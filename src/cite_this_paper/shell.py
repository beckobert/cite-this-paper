"""Interactive, shell-only active-corpus experience."""

from __future__ import annotations

import argparse
import cmd
import shlex
from functools import cache
from pathlib import Path

from .catalog import (
    CorpusSummary,
    create_named_corpus,
    corpus_path,
    inspect_corpus,
    list_corpora,
    open_named_corpus,
    resolve_catalog_root,
)
from .cleanup import CleanupResult, cleanup_corpora, find_inactive_corpora, inspect_corpus as inspect_cleanup_corpus
from .corpus import Corpus, CorpusError
from .settings import SettingsError, UserSettings, load_settings, prompt_uses_color, save_settings, user_settings_path


PROMPT_COLOR_CODES = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}

COMMAND_HELP = (
    ("create NAME", "Create and activate a named corpus."),
    ("load NAME", "Load an existing corpus into this shell."),
    ("logout", "Clear the active corpus without deleting it."),
    ("list", "List corpora in the current catalog."),
    ("info [NAME]", "Show detailed information for a corpus."),
    ("cleanup NAME …", "Preview or delete named corpora."),
    ("settings", "View or change user-wide shell preferences."),
    ("add-pdf", "Ingest one PDF into the active corpus."),
    ("add-directory", "Ingest PDFs from a directory into the active corpus."),
    ("rebuild-index", "Rebuild the active corpus search indexes."),
    ("verify-claim", "Retrieve, rerank, and verify evidence for a claim."),
    ("show-sentences", "Render highlighted source sentences."),
    ("help [COMMAND]", "Show the command reference or one command description."),
    ("exit", "Leave the shell and discard the active selection."),
    ("quit", "Leave the shell and discard the active selection."),
)


def _matches(candidates: list[str] | tuple[str, ...], text: str) -> list[str]:
    """Return stable completion candidates that match the text at the cursor."""
    return [candidate for candidate in candidates if candidate.startswith(text)]


@cache
def _operational_options(command: str) -> dict[str, tuple[str, ...]]:
    """Read an operational command's flags and constrained values from its parser."""
    from . import cli

    parser = cli.build_parser(session=True)
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    command_parser = subparsers.choices[command]
    options: dict[str, tuple[str, ...]] = {}
    for action in command_parser._actions:
        choices = tuple(str(choice) for choice in action.choices) if action.choices else ()
        for option in action.option_strings:
            options[option] = choices
    return options


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    raise AssertionError("Unreachable")


class CorpusShell(cmd.Cmd):
    """A command session whose active corpus is never persisted to disk."""

    intro = "Interactive cite-this-paper shell. Type 'help' for commands."
    identchars = cmd.Cmd.identchars + "-"

    def __init__(self, root: Path, *, stdin=None, stdout=None, settings_path: Path | None = None) -> None:
        super().__init__(stdin=stdin, stdout=stdout)
        self.root = root
        self.active_name: str | None = None
        self.active_corpus: Corpus | None = None
        self.settings_path = settings_path or user_settings_path()
        self.settings, warning = load_settings(self.settings_path)
        self._readline_delimiters: str | None = None
        self._refresh_prompt()
        if warning:
            self._write(f"WARNING: {warning}")

    def _write(self, message: str = "") -> None:
        print(message, file=self.stdout)

    def _refresh_prompt(self) -> None:
        context = f"cite-this-paper [{self.active_name or 'no corpus'}]"
        if prompt_uses_color(self.settings, self.stdout):
            attributes = f"1;{PROMPT_COLOR_CODES[self.settings.prompt_color]}" if self.settings.prompt_bold else str(PROMPT_COLOR_CODES[self.settings.prompt_color])
            context = f"\001\033[{attributes}m\002{context}\001\033[0m\002"
        self.prompt = f"{context} {self.settings.prompt_marker} "

    def _activate(self, name: str, corpus: Corpus) -> None:
        corpus.touch_access()
        self.active_name = name
        self.active_corpus = corpus
        self._refresh_prompt()
        self._write(f"Active corpus: {name} ({corpus.root})")

    def _parse_names(self, argument: str, command: str, *, optional: bool = False) -> list[str] | None:
        try:
            names = shlex.split(argument)
        except ValueError as error:
            self._write(f"ERROR: {error}")
            return None
        if not optional and len(names) != 1:
            self._write(f"Usage: {command} NAME")
            return None
        if optional and len(names) > 1:
            self._write(f"Usage: {command} [NAME]")
            return None
        return names

    def _corpus_name_completions(self, text: str) -> list[str]:
        return _matches([summary.name for summary in list_corpora(self.root)], text)

    @staticmethod
    def _tokens_before_cursor(line: str, begidx: int) -> list[str]:
        """Split only completed text so an unfinished quoted value does not break Tab."""
        try:
            return shlex.split(line[:begidx])
        except ValueError:
            return line[:begidx].split()

    def _complete_options(
        self,
        command: str,
        text: str,
        line: str,
        begidx: int,
        *,
        options: dict[str, tuple[str, ...]] | None = None,
    ) -> list[str]:
        options = options if options is not None else _operational_options(command)
        preceding = self._tokens_before_cursor(line, begidx)
        previous = preceding[-1] if preceding else ""
        if previous in options and options[previous]:
            return _matches(options[previous], text)
        if not text.startswith("-"):
            return []
        used = set(preceding[1:])
        return _matches(tuple(option for option in options if option not in used), text)

    def do_create(self, argument: str) -> None:
        """create NAME -- create and immediately activate a named corpus."""
        names = self._parse_names(argument, "create")
        if not names:
            return
        try:
            self._activate(names[0], create_named_corpus(self.root, names[0]))
        except CorpusError as error:
            self._write(f"ERROR: {error}")

    def complete_load(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        return self._corpus_name_completions(text)

    def complete_info(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        return self._corpus_name_completions(text)

    def do_load(self, argument: str) -> None:
        """load NAME -- load an existing named corpus into this session."""
        names = self._parse_names(argument, "load")
        if not names:
            return
        try:
            self._activate(names[0], open_named_corpus(self.root, names[0]))
        except CorpusError as error:
            self._write(f"ERROR: {error}")

    def do_logout(self, argument: str) -> None:
        """logout -- clear the active corpus without deleting anything."""
        if argument.strip():
            self._write("Usage: logout")
            return
        self.active_name = None
        self.active_corpus = None
        self._refresh_prompt()
        self._write("No corpus is active.")

    def do_list(self, argument: str) -> None:
        """list -- display all named corpora below this shell's catalog root."""
        if argument.strip():
            self._write("Usage: list")
            return
        summaries = list_corpora(self.root)
        self._write(f"Catalog root: {self.root}")
        if not summaries:
            self._write("No named corpora are available.")
            return
        self._write(f"{'ACTIVE':<8} {'NAME':<20} {'STATUS':<18} {'DOCS':>5} {'SIZE':>10} LAST ACCESSED")
        self._write("-" * 88)
        for summary in summaries:
            active = "*" if summary.name == self.active_name else ""
            status = summary.status if summary.status != "ready" else (summary.index_status or "ready")
            documents = "-" if summary.document_count is None else str(summary.document_count)
            accessed = summary.last_accessed_at or "-"
            self._write(
                f"{active:<8} {summary.name:<20} {status:<18} {documents:>5} "
                f"{_format_size(summary.size_bytes):>10} {accessed}"
            )
            if summary.message and summary.status != "incompatible":
                self._write(f"         {summary.message}")

    def do_info(self, argument: str) -> None:
        """info [NAME] -- display detailed information for one corpus."""
        names = self._parse_names(argument, "info", optional=True)
        if names is None:
            return
        name = names[0] if names else self.active_name
        if name is None:
            self._write("ERROR: No active corpus. Use 'info NAME', 'load NAME', or 'create NAME'.")
            return
        try:
            path = corpus_path(self.root, name)
            if not path.exists():
                self._write(f"ERROR: Corpus '{name}' does not exist.")
                return
            summary = inspect_corpus(self.root, name)
        except CorpusError as error:
            self._write(f"ERROR: {error}")
            return
        if name == self.active_name and self.active_corpus is not None:
            self.active_corpus.touch_access()
            summary = inspect_corpus(self.root, name)
        self._print_info(summary)

    def _print_info(self, summary: CorpusSummary) -> None:
        self._write(f"Corpus: {summary.name}")
        self._write(f"Path:   {summary.root}")
        self._write(f"Status: {summary.status}")
        self._write(f"Size:   {_format_size(summary.size_bytes)}")
        if summary.message:
            self._write(f"Detail: {summary.message}")
        if summary.schema_version is None:
            return
        self._write(f"Schema: {summary.schema_version}")
        self._write(f"Index:  {summary.index_status or '-'}")
        if summary.document_count is None:
            return
        self._write(
            "Content: "
            f"{summary.document_count} documents, {summary.page_count} pages, "
            f"{summary.sentence_count} sentences, {summary.passage_count} passages, "
            f"{summary.eligible_passage_count} eligible"
        )
        self._write(
            "Index data: "
            f"{summary.indexed_passage_count} rows, {summary.embedding_dimensions if summary.embedding_dimensions is not None else '-'} dimensions, "
            f"model {summary.embedding_model or '-'}"
        )
        self._write(
            "Activity: "
            f"last indexed {summary.last_indexed_at or '-'}; last accessed {summary.last_accessed_at or '-'}"
        )
        self._write(
            "History: "
            f"{summary.verification_run_count} verification runs; {summary.ingestion_error_count} ingestion errors"
        )

    def do_settings(self, argument: str) -> None:
        """settings [set KEY VALUE | reset [KEY]] -- view or update shell preferences."""
        try:
            tokens = shlex.split(argument)
        except ValueError as error:
            self._write(f"ERROR: {error}")
            return
        if not tokens:
            self._write(f"Settings file: {self.settings_path}")
            for key, value in self.settings.values().items():
                self._write(f"{key} = {value}")
            self._write("Use 'settings set KEY VALUE' or 'settings reset [KEY]'.")
            return
        if tokens[0] == "set" and len(tokens) == 3:
            try:
                updated = self.settings.with_value(tokens[1], tokens[2])
            except SettingsError as error:
                self._write(f"ERROR: {error}")
                return
        elif tokens[0] == "reset" and len(tokens) in {1, 2}:
            if len(tokens) == 1:
                updated = UserSettings.defaults()
            else:
                try:
                    updated = self.settings.with_value(tokens[1], UserSettings.defaults().values()[tokens[1]])
                except (KeyError, SettingsError):
                    self._write(f"ERROR: Unknown setting: {tokens[1]}")
                    return
        else:
            self._write("Usage: settings [set KEY VALUE | reset [KEY]]")
            return
        try:
            self.settings_path = save_settings(updated, self.settings_path)
        except OSError as error:
            self._write(f"ERROR: Could not save settings: {error}")
            return
        self.settings = updated
        self._refresh_prompt()
        self._write("Settings saved.")

    def complete_settings(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        preceding = self._tokens_before_cursor(line, begidx)[1:]
        keys = tuple(self.settings.values())
        if not preceding:
            return _matches(("set", "reset"), text)
        if preceding[0] == "set":
            if len(preceding) == 1:
                return _matches(keys, text)
            if len(preceding) == 2:
                values = {
                    "prompt.color": tuple(sorted(PROMPT_COLOR_CODES)),
                    "prompt.bold": ("true", "false"),
                    "prompt.marker": (self.settings.prompt_marker,),
                    "color.mode": ("auto", "always", "never"),
                }.get(preceding[1], ())
                return _matches(values, text)
        if preceding[0] == "reset" and len(preceding) == 1:
            return _matches(keys, text)
        return []

    def do_cleanup(self, argument: str) -> None:
        """cleanup [NAME ...] [--unused-for DAYS] [--apply] -- preview or delete named corpora."""
        parser = argparse.ArgumentParser(prog="cleanup", add_help=False)
        parser.add_argument("names", nargs="*")
        parser.add_argument("--unused-for", type=int, metavar="DAYS")
        parser.add_argument("--apply", action="store_true")
        try:
            args = parser.parse_args(shlex.split(argument))
        except (SystemExit, ValueError):
            return
        explicit = bool(args.names)
        age_based = args.unused_for is not None
        if explicit == age_based:
            self._write("ERROR: Specify corpus names or --unused-for DAYS, but not both.")
            return
        try:
            if age_based:
                discovered = find_inactive_corpora(self.root, args.unused_for, recursive=False)
                outcomes = self._cleanup_results(
                    discovered,
                    apply=args.apply,
                    protected_root=self.root,
                )
                mode = f"inactive for at least {args.unused_for} day(s) below {self.root}"
            else:
                paths = [corpus_path(self.root, name) for name in args.names]
                inspected = [inspect_cleanup_corpus(path) for path in paths]
                outcomes = self._cleanup_results(inspected, apply=args.apply)
                mode = "named corpora"
        except (CorpusError, ValueError) as error:
            self._write(f"ERROR: {error}")
            return
        self._print_cleanup(outcomes, mode=mode, apply=args.apply)

    def complete_cleanup(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        if text.startswith("-"):
            return self._complete_options(
                "cleanup",
                text,
                line,
                begidx,
                options={"--unused-for": (), "--apply": ()},
            )
        return self._corpus_name_completions(text)

    def _cleanup_results(
        self,
        results: list[CleanupResult],
        *,
        apply: bool,
        protected_root: Path | None = None,
    ) -> list[CleanupResult]:
        if apply and any(result.status == "invalid" for result in results):
            return results
        protected: list[CleanupResult] = []
        candidates: list[Path] = []
        for result in results:
            if self.active_corpus is not None and result.root == self.active_corpus.root:
                protected.append(
                    CleanupResult(
                        result.root,
                        result.last_accessed_at,
                        result.size_bytes,
                        "protected",
                        "The active corpus cannot be deleted. Use logout or load another corpus first.",
                    )
                )
            elif result.status == "ready":
                candidates.append(result.root)
            else:
                protected.append(result)
        outcomes = cleanup_corpora(candidates, apply=apply, protected_root=protected_root)
        return sorted([*protected, *outcomes], key=lambda result: str(result.root))

    def _print_cleanup(self, results: list[CleanupResult], *, mode: str, apply: bool) -> None:
        title = "CORPUS CLEANUP REPORT" if apply else "CORPUS CLEANUP PREVIEW"
        self._write(title)
        self._write(f"Mode: {mode}")
        if not results:
            self._write("No corpora matched this cleanup request.")
            return
        self._write(f"{'STATUS':<10} {'SIZE':>10}  {'LAST ACCESSED':<25} CORPUS")
        self._write("-" * 88)
        for result in results:
            self._write(
                f"{result.status.upper():<10} {_format_size(result.size_bytes):>10}  "
                f"{result.last_accessed_at or 'not tracked':<25} {result.root.name}"
            )
            if result.message:
                self._write(f"           {result.message}")
        if not apply:
            self._write("No data was removed. Re-run with --apply to permanently delete listed corpora.")

    def do_exit(self, argument: str) -> bool:
        """exit -- leave the shell and discard the in-memory active selection."""
        if argument.strip():
            self._write("Usage: exit")
            return False
        return True

    def do_quit(self, argument: str) -> bool:
        """quit -- alias for exit."""
        return self.do_exit(argument)

    def do_help(self, argument: str) -> None:
        """help [COMMAND] -- display a concise shell command reference."""
        command = argument.strip()
        descriptions = {entry.split()[0]: description for entry, description in COMMAND_HELP}
        if command:
            description = descriptions.get(command)
            if description is None:
                self._write(f"No help is available for: {command}")
            else:
                self._write(f"{command}: {description}")
            return
        self._write("Commands:")
        for command_usage, description in COMMAND_HELP:
            self._write(f"  {command_usage:<24} {description}")
        self._write("Use 'help COMMAND' for a command description.")

    def complete_help(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        return _matches(tuple(entry.split()[0] for entry, _ in COMMAND_HELP), text)

    def completenames(self, text: str, *ignored: object) -> list[str]:
        """Include operational commands, which are dispatched through default()."""
        return _matches(tuple(entry.split()[0] for entry, _ in COMMAND_HELP), text)

    def completedefault(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        """Complete flags for commands that share the normal CLI dispatcher."""
        command, _, _ = self.parseline(line)
        from . import cli

        if command in cli.OPERATIONAL_COMMANDS:
            return self._complete_options(command, text, line, begidx)
        return []

    def preloop(self) -> None:
        """Make hyphenated commands and long options one readline completion word."""
        try:
            import readline
        except ImportError:
            return
        if self._readline_delimiters is None:
            self._readline_delimiters = readline.get_completer_delims()
            readline.set_completer_delims(self._readline_delimiters.replace("-", ""))

    def postloop(self) -> None:
        self._restore_readline_delimiters()

    def _restore_readline_delimiters(self) -> None:
        if self._readline_delimiters is None:
            return
        try:
            import readline
        except ImportError:
            return
        readline.set_completer_delims(self._readline_delimiters)
        self._readline_delimiters = None

    def cmdloop(self, intro: str | None = None) -> None:
        """Restore process-wide readline state even when the shell exits unexpectedly."""
        try:
            super().cmdloop(intro)
        finally:
            self._restore_readline_delimiters()

    def do_EOF(self, argument: str) -> bool:  # noqa: N802
        self._write()
        return True

    def default(self, line: str) -> None:
        """Dispatch regular corpus commands through the shared CLI implementation."""
        from . import cli

        try:
            tokens = shlex.split(line)
        except ValueError as error:
            self._write(f"ERROR: {error}")
            return
        if not tokens:
            return
        command = tokens[0]
        if command in {"init-db", "cleanup-databases", "shell"}:
            self._write("Use create, cleanup, and the current interactive shell instead.")
            return
        if command not in cli.OPERATIONAL_COMMANDS:
            self._write(f"Unknown command: {command}. Type 'help' for commands.")
            return
        if self.active_corpus is None:
            self._write("ERROR: No active corpus. Use 'load NAME' or 'create NAME'.")
            return
        try:
            args = cli.build_parser(session=True).parse_args(tokens)
        except SystemExit:
            return
        try:
            cli.execute_operational(args, self.active_corpus, interactive=True)
        except CorpusError as error:
            self._write(f"ERROR: {error}")


def run_shell(root: Path | None = None) -> int:
    """Start one interactive session using the resolved named-corpus catalog."""
    shell = CorpusShell(resolve_catalog_root(root))
    shell.cmdloop()
    return 0
