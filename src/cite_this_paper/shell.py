"""Interactive, shell-only active-corpus experience."""

from __future__ import annotations

import argparse
import cmd
import shlex
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

    def __init__(self, root: Path, *, stdin=None, stdout=None) -> None:
        super().__init__(stdin=stdin, stdout=stdout)
        self.root = root
        self.active_name: str | None = None
        self.active_corpus: Corpus | None = None
        self._refresh_prompt()

    def _write(self, message: str = "") -> None:
        print(message, file=self.stdout)

    def _refresh_prompt(self) -> None:
        suffix = f"[{self.active_name}]" if self.active_name else ""
        self.prompt = f"cite-this-paper{suffix}> "

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

    def do_create(self, argument: str) -> None:
        """create NAME -- create and immediately activate a named corpus."""
        names = self._parse_names(argument, "create")
        if not names:
            return
        try:
            self._activate(names[0], create_named_corpus(self.root, names[0]))
        except CorpusError as error:
            self._write(f"ERROR: {error}")

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
            if summary.message:
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
