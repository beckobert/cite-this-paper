from __future__ import annotations

import json
import os
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from cite_this_paper import cli
from cite_this_paper.catalog import resolve_catalog_root
from cite_this_paper.corpus import Corpus, CorpusError
from cite_this_paper.shell import CorpusShell

from test_support import CorpusTestCase, TtyStringIO


class ShellAndCliTests(CorpusTestCase):
    def test_normal_cli_command_refreshes_last_accessed_timestamp(self):
        with self.corpus.connect() as connection:
            connection.execute(
                "UPDATE corpus_state SET last_accessed_at = '2000-01-01T00:00:00+00:00' WHERE id = 1"
            )
            connection.commit()
        with patch("cite_this_paper.cli.render_sentences", return_value=[]), redirect_stdout(StringIO()):
            self.assertEqual(
                cli.main(["show-sentences", "--database", str(self.corpus.root), "sentence-id"]), 0
            )
        self.assertNotEqual(self.corpus.state()["last_accessed_at"], "2000-01-01T00:00:00+00:00")

    def test_catalog_root_precedence_and_shell_selection_lifecycle(self):
        environment_root = self.root / "environment-root"
        explicit_root = self.root / "explicit-root"
        with patch.dict(os.environ, {"CITE_THIS_PAPER_ROOT": str(environment_root)}, clear=False):
            self.assertEqual(resolve_catalog_root(), environment_root.resolve())
            self.assertEqual(resolve_catalog_root(explicit_root), explicit_root.resolve())

        output = StringIO()
        shell = CorpusShell(environment_root, stdout=output)
        shell.onecmd("rebuild-index")
        self.assertIn("No active corpus", output.getvalue())
        shell.onecmd("create water")
        self.assertEqual(shell.active_name, "water")
        self.assertTrue((environment_root / "water" / "corpus.sqlite").exists())
        shell.onecmd("logout")
        self.assertIsNone(shell.active_name)
        shell.onecmd("load water")
        self.assertEqual(shell.active_name, "water")
        self.assertIsNone(CorpusShell(environment_root, stdout=StringIO()).active_name)

    def test_catalog_views_handle_healthy_empty_incompatible_and_missing_corpora(self):
        catalog_root = self.root / "catalog"
        Corpus.create(catalog_root / "healthy")
        legacy = Corpus.create(catalog_root / "legacy")
        empty_vectors = Corpus.create(catalog_root / "empty-vectors")
        empty_vectors.matrix_path.touch()
        with legacy.connect() as connection:
            connection.execute("UPDATE corpus_state SET schema_version = 1 WHERE id = 1")
            connection.commit()

        output = StringIO()
        shell = CorpusShell(catalog_root, stdout=output)
        shell.onecmd("load healthy")
        shell.onecmd("list")
        list_text = output.getvalue()
        shell.onecmd("info empty-vectors")
        shell.onecmd("info legacy")
        shell.onecmd("info missing")
        text = output.getvalue()

        self.assertIn("healthy", text)
        self.assertIn("legacy", text)
        self.assertIn("empty-vectors", text)
        self.assertIn("incompatible", text)
        self.assertNotIn("Schema 1", list_text)
        self.assertIn("Schema 1", text)
        self.assertIn("Index data: 0 rows, - dimensions", text)
        self.assertIn("Corpus 'missing' does not exist", text)
        with self.assertRaises(CorpusError):
            Corpus.open(legacy.root)
        shell.onecmd("load legacy")
        self.assertIn("Recreate and reingest", output.getvalue())

    def test_shell_help_exposes_core_commands_and_command_specific_guidance(self):
        output = StringIO()
        shell = CorpusShell(self.root / "catalog", stdout=output)
        shell.onecmd("help")
        shell.onecmd("help create")
        text = output.getvalue()
        for command in ("create NAME", "add-pdf", "verify-claim", "show-sentences", "cleanup NAME"):
            self.assertIn(command, text)
        self.assertIn("create: Create and activate a named corpus.", text)

    def test_shell_settings_control_prompt_and_persist(self):
        settings_path = self.root / "user-settings.json"
        output = TtyStringIO()
        with patch.dict(os.environ, {}, clear=True):
            shell = CorpusShell(self.root / "catalog", stdout=output, settings_path=settings_path)
        self.assertIn("\033[1;36m", shell.prompt)
        self.assertIn("\001\033[1;36m\002", shell.prompt)
        self.assertIn("\001\033[0m\002", shell.prompt)

        shell.onecmd("settings set prompt.color magenta")
        shell.onecmd('settings set prompt.marker \">\"')
        shell.onecmd("settings set color.mode never")
        self.assertEqual(shell.prompt, "cite-this-paper [no corpus] > ")
        self.assertEqual(json.loads(settings_path.read_text(encoding="utf-8"))["prompt"]["color"], "magenta")

        reloaded = CorpusShell(self.root / "catalog", stdout=TtyStringIO(), settings_path=settings_path)
        self.assertEqual((reloaded.settings.prompt_color, reloaded.settings.prompt_marker, reloaded.settings.color_mode), ("magenta", ">", "never"))
        shell.onecmd("settings reset prompt.color")
        self.assertEqual(shell.settings.prompt_color, "cyan")
        shell.onecmd("settings set prompt.color orange")
        self.assertIn("prompt.color must be one of", output.getvalue())

    def test_shell_settings_fall_back_from_invalid_files_and_honor_no_color(self):
        invalid_files = (
            ("invalid-json.json", lambda path: path.write_text("{not json", encoding="utf-8")),
            ("invalid-utf8.json", lambda path: path.write_bytes(b"\xff\xfe")),
        )
        for filename, write_invalid_file in invalid_files:
            with self.subTest(filename=filename):
                settings_path = self.root / filename
                write_invalid_file(settings_path)
                output = TtyStringIO()
                with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
                    shell = CorpusShell(self.root / "catalog", stdout=output, settings_path=settings_path)
                self.assertEqual(shell.settings.prompt_color, "cyan")
                self.assertNotIn("\033[", shell.prompt)
                self.assertIn("Ignoring invalid settings file", output.getvalue())

    def test_shell_operational_commands_use_session_parser_and_active_corpus(self):
        session_args = cli.build_parser(session=True).parse_args(["show-sentences", "sentence-id"])
        self.assertFalse(hasattr(session_args, "database"))
        direct_parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            direct_parser.parse_args(["show-sentences", "sentence-id"])

        catalog_root = self.root / "catalog"
        Corpus.create(catalog_root / "water")
        shell = CorpusShell(catalog_root, stdout=StringIO(), settings_path=self.root / "settings.json")
        shell.onecmd("load water")
        with patch("cite_this_paper.cli.execute_operational") as execute:
            shell.onecmd('verify-claim "a scientific claim" --quiet')
        arguments, keyword_arguments = execute.call_args
        self.assertEqual(arguments[0].command, "verify-claim")
        self.assertEqual(arguments[0].claim, "a scientific claim")
        self.assertTrue(arguments[0].quiet)
        self.assertTrue(keyword_arguments["interactive"])

    def test_shell_completion_covers_commands_corpora_options_and_settings(self):
        catalog_root = self.root / "catalog"
        Corpus.create(catalog_root / "water")
        Corpus.create(catalog_root / "weather")
        shell = CorpusShell(catalog_root, stdout=StringIO(), settings_path=self.root / "settings.json")

        self.assertEqual(shell.completenames("ver"), ["verify-claim"])
        self.assertEqual(shell.complete_load("w", "load w", 5, 6), ["water", "weather"])
        self.assertEqual(shell.complete_info("wea", "info wea", 5, 8), ["weather"])
        self.assertEqual(shell.complete_cleanup("wa", "cleanup wa", 8, 10), ["water"])
        self.assertIn("--verbose", shell.completedefault("--v", "verify-claim --v", 13, 16))
        self.assertEqual(shell.completedefault("r", "add-pdf source.pdf --on-duplicate r", 34, 35), ["replace"])
        self.assertEqual(shell.completedefault("--q", "verify-claim a claim --quiet --q", 29, 32), [])
        self.assertEqual(
            shell.complete_settings("prompt.", "settings set prompt.", 13, 20),
            ["prompt.color", "prompt.bold", "prompt.marker"],
        )
        self.assertEqual(shell.complete_settings("a", "settings set color.mode a", 24, 25), ["auto", "always"])
        self.assertEqual(shell.complete_help("show", "help show", 5, 9), ["show-sentences"])

    def test_shell_completion_restores_readline_delimiters(self):
        try:
            import readline
        except ImportError:
            self.skipTest("readline is unavailable")
        original = readline.get_completer_delims()
        shell = CorpusShell(self.root / "catalog", stdout=StringIO(), settings_path=self.root / "settings.json")
        try:
            shell.preloop()
            self.assertNotIn("-", readline.get_completer_delims())
        finally:
            shell.postloop()
        self.assertEqual(readline.get_completer_delims(), original)
