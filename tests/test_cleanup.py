from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

from cite_this_paper import cli
from cite_this_paper.corpus import Corpus
from cite_this_paper.shell import CorpusShell

from test_support import CorpusTestCase


class CleanupTests(CorpusTestCase):
    def _mark_old(self, corpus: Corpus) -> None:
        with corpus.connect() as connection:
            connection.execute(
                "UPDATE corpus_state SET last_accessed_at = '2000-01-01T00:00:00+00:00' WHERE id = 1"
            )
            connection.commit()

    def test_cleanup_preview_and_applied_explicit_deletion(self):
        target = Corpus.create(self.root / "remove-me")
        preview = StringIO()
        with redirect_stdout(preview):
            exit_code = cli.main(["cleanup-databases", str(target.root)])
        self.assertEqual(exit_code, 0)
        self.assertTrue(target.root.exists())
        self.assertIn("DATABASE CLEANUP PREVIEW", preview.getvalue())

        applied = StringIO()
        with redirect_stdout(applied):
            exit_code = cli.main(["cleanup-databases", str(target.root), "--apply"])
        self.assertEqual(exit_code, 0)
        self.assertFalse(target.root.exists())
        self.assertIn("DELETED", applied.getvalue())

    def test_cleanup_refuses_all_deletion_when_one_requested_target_is_invalid(self):
        target = Corpus.create(self.root / "keep-me")
        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main([
                "cleanup-databases", str(target.root), str(self.root / "not-a-corpus"), "--apply"
            ])
        self.assertEqual(exit_code, 2)
        self.assertTrue(target.root.exists())
        self.assertIn("INVALID", output.getvalue())

    def test_age_based_cleanup_uses_last_accessed_timestamp(self):
        old = Corpus.create(self.root / "old")
        recent = Corpus.create(self.root / "recent")
        self._mark_old(old)

        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main(["cleanup-databases", "--unused-for", "30", "--root", str(self.root)])
        self.assertEqual(exit_code, 0)
        self.assertTrue(old.root.exists())
        self.assertTrue(recent.root.exists())
        self.assertIn(str(old.root), output.getvalue())
        self.assertNotIn(str(recent.root), output.getvalue())

        with redirect_stdout(StringIO()):
            exit_code = cli.main([
                "cleanup-databases", "--unused-for", "30", "--root", str(self.root), "--apply"
            ])
        self.assertEqual(exit_code, 0)
        self.assertFalse(old.root.exists())
        self.assertTrue(recent.root.exists())

    def test_age_based_cleanup_never_deletes_the_scan_root(self):
        scan_root = self.root / "scan-root"
        corpus = Corpus.create(scan_root)
        self._mark_old(corpus)
        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main([
                "cleanup-databases", "--unused-for", "30", "--root", str(scan_root), "--apply"
            ])
        self.assertEqual(exit_code, 2)
        self.assertTrue(scan_root.exists())
        self.assertIn("cleanup root itself", output.getvalue())

    def test_shell_cleanup_rejects_symlinked_corpus_targets(self):
        external = Corpus.create(self.root / "external")
        self._mark_old(external)
        catalog_root = self.root / "catalog"
        catalog_root.mkdir()
        linked = catalog_root / "linked"
        linked.symlink_to(external.root, target_is_directory=True)

        output = StringIO()
        shell = CorpusShell(catalog_root, stdout=output)
        shell.onecmd("cleanup linked --apply")
        shell.onecmd("cleanup --unused-for 1 --apply")

        self.assertTrue(linked.is_symlink())
        self.assertTrue(external.root.exists())
        self.assertIn("Symlinked corpus directories cannot be cleaned", output.getvalue())

    def test_shell_cleanup_protects_active_corpus_until_logout(self):
        catalog_root = self.root / "catalog"
        output = StringIO()
        shell = CorpusShell(catalog_root, stdout=output)
        shell.onecmd("create water")
        water_path = catalog_root / "water"
        shell.onecmd("cleanup water --apply")
        self.assertTrue(water_path.exists())
        self.assertIn("PROTECTED", output.getvalue())

        shell.onecmd("logout")
        shell.onecmd("cleanup water --apply")
        self.assertFalse(water_path.exists())

    def test_shell_cleanup_refuses_all_deletion_when_one_name_is_invalid(self):
        catalog_root = self.root / "catalog"
        Corpus.create(catalog_root / "keep-me")
        output = StringIO()
        shell = CorpusShell(catalog_root, stdout=output)
        shell.onecmd("cleanup keep-me missing --apply")
        self.assertTrue((catalog_root / "keep-me").exists())
        self.assertIn("INVALID", output.getvalue())
