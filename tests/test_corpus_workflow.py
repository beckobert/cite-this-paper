from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pymupdf

from cite_this_paper.corpus import Corpus, CorpusError
from cite_this_paper import cli
from cite_this_paper.catalog import resolve_catalog_root
from cite_this_paper.indexing import IndexResult, rebuild_index
from cite_this_paper.ingest import ingest_pdf
from cite_this_paper.models import (
    VERDICT_LABELS,
    VERIFIER_PROMPT,
    VERIFIER_PROMPT_VERSION,
    VerificationOutput,
    parse_verification_output,
)
from cite_this_paper.progress import ConsoleReporter
from cite_this_paper.processing import sentences as sentence_processing
from cite_this_paper.review import render_sentences
from cite_this_paper.retrieval import verify_claim
from cite_this_paper.shell import CorpusShell


class FakeEmbeddingModel:
    name = "test-embedding"

    def encode(self, texts):
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


class FakeReranker:
    name = "test-reranker"

    def rerank(self, claim, passages):
        return [(1.0 / (position + 1), -float(position)) for position, _ in enumerate(passages)]


class FakeVerifier:
    name = "test-verifier"

    def verify(self, claim, numbered_sentences):
        return VerificationOutput("DIRECT_SUPPORT", ["S1"], "Test verdict")


class NoEvidenceVerifier:
    name = "test-no-evidence-verifier"

    def verify(self, claim, numbered_sentences):
        return VerificationOutput("RELATED_ONLY", [], "No individual sentence was selected")


class NotMentionedVerifier:
    name = "test-not-mentioned-verifier"

    def verify(self, claim, numbered_sentences):
        return VerificationOutput("NOT_MENTIONED", [], "The passage is unrelated to the claim")


class TtyStringIO(StringIO):
    def isatty(self):
        return True


class NoTagContradictionVerifier:
    name = "test-no-tag-contradiction-verifier"

    def verify(self, claim, numbered_sentences):
        return VerificationOutput("CONTRADICTS", [], "Test contradiction without sentence tags")


class RecordingProgress:
    def __init__(self, description, total):
        self.description = description
        self.total = total
        self.advanced = 0
        self.closed = False

    def advance(self, amount=1):
        self.advanced += amount

    def close(self):
        self.closed = True


class RecordingReporter:
    def __init__(self):
        self.stages = []
        self.bars = []

    def stage(self, message):
        self.stages.append(message)

    def progress(self, description, total):
        bar = RecordingProgress(description, total)
        self.bars.append(bar)
        return bar


class ClosingEmbeddingModel(FakeEmbeddingModel):
    name = "closing-embedding"

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class ClosingReranker(FakeReranker):
    name = "closing-reranker"

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class ClosingVerifier(FakeVerifier):
    name = "closing-verifier"

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def create_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def create_two_page_pdf(path: Path, page_texts: list[str]) -> None:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class CorpusWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = Corpus.create(self.root / "corpus")
        self.pdf = self.root / "paper.pdf"
        create_pdf(self.pdf, "This scientific passage contains enough words to be eligible for evidence retrieval.")

    def tearDown(self):
        self.temporary.cleanup()

    def test_ingestion_deduplicates_and_keeps_highlight_provenance(self):
        first = ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        discarded = ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        replaced = ingest_pdf(self.corpus, self.pdf, on_duplicate="replace")
        self.assertEqual((first.status, discarded.status, replaced.status), ("added", "discarded", "replaced"))
        with self.corpus.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM documents").fetchone()[0], 1)
            self.assertGreater(connection.execute("SELECT count(*) FROM sentences").fetchone()[0], 0)
            self.assertGreater(connection.execute("SELECT count(*) FROM sentence_boxes").fetchone()[0], 0)
            self.assertGreater(connection.execute("SELECT count(*) FROM passages").fetchone()[0], 0)

    def test_cleanup_preview_and_applied_explicit_deletion(self):
        target = Corpus.create(self.root / "remove-me")
        preview = StringIO()
        with redirect_stdout(preview):
            exit_code = cli.main(["cleanup-databases", str(target.root)])
        self.assertEqual(exit_code, 0)
        self.assertTrue(target.root.exists())
        self.assertIn("DATABASE CLEANUP PREVIEW", preview.getvalue())
        self.assertIn("PREVIEW", preview.getvalue())

        applied = StringIO()
        with redirect_stdout(applied):
            exit_code = cli.main(["cleanup-databases", str(target.root), "--apply"])
        self.assertEqual(exit_code, 0)
        self.assertFalse(target.root.exists())
        self.assertIn("DATABASE CLEANUP REPORT", applied.getvalue())
        self.assertIn("DELETED", applied.getvalue())

    def test_cleanup_refuses_to_delete_valid_target_when_request_has_invalid_target(self):
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
        with old.connect() as connection:
            connection.execute("UPDATE corpus_state SET last_accessed_at = '2000-01-01T00:00:00+00:00' WHERE id = 1")
            connection.commit()

        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main([
                "cleanup-databases", "--unused-for", "30", "--root", str(self.root)
            ])
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
        with corpus.connect() as connection:
            connection.execute("UPDATE corpus_state SET last_accessed_at = '2000-01-01T00:00:00+00:00' WHERE id = 1")
            connection.commit()
        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main([
                "cleanup-databases", "--unused-for", "30", "--root", str(scan_root), "--apply"
            ])
        self.assertEqual(exit_code, 2)
        self.assertTrue(scan_root.exists())
        self.assertIn("cleanup root itself", output.getvalue())

    def test_cleanup_rejects_symlinked_corpus_targets(self):
        external = Corpus.create(self.root / "external")
        with external.connect() as connection:
            connection.execute("UPDATE corpus_state SET last_accessed_at = '2000-01-01T00:00:00+00:00' WHERE id = 1")
            connection.commit()
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

    def test_normal_cli_command_refreshes_last_accessed_timestamp(self):
        with self.corpus.connect() as connection:
            connection.execute("UPDATE corpus_state SET last_accessed_at = '2000-01-01T00:00:00+00:00' WHERE id = 1")
            connection.commit()
        with patch("cite_this_paper.cli.render_sentences", return_value=[]), redirect_stdout(StringIO()):
            self.assertEqual(
                cli.main(["show-sentences", "--database", str(self.corpus.root), "sentence-id"]),
                0,
            )
        self.assertNotEqual(self.corpus.state()["last_accessed_at"], "2000-01-01T00:00:00+00:00")

    def test_rebuild_and_verified_claim_are_audited(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        result = rebuild_index(self.corpus, FakeEmbeddingModel())
        self.assertGreater(result.indexed_passages, 0)
        run_id, warning, verified = verify_claim(
            self.corpus,
            "This passage is scientific evidence.",
            embedding_model=FakeEmbeddingModel(),
            reranker=FakeReranker(),
            verifier=FakeVerifier(),
            candidate_k=10,
            rerank_k=5,
            verify_k=1,
        )
        self.assertIsNone(warning)
        self.assertEqual(verified[0].verification.label, "DIRECT_SUPPORT")
        with self.corpus.connect() as connection:
            self.assertEqual(connection.execute("SELECT status FROM verification_runs WHERE id = ?", (run_id,)).fetchone()[0], "completed")
            self.assertGreater(connection.execute("SELECT count(*) FROM verification_evidence").fetchone()[0], 0)
            config = json.loads(
                connection.execute(
                    "SELECT configuration_json FROM verification_runs WHERE id = ?", (run_id,)
                ).fetchone()[0]
            )
        self.assertEqual(config["verifier_prompt_version"], VERIFIER_PROMPT_VERSION)

    def test_processing_reporter_describes_stages_and_verifier_progress(self):
        reporter = RecordingReporter()
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard", reporter=reporter)
        self.assertEqual(reporter.stages, ["Processing PDF: paper.pdf"])
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard", reporter=reporter)
        self.assertIn("Duplicate document found; keeping the existing copy.", reporter.stages)

        rebuild_index(self.corpus, FakeEmbeddingModel(), reporter=reporter)
        self.assertIn("Creating embeddings for 1 passage(s)...", reporter.stages)
        self.assertIn("Rebuilding the lexical search index...", reporter.stages)

        verify_claim(
            self.corpus,
            "This passage is scientific evidence.",
            embedding_model=FakeEmbeddingModel(),
            reranker=FakeReranker(),
            verifier=FakeVerifier(),
            candidate_k=10,
            rerank_k=5,
            verify_k=1,
            reporter=reporter,
        )
        self.assertIn("Creating claim embedding and retrieving dense and lexical candidates...", reporter.stages)
        self.assertIn("Saving the verification audit record...", reporter.stages)
        self.assertEqual(len(reporter.bars), 1)
        self.assertEqual(reporter.bars[0].description, "Verifying passages")
        self.assertEqual(reporter.bars[0].total, 1)
        self.assertEqual(reporter.bars[0].advanced, 1)
        self.assertTrue(reporter.bars[0].closed)

    def test_ingestion_report_summarizes_results_and_failures(self):
        output = StringIO()
        results = [
            cli.IngestResult(self.pdf, "added"),
            cli.IngestResult(self.root / "duplicate.pdf", "discarded"),
            cli.IngestResult(self.root / "broken.pdf", "failed", message="Unreadable PDF"),
        ]
        with redirect_stdout(output):
            cli._print_ingestion_report(self.corpus, results, "Deferred (new PDFs are not searchable until rebuild-index runs)")
        text = output.getvalue()
        self.assertIn("INGESTION REPORT", text)
        self.assertIn("Processed: 3 PDF(s)", text)
        self.assertIn("Added:     1", text)
        self.assertIn("Duplicates kept: 1", text)
        self.assertIn("Failed:    1", text)
        self.assertIn("broken.pdf: Unreadable PDF", text)

    def test_quiet_ingestion_keeps_the_final_report(self):
        output = StringIO()
        with patch("cite_this_paper.cli._ingest_one", return_value=cli.IngestResult(self.pdf, "added")), redirect_stdout(output):
            result = cli.main([
                "add-pdf", "--database", str(self.corpus.root), str(self.pdf), "--defer-rebuild", "--quiet"
            ])
        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("INGESTION REPORT", text)
        self.assertNotIn("Processing PDF:", text)

    def test_automatic_models_are_released_after_verification(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        embedding = ClosingEmbeddingModel()
        reranker = ClosingReranker()
        verifier = ClosingVerifier()
        with (
            patch("cite_this_paper.retrieval.BGEEmbeddingModel", return_value=embedding),
            patch("cite_this_paper.retrieval.QwenPassageReranker", return_value=reranker),
            patch("cite_this_paper.retrieval.QwenClaimVerifier", return_value=verifier),
        ):
            verify_claim(self.corpus, "This passage is scientific evidence.", candidate_k=10, rerank_k=5, verify_k=1)
        self.assertTrue(embedding.closed)
        self.assertTrue(reranker.closed)
        self.assertTrue(verifier.closed)

    def test_console_reporter_quiet_mode_and_cli_flags(self):
        output = StringIO()
        ConsoleReporter(stream=output).stage("Visible stage")
        ConsoleReporter(quiet=True, stream=output).stage("Hidden stage")
        self.assertEqual(output.getvalue(), "Visible stage\n")
        parser = cli.build_parser()
        self.assertTrue(parser.parse_args(["rebuild-index", "--database", "corpus", "--quiet"]).quiet)
        self.assertTrue(parser.parse_args(["verify-claim", "--database", "corpus", "claim", "--quiet"]).quiet)

    def test_physical_block_merge_diagnostics_require_debug(self):
        page = {"document_id": "paper", "page_number": 1, "words": [{"text": "placeholder"}]}
        physical_blocks = OrderedDict([(1, []), (2, [])])
        output = StringIO()
        with (
            patch.object(sentence_processing, "group_words_by_block_and_line", return_value=physical_blocks),
            patch.object(sentence_processing, "build_logical_block_groups", return_value=[[1, 2]]),
            patch.object(sentence_processing, "reconstruct_logical_block", return_value=("", [])),
            redirect_stdout(output),
        ):
            sentence_processing.build_sentences_for_page(page, None, 0)
        self.assertEqual(output.getvalue(), "")

        with (
            patch.object(sentence_processing, "group_words_by_block_and_line", return_value=physical_blocks),
            patch.object(sentence_processing, "build_logical_block_groups", return_value=[[1, 2]]),
            patch.object(sentence_processing, "reconstruct_logical_block", return_value=("", [])),
            redirect_stdout(output),
        ):
            sentence_processing.build_sentences_for_page(page, None, 0, debug=True)
        self.assertIn("merged physical blocks [1, 2]", output.getvalue())

    def test_add_commands_accept_debug_flag(self):
        parser = cli.build_parser()
        self.assertTrue(parser.parse_args(["add-pdf", "--database", "corpus", "paper.pdf", "--debug"]).debug)
        self.assertTrue(parser.parse_args(["add-directory", "--database", "corpus", "papers", "--debug"]).debug)

    def test_verifier_label_contract_and_prompt_distinguish_unrelated_content(self):
        self.assertIn("NOT_MENTIONED", VERDICT_LABELS)
        self.assertNotIn("REFERENCES", VERDICT_LABELS)
        self.assertIn("NEVER a contradiction", VERIFIER_PROMPT)
        self.assertIn("NOT_MENTIONED", VERIFIER_PROMPT)
        accepted = parse_verification_output(
            '{"label": "NOT_MENTIONED", "evidence": [], "reason": "Unrelated."}'
        )
        rejected = parse_verification_output(
            '{"label": "REFERENCES", "evidence": [], "reason": "Legacy label."}'
        )
        self.assertEqual(accepted.label, "NOT_MENTIONED")
        self.assertTrue(accepted.parse_success)
        self.assertEqual(rejected.label, "VERIFICATION_ERROR")
        self.assertFalse(rejected.parse_success)

    def test_pending_documents_warn_after_a_previous_rebuild(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        second = self.root / "second.pdf"
        create_pdf(second, "Another scientific passage contains enough words for independent evidence retrieval now.")
        ingest_pdf(self.corpus, second, on_duplicate="discard")
        _, warning, _ = verify_claim(
            self.corpus,
            "scientific evidence",
            embedding_model=FakeEmbeddingModel(),
            reranker=FakeReranker(),
            verifier=FakeVerifier(),
            candidate_k=10,
            rerank_k=5,
            verify_k=1,
        )
        self.assertIn("pending index rebuild", warning)

    def test_evidence_focused_output_includes_sentence_render_commands(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        run_id, warning, verified = verify_claim(
            self.corpus,
            "This passage is scientific evidence.",
            embedding_model=FakeEmbeddingModel(),
            reranker=FakeReranker(),
            verifier=FakeVerifier(),
            candidate_k=10,
            rerank_k=5,
            verify_k=1,
        )
        output = StringIO()
        with redirect_stdout(output):
            cli._print_verification_output(
                self.corpus,
                "This passage is scientific evidence.",
                run_id,
                warning,
                verified,
                verbose=False,
            )
        text = output.getvalue()
        sentence_id = self.corpus.connect().execute("SELECT display_id FROM sentences LIMIT 1").fetchone()[0]
        self.assertIn("CLAIM VERIFICATION", text)
        self.assertIn("Verifier-selected evidence:", text)
        self.assertIn(sentence_id, text)
        self.assertIn(f"show-sentences --database {self.corpus.root} {sentence_id}", text)
        self.assertEqual(text.count("show-sentences --database"), 1)
        self.assertIn(cli.RESPONSIBILITY_NOTICE, text)
        self.assertTrue(text.rstrip().endswith(cli.RESPONSIBILITY_NOTICE))
        self.assertNotIn("Retrieval diagnostics:", text)

    def test_no_evidence_selection_displays_every_passage_sentence(self):
        multi_sentence_pdf = self.root / "multi-sentence.pdf"
        create_pdf(
            multi_sentence_pdf,
            "This first scientific sentence is eligible evidence. "
            "This second scientific sentence provides additional context.",
        )
        ingest_pdf(self.corpus, multi_sentence_pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        run_id, warning, verified = verify_claim(
            self.corpus,
            "This passage is scientific evidence.",
            embedding_model=FakeEmbeddingModel(),
            reranker=FakeReranker(),
            verifier=NoEvidenceVerifier(),
            candidate_k=10,
            rerank_k=5,
            verify_k=1,
        )
        output = StringIO()
        with redirect_stdout(output):
            cli._print_verification_output(
                self.corpus,
                "This passage is scientific evidence.",
                run_id,
                warning,
                verified,
                verbose=True,
            )
        text = output.getvalue()
        self.assertIn("Passage-wide evidence (fallback: the verifier selected no individual sentences):", text)
        self.assertIn("Retrieval diagnostics:", text)
        with self.corpus.connect() as connection:
            sentence_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT display_id FROM sentences ORDER BY document_sentence_index"
                )
            ]
        self.assertGreaterEqual(len(sentence_ids), 2)
        for sentence_id in sentence_ids:
            self.assertIn(sentence_id, text)
        command = f"show-sentences --database {self.corpus.root} {' '.join(sentence_ids)}"
        self.assertIn(command, text)
        self.assertEqual(text.count("show-sentences --database"), 1)

    def test_not_mentioned_and_untagged_contradiction_keep_whole_passage_visible(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        for verifier, expected_label in (
            (NotMentionedVerifier(), "NOT_MENTIONED"),
            (NoTagContradictionVerifier(), "CONTRADICTS"),
        ):
            run_id, warning, verified = verify_claim(
                self.corpus,
                "An unrelated claim.",
                embedding_model=FakeEmbeddingModel(),
                reranker=FakeReranker(),
                verifier=verifier,
                candidate_k=10,
                rerank_k=5,
                verify_k=1,
            )
            output = StringIO()
            with redirect_stdout(output):
                cli._print_verification_output(
                    self.corpus,
                    "An unrelated claim.",
                    run_id,
                    warning,
                    verified,
                    verbose=False,
                )
            text = output.getvalue()
            self.assertEqual(verified[0].verification.label, expected_label)
            self.assertIn("Passage-wide evidence", text)
            self.assertIn("show-sentences --database", text)
            if expected_label == "NOT_MENTIONED":
                self.assertIn("this absence is not a contradiction", text)

    def test_show_sentences_renders_one_image_per_affected_page(self):
        multi_page_pdf = self.root / "multi-page.pdf"
        create_two_page_pdf(
            multi_page_pdf,
            [
                "The first page contains enough scientific words for evidence retrieval.",
                "The second page contains enough scientific words for evidence retrieval.",
            ],
        )
        ingest_pdf(self.corpus, multi_page_pdf, on_duplicate="discard")
        with self.corpus.connect() as connection:
            sentence_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT display_id FROM sentences ORDER BY document_sentence_index"
                )
            ]
        output_directory = self.root / "rendered"
        rendered_pages = render_sentences(self.corpus, sentence_ids, output_directory)
        self.assertEqual(len(rendered_pages), 2)
        self.assertTrue(all(rendered_page.output_path.exists() for rendered_page in rendered_pages))
        self.assertEqual(
            {sentence_id for rendered_page in rendered_pages for sentence_id in rendered_page.sentence_ids},
            set(sentence_ids),
        )

    def test_show_sentences_validates_all_ids_before_creating_output(self):
        output_directory = self.root / "invalid-rendered"
        with self.assertRaises(CorpusError):
            render_sentences(self.corpus, ["missing-sentence-id"], output_directory)
        self.assertFalse(output_directory.exists())

    def test_stale_index_requires_explicit_noninteractive_override(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        second = self.root / "second.pdf"
        create_pdf(second, "Another scientific passage contains enough words for independent evidence retrieval now.")
        ingest_pdf(self.corpus, second, on_duplicate="discard")

        stderr = StringIO()
        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch("cite_this_paper.cli.verify_claim") as verify:
            stdin.isatty.return_value = False
            with redirect_stderr(stderr):
                exit_code = cli.main([
                    "verify-claim", "--database", str(self.corpus.root), "scientific evidence",
                ])
        self.assertEqual(exit_code, 2)
        verify.assert_not_called()
        self.assertIn("--allow-stale-index", stderr.getvalue())

        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch(
            "cite_this_paper.cli.verify_claim", return_value=(99, "stale index", [])
        ) as verify:
            stdin.isatty.return_value = False
            exit_code = cli.main([
                "verify-claim", "--database", str(self.corpus.root), "--allow-stale-index", "scientific evidence",
            ])
        self.assertEqual(exit_code, 0)
        verify.assert_called_once()

    def test_interactive_stale_index_choices_continue_or_quit(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        second = self.root / "second.pdf"
        create_pdf(second, "Another scientific passage contains enough words for independent evidence retrieval now.")
        ingest_pdf(self.corpus, second, on_duplicate="discard")
        args = SimpleNamespace(allow_stale_index=False)

        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch("builtins.input", return_value="continue"):
            stdin.isatty.return_value = True
            self.assertTrue(cli._prepare_verification(self.corpus, args))
        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch("builtins.input", return_value="quit"):
            stdin.isatty.return_value = True
            self.assertFalse(cli._prepare_verification(self.corpus, args))
        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch(
            "builtins.input", return_value="rebuild"
        ), patch(
            "cite_this_paper.cli.rebuild_index",
            return_value=IndexResult(4, 2, self.corpus.matrix_path),
        ) as rebuild:
            stdin.isatty.return_value = True
            self.assertTrue(cli._prepare_verification(self.corpus, args))
        rebuild.assert_called_once_with(self.corpus)

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

        new_shell = CorpusShell(environment_root, stdout=StringIO())
        self.assertIsNone(new_shell.active_name)

    def test_catalog_list_info_and_incompatible_corpus_are_safe(self):
        catalog_root = self.root / "catalog"
        healthy = Corpus.create(catalog_root / "healthy")
        legacy = Corpus.create(catalog_root / "legacy")
        with legacy.connect() as connection:
            connection.execute("UPDATE corpus_state SET schema_version = 1 WHERE id = 1")
            connection.commit()

        output = StringIO()
        shell = CorpusShell(catalog_root, stdout=output)
        shell.onecmd("load healthy")
        shell.onecmd("list")
        list_text = output.getvalue()
        shell.onecmd("info healthy")
        shell.onecmd("info legacy")
        text = output.getvalue()
        self.assertIn("healthy", text)
        self.assertIn("legacy", text)
        self.assertIn("incompatible", text)
        self.assertNotIn("Schema 1", list_text)
        self.assertIn("Schema 1", text)
        self.assertIn("Content: 0 documents", text)

        with self.assertRaises(CorpusError):
            Corpus.open(legacy.root)
        shell.onecmd("load legacy")
        self.assertIn("Recreate and reingest", output.getvalue())

    def test_catalog_views_handle_an_empty_vector_file(self):
        catalog_root = self.root / "catalog"
        corpus = Corpus.create(catalog_root / "empty-vectors")
        corpus.matrix_path.touch()

        output = StringIO()
        shell = CorpusShell(catalog_root, stdout=output)
        shell.onecmd("list")
        shell.onecmd("info empty-vectors")

        self.assertIn("empty-vectors", output.getvalue())
        self.assertIn("Index data: 0 rows, - dimensions", output.getvalue())

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

    def test_shell_cleanup_keeps_all_targets_when_one_name_is_invalid(self):
        catalog_root = self.root / "catalog"
        Corpus.create(catalog_root / "keep-me")
        output = StringIO()
        shell = CorpusShell(catalog_root, stdout=output)
        shell.onecmd("cleanup keep-me missing --apply")
        self.assertTrue((catalog_root / "keep-me").exists())
        self.assertIn("INVALID", output.getvalue())

    def test_shell_info_for_missing_corpus_is_one_concise_error(self):
        output = StringIO()
        shell = CorpusShell(self.root / "catalog", stdout=output)
        shell.onecmd("info missing")
        self.assertEqual(output.getvalue(), "ERROR: Corpus 'missing' does not exist.\n")

    def test_shell_help_lists_every_supported_command(self):
        output = StringIO()
        shell = CorpusShell(self.root / "catalog", stdout=output)
        shell.onecmd("help")
        text = output.getvalue()
        for command in (
            "create NAME", "load NAME", "logout", "list", "info [NAME]", "cleanup NAME",
            "settings", "add-pdf", "add-directory", "rebuild-index", "verify-claim",
            "show-sentences", "help [COMMAND]", "exit", "quit",
        ):
            self.assertIn(command, text)
        self.assertIn("Create and activate a named corpus.", text)

    def test_shell_settings_control_prompt_and_persist(self):
        settings_path = self.root / "user-settings.json"
        output = TtyStringIO()
        with patch.dict(os.environ, {}, clear=True):
            shell = CorpusShell(self.root / "catalog", stdout=output, settings_path=settings_path)
        self.assertIn("\033[1;36m", shell.prompt)
        self.assertIn("\001\033[1;36m\002", shell.prompt)
        self.assertIn("\001\033[0m\002", shell.prompt)

        shell.onecmd("settings set prompt.color magenta")
        shell.onecmd('settings set prompt.marker ">"')
        shell.onecmd("settings set color.mode never")
        self.assertEqual(shell.prompt, "cite-this-paper [no corpus] > ")
        self.assertEqual(json.loads(settings_path.read_text(encoding="utf-8"))["prompt"]["color"], "magenta")

        reloaded = CorpusShell(self.root / "catalog", stdout=TtyStringIO(), settings_path=settings_path)
        self.assertEqual(reloaded.settings.prompt_color, "magenta")
        self.assertEqual(reloaded.settings.prompt_marker, ">")
        self.assertEqual(reloaded.settings.color_mode, "never")

        shell.onecmd("settings reset prompt.color")
        self.assertEqual(shell.settings.prompt_color, "cyan")
        shell.onecmd("settings set prompt.color orange")
        self.assertIn("prompt.color must be one of", output.getvalue())

    def test_shell_settings_fall_back_when_file_is_invalid_and_honor_no_color(self):
        settings_path = self.root / "invalid-settings.json"
        settings_path.write_text("{not json", encoding="utf-8")
        output = TtyStringIO()
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            shell = CorpusShell(self.root / "catalog", stdout=output, settings_path=settings_path)
        self.assertNotIn("\033[", shell.prompt)
        self.assertIn("Ignoring invalid settings file", output.getvalue())

    def test_shell_settings_fall_back_when_file_is_not_utf8(self):
        settings_path = self.root / "invalid-settings.json"
        settings_path.write_bytes(b"\xff\xfe")
        output = StringIO()
        shell = CorpusShell(self.root / "catalog", stdout=output, settings_path=settings_path)
        self.assertEqual(shell.settings.prompt_color, "cyan")
        self.assertIn("Ignoring invalid settings file", output.getvalue())

    def test_session_parser_omits_database_but_direct_parser_keeps_it_required(self):
        session_args = cli.build_parser(session=True).parse_args(["show-sentences", "sentence-id"])
        self.assertFalse(hasattr(session_args, "database"))
        direct_parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            direct_parser.parse_args(["show-sentences", "sentence-id"])

    def test_evidence_command_matches_its_execution_context(self):
        self.assertEqual(
            cli._format_render_command(self.corpus, ["sentence-id"], interactive=True),
            "show-sentences sentence-id",
        )
        self.assertEqual(
            cli._format_render_command(self.corpus, ["sentence-id"]),
            f"cite-this-paper show-sentences --database {self.corpus.root} sentence-id",
        )

    def test_shell_completion_covers_commands_corpora_options_and_settings(self):
        catalog_root = self.root / "catalog"
        Corpus.create(catalog_root / "water")
        Corpus.create(catalog_root / "weather")
        shell = CorpusShell(catalog_root, stdout=StringIO(), settings_path=self.root / "settings.json")

        self.assertEqual(shell.completenames("ver"), ["verify-claim"])
        self.assertEqual(shell.complete_load("w", "load w", 5, 6), ["water", "weather"])
        self.assertEqual(shell.complete_info("wea", "info wea", 5, 8), ["weather"])
        self.assertEqual(shell.complete_cleanup("wa", "cleanup wa", 8, 10), ["water"])
        self.assertIn(
            "--verbose",
            shell.completedefault("--v", "verify-claim --v", 13, 16),
        )
        self.assertEqual(
            shell.completedefault(
                "r", "add-pdf source.pdf --on-duplicate r", 34, 35
            ),
            ["replace"],
        )
        self.assertEqual(
            shell.completedefault(
                "--q", "verify-claim a claim --quiet --q", 29, 32
            ),
            [],
        )
        self.assertEqual(
            shell.complete_settings("prompt.", "settings set prompt.", 13, 20),
            ["prompt.color", "prompt.bold", "prompt.marker"],
        )
        self.assertEqual(
            shell.complete_settings("a", "settings set color.mode a", 24, 25),
            ["auto", "always"],
        )
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

    def test_shell_dispatches_hyphenated_operational_commands(self):
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
