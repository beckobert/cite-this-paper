from __future__ import annotations

import json
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
