from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pymupdf

from cite_this_paper.corpus import Corpus
from cite_this_paper import cli
from cite_this_paper.indexing import IndexResult, rebuild_index
from cite_this_paper.ingest import ingest_pdf
from cite_this_paper.models import VerificationOutput
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


def create_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
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
        self.assertIn(f"show-sentence --database {self.corpus.root} {sentence_id}", text)
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
            self.assertIn(f"show-sentence --database {self.corpus.root} {sentence_id}", text)

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
