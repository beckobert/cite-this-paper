from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pymupdf

from cite_this_paper.corpus import Corpus
from cite_this_paper.indexing import rebuild_index
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
