from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from cite_this_paper import cli
from cite_this_paper.indexing import rebuild_index
from cite_this_paper.ingest import ingest_pdf
from cite_this_paper.models import VERIFIER_PROMPT_VERSION
from cite_this_paper.processing import sentences as sentence_processing
from cite_this_paper.retrieval import verify_claim

from test_support import CorpusTestCase, FakeEmbeddingModel, FakeReranker, FakeVerifier, RecordingReporter


class IngestionAndIndexingTests(CorpusTestCase):
    def test_ingestion_deduplicates_and_keeps_highlight_provenance(self):
        first = ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        discarded = ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        replaced = ingest_pdf(self.corpus, self.pdf, on_duplicate="replace")

        self.assertEqual((first.status, discarded.status, replaced.status), ("added", "discarded", "replaced"))
        with self.corpus.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM documents").fetchone()[0], 1)
            for table in ("sentences", "sentence_boxes", "passages"):
                self.assertGreater(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

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
            status = connection.execute(
                "SELECT status FROM verification_runs WHERE id = ?", (run_id,)
            ).fetchone()[0]
            evidence_count = connection.execute("SELECT count(*) FROM verification_evidence").fetchone()[0]
            config = json.loads(
                connection.execute(
                    "SELECT configuration_json FROM verification_runs WHERE id = ?", (run_id,)
                ).fetchone()[0]
            )
        self.assertEqual(status, "completed")
        self.assertGreater(evidence_count, 0)
        self.assertEqual(config["verifier_prompt_version"], VERIFIER_PROMPT_VERSION)

    def test_processing_reports_key_stages_and_completes_verifier_progress(self):
        reporter = RecordingReporter()
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard", reporter=reporter)
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard", reporter=reporter)
        rebuild_index(self.corpus, FakeEmbeddingModel(), reporter=reporter)
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

        self.assertIn("Processing PDF: paper.pdf", reporter.stages)
        self.assertIn("Duplicate document found; keeping the existing copy.", reporter.stages)
        self.assertIn("Creating embeddings for 1 passage(s)...", reporter.stages)
        self.assertIn("Creating claim embedding and retrieving dense and lexical candidates...", reporter.stages)
        self.assertIn("Saving the verification audit record...", reporter.stages)
        self.assertEqual(len(reporter.bars), 1)
        bar = reporter.bars[0]
        self.assertEqual((bar.description, bar.total, bar.advanced, bar.closed), ("Verifying passages", 1, 1, True))

    def test_ingestion_report_summarizes_results_and_failures(self):
        output = StringIO()
        results = [
            cli.IngestResult(self.pdf, "added"),
            cli.IngestResult(self.root / "duplicate.pdf", "discarded"),
            cli.IngestResult(self.root / "broken.pdf", "failed", message="Unreadable PDF"),
        ]
        with redirect_stdout(output):
            cli._print_ingestion_report(
                self.corpus, results, "Deferred (new PDFs are not searchable until rebuild-index runs)"
            )
        text = output.getvalue()
        for expected in ("INGESTION REPORT", "Processed: 3 PDF(s)", "Added:     1", "Duplicates kept: 1", "Failed:    1", "broken.pdf: Unreadable PDF"):
            self.assertIn(expected, text)

    def test_quiet_ingestion_suppresses_updates_but_keeps_the_final_report(self):
        output = StringIO()
        with (
            patch("cite_this_paper.cli._ingest_one", return_value=cli.IngestResult(self.pdf, "added")),
            redirect_stdout(output),
        ):
            result = cli.main([
                "add-pdf", "--database", str(self.corpus.root), str(self.pdf), "--defer-rebuild", "--quiet"
            ])
        self.assertEqual(result, 0)
        self.assertIn("INGESTION REPORT", output.getvalue())
        self.assertNotIn("Processing PDF:", output.getvalue())

    def test_physical_block_merge_diagnostics_require_debug(self):
        page = {"document_id": "paper", "page_number": 1, "words": [{"text": "placeholder"}]}
        physical_blocks = OrderedDict([(1, []), (2, [])])

        def build_sentences(*, debug: bool) -> str:
            output = StringIO()
            with (
                patch.object(sentence_processing, "group_words_by_block_and_line", return_value=physical_blocks),
                patch.object(sentence_processing, "build_logical_block_groups", return_value=[[1, 2]]),
                patch.object(sentence_processing, "reconstruct_logical_block", return_value=("", [])),
                redirect_stdout(output),
            ):
                sentence_processing.build_sentences_for_page(page, None, 0, debug=debug)
            return output.getvalue()

        self.assertEqual(build_sentences(debug=False), "")
        self.assertIn("merged physical blocks [1, 2]", build_sentences(debug=True))
