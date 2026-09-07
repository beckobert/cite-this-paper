from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from cite_this_paper import cli
from cite_this_paper.corpus import CorpusError
from cite_this_paper.indexing import IndexResult, rebuild_index
from cite_this_paper.ingest import ingest_pdf
from cite_this_paper.models import VERDICT_LABELS, VERIFIER_PROMPT, parse_verification_output
from cite_this_paper.review import render_sentences
from cite_this_paper.retrieval import verify_claim

from test_support import (
    ClosingEmbeddingModel,
    ClosingReranker,
    ClosingVerifier,
    CorpusTestCase,
    FakeEmbeddingModel,
    FakeReranker,
    FakeVerifier,
    NoEvidenceVerifier,
    NoTagContradictionVerifier,
    NotMentionedVerifier,
    create_pdf,
    create_two_page_pdf,
)


class VerificationAndReviewTests(CorpusTestCase):
    def _index_document(self) -> None:
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())

    def _make_stale_index(self) -> None:
        self._index_document()
        second = self.root / "second.pdf"
        create_pdf(second, "Another scientific passage contains enough words for independent evidence retrieval now.")
        ingest_pdf(self.corpus, second, on_duplicate="discard")

    def _verify(self, verifier, claim: str = "This passage is scientific evidence."):
        return verify_claim(
            self.corpus,
            claim,
            embedding_model=FakeEmbeddingModel(),
            reranker=FakeReranker(),
            verifier=verifier,
            candidate_k=10,
            rerank_k=5,
            verify_k=1,
        )

    def _render_verification_output(self, run_id, warning, verified, *, claim: str, verbose: bool, interactive: bool = False) -> str:
        output = StringIO()
        with redirect_stdout(output):
            cli._print_verification_output(
                self.corpus, claim, run_id, warning, verified, verbose=verbose, interactive=interactive
            )
        return output.getvalue()

    def test_automatic_models_are_released_after_verification(self):
        self._index_document()
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

    def test_verifier_label_contract_distinguishes_unrelated_content(self):
        self.assertIn("NOT_MENTIONED", VERDICT_LABELS)
        self.assertNotIn("REFERENCES", VERDICT_LABELS)
        self.assertIn("NEVER a contradiction", VERIFIER_PROMPT)
        self.assertIn("NOT_MENTIONED", VERIFIER_PROMPT)

        accepted = parse_verification_output('{"label": "NOT_MENTIONED", "evidence": [], "reason": "Unrelated."}')
        rejected = parse_verification_output('{"label": "REFERENCES", "evidence": [], "reason": "Legacy label."}')
        self.assertEqual((accepted.label, accepted.parse_success), ("NOT_MENTIONED", True))
        self.assertEqual((rejected.label, rejected.parse_success), ("VERIFICATION_ERROR", False))

    def test_pending_documents_warn_after_a_previous_rebuild(self):
        self._make_stale_index()
        _, warning, _ = self._verify(FakeVerifier(), claim="scientific evidence")
        self.assertIn("pending index rebuild", warning)

    def test_verification_output_shows_selected_evidence_and_contextual_render_commands(self):
        self._index_document()
        claim = "This passage is scientific evidence."
        run_id, warning, verified = self._verify(FakeVerifier(), claim)
        direct_output = self._render_verification_output(run_id, warning, verified, claim=claim, verbose=False)
        shell_output = self._render_verification_output(run_id, warning, verified, claim=claim, verbose=False, interactive=True)
        sentence_id = self.corpus.connect().execute("SELECT display_id FROM sentences LIMIT 1").fetchone()[0]

        self.assertIn("Verifier-selected evidence:", direct_output)
        self.assertIn(sentence_id, direct_output)
        self.assertIn(f"show-sentences --database {self.corpus.root} {sentence_id}", direct_output)
        self.assertIn(cli.RESPONSIBILITY_NOTICE, direct_output)
        self.assertNotIn("Retrieval diagnostics:", direct_output)
        self.assertIn(f"show-sentences {sentence_id}", shell_output)
        self.assertNotIn("--database", shell_output)

    def test_verification_output_falls_back_to_whole_passage_without_sentence_tags(self):
        multi_sentence_pdf = self.root / "multi-sentence.pdf"
        create_pdf(
            multi_sentence_pdf,
            "This first scientific sentence is eligible evidence. "
            "This second scientific sentence provides additional context.",
        )
        ingest_pdf(self.corpus, multi_sentence_pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        with self.corpus.connect() as connection:
            sentence_ids = [
                row[0]
                for row in connection.execute("SELECT display_id FROM sentences ORDER BY document_sentence_index")
            ]

        cases = (
            (NoEvidenceVerifier(), "RELATED_ONLY", "fallback: the verifier selected no individual sentences", True, None),
            (NotMentionedVerifier(), "NOT_MENTIONED", "Passage-wide evidence", False, "this absence is not a contradiction"),
            (NoTagContradictionVerifier(), "CONTRADICTS", "Passage-wide evidence", False, None),
        )
        for verifier, label, expected_text, verbose, interpretation in cases:
            with self.subTest(label=label):
                claim = "An unrelated claim." if label != "RELATED_ONLY" else "This passage is scientific evidence."
                run_id, warning, verified = self._verify(verifier, claim)
                text = self._render_verification_output(run_id, warning, verified, claim=claim, verbose=verbose)
                self.assertEqual(verified[0].verification.label, label)
                self.assertIn(expected_text, text)
                self.assertIn("show-sentences --database", text)
                if label == "RELATED_ONLY":
                    self.assertIn("Retrieval diagnostics:", text)
                    for sentence_id in sentence_ids:
                        self.assertIn(sentence_id, text)
                if interpretation:
                    self.assertIn(interpretation, text)

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
                for row in connection.execute("SELECT display_id FROM sentences ORDER BY document_sentence_index")
            ]
        rendered_pages = render_sentences(self.corpus, sentence_ids, self.root / "rendered")
        self.assertEqual(len(rendered_pages), 2)
        self.assertTrue(all(page.output_path.exists() for page in rendered_pages))
        self.assertEqual(
            {sentence_id for page in rendered_pages for sentence_id in page.sentence_ids}, set(sentence_ids)
        )

    def test_show_sentences_validates_all_ids_before_creating_output(self):
        output_directory = self.root / "invalid-rendered"
        with self.assertRaises(CorpusError):
            render_sentences(self.corpus, ["missing-sentence-id"], output_directory)
        self.assertFalse(output_directory.exists())

    def test_stale_index_requires_explicit_noninteractive_override(self):
        self._make_stale_index()
        stderr = StringIO()
        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch("cite_this_paper.cli.verify_claim") as verify:
            stdin.isatty.return_value = False
            with redirect_stderr(stderr):
                exit_code = cli.main(["verify-claim", "--database", str(self.corpus.root), "scientific evidence"])
        self.assertEqual(exit_code, 2)
        verify.assert_not_called()
        self.assertIn("--allow-stale-index", stderr.getvalue())

        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch(
            "cite_this_paper.cli.verify_claim", return_value=(99, "stale index", [])
        ) as verify:
            stdin.isatty.return_value = False
            exit_code = cli.main([
                "verify-claim", "--database", str(self.corpus.root), "--allow-stale-index", "scientific evidence"
            ])
        self.assertEqual(exit_code, 0)
        verify.assert_called_once()

    def test_interactive_stale_index_choices_continue_quit_or_rebuild(self):
        self._make_stale_index()
        args = SimpleNamespace(allow_stale_index=False)
        for answer, expected in (("continue", True), ("quit", False)):
            with self.subTest(answer=answer), patch("cite_this_paper.cli.sys.stdin") as stdin, patch(
                "builtins.input", return_value=answer
            ):
                stdin.isatty.return_value = True
                self.assertEqual(cli._prepare_verification(self.corpus, args), expected)

        with patch("cite_this_paper.cli.sys.stdin") as stdin, patch(
            "builtins.input", return_value="rebuild"
        ), patch(
            "cite_this_paper.cli.rebuild_index", return_value=IndexResult(4, 2, self.corpus.matrix_path)
        ) as rebuild:
            stdin.isatty.return_value = True
            self.assertTrue(cli._prepare_verification(self.corpus, args))
        rebuild.assert_called_once_with(self.corpus)
