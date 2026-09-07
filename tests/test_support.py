from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import numpy as np
import pymupdf

from cite_this_paper.corpus import Corpus


class FakeEmbeddingModel:
    name = "test-embedding"

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


class FakeReranker:
    name = "test-reranker"

    def rerank(self, claim: str, passages: Iterable[object]) -> list[tuple[float, float]]:
        return [(1.0 / (position + 1), -float(position)) for position, _ in enumerate(passages)]


class FakeVerifier:
    name = "test-verifier"

    def verify(self, claim: str, numbered_sentences: list[str]):
        from cite_this_paper.models import VerificationOutput

        return VerificationOutput("DIRECT_SUPPORT", ["S1"], "Test verdict")


class NoEvidenceVerifier:
    name = "test-no-evidence-verifier"

    def verify(self, claim: str, numbered_sentences: list[str]):
        from cite_this_paper.models import VerificationOutput

        return VerificationOutput("RELATED_ONLY", [], "No individual sentence was selected")


class NotMentionedVerifier:
    name = "test-not-mentioned-verifier"

    def verify(self, claim: str, numbered_sentences: list[str]):
        from cite_this_paper.models import VerificationOutput

        return VerificationOutput("NOT_MENTIONED", [], "The passage is unrelated to the claim")


class NoTagContradictionVerifier:
    name = "test-no-tag-contradiction-verifier"

    def verify(self, claim: str, numbered_sentences: list[str]):
        from cite_this_paper.models import VerificationOutput

        return VerificationOutput("CONTRADICTS", [], "Test contradiction without sentence tags")


class TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class RecordingProgress:
    def __init__(self, description: str, total: int):
        self.description = description
        self.total = total
        self.advanced = 0
        self.closed = False

    def advance(self, amount: int = 1) -> None:
        self.advanced += amount

    def close(self) -> None:
        self.closed = True


class RecordingReporter:
    def __init__(self):
        self.stages: list[str] = []
        self.bars: list[RecordingProgress] = []

    def stage(self, message: str) -> None:
        self.stages.append(message)

    def progress(self, description: str, total: int) -> RecordingProgress:
        bar = RecordingProgress(description, total)
        self.bars.append(bar)
        return bar


class ClosingEmbeddingModel(FakeEmbeddingModel):
    name = "closing-embedding"

    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ClosingReranker(FakeReranker):
    name = "closing-reranker"

    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ClosingVerifier(FakeVerifier):
    name = "closing-verifier"

    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


def create_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def create_two_page_pdf(path: Path, page_texts: Iterable[str]) -> None:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class CorpusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = Corpus.create(self.root / "corpus")
        self.pdf = self.root / "paper.pdf"
        create_pdf(self.pdf, "This scientific passage contains enough words to be eligible for evidence retrieval.")

    def tearDown(self) -> None:
        self.temporary.cleanup()
