"""Corpus directory lifecycle and shared database operations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import connect, initialize


class CorpusError(RuntimeError):
    """Raised when a corpus cannot satisfy an operation."""


class DuplicateDocumentError(CorpusError):
    """Raised when ingestion needs an explicit duplicate choice."""

    def __init__(self, document: dict[str, Any]):
        super().__init__(f"Document already exists: {document['filename']}")
        self.document = document


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Corpus:
    """Paths and connections belonging to one independent paper corpus."""

    root: Path

    @property
    def database_path(self) -> Path:
        return self.root / "corpus.sqlite"

    @property
    def pdfs_path(self) -> Path:
        return self.root / "pdfs"

    @property
    def vectors_path(self) -> Path:
        return self.root / "vectors"

    @property
    def matrix_path(self) -> Path:
        return self.vectors_path / "embeddings.npy"

    @property
    def config_path(self) -> Path:
        return self.root / "corpus-config.json"

    @classmethod
    def create(cls, root: Path) -> "Corpus":
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        corpus = cls(root)
        corpus.pdfs_path.mkdir(exist_ok=True)
        corpus.vectors_path.mkdir(exist_ok=True)
        initialize(corpus.database_path)
        if not corpus.config_path.exists():
            corpus.config_path.write_text(
                json.dumps(
                    {
                        "embedding_model": "BAAI/bge-m3",
                        "reranker_model": "Qwen/Qwen3-Reranker-4B",
                        "verifier_model": "Qwen/Qwen3-4B-Instruct-2507",
                        "passage_max_words": 180,
                        "passage_overlap_sentences": 1,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return corpus

    @classmethod
    def open(cls, root: Path) -> "Corpus":
        corpus = cls(root.expanduser().resolve())
        if not corpus.database_path.exists():
            raise CorpusError(f"Not a corpus database: {corpus.root}")
        return corpus

    def connect(self):
        return connect(self.database_path)

    def config(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def state(self) -> dict[str, Any]:
        with self.connect() as connection:
            return dict(connection.execute("SELECT * FROM corpus_state WHERE id = 1").fetchone())

    def touch_access(self) -> None:
        """Record use by a normal package command for retention cleanup."""
        with self.connect() as connection:
            connection.execute(
                "UPDATE corpus_state SET last_accessed_at = ? WHERE id = 1",
                (utc_now(),),
            )
            connection.commit()

    def store_pdf(self, source: Path, sha256: str, *, replace: bool) -> Path:
        suffix = source.suffix.lower() or ".pdf"
        target = self.pdfs_path / f"{sha256}{suffix}"
        if replace or not target.exists():
            shutil.copy2(source, target)
        return target

    def mark_rebuild_required(self) -> None:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT indexed_passage_count FROM corpus_state WHERE id = 1"
            ).fetchone()[0]
            status = "rebuild_required" if existing else "empty"
            connection.execute("UPDATE corpus_state SET index_status = ? WHERE id = 1", (status,))
            connection.commit()
