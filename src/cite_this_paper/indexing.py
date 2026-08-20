"""Dense and lexical index construction for a corpus."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .corpus import Corpus, CorpusError


class EmbeddingModel(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one dense vector per text."""


class BGEEmbeddingModel:
    """Lazy BGE-M3 adapter so database-only commands have no model startup cost."""

    def __init__(self, name: str, batch_size: int = 32, max_length: int = 512):
        self.name = name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.name, use_fp16=True)
        output = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return normalize_rows(np.asarray(output["dense_vecs"], dtype=np.float32))


@dataclass(frozen=True)
class IndexResult:
    indexed_passages: int
    dimensions: int
    matrix_path: Path


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Embedding model must return a two-dimensional matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-12)


def rebuild_index(corpus: Corpus, model: EmbeddingModel | None = None) -> IndexResult:
    """Replace the current matrix and FTS contents with the current eligible passages."""
    config = corpus.config()
    model = model or BGEEmbeddingModel(config["embedding_model"])
    with corpus.connect() as connection:
        rows = connection.execute(
            """
            SELECT p.id, p.document_id, pg.page_number, p.content_type, p.normalized_text
            FROM passages AS p
            JOIN pages AS pg ON pg.id = p.page_id
            WHERE p.retrieval_eligible = 1
            ORDER BY p.id
            """
        ).fetchall()
    texts = [row["normalized_text"] for row in rows]
    if texts:
        matrix = normalize_rows(model.encode(texts))
    else:
        matrix = np.empty((0, 0), dtype=np.float32)
    temporary_path = corpus.vectors_path / f".embeddings-{uuid.uuid4().hex}.npy"
    np.save(temporary_path, matrix)

    try:
        with corpus.connect() as connection:
            connection.execute("UPDATE passages SET embedding_row = NULL")
            connection.execute("DELETE FROM passages_fts")
            for row_number, row in enumerate(rows):
                connection.execute("UPDATE passages SET embedding_row = ? WHERE id = ?", (row_number, row["id"]))
                connection.execute(
                    """
                    INSERT INTO passages_fts (passage_id, document_id, page_number, content_type, normalized_text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(row["id"]), str(row["document_id"]), row["page_number"], row["content_type"], row["normalized_text"]),
                )
            connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('optimize')")
            connection.execute(
                """
                UPDATE corpus_state
                SET index_status = 'ready', embedding_model = ?, embedding_config_json = ?,
                    indexed_passage_count = ?, last_indexed_at = ?
                WHERE id = 1
                """,
                (
                    model.name,
                    json.dumps({"batch_size": getattr(model, "batch_size", None), "max_length": getattr(model, "max_length", None)}),
                    len(rows),
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            connection.commit()
        os.replace(temporary_path, corpus.matrix_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    dimensions = int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else 0
    return IndexResult(len(rows), dimensions, corpus.matrix_path)


def require_matrix(corpus: Corpus) -> np.ndarray:
    """Load the active matrix or explain why the corpus is not queryable yet."""
    if not corpus.matrix_path.exists():
        raise CorpusError("This corpus has no index yet. Run rebuild-index first.")
    matrix = np.load(corpus.matrix_path)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise CorpusError("This corpus has no retrieval-eligible passages.")
    return matrix
