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
from .embeddings import (
    create_embedding_backend,
    descriptor_for_legacy_model,
    encode_passages,
)
from .progress import ProgressReporter, report_stage


class EmbeddingModel(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one dense vector per text."""


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


def _collect_model_memory() -> None:
    """Best-effort release that keeps CPU-only installations lightweight."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def close_model(model: object) -> None:
    """Close built-in model adapters without imposing cleanup on custom models."""
    close = getattr(model, "close", None)
    if callable(close):
        close()


def rebuild_index(
    corpus: Corpus,
    model: EmbeddingModel | None = None,
    *,
    reporter: ProgressReporter | None = None,
) -> IndexResult:
    """Replace the current matrix and FTS contents with the current eligible passages."""
    owns_model = model is None
    if model is None:
        model, descriptor = create_embedding_backend(corpus.embedding_spec(), corpus.root)
    else:
        descriptor = descriptor_for_legacy_model(model)
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
    try:
        if texts:
            report_stage(reporter, f"Loading embedding model: {model.name}")
            report_stage(reporter, f"Creating embeddings for {len(texts)} passage(s)...")
            matrix = normalize_rows(encode_passages(model, texts))
            report_stage(reporter, "Passage embeddings created.")
        else:
            report_stage(reporter, "No retrieval-eligible passages found; creating an empty index.")
            matrix = np.empty((0, 0), dtype=np.float32)
        temporary_path = corpus.vectors_path / f".embeddings-{uuid.uuid4().hex}.npy"
        report_stage(reporter, "Writing the new vector matrix...")
        np.save(temporary_path, matrix)

        with corpus.connect() as connection:
            report_stage(reporter, "Rebuilding the lexical search index...")
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
                    getattr(model, "name", type(model).__name__),
                    json.dumps(descriptor, sort_keys=True),
                    len(rows),
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            connection.commit()
        report_stage(reporter, "Activating the rebuilt index...")
        os.replace(temporary_path, corpus.matrix_path)
    except Exception:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if owns_model:
            report_stage(reporter, "Dropping embedding model from memory...")
            close_model(model)
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
