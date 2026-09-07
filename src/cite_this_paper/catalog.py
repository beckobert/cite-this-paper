"""Named-corpus discovery and summaries for the interactive shell."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cleanup import corpus_size
from .corpus import Corpus, CorpusError
from .schema import SCHEMA_VERSION


DEFAULT_CORPUS_ROOT = Path("data/corpora")
CORPUS_NAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


@dataclass(frozen=True)
class CorpusSummary:
    """A read-only catalog entry suitable for listing and information views."""

    name: str
    root: Path
    status: str
    message: str | None
    schema_version: int | None = None
    index_status: str | None = None
    document_count: int | None = None
    page_count: int | None = None
    sentence_count: int | None = None
    passage_count: int | None = None
    eligible_passage_count: int | None = None
    verification_run_count: int | None = None
    ingestion_error_count: int | None = None
    embedding_model: str | None = None
    indexed_passage_count: int | None = None
    embedding_dimensions: int | None = None
    last_indexed_at: str | None = None
    last_accessed_at: str | None = None
    size_bytes: int = 0


def resolve_catalog_root(explicit_root: Path | None = None) -> Path:
    """Resolve a catalog root from an option, environment, or project default."""
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()
    environment_root = os.environ.get("CITE_THIS_PAPER_ROOT")
    if environment_root:
        return Path(environment_root).expanduser().resolve()
    return DEFAULT_CORPUS_ROOT.resolve()


def validate_corpus_name(name: str) -> str:
    """Accept one safe direct-child corpus name, never an arbitrary path."""
    if not CORPUS_NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise CorpusError(
            "Corpus names must contain only letters, digits, '.', '_', or '-', "
            "and cannot be paths."
        )
    return name


def corpus_path(root: Path, name: str) -> Path:
    """Return a named corpus's direct-child path beneath its catalog root."""
    name = validate_corpus_name(name)
    root = root.expanduser().resolve()
    path = root / name
    try:
        path.relative_to(root)
    except ValueError as error:  # Defensive even though names are validated.
        raise CorpusError(f"Corpus name escapes the catalog root: {name}") from error
    return path


def create_named_corpus(root: Path, name: str) -> Corpus:
    """Create a new named corpus without reusing an existing directory."""
    path = corpus_path(root, name)
    if path.exists():
        raise CorpusError(f"A corpus named '{name}' already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return Corpus.create(path)


def open_named_corpus(root: Path, name: str) -> Corpus:
    """Open one current-schema named corpus for normal work."""
    path = corpus_path(root, name)
    if path.is_symlink():
        raise CorpusError("Symlinked corpus directories cannot be opened from the catalog.")
    return Corpus.open(path)


def _count(connection: sqlite3.Connection, table: str, where: str = "") -> int:
    return int(connection.execute(f"SELECT count(*) FROM {table}{where}").fetchone()[0])


def _matrix_dimensions(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        matrix = np.load(path, mmap_mode="r")
        return int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else 0
    except (EOFError, OSError, ValueError):
        return None


def inspect_corpus(root: Path, name: str) -> CorpusSummary:
    """Read a corpus summary without requiring schema compatibility."""
    path = corpus_path(root, name)
    if path.is_symlink():
        return CorpusSummary(
            name,
            path,
            "invalid",
            "Symlinked corpus directories are not supported.",
        )
    database_path = path / "corpus.sqlite"
    if database_path.is_symlink():
        return CorpusSummary(name, path, "invalid", "Corpus database files cannot be symlinks.")
    if not path.is_dir() or not database_path.is_file():
        return CorpusSummary(name, path, "invalid", "Not a corpus database directory.")

    try:
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        try:
            state = connection.execute("SELECT * FROM corpus_state WHERE id = 1").fetchone()
            if state is None:
                return CorpusSummary(name, path, "invalid", "Missing corpus state.", size_bytes=corpus_size(path))
            schema_version = int(state["schema_version"])
            size = corpus_size(path)
            if schema_version != SCHEMA_VERSION:
                return CorpusSummary(
                    name,
                    path,
                    "incompatible",
                    f"Schema {schema_version}; this version requires schema {SCHEMA_VERSION}. Recreate and reingest this corpus.",
                    schema_version=schema_version,
                    index_status=state["index_status"],
                    last_indexed_at=state["last_indexed_at"],
                    size_bytes=size,
                )
            return CorpusSummary(
                name=name,
                root=path,
                status="ready",
                message=None,
                schema_version=schema_version,
                index_status=state["index_status"],
                document_count=_count(connection, "documents"),
                page_count=_count(connection, "pages"),
                sentence_count=_count(connection, "sentences"),
                passage_count=_count(connection, "passages"),
                eligible_passage_count=_count(connection, "passages", " WHERE retrieval_eligible = 1"),
                verification_run_count=_count(connection, "verification_runs"),
                ingestion_error_count=_count(connection, "ingestion_errors"),
                embedding_model=state["embedding_model"],
                indexed_passage_count=int(state["indexed_passage_count"]),
                embedding_dimensions=_matrix_dimensions(path / "vectors" / "embeddings.npy"),
                last_indexed_at=state["last_indexed_at"],
                last_accessed_at=state["last_accessed_at"],
                size_bytes=size,
            )
        finally:
            connection.close()
    except sqlite3.Error as error:
        return CorpusSummary(name, path, "invalid", f"Could not read corpus database: {error}", size_bytes=corpus_size(path))


def list_corpora(root: Path) -> list[CorpusSummary]:
    """List direct-child named corpora without creating the catalog root."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    return [
        inspect_corpus(root, child.name)
        for child in sorted(root.iterdir(), key=lambda path: path.name.casefold())
        if child.is_dir() and (child / "corpus.sqlite").is_file()
    ]
