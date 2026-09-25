"""Corpus directory lifecycle and shared database operations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, connect, initialize


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
            from .embeddings import default_embedding_config

            corpus.config_path.write_text(
                json.dumps(
                    {
                        "embedding": default_embedding_config(),
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
        if not corpus.database_path.is_file():
            raise CorpusError(f"Not a corpus database: {corpus.root}")
        try:
            with corpus.connect() as connection:
                row = connection.execute(
                    "SELECT schema_version FROM corpus_state WHERE id = 1"
                ).fetchone()
        except Exception as error:
            raise CorpusError(f"Could not open corpus database: {corpus.root}") from error
        if row is None:
            raise CorpusError(f"Corpus database has no state record: {corpus.root}")
        if int(row["schema_version"]) != SCHEMA_VERSION:
            raise CorpusError(
                f"Corpus schema version {row['schema_version']} is incompatible with this "
                f"version (requires {SCHEMA_VERSION}). Recreate and reingest the corpus."
            )
        return corpus

    def connect(self):
        return connect(self.database_path)

    def config(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def embedding_spec(self):
        """Return the desired embedding setup, including legacy BGE-only config."""
        from .embeddings import embedding_spec_from_config

        return embedding_spec_from_config(self.config())

    def configure_embedding(self, spec) -> bool:
        """Persist a desired backend and flag only incompatible active indexes.

        Returns whether the selected setup requires a rebuild of an existing index.
        """
        from .embeddings import active_descriptor, descriptor_for_spec, validate_embedding_spec

        validate_embedding_spec(spec)
        desired = descriptor_for_spec(spec, self.root)
        state = self.state()
        active = active_descriptor(state)
        # A rebuilt empty corpus still has an active index configuration.  Its
        # vector matrix happens to be empty, but changing the backend must not
        # silently make that index configuration appear current.
        requires_rebuild = active is not None and active.get("fingerprint") != desired["fingerprint"]
        config = self.config()
        config["embedding"] = spec.as_dict()
        config.pop("embedding_model", None)
        temporary = self.config_path.with_name(f".{self.config_path.name}.tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.config_path)
        if requires_rebuild:
            self.mark_rebuild_required()
        elif active is not None:
            self.restore_ready_index_if_current()
        return requires_rebuild

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

    def restore_ready_index_if_current(self) -> bool:
        """Clear a configuration-only rebuild flag when the active index is current.

        A pending document change leaves an eligible passage without an embedding
        row (and normally changes the count), so it must continue to require a
        rebuild.
        """
        if not self.matrix_path.is_file():
            return False
        with self.connect() as connection:
            state = connection.execute(
                "SELECT index_status, indexed_passage_count FROM corpus_state WHERE id = 1"
            ).fetchone()
            eligible_count = int(
                connection.execute(
                    "SELECT count(*) FROM passages WHERE retrieval_eligible = 1"
                ).fetchone()[0]
            )
            unindexed_count = int(
                connection.execute(
                    "SELECT count(*) FROM passages "
                    "WHERE retrieval_eligible = 1 AND embedding_row IS NULL"
                ).fetchone()[0]
            )
            if (
                state["index_status"] != "rebuild_required"
                or int(state["indexed_passage_count"]) != eligible_count
                or unindexed_count
            ):
                return False
            connection.execute("UPDATE corpus_state SET index_status = 'ready' WHERE id = 1")
            connection.commit()
            return True
