"""SQLite schema and connection helpers for a single corpus."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 2


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS corpus_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    index_status TEXT NOT NULL CHECK (index_status IN ('empty', 'ready', 'rebuild_required')),
    embedding_model TEXT,
    embedding_config_json TEXT NOT NULL DEFAULT '{}',
    indexed_passage_count INTEGER NOT NULL DEFAULT 0,
    last_indexed_at TEXT,
    last_accessed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    stored_path TEXT NOT NULL,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    title TEXT,
    authors_json TEXT NOT NULL DEFAULT '[]',
    publication_year INTEGER,
    journal TEXT,
    volume TEXT,
    issue TEXT,
    page_range TEXT,
    doi TEXT,
    abstract TEXT,
    citation_key TEXT,
    raw_metadata_json TEXT NOT NULL DEFAULT '{}',
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_doi_idx ON documents(doi);

CREATE TABLE IF NOT EXISTS ingestion_errors (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    rotation INTEGER NOT NULL,
    extraction_status TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    block_count INTEGER NOT NULL,
    UNIQUE(document_id, page_index)
);

CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    display_id TEXT NOT NULL UNIQUE,
    page_sentence_index INTEGER NOT NULL,
    document_sentence_index INTEGER NOT NULL,
    logical_block_index INTEGER NOT NULL,
    primary_block_no INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    character_count INTEGER NOT NULL,
    source_word_count INTEGER NOT NULL,
    source_word_start INTEGER,
    source_word_end INTEGER,
    UNIQUE(page_id, page_sentence_index)
);

CREATE INDEX IF NOT EXISTS sentences_document_idx ON sentences(document_id, document_sentence_index);

CREATE TABLE IF NOT EXISTS sentence_boxes (
    sentence_id INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    box_index INTEGER NOT NULL,
    block_no INTEGER NOT NULL,
    line_no INTEGER NOT NULL,
    x0 REAL NOT NULL,
    y0 REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    PRIMARY KEY(sentence_id, box_index)
);

CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    display_id TEXT NOT NULL UNIQUE,
    logical_block_index INTEGER NOT NULL,
    block_passage_index INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    sentence_count INTEGER NOT NULL,
    retrieval_eligible_base INTEGER NOT NULL CHECK (retrieval_eligible_base IN (0, 1)),
    retrieval_eligible INTEGER NOT NULL CHECK (retrieval_eligible IN (0, 1)),
    content_type TEXT NOT NULL CHECK (content_type IN ('body', 'end_matter')),
    classification_json TEXT,
    embedding_row INTEGER UNIQUE,
    UNIQUE(document_id, page_id, logical_block_index, block_passage_index)
);

CREATE INDEX IF NOT EXISTS passages_eligible_idx ON passages(retrieval_eligible, embedding_row);

CREATE TABLE IF NOT EXISTS passage_sentences (
    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    sentence_id INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY(passage_id, position),
    UNIQUE(passage_id, sentence_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    passage_id UNINDEXED,
    document_id UNINDEXED,
    page_number UNINDEXED,
    content_type UNINDEXED,
    normalized_text,
    tokenize = 'unicode61 remove_diacritics 1'
);

CREATE TABLE IF NOT EXISTS verification_runs (
    id INTEGER PRIMARY KEY,
    claim TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    reranker_model TEXT NOT NULL,
    verifier_model TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    warning TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS verification_candidates (
    id INTEGER PRIMARY KEY,
    verification_run_id INTEGER NOT NULL REFERENCES verification_runs(id) ON DELETE CASCADE,
    passage_id INTEGER NOT NULL REFERENCES passages(id),
    dense_rank INTEGER,
    dense_score REAL,
    bm25_rank INTEGER,
    bm25_score REAL,
    fusion_rank INTEGER,
    fusion_score REAL,
    rerank_rank INTEGER,
    rerank_score REAL,
    rerank_logit REAL,
    verifier_label TEXT,
    verifier_reason TEXT,
    verifier_raw_output TEXT,
    verifier_parse_success INTEGER,
    UNIQUE(verification_run_id, passage_id)
);

CREATE TABLE IF NOT EXISTS verification_evidence (
    verification_candidate_id INTEGER NOT NULL REFERENCES verification_candidates(id) ON DELETE CASCADE,
    sentence_id INTEGER NOT NULL REFERENCES sentences(id),
    position INTEGER NOT NULL,
    PRIMARY KEY(verification_candidate_id, position)
);
"""


def connect(database_path: Path) -> sqlite3.Connection:
    """Open a corpus connection with the project-wide SQLite settings."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize(database_path: Path) -> None:
    """Create the schema for an empty corpus database."""
    connection = connect(database_path)
    try:
        connection.executescript(SCHEMA)
        connection.execute(
            """
            INSERT INTO corpus_state (id, schema_version, index_status, last_accessed_at)
            VALUES (1, ?, 'empty', ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (SCHEMA_VERSION, datetime.now(UTC).isoformat(timespec="seconds")),
        )
        connection.commit()
    finally:
        connection.close()
