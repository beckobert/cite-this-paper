"""Idempotent PDF ingestion backed by the existing extraction heuristics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .processing.classification import classify_document
from .processing.passages import build_passages_for_block, group_sentences
from .processing.pdf_extraction import extract_pdf
from .processing.sentences import build_sentences_for_page, create_nlp

from .corpus import Corpus, CorpusError, DuplicateDocumentError, utc_now

DuplicatePolicy = Literal["ask", "discard", "replace"]


@dataclass(frozen=True)
class IngestResult:
    source: Path
    status: Literal["added", "discarded", "replaced", "failed"]
    document_id: int | None = None
    message: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_values(document: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    raw = document.get("metadata") or {}
    values: dict[str, Any] = {
        "title": raw.get("title") or None,
        "authors_json": json.dumps([raw["author"]]) if raw.get("author") else "[]",
        "publication_year": None,
        "journal": None,
        "volume": None,
        "issue": None,
        "page_range": None,
        "doi": None,
        "abstract": raw.get("subject") or None,
        "citation_key": None,
        "raw_metadata_json": json.dumps(raw, ensure_ascii=False),
    }
    if overrides:
        for key, value in overrides.items():
            if value is None or key not in values:
                continue
            values[key] = json.dumps(value) if key == "authors_json" and not isinstance(value, str) else value
    return values


def _build_sentences(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nlp = create_nlp("en")
    by_document_index: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for page in pages:
        if not page.get("words"):
            continue
        document_id = page["document_id"]
        starting_index = by_document_index.get(document_id, 0)
        sentences, next_index = build_sentences_for_page(page, nlp, starting_index)
        by_document_index[document_id] = next_index
        records.extend(sentences)
    return records


def _build_passages(sentences: list[dict[str, Any]], max_words: int, overlap_sentences: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block_sentences in group_sentences(sentences).values():
        records.extend(build_passages_for_block(block_sentences, max_words, overlap_sentences))
    return records


def _record_error(corpus: Corpus, source: Path, error: Exception) -> None:
    with corpus.connect() as connection:
        connection.execute(
            """
            INSERT INTO ingestion_errors (source_path, filename, error_type, error_message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(source), source.name, type(error).__name__, str(error), utc_now()),
        )
        connection.commit()


def _replace_duplicate(
    corpus: Corpus,
    source: Path,
    sha256: str,
    existing: dict[str, Any],
    metadata_overrides: dict[str, Any] | None,
) -> IngestResult:
    stored = corpus.store_pdf(source, sha256, replace=True)
    now = utc_now()
    assignments = [
        "stored_path = ?",
        "source_path = ?",
        "filename = ?",
        "byte_size = ?",
        "updated_at = ?",
    ]
    parameters: list[Any] = [str(stored), str(source), source.name, source.stat().st_size, now]
    metadata_columns = {
        "title", "authors_json", "publication_year", "journal", "volume", "issue",
        "page_range", "doi", "abstract", "citation_key",
    }
    for key, value in (metadata_overrides or {}).items():
        if key in metadata_columns and value is not None:
            assignments.append(f"{key} = ?")
            parameters.append(json.dumps(value) if key == "authors_json" and not isinstance(value, str) else value)
    parameters.append(existing["id"])
    with corpus.connect() as connection:
        connection.execute(
            f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?",
            parameters,
        )
        connection.commit()
    return IngestResult(source, "replaced", existing["id"])


def ingest_pdf(
    corpus: Corpus,
    source: Path,
    *,
    on_duplicate: DuplicatePolicy = "ask",
    metadata_overrides: dict[str, Any] | None = None,
) -> IngestResult:
    """Add one PDF, or apply the requested policy to an exact-content duplicate."""
    source = source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pdf":
        raise CorpusError(f"Not a PDF file: {source}")
    sha256 = sha256_file(source)
    with corpus.connect() as connection:
        row = connection.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    if row is not None:
        existing = dict(row)
        if on_duplicate == "ask":
            raise DuplicateDocumentError(existing)
        if on_duplicate == "discard":
            return IngestResult(source, "discarded", existing["id"])
        return _replace_duplicate(corpus, source, sha256, existing, metadata_overrides)

    try:
        extracted_document, pages = extract_pdf(source, source.parent)
        sentences = _build_sentences(pages)
        config = corpus.config()
        passages = _build_passages(
            sentences,
            int(config["passage_max_words"]),
            int(config["passage_overlap_sentences"]),
        )
        classified, report = classify_document(
            passages=passages,
            sentences={sentence["sentence_id"]: sentence for sentence in sentences},
            document=extracted_document,
            pdf_path=source,
        )
    except Exception as error:
        _record_error(corpus, source, error)
        return IngestResult(source, "failed", message=f"{type(error).__name__}: {error}")

    stored = corpus.store_pdf(source, sha256, replace=False)
    now = utc_now()
    metadata = _metadata_values(extracted_document, metadata_overrides)
    try:
        with corpus.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents (
                    sha256, stored_path, source_path, filename, byte_size, page_count,
                    title, authors_json, publication_year, journal, volume, issue, page_range,
                    doi, abstract, citation_key, raw_metadata_json, added_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sha256, str(stored), str(source), source.name, source.stat().st_size,
                    extracted_document["page_count"], metadata["title"], metadata["authors_json"],
                    metadata["publication_year"], metadata["journal"], metadata["volume"], metadata["issue"],
                    metadata["page_range"], metadata["doi"], metadata["abstract"], metadata["citation_key"],
                    metadata["raw_metadata_json"], now, now,
                ),
            )
            document_id = int(cursor.lastrowid)
            page_ids: dict[int, int] = {}
            for page in pages:
                page_cursor = connection.execute(
                    """
                    INSERT INTO pages (
                        document_id, page_index, page_number, width, height, rotation,
                        extraction_status, extraction_method, extracted_text, character_count,
                        word_count, block_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id, page["page_index"], page["page_number"], page["width"], page["height"],
                        page["rotation"], page["extraction_status"], page["text_extraction_method"], page["text"],
                        page["character_count"], page["word_count"], page["block_count"],
                    ),
                )
                page_ids[page["page_index"]] = int(page_cursor.lastrowid)

            sentence_ids: dict[str, int] = {}
            for sentence in sentences:
                sentence_cursor = connection.execute(
                    """
                    INSERT INTO sentences (
                        document_id, page_id, display_id, page_sentence_index,
                        document_sentence_index, logical_block_index, primary_block_no,
                        source_text, normalized_text, character_count, source_word_count,
                        source_word_start, source_word_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id, page_ids[sentence["page_index"]], sentence["sentence_id"],
                        sentence["page_sentence_index"], sentence["document_sentence_index"],
                        sentence.get("logical_block_index", sentence["block_no"]), sentence["block_no"],
                        sentence["text"], sentence["text_normalized"], sentence["character_count"],
                        sentence["source_word_count"], sentence.get("source_word_start"), sentence.get("source_word_end"),
                    ),
                )
                sentence_id = int(sentence_cursor.lastrowid)
                sentence_ids[sentence["sentence_id"]] = sentence_id
                for position, box in enumerate(sentence.get("boxes", [])):
                    x0, y0, x1, y1 = box["bbox"]
                    connection.execute(
                        """
                        INSERT INTO sentence_boxes (sentence_id, box_index, block_no, line_no, x0, y0, x1, y1)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (sentence_id, position, box["block_no"], box["line_no"], x0, y0, x1, y1),
                    )

            classification_json = json.dumps(report.get("cutoff"), ensure_ascii=False) if report.get("cutoff") else None
            for passage in classified:
                passage_cursor = connection.execute(
                    """
                    INSERT INTO passages (
                        document_id, page_id, display_id, logical_block_index, block_passage_index,
                        source_text, normalized_text, word_count, sentence_count,
                        retrieval_eligible_base, retrieval_eligible, content_type, classification_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id, page_ids[passage["page_index"]], passage["passage_id"],
                        passage.get("logical_block_index", passage["physical_block_nos"][0]),
                        passage["block_passage_index"], passage["text"], passage["text_normalized"],
                        passage["word_count"], passage["sentence_count"],
                        int(bool(passage.get("retrieval_eligible_base", passage["retrieval_eligible"]))),
                        int(bool(passage["retrieval_eligible"])), passage["content_type"], classification_json,
                    ),
                )
                passage_id = int(passage_cursor.lastrowid)
                for position, display_id in enumerate(passage["sentence_ids"]):
                    connection.execute(
                        "INSERT INTO passage_sentences (passage_id, sentence_id, position) VALUES (?, ?, ?)",
                        (passage_id, sentence_ids[display_id], position),
                    )
            connection.commit()
    except Exception:
        # The managed copy is content-addressed and harmless; rows are rolled back.
        raise

    corpus.mark_rebuild_required()
    return IngestResult(source, "added", document_id)


def ingest_directory(
    corpus: Corpus,
    directory: Path,
    *,
    on_duplicate: DuplicatePolicy = "ask",
    metadata_overrides: dict[str, Any] | None = None,
) -> list[IngestResult]:
    """Ingest every PDF below a directory in stable path order."""
    directory = directory.expanduser().resolve()
    return [
        ingest_pdf(corpus, path, on_duplicate=on_duplicate, metadata_overrides=metadata_overrides)
        for path in sorted(directory.rglob("*.pdf"))
    ]
