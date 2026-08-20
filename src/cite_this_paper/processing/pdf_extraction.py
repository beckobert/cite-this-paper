#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def extract_word(word_tuple: tuple) -> dict:
    """
    Convert a PyMuPDF word tuple into a readable dictionary.

    PyMuPDF returns:
        x0, y0, x1, y1, text, block_no, line_no, word_no
    """
    x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple

    return {
        "text": text,
        "bbox": [
            round(x0, 3),
            round(y0, 3),
            round(x1, 3),
            round(y1, 3),
        ],
        "block_no": block_no,
        "line_no": line_no,
        "word_no": word_no,
    }


def extract_block(block_tuple: tuple) -> dict:
    """
    Convert a PyMuPDF block tuple into a dictionary.

    PyMuPDF returns:
        x0, y0, x1, y1, text, block_no, block_type

    block_type == 0 means text.
    """
    x0, y0, x1, y1, text, block_no, block_type = block_tuple

    return {
        "text": text,
        "bbox": [
            round(x0, 3),
            round(y0, 3),
            round(x1, 3),
            round(y1, 3),
        ],
        "block_no": block_no,
        "block_type": block_type,
    }


def extract_pdf(pdf_path: Path, input_dir: Path) -> tuple[dict, list[dict]]:
    """
    Extract document-level metadata and page-level text/provenance.
    """
    file_hash = sha256_file(pdf_path)

    # The short ID is primarily for convenient human-readable references.
    document_id = file_hash[:16]

    doc = pymupdf.open(str(pdf_path))

    try:
        if doc.needs_pass:
            raise RuntimeError("PDF is password protected")

        metadata = dict(doc.metadata or {})

        document_record = {
            "document_id": document_id,
            "sha256": file_hash,
            "filename": pdf_path.name,
            "relative_path": str(pdf_path.relative_to(input_dir)),
            "page_count": doc.page_count,
            "metadata": metadata,
        }

        pages = []

        for page_index in range(doc.page_count):
            page = doc[page_index]

            # Native extraction order proved substantially more reliable
            # for the two-column publications in this corpus.
            text = page.get_text("text", sort=False)

            # Keep words in their native PDF extraction order.
            words_raw = page.get_text("words", sort=False)
            words = [extract_word(word) for word in words_raw]

            # Blocks give us a useful higher-level representation of layout.
            blocks_raw = page.get_text("blocks", sort=False)

            # Keep only text blocks for now.
            blocks = [extract_block(block) for block in blocks_raw if block[6] == 0]

            stripped_text = text.strip()

            if len(stripped_text) == 0:
                extraction_status = "empty"
            elif len(stripped_text) < 100:
                extraction_status = "sparse"
            else:
                extraction_status = "ok"

            page_record = {
                "schema_version": 1,
                "document_id": document_id,
                # Internal index is zero-based.
                "page_index": page_index,
                # Human-facing page number is one-based.
                "page_number": page_index + 1,
                "width": round(page.rect.width, 3),
                "height": round(page.rect.height, 3),
                "rotation": page.rotation,
                "extraction_status": extraction_status,
                "text": text,
                "text_extraction_method": "pymypdf_native",
                "blocks": blocks,
                "words": words,
                "character_count": len(text),
                "word_count": len(words),
                "block_count": len(blocks),
            }

            pages.append(page_record)

        return document_record, pages

    finally:
        doc.close()


