"""Database-backed sentence inspection and source-page rendering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .corpus import Corpus, CorpusError


@dataclass(frozen=True)
class RenderedPage:
    """One source page rendered for a grouped sentence request."""

    output_path: Path
    filename: str
    page_number: int
    sentence_ids: tuple[str, ...]


def sentence_details(corpus: Corpus, display_id: str) -> dict:
    with corpus.connect() as connection:
        row = connection.execute(
            """
            SELECT s.id, s.display_id, s.source_text, s.normalized_text, s.page_sentence_index,
                   pg.page_index, pg.page_number, d.filename, d.stored_path, d.title
            FROM sentences AS s
            JOIN pages AS pg ON pg.id = s.page_id
            JOIN documents AS d ON d.id = s.document_id
            WHERE s.display_id = ?
            """,
            (display_id,),
        ).fetchone()
        if row is None:
            raise CorpusError(f"Sentence not found: {display_id}")
        result = dict(row)
        result["boxes"] = [
            dict(box)
            for box in connection.execute(
                "SELECT * FROM sentence_boxes WHERE sentence_id = ? ORDER BY box_index", (row["id"],)
            )
        ]
        result["context"] = [
            dict(sentence)
            for sentence in connection.execute(
                """
                SELECT display_id, source_text FROM sentences
                WHERE page_id = (SELECT page_id FROM sentences WHERE id = ?)
                ORDER BY page_sentence_index
                """,
                (row["id"],),
            )
        ]
        return result


def render_sentences(
    corpus: Corpus,
    display_ids: list[str],
    output_directory: Path,
    dpi: int = 150,
) -> list[RenderedPage]:
    """Render one highlighted image for each PDF page containing requested sentences."""
    unique_ids = list(dict.fromkeys(display_ids))
    if not unique_ids:
        raise CorpusError("At least one sentence ID is required.")

    # Resolve every ID before creating the output directory or any image, so an
    # invalid request cannot leave partial results behind.
    details = [sentence_details(corpus, display_id) for display_id in unique_ids]
    groups: dict[tuple[str, int], list[dict]] = {}
    for detail in details:
        key = (detail["stored_path"], detail["page_index"])
        groups.setdefault(key, []).append(detail)

    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    selection_digest = hashlib.sha256("\0".join(unique_ids).encode("utf-8")).hexdigest()[:12]
    rendered_pages: list[RenderedPage] = []

    for (stored_path, page_index), page_details in groups.items():
        document = pymupdf.open(stored_path)
        try:
            page = document[page_index]
            for detail in page_details:
                for box in detail["boxes"]:
                    annotation = page.add_highlight_annot(
                        pymupdf.Rect(box["x0"], box["y0"], box["x1"], box["y1"])
                    )
                    if annotation:
                        annotation.set_opacity(0.35)
                        annotation.update()
            filename = (
                f"{Path(stored_path).stem}_page_{page_details[0]['page_number']:04d}_"
                f"{selection_digest}.png"
            )
            output_path = output_directory / filename
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                alpha=False,
                annots=True,
            )
            pixmap.save(output_path)
            rendered_pages.append(
                RenderedPage(
                    output_path=output_path,
                    filename=page_details[0]["filename"],
                    page_number=page_details[0]["page_number"],
                    sentence_ids=tuple(detail["display_id"] for detail in page_details),
                )
            )
        finally:
            document.close()

    return rendered_pages
