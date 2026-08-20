"""Database-backed sentence inspection and source-page rendering."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from .corpus import Corpus, CorpusError


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


def render_sentence(corpus: Corpus, display_id: str, output_directory: Path, dpi: int = 150) -> Path:
    """Render the source PDF page with the stored sentence boxes highlighted."""
    detail = sentence_details(corpus, display_id)
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{display_id.replace(':', '_').replace('/', '_')}.png"
    document = pymupdf.open(detail["stored_path"])
    try:
        page = document[detail["page_index"]]
        for box in detail["boxes"]:
            annotation = page.add_highlight_annot(pymupdf.Rect(box["x0"], box["y0"], box["x1"], box["y1"]))
            if annotation:
                annotation.set_opacity(0.35)
                annotation.update()
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False, annots=True)
        pixmap.save(output_path)
    finally:
        document.close()
    return output_path
