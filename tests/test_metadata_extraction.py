from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pymupdf

from cite_this_paper.ingest import ingest_pdf
from cite_this_paper.processing import metadata as metadata_processing
from cite_this_paper.processing.metadata import extract_document_metadata, extract_identifiers
from cite_this_paper.processing.pdf_extraction import extract_pdf

from test_support import CorpusTestCase


def create_cover_page_pdf(path: Path) -> None:
    document = pymupdf.open()
    for index in range(3):
        page = document.new_page()
        if index:
            page.insert_text((72, 30), "Journal of Testing 2025, 12", fontsize=9)
        page.insert_text((72, 810), "doi:10.1234/example.article", fontsize=9)
        if index == 0:
            page.insert_text((72, 100), "Cover", fontsize=10)
        elif index == 1:
            page.insert_text((72, 120), "A Font-Aware Article Title", fontsize=24, fontname="hebo")
            page.insert_text((72, 160), "Ada Lovelace and Grace Hopper", fontsize=12)
            page.insert_text((72, 200), "doi:10.1234/example.article", fontsize=10)
            page.insert_text((72, 250), "This article contains searchable scientific evidence for testing.", fontsize=10)
        else:
            page.insert_text((72, 100), "Additional article text for recurring-header detection.", fontsize=10)
    document.save(path)
    document.close()


class MetadataExtractionTests(CorpusTestCase):
    def setUp(self) -> None:
        super().setUp()
        create_cover_page_pdf(self.pdf)

    def test_second_page_layout_and_recurring_margins_produce_reviewable_candidates(self):
        document, pages = extract_pdf(self.pdf)
        metadata = extract_document_metadata(document, pages, self.pdf)

        self.assertEqual(metadata["title"], "A Font-Aware Article Title")
        self.assertEqual(metadata["doi"], "10.1234/example.article")
        self.assertIn("Ada Lovelace", metadata["authors_raw"])
        self.assertIn("Journal of Testing", metadata["journal"])
        self.assertIn("second_page", metadata["fields"]["doi"]["sources"])
        self.assertIn("recurring_footer", metadata["fields"]["doi"]["sources"])
        self.assertIn("recurring_margin", metadata["fields"]["journal"]["sources"])
        self.assertEqual(set(metadata["cleanup"]["fields"]), {"title", "authors", "journal", "doi"})

    def test_journal_cleanup_preserves_letter_in_a_journal_name(self):
        self.assertEqual(
            metadata_processing._clean_journal("Journal of Experimental Letter"),
            ("Journal of Experimental Letter", None),
        )
        self.assertEqual(
            metadata_processing._clean_journal("Journal of Testing Article"),
            ("Journal of Testing", None),
        )

    def test_ingestion_stores_candidates_but_not_page_layout_blocks_and_manual_values_win(self):
        result = ingest_pdf(self.corpus, self.pdf, on_duplicate="discard", metadata_overrides={"title": "Manual title"})
        self.assertEqual(result.status, "added")
        with self.corpus.connect() as connection:
            document = connection.execute("SELECT title, metadata_candidates_json FROM documents").fetchone()
            page_columns = {row[1] for row in connection.execute("PRAGMA table_info(pages)")}
        candidates = json.loads(document["metadata_candidates_json"])
        self.assertEqual(document["title"], "Manual title")
        self.assertIn("A Font-Aware Article Title", [item["value"] for item in candidates["fields"]["title"]["candidates"]])
        self.assertNotIn("blocks", page_columns)

    def test_metadata_reader_failure_does_not_block_ingestion(self):
        with patch("cite_this_paper.ingest.extract_document_metadata", side_effect=RuntimeError("metadata failure")):
            result = ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        self.assertEqual(result.status, "added")
        with self.corpus.connect() as connection:
            document = connection.execute("SELECT title, metadata_candidates_json FROM documents").fetchone()
        self.assertIsNone(document["title"])
        self.assertEqual(json.loads(document["metadata_candidates_json"]), {})

    def test_visible_eissn_is_not_stored_as_print_issn(self):
        identifiers = extract_identifiers(
            texts=["ISSN: 1234-567X\neISSN: 9876-543X"],
            xmp={"issn": [], "eissn": [], "identifiers": []},
            selected_doi=None,
        )
        self.assertEqual(identifiers["issn"], ["1234-567X"])
        self.assertEqual(identifiers["eissn"], ["9876-543X"])
