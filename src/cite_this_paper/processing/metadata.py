"""Font-aware, best-effort bibliographic metadata extraction for PDFs.

The extractor deliberately keeps its intermediate layout evidence in memory.  Its
return value contains the selected catalogue values and a review-ready record of
every distinct candidate value, with its sources, score, and page references.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pymupdf

DOI_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)?(?P<doi>10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
ARXIV_RE = re.compile(r"\barXiv\s*:\s*(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)\b", re.I)
PMID_RE = re.compile(r"\bPMID\s*:?\s*(?P<id>\d+)\b", re.I)
PMC_RE = re.compile(r"\bPMC\s*:?\s*(?P<id>PMC\d+)\b", re.I)
ISSN_RE = re.compile(r"\b(?P<label>e?ISSN|pISSN)\s*:?\s*(?P<id>\d{4}-\d{3}[\dXx])\b", re.I)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
JOURNAL_HINT_RE = re.compile(r"\b(journal|letters|review|reviews|communications|proceedings|transactions|physical review|nature|science)\b", re.I)
BOILERPLATE_RE = re.compile(r"\b(copyright|all rights reserved|downloaded from|received|accepted|published|open access|creative commons)\b", re.I)
AFFILIATION_RE = re.compile(r"\b(university|universität|institute|institut|department|laboratory|centre|center|school|faculty|college|corporation|gmbh|ltd|inc)\b", re.I)
ABSTRACT_RE = re.compile(r"^\s*(abstract|summary)\s*:?\s*$", re.I)


@dataclass(frozen=True)
class PageLine:
    page_index: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    bold: bool
    dir_x: float
    dir_y: float
    page_width: float
    page_height: float

    @property
    def horizontal(self) -> bool:
        return abs(self.dir_x) >= 0.95 and abs(self.dir_y) <= 0.30

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def _space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _compare(value: str) -> str:
    return _space(re.sub(r"[^\w]+", " ", value.casefold()))


def _clean_doi(value: str) -> str:
    value = value.strip().rstrip(".,;:")
    while value.endswith(")") and value.count("(") < value.count(")"):
        value = value[:-1]
    while value.endswith("]") and value.count("[") < value.count("]"):
        value = value[:-1]
    return value.lower()


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = _space(value)
        if value and _compare(value) not in seen:
            result.append(value)
            seen.add(_compare(value))
    return result


def _xml_tag(tag: str) -> tuple[str, str]:
    if tag.startswith("{") and "}" in tag:
        namespace, name = tag[1:].split("}", 1)
        return namespace, name
    return "", tag


def _xml_values(element: Any) -> list[str]:
    listed = [_space(" ".join(node.itertext())) for node in element.iter() if _xml_tag(node.tag)[1].casefold() == "li"]
    return _unique(listed) if listed else _unique([" ".join(element.itertext())])


def extract_xmp_metadata(pdf: pymupdf.Document) -> dict[str, Any]:
    """Read the Dublin Core and PRISM fields commonly embedded by publishers."""
    result: dict[str, Any] = {
        "present": False, "parse_error": False, "title": [], "authors": [], "journal": [], "doi": [],
        "issn": [], "eissn": [], "volume": [], "issue": [], "starting_page": [], "ending_page": [],
        "publication_date": [],
    }
    xml = pdf.get_xml_metadata()
    if not xml.strip():
        return result
    result["present"] = True
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        result["parse_error"] = True
        return result
    for element in root.iter():
        namespace, local_name = _xml_tag(element.tag)
        namespace, name = namespace.casefold(), local_name.casefold()
        values = _xml_values(element)
        if not values:
            continue
        if "purl.org/dc/" in namespace:
            if name == "title":
                result["title"].extend(values)
            elif name == "creator":
                result["authors"].extend(values)
            elif name == "identifier":
                for value in values:
                    result["doi"].extend(_clean_doi(match.group("doi")) for match in DOI_RE.finditer(value))
        if "prismstandard.org" in namespace:
            if name in {"publicationname", "journal", "journalname"}:
                result["journal"].extend(values)
            elif name == "doi":
                result["doi"].extend(_clean_doi(match.group("doi")) for value in values for match in DOI_RE.finditer(value))
            elif name == "issn":
                result["issn"].extend(values)
            elif name in {"eissn", "electronicissn"}:
                result["eissn"].extend(values)
            elif name == "volume":
                result["volume"].extend(values)
            elif name in {"number", "issue"}:
                result["issue"].extend(values)
            elif name == "startingpage":
                result["starting_page"].extend(values)
            elif name == "endingpage":
                result["ending_page"].extend(values)
            elif name in {"publicationdate", "coverdate"}:
                result["publication_date"].extend(values)
    for key, value in result.items():
        if isinstance(value, list):
            result[key] = _unique(value)
    return result


def extract_page_lines(page: pymupdf.Page, page_index: int) -> list[PageLine]:
    """Extract font and geometry-aware text lines from one page."""
    lines: list[PageLine] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            bbox = line.get("bbox")
            if not spans or not bbox:
                continue
            text = _space("".join(str(span.get("text", "")) for span in spans))
            if not text:
                continue
            direction = line.get("dir", (1.0, 0.0))
            lines.append(PageLine(
                page_index, text, *map(float, bbox),
                max(float(span.get("size", 0.0)) for span in spans),
                any("bold" in str(span.get("font", "")).casefold() for span in spans),
                float(direction[0]), float(direction[1]), float(page.rect.width), float(page.rect.height),
            ))
    return sorted(lines, key=lambda line: (line.page_index, line.y0, line.x0))


class CandidateCollector:
    """Deduplicate values while retaining all evidence useful for later review."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def add(self, value: str | None, source: str, score: float, page_index: int | None = None) -> None:
        if not value or not (value := _space(str(value))):
            return
        key = _compare(value)
        record = self._records.setdefault(key, {"value": value, "score": 0.0, "sources": set(), "page_numbers": set()})
        record["score"] += score
        record["sources"].add(source)
        if page_index is not None:
            record["page_numbers"].add(page_index + 1)

    def records(self) -> list[dict[str, Any]]:
        return [
            {"value": record["value"], "score": round(record["score"], 3), "sources": sorted(record["sources"]), "page_numbers": sorted(record["page_numbers"])}
            for record in sorted(self._records.values(), key=lambda item: (-item["score"], _compare(item["value"])))
        ]

    def selected(self) -> str | None:
        records = self.records()
        return records[0]["value"] if records else None


def _metadata_text(document: dict[str, Any]) -> str:
    return "\n".join(str(value) for value in (document.get("metadata") or {}).values() if value)


def _layout_title_candidates(lines: list[PageLine]) -> list[tuple[PageLine, float]]:
    eligible = [line for line in lines if line.horizontal and line.y0 < line.page_height * 0.60 and len(line.text) >= 12
                and not DOI_RE.search(line.text) and not ARXIV_RE.search(line.text) and "@" not in line.text
                and not ABSTRACT_RE.match(line.text) and not BOILERPLATE_RE.search(line.text)]
    if not eligible:
        return []
    largest = max(line.font_size for line in eligible)
    result: list[tuple[PageLine, float]] = []
    for line in eligible:
        if line.font_size < largest * 0.82:
            continue
        score = 5.0 * line.font_size / largest + 2.0 * (1 - line.y0 / line.page_height)
        score += min(line.width / line.page_width, 1.0) + (0.5 if line.bold else 0.0)
        score += 1.0 if len(line.text.split()) >= 5 else 0.0
        result.append((line, score))
    return sorted(result, key=lambda item: item[1], reverse=True)[:5]


def _authors_below_title(lines: list[PageLine], title: PageLine) -> str | None:
    candidates: list[str] = []
    for line in lines:
        if line.page_index != title.page_index or line.y0 <= title.y1 or line.y0 > title.y1 + title.page_height * 0.22:
            continue
        if ABSTRACT_RE.match(line.text) or AFFILIATION_RE.search(line.text) or BOILERPLATE_RE.search(line.text):
            break
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", line.text)
        if line.horizontal and len(words) >= 2 and sum(word[0].isupper() for word in words) / len(words) >= 0.45:
            candidates.append(line.text)
    return _space(" ".join(candidates)) or None


def _margin_blocks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in pages:
        height = float(page["height"])
        for block in page.get("blocks", []):
            x0, y0, x1, y1 = block["bbox"]
            region = "header" if y1 <= height * 0.18 else "footer" if y0 >= height * 0.82 else None
            if region:
                result.append({"text": block["text"], "page_index": page["page_index"], "region": region})
    return result


def _recurring_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        signature = DOI_RE.sub("<doi>", block["text"].casefold())
        signature = re.sub(r"https?://\S+", "<url>", signature)
        signature = re.sub(r"\b\d{1,6}\b", "<n>", signature)
        signature = _space(signature)
        if len(signature) >= 4:
            grouped[signature].append(block)
    return [block for matches in grouped.values() if len({match["page_index"] for match in matches}) >= 2 for block in matches]


def _journal_prefix(text: str) -> str | None:
    text = _space(DOI_RE.sub("", text))
    if not text or AFFILIATION_RE.search(text):
        return None
    match = re.match(r"(.+?)\s+(?:19\d{2}|20\d{2}|\d{1,4})(?=\s*[,;:|()]|\s+\d|$)", text)
    value = _space((match.group(1) if match else text).strip(" |,;:-–—"))
    if not 3 <= len(value) <= 120 or not re.search(r"[A-Za-z].*[A-Za-z].*[A-Za-z]", value):
        return None
    return value if JOURNAL_HINT_RE.search(value) or match else None


def _identifier_candidates(texts: list[tuple[str, str, int | None]], xmp: dict[str, Any]) -> dict[str, CandidateCollector]:
    fields = {name: CandidateCollector() for name in ("issn", "eissn", "arxiv", "pmid", "pmc")}
    for text, source, page_index in texts:
        for match in ISSN_RE.finditer(text):
            fields["eissn" if match.group("label").casefold().startswith("e") else "issn"].add(match.group("id").upper(), source, 2.0, page_index)
        for name, regex in (("arxiv", ARXIV_RE), ("pmid", PMID_RE), ("pmc", PMC_RE)):
            for match in regex.finditer(text):
                fields[name].add(match.group("id").upper() if name == "pmc" else match.group("id"), source, 2.0, page_index)
    for name in ("issn", "eissn"):
        for value in xmp[name]:
            fields[name].add(value.upper(), "xmp", 8.0)
    return fields


def _xmp_candidates(xmp: dict[str, Any], key: str) -> CandidateCollector:
    result = CandidateCollector()
    for value in xmp[key]:
        result.add(value, "xmp", 8.0)
    return result


def extract_document_metadata(document: dict[str, Any], pages: list[dict[str, Any]], pdf_path: Path) -> dict[str, Any]:
    """Return selected metadata and reviewable alternatives for a document.

    This function is intentionally independent of database code so both package
    ingestion and future metadata-review tooling share the same reader.
    """
    with pymupdf.open(pdf_path) as pdf:
        xmp = extract_xmp_metadata(pdf)
        front_lines = [line for index in range(min(2, len(pdf))) for line in extract_page_lines(pdf[index], index)]

    raw = document.get("metadata") or {}
    margins = _margin_blocks(pages)
    recurring_margins = _recurring_blocks(margins)
    title = CandidateCollector()
    author = CandidateCollector()
    journal = CandidateCollector()
    doi = CandidateCollector()
    if raw.get("title") and len(_space(raw["title"])) >= 8:
        title.add(raw["title"], "pdf_metadata", 6.0)
    if raw.get("author"):
        author.add(raw["author"], "pdf_metadata", 5.0)
    for value in xmp["title"]:
        title.add(value, "xmp", 7.0)
    if xmp["authors"]:
        author.add(", ".join(xmp["authors"]), "xmp", 8.0)
    layout_titles = _layout_title_candidates(front_lines)
    for line, score in layout_titles:
        title.add(line.text, "page_layout", score, line.page_index)
    for line, _ in layout_titles[:2]:
        visible_authors = _authors_below_title(front_lines, line)
        if visible_authors:
            author.add(visible_authors, "page_layout", 5.0, line.page_index)

    metadata_text = _metadata_text(document)
    for match in DOI_RE.finditer(metadata_text):
        doi.add(_clean_doi(match.group("doi")), "pdf_metadata", 6.0)
    for value in xmp["doi"]:
        doi.add(value, "xmp", 9.0)
    for line in front_lines:
        for match in DOI_RE.finditer(line.text):
            explicit = "doi:" in line.text.casefold() or "doi.org" in line.text.casefold()
            doi.add(_clean_doi(match.group("doi")), "first_page" if line.page_index == 0 else "second_page", 7.0 if explicit else 3.0, line.page_index)
    doi_pages: dict[str, set[int]] = defaultdict(set)
    for block in recurring_margins:
        for match in DOI_RE.finditer(block["text"]):
            value = _clean_doi(match.group("doi"))
            doi.add(value, f"recurring_{block['region']}", 3.0, block["page_index"])
            doi_pages[value].add(block["page_index"])
    for value, page_indices in doi_pages.items():
        if len(page_indices) >= 2:
            doi.add(value, "margin_recurrence", 8.0)

    for value in xmp["journal"]:
        journal.add(value, "xmp", 9.0)
    if raw.get("subject") and JOURNAL_HINT_RE.search(raw["subject"]):
        journal.add(raw["subject"], "pdf_metadata.subject", 4.0)
    for block in recurring_margins:
        if value := _journal_prefix(block["text"]):
            journal.add(value, "recurring_margin", 4.0, block["page_index"])
    for line in front_lines:
        if line.y0 <= line.page_height * 0.20 and (value := _journal_prefix(line.text)):
            journal.add(value, "first_page" if line.page_index == 0 else "second_page", 3.0, line.page_index)

    identifier_texts = [(metadata_text, "pdf_metadata", None)] + [(line.text, "first_page" if line.page_index == 0 else "second_page", line.page_index) for line in front_lines] + [(block["text"], f"recurring_{block['region']}", block["page_index"]) for block in recurring_margins]
    identifiers = _identifier_candidates(identifier_texts, xmp)
    bibliography = {key: _xmp_candidates(xmp, key) for key in ("volume", "issue", "starting_page", "ending_page", "publication_date")}
    year = CandidateCollector()
    for candidate in bibliography["publication_date"].records():
        match = YEAR_RE.search(candidate["value"])
        if match:
            year.add(match.group(1), "xmp", candidate["score"])

    candidates = {
        "title": title.records(), "authors": author.records(), "journal": journal.records(), "doi": doi.records(),
        "identifiers": {name: collector.records() for name, collector in identifiers.items()},
        "bibliographic": {**{name: collector.records() for name, collector in bibliography.items()}, "publication_year": year.records()},
    }
    selected = {
        "title": title.selected(), "authors": author.selected(), "journal": journal.selected(), "doi": doi.selected(),
        "issn": [item["value"] for item in identifiers["issn"].records()], "eissn": [item["value"] for item in identifiers["eissn"].records()],
        "arxiv": [item["value"] for item in identifiers["arxiv"].records()], "pmid": [item["value"] for item in identifiers["pmid"].records()], "pmc": [item["value"] for item in identifiers["pmc"].records()],
        "volume": bibliography["volume"].selected(), "issue": bibliography["issue"].selected(),
        "starting_page": bibliography["starting_page"].selected(), "ending_page": bibliography["ending_page"].selected(),
        "publication_date": bibliography["publication_date"].selected(),
        "publication_year": int(year.selected()) if year.selected() else None,
    }
    if selected["starting_page"] and selected["ending_page"]:
        selected["page_range"] = f"{selected['starting_page']}–{selected['ending_page']}"
    else:
        selected["page_range"] = selected["starting_page"] or selected["ending_page"]
    return {"selected": selected, "candidates": candidates}
