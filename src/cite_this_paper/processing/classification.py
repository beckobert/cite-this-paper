#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import pymupdf

CLASSIFIER_VERSION = 1


class TerminalHeadingMatch(TypedDict):
    section_type: str
    heading_position: int
    matched_text: str
    match_type: str


# ------------------------------------------------------------
# Terminal headings
# ------------------------------------------------------------

TERMINAL_HEADINGS = {
    "references": [
        r"references?",
        r"references?\s+and\s+notes?",
        r"bibliography",
        r"literature\s+cited",
        r"works\s+cited",
    ],
    "acknowledgements": [
        r"acknowledg(?:e)?ments?",
    ],
    "author_contributions": [
        r"author\s+contributions?",
        r"authors?['’]?\s+contributions?",
        r"credit\s+authorship\s+contribution\s+statement",
        r"author\s+statement",
    ],
    "conflicts_of_interest": [
        r"conflicts?\s+of\s+interest",
        r"conflicting\s+interests?",
        r"competing\s+interests?",
        r"declaration\s+of\s+competing\s+interest",
        r"declarations?\s+of\s+interest",
    ],
    "data_availability": [
        r"data\s+availability",
        r"data\s+availability\s+statement",
        r"data\s+and\s+code\s+availability",
        r"code\s+availability",
        r"availability\s+of\s+data",
    ],
    "funding": [
        r"funding",
        r"funding\s+information",
    ],
}


def normalize_heading_text(text: str) -> str:
    text = text.strip().casefold()

    # Remove common section numbering:
    #
    # 6. References
    # VI. REFERENCES
    # 5 References
    text = re.sub(
        r"^\s*(?:"
        r"\d+(?:\.\d+)*"
        r"|[ivxlcdm]+"
        r")"
        r"[.)]?\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def detect_terminal_heading(
    text: str,
) -> TerminalHeadingMatch | None:
    """
    Detect a terminal-section heading.

    A heading is accepted when it appears:

    1. at the beginning of the passage, optionally preceded by
       decorative symbols or section numbering;

    2. after a strong structural separator such as a black square;

    3. at the very end of a passage, following sentence-ending
       punctuation.

    Returns a dictionary describing the match, or None.
    """

    # Normalize whitespace. Character positions returned below refer
    # to this normalized string, not the original input string.
    normalized = re.sub(
        r"\s+",
        " ",
        text.strip(),
    )

    for section_type, patterns in TERMINAL_HEADINGS.items():
        for pattern in patterns:
            # ----------------------------------------------------
            # Case 1:
            # Heading at beginning of passage.
            #
            # Examples:
            #
            #   REFERENCES
            #   ■REFERENCES
            #   ■ REFERENCES
            #   6. References
            #   VI. REFERENCES
            # ----------------------------------------------------

            beginning_re = re.compile(
                rf"""
                ^
                \s*

                # Optional decorative prefix.
                [■▪◆●□▸►\-–—:]*
                \s*

                # Optional section number.
                (?:
                    (?:\d+(?:\.\d+)*|[IVXLCDM]+)
                    [.)]?
                    \s+
                )?

                (?P<heading>{pattern})

                # Heading must end cleanly. Text belonging to the
                # section itself may follow after whitespace.
                (?=
                    \s*[:.\-–—]?
                    (?:\s|$)
                )
                """,
                re.VERBOSE | re.IGNORECASE,
            )

            match = beginning_re.search(normalized)

            if match:
                return {
                    "section_type": section_type,
                    "heading_position": match.start("heading"),
                    "matched_text": match.group("heading"),
                    "match_type": "passage_start",
                }

            # ----------------------------------------------------
            # Case 2:
            # Heading introduced by an explicit publisher-style
            # structural symbol somewhere inside the passage.
            #
            # Example:
            #
            #   ... research results. ■REFERENCES
            # ----------------------------------------------------

            separator_re = re.compile(
                rf"""
                [■▪◆●□▸►]
                \s*

                (?P<heading>{pattern})

                (?=
                    \s*[:.\-–—]?
                    (?:\s|$)
                )
                """,
                re.VERBOSE | re.IGNORECASE,
            )

            match = separator_re.search(normalized)

            if match:
                return {
                    "section_type": section_type,
                    "heading_position": match.start("heading"),
                    "matched_text": match.group("heading"),
                    "match_type": "structural_symbol",
                }

            # ----------------------------------------------------
            # Case 3:
            # Heading at the very end of a passage after ordinary
            # sentence-ending punctuation.
            #
            # Examples:
            #
            #   ... contributed to these results. REFERENCES
            #   ... contributed to these results. — REFERENCES
            #
            # Requiring the heading to be at the END of the passage
            # makes this substantially safer than searching for the
            # word "references" anywhere in prose.
            # ----------------------------------------------------

            trailing_re = re.compile(
                rf"""
                [.!?]
                \s+

                [■▪◆●□▸►\-–—:]*
                \s*

                (?P<heading>{pattern})

                \s*[:.\-–—]?
                \s*$
                """,
                re.VERBOSE | re.IGNORECASE,
            )

            match = trailing_re.search(normalized)

            if match:
                return {
                    "section_type": section_type,
                    "heading_position": match.start("heading"),
                    "matched_text": match.group("heading"),
                    "match_type": "passage_end",
                }

    return None


# ------------------------------------------------------------
# Reference-like text
# ------------------------------------------------------------

REFERENCE_NUMBER_RE = re.compile(
    r"""
    ^\s*
    (?:
        \[\s*\d+\s*\]
        |
        \d{1,4}[.)]
    )
    \s+
    """,
    re.VERBOSE,
)

YEAR_RE = re.compile(
    r"\b(?:18|19|20)\d{2}[a-z]?\b",
    re.IGNORECASE,
)

DOI_RE = re.compile(
    r"""
    (?:
        \bdoi\s*:
        |
        \b10\.\d{4,9}/
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

ET_AL_RE = re.compile(
    r"\bet\s+al\.",
    re.IGNORECASE,
)


def reference_likeness(
    text: str,
) -> tuple[bool, dict]:
    """
    Decide whether a piece of text looks strongly like a bibliography.

    This is deliberately conservative because it is used only to
    confirm horizontal-rule candidates.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    numbered_entries = sum(bool(REFERENCE_NUMBER_RE.match(line)) for line in lines)

    years = len(YEAR_RE.findall(text))

    dois = len(DOI_RE.findall(text))

    et_al = len(ET_AL_RE.findall(text))

    # Three deliberately different ways to reach high confidence.
    looks_like_references = (
        numbered_entries >= 2
        or (years >= 3 and dois >= 1)
        or (years >= 5 and et_al >= 1)
        or years >= 7
    )

    diagnostics = {
        "numbered_entries": numbered_entries,
        "years": years,
        "dois": dois,
        "et_al": et_al,
    }

    return looks_like_references, diagnostics


# ------------------------------------------------------------
# Horizontal separator detection
# ------------------------------------------------------------


def horizontal_rule_candidates(
    page,
    min_width_fraction: float = 0.40,
    max_slope: float = 1.5,
) -> list[dict]:
    """
    Find long horizontal vector rules.

    Handles both explicit line drawing commands and very thin
    rectangles, since publishers may encode rules either way.
    """
    page_width = page.rect.width
    page_height = page.rect.height

    minimum_width = page_width * min_width_fraction

    candidates = []

    for path in page.get_drawings():
        for item in path["items"]:
            # -----------------------------------------------
            # Explicit line
            # -----------------------------------------------

            if item[0] == "l":
                _, p1, p2 = item

                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)

                if dx < minimum_width:
                    continue

                if dy > max_slope:
                    continue

                y = (p1.y + p2.y) / 2

                # Ignore typical page-border / header/footer rules.
                if y < page_height * 0.10 or y > page_height * 0.92:
                    continue

                candidates.append(
                    {
                        "y": float(y),
                        "x0": float(min(p1.x, p2.x)),
                        "x1": float(max(p1.x, p2.x)),
                        "width": float(dx),
                        "source": "line",
                    }
                )

            # -----------------------------------------------
            # Thin rectangle used as a horizontal bar
            # -----------------------------------------------

            elif item[0] == "re":
                _, rect, _orientation = item

                if rect.width < minimum_width:
                    continue

                if rect.height > 3.0:
                    continue

                y = (rect.y0 + rect.y1) / 2

                if y < page_height * 0.10 or y > page_height * 0.92:
                    continue

                candidates.append(
                    {
                        "y": float(y),
                        "x0": float(rect.x0),
                        "x1": float(rect.x1),
                        "width": float(rect.width),
                        "source": "thin_rectangle",
                    }
                )

    # Top-to-bottom order.
    candidates.sort(key=lambda candidate: candidate["y"])

    return candidates


def text_below_rule(
    page,
    y: float,
) -> str:
    """
    Extract text below a proposed separator.
    """
    clip = pymupdf.Rect(
        0,
        y + 2,
        page.rect.width,
        page.rect.height,
    )

    return page.get_text(
        "text",
        clip=clip,
        sort=False,
    )


def find_reference_rule_cutoff(
    pdf_path: Path,
    start_fraction: float = 0.40,
) -> dict | None:
    """
    Search the latter portion of a PDF for a long horizontal rule
    followed by reference-like text.
    """
    doc = pymupdf.open(pdf_path)

    try:
        start_page = int(len(doc) * start_fraction)

        for page_index in range(
            start_page,
            len(doc),
        ):
            page = doc[page_index]

            candidates = horizontal_rule_candidates(page)

            for candidate in candidates:
                below = text_below_rule(
                    page,
                    candidate["y"],
                )

                is_reference_like, diagnostics = reference_likeness(below)

                if not is_reference_like:
                    continue

                return {
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "y": candidate["y"],
                    "reason": ("horizontal_rule_with_reference_like_text"),
                    "diagnostics": diagnostics,
                }

    finally:
        doc.close()

    return None


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    return records


def load_documents(
    path: Path,
) -> dict[str, dict]:
    return {record["document_id"]: record for record in load_jsonl(path)}


def load_sentences(
    path: Path,
) -> dict[str, dict]:
    return {record["sentence_id"]: record for record in load_jsonl(path)}


# ------------------------------------------------------------
# Passage geometry
# ------------------------------------------------------------


def passage_top_y(
    passage: dict,
    sentences: dict[str, dict],
) -> float | None:
    """
    Return the top-most y coordinate occupied by the passage.
    """
    y_values = []

    for sentence_id in passage["sentence_ids"]:
        sentence = sentences.get(sentence_id)

        if sentence is None:
            continue

        for box in sentence.get("boxes", []):
            y_values.append(box["bbox"][1])

    if not y_values:
        return None

    return min(y_values)


# ------------------------------------------------------------
# Terminal heading cutoff
# ------------------------------------------------------------


def find_heading_cutoff(
    passages: list[dict],
    page_count: int,
    start_fraction: float = 0.40,
) -> dict | None:
    """
    Locate the earliest recognized terminal heading in the latter
    portion of the publication.
    """
    minimum_page_index = int(page_count * start_fraction)

    for passage_index, passage in enumerate(passages):
        if passage["page_index"] < minimum_page_index:
            continue

        text = passage["text_normalized"]

        detection = detect_terminal_heading(text)

        if detection is None:
            continue

        section_type = detection["section_type"]

        heading_position = detection["heading_position"]

        relative_position = heading_position / max(len(text), 1)

        # If the heading appears in the first 25% of the passage,
        # consider the passage itself end matter.
        #
        # Otherwise retain this mixed passage and begin filtering
        # with the following passage.
        if relative_position <= 0.25:
            cutoff_passage_index = passage_index
            filtering_begins = "current_passage"
        else:
            cutoff_passage_index = passage_index + 1
            filtering_begins = "next_passage"

        return {
            "passage_index": cutoff_passage_index,
            "heading_passage_index": passage_index,
            "page_index": passage["page_index"],
            "page_number": passage["page_number"],
            "section_type": section_type,
            "reason": "terminal_heading",
            "matched_text": detection["matched_text"],
            "match_type": detection["match_type"],
            "heading_position": heading_position,
            "heading_relative_position": relative_position,
            "filtering_begins": filtering_begins,
            "text": text,
        }

    return None


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------


def classify_document(
    passages: list[dict],
    sentences: dict[str, dict],
    document: dict,
    pdf_path: Path,
) -> tuple[list[dict], dict]:
    """
    Classify one document's passages.
    """
    page_count = document["page_count"]

    heading_cutoff = find_heading_cutoff(
        passages=passages,
        page_count=page_count,
    )

    rule_cutoff = None

    # Heading detection is preferable.
    # Only invoke the graphics/reference heuristic if no heading
    # was found.
    if heading_cutoff is None:
        rule_cutoff = find_reference_rule_cutoff(
            pdf_path=pdf_path,
        )

    cutoff = heading_cutoff if heading_cutoff is not None else rule_cutoff

    classified = []
    filtered_count = 0

    for passage_index, passage in enumerate(passages):
        result = dict(passage)

        # Preserve the old eligibility decision.
        base_eligible = passage.get(
            "retrieval_eligible",
            True,
        )

        result["retrieval_eligible_base"] = base_eligible

        is_end_matter = False

        if heading_cutoff is not None:
            is_end_matter = passage_index >= heading_cutoff["passage_index"]

        elif rule_cutoff is not None:
            if passage["page_index"] > rule_cutoff["page_index"]:
                is_end_matter = True

            elif passage["page_index"] == rule_cutoff["page_index"]:
                top_y = passage_top_y(
                    passage=passage,
                    sentences=sentences,
                )

                if top_y is not None and top_y > rule_cutoff["y"]:
                    is_end_matter = True

        if is_end_matter:
            result["content_type"] = "end_matter"

            result["retrieval_eligible"] = False

            filtered_count += 1

        else:
            result["content_type"] = "body"

            result["retrieval_eligible"] = base_eligible

        classified.append(result)

    report = {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "passages": len(passages),
        "filtered_passages": filtered_count,
        "cutoff": cutoff,
    }

    return classified, report


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


