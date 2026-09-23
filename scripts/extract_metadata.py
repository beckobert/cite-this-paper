#!/usr/bin/env python3

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import fitz

SCHEMA_VERSION = 3


# ======================================================================
# Regular expressions
# ======================================================================


DOI_RE = re.compile(
    r"""
    (?:
        https?://(?:dx\.)?doi\.org/
        |
        doi\s*:\s*
    )?
    (?P<doi>
        10\.\d{4,9}/
        [-._;()/:A-Z0-9]+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


ARXIV_RE = re.compile(
    r"""
    \b
    arXiv
    \s*:\s*
    (?P<id>
        \d{4}\.\d{4,5}
        (?:v\d+)?
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


PMID_RE = re.compile(
    r"""
    \b
    PMID
    \s*:?\s*
    (?P<id>\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


PMC_RE = re.compile(
    r"""
    \b
    PMC
    \s*:?\s*
    (?P<id>PMC\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


ISSN_RE = re.compile(
    r"""
    \b
    (?:
        ISSN
        |
        eISSN
        |
        pISSN
    )
    \s*:?\s*
    (?P<id>
        \d{4}-\d{3}[\dXx]
    )
    \b
    """,
    re.VERBOSE,
)


YEAR_RE = re.compile(
    r"""
    \b
    (?P<year>
        19\d{2}
        |
        20\d{2}
    )
    \b
    """,
    re.VERBOSE,
)


RELATED_HEADING_RE = re.compile(
    r"""
    ^\s*
    (?:
        related\s+(?:articles?|content|research)
        |
        recommended\s+(?:articles?|content)
        |
        you\s+may\s+also\s+like
        |
        similar\s+articles?
        |
        more\s+like\s+this
        |
        other\s+articles?
        |
        cited\s+by
    )
    \s*:?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


ABSTRACT_RE = re.compile(
    r"""
    ^\s*
    (?:abstract|summary)
    \s*:?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


AFFILIATION_HINT_RE = re.compile(
    r"""
    \b
    (?:
        university
        |
        universität
        |
        institute
        |
        institut
        |
        department
        |
        laboratory
        |
        centre
        |
        center
        |
        school
        |
        faculty
        |
        college
        |
        corporation
        |
        gmbh
        |
        ltd
        |
        inc
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


BOILERPLATE_RE = re.compile(
    r"""
    \b
    (?:
        copyright
        |
        all\s+rights\s+reserved
        |
        downloaded\s+from
        |
        downloaded\s+by
        |
        received
        |
        accepted
        |
        published
        |
        open\s+access
        |
        creative\s+commons
        |
        terms\s+of\s+use
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


JOURNAL_HINT_RE = re.compile(
    r"""
    \b
    (?:
        journal
        |
        letters
        |
        review
        |
        reviews
        |
        communications
        |
        proceedings
        |
        transactions
        |
        physical\s+review
        |
        nature
        |
        science
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


JOURNAL_BOILERPLATE_RE = re.compile(
    r"""
    (?:
        ©
        |
        copyright
        |
        owner\s+societ
        |
        this\s+journal\s+is
        |
        all\s+rights\s+reserved
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ======================================================================
# Page layout representation
# ======================================================================


@dataclass
class PageLine:
    page_index: int

    text: str

    x0: float
    y0: float
    x1: float
    y1: float

    font_size: float
    bold: bool

    # PyMuPDF line direction vector.
    #
    # Horizontal left-to-right text is normally approximately:
    #
    #     (1, 0)
    #
    # Vertical text is closer to:
    #
    #     (0, 1)
    #
    dir_x: float
    dir_y: float

    page_width: float
    page_height: float

    block_no: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def is_horizontal(self) -> bool:
        """
        Accept text whose baseline is approximately horizontal.

        abs(dir_x) is used so right-to-left horizontal text would
        also count as horizontal.
        """

        return abs(self.dir_x) >= 0.95 and abs(self.dir_y) <= 0.30


# ======================================================================
# Generic helpers
# ======================================================================


def normalize_space(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_compare(
    text: str,
) -> str:
    text = text.casefold()

    text = re.sub(
        r"[^\w]+",
        " ",
        text,
    )

    return normalize_space(text)


def text_similarity(
    a: str,
    b: str,
) -> float:
    a = normalize_compare(a)
    b = normalize_compare(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def clean_doi(
    doi: str,
) -> str:
    """
    Normalize a DOI extracted from text.

    DOI suffixes may legitimately contain parentheses, so closing
    brackets are stripped only when they are clearly unbalanced.
    """

    doi = doi.strip()

    doi = doi.rstrip(".,;:")

    while doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1]

    while doi.endswith("]") and doi.count("[") < doi.count("]"):
        doi = doi[:-1]

    return doi.lower()


def deduplicate_strings(
    values: list[str],
) -> list[str]:
    result = []

    seen = set()

    for value in values:
        value = normalize_space(value)

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)

        result.append(value)

    return result


def horizontal_overlap(
    a: PageLine,
    b: PageLine,
) -> float:
    overlap = max(
        0.0,
        min(
            a.x1,
            b.x1,
        )
        - max(
            a.x0,
            b.x0,
        ),
    )

    denominator = min(
        max(
            a.width,
            1.0,
        ),
        max(
            b.width,
            1.0,
        ),
    )

    return overlap / denominator


def plausible_pdf_title(
    text: str | None,
    filename: str,
) -> bool:
    if not text:
        return False

    text = normalize_space(text)

    if not 8 <= len(text) <= 500:
        return False

    lower = text.casefold()

    if lower in {
        "untitled",
        "document",
        "article",
        "manuscript",
        "paper",
    }:
        return False

    if lower.startswith("microsoft word"):
        return False

    stem = normalize_compare(Path(filename).stem)

    if normalize_compare(text) == stem:
        return False

    return True


def plausible_pdf_author(
    text: str | None,
) -> bool:
    if not text:
        return False

    text = normalize_space(text)

    if not 3 <= len(text) <= 500:
        return False

    if text.casefold() in {
        "author",
        "unknown",
        "user",
        "administrator",
    }:
        return False

    return True


def first_or_none(
    values: list[str],
) -> str | None:
    if values:
        return values[0]

    return None


# ======================================================================
# Existing extracted data
# ======================================================================


def load_documents(
    path: Path,
) -> list[dict]:
    documents = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            documents.append(json.loads(line))

    return documents


def load_pages_by_document(
    path: Path,
) -> dict[str, list[dict]]:
    result = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            page = json.loads(line)

            result.setdefault(
                page["document_id"],
                [],
            ).append(page)

    for pages in result.values():
        pages.sort(key=lambda page: page["page_index"])

    return result


def resolve_pdf_path(
    document: dict,
    pdf_root: Path,
) -> Path:
    relative_path = document.get("relative_path")

    if relative_path:
        path = pdf_root / relative_path

        if path.exists():
            return path

    filename = document.get("filename")

    if filename:
        path = pdf_root / filename

        if path.exists():
            return path

    raise FileNotFoundError(f"Could not resolve PDF for {document.get('document_id')}")


# ======================================================================
# XMP extraction
# ======================================================================


def split_xml_tag(
    tag: str,
) -> tuple[str, str]:
    if tag.startswith("{") and "}" in tag:
        uri, local_name = tag[1:].split(
            "}",
            1,
        )

        return (
            uri,
            local_name,
        )

    return (
        "",
        tag,
    )


def xml_element_values(
    element,
) -> list[str]:
    """
    Handle both ordinary XML text and RDF list structures such as:

        <dc:creator>
            <rdf:Seq>
                <rdf:li>Author A</rdf:li>
                <rdf:li>Author B</rdf:li>
            </rdf:Seq>
        </dc:creator>
    """

    list_values = []

    for descendant in element.iter():
        _, local_name = split_xml_tag(descendant.tag)

        if local_name.casefold() == "li":
            value = normalize_space(" ".join(descendant.itertext()))

            if value:
                list_values.append(value)

    if list_values:
        return deduplicate_strings(list_values)

    value = normalize_space(" ".join(element.itertext()))

    if value:
        return [value]

    return []


def extract_xmp_metadata(
    pdf,
) -> dict:
    result = {
        "present": False,
        "title": [],
        "authors": [],
        "journal": [],
        "doi": [],
        "issn": [],
        "eissn": [],
        "identifiers": [],
        "volume": [],
        "issue": [],
        "starting_page": [],
        "ending_page": [],
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
        uri, local_name = split_xml_tag(element.tag)

        name = local_name.casefold()

        uri_lower = uri.casefold()

        values = xml_element_values(element)

        if not values:
            continue

        is_dc = "purl.org/dc/" in uri_lower

        is_prism = "prismstandard.org" in uri_lower

        # ----------------------------------------------------------
        # Dublin Core
        # ----------------------------------------------------------

        if is_dc and name == "title":
            result["title"].extend(values)

        elif is_dc and name == "creator":
            result["authors"].extend(values)

        elif is_dc and name == "identifier":
            result["identifiers"].extend(values)

            for value in values:
                for match in DOI_RE.finditer(value):
                    result["doi"].append(clean_doi(match.group("doi")))

        # ----------------------------------------------------------
        # PRISM / publisher bibliographic metadata
        # ----------------------------------------------------------

        if is_prism and name in {
            "publicationname",
            "journal",
            "journalname",
        }:
            result["journal"].extend(values)

        elif is_prism and name == "doi":
            for value in values:
                match = DOI_RE.search(value)

                if match:
                    result["doi"].append(clean_doi(match.group("doi")))

        elif is_prism and name == "issn":
            result["issn"].extend(values)

        elif is_prism and name in {
            "eissn",
            "electronicissn",
        }:
            result["eissn"].extend(values)

        elif is_prism and name == "volume":
            result["volume"].extend(values)

        elif is_prism and name in {
            "number",
            "issue",
        }:
            result["issue"].extend(values)

        elif is_prism and name == "startingpage":
            result["starting_page"].extend(values)

        elif is_prism and name == "endingpage":
            result["ending_page"].extend(values)

        elif is_prism and name in {
            "publicationdate",
            "coverdate",
        }:
            result["publication_date"].extend(values)

    for key in result:
        if isinstance(
            result[key],
            list,
        ):
            result[key] = deduplicate_strings(result[key])

    return result


# ======================================================================
# Page typography
# ======================================================================


def extract_page_lines(
    page,
    page_index: int,
) -> list[PageLine]:
    data = page.get_text("dict")

    page_width = float(page.rect.width)

    page_height = float(page.rect.height)

    lines = []

    for block_no, block in enumerate(
        data.get(
            "blocks",
            [],
        )
    ):
        if block.get("type") != 0:
            continue

        for line in block.get(
            "lines",
            [],
        ):
            spans = line.get(
                "spans",
                [],
            )

            if not spans:
                continue

            text = normalize_space(
                "".join(
                    span.get(
                        "text",
                        "",
                    )
                    for span in spans
                )
            )

            if not text:
                continue

            bbox = line.get("bbox")

            if not bbox:
                continue

            direction = line.get(
                "dir",
                (1.0, 0.0),
            )

            font_size = max(
                float(
                    span.get(
                        "size",
                        0.0,
                    )
                )
                for span in spans
            )

            bold = any(
                "bold"
                in span.get(
                    "font",
                    "",
                ).casefold()
                for span in spans
            )

            lines.append(
                PageLine(
                    page_index=(page_index),
                    text=text,
                    x0=float(bbox[0]),
                    y0=float(bbox[1]),
                    x1=float(bbox[2]),
                    y1=float(bbox[3]),
                    font_size=(font_size),
                    bold=bold,
                    dir_x=float(direction[0]),
                    dir_y=float(direction[1]),
                    page_width=(page_width),
                    page_height=(page_height),
                    block_no=(block_no),
                )
            )

    lines.sort(
        key=lambda line: (
            line.y0,
            line.x0,
        )
    )

    return lines


# ======================================================================
# Related-content detection
# ======================================================================


def find_related_headings(
    lines: list[PageLine],
) -> list[PageLine]:
    return [line for line in lines if RELATED_HEADING_RE.match(line.text)]


def line_is_in_related_region(
    line: PageLine,
    related_headings: list[PageLine],
) -> bool:
    for heading in related_headings:
        if line.page_index != heading.page_index:
            continue

        if line.y0 <= heading.y0:
            continue

        if line.y0 - heading.y0 > 0.65 * line.page_height:
            continue

        center_distance = abs(line.center_x - heading.center_x)

        same_region = (
            horizontal_overlap(
                line,
                heading,
            )
            >= 0.15
            or center_distance <= 0.20 * line.page_width
        )

        if same_region:
            return True

    return False


# ======================================================================
# DOI hints available before title selection
# ======================================================================


def collect_known_dois(
    document: dict,
    xmp: dict,
) -> list[str]:
    values = []

    values.extend(xmp["doi"])

    metadata_text = "\n".join(
        str(value)
        for value in document.get(
            "metadata",
            {},
        ).values()
        if value
    )

    for match in DOI_RE.finditer(metadata_text):
        values.append(clean_doi(match.group("doi")))

    return list(dict.fromkeys(values))


# ======================================================================
# Title extraction
# ======================================================================


def looks_like_internal_title(
    text: str,
    known_dois: list[str] | None = None,
) -> bool:
    """
    Detect likely publisher/typesetting identifiers accidentally stored
    as a PDF title.

    Examples include:

        RSC_CP_C3CP54374A 3..23

    and arXiv stamps.
    """

    text = normalize_space(text)

    lower = text.casefold()

    if ARXIV_RE.search(text):
        return True

    if lower.startswith("arxiv:"):
        return True

    # Typesetting / production identifiers often contain multiple
    # underscores and very few natural-language words.
    if text.count("_") >= 2 and len(text.split()) <= 6:
        return True

    # Page-range-like production metadata.
    if ".." in text and len(text.split()) <= 6:
        return True

    letters = sum(character.isalpha() for character in text)

    uppercase = sum(character.isupper() for character in text)

    # Highly uppercase short identifiers are suspicious.
    if letters >= 6 and uppercase / letters > 0.80 and len(text.split()) <= 5:
        return True

    # Some publisher production names contain the DOI suffix.
    if known_dois:
        compact_title = re.sub(
            r"[^a-z0-9]",
            "",
            lower,
        )

        for doi in known_dois:
            suffix = doi.split(
                "/",
                1,
            )[-1].casefold()

            compact_suffix = re.sub(
                r"[^a-z0-9]",
                "",
                suffix,
            )

            if (
                len(compact_suffix) >= 6
                and compact_suffix in compact_title
                and len(text.split()) <= 6
            ):
                return True

    return False


def title_line_is_eligible(
    line: PageLine,
    related_headings: list[PageLine],
) -> bool:
    text = line.text

    # This explicitly rejects vertical arXiv sidebars and similar
    # large-font publisher decorations.
    if not line.is_horizontal:
        return False

    if line.y0 > 0.60 * line.page_height:
        return False

    if len(text) < 8:
        return False

    if DOI_RE.search(text):
        return False

    if ARXIV_RE.search(text):
        return False

    if text.casefold().startswith("arxiv:"):
        return False

    if "@" in text:
        return False

    if BOILERPLATE_RE.search(text):
        return False

    if RELATED_HEADING_RE.match(text):
        return False

    if ABSTRACT_RE.match(text):
        return False

    if line_is_in_related_region(
        line,
        related_headings,
    ):
        return False

    return True


def build_visible_title_candidates(
    lines: list[PageLine],
) -> list[dict]:
    if not lines:
        return []

    related_headings = find_related_headings(lines)

    eligible = [
        line
        for line in lines
        if title_line_is_eligible(
            line,
            related_headings,
        )
    ]

    if not eligible:
        return []

    max_size = max(line.font_size for line in eligible)

    # Candidate title lines must be close to the largest horizontal
    # typography present in the top part of the article.
    large_lines = [line for line in eligible if line.font_size >= 0.82 * max_size]

    groups = []

    current = []

    for line in large_lines:
        if not current:
            current = [line]

            continue

        previous = current[-1]

        vertical_gap = line.y0 - previous.y1

        size_ratio = min(
            line.font_size,
            previous.font_size,
        ) / max(
            line.font_size,
            previous.font_size,
            0.1,
        )

        center_distance = abs(line.center_x - previous.center_x)

        compatible = (
            vertical_gap
            <= 2.0
            * max(
                line.font_size,
                previous.font_size,
            )
            and size_ratio >= 0.82
            and center_distance <= 0.35 * line.page_width
        )

        if compatible:
            current.append(line)

        else:
            groups.append(current)

            current = [line]

    if current:
        groups.append(current)

    candidates = []

    for group in groups:
        text = normalize_space(" ".join(line.text for line in group))

        if len(text) < 15:
            continue

        first = group[0]

        last = group[-1]

        average_size = sum(line.font_size for line in group) / len(group)

        width = max(line.x1 for line in group) - min(line.x0 for line in group)

        width_fraction = width / first.page_width

        y_fraction = first.y0 / first.page_height

        score = 5.0 * average_size / max_size

        # Titles near the top of the page are favored.
        score += max(
            0.0,
            2.0 * (1.0 - y_fraction),
        )

        # Wide horizontal text is more title-like than narrow
        # marginal labels.
        score += min(
            width_fraction,
            1.0,
        )

        if any(line.bold for line in group):
            score += 0.5

        # Natural article titles typically contain several words.
        if len(text.split()) >= 5:
            score += 1.0

        candidates.append(
            {
                "value": text,
                "score": score,
                "source": "page_layout",
                "bbox": [
                    min(line.x0 for line in group),
                    first.y0,
                    max(line.x1 for line in group),
                    last.y1,
                ],
                "font_size": average_size,
                "width_fraction": width_fraction,
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates


def cluster_text_candidates(
    candidates: list[dict],
    similarity_threshold: float,
) -> list[dict]:
    """
    Group similar strings while retaining every evidence source.

    The initial score is simply the sum of member scores. Individual
    consumers may rescore the clusters afterward.
    """

    clusters = []

    for candidate in candidates:
        placed = False

        for cluster in clusters:
            if (
                text_similarity(
                    candidate["value"],
                    cluster["value"],
                )
                >= similarity_threshold
            ):
                cluster["score"] += candidate["score"]

                cluster["sources"].append(candidate["source"])

                cluster["members"].append(candidate)

                if candidate["score"] > cluster["best_member_score"]:
                    cluster["value"] = candidate["value"]

                    cluster["best_member_score"] = candidate["score"]

                placed = True

                break

        if not placed:
            clusters.append(
                {
                    "value": candidate["value"],
                    "score": candidate["score"],
                    "best_member_score": candidate["score"],
                    "sources": [candidate["source"]],
                    "members": [candidate],
                }
            )

    for cluster in clusters:
        cluster["sources"] = sorted(set(cluster["sources"]))

    clusters.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return clusters


def title_source_family(
    source: str,
) -> str:
    """
    PDF Info and XMP frequently duplicate the same publisher metadata.

    They therefore count as one evidence family rather than two
    independent confirmations.
    """

    if source in {
        "pdf_metadata",
        "xmp",
    }:
        return "embedded"

    if source == "page_layout":
        return "visual"

    return source


def rescore_title_clusters(
    clusters: list[dict],
) -> list[dict]:
    for cluster in clusters:
        family_scores = {}

        for member in cluster["members"]:
            family = title_source_family(member["source"])

            family_scores[family] = max(
                family_scores.get(
                    family,
                    0.0,
                ),
                member["score"],
            )

        # Sum the strongest evidence from each independent family.
        score = sum(family_scores.values())

        # Agreement between visible title and embedded metadata is
        # particularly strong evidence.
        if "embedded" in family_scores and "visual" in family_scores:
            score += 4.0

        cluster["score"] = score

        cluster["source_families"] = sorted(family_scores)

    clusters.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return clusters


def select_title(
    document: dict,
    xmp: dict,
    first_page_lines: list[PageLine],
    known_dois: list[str],
) -> dict:
    candidates = []

    embedded = document.get(
        "metadata",
        {},
    ).get("title")

    if plausible_pdf_title(
        embedded,
        document.get(
            "filename",
            "",
        ),
    ) and not looks_like_internal_title(
        embedded,
        known_dois,
    ):
        candidates.append(
            {
                "value": normalize_space(embedded),
                "score": 6.0,
                "source": "pdf_metadata",
            }
        )

    for value in xmp["title"]:
        if plausible_pdf_title(
            value,
            document.get(
                "filename",
                "",
            ),
        ) and not looks_like_internal_title(
            value,
            known_dois,
        ):
            candidates.append(
                {
                    "value": value,
                    "score": 7.0,
                    "source": "xmp",
                }
            )

    visible = build_visible_title_candidates(first_page_lines)

    for candidate in visible[:3]:
        candidates.append(candidate)

    clusters = cluster_text_candidates(
        candidates,
        similarity_threshold=0.78,
    )

    clusters = rescore_title_clusters(clusters)

    # Preserve the visible title location independently from which
    # candidate ultimately supplies the resolved title.
    #
    # This is later used to locate the author block.
    title_bbox = visible[0]["bbox"] if visible else None

    if not clusters:
        return {
            "value": None,
            "confidence": "none",
            "sources": [],
            "candidates": [],
            "title_bbox": title_bbox,
        }

    best = clusters[0]

    families = set(
        best.get(
            "source_families",
            [],
        )
    )

    if "embedded" in families and "visual" in families:
        confidence = "high"

    elif best["score"] >= 7.0:
        confidence = "medium"

    else:
        confidence = "none"

    return {
        "value": (best["value"] if confidence != "none" else None),
        "confidence": confidence,
        "sources": best["sources"],
        "source_families": best.get(
            "source_families",
            [],
        ),
        "candidates": clusters[:5],
        "title_bbox": title_bbox,
    }


# ======================================================================
# Author extraction
# ======================================================================


def looks_like_author_line(
    line: PageLine,
) -> bool:
    text = line.text

    if not line.is_horizontal:
        return False

    if not 3 <= len(text) <= 350:
        return False

    if DOI_RE.search(text):
        return False

    if ARXIV_RE.search(text):
        return False

    if "@" in text:
        return False

    if ABSTRACT_RE.match(text):
        return False

    if AFFILIATION_HINT_RE.search(text):
        return False

    if BOILERPLATE_RE.search(text):
        return False

    words = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ]+",
        text,
    )

    if len(words) < 2:
        return False

    capitalized = sum(bool(word and word[0].isupper()) for word in words)

    cap_ratio = capitalized / len(words)

    if cap_ratio < 0.45:
        return False

    return True


def find_visible_authors(
    lines: list[PageLine],
    title_bbox: list[float] | None,
) -> str | None:
    if not title_bbox:
        return None

    if not lines:
        return None

    title_bottom = float(title_bbox[3])

    page_height = lines[0].page_height

    related_headings = find_related_headings(lines)

    candidates = []

    for line in lines:
        if line.y0 <= title_bottom:
            continue

        # Author block should be fairly close beneath the title.
        if line.y0 > title_bottom + 0.22 * page_height:
            break

        if line_is_in_related_region(
            line,
            related_headings,
        ):
            continue

        if ABSTRACT_RE.match(line.text):
            break

        if looks_like_author_line(line):
            candidates.append(line.text)

        elif candidates and AFFILIATION_HINT_RE.search(line.text):
            break

    if not candidates:
        return None

    return normalize_space(" ".join(candidates))


def author_token_overlap(
    a: str,
    b: str,
) -> float:
    def tokens(
        text: str,
    ) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(
                r"[A-Za-zÀ-ÖØ-öø-ÿ]+",
                text,
            )
            if len(token) >= 2
        }

    ta = tokens(a)

    tb = tokens(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def select_authors(
    document: dict,
    xmp: dict,
    first_page_lines: list[PageLine],
    title_result: dict,
) -> dict:
    candidates = []

    embedded = document.get(
        "metadata",
        {},
    ).get("author")

    if plausible_pdf_author(embedded):
        candidates.append(
            {
                "value": normalize_space(embedded),
                "score": 5.0,
                "source": "pdf_metadata",
            }
        )

    if xmp["authors"]:
        xmp_authors = normalize_space(", ".join(xmp["authors"]))

        candidates.append(
            {
                "value": xmp_authors,
                "score": 8.0,
                "source": "xmp",
            }
        )

    visible = find_visible_authors(
        first_page_lines,
        title_result.get("title_bbox"),
    )

    if visible:
        candidates.append(
            {
                "value": visible,
                "score": 5.0,
                "source": "page_layout",
            }
        )

    if not candidates:
        return {
            "value": None,
            "confidence": "none",
            "sources": [],
            "candidates": [],
        }

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    agreeing_sources = {best["source"]}

    for candidate in candidates[1:]:
        if (
            author_token_overlap(
                best["value"],
                candidate["value"],
            )
            >= 0.40
        ):
            agreeing_sources.add(candidate["source"])

    if best["source"] == "xmp" or len(agreeing_sources) >= 2:
        confidence = "high"

    else:
        confidence = "medium"

    return {
        "value": best["value"],
        "confidence": confidence,
        "sources": sorted(agreeing_sources),
        "candidates": candidates,
    }


# ======================================================================
# Header/footer evidence from pages.jsonl
# ======================================================================


def get_margin_blocks(
    page: dict,
    top_fraction: float = 0.12,
    bottom_fraction: float = 0.15,
) -> list[dict]:
    page_height = float(page["height"])

    top_cutoff = top_fraction * page_height

    bottom_cutoff = (1.0 - bottom_fraction) * page_height

    result = []

    for block in page.get(
        "blocks",
        [],
    ):
        bbox = block.get("bbox")

        if not bbox:
            continue

        text = normalize_space(
            block.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        y0 = float(bbox[1])

        y1 = float(bbox[3])

        region = None

        if y1 <= top_cutoff:
            region = "header"

        elif y0 >= bottom_cutoff:
            region = "footer"

        if region is None:
            continue

        result.append(
            {
                "page_index": page["page_index"],
                "page_number": page.get("page_number"),
                "region": region,
                "text": text,
                "bbox": bbox,
            }
        )

    return result


def collect_margin_blocks(
    pages: list[dict],
) -> list[dict]:
    result = []

    for page in pages:
        result.extend(get_margin_blocks(page))

    return result


def normalize_margin_signature(
    text: str,
) -> str:
    text = text.casefold()

    text = DOI_RE.sub(
        "<doi>",
        text,
    )

    text = re.sub(
        r"https?://\S+",
        "<url>",
        text,
    )

    # Page numbers and changing article numbers should not prevent
    # otherwise-identical footer strings from clustering.
    text = re.sub(
        r"\b\d{1,5}\b",
        "<n>",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def find_recurring_margin_signatures(
    margin_blocks: list[dict],
) -> list[dict]:
    grouped = defaultdict(list)

    for block in margin_blocks:
        signature = normalize_margin_signature(block["text"])

        if len(signature) < 4:
            continue

        grouped[signature].append(block)

    result = []

    for signature, hits in grouped.items():
        pages = sorted({hit["page_index"] for hit in hits})

        if len(pages) < 2:
            continue

        result.append(
            {
                "signature": signature,
                "page_count": len(pages),
                "pages": pages,
                "examples": [hit["text"] for hit in hits[:3]],
            }
        )

    result.sort(
        key=lambda item: item["page_count"],
        reverse=True,
    )

    return result


# ======================================================================
# DOI extraction
# ======================================================================


def add_doi_evidence(
    store: dict,
    doi: str,
    source: str,
    score: float,
    detail: dict | None = None,
):
    doi = clean_doi(doi)

    record = store.setdefault(
        doi,
        {
            "value": doi,
            "score": 0.0,
            "sources": set(),
            "evidence": [],
            "margin_pages": set(),
        },
    )

    record["score"] += score

    record["sources"].add(source)

    evidence = {
        "source": source,
        "score": score,
    }

    if detail:
        evidence.update(detail)

    record["evidence"].append(evidence)

    if (
        detail
        and detail.get("page_index") is not None
        and source.startswith("recurring_")
    ):
        record["margin_pages"].add(detail["page_index"])


def build_doi_candidates(
    document: dict,
    xmp: dict,
    front_page_lines: list[PageLine],
    front_page_text: str,
    margin_blocks: list[dict],
) -> list[dict]:
    store = {}

    # ----------------------------------------------------------
    # Standard PDF metadata
    # ----------------------------------------------------------

    metadata_text = "\n".join(
        str(value)
        for value in document.get(
            "metadata",
            {},
        ).values()
        if value
    )

    for match in DOI_RE.finditer(metadata_text):
        add_doi_evidence(
            store,
            match.group("doi"),
            source=("pdf_metadata"),
            score=6.0,
        )

    # ----------------------------------------------------------
    # XMP
    # ----------------------------------------------------------

    for doi in xmp["doi"]:
        add_doi_evidence(
            store,
            doi,
            source="xmp",
            score=9.0,
        )

    # ----------------------------------------------------------
    # First two pages with geometry
    # ----------------------------------------------------------

    related_headings = find_related_headings(front_page_lines)

    for line in front_page_lines:
        for match in DOI_RE.finditer(line.text):
            if line.page_index == 0:
                score = 3.0

                source = "first_page"

            else:
                score = 2.0

                source = "second_page"

            lower = line.text.casefold()

            # Explicit DOI labels are highly informative.
            if "doi:" in lower or "doi.org" in lower:
                score += 4.0

            if line.y0 < 0.40 * line.page_height:
                score += 1.0

            if line_is_in_related_region(
                line,
                related_headings,
            ):
                score -= 8.0

            add_doi_evidence(
                store,
                match.group("doi"),
                source=source,
                score=score,
                detail={
                    "page_index": line.page_index,
                    "context": line.text,
                },
            )

    # ----------------------------------------------------------
    # Flattened front-page text as weak fallback.
    # ----------------------------------------------------------

    for match in DOI_RE.finditer(front_page_text):
        add_doi_evidence(
            store,
            match.group("doi"),
            source=("front_text"),
            score=1.0,
        )

    # ----------------------------------------------------------
    # Recurring header/footer DOI evidence
    # ----------------------------------------------------------

    margin_occurrences = defaultdict(list)

    for block in margin_blocks:
        seen_in_block = set()

        for match in DOI_RE.finditer(block["text"]):
            doi = clean_doi(match.group("doi"))

            if doi in seen_in_block:
                continue

            seen_in_block.add(doi)

            margin_occurrences[doi].append(block)

    for doi, occurrences in margin_occurrences.items():
        by_page = {}

        for occurrence in occurrences:
            by_page.setdefault(
                occurrence["page_index"],
                occurrence,
            )

        page_count = len(by_page)

        for page_index, occurrence in by_page.items():
            add_doi_evidence(
                store,
                doi,
                source=(f"recurring_{occurrence['region']}"),
                score=3.0,
                detail={
                    "page_index": page_index,
                    "context": occurrence["text"],
                },
            )

        # Recurrence across pages is strong evidence that the DOI
        # belongs to the paper rather than to a related-article box.
        if page_count >= 2:
            add_doi_evidence(
                store,
                doi,
                source=("margin_recurrence"),
                score=8.0,
                detail={
                    "page_count": page_count,
                },
            )

        if page_count >= 4:
            add_doi_evidence(
                store,
                doi,
                source=("strong_margin_recurrence"),
                score=4.0,
                detail={
                    "page_count": page_count,
                },
            )

    result = []

    for candidate in store.values():
        candidate["sources"] = sorted(candidate["sources"])

        candidate["margin_pages"] = sorted(candidate["margin_pages"])

        result.append(candidate)

    result.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return result


def select_doi(
    candidates: list[dict],
) -> dict:
    if not candidates:
        return {
            "value": None,
            "confidence": "none",
            "sources": [],
            "candidates": [],
        }

    best = candidates[0]

    second_score = candidates[1]["score"] if len(candidates) > 1 else None

    sources = set(best["sources"])

    repeated_margin = (
        "margin_recurrence" in sources or "strong_margin_recurrence" in sources
    )

    xmp_and_other = "xmp" in sources and len(sources) >= 2

    clearly_dominant = second_score is None or best["score"] - second_score >= 4.0

    if repeated_margin or xmp_and_other or (best["score"] >= 14 and clearly_dominant):
        confidence = "high"

    elif best["score"] >= 6 and clearly_dominant:
        confidence = "medium"

    else:
        confidence = "none"

    return {
        "value": (best["value"] if confidence != "none" else None),
        "confidence": confidence,
        "sources": best["sources"],
        "candidates": candidates,
    }


# ======================================================================
# Journal extraction
# ======================================================================


def clean_journal_candidate(
    text: str,
) -> str:
    """
    Remove common copyright/publisher boilerplate surrounding
    otherwise-useful recurring journal footer text.

    Example:

        This journal is ©the Owner Societies 2014
        Phys. Chem. Chem. Phys.

    ->

        Phys. Chem. Chem. Phys.
    """

    text = normalize_space(text)

    original = text

    if JOURNAL_BOILERPLATE_RE.search(text):
        year_match = YEAR_RE.search(text)

        if year_match:
            suffix = normalize_space(text[year_match.end() :])

            suffix = suffix.strip(" |,;:-–—")

            if len(suffix) >= 3:
                text = suffix

    text = re.sub(
        r"""
        ^\s*
        this\s+journal\s+is
        \s*
        """,
        "",
        text,
        flags=(re.IGNORECASE | re.VERBOSE),
    )

    text = normalize_space(text)

    if not text:
        return original

    return text


def extract_journal_prefix(
    text: str,
) -> str | None:
    """
    Extract a likely journal name from header/footer text.

    Examples:

        J. Chem. Phys. 159, 123456 (2023)
        Phys. Rev. B 107, 123456
        Journal of Foo 2024, 19, 123-130

    The function also handles copyright-prefixed publisher footers.
    """

    original = normalize_space(text)

    text = clean_journal_candidate(original)

    text = DOI_RE.sub(
        "",
        text,
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    text = normalize_space(text)

    if not text:
        return None

    # If cleanup removed a publisher copyright prefix, the remaining
    # string is already a plausible journal/footer segment.
    boilerplate_was_removed = normalize_compare(text) != normalize_compare(original)

    # Capture text before a typical year / volume / issue number.
    match = re.search(
        r"""
        ^
        (?P<prefix>.*?)
        \s+
        (?:
            19\d{2}
            |
            20\d{2}
            |
            \d{1,3}
        )
        (?=
            \s*
            [,;:|()]
            |
            \s+
            \d
            |
            $
        )
        """,
        text,
        re.VERBOSE,
    )

    if match:
        prefix = normalize_space(match.group("prefix"))

    elif JOURNAL_HINT_RE.search(text):
        prefix = text

    elif boilerplate_was_removed:
        # Example:
        #
        # Phys. Chem. Chem. Phys.
        #
        # may contain no literal word "Journal", yet be a strong
        # recurring footer candidate.
        prefix = text

    else:
        return None

    prefix = prefix.strip(" |,;:-–—")

    # Strip trailing bibliographic numbers if cleanup left them.
    prefix = re.sub(
        r"""
        \s*
        [,;:]?
        \s+
        (?:
            19\d{2}
            |
            20\d{2}
            |
            \d{1,6}
        )
        .*$
        """,
        "",
        prefix,
        flags=re.VERBOSE,
    )

    prefix = normalize_space(prefix)

    if not 3 <= len(prefix) <= 120:
        return None

    if AFFILIATION_HINT_RE.search(prefix):
        return None

    if re.search(
        r"""
        \b
        (?:
            page
            |
            volume
            |
            vol
            |
            issue
            |
            doi
        )
        \b
        """,
        prefix,
        re.IGNORECASE | re.VERBOSE,
    ):
        return None

    if (
        len(
            re.findall(
                r"[A-Za-z]",
                prefix,
            )
        )
        < 3
    ):
        return None

    return prefix


def build_recurring_journal_candidates(
    margin_blocks: list[dict],
) -> list[dict]:
    grouped = {}

    for block in margin_blocks:
        prefix = extract_journal_prefix(block["text"])

        if not prefix:
            continue

        key = normalize_compare(prefix)

        record = grouped.setdefault(
            key,
            {
                "value": prefix,
                "pages": set(),
                "regions": set(),
                "examples": [],
            },
        )

        record["pages"].add(block["page_index"])

        record["regions"].add(block["region"])

        if len(record["examples"]) < 3:
            record["examples"].append(block["text"])

    result = []

    for record in grouped.values():
        page_count = len(record["pages"])

        if page_count < 2:
            continue

        score = 2.0 + 2.0 * min(
            page_count,
            5,
        )

        result.append(
            {
                "value": record["value"],
                "score": score,
                "source": "recurring_margin",
                "page_count": page_count,
                "pages": sorted(record["pages"]),
                "regions": sorted(record["regions"]),
                "examples": record["examples"],
            }
        )

    result.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return result


def build_journal_candidates(
    document: dict,
    xmp: dict,
    first_page_lines: list[PageLine],
    recurring: list[dict],
) -> list[dict]:
    candidates = []

    # ----------------------------------------------------------
    # XMP publication name
    # ----------------------------------------------------------

    for value in xmp["journal"]:
        candidates.append(
            {
                "value": value,
                "score": 9.0,
                "source": "xmp",
            }
        )

    # ----------------------------------------------------------
    # Standard PDF subject
    # ----------------------------------------------------------

    subject = document.get(
        "metadata",
        {},
    ).get("subject")

    if subject and JOURNAL_HINT_RE.search(subject):
        candidates.append(
            {
                "value": normalize_space(subject),
                "score": 4.0,
                "source": "pdf_metadata.subject",
            }
        )

    # ----------------------------------------------------------
    # First-page header/footer-like lines
    # ----------------------------------------------------------

    related_headings = find_related_headings(first_page_lines)

    for line in first_page_lines:
        if line.y0 > 0.20 * line.page_height:
            continue

        if line_is_in_related_region(
            line,
            related_headings,
        ):
            continue

        prefix = extract_journal_prefix(line.text)

        if prefix:
            candidates.append(
                {
                    "value": prefix,
                    "score": 3.0,
                    "source": "first_page",
                }
            )

    # ----------------------------------------------------------
    # Recurring margins
    # ----------------------------------------------------------

    candidates.extend(recurring)

    return candidates


def select_journal(
    candidates: list[dict],
) -> dict:
    clusters = cluster_text_candidates(
        candidates,
        similarity_threshold=0.72,
    )

    if not clusters:
        return {
            "value": None,
            "confidence": "none",
            "sources": [],
            "candidates": [],
        }

    best = clusters[0]

    sources = set(best["sources"])

    has_xmp = "xmp" in sources

    recurring_page_count = max(
        (
            member.get(
                "page_count",
                0,
            )
            for member in best["members"]
        ),
        default=0,
    )

    if (has_xmp and len(sources) >= 2) or (recurring_page_count >= 3):
        confidence = "high"

    elif has_xmp or (best["score"] >= 6):
        confidence = "medium"

    else:
        confidence = "none"

    return {
        "value": (best["value"] if confidence != "none" else None),
        "confidence": confidence,
        "sources": best["sources"],
        "candidates": clusters[:5],
    }


# ======================================================================
# Other identifiers
# ======================================================================


def extract_identifiers(
    texts: list[str],
    xmp: dict,
    selected_doi: str | None,
) -> dict:
    full_text = "\n".join(text for text in texts if text)

    arxiv = {match.group("id") for match in ARXIV_RE.finditer(full_text)}

    pmid = {match.group("id") for match in PMID_RE.finditer(full_text)}

    pmc = {match.group("id").upper() for match in PMC_RE.finditer(full_text)}

    issn = {match.group("id").upper() for match in ISSN_RE.finditer(full_text)}

    issn.update(value.upper() for value in xmp["issn"])

    eissn = {value.upper() for value in xmp["eissn"]}

    return {
        "doi": ([selected_doi] if selected_doi else []),
        "issn": sorted(issn),
        "eissn": sorted(eissn),
        "arxiv": sorted(arxiv),
        "pmid": sorted(pmid),
        "pmc": sorted(pmc),
        "xmp_identifiers": xmp["identifiers"],
    }


# ======================================================================
# Bibliographic fields from XMP
# ======================================================================


def build_bibliographic_fields(
    xmp: dict,
) -> dict:
    publication_date = first_or_none(xmp["publication_date"])

    year = None

    if publication_date:
        match = YEAR_RE.search(publication_date)

        if match:
            year = match.group("year")

    return {
        "volume": first_or_none(xmp["volume"]),
        "issue": first_or_none(xmp["issue"]),
        "year": year,
        "publication_date": publication_date,
        "starting_page": first_or_none(xmp["starting_page"]),
        "ending_page": first_or_none(xmp["ending_page"]),
    }


# ======================================================================
# One document
# ======================================================================


def extract_document_metadata(
    document: dict,
    pages: list[dict],
    pdf_path: Path,
) -> dict:
    # --------------------------------------------------------------
    # PDF-native metadata and front-page typography
    # --------------------------------------------------------------

    with fitz.open(pdf_path) as pdf:
        xmp = extract_xmp_metadata(pdf)

        front_page_lines = []

        # Use detailed geometric extraction for the first two pages.
        # Page 1 is needed for title/authors; page 2 improves DOI
        # coverage for publishers that put bibliographic information
        # after a cover/front-matter page.
        for page_index in range(
            min(
                len(pdf),
                2,
            )
        ):
            front_page_lines.extend(
                extract_page_lines(
                    pdf[page_index],
                    page_index=(page_index),
                )
            )

    first_page_lines = [line for line in front_page_lines if line.page_index == 0]

    # --------------------------------------------------------------
    # Existing pages.jsonl for whole-document recurring evidence
    # --------------------------------------------------------------

    margin_blocks = collect_margin_blocks(pages)

    recurring_signatures = find_recurring_margin_signatures(margin_blocks)

    recurring_journals = build_recurring_journal_candidates(margin_blocks)

    # --------------------------------------------------------------
    # Flattened text from first two pages
    # --------------------------------------------------------------

    front_page_text = "\n".join(page.get("text", "") for page in pages[:2])

    # --------------------------------------------------------------
    # DOI hints needed for filtering broken embedded titles
    # --------------------------------------------------------------

    known_dois = collect_known_dois(
        document=document,
        xmp=xmp,
    )

    # --------------------------------------------------------------
    # Title
    # --------------------------------------------------------------

    title = select_title(
        document=document,
        xmp=xmp,
        first_page_lines=(first_page_lines),
        known_dois=(known_dois),
    )

    # --------------------------------------------------------------
    # Authors
    # --------------------------------------------------------------

    authors = select_authors(
        document=document,
        xmp=xmp,
        first_page_lines=(first_page_lines),
        title_result=title,
    )

    # --------------------------------------------------------------
    # DOI
    # --------------------------------------------------------------

    doi_candidates = build_doi_candidates(
        document=document,
        xmp=xmp,
        front_page_lines=(front_page_lines),
        front_page_text=(front_page_text),
        margin_blocks=(margin_blocks),
    )

    doi = select_doi(doi_candidates)

    # --------------------------------------------------------------
    # Journal
    # --------------------------------------------------------------

    journal_candidates = build_journal_candidates(
        document=document,
        xmp=xmp,
        first_page_lines=(first_page_lines),
        recurring=(recurring_journals),
    )

    journal = select_journal(journal_candidates)

    # --------------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------------

    metadata_text = "\n".join(
        str(value)
        for value in document.get(
            "metadata",
            {},
        ).values()
        if value
    )

    margin_text = "\n".join(block["text"] for block in margin_blocks)

    identifiers = extract_identifiers(
        texts=[
            metadata_text,
            front_page_text,
            margin_text,
        ],
        xmp=xmp,
        selected_doi=(doi["value"]),
    )

    bibliographic = build_bibliographic_fields(xmp)

    # --------------------------------------------------------------
    # Review flags
    # --------------------------------------------------------------

    needs_review = []

    for field_name, field in [
        (
            "title",
            title,
        ),
        (
            "authors",
            authors,
        ),
        (
            "journal",
            journal,
        ),
        (
            "doi",
            doi,
        ),
    ]:
        if field["value"] is None:
            needs_review.append(field_name)

    # Layout bbox is an internal processing detail rather than useful
    # bibliographic output.
    output_title = {key: value for key, value in title.items() if key != "title_bbox"}

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document["document_id"],
        "filename": document.get("filename"),
        "relative_path": document.get("relative_path"),
        # ----------------------------------------------------------
        # Convenient resolved values
        # ----------------------------------------------------------
        "title": title["value"],
        "authors_raw": authors["value"],
        "journal": journal["value"],
        "doi": doi["value"],
        "identifiers": identifiers,
        "bibliographic": bibliographic,
        # ----------------------------------------------------------
        # Full provenance / diagnostics
        # ----------------------------------------------------------
        "fields": {
            "title": output_title,
            "authors": authors,
            "journal": journal,
            "doi": doi,
        },
        "metadata_sources": {
            "pdf_metadata_present": bool(document.get("metadata")),
            "xmp_present": xmp.get(
                "present",
                False,
            ),
            "xmp_parse_error": xmp.get(
                "parse_error",
                False,
            ),
        },
        "recurring_margins": {
            "signatures": recurring_signatures[:10],
            "journal_candidates": recurring_journals[:5],
        },
        "needs_review": needs_review,
    }


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract bibliographic metadata using "
            "standard PDF metadata, XMP, first-page "
            "layout, and recurring header/footer evidence."
        )
    )

    parser.add_argument(
        "documents",
        type=Path,
        help=("Existing data/extracted/documents.jsonl"),
    )

    parser.add_argument(
        "pages",
        type=Path,
        help=("Existing data/extracted/pages.jsonl"),
    )

    parser.add_argument(
        "pdf_root",
        type=Path,
        help=("Root relative to which document relative_path values are resolved"),
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output metadata JSONL",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Print complete metadata record for every processed PDF"),
    )

    args = parser.parse_args()

    documents_path = args.documents.resolve()

    pages_path = args.pages.resolve()

    pdf_root = args.pdf_root.resolve()

    output_path = args.output.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = load_documents(documents_path)

    pages_by_document = load_pages_by_document(pages_path)

    successful = 0
    failed = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out:
        for index, document in enumerate(
            documents,
            start=1,
        ):
            document_id = document["document_id"]

            filename = document.get(
                "filename",
                "",
            )

            print(f"[{index}/{len(documents)}] {filename}")

            try:
                pdf_path = resolve_pdf_path(
                    document,
                    pdf_root,
                )

                pages = pages_by_document.get(
                    document_id,
                    [],
                )

                if not pages:
                    raise RuntimeError(f"No extracted pages found for {document_id}")

                result = extract_document_metadata(
                    document=document,
                    pages=pages,
                    pdf_path=pdf_path,
                )

                out.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                successful += 1

                if args.debug:
                    print(
                        json.dumps(
                            result,
                            indent=2,
                            ensure_ascii=False,
                        )
                    )

            except Exception as exc:
                failed += 1

                error_record = {
                    "schema_version": SCHEMA_VERSION,
                    "document_id": document_id,
                    "filename": filename,
                    "error": str(exc),
                }

                out.write(
                    json.dumps(
                        error_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                print(f"  ERROR: {exc}")

    print()
    print("Finished")

    print(f"  Successful: {successful}")

    print(f"  Failed    : {failed}")

    print(f"  Output    : {output_path}")


if __name__ == "__main__":
    main()
