#!/usr/bin/env python3

import argparse
import json
import math
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import spacy

SENTENCE_BUILDER_VERSION = 2


# Common abbreviations in scientific writing where a simple rule-based
# sentence splitter may otherwise occasionally create a false boundary.
ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "fig.",
    "figs.",
    "eq.",
    "eqs.",
    "sec.",
    "secs.",
    "sect.",
    "sects.",
    "ref.",
    "refs.",
    "no.",
    "nos.",
    "dr.",
    "prof.",
    "vs.",
    "cf.",
    "approx.",
)
CITATION_BOUNDARY_RE = re.compile(
    r"""
    (?<=[A-Za-z)\]])        # punctuation follows ordinary text
    [.!?]                   # sentence-ending punctuation

    (?:
        \[
            \d+
            (?:\s*[,;–-]\s*\d+)*
        \]
        |
        \(
            \d+
            (?:\s*[,;–-]\s*\d+)*
        \)
        |
        \d+
        (?:\s*[,;–-]\s*\d+)*
    )

    (?P<gap>\s+)

    (?=[A-Z])               # next sentence starts uppercase
    """,
    re.VERBOSE,
)


def create_nlp(language: str = "en"):
    """
    Create a lightweight spaCy pipeline used only for sentence splitting.

    No downloaded language model is required.
    """
    nlp = spacy.blank(language)

    nlp.add_pipe(
        "sentencizer",
        config={
            "punct_chars": [".", "!", "?", "…"],
        },
    )

    return nlp


def furniture_signature(text: str) -> str:
    """
    Normalize recurring header/footer text.

    Numbers are replaced so that:

        160, 114107-2
        160, 114107-3

    receive the same signature.
    """
    text = text.casefold()

    text = re.sub(
        r"\d+",
        "#",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def is_margin_block(
    block: dict,
    page: dict,
    margin_ratio: float = 0.15,
) -> bool:
    """
    Return True if the center of a block lies in the upper or lower
    margin region of the page.
    """
    _, y0, _, y1 = block["bbox"]

    center_y = (y0 + y1) / 2
    height = page["height"]

    return center_y < height * margin_ratio or center_y > height * (1.0 - margin_ratio)


def find_furniture_signatures(
    pages: list[dict],
    margin_ratio: float = 0.15,
    min_fraction: float = 0.30,
) -> set[str]:
    """
    Find recurring text blocks in page margins.

    A block must occur on at least 30% of pages, with an absolute
    minimum of 2 pages.
    """
    occurrences = defaultdict(set)

    for page in pages:
        for block in page["blocks"]:
            if not is_margin_block(
                block=block,
                page=page,
                margin_ratio=margin_ratio,
            ):
                continue

            signature = furniture_signature(block["text"])

            if len(signature) < 3:
                continue

            occurrences[signature].add(page["page_index"])

    threshold = max(
        2,
        math.ceil(len(pages) * min_fraction),
    )

    return {
        signature
        for signature, page_indices in occurrences.items()
        if len(page_indices) >= threshold
    }


def get_body_block_numbers(
    page: dict,
    furniture_signatures: set[str],
    margin_ratio: float = 0.15,
) -> set[int]:
    """
    Return block numbers that should participate in sentence building.
    """
    result = set()

    for block in page["blocks"]:
        signature = furniture_signature(block["text"])

        recurring_margin_material = (
            is_margin_block(
                block=block,
                page=page,
                margin_ratio=margin_ratio,
            )
            and signature in furniture_signatures
        )

        # Also discard margin blocks consisting almost entirely
        # of standalone page numbers / punctuation.
        numeric_margin_material = (
            is_margin_block(
                block=block,
                page=page,
                margin_ratio=margin_ratio,
            )
            and re.fullmatch(
                r"[\W\d_]+",
                block["text"].strip(),
            )
            is not None
        )

        if recurring_margin_material or numeric_margin_material:
            continue

        result.add(block["block_no"])

    return result


def group_words_by_block_and_line(
    words: list[dict],
    allowed_block_numbers: set[int] | None = None,
):
    """
    Group page words by block and line while preserving the original
    native extraction order.

    Returns an OrderedDict:

        block_no
            -> OrderedDict
                line_no
                    -> [(page_word_index, word_record), ...]
    """
    blocks = OrderedDict()

    for page_word_index, word in enumerate(words):
        block_no = word["block_no"]
        line_no = word["line_no"]

        if block_no not in blocks:
            blocks[block_no] = OrderedDict()

        if line_no not in blocks[block_no]:
            blocks[block_no][line_no] = []

        blocks[block_no][line_no].append((page_word_index, word))

    return blocks


def needs_space(
    previous_text: str,
    current_text: str,
    is_new_line: bool,
) -> bool:
    """
    Decide whether a space should be inserted between two extracted words.

    The important special case is line-end hyphenation:

        multi-
        modal

    becomes:

        multi-modal

    We deliberately KEEP the hyphen in v1.
    """
    if not previous_text:
        return False

    # At a physical line break:
    #
    #     multi-
    #     modal
    #
    # becomes "multi-modal".
    if is_new_line and previous_text.endswith("-"):
        return False

    # Avoid spaces before common closing punctuation when PyMuPDF
    # happens to return punctuation as an independent word.
    no_space_before = ",.;:!?%)]}»”’"

    if current_text and current_text[0] in no_space_before:
        return False

    # Avoid spaces immediately after opening punctuation.
    no_space_after = "([{«“‘"

    if previous_text and previous_text[-1] in no_space_after:  # noqa: SIM103
        return False

    return True


def reconstruct_words(
    source_word_indices: list[int],
    page_words: list[dict],
) -> str:
    """
    Reconstruct text from selected source words.

    If dehyphenate=True, remove likely artificial line-wrap hyphens:

        partici-
        pants

    -> participants

    The original PDF word records are never modified.
    """
    parts = []
    previous_word = None

    for page_word_index in source_word_indices:
        word = page_words[page_word_index]
        current_text = word["text"]

        if previous_word is None:
            parts.append(current_text)
            previous_word = word
            continue

        is_new_line = (
            word["block_no"] != previous_word["block_no"]
            or word["line_no"] != previous_word["line_no"]
        )

        previous_text = previous_word["text"]

        likely_artificial_hyphen = (
            is_new_line
            and previous_text.endswith("-")
            and len(previous_text) > 2
            and current_text
            and current_text[0].islower()
        )

        if likely_artificial_hyphen:
            # Previous word is currently the final text component.
            #
            # "partici-" + "pants"
            # becomes
            # "partici" + "pants"
            parts[-1] = parts[-1].removesuffix("-")

            parts.append(current_text)

        else:
            if needs_space(
                previous_text=previous_text,
                current_text=current_text,
                is_new_line=is_new_line,
            ):
                parts.append(" ")

            parts.append(current_text)

        previous_word = word

    return "".join(parts)


def reconstruct_block(lines: OrderedDict):
    """
    Reconstruct normalized text for one PyMuPDF text block.

    At the same time, retain character spans mapping every reconstructed
    word back to its original page word index.

    Returns:
        block_text: str

        word_spans: [
            {
                "page_word_index": ...,
                "start": ...,
                "end": ...,
                ...
            },
            ...
        ]
    """
    parts = []
    word_spans = []

    cursor = 0
    previous_text = ""
    first_word = True

    for line_number_in_block, (line_no, line_words) in enumerate(lines.items()):
        for word_number_in_line, (page_word_index, word) in enumerate(line_words):
            word_text = word["text"]

            is_new_line = line_number_in_block > 0 and word_number_in_line == 0

            if not first_word and needs_space(
                previous_text=previous_text,
                current_text=word_text,
                is_new_line=is_new_line,
            ):
                parts.append(" ")
                cursor += 1

            start = cursor

            parts.append(word_text)
            cursor += len(word_text)

            end = cursor

            word_spans.append(
                {
                    "page_word_index": page_word_index,
                    "start": start,
                    "end": end,
                    "block_no": word["block_no"],
                    "line_no": word["line_no"],
                    "word_no": word["word_no"],
                }
            )

            previous_text = word_text
            first_word = False

    return "".join(parts), word_spans


def looks_like_abbreviation(text: str) -> bool:
    """
    Return True when the end of `text` looks like a common scientific
    abbreviation rather than a genuine sentence ending.
    """
    stripped = text.rstrip()
    lowered = stripped.lower()

    if any(lowered.endswith(abbr) for abbr in ABBREVIATIONS):
        return True

    # Author initials, for example:
    #
    #     Smith, J. Doe ...
    #
    # Keep this rule deliberately narrow.
    if re.search(r"\b[A-Z]\.$", stripped):  # noqa: SIM103
        return True

    return False


def apply_citation_boundaries(
    text: str,
    spans: list[tuple[int, int]],
):
    """
    Split sentence spans where a sentence-final numeric citation
    prevented the normal sentence segmenter from finding the boundary.
    """
    result = []

    for span_start, span_end in spans:
        cursor = span_start

        for match in CITATION_BOUNDARY_RE.finditer(
            text,
            span_start,
            span_end,
        ):
            # End the previous sentence after the citation numbers,
            # but before the whitespace.
            previous_end = match.start("gap")

            # Start the next sentence after the whitespace.
            next_start = match.end("gap")

            if previous_end > cursor:
                result.append((cursor, previous_end))

            cursor = next_start

        if cursor < span_end:
            result.append((cursor, span_end))

    return result


def split_sentence_spans(text: str, nlp):
    """
    Return sentence character spans for one reconstructed block.

    spaCy performs the initial segmentation. A small post-processing
    step repairs a few obvious scientific-abbreviation boundaries.
    """
    if not text.strip():
        return []

    doc = nlp(text)

    spans = [
        (sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()
    ]

    if not spans:
        return []

    merged = []

    for start, end in spans:
        if not merged:
            merged.append([start, end])
            continue

        previous_start, previous_end = merged[-1]
        previous_text = text[previous_start:previous_end]

        if looks_like_abbreviation(previous_text):
            merged[-1][1] = end
        else:
            merged.append([start, end])

    merged = [(start, end) for start, end in merged]

    return apply_citation_boundaries(text=text, spans=merged)


def source_words_for_sentence(
    sentence_start: int,
    sentence_end: int,
    word_spans: list[dict],
):
    """
    Find all source PDF words overlapping a sentence character span.
    """
    selected = []

    for span in word_spans:
        overlaps = span["end"] > sentence_start and span["start"] < sentence_end

        if overlaps:
            selected.append(span)

    return selected


def make_line_boxes(
    source_word_indices: list[int],
    page_words: list[dict],
):
    """
    Build one bounding rectangle per physical PDF line occupied
    by the sentence.

    This is substantially better for highlighting than one large
    rectangle around the complete sentence.
    """
    grouped = OrderedDict()

    for page_word_index in source_word_indices:
        word = page_words[page_word_index]

        key = (
            word["block_no"],
            word["line_no"],
        )

        if key not in grouped:
            grouped[key] = []

        grouped[key].append(word)

    line_boxes = []

    for (block_no, line_no), words in grouped.items():
        x0 = min(word["bbox"][0] for word in words)
        y0 = min(word["bbox"][1] for word in words)
        x1 = max(word["bbox"][2] for word in words)
        y1 = max(word["bbox"][3] for word in words)

        line_boxes.append(
            {
                "block_no": block_no,
                "line_no": line_no,
                "bbox": [
                    round(x0, 3),
                    round(y0, 3),
                    round(x1, 3),
                    round(y1, 3),
                ],
            }
        )

    return line_boxes


def build_sentences_for_page(
    page: dict,
    nlp,
    allowed_block_numbers: set[int],
):
    """
    Build sentence records for one page.

    Returns:
        sentences
    """
    page_words = page["words"]

    if not page_words:
        return []

    blocks = group_words_by_block_and_line(
        page_words,
        allowed_block_numbers=allowed_block_numbers,
    )

    sentences = []

    for block_no, lines in blocks.items():
        block_text, word_spans = reconstruct_block(lines)

        if not block_text.strip():
            continue

        sentence_spans = split_sentence_spans(
            text=block_text,
            nlp=nlp,
        )

        for sentence_start, sentence_end in sentence_spans:
            source_text = block_text[sentence_start:sentence_end].strip()

            if not source_text:
                continue

            source_spans = source_words_for_sentence(
                sentence_start=sentence_start,
                sentence_end=sentence_end,
                word_spans=word_spans,
            )

            if not source_spans:
                continue

            source_word_indices = [span["page_word_index"] for span in source_spans]

            normalized_text = reconstruct_words(
                source_word_indices=source_word_indices,
                page_words=page_words,
            )

            line_boxes = make_line_boxes(
                source_word_indices=source_word_indices,
                page_words=page_words,
            )

            source_segment = {
                "page_index": page["page_index"],
                "page_number": page["page_number"],
                "block_no": block_no,
                "source_word_indices": source_word_indices,
                "boxes": line_boxes,
                "character_count": len(source_text),
                "source_word_count": len(source_word_indices),
            }

            sentence_record = {
                "schema_version": 2,
                "sentence_builder_version": 2,
                "document_id": page["document_id"],
                "page_numbers": [page["page_number"]],
                "text": source_text,
                "text_normalized": normalized_text,
                "source_segments": [source_segment],
            }

            sentences.append(sentence_record)

    return sentences


def ends_like_complete_sentence(text: str) -> bool:
    text = text.strip()

    # Ordinary sentence ending.
    if re.search(
        r'[.!?]["”’)\]]*$',
        text,
    ):
        return True

    # Sentence-ending punctuation followed by a numeric citation.
    if re.search(  # noqa: SIM103
        r"""
        [.!?]
        (?:
            \[\d+(?:\s*[,;–-]\s*\d+)*\]
            |
            \(\d+(?:\s*[,;–-]\s*\d+)*\)
            |
            \d+(?:\s*[,;–-]\s*\d+)*
        )
        ["”’)\]]*
        $
        """,
        text,
        re.VERBOSE,
    ):
        return True

    return False


def starts_like_continuation(text: str) -> bool:
    """
    Conservative test.

    Page-spanning sentence continuations very often start lowercase.
    """
    for char in text.lstrip():
        if char.isalpha():
            return char.islower()

        if char.isdigit():
            return True

    return False


def join_across_page(
    left: str,
    right: str,
    dehyphenate: bool,
) -> str:
    left = left.rstrip()
    right = right.lstrip()

    if left.endswith("-") and right and right[0].islower():
        if dehyphenate:
            return left[:-1] + right

        return left + right

    return left + " " + right


def merge_cross_page_fragments(
    sentences: list[dict],
) -> list[dict]:
    if not sentences:
        return []

    result = [sentences[0]]

    for current in sentences[1:]:
        previous = result[-1]

        previous_last_page = previous["page_numbers"][-1]
        current_first_page = current["page_numbers"][0]

        adjacent_pages = current_first_page == previous_last_page + 1

        likely_continuation = (
            adjacent_pages
            and not ends_like_complete_sentence(previous["text"])
            and starts_like_continuation(current["text"])
        )

        if not likely_continuation:
            result.append(current)
            continue

        previous["text"] = join_across_page(
            previous["text"],
            current["text"],
            dehyphenate=False,
        )

        previous["text_normalized"] = join_across_page(
            previous["text_normalized"],
            current["text_normalized"],
            dehyphenate=True,
        )

        previous["page_numbers"].extend(
            page
            for page in current["page_numbers"]
            if page not in previous["page_numbers"]
        )

        previous["source_segments"].extend(current["source_segments"])

        previous["cross_page"] = True

    return result


def main():
    parser = argparse.ArgumentParser(
        description=("Build sentence-level records from extracted PyMuPDF page data.")
    )

    parser.add_argument(
        "pages_jsonl",
        type=Path,
        help="pages.jsonl produced by extract_pdfs.py",
    )

    parser.add_argument(
        "output_jsonl",
        type=Path,
        help="Output sentence JSONL file",
    )

    parser.add_argument(
        "--language",
        default="en",
        required=False,
        help=(
            "spaCy language code used for rule-based sentence segmentation. Default: en"
        ),
    )

    args = parser.parse_args()

    pages_path = args.pages_jsonl.resolve()
    output_path = args.output_jsonl.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading sentence segmenter: {args.language}")
    nlp = create_nlp(args.language)

    # ------------------------------------------------------------
    # Load pages and group them by document.
    # ------------------------------------------------------------

    pages_by_document = defaultdict(list)

    page_count = 0
    empty_page_count = 0

    with pages_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                page = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}") from exc

            page_count += 1

            if not page.get("words"):
                empty_page_count += 1

            pages_by_document[page["document_id"]].append(page)

    # ------------------------------------------------------------
    # Statistics.
    # ------------------------------------------------------------

    document_count = len(pages_by_document)

    sentence_count = 0
    short_sentence_count = 0
    cross_page_sentence_count = 0

    detected_furniture_signature_count = 0
    removed_furniture_block_count = 0

    sentences_before_merge = 0
    sentences_after_merge = 0

    print()
    print(f"Loaded {page_count} pages from {document_count} document(s)")
    print()

    # ------------------------------------------------------------
    # Process complete documents.
    # ------------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for document_number, (
            document_id,
            pages,
        ) in enumerate(
            pages_by_document.items(),
            start=1,
        ):
            pages.sort(key=lambda page: page["page_index"])

            print(
                f"[{document_number}/{document_count}] "
                f"{document_id} "
                f"({len(pages)} pages)"
            )

            # ----------------------------------------------------
            # Detect repeating headers / footers for this document.
            # ----------------------------------------------------

            furniture_signatures = find_furniture_signatures(pages)

            detected_furniture_signature_count += len(furniture_signatures)

            document_sentences = []
            document_removed_blocks = 0

            # ----------------------------------------------------
            # Build sentences page by page.
            # ----------------------------------------------------

            for page in pages:
                if not page.get("words"):
                    continue

                body_block_numbers = get_body_block_numbers(
                    page=page,
                    furniture_signatures=(furniture_signatures),
                )

                all_text_block_numbers = {
                    block["block_no"]
                    for block in page["blocks"]
                    if block["block_type"] == 0
                }

                removed_blocks = all_text_block_numbers - body_block_numbers

                document_removed_blocks += len(removed_blocks)

                page_sentences = build_sentences_for_page(
                    page=page,
                    nlp=nlp,
                    allowed_block_numbers=(body_block_numbers),
                )

                document_sentences.extend(page_sentences)

            before_merge = len(document_sentences)

            sentences_before_merge += before_merge

            # ----------------------------------------------------
            # Merge likely sentence fragments across page breaks.
            # ----------------------------------------------------

            document_sentences = merge_cross_page_fragments(document_sentences)

            after_merge = len(document_sentences)

            sentences_after_merge += after_merge

            merged_count = before_merge - after_merge

            # ----------------------------------------------------
            # Assign final document-level sentence IDs.
            # ----------------------------------------------------

            document_cross_page_count = 0
            document_short_count = 0

            for index, sentence in enumerate(document_sentences):
                sentence["document_sentence_index"] = index

                sentence["sentence_id"] = f"{document_id}:s{index:06d}"

                if sentence.get(
                    "cross_page",
                    False,
                ):
                    document_cross_page_count += 1

                # Count using normalized text because this is
                # what retrieval will ultimately see.
                word_count = len(sentence["text_normalized"].split())

                if word_count < 4:
                    document_short_count += 1

                output_file.write(
                    json.dumps(
                        sentence,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            sentence_count += after_merge

            short_sentence_count += document_short_count

            cross_page_sentence_count += document_cross_page_count

            removed_furniture_block_count += document_removed_blocks

            print(
                f"    sentences={after_merge}, "
                f"cross-page merges={merged_count}, "
                f"furniture blocks removed="
                f"{document_removed_blocks}"
            )

    # ------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------

    print()
    print("Finished")
    print(f"  Documents processed        : {document_count}")
    print(f"  Pages processed            : {page_count}")
    print(f"  Pages without words        : {empty_page_count}")
    print()
    print(f"  Sentences before page merge: {sentences_before_merge}")
    print(f"  Sentences after page merge : {sentences_after_merge}")
    print(f"  Cross-page sentences       : {cross_page_sentence_count}")
    print(f"  Very short sentences       : {short_sentence_count}")
    print()
    print(f"  Furniture signatures found : {detected_furniture_signature_count}")
    print(f"  Furniture blocks removed   : {removed_furniture_block_count}")
    print()
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
