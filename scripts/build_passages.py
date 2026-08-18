#!/usr/bin/env python3

import argparse
import json
from collections import OrderedDict
from pathlib import Path

PASSAGE_BUILDER_VERSION = 2


def word_count(text: str) -> int:
    """
    Simple retrieval-oriented word count.

    This is deliberately tokenizer-independent. We will use the
    BGE-M3 tokenizer later when building embeddings if necessary.
    """
    return len(text.split())


def group_sentences(sentences: list[dict]):
    """
    Group sentences by document, page, and PyMuPDF block while
    preserving their existing order.

    Returns:

        (document_id, page_index, block_no)
            -> [sentence, sentence, ...]
    """
    groups = OrderedDict()

    for sentence in sentences:
        logical_block_index = sentence.get(
            "logical_block_index",
            sentence["block_no"],
        )

        key = (
            sentence["document_id"],
            sentence["page_index"],
            logical_block_index,
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(sentence)

    return groups


def make_passage(
    sentences: list[dict],
    passage_index: int,
) -> dict:
    """
    Create one passage record from consecutive sentences.
    """
    if not sentences:
        raise ValueError("Cannot create an empty passage")

    first = sentences[0]
    last = sentences[-1]

    source_text = " ".join(sentence["text"].strip() for sentence in sentences)

    normalized_text = " ".join(
        sentence["text_normalized"].strip() for sentence in sentences
    )

    logical_block_index = first.get(
        "logical_block_index",
        first["block_no"],
    )

    physical_block_nos = sorted(
        {
            block_no
            for sentence in sentences
            for block_no in sentence.get(
                "block_nos",
                [sentence["block_no"]],
            )
        }
    )

    sentence_ids = [sentence["sentence_id"] for sentence in sentences]

    passage_id = (
        f"{first['document_id']}:"
        f"p{first['page_number']:04d}:"
        f"lb{logical_block_index:04d}:"
        f"c{passage_index:03d}"
    )

    normalized_word_count = word_count(normalized_text)

    return {
        "schema_version": 2,
        "passage_builder_version": PASSAGE_BUILDER_VERSION,
        "passage_id": passage_id,
        "document_id": first["document_id"],
        "page_index": first["page_index"],
        "page_number": first["page_number"],
        "physical_block_nos": physical_block_nos,
        "block_passage_index": passage_index,
        "sentence_ids": sentence_ids,
        "first_page_sentence_index": (first["page_sentence_index"]),
        "last_page_sentence_index": (last["page_sentence_index"]),
        "sentence_count": len(sentences),
        "text": source_text,
        "text_normalized": normalized_text,
        "word_count": normalized_word_count,
        # Useful later for diagnosing noisy retrieval results.
        "retrieval_eligible": (normalized_word_count >= 8),
    }


def build_passages_for_block(
    sentences: list[dict],
    max_words: int,
    overlap_sentences: int,
) -> list[dict]:
    """
    Split consecutive sentences from one PDF block into passages.

    Passages never cross a block or page boundary.

    If a block exceeds max_words, consecutive passages overlap by
    overlap_sentences.
    """
    if not sentences:
        return []

    passages = []

    start = 0
    passage_index = 0

    while start < len(sentences):
        end = start
        current_words = 0

        while end < len(sentences):
            sentence_words = word_count(sentences[end]["text_normalized"])

            # Always include at least one sentence, even if that
            # sentence itself exceeds max_words.
            if end > start and current_words + sentence_words > max_words:
                break

            current_words += sentence_words
            end += 1

        passage_sentences = sentences[start:end]

        passages.append(
            make_passage(
                sentences=passage_sentences,
                passage_index=passage_index,
            )
        )

        passage_index += 1

        # Finished the block.
        if end >= len(sentences):
            break

        # Overlap the next passage by the requested number
        # of complete sentences.
        next_start = max(
            start + 1,
            end - overlap_sentences,
        )

        start = next_start

    return passages


def main():
    parser = argparse.ArgumentParser(
        description=("Build retrieval passages from sentence-level PDF records.")
    )

    parser.add_argument(
        "sentences_jsonl",
        type=Path,
        help="sentences.jsonl produced by build_sentences.py",
    )

    parser.add_argument(
        "output_jsonl",
        type=Path,
        help="Output passages JSONL file",
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=180,
        help=("Maximum approximate words per passage. Default: 180"),
    )

    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
        help=(
            "Number of sentences shared by consecutive "
            "passages from the same block. Default: 1"
        ),
    )

    args = parser.parse_args()

    sentences_path = args.sentences_jsonl.resolve()

    output_path = args.output_jsonl.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.max_words <= 0:
        raise ValueError("--max-words must be greater than zero")

    if args.overlap_sentences < 0:
        raise ValueError("--overlap-sentences cannot be negative")

    # ------------------------------------------------------------
    # Load sentences.
    #
    # For a bibliography-sized corpus this is entirely reasonable.
    # We can stream/group differently later if needed.
    # ------------------------------------------------------------

    sentences = []

    with sentences_path.open(
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
                sentence = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}") from exc

            sentences.append(sentence)

    groups = group_sentences(sentences)

    # ------------------------------------------------------------
    # Statistics.
    # ------------------------------------------------------------

    passage_count = 0
    eligible_passage_count = 0
    short_passage_count = 0

    split_block_count = 0
    block_count = len(groups)

    total_words = 0

    # ------------------------------------------------------------
    # Build passages.
    # ------------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for (
            document_id,
            page_index,
            block_no,
        ), block_sentences in groups.items():  # noqa: PERF102
            passages = build_passages_for_block(
                sentences=block_sentences,
                max_words=args.max_words,
                overlap_sentences=(args.overlap_sentences),
            )

            if len(passages) > 1:
                split_block_count += 1

            for passage in passages:
                output_file.write(
                    json.dumps(
                        passage,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                passage_count += 1

                total_words += passage["word_count"]

                if passage["retrieval_eligible"]:
                    eligible_passage_count += 1
                else:
                    short_passage_count += 1

    average_words = total_words / passage_count if passage_count else 0

    # ------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------

    print()
    print("Finished")
    print(f"  Sentences read       : {len(sentences)}")
    print(f"  Blocks processed     : {block_count}")
    print(f"  Blocks split         : {split_block_count}")
    print(f"  Passages created     : {passage_count}")
    print(f"  Retrieval eligible   : {eligible_passage_count}")
    print(f"  Very short passages  : {short_passage_count}")
    print(f"  Mean passage words   : {average_words:.1f}")
    print()
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
