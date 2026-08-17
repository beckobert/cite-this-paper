#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

import pymupdf


def find_sentence(
    sentences_path: Path,
    sentence_id: str,
) -> tuple[dict, list[dict]]:
    """
    Find a sentence by ID.

    Also return all sentences from the same page so that surrounding
    context can optionally be printed.
    """
    target = None
    page_sentences = []

    with sentences_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            sentence = json.loads(line)

            if sentence["sentence_id"] == sentence_id:
                target = sentence

    if target is None:
        raise RuntimeError(f"Sentence ID not found: {sentence_id}")

    # Second pass: collect sentences from the same page.
    with sentences_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            sentence = json.loads(line)

            if (
                sentence["document_id"] == target["document_id"]
                and sentence["page_index"] == target["page_index"]
            ):
                page_sentences.append(sentence)

    page_sentences.sort(key=lambda s: s["page_sentence_index"])

    return target, page_sentences


def find_document(
    documents_path: Path,
    document_id: str,
) -> dict:
    """
    Find the document metadata record corresponding to document_id.
    """
    with documents_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            document = json.loads(line)

            if document["document_id"] == document_id:
                return document

    raise RuntimeError(f"Document ID not found: {document_id}")


def print_sentence_info(
    sentence: dict,
    document: dict,
    page_sentences: list[dict],
    context: int,
):
    """
    Print sentence metadata and optional surrounding context.
    """
    print()
    print("=" * 80)
    print("SENTENCE")
    print("=" * 80)

    print(f"Sentence ID : {sentence['sentence_id']}")
    print(f"Document    : {document['filename']}")
    print(f"Document ID : {sentence['document_id']}")
    print(f"PDF page    : {sentence['page_number']}")
    print(f"Block       : {sentence['block_no']}")
    print()

    print(sentence["text"])

    normalized = sentence.get("text_normalized")

    if normalized and normalized != sentence["text"]:
        print()
        print("Normalized:")
        print(normalized)

    if context <= 0:
        return

    target_index = sentence["page_sentence_index"]

    lower = max(0, target_index - context)
    upper = min(
        len(page_sentences),
        target_index + context + 1,
    )

    print()
    print("-" * 80)
    print("PAGE CONTEXT")
    print("-" * 80)

    for candidate in page_sentences[lower:upper]:
        marker = ">>>" if candidate["sentence_id"] == sentence["sentence_id"] else "   "

        print(f"{marker} s{candidate['page_sentence_index']:03d}: {candidate['text']}")

    print()


def render_sentence(
    pdf_path: Path,
    sentence: dict,
    output_path: Path,
    dpi: int,
):
    """
    Render the source PDF page with the sentence highlighted.
    """
    doc = pymupdf.open(pdf_path)

    try:
        page_index = sentence["page_index"]

        if page_index < 0 or page_index >= len(doc):
            raise RuntimeError(
                f"Invalid page index {page_index} for PDF with {len(doc)} pages"
            )

        page = doc[page_index]

        # Each box corresponds to one physical line occupied
        # by the sentence.
        for box_record in sentence["boxes"]:
            rect = pymupdf.Rect(box_record["bbox"])

            annotation = page.add_highlight_annot(rect)

            if annotation is not None:
                annotation.set_opacity(0.35)
                annotation.update()

        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
            annots=True,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pixmap.save(output_path)

    finally:
        doc.close()


def safe_filename(sentence_id: str) -> str:
    """
    Convert a sentence ID into a convenient filename.
    """
    return sentence_id.replace(":", "_").replace("/", "_")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render a sentence from sentences.jsonl at its original PDF location."
        )
    )

    parser.add_argument(
        "sentence_id",
        help="Sentence ID to display",
    )

    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path("data/sentences/sentences.jsonl"),
        help="Path to sentences.jsonl",
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/extracted/documents.jsonl"),
        help="Path to documents.jsonl",
    )

    parser.add_argument(
        "--papers",
        type=Path,
        default=Path("papers"),
        help="Root directory containing the PDFs",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/review/sentences"),
        help="Directory for rendered images",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering resolution. Default: 150 DPI",
    )

    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help=("Number of surrounding sentences to show in the terminal. Default: 2"),
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the rendered image using xdg-open",
    )

    args = parser.parse_args()

    sentences_path = args.sentences.resolve()
    documents_path = args.documents.resolve()
    papers_dir = args.papers.resolve()
    output_dir = args.output_dir.resolve()

    sentence, page_sentences = find_sentence(
        sentences_path=sentences_path,
        sentence_id=args.sentence_id,
    )

    document = find_document(
        documents_path=documents_path,
        document_id=sentence["document_id"],
    )

    pdf_path = (papers_dir / document["relative_path"]).resolve()

    if not pdf_path.exists():
        raise RuntimeError(f"PDF does not exist: {pdf_path}")

    print_sentence_info(
        sentence=sentence,
        document=document,
        page_sentences=page_sentences,
        context=args.context,
    )

    output_filename = safe_filename(args.sentence_id) + ".png"

    output_path = output_dir / output_filename

    render_sentence(
        pdf_path=pdf_path,
        sentence=sentence,
        output_path=output_path,
        dpi=args.dpi,
    )

    print(f"PDF          : {pdf_path}")
    print(f"Rendered page: {output_path}")
    print()

    if args.open:
        subprocess.Popen(
            [
                "xdg-open",
                str(output_path),
            ]
        )


if __name__ == "__main__":
    main()
