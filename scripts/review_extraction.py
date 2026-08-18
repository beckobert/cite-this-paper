#!/usr/bin/env python3

import argparse
import html
from pathlib import Path

import pymupdf

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>

<style>
body {{
    font-family: sans-serif;
    margin: 20px;
}}

.metadata {{
    margin-bottom: 20px;
    padding: 12px;
    background: #eee;
}}

.container {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    align-items: start;
}}

.panel {{
    min-width: 0;
}}

.panel h2 {{
    margin-top: 0;
}}

.page-image {{
    width: 100%;
    border: 1px solid #999;
}}

.text {{
    white-space: pre-wrap;
    font-family: monospace;
    font-size: 13px;
    line-height: 1.4;
    border: 1px solid #ccc;
    padding: 12px;
    max-height: 90vh;
    overflow: auto;
}}
</style>
</head>

<body>

<h1>{title}</h1>

<div class="metadata">
    <b>PDF:</b> {pdf_name}<br>
    <b>Page:</b> {page_number} / {page_count}<br>
    <b>Words:</b> {word_count}<br>
    <b>Blocks:</b> {block_count}
</div>

<div class="container">

<div class="panel">
<h2>Original page</h2>
<img class="page-image" src="{image_name}">
</div>

<div class="panel">
<h2>Native extraction</h2>
<div class="text">{native_text}</div>
</div>

<div class="panel">
<h2>Sorted extraction</h2>
<div class="text">{sorted_text}</div>
</div>

</div>

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pdf",
        type=Path,
        help="PDF to inspect",
    )

    parser.add_argument(
        "page",
        type=int,
        help="Human-readable page number (starts at 1)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/review"),
    )

    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)

    page_index = args.page - 1

    if page_index < 0 or page_index >= len(doc):
        raise SystemExit(f"Page must be between 1 and {len(doc)}")

    page = doc[page_index]

    # Extract text.
    native_text = page.get_text("text", sort=False)
    sorted_text = page.get_text("text", sort=True)

    words = page.get_text("words", sort=False)
    blocks = page.get_text("blocks", sort=False)

    # Render page at approximately 150 DPI.
    zoom = 150 / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    stem = pdf_path.stem
    base_name = f"{stem}_page_{args.page:04d}"

    image_path = output_dir / f"{base_name}.png"
    html_path = output_dir / f"{base_name}.html"

    pixmap.save(image_path)

    report = HTML_TEMPLATE.format(
        title=f"{pdf_path.name} — page {args.page}",
        pdf_name=html.escape(pdf_path.name),
        page_number=args.page,
        page_count=len(doc),
        word_count=len(words),
        block_count=len(blocks),
        image_name=image_path.name,
        native_text=html.escape(native_text),
        sorted_text=html.escape(sorted_text),
    )

    html_path.write_text(report, encoding="utf-8")

    doc.close()

    print(html_path)


if __name__ == "__main__":
    main()
