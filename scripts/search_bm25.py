#!/usr/bin/env python3

import argparse
import json
import re
import sqlite3
from pathlib import Path


def load_documents(
    path: Path,
) -> dict[str, dict]:
    documents = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            document = json.loads(line)

            documents[document["document_id"]] = document

    return documents


def extract_query_terms(
    text: str,
) -> list[str]:
    """
    Convert arbitrary claim text into safe lexical query terms.

    This deliberately mirrors the broad behavior of SQLite's
    unicode61 tokenizer:

        - letters and numbers are retained
        - punctuation separates tokens

    Terms are deduplicated while preserving their original order.
    """

    # [^\\W_] means:
    #     a Unicode "word" character,
    #     excluding underscore.
    #
    # This captures letters and digits without interpreting
    # punctuation as FTS5 query syntax.
    terms = re.findall(
        r"[^\W_]+",
        text.casefold(),
        flags=re.UNICODE,
    )

    seen = set()
    unique_terms = []

    for term in terms:
        if term in seen:
            continue

        seen.add(term)
        unique_terms.append(term)

    return unique_terms


def build_fts_query(
    terms: list[str],
) -> str:
    """
    Build a safe FTS5 OR query.

    Each token is quoted so that words such as AND, OR or NOT
    are interpreted as terms rather than FTS5 operators.
    """
    if not terms:
        raise ValueError("No searchable terms found in claim.")

    return " OR ".join(f'"{term}"' for term in terms)


def main():
    parser = argparse.ArgumentParser(
        description=("Search passages using SQLite FTS5/BM25.")
    )

    parser.add_argument(
        "claim",
        help="Thesis claim to search for",
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/lexical/bm25.sqlite"),
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/extracted/documents.jsonl"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show generated FTS query",
    )

    args = parser.parse_args()

    database_path = args.database.resolve()

    documents = load_documents(args.documents.resolve())

    terms = extract_query_terms(args.claim)

    fts_query = build_fts_query(terms)

    if args.debug:
        print("Query terms:")
        print(terms)
        print()
        print("FTS5 query:")
        print(fts_query)
        print()

    conn = sqlite3.connect(database_path)

    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT
                passage_id,
                document_id,
                page_index,
                page_number,
                logical_block_index,
                sentence_ids,
                content_type,
                text,
                text_normalized,
                rank
            FROM passages_fts
            WHERE passages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (
                fts_query,
                args.top_k,
            ),
        ).fetchall()

    finally:
        conn.close()

    print()
    print("=" * 80)
    print("CLAIM")
    print("=" * 80)
    print(args.claim)

    if not rows:
        print()
        print("No lexical matches.")
        return

    for result_rank, row in enumerate(
        rows,
        start=1,
    ):
        document = documents.get(
            row["document_id"],
            {},
        )

        filename = document.get(
            "filename",
            row["document_id"],
        )

        # SQLite's native BM25 rank is numerically LOWER
        # for better results. Negating it is easier to read,
        # so our displayed lexical_score is HIGHER-is-better.
        lexical_score = -float(row["rank"])

        sentence_ids = json.loads(row["sentence_ids"])

        print()
        print("=" * 80)
        print(f"{result_rank}. lexical_score={lexical_score:.6f}")
        print(f"PDF      : {filename}")
        print(f"Page     : {row['page_number']}")
        print(f"Type     : {row['content_type']}")
        print(f"Passage  : {row['passage_id']}")
        print("Sentences: " + ", ".join(sentence_ids))
        print()
        print(row["text_normalized"])


if __name__ == "__main__":
    main()
