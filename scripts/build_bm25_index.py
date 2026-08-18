#!/usr/bin/env python3

import argparse
import json
import sqlite3
from pathlib import Path

INDEX_SCHEMA_VERSION = 1


def load_passages(path: Path):
    """
    Yield retrieval-eligible passages from classified passages.jsonl.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                passage = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}") from exc

            if not passage.get(
                "retrieval_eligible",
                True,
            ):
                continue

            yield passage


def create_database(
    database_path: Path,
):
    """
    Create a fresh SQLite FTS5 database.
    """
    if database_path.exists():
        database_path.unlink()

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(database_path)

    # Metadata about how this index was built.
    conn.execute(
        """
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    # Only text_normalized is indexed.
    #
    # The other columns are stored alongside it so search results
    # are self-contained, but UNINDEXED means they do not contribute
    # to full-text search.
    conn.execute(
        """
        CREATE VIRTUAL TABLE passages_fts USING fts5(
            passage_id UNINDEXED,
            document_id UNINDEXED,
            page_index UNINDEXED,
            page_number UNINDEXED,
            logical_block_index UNINDEXED,
            sentence_ids UNINDEXED,
            content_type UNINDEXED,
            text UNINDEXED,
            text_normalized,
            tokenize = 'unicode61 remove_diacritics 1'
        )
        """
    )

    return conn


def main():
    parser = argparse.ArgumentParser(
        description=("Build a SQLite FTS5/BM25 lexical index from classified passages.")
    )

    parser.add_argument(
        "passages_jsonl",
        type=Path,
        help="Classified passages.jsonl",
    )

    parser.add_argument(
        "database",
        type=Path,
        help="Output SQLite database",
    )

    args = parser.parse_args()

    passages_path = args.passages_jsonl.resolve()

    database_path = args.database.resolve()

    print(f"Reading passages from: {passages_path}")

    conn = create_database(database_path)

    inserted = 0

    try:
        with conn:
            for passage in load_passages(passages_path):
                conn.execute(
                    """
                    INSERT INTO passages_fts (
                        passage_id,
                        document_id,
                        page_index,
                        page_number,
                        logical_block_index,
                        sentence_ids,
                        content_type,
                        text,
                        text_normalized
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        passage["passage_id"],
                        passage["document_id"],
                        passage["page_index"],
                        passage["page_number"],
                        passage.get("logical_block_index"),
                        json.dumps(passage["sentence_ids"]),
                        passage.get(
                            "content_type",
                            "body",
                        ),
                        passage["text"],
                        passage["text_normalized"],
                    ),
                )

                inserted += 1

            metadata = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "source_passages": str(passages_path),
                "passage_count": inserted,
                "tokenizer": "unicode61 remove_diacritics 1",
                "ranking": "SQLite FTS5 BM25",
            }

            for key, value in metadata.items():
                conn.execute(
                    """
                    INSERT INTO index_meta (
                        key,
                        value
                    )
                    VALUES (?, ?)
                    """,
                    (
                        key,
                        json.dumps(value),
                    ),
                )

        # Compact/optimize the FTS index after the bulk insert.
        conn.execute(
            """
            INSERT INTO passages_fts(
                passages_fts
            )
            VALUES('optimize')
            """
        )

        conn.commit()

    finally:
        conn.close()

    size_mb = database_path.stat().st_size / 1024**2

    print()
    print("Finished")
    print(f"  Passages indexed : {inserted}")
    print(f"  Database size    : {size_mb:.1f} MiB")
    print()
    print(f"Database: {database_path}")


if __name__ == "__main__":
    main()
