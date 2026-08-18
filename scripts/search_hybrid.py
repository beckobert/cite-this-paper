#!/usr/bin/env python3

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

# ----------------------------------------------------------------------
# General loading
# ----------------------------------------------------------------------


def load_index(
    path: Path,
) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


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


def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return vector

    return vector / norm


# ----------------------------------------------------------------------
# Dense retrieval
# ----------------------------------------------------------------------


def dense_search(
    claim: str,
    model,
    embeddings: np.ndarray,
    index: list[dict],
    candidate_k: int,
    max_length: int,
) -> list[dict]:
    output = model.encode(
        [claim],
        batch_size=1,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    query = normalize_vector(output["dense_vecs"][0])

    scores = embeddings @ query

    candidate_k = min(
        candidate_k,
        len(scores),
    )

    if candidate_k == len(scores):
        candidate_indices = np.argsort(scores)[::-1]

    else:
        candidate_indices = np.argpartition(
            scores,
            -candidate_k,
        )[-candidate_k:]

        candidate_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

    results = []

    for rank, embedding_index in enumerate(
        candidate_indices,
        start=1,
    ):
        record = index[int(embedding_index)]

        results.append(
            {
                "passage_id": record["passage_id"],
                "rank": rank,
                "score": float(scores[embedding_index]),
                "record": record,
            }
        )

    return results


# ----------------------------------------------------------------------
# BM25 retrieval
# ----------------------------------------------------------------------


def extract_query_terms(
    text: str,
) -> list[str]:
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
    if not terms:
        raise ValueError("No searchable lexical terms found.")

    return " OR ".join(f'"{term}"' for term in terms)


def bm25_search(
    claim: str,
    database_path: Path,
    candidate_k: int,
) -> list[dict]:
    terms = extract_query_terms(claim)

    fts_query = build_fts_query(terms)

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
                candidate_k,
            ),
        ).fetchall()

    finally:
        conn.close()

    results = []

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        results.append(
            {
                "passage_id": row["passage_id"],
                "rank": rank,
                # Keep native SQLite BM25 score for diagnostics.
                # Lower is better.
                "score": float(row["rank"]),
                "record": {
                    "passage_id": row["passage_id"],
                    "document_id": row["document_id"],
                    "page_index": row["page_index"],
                    "page_number": row["page_number"],
                    "logical_block_index": row["logical_block_index"],
                    "sentence_ids": json.loads(row["sentence_ids"]),
                    "content_type": row["content_type"],
                    "text": row["text"],
                    "text_normalized": row["text_normalized"],
                },
            }
        )

    return results


# ----------------------------------------------------------------------
# Weighted Reciprocal Rank Fusion
# ----------------------------------------------------------------------


def reciprocal_rank(
    rank: int,
    weight: float,
    rrf_k: float,
) -> float:
    return weight / (rrf_k + rank)


def fuse_results(
    dense_results: list[dict],
    bm25_results: list[dict],
    dense_weight: float,
    bm25_weight: float,
    rrf_k: float,
) -> list[dict]:

    combined = {}

    # --------------------------------------------------------------
    # Dense candidates
    # --------------------------------------------------------------

    for result in dense_results:
        passage_id = result["passage_id"]

        combined[passage_id] = {
            "passage_id": passage_id,
            "record": result["record"],
            "dense_rank": result["rank"],
            "dense_score": result["score"],
            "bm25_rank": None,
            "bm25_score": None,
            "hybrid_score": reciprocal_rank(
                rank=result["rank"],
                weight=dense_weight,
                rrf_k=rrf_k,
            ),
        }

    # --------------------------------------------------------------
    # BM25 candidates
    # --------------------------------------------------------------

    for result in bm25_results:
        passage_id = result["passage_id"]

        contribution = reciprocal_rank(
            rank=result["rank"],
            weight=bm25_weight,
            rrf_k=rrf_k,
        )

        if passage_id in combined:
            combined[passage_id]["bm25_rank"] = result["rank"]

            combined[passage_id]["bm25_score"] = result["score"]

            combined[passage_id]["hybrid_score"] += contribution

        else:
            combined[passage_id] = {
                "passage_id": passage_id,
                "record": result["record"],
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": result["rank"],
                "bm25_score": result["score"],
                "hybrid_score": contribution,
            }

    results = list(combined.values())

    results.sort(
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )

    return results


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def format_rank(
    rank: int | None,
) -> str:
    if rank is None:
        return "—"

    return str(rank)


def main():
    parser = argparse.ArgumentParser(
        description=("Hybrid evidence retrieval using BGE-M3 dense search and BM25.")
    )

    parser.add_argument(
        "claim",
        help="Thesis claim to search for",
    )

    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/embeddings/bge_m3_dense_filtered"),
    )

    parser.add_argument(
        "--bm25-database",
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
        help=("Number of final hybrid results. Default: 10"),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help=("Candidates retrieved independently by each search method. Default: 50"),
    )

    parser.add_argument(
        "--dense-weight",
        type=float,
        default=2.0,
        help=("Dense RRF weight. Default: 2.0"),
    )

    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help=("BM25 RRF weight. Default: 1.0"),
    )

    parser.add_argument(
        "--rrf-k",
        type=float,
        default=60.0,
        help=("Reciprocal-rank smoothing constant. Default: 60"),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Display additional retrieval diagnostics."),
    )

    args = parser.parse_args()

    embedding_dir = args.embedding_dir.resolve()

    database_path = args.bm25_database.resolve()

    documents = load_documents(args.documents.resolve())

    # --------------------------------------------------------------
    # Dense index
    # --------------------------------------------------------------

    embeddings = np.load(embedding_dir / "embeddings.npy")

    dense_index = load_index(embedding_dir / "index.jsonl")

    if len(dense_index) != embeddings.shape[0]:
        raise RuntimeError("Dense index size does not match embedding matrix.")

    manifest = json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))

    model_name = manifest["model"]

    max_length = manifest.get(
        "max_length",
        512,
    )

    print(f"Loading model: {model_name}")

    model = BGEM3FlagModel(
        model_name,
        use_fp16=True,
    )

    # --------------------------------------------------------------
    # Independent retrieval
    # --------------------------------------------------------------

    dense_results = dense_search(
        claim=args.claim,
        model=model,
        embeddings=embeddings,
        index=dense_index,
        candidate_k=args.candidate_k,
        max_length=max_length,
    )

    bm25_results = bm25_search(
        claim=args.claim,
        database_path=database_path,
        candidate_k=args.candidate_k,
    )

    # --------------------------------------------------------------
    # Fusion
    # --------------------------------------------------------------

    results = fuse_results(
        dense_results=dense_results,
        bm25_results=bm25_results,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
    )

    results = results[: args.top_k]

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("CLAIM")
    print("=" * 80)
    print(args.claim)

    print()
    print(f"Dense weight : {args.dense_weight}")
    print(f"BM25 weight  : {args.bm25_weight}")
    print(f"Candidate k  : {args.candidate_k}")

    for result_rank, result in enumerate(
        results,
        start=1,
    ):
        record = result["record"]

        document = documents.get(
            record["document_id"],
            {},
        )

        filename = document.get(
            "filename",
            record["document_id"],
        )

        print()
        print("=" * 80)

        print(f"{result_rank}. hybrid={result['hybrid_score']:.6f}")

        print(f"Dense rank : {format_rank(result['dense_rank'])}")

        print(f"BM25 rank  : {format_rank(result['bm25_rank'])}")

        if args.debug:
            if result["dense_score"] is not None:
                print(f"Dense score: {result['dense_score']:.4f}")

            if result["bm25_score"] is not None:
                print(f"BM25 score : {result['bm25_score']:.6f}")

        print(f"PDF        : {filename}")

        print(f"Page       : {record['page_number']}")

        print(f"Passage    : {record['passage_id']}")

        print("Sentences  : " + ", ".join(record["sentence_ids"]))

        print()
        print(record["text_normalized"])


if __name__ == "__main__":
    main()
