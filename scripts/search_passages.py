#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel


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

    if not path.exists():
        return documents

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


def main():
    parser = argparse.ArgumentParser(
        description=("Search passage embeddings using a thesis claim.")
    )

    parser.add_argument(
        "claim",
        help="Claim to search for",
    )

    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/embeddings/bge_m3_dense_filtered"),
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

    args = parser.parse_args()

    embedding_dir = args.embedding_dir.resolve()

    embeddings = np.load(embedding_dir / "embeddings.npy")

    index = load_index(embedding_dir / "index.jsonl")

    if len(index) != embeddings.shape[0]:
        raise RuntimeError("Index size does not match embedding matrix.")

    documents = load_documents(args.documents.resolve())

    manifest = json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))

    model_name = manifest["model"]

    print(f"Loading model: {model_name}")

    model = BGEM3FlagModel(
        model_name,
        use_fp16=True,
    )

    output = model.encode(
        [args.claim],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    query = normalize_vector(output["dense_vecs"][0])

    # Because both passage and query vectors are normalized,
    # this inner product is cosine similarity.
    scores = embeddings @ query

    top_k = min(
        args.top_k,
        len(scores),
    )

    # Efficient top-k selection without sorting the entire corpus.
    candidate_indices = np.argpartition(
        scores,
        -top_k,
    )[-top_k:]

    candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    print()
    print("=" * 80)
    print("CLAIM")
    print("=" * 80)
    print(args.claim)

    for rank, embedding_index in enumerate(
        candidate_indices,
        start=1,
    ):
        record = index[int(embedding_index)]

        score = float(scores[embedding_index])

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
        print(f"{rank}. score={score:.4f}")
        print(f"PDF      : {filename}")
        print(f"Page     : {record['page_number']}")
        print(f"Passage  : {record['passage_id']}")
        print("Sentences: " + ", ".join(record["sentence_ids"]))
        print()
        print(record["text_normalized"])


if __name__ == "__main__":
    main()
