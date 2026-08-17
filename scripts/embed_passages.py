#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel


def load_passages(path: Path) -> list[dict]:
    passages = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                passage = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}") from exc

            # Keep short/noisy passages in passages.jsonl,
            # but do not put them into the retrieval index.
            if not passage.get(
                "retrieval_eligible",
                True,
            ):
                continue

            passages.append(passage)

    return passages


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embeddings so that inner product equals
    cosine similarity.
    """
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return embeddings / norms


def main():
    parser = argparse.ArgumentParser(
        description=("Create dense BGE-M3 embeddings for retrieval passages.")
    )

    parser.add_argument(
        "passages_jsonl",
        type=Path,
        help="passages.jsonl",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for embedding files",
    )

    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Hugging Face model name",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size. Default: 32",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help=("Maximum tokenizer length. Default: 512"),
    )

    args = parser.parse_args()

    passages_path = args.passages_jsonl.resolve()

    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading passages...")

    passages = load_passages(passages_path)

    if not passages:
        raise RuntimeError("No retrieval-eligible passages found.")

    texts = [passage["text_normalized"] for passage in passages]

    print(f"Passages to embed: {len(passages)}")

    print(f"Loading model: {args.model}")

    model = BGEM3FlagModel(
        args.model,
        use_fp16=True,
    )

    print("Encoding...")

    output = model.encode(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    embeddings = normalize_embeddings(output["dense_vecs"])

    print(f"Embedding matrix: {embeddings.shape}")

    # --------------------------------------------------------
    # Save matrix
    # --------------------------------------------------------

    embeddings_path = output_dir / "embeddings.npy"

    np.save(
        embeddings_path,
        embeddings,
    )

    # --------------------------------------------------------
    # Save row -> passage mapping.
    #
    # We deliberately include enough metadata here that the
    # search script does not need to scan passages.jsonl.
    # --------------------------------------------------------

    index_path = output_dir / "index.jsonl"

    with index_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for embedding_index, passage in enumerate(passages):
            record = {
                "embedding_index": embedding_index,
                "passage_id": passage["passage_id"],
                "document_id": passage["document_id"],
                "page_index": passage["page_index"],
                "page_number": passage["page_number"],
                "logical_block_index": passage.get("logical_block_index"),
                "sentence_ids": passage["sentence_ids"],
                "text": passage["text"],
                "text_normalized": passage["text_normalized"],
                "word_count": passage["word_count"],
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "model": args.model,
        "embedding_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "source_passages": str(passages_path),
    }

    manifest_path = output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Finished")
    print(f"  Passages embedded : {len(passages)}")
    print(f"  Vector dimension  : {embeddings.shape[1]}")
    print(f"  Matrix size       : {embeddings.nbytes / 1024**2:.1f} MiB")
    print()
    print(f"Embeddings: {embeddings_path}")
    print(f"Index:      {index_path}")
    print(f"Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
