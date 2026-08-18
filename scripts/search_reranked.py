#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from search_hybrid import (
    bm25_search,
    dense_search,
    fuse_results,
    load_documents,
    load_index,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------


def format_rank(rank: int | None) -> str:
    if rank is None:
        return "—"

    return str(rank)


# ----------------------------------------------------------------------
# Reranker loading
# ----------------------------------------------------------------------


def load_reranker(
    model_name: str,
    device: str,
):
    """
    Load bge-reranker-v2-m3 directly through Transformers.

    This deliberately does NOT use FlagEmbedding's reranker wrapper.
    """

    print(f"Loading reranker tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    print(f"Tokenizer: {type(tokenizer).__name__}")

    print(f"Loading reranker model: {model_name}")

    # Load normally first.
    #
    # This avoids depending on changing from_pretrained()
    # dtype / torch_dtype API details.
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device.startswith("cuda"):
        model = model.to(
            device=device,
            dtype=torch.float16,
        )
    else:
        model = model.to(
            device=device,
            dtype=torch.float32,
        )

    model.eval()

    return tokenizer, model


# ----------------------------------------------------------------------
# Reranking
# ----------------------------------------------------------------------


def rerank_candidates(
    claim: str,
    candidates: list[dict],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[dict]:
    """
    Rerank hybrid candidates using a cross-encoder.

    The claim and passage are presented jointly to the model.

    Both the raw reranker logit and sigmoid-normalized score are
    retained. Ordering by either is equivalent because sigmoid is
    monotonic.
    """

    if not candidates:
        return []

    results = []

    for batch_start in range(
        0,
        len(candidates),
        batch_size,
    ):
        batch_candidates = candidates[batch_start : batch_start + batch_size]

        pairs = [
            [
                claim,
                candidate["record"]["text_normalized"],
            ]
            for candidate in batch_candidates
        ]

        # Just for debugging, delete later
        lengths = tokenizer(
            pairs,
            truncation=False,
            padding=False,
        )["input_ids"]

        for candidate, ids in zip(batch_candidates, lengths):
            print(
                len(ids),
                candidate["record"]["passage_id"],
            )

        # This follows the input format documented by BAAI for
        # bge-reranker-v2-m3:
        #
        # [
        #     [query, passage],
        #     [query, passage],
        #     ...
        # ]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = (
                model(
                    **inputs,
                    return_dict=True,
                )
                .logits.view(-1)
                .float()
            )

        # BAAI documents sigmoid as the normalization that maps
        # reranker logits into [0, 1].
        normalized_scores = torch.sigmoid(logits)

        logits_cpu = logits.cpu().tolist()

        scores_cpu = normalized_scores.cpu().tolist()

        for (
            candidate,
            logit,
            score,
        ) in zip(
            batch_candidates,
            logits_cpu,
            scores_cpu,
        ):
            result = dict(candidate)

            result["reranker_logit"] = float(logit)

            result["reranker_score"] = float(score)

            results.append(result)

    results.sort(
        key=lambda result: result["reranker_logit"],
        reverse=True,
    )

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid dense + BM25 retrieval followed by BGE cross-encoder reranking."
        )
    )

    parser.add_argument(
        "claim",
        help="Thesis claim to search for",
    )

    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/embeddings/bge_m3_dense_filtered"),
        help=("Directory containing embeddings.npy, index.jsonl and manifest.json"),
    )

    parser.add_argument(
        "--bm25-database",
        type=Path,
        default=Path("data/lexical/bm25.sqlite"),
        help="SQLite FTS5/BM25 database",
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/extracted/documents.jsonl"),
    )

    parser.add_argument(
        "--reranker",
        default=("BAAI/bge-reranker-v2-m3"),
        help=("Transformers reranker model. Default: BAAI/bge-reranker-v2-m3"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=("Final reranked results displayed. Default: 10"),
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=30,
        help=("Number of hybrid candidates sent to the reranker. Default: 30"),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help=(
            "Number of candidates retrieved "
            "independently by dense and BM25. "
            "Default: 100"
        ),
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
        help=("RRF smoothing constant. Default: 60"),
    )

    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=8,
        help=("Cross-encoder inference batch size. Default: 8"),
    )

    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help=("Maximum query+passage token length for reranking. Default: 512"),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help=("Reranker device. Default: cuda:0"),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Show raw retrieval and reranker scores."),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Validate options.
    # --------------------------------------------------------------

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    if args.rerank_k <= 0:
        raise ValueError("--rerank-k must be > 0")

    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be > 0")

    if args.rerank_batch_size <= 0:
        raise ValueError("--rerank-batch-size must be > 0")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    # --------------------------------------------------------------
    # Resolve paths.
    # --------------------------------------------------------------

    embedding_dir = args.embedding_dir.resolve()

    database_path = args.bm25_database.resolve()

    documents_path = args.documents.resolve()

    # --------------------------------------------------------------
    # Load document metadata.
    # --------------------------------------------------------------

    documents = load_documents(documents_path)

    # --------------------------------------------------------------
    # Load dense embedding index.
    # --------------------------------------------------------------

    embeddings_path = embedding_dir / "embeddings.npy"

    index_path = embedding_dir / "index.jsonl"

    manifest_path = embedding_dir / "manifest.json"

    embeddings = np.load(embeddings_path)

    dense_index = load_index(index_path)

    if len(dense_index) != embeddings.shape[0]:
        raise RuntimeError("Dense index size does not match embedding matrix.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    embedding_model_name = manifest["model"]

    embedding_max_length = manifest.get(
        "max_length",
        512,
    )

    # --------------------------------------------------------------
    # Dense retrieval.
    # --------------------------------------------------------------

    print(f"Loading embedding model: {embedding_model_name}")

    embedding_model = BGEM3FlagModel(
        embedding_model_name,
        use_fp16=True,
    )

    dense_results = dense_search(
        claim=args.claim,
        model=embedding_model,
        embeddings=embeddings,
        index=dense_index,
        candidate_k=args.candidate_k,
        max_length=embedding_max_length,
    )

    # --------------------------------------------------------------
    # Release embedding model before loading reranker.
    # --------------------------------------------------------------

    del embedding_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # BM25 retrieval.
    # --------------------------------------------------------------

    bm25_results = bm25_search(
        claim=args.claim,
        database_path=database_path,
        candidate_k=args.candidate_k,
    )

    # --------------------------------------------------------------
    # Weighted RRF fusion.
    # --------------------------------------------------------------

    hybrid_results = fuse_results(
        dense_results=dense_results,
        bm25_results=bm25_results,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
    )

    # Record the hybrid position before reranking.
    for hybrid_rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        result["hybrid_rank"] = hybrid_rank

    rerank_candidates_list = hybrid_results[: args.rerank_k]

    # --------------------------------------------------------------
    # Load reranker directly through Transformers.
    # --------------------------------------------------------------

    tokenizer, reranker_model = load_reranker(
        model_name=args.reranker,
        device=args.device,
    )

    # --------------------------------------------------------------
    # Rerank hybrid candidates.
    # --------------------------------------------------------------

    reranked_results = rerank_candidates(
        claim=args.claim,
        candidates=(rerank_candidates_list),
        tokenizer=tokenizer,
        model=reranker_model,
        device=args.device,
        batch_size=(args.rerank_batch_size),
        max_length=(args.reranker_max_length),
    )

    final_results = reranked_results[: args.top_k]

    # --------------------------------------------------------------
    # Display.
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("CLAIM")
    print("=" * 80)
    print(args.claim)

    print()
    print("Retrieval configuration")
    print("-" * 80)

    print(f"Dense weight       : {args.dense_weight}")

    print(f"BM25 weight        : {args.bm25_weight}")

    print(f"Candidates/retriever: {args.candidate_k}")

    print(f"Hybrid candidates  : {len(hybrid_results)}")

    print(f"Reranked candidates: {len(rerank_candidates_list)}")

    print(f"Reranker            : {args.reranker}")

    print(f"Reranker device     : {args.device}")

    print(f"Reranker batch size : {args.rerank_batch_size}")

    for final_rank, result in enumerate(
        final_results,
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

        print(f"{final_rank}. reranker={result['reranker_score']:.4f}")

        print(f"Hybrid rank : {result['hybrid_rank']}")

        print(f"Dense rank  : {format_rank(result['dense_rank'])}")

        print(f"BM25 rank   : {format_rank(result['bm25_rank'])}")

        if args.debug:
            print(f"Reranker logit: {result['reranker_logit']:.4f}")

            print(f"Hybrid score  : {result['hybrid_score']:.6f}")

            if result["dense_score"] is not None:
                print(f"Dense score   : {result['dense_score']:.4f}")

            if result["bm25_score"] is not None:
                print(f"BM25 score    : {result['bm25_score']:.6f}")

        print(f"PDF         : {filename}")

        print(f"Page        : {record['page_number']}")

        print(f"Passage     : {record['passage_id']}")

        print("Sentences   : " + ", ".join(record["sentence_ids"]))

        print()
        print(record["text_normalized"])


if __name__ == "__main__":
    main()
