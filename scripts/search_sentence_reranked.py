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
# General utilities
# ----------------------------------------------------------------------


def format_rank(rank: int | None) -> str:
    if rank is None:
        return "—"

    return str(rank)


def load_sentences(
    path: Path,
) -> dict[str, dict]:
    """
    Load sentence records keyed by sentence_id.
    """
    sentences = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                sentence = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc

            sentence_id = sentence["sentence_id"]

            sentences[sentence_id] = sentence

    return sentences


# ----------------------------------------------------------------------
# Sentence candidate construction
# ----------------------------------------------------------------------


def sentence_text(
    sentence: dict,
) -> str:
    """
    Prefer normalized text for model input.
    """
    return sentence.get("text_normalized") or sentence.get(
        "text",
        "",
    )


def build_context_text(
    target_sentence_id: str,
    parent_passage: dict,
    sentences: dict[str, dict],
    context_window: int,
) -> str:
    """
    Build the text given to the reranker.

    context_window = 0:
        target sentence only

    context_window = 1:
        previous + target + next sentence

    Context never leaves the parent passage.
    """

    passage_sentence_ids = parent_passage["sentence_ids"]

    try:
        target_index = passage_sentence_ids.index(target_sentence_id)

    except ValueError:
        # Should not happen, but fall back safely.
        sentence = sentences[target_sentence_id]

        return sentence_text(sentence)

    start = max(
        0,
        target_index - context_window,
    )

    end = min(
        len(passage_sentence_ids),
        target_index + context_window + 1,
    )

    selected_ids = passage_sentence_ids[start:end]

    parts = []

    for sentence_id in selected_ids:
        sentence = sentences.get(sentence_id)

        if sentence is None:
            continue

        text = sentence_text(sentence).strip()

        if text:
            parts.append(text)

    return " ".join(parts)


def collect_sentence_candidates(
    hybrid_results: list[dict],
    sentences: dict[str, dict],
    passage_k: int,
    context_window: int,
    debug: bool = False,
) -> list[dict]:
    """
    Expand the top hybrid passages into unique sentence candidates.

    If the same sentence occurs in overlapping passages, keep the
    occurrence from the highest-ranked parent passage.
    """

    candidates_by_id = {}

    selected_passages = hybrid_results[:passage_k]

    for hybrid_rank, result in enumerate(
        selected_passages,
        start=1,
    ):
        passage = result["record"]

        for sentence_id in passage["sentence_ids"]:
            if sentence_id in candidates_by_id:
                continue

            sentence = sentences.get(sentence_id)

            if sentence is None:
                if debug:
                    print(f"WARNING: sentence not found: {sentence_id}")

                continue

            target_text = sentence_text(sentence).strip()

            if not target_text:
                continue

            rerank_text = build_context_text(
                target_sentence_id=sentence_id,
                parent_passage=passage,
                sentences=sentences,
                context_window=context_window,
            )

            candidates_by_id[sentence_id] = {
                "sentence_id": sentence_id,
                "sentence": sentence,
                "target_text": target_text,
                "rerank_text": rerank_text,
                "parent_passage": passage,
                "parent_hybrid_rank": hybrid_rank,
                "parent_hybrid_score": result["hybrid_score"],
                "parent_dense_rank": result["dense_rank"],
                "parent_dense_score": result["dense_score"],
                "parent_bm25_rank": result["bm25_rank"],
                "parent_bm25_score": result["bm25_score"],
            }

    return list(candidates_by_id.values())


# ----------------------------------------------------------------------
# Reranker
# ----------------------------------------------------------------------


def load_reranker(
    model_name: str,
    device: str,
):
    """
    Load the reranker directly through Transformers.

    FlagEmbedding is deliberately not used for reranking.
    """

    print(f"Loading reranker tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    print(f"Tokenizer: {type(tokenizer).__name__}")

    print(f"Loading reranker model: {model_name}")

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


def rerank_sentences(
    claim: str,
    candidates: list[dict],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[dict]:
    """
    Rerank sentence candidates against the claim.
    """

    if not candidates:
        return []

    results = []

    for batch_start in range(
        0,
        len(candidates),
        batch_size,
    ):
        batch = candidates[batch_start : batch_start + batch_size]

        queries = [claim for _ in batch]

        texts = [candidate["rerank_text"] for candidate in batch]

        inputs = tokenizer(
            queries,
            texts,
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

        scores = torch.sigmoid(logits)

        logits_cpu = logits.cpu().tolist()

        scores_cpu = scores.cpu().tolist()

        for (
            candidate,
            logit,
            score,
        ) in zip(
            batch,
            logits_cpu,
            scores_cpu,
        ):
            result = dict(candidate)

            result["reranker_logit"] = float(logit)

            result["reranker_score"] = float(score)

            results.append(result)

    # Sigmoid is monotonic, so ordering by raw logit or
    # normalized score is equivalent.
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
            "Retrieve passages using dense + BM25 hybrid "
            "search, then rerank their constituent sentences."
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
        "--sentences",
        type=Path,
        default=Path("data/sentences/sentences.jsonl"),
    )

    parser.add_argument(
        "--reranker",
        default=("BAAI/bge-reranker-v2-m3"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=("Number of final sentence results displayed. Default: 10"),
    )

    parser.add_argument(
        "--passage-k",
        type=int,
        default=30,
        help=(
            "Number of top hybrid passages whose "
            "sentences become reranking candidates. "
            "Default: 30"
        ),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help=(
            "Candidates retrieved independently by dense and BM25 search. Default: 100"
        ),
    )

    parser.add_argument(
        "--dense-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--rrf-k",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help=(
            "Number of neighboring sentences on "
            "each side supplied to the reranker. "
            "0 = sentence only. Default: 0"
        ),
    )

    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    if args.context_window < 0:
        raise ValueError("--context-window must be >= 0")

    if args.passage_k <= 0:
        raise ValueError("--passage-k must be > 0")

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    embedding_dir = args.embedding_dir.resolve()

    # --------------------------------------------------------------
    # Metadata.
    # --------------------------------------------------------------

    documents = load_documents(args.documents.resolve())

    sentences = load_sentences(args.sentences.resolve())

    # --------------------------------------------------------------
    # Dense index.
    # --------------------------------------------------------------

    embeddings = np.load(embedding_dir / "embeddings.npy")

    dense_index = load_index(embedding_dir / "index.jsonl")

    if len(dense_index) != embeddings.shape[0]:
        raise RuntimeError("Dense index size does not match embedding matrix.")

    manifest = json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))

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
        max_length=(embedding_max_length),
    )

    # Dense model no longer needed.
    del embedding_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # BM25 retrieval.
    # --------------------------------------------------------------

    bm25_results = bm25_search(
        claim=args.claim,
        database_path=(args.bm25_database.resolve()),
        candidate_k=args.candidate_k,
    )

    # --------------------------------------------------------------
    # Hybrid passage retrieval.
    # --------------------------------------------------------------

    hybrid_results = fuse_results(
        dense_results=dense_results,
        bm25_results=bm25_results,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
    )

    # --------------------------------------------------------------
    # Expand top passages into sentences.
    # --------------------------------------------------------------

    sentence_candidates = collect_sentence_candidates(
        hybrid_results=hybrid_results,
        sentences=sentences,
        passage_k=args.passage_k,
        context_window=(args.context_window),
        debug=args.debug,
    )

    print(f"Hybrid passages selected : {min(args.passage_k, len(hybrid_results))}")

    print(f"Unique sentence candidates: {len(sentence_candidates)}")

    print(f"Context window            : {args.context_window}")

    if not sentence_candidates:
        print("No sentence candidates found.")
        return

    # --------------------------------------------------------------
    # Sentence reranker.
    # --------------------------------------------------------------

    tokenizer, reranker_model = load_reranker(
        model_name=args.reranker,
        device=args.device,
    )

    results = rerank_sentences(
        claim=args.claim,
        candidates=sentence_candidates,
        tokenizer=tokenizer,
        model=reranker_model,
        device=args.device,
        batch_size=(args.rerank_batch_size),
        max_length=(args.reranker_max_length),
    )

    final_results = results[: args.top_k]

    # --------------------------------------------------------------
    # Output.
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("CLAIM")
    print("=" * 80)
    print(args.claim)

    print()
    print(f"Dense weight   : {args.dense_weight}")

    print(f"BM25 weight    : {args.bm25_weight}")

    print(f"Passages used  : {args.passage_k}")

    print(f"Sentence candidates: {len(sentence_candidates)}")

    print(f"Context window : {args.context_window}")

    for final_rank, result in enumerate(
        final_results,
        start=1,
    ):
        sentence = result["sentence"]

        passage = result["parent_passage"]

        document_id = sentence.get(
            "document_id",
            passage["document_id"],
        )

        document = documents.get(
            document_id,
            {},
        )

        filename = document.get(
            "filename",
            document_id,
        )

        page_number = sentence.get(
            "page_number",
            passage["page_number"],
        )

        print()
        print("=" * 80)

        print(f"{final_rank}. sentence_score={result['reranker_score']:.4f}")

        print(f"Parent hybrid rank : {result['parent_hybrid_rank']}")

        print(f"Parent dense rank  : {format_rank(result['parent_dense_rank'])}")

        print(f"Parent BM25 rank   : {format_rank(result['parent_bm25_rank'])}")

        print(f"PDF                : {filename}")

        print(f"Page               : {page_number}")

        print(f"Sentence           : {result['sentence_id']}")

        print(f"Parent passage     : {passage['passage_id']}")

        if args.debug:
            print(f"Reranker logit     : {result['reranker_logit']:.4f}")

            print(f"Parent hybrid score: {result['parent_hybrid_score']:.6f}")

        print()
        print("TARGET SENTENCE")
        print(result["target_text"])

        if args.context_window > 0:
            print()
            print("RERANKER CONTEXT")
            print(result["rerank_text"])


if __name__ == "__main__":
    main()
