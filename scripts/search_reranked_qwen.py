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
    AutoModelForCausalLM,
    AutoTokenizer,
)

DEFAULT_RERANKER = "Qwen/Qwen3-Reranker-4B"

DEFAULT_INSTRUCTION = (
    "Given a scientific claim, determine whether the document "
    "provides evidence that supports that claim. Prefer passages "
    "containing explicit scientific statements, experimental or "
    "computational results, quantitative findings, methodological "
    "details, or conclusions that substantiate the claim. "
    "Do not rank a document highly merely because it discusses "
    "the same topic. Well cited introductions can be valuable to"
    "some degree as well."
)


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------


def format_rank(
    rank: int | None,
) -> str:
    if rank is None:
        return "—"

    return str(rank)


# ----------------------------------------------------------------------
# Qwen reranker
# ----------------------------------------------------------------------


def format_instruction(
    instruction: str,
    query: str,
    document: str,
) -> str:
    """
    Format one query/document pair for Qwen3-Reranker.
    """

    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def load_qwen_reranker(
    model_name: str,
    device: str,
):
    """
    Load Qwen3-Reranker directly through Transformers.

    The model scores each query/document pair by comparing the
    logits of the tokens "yes" and "no".
    """

    print(f"Loading reranker tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )

    print(f"Tokenizer: {type(tokenizer).__name__}")

    print(f"Loading reranker model: {model_name}")

    if device.startswith("cuda"):
        dtype = torch.float16

    else:
        dtype = torch.float32

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )

    token_false_id = tokenizer.convert_tokens_to_ids("no")

    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    if token_false_id is None or token_true_id is None:
        raise RuntimeError("Could not resolve Qwen reranker 'yes'/'no' token IDs.")

    return (
        tokenizer,
        model,
        token_false_id,
        token_true_id,
    )


def build_qwen_prompt_tokens(
    tokenizer,
):
    """
    Build the fixed prompt prefix and suffix used by
    Qwen3-Reranker.
    """

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. "
        "Note that the answer can only be "
        '"yes" or "no".'
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prefix_tokens = tokenizer.encode(
        prefix,
        add_special_tokens=False,
    )

    suffix_tokens = tokenizer.encode(
        suffix,
        add_special_tokens=False,
    )

    return (
        prefix_tokens,
        suffix_tokens,
    )


def prepare_qwen_batch(
    tokenizer,
    formatted_pairs: list[str],
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    max_length: int,
    device: str,
):
    """
    Tokenize and format a batch for Qwen3-Reranker.
    """

    available_length = max_length - len(prefix_tokens) - len(suffix_tokens)

    if available_length <= 0:
        raise ValueError(
            "--reranker-max-length is too small for the Qwen prompt wrapper."
        )

    encoded = tokenizer(
        formatted_pairs,
        padding=False,
        truncation=True,
        return_attention_mask=False,
        max_length=available_length,
    )

    for i, token_ids in enumerate(encoded["input_ids"]):
        encoded["input_ids"][i] = prefix_tokens + token_ids + suffix_tokens

    inputs = tokenizer.pad(
        encoded,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    return inputs


def rerank_candidates(
    claim: str,
    candidates: list[dict],
    tokenizer,
    model,
    token_false_id: int,
    token_true_id: int,
    instruction: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[dict]:
    """
    Rerank complete passage candidates with Qwen3-Reranker.
    """

    if not candidates:
        return []

    (
        prefix_tokens,
        suffix_tokens,
    ) = build_qwen_prompt_tokens(tokenizer)

    results = []

    for batch_start in range(
        0,
        len(candidates),
        batch_size,
    ):
        batch_candidates = candidates[batch_start : batch_start + batch_size]

        formatted_pairs = []

        for candidate in batch_candidates:
            record = candidate["record"]

            passage_text = record.get("text_normalized") or record.get(
                "text",
                "",
            )

            formatted_pairs.append(
                format_instruction(
                    instruction=instruction,
                    query=claim,
                    document=passage_text,
                )
            )

        inputs = prepare_qwen_batch(
            tokenizer=tokenizer,
            formatted_pairs=formatted_pairs,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            max_length=max_length,
            device=device,
        )

        with torch.inference_mode():
            final_logits = model(**inputs).logits[:, -1, :].float()

            true_logits = final_logits[:, token_true_id]

            false_logits = final_logits[:, token_false_id]

            yes_no_logits = torch.stack(
                [
                    false_logits,
                    true_logits,
                ],
                dim=1,
            )

            probabilities = torch.nn.functional.softmax(
                yes_no_logits,
                dim=1,
            )[:, 1]

            # Difference between yes and no logits.
            # Useful as an unbounded diagnostic score.
            logit_differences = true_logits - false_logits

        probabilities = probabilities.cpu().tolist()

        logit_differences = logit_differences.cpu().tolist()

        for (
            candidate,
            score,
            logit_difference,
        ) in zip(
            batch_candidates,
            probabilities,
            logit_differences,
        ):
            result = dict(candidate)

            result["reranker_score"] = float(score)

            result["reranker_logit"] = float(logit_difference)

            results.append(result)

    results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True,
    )

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid dense + BM25 retrieval followed by Qwen3 passage reranking."
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
        "--reranker",
        default=DEFAULT_RERANKER,
        help=("Qwen reranker model. Default: Qwen/Qwen3-Reranker-4B"),
    )

    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help=("Task-specific instruction supplied to Qwen3-Reranker."),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=("Final reranked passages displayed. Default: 10"),
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=30,
        help=("Number of hybrid passages sent to Qwen. Default: 30"),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help=("Candidates retrieved independently by dense and BM25. Default: 100"),
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
        help=("Qwen reranking batch size. Default: 8"),
    )

    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=1024,
        help=("Maximum Qwen prompt length. Default: 1024"),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help=("Reranker device. Default: cuda:0"),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Display retrieval and reranker diagnostics."),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Validate.
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
        raise RuntimeError("CUDA requested but unavailable.")

    # --------------------------------------------------------------
    # Paths and metadata.
    # --------------------------------------------------------------

    embedding_dir = args.embedding_dir.resolve()

    documents = load_documents(args.documents.resolve())

    # --------------------------------------------------------------
    # Load dense index.
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

    # BGE-M3 is no longer needed.
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
    # Weighted RRF.
    # --------------------------------------------------------------

    hybrid_results = fuse_results(
        dense_results=dense_results,
        bm25_results=bm25_results,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
    )

    for hybrid_rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        result["hybrid_rank"] = hybrid_rank

    rerank_candidates_list = hybrid_results[: args.rerank_k]

    # --------------------------------------------------------------
    # Load Qwen.
    # --------------------------------------------------------------

    (
        tokenizer,
        reranker_model,
        token_false_id,
        token_true_id,
    ) = load_qwen_reranker(
        model_name=args.reranker,
        device=args.device,
    )

    # --------------------------------------------------------------
    # Passage reranking.
    # --------------------------------------------------------------

    reranked_results = rerank_candidates(
        claim=args.claim,
        candidates=(rerank_candidates_list),
        tokenizer=tokenizer,
        model=reranker_model,
        token_false_id=(token_false_id),
        token_true_id=(token_true_id),
        instruction=(args.instruction),
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
    print(f"Dense weight        : {args.dense_weight}")

    print(f"BM25 weight         : {args.bm25_weight}")

    print(f"Candidates/retriever: {args.candidate_k}")

    print(f"Passages reranked   : {len(rerank_candidates_list)}")

    print(f"Reranker            : {args.reranker}")

    if args.debug:
        print()
        print("RERANKER INSTRUCTION")
        print("-" * 80)
        print(args.instruction)

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

        print(f"{final_rank}. Qwen score={result['reranker_score']:.4f}")

        print(f"Hybrid rank : {result['hybrid_rank']}")

        print(f"Dense rank  : {format_rank(result['dense_rank'])}")

        print(f"BM25 rank   : {format_rank(result['bm25_rank'])}")

        if args.debug:
            print(f"Qwen logit  : {result['reranker_logit']:.4f}")

            print(f"Hybrid score: {result['hybrid_score']:.6f}")

            if result["dense_score"] is not None:
                print(f"Dense score : {result['dense_score']:.4f}")

            if result["bm25_score"] is not None:
                print(f"BM25 score  : {result['bm25_score']:.6f}")

        print(f"PDF         : {filename}")

        print(f"Page        : {record['page_number']}")

        print(f"Passage     : {record['passage_id']}")

        print("Sentences   : " + ", ".join(record["sentence_ids"]))

        print()
        print(
            record.get("text_normalized")
            or record.get(
                "text",
                "",
            )
        )


if __name__ == "__main__":
    main()
