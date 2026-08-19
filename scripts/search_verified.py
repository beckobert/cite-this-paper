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
from search_reranked_qwen import (
    DEFAULT_INSTRUCTION as DEFAULT_RERANK_INSTRUCTION,
)
from search_reranked_qwen import (
    load_qwen_reranker,
    rerank_candidates,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

DEFAULT_VERIFIER = "Qwen/Qwen3-4B-Instruct-2507"

VALID_LABELS = {
    "DIRECT_SUPPORT",
    "PARTIAL_SUPPORT",
    "CONTRADICTS",
    "RELATED_ONLY",
    "REFERENCES",
}


VERIFIER_SYSTEM_PROMPT = """
You are a strict evidence verifier.

Your task is to compare one claim against one passage from an
academic publication.

Judge ONLY from the supplied passage. Do not use outside knowledge.

Classify the relationship using exactly one of these labels:

DIRECT_SUPPORT
The passage provides sufficient evidence for the claim as written.
The wording does not need to be identical, but the important meaning,
scope, direction, comparison, qualifiers, and causal or quantitative
content must be supported.

PARTIAL_SUPPORT
The passage supports a meaningful part of the claim, but not the whole
claim as written. For example, an important qualifier, condition,
comparison, magnitude, causal statement, or generalization is not
supported.

CONTRADICTS
The passage contains evidence that is incompatible with the claim.

RELATED_ONLY
The passage discusses the same topic or concepts, but does not provide
evidence that supports or contradicts the claim.

REFERENCES
The passage itself does not support the claim with its own findings, but
references other publications that do, either by name or citation.

The passage is supplied as numbered sentences such as S1, S2, S3.

Select the MINIMAL set of sentence labels needed to justify your
classification.

For DIRECT_SUPPORT, PARTIAL_SUPPORT, CONTRADICTS, and REFERENCE evidence
should normally contain at least one sentence.

For RELATED_ONLY, evidence should normally be empty.

Do not invent quotations or sentence labels.

Return ONLY a JSON object with exactly these fields:

{
  "label": "DIRECT_SUPPORT",
  "evidence": ["S1", "S2"],
  "reason": "Brief explanation."
}
""".strip()


# ----------------------------------------------------------------------
# General data loading
# ----------------------------------------------------------------------


def load_sentences(
    path: Path,
) -> dict[str, dict]:
    """
    Load sentences.jsonl keyed by sentence_id.
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


def sentence_model_text(
    sentence: dict,
) -> str:
    """
    Prefer normalized text for model input.
    """

    return (
        sentence.get("text_normalized")
        or sentence.get(
            "text",
            "",
        )
    ).strip()


def sentence_source_text(
    sentence: dict,
) -> str:
    """
    Prefer source-faithful text for human display.
    """

    return (
        sentence.get("text")
        or sentence.get(
            "text_normalized",
            "",
        )
    ).strip()


def format_rank(
    rank: int | None,
) -> str:
    if rank is None:
        return "—"

    return str(rank)


# ----------------------------------------------------------------------
# Passage → numbered sentence representation
# ----------------------------------------------------------------------


def build_passage_sentence_input(
    passage: dict,
    sentences: dict[str, dict],
) -> tuple[str, dict[str, str]]:
    """
    Represent a passage as:

        [S1] First sentence.
        [S2] Second sentence.

    Returns:
        model_text
        tag_to_sentence_id

    Using short tags makes it much easier for the verifier to
    reliably select evidence while preserving our real sentence IDs.
    """

    lines = []

    tag_to_sentence_id = {}

    for index, sentence_id in enumerate(
        passage["sentence_ids"],
        start=1,
    ):
        sentence = sentences.get(sentence_id)

        if sentence is None:
            continue

        text = sentence_model_text(sentence)

        if not text:
            continue

        tag = f"S{index}"

        tag_to_sentence_id[tag] = sentence_id

        lines.append(f"[{tag}] {text}")

    if not lines:
        raise RuntimeError(
            f"Passage contains no resolvable sentences: {passage['passage_id']}"
        )

    return (
        "\n".join(lines),
        tag_to_sentence_id,
    )


# ----------------------------------------------------------------------
# Verifier model
# ----------------------------------------------------------------------


def load_verifier(
    model_name: str,
    device: str,
):
    """
    Load a local instruction-following language model.
    """

    print(f"Loading verifier tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading verifier model: {model_name}")

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

    return tokenizer, model


# ----------------------------------------------------------------------
# Verifier response parsing
# ----------------------------------------------------------------------


def extract_json_object(
    text: str,
) -> dict:
    """
    Extract one JSON object even if the model accidentally adds
    surrounding whitespace or a Markdown code fence.
    """

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in verifier output.")

    json_text = text[start : end + 1]

    return json.loads(json_text)


def validate_verdict(
    raw: dict,
    tag_to_sentence_id: dict[str, str],
) -> dict:
    """
    Validate and normalize the model's verdict.

    Evidence tags that do not exist in the supplied passage
    are discarded.
    """

    label = (
        str(
            raw.get(
                "label",
                "",
            )
        )
        .strip()
        .upper()
    )

    if label not in VALID_LABELS:
        raise ValueError(f"Invalid verifier label: {label!r}")

    raw_evidence = raw.get(
        "evidence",
        [],
    )

    if not isinstance(
        raw_evidence,
        list,
    ):
        raw_evidence = []

    evidence_tags = []

    evidence_sentence_ids = []

    seen = set()

    for raw_tag in raw_evidence:
        tag = str(raw_tag).strip().upper()

        if tag in seen:
            continue

        if tag not in tag_to_sentence_id:
            continue

        seen.add(tag)

        evidence_tags.append(tag)

        evidence_sentence_ids.append(tag_to_sentence_id[tag])

    reason = str(
        raw.get(
            "reason",
            "",
        )
    ).strip()

    return {
        "label": label,
        "evidence_tags": evidence_tags,
        "evidence_sentence_ids": evidence_sentence_ids,
        "reason": reason,
    }


# ----------------------------------------------------------------------
# Verify one passage
# ----------------------------------------------------------------------


def verify_passage(
    claim: str,
    passage: dict,
    sentences: dict[str, dict],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    debug: bool = False,
) -> dict:
    """
    Ask the verifier to judge a single passage.
    """

    (
        passage_text,
        tag_to_sentence_id,
    ) = build_passage_sentence_input(
        passage=passage,
        sentences=sentences,
    )

    user_prompt = f"CLAIM\n-----\n{claim}\n\nPASSAGE\n-------\n{passage_text}"

    messages = [
        {
            "role": "system",
            "content": VERIFIER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=(max_new_tokens),
            do_sample=False,
            pad_token_id=(tokenizer.eos_token_id),
        )

    output_ids = generated[0, input_length:]

    raw_output = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip()

    if debug:
        print()
        print("RAW VERIFIER OUTPUT")
        print("-" * 80)
        print(raw_output)
        print()

    try:
        raw_json = extract_json_object(raw_output)

        verdict = validate_verdict(
            raw=raw_json,
            tag_to_sentence_id=(tag_to_sentence_id),
        )

        verdict["parse_success"] = True

    except (
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        verdict = {
            "label": "VERIFICATION_ERROR",
            "evidence_tags": [],
            "evidence_sentence_ids": [],
            "reason": str(exc),
            "parse_success": False,
            "raw_output": raw_output,
        }

    return verdict


# ----------------------------------------------------------------------
# Verify multiple reranked passages
# ----------------------------------------------------------------------


def verify_candidates(
    claim: str,
    candidates: list[dict],
    sentences: dict[str, dict],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    debug: bool,
) -> list[dict]:
    """
    Verify candidates while preserving their reranked order.

    The verifier deliberately does NOT reorder or filter them.
    """

    results = []

    total = len(candidates)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(f"Verifying passage {index}/{total}...")

        verdict = verify_passage(
            claim=claim,
            passage=(candidate["record"]),
            sentences=sentences,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=(max_new_tokens),
            debug=debug,
        )

        result = dict(candidate)

        result["verification"] = verdict

        results.append(result)

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=("Hybrid retrieval + Qwen reranking + local evidence verification.")
    )

    parser.add_argument(
        "claim",
        help=("Claim for which supporting evidence should be found"),
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

    # ----------------------------------------------------------
    # Reranker configuration
    # ----------------------------------------------------------

    parser.add_argument(
        "--reranker",
        default=("Qwen/Qwen3-Reranker-4B"),
    )

    parser.add_argument(
        "--rerank-instruction",
        default=(DEFAULT_RERANK_INSTRUCTION),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=30,
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
        "--rerank-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=1024,
    )

    # ----------------------------------------------------------
    # Verification configuration
    # ----------------------------------------------------------

    parser.add_argument(
        "--verify-k",
        type=int,
        default=10,
        help=("Number of top reranked passages to verify. Default: 10"),
    )

    parser.add_argument(
        "--verifier",
        default=DEFAULT_VERIFIER,
        help=("Local instruction model used for evidence verification."),
    )

    parser.add_argument(
        "--verifier-max-new-tokens",
        type=int,
        default=220,
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

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be > 0")

    if args.rerank_k <= 0:
        raise ValueError("--rerank-k must be > 0")

    if args.verify_k <= 0:
        raise ValueError("--verify-k must be > 0")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    # ----------------------------------------------------------
    # Load metadata.
    # ----------------------------------------------------------

    embedding_dir = args.embedding_dir.resolve()

    documents = load_documents(args.documents.resolve())

    sentences = load_sentences(args.sentences.resolve())

    # ----------------------------------------------------------
    # Dense index.
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Dense retrieval.
    # ----------------------------------------------------------

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

    del embedding_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # BM25.
    # ----------------------------------------------------------

    bm25_results = bm25_search(
        claim=args.claim,
        database_path=(args.bm25_database.resolve()),
        candidate_k=args.candidate_k,
    )

    # ----------------------------------------------------------
    # Weighted RRF.
    # ----------------------------------------------------------

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

    rerank_input = hybrid_results[: args.rerank_k]

    # ----------------------------------------------------------
    # Qwen reranking.
    # ----------------------------------------------------------

    (
        rerank_tokenizer,
        rerank_model,
        token_false_id,
        token_true_id,
    ) = load_qwen_reranker(
        model_name=args.reranker,
        device=args.device,
    )

    reranked = rerank_candidates(
        claim=args.claim,
        candidates=rerank_input,
        tokenizer=rerank_tokenizer,
        model=rerank_model,
        token_false_id=(token_false_id),
        token_true_id=(token_true_id),
        instruction=(args.rerank_instruction),
        device=args.device,
        batch_size=(args.rerank_batch_size),
        max_length=(args.reranker_max_length),
    )

    for rerank_rank, result in enumerate(
        reranked,
        start=1,
    ):
        result["rerank_rank"] = rerank_rank

    # ----------------------------------------------------------
    # Release reranker completely before verifier.
    # ----------------------------------------------------------

    del rerank_model
    del rerank_tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Load verifier.
    # ----------------------------------------------------------

    (
        verifier_tokenizer,
        verifier_model,
    ) = load_verifier(
        model_name=args.verifier,
        device=args.device,
    )

    verification_input = reranked[: args.verify_k]

    verified = verify_candidates(
        claim=args.claim,
        candidates=verification_input,
        sentences=sentences,
        tokenizer=verifier_tokenizer,
        model=verifier_model,
        device=args.device,
        max_new_tokens=(args.verifier_max_new_tokens),
        debug=args.debug,
    )

    # ----------------------------------------------------------
    # Display.
    # ----------------------------------------------------------

    print()
    print("=" * 80)
    print("CLAIM")
    print("=" * 80)
    print(args.claim)

    print()
    print(f"Verified top {len(verified)} reranked passages")

    for result in verified:
        record = result["record"]

        verdict = result["verification"]

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

        print(f"Rerank #{result['rerank_rank']} | {verdict['label']}")

        print(f"Qwen score   : {result['reranker_score']:.4f}")

        print(f"Hybrid rank  : {result['hybrid_rank']}")

        print(f"Dense rank   : {format_rank(result['dense_rank'])}")

        print(f"BM25 rank    : {format_rank(result['bm25_rank'])}")

        print(f"PDF          : {filename}")

        print(f"Page         : {record['page_number']}")

        print(f"Passage      : {record['passage_id']}")

        print()
        print("VERIFIER REASON")
        print(verdict["reason"])

        evidence_ids = verdict["evidence_sentence_ids"]

        if evidence_ids:
            print()
            print("EVIDENCE SENTENCES")

            for sentence_id in evidence_ids:
                sentence = sentences.get(sentence_id)

                if sentence is None:
                    continue

                print()
                print(f"  {sentence_id}")

                print("  " + sentence_source_text(sentence))

        else:
            print()
            print("EVIDENCE SENTENCES: none")

        if args.debug:
            print()
            print("FULL PASSAGE")
            print(
                record.get("text_normalized")
                or record.get(
                    "text",
                    "",
                )
            )


if __name__ == "__main__":
    main()
