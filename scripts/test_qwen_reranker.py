#!/usr/bin/env python3

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

INSTRUCTION = """
Given a scientific claim, determine whether the document
provides direct evidence supporting that claim.

Prefer explicit experimental or computational results,
quantitative findings, methodological statements, or
conclusions that substantiate the claim.

Do not rank a document highly merely because it discusses
the same topic.
""".strip()


def format_input(
    instruction: str,
    query: str,
    document: str,
) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )

    print(f"Loading model: {MODEL_NAME}")

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
        )
        .to("cuda")
        .eval()
    )

    false_token_id = tokenizer.convert_tokens_to_ids("no")

    true_token_id = tokenizer.convert_tokens_to_ids("yes")

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. "
        'The answer can only be "yes" or "no".'
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

    claim = (
        "Including nuclear quantum effects makes "
        "flexible water models less structured and "
        "improves their density behavior."
    )

    documents = [
        # Strong supporting evidence
        (
            "It is known from classical and DFT-based "
            "simulations that for flexible water models, "
            "the inclusion of nuclear quantum effects "
            "leads to less structured liquid and improves "
            "the density behavior."
        ),
        # Same topic, but not evidence
        (
            "The role of nuclear quantum effects in "
            "liquid water has attracted substantial "
            "attention in previous computational studies."
        ),
        # Related but different claim
        ("Nuclear quantum effects are less pronounced in D2O than in H2O."),
        # Unrelated
        ("The plane-wave calculations employed an energy cutoff of 500 eV."),
    ]

    formatted = [
        format_input(
            INSTRUCTION,
            claim,
            document,
        )
        for document in documents
    ]

    max_length = 1024

    encoded = tokenizer(
        formatted,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=(max_length - len(prefix_tokens) - len(suffix_tokens)),
    )

    for i, token_ids in enumerate(encoded["input_ids"]):
        encoded["input_ids"][i] = prefix_tokens + token_ids + suffix_tokens

    inputs = tokenizer.pad(
        encoded,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
    )

    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.inference_mode():
        last_logits = model(**inputs).logits[:, -1, :]

        true_logits = last_logits[:, true_token_id]

        false_logits = last_logits[:, false_token_id]

        pair_logits = torch.stack(
            [
                false_logits,
                true_logits,
            ],
            dim=1,
        )

        probabilities = torch.nn.functional.softmax(
            pair_logits,
            dim=1,
        )[:, 1]

    scored = list(
        zip(
            probabilities.cpu().tolist(),
            documents,
        )
    )

    scored.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    print()
    print("=" * 80)
    print("CLAIM")
    print("=" * 80)
    print(claim)

    for rank, (
        score,
        document,
    ) in enumerate(
        scored,
        start=1,
    ):
        print()
        print("=" * 80)
        print(f"{rank}. score={score:.4f}")
        print(document)


if __name__ == "__main__":
    main()
