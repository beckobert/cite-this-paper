"""Local model adapters used by the mandatory reranking and verification stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence


class PassageReranker(Protocol):
    name: str

    def rerank(self, claim: str, passages: Sequence[str]) -> list[tuple[float, float]]:
        """Return (probability, logit difference) for each passage."""


class ClaimVerifier(Protocol):
    name: str

    def verify(self, claim: str, numbered_sentences: str) -> "VerificationOutput":
        """Classify one passage based only on its numbered sentence text."""


@dataclass(frozen=True)
class VerificationOutput:
    label: str
    evidence_tags: list[str]
    reason: str
    raw_output: str = ""
    parse_success: bool = True


RERANK_INSTRUCTION = (
    "Given a scientific claim, determine whether the document provides evidence "
    "that supports that claim. Prefer explicit scientific statements, results, "
    "methods, or conclusions. Do not rank a document highly merely because it "
    "discusses the same topic."
)

VERIFIER_PROMPT = """You are a strict evidence verifier. Compare a claim with one academic passage.
Judge only from the supplied passage. Return exactly one JSON object with fields
label, evidence, and reason. label must be one of DIRECT_SUPPORT,
PARTIAL_SUPPORT, CONTRADICTS, RELATED_ONLY, REFERENCES. Evidence must contain
only supplied sentence labels such as S1. DIRECT_SUPPORT requires the important
meaning, scope, qualifiers, and quantitative or causal content of the claim to
be supported. PARTIAL_SUPPORT means a meaningful part is supported but an
important qualifier or scope is missing. REFERENCES means the passage merely
points to another publication. RELATED_ONLY normally has no evidence labels."""


class QwenPassageReranker:
    """Qwen3-Reranker adapter using its yes/no final-token convention."""

    def __init__(self, name: str, device: str = "cuda:0", max_length: int = 1024, batch_size: int = 8):
        self.name = name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._no_token = None
        self._yes_token = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for reranking but is unavailable.")
        self._tokenizer = AutoTokenizer.from_pretrained(self.name, padding_side="left")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.name, dtype=dtype).to(self.device).eval()
        self._no_token = self._tokenizer.convert_tokens_to_ids("no")
        self._yes_token = self._tokenizer.convert_tokens_to_ids("yes")
        if self._no_token is None or self._yes_token is None:
            raise RuntimeError("The reranker tokenizer does not expose yes/no token IDs.")

    def rerank(self, claim: str, passages: Sequence[str]) -> list[tuple[float, float]]:
        self._load()
        import torch

        assert self._tokenizer is not None and self._model is not None
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            "based on the Query and Instruct. Answer only yes or no.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_ids = self._tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self._tokenizer.encode(suffix, add_special_tokens=False)
        available = self.max_length - len(prefix_ids) - len(suffix_ids)
        if available <= 0:
            raise ValueError("Reranker maximum length is too small for its prompt wrapper.")

        results: list[tuple[float, float]] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            pairs = [f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {claim}\n<Document>: {text}" for text in batch]
            encoded = self._tokenizer(
                pairs, padding=False, truncation=True, max_length=available, return_attention_mask=False
            )
            encoded["input_ids"] = [prefix_ids + ids + suffix_ids for ids in encoded["input_ids"]]
            inputs = self._tokenizer.pad(encoded, padding=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self._model(**inputs).logits[:, -1, :].float()
                no_logits = logits[:, self._no_token]
                yes_logits = logits[:, self._yes_token]
                probabilities = torch.softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)[:, 1]
                differences = yes_logits - no_logits
            results.extend(zip(probabilities.cpu().tolist(), differences.cpu().tolist()))
        return [(float(score), float(logit)) for score, logit in results]


class QwenClaimVerifier:
    """Instruction-model adapter that produces a structured evidence verdict."""

    def __init__(self, name: str, device: str = "cuda:0", max_new_tokens: int = 512):
        self.name = name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for verification but is unavailable.")
        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.name, dtype=dtype).to(self.device).eval()

    def verify(self, claim: str, numbered_sentences: str) -> VerificationOutput:
        self._load()
        import torch

        assert self._tokenizer is not None and self._model is not None
        messages = [
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": f"CLAIM\n-----\n{claim}\n\nPASSAGE\n-------\n{numbered_sentences}"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        raw = self._tokenizer.decode(generated[0, input_length:], skip_special_tokens=True).strip()
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            label = str(parsed.get("label", "")).strip().upper()
            valid = {"DIRECT_SUPPORT", "PARTIAL_SUPPORT", "CONTRADICTS", "RELATED_ONLY", "REFERENCES"}
            if label not in valid:
                raise ValueError(f"Invalid verifier label: {label!r}")
            evidence = parsed.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            return VerificationOutput(label, [str(tag).strip().upper() for tag in evidence], str(parsed.get("reason", "")).strip(), raw)
        except (ValueError, json.JSONDecodeError) as error:
            return VerificationOutput("VERIFICATION_ERROR", [], str(error), raw, False)
