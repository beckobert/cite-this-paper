"""Shared hybrid retrieval followed by mandatory reranking and verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

import numpy as np

from .corpus import Corpus, CorpusError
from .indexing import BGEEmbeddingModel, EmbeddingModel, normalize_rows, require_matrix
from .models import (
    VERIFIER_PROMPT_VERSION,
    ClaimVerifier,
    PassageReranker,
    QwenClaimVerifier,
    QwenPassageReranker,
    VerificationOutput,
)


@dataclass
class Candidate:
    passage_id: int
    record: dict[str, Any]
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fusion_rank: int | None = None
    fusion_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    rerank_logit: float | None = None
    verification: VerificationOutput | None = None
    evidence_sentence_ids: list[int] | None = None


def query_terms(claim: str) -> list[str]:
    terms = re.findall(r"[^\W_]+", claim.casefold(), flags=re.UNICODE)
    return list(dict.fromkeys(terms))


def fts_query(claim: str) -> str:
    terms = query_terms(claim)
    if not terms:
        raise CorpusError("The claim contains no searchable lexical terms.")
    return " OR ".join(f'"{term}"' for term in terms)


def _passage_record(connection, passage_id: int) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT p.id, p.display_id, p.source_text, p.normalized_text, p.word_count,
               p.content_type, pg.page_number, d.id AS document_id, d.filename,
               d.title, d.authors_json, d.doi
        FROM passages AS p
        JOIN pages AS pg ON pg.id = p.page_id
        JOIN documents AS d ON d.id = p.document_id
        WHERE p.id = ?
        """,
        (passage_id,),
    ).fetchone()
    if row is None:
        raise CorpusError(f"Indexed passage no longer exists: {passage_id}")
    return dict(row)


def dense_search(corpus: Corpus, claim: str, model: EmbeddingModel, candidate_k: int) -> list[Candidate]:
    matrix = require_matrix(corpus)
    query_matrix = normalize_rows(np.asarray(model.encode([claim]), dtype=np.float32))
    if query_matrix.shape[0] != 1 or query_matrix.shape[1] != matrix.shape[1]:
        raise CorpusError("The embedding model is incompatible with the active matrix.")
    scores = matrix @ query_matrix[0]
    take = min(candidate_k, len(scores))
    indices = np.argsort(scores)[::-1][:take]
    with corpus.connect() as connection:
        rows = connection.execute(
            "SELECT id, embedding_row FROM passages WHERE embedding_row IS NOT NULL ORDER BY embedding_row"
        ).fetchall()
        if len(rows) != len(matrix):
            raise CorpusError("The matrix and database embedding rows are inconsistent. Run rebuild-index.")
        return [
            Candidate(
                passage_id=int(rows[int(index)]["id"]),
                record=_passage_record(connection, int(rows[int(index)]["id"])),
                dense_rank=rank,
                dense_score=float(scores[index]),
            )
            for rank, index in enumerate(indices, start=1)
        ]


def lexical_search(corpus: Corpus, claim: str, candidate_k: int) -> list[Candidate]:
    with corpus.connect() as connection:
        rows = connection.execute(
            """
            SELECT passage_id, bm25(passages_fts) AS lexical_score
            FROM passages_fts
            WHERE passages_fts MATCH ?
            ORDER BY lexical_score
            LIMIT ?
            """,
            (fts_query(claim), candidate_k),
        ).fetchall()
        return [
            Candidate(
                passage_id=int(row["passage_id"]),
                record=_passage_record(connection, int(row["passage_id"])),
                bm25_rank=rank,
                bm25_score=float(row["lexical_score"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]


def hybrid_search(
    corpus: Corpus,
    claim: str,
    embedding_model: EmbeddingModel,
    *,
    candidate_k: int = 100,
    dense_weight: float = 2.0,
    bm25_weight: float = 1.0,
    rrf_k: float = 60.0,
) -> list[Candidate]:
    combined: dict[int, Candidate] = {}
    for candidate in dense_search(corpus, claim, embedding_model, candidate_k):
        candidate.fusion_score = dense_weight / (rrf_k + candidate.dense_rank)
        combined[candidate.passage_id] = candidate
    for lexical in lexical_search(corpus, claim, candidate_k):
        candidate = combined.get(lexical.passage_id)
        if candidate is None:
            lexical.fusion_score = bm25_weight / (rrf_k + lexical.bm25_rank)
            combined[lexical.passage_id] = lexical
            continue
        candidate.bm25_rank = lexical.bm25_rank
        candidate.bm25_score = lexical.bm25_score
        candidate.fusion_score = (candidate.fusion_score or 0.0) + bm25_weight / (rrf_k + lexical.bm25_rank)
    results = sorted(combined.values(), key=lambda item: item.fusion_score or 0.0, reverse=True)
    for rank, candidate in enumerate(results, start=1):
        candidate.fusion_rank = rank
    return results


def _numbered_sentences(corpus: Corpus, passage_id: int) -> tuple[str, dict[str, int]]:
    with corpus.connect() as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.normalized_text
            FROM passage_sentences AS ps
            JOIN sentences AS s ON s.id = ps.sentence_id
            WHERE ps.passage_id = ?
            ORDER BY ps.position
            """,
            (passage_id,),
        ).fetchall()
    mapping: dict[str, int] = {}
    lines: list[str] = []
    for position, row in enumerate(rows, start=1):
        tag = f"S{position}"
        mapping[tag] = int(row["id"])
        lines.append(f"[{tag}] {row['normalized_text']}")
    if not lines:
        raise CorpusError(f"Passage {passage_id} does not contain sentences.")
    return "\n".join(lines), mapping


def _create_run(corpus: Corpus, claim: str, embedding: EmbeddingModel, reranker: PassageReranker, verifier: ClaimVerifier, warning: str | None, config: dict[str, Any]) -> int:
    with corpus.connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO verification_runs (
                claim, embedding_model, reranker_model, verifier_model, configuration_json,
                status, warning, started_at
            ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (claim, embedding.name, reranker.name, verifier.name, json.dumps(config), warning, datetime.now(UTC).isoformat(timespec="seconds")),
        )
        connection.commit()
        return int(cursor.lastrowid)


def _save_run(corpus: Corpus, run_id: int, candidates: Sequence[Candidate]) -> None:
    with corpus.connect() as connection:
        candidate_rows: dict[int, int] = {}
        for candidate in candidates:
            cursor = connection.execute(
                """
                INSERT INTO verification_candidates (
                    verification_run_id, passage_id, dense_rank, dense_score, bm25_rank,
                    bm25_score, fusion_rank, fusion_score, rerank_rank, rerank_score,
                    rerank_logit, verifier_label, verifier_reason, verifier_raw_output,
                    verifier_parse_success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, candidate.passage_id, candidate.dense_rank, candidate.dense_score,
                    candidate.bm25_rank, candidate.bm25_score, candidate.fusion_rank,
                    candidate.fusion_score, candidate.rerank_rank, candidate.rerank_score,
                    candidate.rerank_logit, candidate.verification.label if candidate.verification else None,
                    candidate.verification.reason if candidate.verification else None,
                    candidate.verification.raw_output if candidate.verification else None,
                    int(candidate.verification.parse_success) if candidate.verification else None,
                ),
            )
            candidate_rows[candidate.passage_id] = int(cursor.lastrowid)
        for candidate in candidates:
            for position, sentence_id in enumerate(candidate.evidence_sentence_ids or [], start=1):
                connection.execute(
                    "INSERT INTO verification_evidence (verification_candidate_id, sentence_id, position) VALUES (?, ?, ?)",
                    (candidate_rows[candidate.passage_id], sentence_id, position),
                )
        connection.execute(
            "UPDATE verification_runs SET status = 'completed', completed_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(timespec="seconds"), run_id),
        )
        connection.commit()


def verify_claim(
    corpus: Corpus,
    claim: str,
    *,
    embedding_model: EmbeddingModel | None = None,
    reranker: PassageReranker | None = None,
    verifier: ClaimVerifier | None = None,
    candidate_k: int = 100,
    rerank_k: int = 30,
    verify_k: int = 10,
    device: str = "cuda:0",
) -> tuple[int, str | None, list[Candidate]]:
    """Run the required hybrid retrieval, reranking, and verification sequence."""
    if not claim.strip():
        raise CorpusError("The claim cannot be empty.")
    config = corpus.config()
    embedding_model = embedding_model or BGEEmbeddingModel(config["embedding_model"])
    reranker = reranker or QwenPassageReranker(config["reranker_model"], device=device)
    verifier = verifier or QwenClaimVerifier(config["verifier_model"], device=device)
    state = corpus.state()
    warning = None
    if state["index_status"] == "rebuild_required":
        warning = "The corpus has newly ingested documents that are pending index rebuild and were not searched."
    if state["index_status"] == "empty":
        raise CorpusError("This corpus has no index yet. Run rebuild-index first.")
    run_id = _create_run(
        corpus, claim, embedding_model, reranker, verifier, warning,
        {
            "candidate_k": candidate_k,
            "rerank_k": rerank_k,
            "verify_k": verify_k,
            "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        },
    )
    try:
        candidates = hybrid_search(corpus, claim, embedding_model, candidate_k=candidate_k)
        reranked = candidates[:rerank_k]
        scores = reranker.rerank(claim, [candidate.record["normalized_text"] for candidate in reranked])
        for candidate, (score, logit) in zip(reranked, scores, strict=True):
            candidate.rerank_score, candidate.rerank_logit = score, logit
        reranked.sort(key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)
        for rank, candidate in enumerate(reranked, start=1):
            candidate.rerank_rank = rank
        verified = reranked[:verify_k]
        for candidate in verified:
            numbered, tags = _numbered_sentences(corpus, candidate.passage_id)
            verdict = verifier.verify(claim, numbered)
            candidate.verification = verdict
            candidate.evidence_sentence_ids = [tags[tag] for tag in verdict.evidence_tags if tag in tags]
        _save_run(corpus, run_id, candidates)
        return run_id, warning, verified
    except Exception as error:
        with corpus.connect() as connection:
            connection.execute(
                "UPDATE verification_runs SET status = 'failed', error_message = ?, completed_at = ? WHERE id = ?",
                (str(error), datetime.now(UTC).isoformat(timespec="seconds"), run_id),
            )
            connection.commit()
        raise
