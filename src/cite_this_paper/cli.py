"""Command-line interface for independent paper corpora."""

from __future__ import annotations

import argparse
import shlex
import sys
import textwrap
from pathlib import Path

from .cleanup import CleanupResult, cleanup_corpora, find_inactive_corpora
from .corpus import Corpus, CorpusError, DuplicateDocumentError
from .indexing import rebuild_index
from .ingest import IngestResult, ingest_directory, ingest_pdf
from .progress import ConsoleReporter, ProgressReporter
from .retrieval import verify_claim
from .review import render_sentences


RESPONSIBILITY_NOTICE = (
    "IMPORTANT: This output is not guaranteed to be correct. You are solely "
    "responsible for verifying it and deciding how to use or further process "
    "the result."
)


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    raise AssertionError("Unreachable")


def _print_cleanup_report(results: list[CleanupResult], *, mode: str, apply: bool) -> None:
    title = "DATABASE CLEANUP REPORT" if apply else "DATABASE CLEANUP PREVIEW"
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(f"Mode: {mode}")
    if not results:
        print("No corpus databases matched this cleanup request.")
    else:
        print(f"{'STATUS':<10} {'SIZE':>10}  {'LAST ACCESSED':<25} DATABASE")
        print("-" * 88)
        for result in results:
            accessed = result.last_accessed_at or "not tracked"
            print(f"{result.status.upper():<10} {_format_size(result.size_bytes):>10}  {accessed:<25} {result.root}")
            if result.message:
                print(f"           {result.message}")
    print("=" * 88)
    if not apply:
        print("No data was removed. Re-run this command with --apply to permanently delete the listed databases.")


def _cleanup_command(args: argparse.Namespace) -> int:
    explicit = bool(args.databases)
    age_based = args.unused_for is not None
    if explicit == age_based:
        raise CorpusError("Specify database paths or --unused-for DAYS, but not both.")
    if age_based:
        try:
            discovered = find_inactive_corpora(args.root, args.unused_for)
        except ValueError as error:
            raise CorpusError(str(error)) from error
        candidates = [result.root for result in discovered if result.status == "ready"]
        outcomes = cleanup_corpora(candidates, apply=args.apply, protected_root=args.root)
        results = [result for result in discovered if result.status != "ready"] + outcomes
        results.sort(key=lambda result: str(result.root))
        _print_cleanup_report(
            results,
            mode=f"inactive for at least {args.unused_for} day(s) below {args.root.expanduser().resolve()}",
            apply=args.apply,
        )
    else:
        results = cleanup_corpora(args.databases, apply=args.apply)
        _print_cleanup_report(results, mode="explicit database paths", apply=args.apply)
    return 2 if any(result.status == "invalid" for result in results) else 0


def _metadata_from_args(args: argparse.Namespace) -> dict:
    return {
        "title": args.title,
        "authors_json": args.author if args.author else None,
        "publication_year": args.year,
        "journal": args.journal,
        "doi": args.doi,
        "citation_key": args.citation_key,
    }


def _add_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title")
    parser.add_argument("--author", action="append")
    parser.add_argument("--year", type=int)
    parser.add_argument("--journal")
    parser.add_argument("--doi")
    parser.add_argument("--citation-key")


def _ingest_one(
    corpus: Corpus,
    source: Path,
    args: argparse.Namespace,
    reporter: ProgressReporter | None = None,
) -> IngestResult:
    try:
        return ingest_pdf(
            corpus,
            source,
            on_duplicate=args.on_duplicate,
            metadata_overrides=_metadata_from_args(args),
            reporter=reporter,
            debug=getattr(args, "debug", False),
        )
    except DuplicateDocumentError as error:
        if not sys.stdin.isatty():
            raise CorpusError(
                f"Duplicate PDF: {source}. Use --on-duplicate discard or --on-duplicate replace."
            ) from error
        answer = input(f"{source.name} already exists. [d]iscard or [r]eplace? ").strip().lower()
        if answer in {"r", "replace"}:
            return ingest_pdf(
                corpus, source, on_duplicate="replace", metadata_overrides=_metadata_from_args(args),
                reporter=reporter, debug=getattr(args, "debug", False),
            )
        if answer in {"d", "discard", ""}:
            return ingest_pdf(
                corpus, source, on_duplicate="discard", metadata_overrides=_metadata_from_args(args),
                reporter=reporter, debug=getattr(args, "debug", False),
            )
        raise CorpusError("Duplicate choice must be discard or replace.")


def _maybe_rebuild(
    corpus: Corpus,
    args: argparse.Namespace,
    added: bool,
    reporter: ProgressReporter | None = None,
) -> str:
    if not added:
        return "Unchanged (no new PDFs added)"
    rebuild = args.rebuild
    if not args.rebuild and not args.defer_rebuild and sys.stdin.isatty():
        rebuild = input("Rebuild the retrieval index now? [y/N] ").strip().lower() in {"y", "yes"}
    if rebuild:
        result = rebuild_index(corpus, reporter=reporter)
        return f"Rebuilt ({result.indexed_passages} passages, {result.dimensions} dimensions)"
    return "Deferred (new PDFs are not searchable until rebuild-index runs)"


def _print_ingestion_report(corpus: Corpus, results: list[IngestResult], index_status: str) -> None:
    """Render one compact result summary for an ingestion command."""
    statuses = ("added", "replaced", "discarded", "failed")
    counts = {status: sum(result.status == status for result in results) for status in statuses}
    print()
    print("=" * 72)
    print("INGESTION REPORT")
    print("=" * 72)
    print(f"Corpus:    {corpus.root}")
    print(f"Processed: {len(results)} PDF(s)")
    print()
    print("Results")
    print(f"  Added:     {counts['added']}")
    print(f"  Replaced:  {counts['replaced']}")
    print(f"  Duplicates kept: {counts['discarded']}")
    print(f"  Failed:    {counts['failed']}")
    print()
    print(f"Index: {index_status}")
    failures = [result for result in results if result.status == "failed"]
    if failures:
        print()
        print("Failures")
        for result in failures:
            print(f"  - {result.source.name}: {result.message or 'Unknown error'}")
    print("=" * 72)


def _sentence_records(corpus: Corpus, sentence_ids: list[int]) -> list[dict]:
    """Resolve internal sentence IDs while preserving their requested order."""
    if not sentence_ids:
        return []
    placeholders = ", ".join("?" for _ in sentence_ids)
    with corpus.connect() as connection:
        rows = connection.execute(
            f"SELECT id, display_id, source_text FROM sentences WHERE id IN ({placeholders})",
            sentence_ids,
        ).fetchall()
    records_by_id = {int(row["id"]): dict(row) for row in rows}
    return [records_by_id[sentence_id] for sentence_id in sentence_ids if sentence_id in records_by_id]


def _passage_sentence_records(corpus: Corpus, passage_id: int) -> list[dict]:
    """Return every sentence in a passage in its stored passage order."""
    with corpus.connect() as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.display_id, s.source_text
            FROM passage_sentences AS ps
            JOIN sentences AS s ON s.id = ps.sentence_id
            WHERE ps.passage_id = ?
            ORDER BY ps.position
            """,
            (passage_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _format_render_command(corpus: Corpus, sentence_ids: list[str]) -> str:
    return shlex.join(
        [
            "cite-this-paper",
            "show-sentences",
            "--database",
            str(corpus.root),
            *sentence_ids,
        ]
    )


def _print_sentence_evidence(sentence: dict) -> None:
    print(f"  Sentence: {sentence['display_id']}")
    print("  Text:")
    print(
        textwrap.fill(
            sentence["source_text"],
            width=88,
            initial_indent="    ",
            subsequent_indent="    ",
        )
    )


def _format_rank_score(label: str, rank: int | None, score: float | None, score_name: str) -> str:
    if rank is None:
        return f"  {label}: not retrieved"
    if score is None:
        return f"  {label}: rank {rank}"
    return f"  {label}: rank {rank} | {score_name} {score:.4f}"


def _print_verification_output(
    corpus: Corpus,
    claim: str,
    run_id: int,
    warning: str | None,
    results: list,
    *,
    verbose: bool,
) -> None:
    print()
    print("=" * 80)
    print("CLAIM VERIFICATION")
    print("=" * 80)
    print(f"Run:    {run_id}")
    print(f"Corpus: {corpus.root}")
    print("Claim:")
    print(textwrap.fill(claim, width=80, initial_indent="  ", subsequent_indent="  "))
    print(f"Results: {len(results)} verified passage(s)")
    if warning:
        print(f"WARNING: {warning}")

    if not results:
        print()
        print("No passages were verified for this claim.")

    for rank, candidate in enumerate(results, start=1):
        record = candidate.record
        verdict = candidate.verification
        label = verdict.label if verdict else "UNVERIFIED"
        title = record["title"] or record["filename"]

        print()
        print("-" * 80)
        print(f"RESULT {rank}: {label}")
        print("-" * 80)
        print(f"Source: {title}")
        print(f"File:   {record['filename']}")
        print(f"Page:   {record['page_number']}")
        if record.get("doi"):
            print(f"DOI:    {record['doi']}")
        if verdict:
            print("Reason:")
            print(textwrap.fill(verdict.reason or "No reason was provided.", width=80, initial_indent="  ", subsequent_indent="  "))
        if label == "NOT_MENTIONED":
            print(
                "Interpretation: The passage does not meaningfully address the claim; "
                "this absence is not a contradiction."
            )

        evidence = _sentence_records(corpus, candidate.evidence_sentence_ids or [])
        if evidence:
            print("Verifier-selected evidence:")
        else:
            evidence = _passage_sentence_records(corpus, candidate.passage_id)
            print("Passage-wide evidence (fallback: the verifier selected no individual sentences):")

        if evidence:
            for sentence in evidence:
                print()
                _print_sentence_evidence(sentence)
            print()
            print("Show all displayed evidence:")
            print(
                "  "
                + _format_render_command(
                    corpus,
                    [sentence["display_id"] for sentence in evidence],
                )
            )
        else:
            print("  No stored sentences are available for this passage.")

        if verbose:
            print("Retrieval diagnostics:")
            print(f"  Passage: {record['display_id']}")
            print(_format_rank_score("Dense", candidate.dense_rank, candidate.dense_score, "cosine"))
            print(_format_rank_score("BM25", candidate.bm25_rank, candidate.bm25_score, "native score"))
            print(_format_rank_score("Fusion", candidate.fusion_rank, candidate.fusion_score, "RRF score"))
            print(_format_rank_score("Reranker", candidate.rerank_rank, candidate.rerank_score, "probability"))
            if candidate.rerank_logit is not None:
                print(f"  Reranker logit: {candidate.rerank_logit:.4f}")

    print()
    print(RESPONSIBILITY_NOTICE)


def _prepare_verification(
    corpus: Corpus,
    args: argparse.Namespace,
    reporter: ProgressReporter | None = None,
) -> bool:
    """Resolve a pending index rebuild before any verification run is created."""
    if corpus.state()["index_status"] != "rebuild_required":
        return True

    if not sys.stdin.isatty():
        if args.allow_stale_index:
            return True
        raise CorpusError(
            "This corpus has documents pending an index rebuild. Run rebuild-index first "
            "or pass --allow-stale-index to verify against the previous index."
        )

    while True:
        answer = input(
            "This corpus has documents pending an index rebuild. "
            "[r]ebuild now, [c]ontinue with the previous index, or [q]uit? [q]: "
        ).strip().lower()
        if answer in {"r", "rebuild"}:
            result = rebuild_index(corpus) if reporter is None else rebuild_index(corpus, reporter=reporter)
            print(f"Rebuilt {result.indexed_passages} passages ({result.dimensions} dimensions).")
            return True
        if answer in {"c", "continue"}:
            return True
        if answer in {"q", "quit", ""}:
            print("Verification cancelled.")
            return False
        print("Please choose rebuild, continue, or quit.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cite-this-paper", description="Manage and query source-grounded PDF corpora.")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init-db", help="Create a new independent corpus.")
    init.add_argument("database", type=Path, help="Directory for the new corpus")

    for name, source_help in (("add-pdf", "PDF to ingest"), ("add-directory", "Directory scanned recursively for PDFs")):
        add = commands.add_parser(name, help=f"Ingest {source_help.lower()}.")
        add.add_argument("--database", required=True, type=Path)
        add.add_argument("source", type=Path, help=source_help)
        add.add_argument("--on-duplicate", choices=["ask", "discard", "replace"], default="ask")
        rebuild_group = add.add_mutually_exclusive_group()
        rebuild_group.add_argument("--rebuild", action="store_true")
        rebuild_group.add_argument("--defer-rebuild", action="store_true")
        add.add_argument("--quiet", action="store_true", help="Suppress processing updates and progress bars.")
        add.add_argument("--debug", action="store_true", help="Show low-level PDF extraction diagnostics.")
        _add_metadata_arguments(add)

    rebuild = commands.add_parser("rebuild-index", help="Rebuild dense and lexical indexes.")
    rebuild.add_argument("--database", required=True, type=Path)
    rebuild.add_argument("--quiet", action="store_true", help="Suppress processing updates and progress bars.")

    verify = commands.add_parser("verify-claim", help="Retrieve, rerank, and verify evidence for a claim.")
    verify.add_argument("--database", required=True, type=Path)
    verify.add_argument("claim")
    verify.add_argument("--candidate-k", type=int, default=100)
    verify.add_argument("--rerank-k", type=int, default=30)
    verify.add_argument("--verify-k", type=int, default=10)
    verify.add_argument("--device", default="cuda:0")
    verify.add_argument("--verbose", action="store_true", help="Show retrieval and reranking diagnostics.")
    verify.add_argument("--quiet", action="store_true", help="Suppress processing updates and progress bars.")
    verify.add_argument(
        "--allow-stale-index",
        action="store_true",
        help="In non-interactive use, verify with the previous index when documents await rebuilding.",
    )

    cleanup = commands.add_parser("cleanup-databases", help="Preview or permanently remove corpus databases.")
    cleanup.add_argument("databases", nargs="*", type=Path, metavar="DATABASE")
    cleanup.add_argument("--unused-for", type=int, metavar="DAYS", help="Select databases not accessed for this many days.")
    cleanup.add_argument("--root", type=Path, default=Path("data/corpora"), help="Parent directory scanned with --unused-for.")
    cleanup.add_argument("--apply", action="store_true", help="Permanently delete selected databases instead of previewing them.")

    show = commands.add_parser(
        "show-sentences",
        help="Render highlighted source sentences, grouped into one image per PDF page.",
    )
    show.add_argument("--database", required=True, type=Path)
    show.add_argument("sentence_ids", nargs="+", help="One or more sentence IDs to highlight")
    show.add_argument("--output-dir", type=Path)
    show.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-db":
            Corpus.create(args.database)
            print(f"Created corpus: {args.database.resolve()}")
            return 0
        if args.command == "cleanup-databases":
            return _cleanup_command(args)
        corpus = Corpus.open(args.database)
        corpus.touch_access()
        reporter = ConsoleReporter(quiet=getattr(args, "quiet", False))
        if args.command == "add-pdf":
            result = _ingest_one(corpus, args.source, args, reporter)
            index_status = _maybe_rebuild(corpus, args, result.status == "added", reporter)
            _print_ingestion_report(corpus, [result], index_status)
            return 0 if result.status != "failed" else 1
        if args.command == "add-directory":
            results: list[IngestResult] = []
            sources = sorted(args.source.expanduser().resolve().rglob("*.pdf"))
            for source in sources:
                result = _ingest_one(corpus, source, args, reporter)
                results.append(result)
            index_status = _maybe_rebuild(corpus, args, any(result.status == "added" for result in results), reporter)
            _print_ingestion_report(corpus, results, index_status)
            return 1 if any(result.status == "failed" for result in results) else 0
        if args.command == "rebuild-index":
            result = rebuild_index(corpus, reporter=reporter)
            print(f"Rebuilt {result.indexed_passages} passages ({result.dimensions} dimensions).")
            return 0
        if args.command == "verify-claim":
            if not _prepare_verification(corpus, args, reporter):
                return 0
            run_id, warning, results = verify_claim(
                corpus, args.claim, candidate_k=args.candidate_k, rerank_k=args.rerank_k,
                verify_k=args.verify_k, device=args.device, reporter=reporter,
            )
            _print_verification_output(
                corpus,
                args.claim,
                run_id,
                warning,
                results,
                verbose=args.verbose,
            )
            return 0
        if args.command == "show-sentences":
            rendered_pages = render_sentences(
                corpus,
                args.sentence_ids,
                args.output_dir or corpus.root / "review",
                args.dpi,
            )
            print(f"Rendered {len(args.sentence_ids)} sentence(s) across {len(rendered_pages)} page(s):")
            for rendered_page in rendered_pages:
                print(f"{rendered_page.filename} — page {rendered_page.page_number}")
                print("  Sentences: " + ", ".join(rendered_page.sentence_ids))
                print(f"  Rendered page: {rendered_page.output_path}")
            return 0
    except CorpusError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 2
