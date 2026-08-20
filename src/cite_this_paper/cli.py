"""Command-line interface for independent paper corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .corpus import Corpus, CorpusError, DuplicateDocumentError
from .indexing import rebuild_index
from .ingest import IngestResult, ingest_directory, ingest_pdf
from .retrieval import verify_claim
from .review import render_sentence, sentence_details


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


def _ingest_one(corpus: Corpus, source: Path, args: argparse.Namespace) -> IngestResult:
    try:
        return ingest_pdf(corpus, source, on_duplicate=args.on_duplicate, metadata_overrides=_metadata_from_args(args))
    except DuplicateDocumentError as error:
        if not sys.stdin.isatty():
            raise CorpusError(
                f"Duplicate PDF: {source}. Use --on-duplicate discard or --on-duplicate replace."
            ) from error
        answer = input(f"{source.name} already exists. [d]iscard or [r]eplace? ").strip().lower()
        if answer in {"r", "replace"}:
            return ingest_pdf(corpus, source, on_duplicate="replace", metadata_overrides=_metadata_from_args(args))
        if answer in {"d", "discard", ""}:
            return ingest_pdf(corpus, source, on_duplicate="discard", metadata_overrides=_metadata_from_args(args))
        raise CorpusError("Duplicate choice must be discard or replace.")


def _maybe_rebuild(corpus: Corpus, args: argparse.Namespace, added: bool) -> None:
    if not added:
        return
    rebuild = args.rebuild
    if not args.rebuild and not args.defer_rebuild and sys.stdin.isatty():
        rebuild = input("Rebuild the retrieval index now? [y/N] ").strip().lower() in {"y", "yes"}
    if rebuild:
        result = rebuild_index(corpus)
        print(f"Rebuilt {result.indexed_passages} passages ({result.dimensions} dimensions).")
    else:
        print("Index rebuild deferred. Existing verification uses the previous index with a warning.")


def _print_result(result: IngestResult) -> None:
    detail = f" ({result.message})" if result.message else ""
    print(f"{result.status:9} {result.source}{detail}")


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
        _add_metadata_arguments(add)

    rebuild = commands.add_parser("rebuild-index", help="Rebuild dense and lexical indexes.")
    rebuild.add_argument("--database", required=True, type=Path)

    verify = commands.add_parser("verify-claim", help="Retrieve, rerank, and verify evidence for a claim.")
    verify.add_argument("--database", required=True, type=Path)
    verify.add_argument("claim")
    verify.add_argument("--candidate-k", type=int, default=100)
    verify.add_argument("--rerank-k", type=int, default=30)
    verify.add_argument("--verify-k", type=int, default=10)
    verify.add_argument("--device", default="cuda:0")

    show = commands.add_parser("show-sentence", help="Render a highlighted source sentence.")
    show.add_argument("--database", required=True, type=Path)
    show.add_argument("sentence_id")
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
        corpus = Corpus.open(args.database)
        if args.command == "add-pdf":
            result = _ingest_one(corpus, args.source, args)
            _print_result(result)
            _maybe_rebuild(corpus, args, result.status == "added")
            return 0 if result.status != "failed" else 1
        if args.command == "add-directory":
            results: list[IngestResult] = []
            for source in sorted(args.source.expanduser().resolve().rglob("*.pdf")):
                result = _ingest_one(corpus, source, args)
                results.append(result)
                _print_result(result)
            _maybe_rebuild(corpus, args, any(result.status == "added" for result in results))
            return 1 if any(result.status == "failed" for result in results) else 0
        if args.command == "rebuild-index":
            result = rebuild_index(corpus)
            print(f"Rebuilt {result.indexed_passages} passages ({result.dimensions} dimensions).")
            return 0
        if args.command == "verify-claim":
            run_id, warning, results = verify_claim(
                corpus, args.claim, candidate_k=args.candidate_k, rerank_k=args.rerank_k,
                verify_k=args.verify_k, device=args.device,
            )
            print(f"Verification run: {run_id}")
            if warning:
                print(f"WARNING: {warning}")
            for rank, candidate in enumerate(results, start=1):
                record = candidate.record
                verdict = candidate.verification
                print()
                print(f"{rank}. {verdict.label if verdict else 'UNVERIFIED'} | rerank={candidate.rerank_score:.4f}")
                print(f"{record['title'] or record['filename']} — page {record['page_number']}")
                print(record["source_text"])
                if verdict:
                    print(verdict.reason)
            return 0
        if args.command == "show-sentence":
            detail = sentence_details(corpus, args.sentence_id)
            output = render_sentence(corpus, args.sentence_id, args.output_dir or corpus.root / "review", args.dpi)
            print(detail["source_text"])
            print(f"Rendered page: {output}")
            return 0
    except CorpusError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 2
