"""Local, source-grounded academic PDF claim verification."""

from .corpus import Corpus, CorpusError, DuplicateDocumentError
from .cleanup import CleanupResult, cleanup_corpora, find_inactive_corpora
from .ingest import ingest_pdf, ingest_directory
from .indexing import rebuild_index
from .retrieval import verify_claim

__all__ = [
    "Corpus",
    "CorpusError",
    "DuplicateDocumentError",
    "CleanupResult",
    "cleanup_corpora",
    "find_inactive_corpora",
    "ingest_pdf",
    "ingest_directory",
    "rebuild_index",
    "verify_claim",
]
