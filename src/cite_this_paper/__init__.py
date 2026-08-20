"""Local, source-grounded academic PDF claim verification."""

from .corpus import Corpus, CorpusError, DuplicateDocumentError
from .cleanup import CleanupResult, cleanup_corpora, find_inactive_corpora
from .ingest import ingest_pdf
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
    "rebuild_index",
    "verify_claim",
]
