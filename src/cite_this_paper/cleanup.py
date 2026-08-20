"""Safe discovery and permanent removal of independent corpus directories."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CleanupResult:
    """One cleanup candidate and its preview or deletion outcome."""

    root: Path
    last_accessed_at: str | None
    size_bytes: int
    status: str
    message: str | None = None


def corpus_size(root: Path) -> int:
    """Return the total regular-file size without following directory links."""
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def _last_accessed_at(root: Path) -> str | None:
    """Read retention metadata without opening a normal corpus session."""
    try:
        connection = sqlite3.connect(root / "corpus.sqlite")
        try:
            row = connection.execute("SELECT last_accessed_at FROM corpus_state WHERE id = 1").fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    return str(row[0]) if row and row[0] else None


def inspect_corpus(root: Path, *, protected_root: Path | None = None) -> CleanupResult:
    """Validate a corpus directory and collect the information shown to users."""
    root = root.expanduser().resolve()
    if protected_root is not None and root == protected_root.expanduser().resolve():
        return CleanupResult(root, None, 0, "invalid", "The cleanup root itself cannot be deleted.")
    database_path = root / "corpus.sqlite"
    if not root.is_dir() or not database_path.is_file():
        return CleanupResult(root, None, 0, "invalid", "Not a corpus database directory.")
    last_accessed_at = _last_accessed_at(root)
    return CleanupResult(root, last_accessed_at, corpus_size(root), "ready")


def find_inactive_corpora(root: Path, unused_for_days: int, *, now: datetime | None = None) -> list[CleanupResult]:
    """Find valid, tracked corpus directories inactive for the requested duration."""
    if unused_for_days <= 0:
        raise ValueError("The unused-for duration must be a positive number of days.")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Cleanup root does not exist: {root}")
    cutoff = (now or datetime.now(UTC)) - timedelta(days=unused_for_days)
    results: list[CleanupResult] = []
    for database_path in sorted(root.rglob("corpus.sqlite")):
        candidate = inspect_corpus(database_path.parent, protected_root=root)
        if candidate.status != "ready":
            results.append(candidate)
            continue
        if candidate.last_accessed_at is None:
            results.append(
                CleanupResult(
                    candidate.root,
                    None,
                    candidate.size_bytes,
                    "untracked",
                    "No last-access timestamp; recreate this development corpus before age-based cleanup.",
                )
            )
            continue
        try:
            last_accessed = datetime.fromisoformat(candidate.last_accessed_at or "")
        except ValueError:
            results.append(CleanupResult(candidate.root, candidate.last_accessed_at, candidate.size_bytes, "untracked", "Invalid last-access timestamp."))
            continue
        if last_accessed <= cutoff:
            results.append(candidate)
    return results


def cleanup_corpora(corpora: Iterable[Path], *, apply: bool, protected_root: Path | None = None) -> list[CleanupResult]:
    """Preview or permanently delete previously validated corpus directories."""
    inspected = [inspect_corpus(root, protected_root=protected_root) for root in corpora]
    if not apply:
        return [
            result if result.status != "ready" else CleanupResult(result.root, result.last_accessed_at, result.size_bytes, "preview")
            for result in inspected
        ]
    if any(result.status == "invalid" for result in inspected):
        return inspected
    deleted: list[CleanupResult] = []
    for result in inspected:
        shutil.rmtree(result.root)
        deleted.append(CleanupResult(result.root, result.last_accessed_at, result.size_bytes, "deleted"))
    return deleted
