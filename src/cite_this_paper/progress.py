"""Small, optional progress-reporting primitives for long-running workflows."""

from __future__ import annotations

import sys
from typing import Protocol, TextIO

from tqdm import tqdm


class ProgressBar(Protocol):
    """A progress bar controlled by a workflow."""

    def advance(self, amount: int = 1) -> None:
        """Advance the bar by ``amount`` completed items."""

    def close(self) -> None:
        """Finish and release the bar."""


class ProgressReporter(Protocol):
    """Optional observer for package workflows; the package is silent without one."""

    def stage(self, message: str) -> None:
        """Report a human-readable workflow stage."""

    def progress(self, description: str, total: int) -> ProgressBar:
        """Create a progress bar for a known number of work items."""


class _NullProgressBar:
    def advance(self, amount: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


class _TqdmProgressBar:
    def __init__(self, bar: tqdm) -> None:
        self._bar = bar

    def advance(self, amount: int = 1) -> None:
        self._bar.update(amount)

    def close(self) -> None:
        self._bar.close()


class ConsoleReporter:
    """Write workflow stages and interactive progress bars to standard output."""

    def __init__(self, *, quiet: bool = False, stream: TextIO | None = None) -> None:
        self.quiet = quiet
        self.stream = stream or sys.stdout

    def stage(self, message: str) -> None:
        if not self.quiet:
            print(message, file=self.stream, flush=True)

    def progress(self, description: str, total: int) -> ProgressBar:
        if self.quiet:
            return _NullProgressBar()
        return _TqdmProgressBar(
            tqdm(
                total=total,
                desc=description,
                unit="passage",
                file=self.stream,
                disable=not self.stream.isatty(),
            )
        )


def report_stage(reporter: ProgressReporter | None, message: str) -> None:
    """Emit a stage only when the caller requested progress reporting."""
    if reporter is not None:
        reporter.stage(message)


def make_progress(reporter: ProgressReporter | None, description: str, total: int) -> ProgressBar:
    """Create a real or no-op bar, keeping non-CLI package use silent."""
    if reporter is None:
        return _NullProgressBar()
    return reporter.progress(description, total)
