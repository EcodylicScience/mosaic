"""Progress callback protocol and implementations.

Defines the callback contract that jobs call at regular intervals to report
progress. Granularities span per-epoch / per-class training progress, per-entry
feature-run progress (``on_entry_*``), and coarse per-phase transitions (e.g.
trex convert/track).

The default backend the Job Contract binds is the append-only JSONL run-log
(:class:`mosaic.core.pipeline.run_log.JsonlRunLog`), which implements this
protocol and writes every event -- keyed by ``execution_id`` -- to one file the
API layer can tail for live monitoring. This module keeps the protocol and the
storage-free backends (``Null`` / ``CSV`` / ``Composite``).

The protocol is exported both as ``TrainingProgressCallback`` (historical
name) and ``ProgressCallback`` (its broader current role).

Also provides ``CSVProgressCallback`` for append-mode CSV writing that
is readable mid-training (no database required).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol (stable interface for future backends)
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingProgressCallback(Protocol):
    """Minimal callback contract for training progress reporting.

    Implement this protocol to create custom progress backends (e.g.
    MLflow, W&B).  The three methods correspond to different granularities
    of training progress.
    """

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        """Called after each training epoch.

        Args:
            epoch: Zero-based epoch index.
            total_epochs: Total number of epochs planned.
            metrics: Metric name-value pairs (e.g. ``{"train_loss": 0.45}``).
        """
        ...

    def on_class_start(
        self,
        class_idx: int,
        total_classes: int,
        class_name: str,
    ) -> None:
        """Called when a one-vs-rest class training begins.

        Args:
            class_idx: Zero-based class index.
            total_classes: Total number of classes.
            class_name: Human-readable class identifier.
        """
        ...

    def on_phase(self, phase: str, message: str) -> None:
        """Called for coarse-grained phase transitions.

        Args:
            phase: Phase identifier (e.g. ``"data_prep"``, ``"training"``).
            message: Free-text description.
        """
        ...

    def on_entry_start(self, index: int, total: int, key: str) -> None:
        """Called just before a per-entry unit of work begins.

        Args:
            index: Zero-based index of the entry about to be processed.
            total: Total number of entries planned for this run.
            key: Human-readable entry identifier (e.g. ``"group/sequence"``).
        """
        ...

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        """Called after a per-entry unit of work completes.

        Args:
            index: One-based count of entries completed so far.
            total: Total number of entries planned for this run.
            key: Human-readable entry identifier (e.g. ``"group/sequence"``).
        """
        ...


# ---------------------------------------------------------------------------
# Null implementation (no-op default)
# ---------------------------------------------------------------------------


class NullProgressCallback:
    """No-op callback used when no monitoring is requested."""

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        pass

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        pass

    def on_phase(self, phase: str, message: str) -> None:
        pass

    def on_entry_start(self, index: int, total: int, key: str) -> None:
        pass

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        pass


# ---------------------------------------------------------------------------
# CSV implementation (append-mode, readable mid-training)
# ---------------------------------------------------------------------------


class CSVProgressCallback:
    """Append-mode CSV writer for epoch metrics.

    Writes one row per ``on_epoch_end`` call.  The file is flushed after
    each write so it can be read from another process (e.g. a notebook
    cell or ``cat`` from the terminal) while training is still running.

    Example::

        cb = CSVProgressCallback(run_root / "summary.csv")
        for epoch in range(n_epochs):
            ...
            cb.on_epoch_end(epoch, n_epochs, {"train_loss": 0.3, "val_f1": 0.9})
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fieldnames: list[str] | None = None
        self._file = None
        self._writer = None

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        row = {"epoch": epoch, "total_epochs": total_epochs, **metrics}

        if self._file is None:
            # First call -- open file in append mode
            write_header = not (self.path.exists() and self.path.stat().st_size > 0)
            self._fieldnames = list(row.keys())
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            if write_header:
                self._writer.writeheader()

        self._writer.writerow(row)  # type: ignore[union-attr]
        self._file.flush()  # type: ignore[union-attr]

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        pass

    def on_phase(self, phase: str, message: str) -> None:
        pass

    def on_entry_start(self, index: int, total: int, key: str) -> None:
        pass

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        pass

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


# ---------------------------------------------------------------------------
# Composite (for running multiple backends simultaneously)
# ---------------------------------------------------------------------------


class CompositeProgressCallback:
    """Fans out calls to multiple callback backends.

    Example::

        cb = CompositeProgressCallback(
            JsonlRunLog(path, execution_id),
            some_mlflow_callback,
        )
    """

    def __init__(self, *backends: TrainingProgressCallback) -> None:
        self._backends = backends

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        for b in self._backends:
            b.on_epoch_end(epoch, total_epochs, metrics)

    def on_class_start(
        self, class_idx: int, total_classes: int, class_name: str
    ) -> None:
        for b in self._backends:
            b.on_class_start(class_idx, total_classes, class_name)

    def on_phase(self, phase: str, message: str) -> None:
        for b in self._backends:
            b.on_phase(phase, message)

    def on_entry_start(self, index: int, total: int, key: str) -> None:
        for b in self._backends:
            b.on_entry_start(index, total, key)

    def on_entry_end(self, index: int, total: int, key: str) -> None:
        for b in self._backends:
            b.on_entry_end(index, total, key)


# ``ProgressCallback`` is the broader current name for the same protocol; the
# ``TrainingProgressCallback`` alias is retained for existing imports.
ProgressCallback = TrainingProgressCallback
