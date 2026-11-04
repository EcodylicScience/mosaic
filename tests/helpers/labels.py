"""Converted label files and their index rows, written as a converter writes them.

A label file is ``individual_pair_v1``: one row per ``(frame, id1, id2, label)``
event. The default vocabulary is CalMS21's, whose every class is a behavior, so
the default ``background_id`` is ``-1``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Annotated, ClassVar, override

import numpy as np
import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.label_converter import LabelConverter, LabelConvertParams, LabelEntry
from mosaic.core.params import Declared
from mosaic.core.pipeline.labels_index import write_labels_row

CALMS21_NAMES: tuple[str, ...] = ("attack", "investigation", "mount", "other")
"""The CalMS21 behavior vocabulary, ids 0 to 3."""

LABELS_VARIANT = "convert-labels-calms21_npy.0.3-aaaaaaaaaa"
"""The labels variant :func:`add_labels` files its tables under."""

Event = tuple[int, int, int, int]
"""One label event: ``(frame, id1, id2, label)``."""

LISTED_FRAMES: tuple[int, ...] = (0, 3)
"""The event frames :class:`ListedEntries` gives each entry it lists."""


def write_sparse_labels(
    path: Path,
    events: Sequence[Event],
    *,
    names: Sequence[str] = CALMS21_NAMES,
    background_id: int | None = -1,
) -> Path:
    """Write an ``individual_pair_v1`` file of *events* at *path*.

    Args:
        path: The ``.npz`` to write. Its directory is created.
        events: The events, in file order.
        names: The vocabulary, ids ``0..len(names) - 1``.
        background_id: The class a frame without an event means. ``None``
            leaves the key out, as a file written before it existed did.

    Returns:
        *path*.
    """
    payload: dict[str, np.ndarray] = {
        "label_format": np.array("individual_pair_v1"),
        "frames": np.array([e[0] for e in events], dtype=np.int64),
        "individual_ids": np.array(
            [[e[1], e[2]] for e in events], dtype=np.int64
        ).reshape(-1, 2),
        "labels": np.array([e[3] for e in events], dtype=np.int64),
        "label_ids": np.arange(len(names), dtype=np.int32),
        "label_names": np.array(list(names), dtype=object),
    }
    if background_id is not None:
        payload["background_id"] = np.array(background_id, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, allow_pickle=True, **payload)
    return path


def add_labels(
    ds: Dataset,
    group: str,
    sequence: str,
    events: Sequence[Event],
    *,
    background_id: int | None = -1,
) -> Path:
    """Write the ``behavior`` labels of one entry and their index row.

    Args:
        ds: The dataset.
        group: The entry's group.
        sequence: The entry's sequence.
        events: The entry's events.
        background_id: As :func:`write_sparse_labels` takes it.

    Returns:
        The written file.
    """
    path = write_sparse_labels(
        ds.get_root("labels")
        / "behavior"
        / LABELS_VARIANT
        / f"{make_entry_key(group, sequence)}.npz",
        events,
        background_id=background_id,
    )
    write_labels_row(
        ds,
        run_id=LABELS_VARIANT,
        group=group,
        sequence=sequence,
        out_path=path,
        producer="convert-labels-calms21_npy",
        label_kind="behavior",
        label_format="individual_pair_v1",
        n_frames=len(events),
    )
    return path


PREDICTIONS_VARIANT = "convert-labels-predictions_table.0.1-dddddddddd"
"""The labels variant :func:`add_predictions` files its tables under."""


def write_predictions(
    path: Path,
    events: Sequence[Event],
    *,
    classes: Sequence[int] = (0, 1),
    probabilities: Sequence[Sequence[float]] | None = None,
    folds: Sequence[int] | None = None,
    blocks: Sequence[int] | None = None,
) -> Path:
    """Write a ``predictions`` file of *events* at *path*, as ``predictions_table`` writes one.

    Args:
        path: The ``.npz`` to write. Its directory is created.
        events: The events, in file order, each predicting its label.
        classes: The class ids, which name the probability columns.
        probabilities: One row per event, one column per class, or ``None``
            for a file without probabilities.
        folds: The fold of each event, ``-1`` in sample, or ``None`` for a
            file without folds.
        blocks: The block of each event beside *folds*, or ``None``.

    Returns:
        *path*.
    """
    payload: dict[str, np.ndarray] = {
        "label_format": np.array("individual_pair_v1"),
        "frames": np.array([e[0] for e in events], dtype=np.int64),
        "individual_ids": np.array(
            [[e[1], e[2]] for e in events], dtype=np.int32
        ).reshape(-1, 2),
        "labels": np.array([e[3] for e in events], dtype=np.int32),
        "label_ids": np.array(list(classes), dtype=np.int64),
        "label_names": np.array([str(c) for c in classes], dtype=object),
        "background_id": np.array(-1, dtype=np.int32),
    }
    if probabilities is not None:
        payload["pred_classes"] = np.array(list(classes), dtype=np.int64)
        payload["pred_probabilities"] = np.array(
            probabilities, dtype=np.float64
        ).reshape(len(events), len(classes))
    if folds is not None:
        payload["pred_fold"] = np.array(list(folds), dtype=np.int64)
    if blocks is not None:
        payload["pred_block"] = np.array(list(blocks), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, allow_pickle=True, **payload)
    return path


def add_predictions(
    ds: Dataset,
    group: str,
    sequence: str,
    events: Sequence[Event],
    *,
    classes: Sequence[int] = (0, 1),
    probabilities: Sequence[Sequence[float]] | None = None,
    folds: Sequence[int] | None = None,
) -> Path:
    """Write the ``predictions`` file of one entry and its index row.

    Args:
        ds: The dataset.
        group: The entry's group.
        sequence: The entry's sequence.
        events: The entry's events.
        classes: As :func:`write_predictions` takes it.
        probabilities: As :func:`write_predictions` takes it.
        folds: As :func:`write_predictions` takes it.

    Returns:
        The written file.
    """
    path = write_predictions(
        ds.get_root("labels")
        / "predictions"
        / PREDICTIONS_VARIANT
        / f"{make_entry_key(group, sequence)}.npz",
        events,
        classes=classes,
        probabilities=probabilities,
        folds=folds,
    )
    write_labels_row(
        ds,
        run_id=PREDICTIONS_VARIANT,
        group=group,
        sequence=sequence,
        out_path=path,
        producer="convert-labels-predictions_table",
        label_kind="predictions",
        label_format="individual_pair_v1",
        n_frames=len(events),
    )
    return path


def scan_predictions_tables(ds: Dataset, tables: Mapping[str, pd.DataFrame]) -> None:
    """Write each table as a predictions file, declare their source and scan it.

    Each table is written to ``labels_raw/predictions/<set>/predictions.parquet``,
    so a ``files`` pattern of ``labels_raw/predictions/<set>/*`` converts that
    set alone. The source reads every set as ``predictions_table`` files of
    several sequences, and its raw rows name no group, so each row's ``group``
    cell names its entry's.

    Args:
        ds: The dataset.
        tables: Each set's name, mapped to its predictions table.
    """
    from mosaic.core.manifest import LabelsScanSource

    root = ds.get_root("labels_raw") / "predictions"
    for name, table in tables.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        table.to_parquet(directory / "predictions.parquet", index=False)
    ds.add_scan_source(
        LabelsScanSource(
            id="predictions",
            path=str(root),
            patterns=("*.parquet",),
            src_format="predictions_table",
            multi_sequences_per_file=True,
        )
    )
    _ = ds.scan_labels()


def write_lineage_run(
    features: Path,
    storage: str,
    run_id: str,
    params: Mapping[str, object],
    upstream: Sequence[tuple[str, str]] = (),
) -> Path:
    """Write the ``params.json`` of a feature run that read the runs *upstream*.

    Args:
        features: The dataset's ``features`` root.
        storage: The run's storage directory.
        run_id: The run.
        params: Its recorded ``_params``.
        upstream: The ``(storage, run_id)`` of each run it read, as its
            ``_resolved`` entries name them.

    Returns:
        The run's directory.
    """
    root = features / storage / run_id
    root.mkdir(parents=True)
    document = {
        "_params": dict(params),
        "_inputs": [],
        "_resolved": [
            {"where": f"inputs[{index}]", "feature": feature, "run_id": upstream_id}
            for index, (feature, upstream_id) in enumerate(upstream)
        ],
    }
    _ = (root / "params.json").write_text(json.dumps(document))
    return root


class ListedEntriesParams(LabelConvertParams):
    """The files a :class:`ListedEntries` conversion selects."""

    name: Annotated[
        str, Declared("The file-name pattern of the files that convert.")
    ] = "*"


class ListedEntries(LabelConverter[ListedEntriesParams]):
    """The entries a ``.npy`` of ``(group, sequence)`` rows lists, in its row order.

    Each entry has one ``individual_pair_v1`` event of the pair ``(0, 1)`` at
    each of :data:`LISTED_FRAMES`, labeled with the entry's row position. A file
    converts when its name matches the ``name`` pattern.
    """

    src_format: ClassVar[str] = "listed_entries_test_npy"
    label_kind: ClassVar[str] = "behavior"
    label_format: ClassVar[str] = "individual_pair_v1"
    Params: ClassVar[type[LabelConvertParams]] = ListedEntriesParams

    @override
    def selects(
        self, raw_row: Mapping[str, object], params: ListedEntriesParams
    ) -> bool:
        return fnmatchcase(Path(str(raw_row["abs_path"])).name, params.name)

    @override
    def convert(
        self,
        src_path: Path,
        params: ListedEntriesParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        rows = np.load(src_path).reshape(-1, 2).tolist()
        frames = np.array(LISTED_FRAMES, dtype=np.int64)
        return [
            LabelEntry(
                group=str(group),
                sequence=str(sequence),
                payload={
                    "label_format": self.label_format,
                    "frames": frames,
                    "labels": np.full(len(frames), position, dtype=np.int32),
                    "individual_ids": np.tile([0, 1], (len(frames), 1)),
                    "label_ids": np.arange(len(rows), dtype=np.int64),
                    "background_id": np.int32(-1),
                },
                n_frames=int(frames.max()) + 1,
            )
            for position, (group, sequence) in enumerate(rows)
        ]


class SharedKeyRefusingEntries(ListedEntries):
    """:class:`ListedEntries` refusing one entry key from two converted entries."""

    src_format: ClassVar[str] = "shared_key_refusing_test_npy"
    refuses_shared_entry_keys: ClassVar[bool] = True


class TracksAxisEntries(ListedEntries):
    """:class:`ListedEntries` whose frames are the tracks table's frames."""

    src_format: ClassVar[str] = "tracks_axis_test_npy"
    frames_on_tracks_axis: ClassVar[bool] = True


def write_listed_entries(
    directory: Path, files: Mapping[str, Sequence[tuple[str, str]]]
) -> Path:
    """Write one ``.npy`` per file stem of *files*, listing its ``(group, sequence)`` rows.

    Args:
        directory: Where the files go. Created if absent.
        files: Each file stem mapped to the entries the file lists.

    Returns:
        *directory*.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for stem, entries in files.items():
        np.save(directory / f"{stem}.npy", np.array(entries, dtype=str).reshape(-1, 2))
    return directory
