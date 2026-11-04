"""Converted label files and their index rows, written as a converter writes them.

A label file is ``individual_pair_v1``: one row per ``(frame, id1, id2, label)``
event. The default vocabulary is CalMS21's, whose every class is a behavior, so
the default ``background_id`` is ``-1``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.labels_index import write_labels_row

CALMS21_NAMES: tuple[str, ...] = ("attack", "investigation", "mount", "other")
"""The CalMS21 behavior vocabulary, ids 0 to 3."""

LABELS_VARIANT = "convert-labels-calms21_npy.0.3-aaaaaaaaaa"
"""The labels variant :func:`add_labels` files its tables under."""

Event = tuple[int, int, int, int]
"""One label event: ``(frame, id1, id2, label)``."""


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
