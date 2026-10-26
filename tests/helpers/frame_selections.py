"""Select-frames runs on disk: an entry's ``selection.parquet`` and its index row."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mosaic.core.helpers import entry_directory
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.tracking.frame_extraction.selection_index import (
    SELECTION_FILE_NAME,
    FrameSelectionRow,
    frame_selection_index,
    selection_index_path,
    selections_kind_root,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["entry_selection", "record_entry_selection"]

_SOURCE_VARIANT = "trex.0.1-0123456789"
"""The tracks variant that a recorded selection names as its source by default."""


def entry_selection(group: str, sequence: str, frames: Sequence[int]) -> pd.DataFrame:
    """Build an entry's selection table with one ``jump`` event of id ``1`` per frame.

    The rows follow *frames* in the order given, which need not be frame order.
    """
    count = len(frames)
    return pd.DataFrame(
        {
            "group": [group] * count,
            "sequence": [sequence] * count,
            "frame": np.asarray(frames, dtype=np.int64),
            "cause": ["jump"] * count,
            "event_id": [f"jump:1:{frame}" for frame in frames],
            "id": ["1"] * count,
            "x": np.full(count, 10.0),
            "y": np.full(count, 20.0),
            "score": np.full(count, 0.9),
            "event_frames": np.ones(count, dtype=np.int64),
        }
    )


def record_entry_selection(
    ds: Dataset,
    *,
    run_id: str,
    group: str,
    sequence: str,
    frames: Sequence[int],
    finished: bool = True,
    source_tracks_run_id: str = _SOURCE_VARIANT,
) -> Path:
    """Write an entry's ``selection.parquet`` of *frames* and append its index row.

    Args:
        ds: The dataset whose selections root gains the file and row.
        run_id: The select-frames run.
        group: The entry's group.
        sequence: The entry's sequence.
        frames: The frames that the file lists, in file order.
        finished: Mark every unfinished row of *run_id* finished afterwards.
        source_tracks_run_id: The tracks variant that the row names.

    Returns:
        The file written, which the row names dataset-relative.
    """
    run_root = selections_kind_root(ds) / run_id
    target = entry_directory(run_root, group, sequence) / SELECTION_FILE_NAME
    _ = write_parquet_atomic(entry_selection(group, sequence, frames), target)
    index = frame_selection_index(selection_index_path(ds))
    index.append(
        [
            FrameSelectionRow(
                run_id=run_id,
                abs_path=Path(ds.relative_to_root(target)),
                group=group,
                sequence=sequence,
                source_tracks_run_id=source_tracks_run_id,
                producer="trex",
                n_candidates=len(frames),
                n_selected=len(frames),
                skipped_criteria="[]",
                body_length_px="12.5",
                params_hash="h",
            )
        ]
    )
    if finished:
        index.mark_finished(run_id)
    return target
