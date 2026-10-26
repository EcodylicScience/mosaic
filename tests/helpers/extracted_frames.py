"""Frames runs on disk: an entry camera's PNGs, ``run_info.json`` and index row."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline.index_csv import index_records
from mosaic.tracking.frame_extraction.frame_runs import (
    FrameFile,
    FramesIndexRow,
    frames_index,
    frames_index_path,
    frames_row_files,
    list_frame_runs,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "extract_entry_camera",
    "record_frames_row",
    "sole_frames_row_files",
    "write_frames_run",
]


def write_frames_run(directory: Path, sizes: Mapping[int, tuple[int, int]]) -> None:
    """Write one PNG per frame index of *sizes* under *directory*, and their records.

    The ``files`` records of ``run_info.json`` are listed in the order of *sizes*.
    Each PNG's bytes name its directory and frame index, which makes two images
    differ.
    """
    directory.mkdir(parents=True)
    records: list[dict[str, int | str]] = []
    for index, (width, height) in sizes.items():
        name = f"frame_{index:06d}.png"
        _ = (directory / name).write_bytes(f"{directory}/{index}".encode())
        records.append(
            {"frame_index": index, "path": name, "width": width, "height": height}
        )
    _ = (directory / "run_info.json").write_text(json.dumps({"files": records}))


def record_frames_row(
    ds: Dataset,
    stored: str,
    *,
    run_id: str,
    group: str,
    sequence: str,
    camera: str,
    media_composition: str = "",
) -> None:
    """Append the ``uniform`` frames-index row of one entry camera.

    Args:
        ds: The dataset whose frames index gains the row.
        stored: The entry camera's directory as the row stores it.
        run_id: The frames run's id.
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The camera.
        media_composition: The composition of the media the frames were cut from.
    """
    index = frames_index(frames_index_path(ds, "uniform"))
    index.ensure()
    index.append(
        [
            FramesIndexRow(
                run_id=run_id,
                method="uniform",
                group=group,
                sequence=sequence,
                camera=camera,
                abs_path=Path(stored),
                video_abs_path="",
                params_hash="h",
                media_composition=media_composition,
            )
        ]
    )


def extract_entry_camera(
    ds: Dataset,
    *,
    run_id: str,
    group: str,
    sequence: str,
    camera: str,
    sizes: Mapping[int, tuple[int, int]],
    stored_absolute: bool = False,
) -> dict[int, Path]:
    """Write and record one entry camera of a ``uniform`` frames run.

    Args:
        ds: The dataset that contains the run.
        run_id: The frames run's id.
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The camera, or ``""`` for the entry directory.
        sizes: Each extracted frame index and its ``(width, height)``.
        stored_absolute: Store the directory absolute, as a row written before
            root-relative storage does.

    Returns:
        Each frame index's PNG, by its dataset-relative path.
    """
    entry = ds.get_root("frames") / "uniform" / run_id / make_entry_key(group, sequence)
    directory = entry / camera
    write_frames_run(directory, sizes)
    stored = str(directory) if stored_absolute else ds.relative_to_root(directory)
    record_frames_row(
        ds, stored, run_id=run_id, group=group, sequence=sequence, camera=camera
    )
    return {
        index: Path(ds.relative_to_root(directory / f"frame_{index:06d}.png"))
        for index in sizes
    }


def sole_frames_row_files(ds: Dataset) -> tuple[FrameFile, ...]:
    """The extracted PNGs that the one row of *ds*'s frames indexes records.

    Raises:
        ValueError: If the frames indexes do not hold exactly one row.
    """
    (row,) = index_records(list_frame_runs(ds))
    return frames_row_files(ds, row)
