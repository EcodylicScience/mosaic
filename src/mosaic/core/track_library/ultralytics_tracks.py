"""Ultralytics tracker output -> a ``trex_v1`` table.

The tracker writes one long-format parquet per entry: a row per
``(frame, track)`` carrying the box, the detection confidence and class, and --
for a pose model -- the keypoint triples. This converter turns that into the
standardized table, which is where identity is densified and the kinematic
columns are derived.

Modeled on the SLEAP converter rather than the DeepLabCut one. DeepLabCut's
per-individual builder assumes a dense array of ``T`` frames per animal and
addresses it with ``np.arange(T)``; an Ultralytics track appears, vanishes under
occlusion and reappears, so its frames are neither dense nor contiguous and
velocity has to be differentiated against the true frame index.

The raw column names are declared here rather than in the tracker, because this
is the module that has to read them back. The tracker imports them, which is the
allowed direction: ``tracking`` may depend on ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mosaic.core.track_converter import (
    EntryHints,
    TrackConvertParams,
    TrackConverter,
    register_track_converter,
)
from mosaic.core.track_library.helpers import angle_from_pca, angle_from_two_points

RAW_FRAME: Final = "frame"
RAW_TRACK_ID: Final = "track_id"
RAW_BOX: Final[tuple[str, str, str, str]] = ("x1", "y1", "x2", "y2")
RAW_CONF: Final = "conf"
RAW_CLS: Final = "cls"


def raw_keypoint_columns(n_keypoints: int) -> tuple[str, ...]:
    """The keypoint triples a raw table carries, in write order."""
    names: list[str] = []
    for k in range(n_keypoints):
        names.extend((f"kpx{k}", f"kpy{k}", f"kpp{k}"))
    return tuple(names)


def raw_columns(n_keypoints: int) -> tuple[str, ...]:
    """Every column of the raw predictions table, in write order."""
    return (
        RAW_FRAME,
        RAW_TRACK_ID,
        *RAW_BOX,
        RAW_CONF,
        RAW_CLS,
        *raw_keypoint_columns(n_keypoints),
    )


_EMPTY_COLUMNS: Final[list[str]] = ["frame", "time", "id", "group", "sequence"]

_HEADING_SPEED_FLOOR: Final = 1e-6
"""Below this speed, a velocity heading is direction-less rather than zero.

Reached only on the one-keypoint (box-only) path, where heading has no other
source. A stationary animal has no heading, and ``0.0`` would read as "pointing
along +x".
"""


@dataclass(frozen=True, slots=True)
class _RawTable:
    """The raw predictions parquet, as typed arrays.

    Pulled out of pandas immediately: everything downstream is arithmetic, and
    the shapes are fixed by ``raw_columns``.
    """

    frame: np.ndarray
    track_id: np.ndarray
    box: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    keypoints: np.ndarray
    """``(n_rows, n_keypoints, 3)`` -- x, y, confidence."""

    @property
    def n_keypoints(self) -> int:
        return int(self.keypoints.shape[1])


def _column_names(table: pd.DataFrame) -> list[str]:
    """The frame's column labels as plain strings."""
    return [str(name) for name in table.columns.astype(str).tolist()]


def _read_raw(path: Path) -> _RawTable | None:
    """Load *path*, or None when it holds no detections."""
    table: pd.DataFrame = pd.read_parquet(path)
    n_keypoints = sum(1 for name in _column_names(table) if name.startswith("kpx"))
    if n_keypoints == 0:
        raise ValueError(
            f"{path} carries no keypoint columns. An Ultralytics tracker writes at "
            "least one triple, synthesizing it from the box centre for a box-only "
            "model, so a table without one was not written by this tracker."
        )
    if len(table) == 0:
        return None

    def column(name: str, dtype: str) -> np.ndarray:
        return np.asarray(table[name], dtype=dtype)

    keypoints = np.stack(
        [
            np.stack(
                [
                    column(f"kpx{k}", "float64"),
                    column(f"kpy{k}", "float64"),
                    column(f"kpp{k}", "float64"),
                ],
                axis=-1,
            )
            for k in range(n_keypoints)
        ],
        axis=1,
    )
    return _RawTable(
        frame=column(RAW_FRAME, "int64"),
        track_id=column(RAW_TRACK_ID, "int64"),
        box=np.stack([column(name, "float64") for name in RAW_BOX], axis=-1),
        conf=column(RAW_CONF, "float64"),
        cls=column(RAW_CLS, "int64"),
        keypoints=keypoints,
    )


def _track_to_trex_df(
    raw: _RawTable,
    rows: np.ndarray,
    *,
    dense_id: int,
    group: str,
    sequence: str,
    fps: float,
    neck_idx: int | None,
    tail_idx: int | None,
) -> pd.DataFrame:
    """One track's rows, already frame-ordered, as a ``trex_v1`` frame."""
    frame_idx = raw.frame[rows]
    frame_f = frame_idx.astype(float)
    n = len(rows)
    n_keypoints = raw.n_keypoints

    x = raw.keypoints[rows, :, 0]
    y = raw.keypoints[rows, :, 1]
    with np.errstate(invalid="ignore"):
        cx = np.nanmean(x, axis=1)
        cy = np.nanmean(y, axis=1)

    if n > 1:
        # Frame spacing is not uniform -- a track drops out and returns -- so
        # differentiate against the frame index and scale to per-second.
        vx = np.gradient(cx, frame_f) * fps
        vy = np.gradient(cy, frame_f) * fps
    else:
        vx = np.zeros(n)
        vy = np.zeros(n)
    speed = np.hypot(vx, vy)

    if (
        neck_idx is not None
        and tail_idx is not None
        and 0 <= neck_idx < n_keypoints
        and 0 <= tail_idx < n_keypoints
    ):
        angle = angle_from_two_points(
            np.stack([x[:, neck_idx], y[:, neck_idx]], axis=-1),
            np.stack([x[:, tail_idx], y[:, tail_idx]], axis=-1),
        )
    elif n_keypoints >= 2:
        angle = angle_from_pca(np.stack([x, y], axis=-1))
    else:
        # A box tracker has one synthetic keypoint, so there is no shape to take
        # a heading from. Direction of travel is the only signal left, and it
        # beats leaving a recommended column permanently NaN.
        angle = np.where(speed > _HEADING_SPEED_FLOOR, np.arctan2(vy, vx), np.nan)

    data: dict[str, np.ndarray] = {
        "frame": frame_idx,
        "time": frame_f / fps,
        "id": np.full(n, dense_id, dtype=int),
        "source_track_id": raw.track_id[rows],
        "X": cx,
        "Y": cy,
        "X#wcentroid": cx,
        "Y#wcentroid": cy,
        "VX": vx,
        "VY": vy,
        "SPEED": speed,
        "ANGLE": angle,
        "group": np.full(n, group),
        "sequence": np.full(n, sequence),
        "det_conf": raw.conf[rows],
        "det_cls": raw.cls[rows],
    }
    for axis, name in enumerate(RAW_BOX):
        data[f"bbox_{name}"] = raw.box[rows, axis]
    for k in range(n_keypoints):
        data[f"poseX{k}"] = x[:, k]
        data[f"poseY{k}"] = y[:, k]
        data[f"poseP{k}"] = raw.keypoints[rows, k, 2]
    return pd.DataFrame(data)


class UltralyticsTracksParams(TrackConvertParams):
    """Parameters for the Ultralytics tracker-output converter.

    Attributes:
        fps: Frame rate used to derive ``time`` and velocities. Filled from the
            source video's measured fps by the tracker bridge, so the value that
            reaches identity is the one that was used.
        neck_idx: Keypoint index of the neck for a two-point heading.
        tail_idx: Keypoint index of the tail for a two-point heading. With
            neither index, heading falls back to per-frame PCA over all
            keypoints, or -- for a box-only model -- to direction of travel.
    """

    fps: float = 30.0
    neck_idx: int | None = None
    tail_idx: int | None = None


@register_track_converter
class UltralyticsTracksConverter(TrackConverter[UltralyticsTracksParams]):
    """Ultralytics tracker predictions parquet -> a ``trex_v1`` table.

    Identity is **densified** here: Ultralytics numbers tracks from 1 with gaps,
    and mosaic's ``id`` is 0-based and dense everywhere else. The tracker's own
    value survives as ``source_track_id``, so a row can still be traced back to
    it.

    For a box-only (``detect``) model the tracker synthesizes one keypoint at the
    box centre with the *detection* confidence as its ``poseP0``: the schema
    requires at least one ``poseX``/``poseY`` pair, and a box centre is the
    honest answer for a model that localizes nothing finer.
    """

    src_format = "ultralytics_tracks"
    output_schema = "trex_v1"
    Params = UltralyticsTracksParams

    def convert(
        self, path: Path, params: UltralyticsTracksParams, hints: EntryHints
    ) -> pd.DataFrame:
        raw = _read_raw(path)
        if raw is None:
            return pd.DataFrame(columns=_EMPTY_COLUMNS)

        group = hints.group or ""
        sequence = hints.sequence or path.stem
        fps = params.fps if params.fps > 0 else 1.0

        # Dense ids follow the tracker's own ascending order, so the mapping is a
        # property of the file rather than of the order its rows happen to be in.
        sources = np.unique(raw.track_id)
        frames = [
            _track_to_trex_df(
                raw,
                np.nonzero(raw.track_id == source)[0][
                    np.argsort(raw.frame[raw.track_id == source], kind="stable")
                ],
                dense_id=dense_id,
                group=group,
                sequence=sequence,
                fps=fps,
                neck_idx=params.neck_idx,
                tail_idx=params.tail_idx,
            )
            for dense_id, source in enumerate(sources)
        ]
        out = pd.concat(frames, ignore_index=True).sort_values(
            ["frame", "id"], kind="stable", ignore_index=True
        )
        return out
