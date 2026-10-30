"""SLEAP analysis-HDF5 track converter.

Converts a SLEAP *analysis* HDF5 export (``sleap-convert --format analysis``)
to the standardized ``mosaic_v1`` parquet schema. mosaic reads the HDF5 with
``h5py`` alone, so no SLEAP package is needed in the mosaic environment -- the
same dependency-free bridge TREx's per-id NPZ converter provides.

SLEAP analysis HDF5 layout
--------------------------
The default (``matlab``) export is column-major/transposed, but every array
dataset carries a ``dims`` attribute naming its axes, e.g.::

    tracks         (n_tracks, 2, n_nodes, n_frames)   dims ["track","xy","node","frame"]
    point_scores   (n_tracks, n_nodes, n_frames)      dims ["track","node","frame"]

This converter reads those ``dims`` and reorders each array to a canonical
``(frame, track, node, xy)`` before flattening, so it works whether the file was
written with the ``matlab`` or ``standard`` preset. When the attribute is
missing (an older ``sleap-io``), it falls back to the documented ``matlab``
order.

Each SLEAP ``Track`` becomes one ``id`` (its 0-based track index), emitted only
on frames where that track is present. Keypoints are written in node order as
``poseX0..N`` / ``poseY0..N`` (with ``poseP0..N`` point confidence), and ``X``/``Y``
carry the mean of that frame's keypoints as the body centre.

Nothing else is derived here. Velocity, speed and heading used to be computed in
this module and written into the table, where they were indistinguishable from
something SLEAP had reported -- and the heading in particular came from a
principal-component fit whose sign is arbitrary. They belong to the
``speed-angvel`` and ``heading`` features, where the method is chosen, recorded
in a run identifier, and visible to whoever reads the numbers.

The readback of tracked instances (:class:`AppearanceReadback`) is the ``.npz``
that a snippet in the SLEAP environment writes from a predictions ``.slp``: one
record per instance with a track, and its appearance vector where it has one.
:func:`read_appearance_readback` reads it with numpy alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.params import Declared
from mosaic.core.pose_columns import keypoint_centroid
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from mosaic.core.track_library.helpers import norm_hint

# Axis-name aliases. sleap-io names the coordinate axis "xy"; the others are
# singular, but tolerate a plural just in case a future writer emits one.
_XY_ALIASES = frozenset({"xy", "coords", "coord"})
_FRAME_ALIASES = frozenset({"frame", "frames"})
_TRACK_ALIASES = frozenset({"track", "tracks"})
_NODE_ALIASES = frozenset({"node", "nodes"})

# The documented matlab-preset axis orders, used when a dataset has no ``dims``.
_TRACKS_MATLAB_DIMS = ("track", "xy", "node", "frame")
_SCORES_MATLAB_DIMS = ("track", "node", "frame")


def _canon_axis(token: str) -> str:
    """Map an axis token to its canonical name (``frame``/``track``/``node``/``xy``)."""
    t = token.lower()
    if t in _XY_ALIASES:
        return "xy"
    if t in _FRAME_ALIASES:
        return "frame"
    if t in _TRACK_ALIASES:
        return "track"
    if t in _NODE_ALIASES:
        return "node"
    return t


def _decode_token(item: object) -> str:
    """Decode one axis-name token (``bytes`` from HDF5, or already a ``str``)."""
    if isinstance(item, (bytes, bytearray)):
        return item.decode("utf-8", "replace")
    return str(item)


def _decode_dims(raw: object) -> tuple[str, ...] | None:
    """Decode a dataset ``dims`` attribute into axis-name tokens, or None.

    sleap-io stores it as a JSON string; tolerate a raw list / bytes array too.
    """
    value: object = raw
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    tokens = [_decode_token(item) for item in value]
    return tuple(tokens) if tokens else None


def _reorder(
    arr: np.ndarray, dims: tuple[str, ...], target: tuple[str, ...]
) -> np.ndarray:
    """Transpose *arr* whose axes are *dims* into *target* canonical order."""
    canon = [_canon_axis(d) for d in dims]
    if arr.ndim != len(canon):
        raise ValueError(f"SLEAP analysis dataset has {arr.ndim} axes but dims={dims}")
    try:
        order = [canon.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(
            f"SLEAP analysis dataset dims {dims} do not cover {target}"
        ) from exc
    return np.transpose(arr, order)


def _read_analysis_h5(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a SLEAP analysis HDF5 into canonical arrays.

    Returns ``(tracks, point_scores)`` where ``tracks`` is
    ``(n_frames, n_tracks, n_nodes, 2)`` and ``point_scores`` is
    ``(n_frames, n_tracks, n_nodes)`` or ``None`` when the file has no scores.
    """
    # Deferred, against the convention that imports sit at module scope, because
    # this is the only h5py import under ``core/`` and importing any
    # ``mosaic.core.*`` module executes it: ``core/__init__`` imports
    # ``track_library``, which imports this module. That made a binary HDF5 stack
    # a hard requirement for reading three field names off a leaf, and an
    # environment without h5py met a ModuleNotFoundError rather than the missing
    # converter it could have done without. Costs 86 ms and 235 modules, paid by
    # the one caller that opens a SLEAP file.
    import h5py

    with h5py.File(str(path), "r") as handle:
        if "tracks" not in handle:
            raise ValueError(f"SLEAP analysis HDF5 has no 'tracks' dataset: {path}")
        tracks_ds = handle["tracks"]
        tracks_raw = np.asarray(tracks_ds, dtype=np.float64)
        tracks_dims = _decode_dims(tracks_ds.attrs.get("dims")) or _TRACKS_MATLAB_DIMS
        tracks = _reorder(tracks_raw, tracks_dims, ("frame", "track", "node", "xy"))

        scores: np.ndarray | None = None
        if "point_scores" in handle:
            scores_ds = handle["point_scores"]
            scores_raw = np.asarray(scores_ds, dtype=np.float64)
            scores_dims = (
                _decode_dims(scores_ds.attrs.get("dims")) or _SCORES_MATLAB_DIMS
            )
            scores = _reorder(scores_raw, scores_dims, ("frame", "track", "node"))

    return tracks, scores


def _track_to_trex_df(
    xy: np.ndarray,
    scores: np.ndarray | None,
    track_id: int,
    group: str,
    sequence: str,
    fps: float,
) -> pd.DataFrame | None:
    """Build a per-frame ``mosaic_v1`` frame for one track, or None if never present.

    *xy* is ``(n_frames, n_nodes, 2)`` for this track; *scores* is
    ``(n_frames, n_nodes)`` or None. Frames where every node is NaN (the track is
    absent) are dropped, so the table is per-detection.
    """
    n_frames, n_nodes = xy.shape[0], xy.shape[1]
    present = ~np.all(np.isnan(xy.reshape(n_frames, -1)), axis=1)
    if not present.any():
        return None
    frame_idx = np.nonzero(present)[0].astype(int)
    fps = fps if fps > 0 else 1.0

    x = xy[present, :, 0]
    y = xy[present, :, 1]
    cx, cy = keypoint_centroid(x, y)

    frame_f = frame_idx.astype(float)

    data: dict[str, np.ndarray] = {
        "frame": frame_idx,
        "time": frame_f / fps,
        "id": np.full(len(frame_idx), track_id, dtype=int),
        "X": cx,
        "Y": cy,
        "group": np.full(len(frame_idx), group),
        "sequence": np.full(len(frame_idx), sequence),
    }
    for k in range(n_nodes):
        data[f"poseX{k}"] = x[:, k]
        data[f"poseY{k}"] = y[:, k]
    if scores is not None:
        track_scores = scores[present, :]
        for k in range(n_nodes):
            data[f"poseP{k}"] = track_scores[:, k]

    return pd.DataFrame(data)


_FPS_DESCRIPTION = (
    "Frame rate used to derive time. Filled from the source video's measured "
    "fps by the tracker bridge."
)


class SleapConvertParams(TrackConvertParams):
    """Parameters for the SLEAP analysis-HDF5 converter."""

    fps: Annotated[float, Declared(_FPS_DESCRIPTION)] = 30.0


@register_track_converter
class SleapAnalysisH5Converter(TrackConverter[SleapConvertParams]):
    """SLEAP analysis ``.h5`` -> a ``mosaic_v1`` table."""

    src_format = "sleap_analysis_h5"
    # 0.2: derived columns (VX/VY/SPEED/ANGLE and a duplicated #wcentroid
    # pair) are no longer invented here -- they belong to features, where
    # the method is chosen and recorded. The params that picked a heading
    # went with them, so the recipe changed twice over.
    version = "0.2"
    output_schema = "mosaic_v1"
    Params = SleapConvertParams

    def convert(
        self, path: Path, params: SleapConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        tracks, scores = _read_analysis_h5(path)
        group = norm_hint(hints.group) or ""
        sequence = norm_hint(hints.sequence) or path.stem

        n_tracks = tracks.shape[1]
        frames: list[pd.DataFrame] = []
        for t in range(n_tracks):
            df = _track_to_trex_df(
                tracks[:, t, :, :],
                scores[:, t, :] if scores is not None else None,
                t,
                group,
                sequence,
                params.fps,
            )
            if df is not None:
                frames.append(df)

        out = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["frame", "time", "id", "group", "sequence"])
        )
        return out


@dataclass(frozen=True, slots=True)
class AppearanceReadback:
    """The tracked instances of a SLEAP predictions file, in file order.

    Row ``i`` of every array describes one instance that has a track: labeled
    frames in stored order, and each frame's instances in stored order.

    Attributes:
        frame_idx: Each instance's frame index, ``(n,)``.
        track_name: The name of each instance's track, ``(n,)``.
        is_user: Whether each instance is a user instance, ``(n,)``.
        has_vector: Whether each instance has an appearance vector, ``(n,)``.
        vectors: ``(n, D)``, zeros where ``has_vector`` is false. A file
            without embeddings has a ``D`` of 0, and ``has_vector`` is false on
            every row.

    Raises:
        ValueError: The arrays' lengths differ, ``vectors`` is not
            two-dimensional, or a row claims a vector that has no columns.
    """

    frame_idx: npt.NDArray[np.int64]
    track_name: npt.NDArray[np.str_]
    is_user: npt.NDArray[np.bool_]
    has_vector: npt.NDArray[np.bool_]
    vectors: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        n = self.frame_idx.shape[0]
        columns = (
            ("frame_idx", self.frame_idx.shape),
            ("track_name", self.track_name.shape),
            ("is_user", self.is_user.shape),
            ("has_vector", self.has_vector.shape),
        )
        for name, shape in columns:
            if shape != (n,):
                msg = f"{name} has shape {shape}, not ({n},)"
                raise ValueError(msg)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != n:
            msg = f"vectors has shape {self.vectors.shape}, not ({n}, D)"
            raise ValueError(msg)
        if self.vectors.shape[1] == 0 and bool(self.has_vector.any()):
            msg = "has_vector is true on a row, but vectors has no columns"
            raise ValueError(msg)


_READBACK_KINDS: Final = {
    "frame_idx": np.dtype(np.int64),
    "is_user": np.dtype(np.bool_),
    "has_vector": np.dtype(np.bool_),
    "vectors": np.dtype(np.float32),
}
"""The dtype of each readback array but ``track_name``, whose width varies."""


def read_appearance_readback(path: Path) -> AppearanceReadback:
    """Read the readback ``.npz`` of tracked instances at *path*.

    Args:
        path: A file written by the readback snippet.

    Returns:
        Its records.

    Raises:
        ValueError: An array is missing, has another dtype, or its length
            disagrees with the others. The message names the array and *path*.
    """
    with np.load(path, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    for name in ("frame_idx", "track_name", "is_user", "has_vector", "vectors"):
        if name not in arrays:
            msg = f"the readback {path} has no {name} array"
            raise ValueError(msg)
    for name, dtype in _READBACK_KINDS.items():
        if arrays[name].dtype != dtype:
            msg = f"{name} of the readback {path} is {arrays[name].dtype}, not {dtype}"
            raise ValueError(msg)
    if arrays["track_name"].dtype.kind != "U":
        found = arrays["track_name"].dtype
        msg = f"track_name of the readback {path} is {found}, not str"
        raise ValueError(msg)
    try:
        return AppearanceReadback(
            frame_idx=arrays["frame_idx"].astype(np.int64, copy=False),
            track_name=arrays["track_name"].astype(np.str_, copy=False),
            is_user=arrays["is_user"].astype(np.bool_, copy=False),
            has_vector=arrays["has_vector"].astype(np.bool_, copy=False),
            vectors=arrays["vectors"].astype(np.float32, copy=False),
        )
    except ValueError as exc:
        msg = f"the readback {path} is malformed: {exc}"
        raise ValueError(msg) from exc
