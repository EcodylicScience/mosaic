"""DeepLabCut track converter (.csv / .h5).

Converts DeepLabCut pose-estimation output to the standardized ``mosaic_v1``
parquet schema. Supports both single-animal and multi-animal (maDLC) exports,
in either CSV or HDF5 form.

DeepLabCut layout
-----------------
DLC stores a wide table whose columns form a MultiIndex:

* single-animal: ``(scorer, bodypart, coord)`` — 3 levels
* multi-animal:  ``(scorer, individual, bodypart, coord)`` — 4 levels

where ``coord`` is one of ``x``, ``y``, ``likelihood``. The (unnamed) first
column of a CSV export is the 0-based frame index. This converter is agnostic to
the scorer name and to the specific bodypart names; keypoints are emitted in
file order as ``poseX0..N`` / ``poseY0..N`` (with ``poseP0..N`` confidence when
the ``likelihood`` column is present).

A single-animal file becomes one ``id=0`` track; a multi-animal file becomes one
``id`` per individual (in file order). ``X``/``Y`` carry the mean of that frame's
keypoints as the body centre; velocity, speed and heading are not derived here --
see the SLEAP converter's module docstring for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd

from mosaic.core.params import Declared
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from mosaic.core.track_library.helpers import norm_hint


@dataclass(frozen=True)
class DlcIndividual:
    """Per-individual pose arrays extracted from a DeepLabCut file.

    Attributes:
        name: Individual identifier from the file (``"animal"`` for
            single-animal exports).
        bodyparts: Ordered bodypart names.
        x: ``(T, n_bodyparts)`` x-coordinates (pixels).
        y: ``(T, n_bodyparts)`` y-coordinates (pixels).
        likelihood: ``(T, n_bodyparts)`` detection confidence, or ``None`` when
            the source file has no ``likelihood`` column.
    """

    name: str
    bodyparts: list[str]
    x: np.ndarray
    y: np.ndarray
    likelihood: Optional[np.ndarray]


def _read_dlc_table(path: Path) -> pd.DataFrame:
    """Read a DLC ``.csv`` / ``.h5`` file into a MultiIndex-column DataFrame.

    The returned frame is indexed by the 0-based frame number and has a column
    MultiIndex whose last level is the coordinate (``x``/``y``/``likelihood``).
    The ``scorer`` level (always the first one in a DLC export) is dropped.
    """
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5", ".hdf"):
        obj = pd.read_hdf(path)
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"DLC HDF5 did not contain a DataFrame: {path}")
        df = obj
    elif suffix == ".csv":
        # Detect how many header rows the export has by inspecting the label
        # column (scorer / individuals / bodyparts / coords).
        probe = pd.read_csv(path, header=None, nrows=4, dtype=str)
        labels = [str(v).strip().lower() for v in probe.iloc[:, 0].tolist()]
        header_rows = [
            i
            for i, v in enumerate(labels)
            if v in {"scorer", "individuals", "bodyparts", "coords"}
        ]
        if not header_rows:
            raise ValueError(
                f"Not a DeepLabCut CSV (no scorer/bodyparts/coords header): {path}"
            )
        df = pd.read_csv(path, header=header_rows, index_col=0)
    else:
        raise ValueError(f"Unsupported DeepLabCut path (expect .csv or .h5): {path}")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"DeepLabCut file lacks MultiIndex columns: {path}")

    # Drop the leading scorer level (present in every DLC export).
    n_levels = df.columns.nlevels
    if n_levels in (3, 4):
        df = df.droplevel(0, axis=1)
    elif n_levels != 2:
        raise ValueError(
            f"Unexpected DeepLabCut column nesting ({n_levels} levels): {path}"
        )
    return df


def load_dlc(path: Path | str) -> list[DlcIndividual]:
    """Load a DeepLabCut file into per-individual pose arrays.

    Args:
        path: Path to a DeepLabCut ``.csv`` or ``.h5`` export.

    Returns:
        One :class:`DlcIndividual` per tracked individual, in file order.
    """
    path = Path(path)
    df = _read_dlc_table(path)

    # After dropping scorer, columns are either (individual, bodypart, coord)
    # for maDLC or (bodypart, coord) for single-animal.
    multi_animal = df.columns.nlevels == 3
    if multi_animal:
        individuals = list(dict.fromkeys(df.columns.get_level_values(0)))
    else:
        individuals = ["animal"]

    out: list[DlcIndividual] = []
    for indiv in individuals:
        sub = df[indiv] if multi_animal else df
        # Bodyparts in file order (the level just above coords).
        bodyparts = list(dict.fromkeys(sub.columns.get_level_values(0)))
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ls: list[np.ndarray] = []
        any_likelihood = False
        for bp in bodyparts:
            cols = sub[bp]
            coord_levels = {str(c).lower() for c in cols.columns}
            xs.append(pd.to_numeric(cols["x"], errors="coerce").to_numpy(dtype=float))
            ys.append(pd.to_numeric(cols["y"], errors="coerce").to_numpy(dtype=float))
            if "likelihood" in coord_levels:
                any_likelihood = True
                ls.append(
                    pd.to_numeric(cols["likelihood"], errors="coerce").to_numpy(
                        dtype=float
                    )
                )
            else:
                ls.append(np.full(len(sub), np.nan))
        out.append(
            DlcIndividual(
                name=str(indiv),
                bodyparts=[str(b) for b in bodyparts],
                x=np.column_stack(xs) if xs else np.empty((len(sub), 0)),
                y=np.column_stack(ys) if ys else np.empty((len(sub), 0)),
                likelihood=np.column_stack(ls) if any_likelihood else None,
            )
        )
    return out


def _individual_to_trex_df(
    indiv: DlcIndividual,
    animal_id: int,
    group: str,
    sequence: str,
    fps: float,
) -> pd.DataFrame:
    """Build a per-frame ``mosaic_v1`` DataFrame for one individual."""
    x = indiv.x
    y = indiv.y
    T = x.shape[0]
    n_lm = x.shape[1]

    # Centroid over keypoints (NaN-robust).
    cx = np.nanmean(x, axis=1) if n_lm else np.full(T, np.nan)
    cy = np.nanmean(y, axis=1) if n_lm else np.full(T, np.nan)

    data: dict[str, np.ndarray] = {
        "frame": np.arange(T, dtype=int),
        "time": np.arange(T, dtype=float) / fps,
        "id": np.full(T, animal_id, dtype=int),
        "X": cx,
        "Y": cy,
        "group": np.full(T, group),
        "sequence": np.full(T, sequence),
    }
    for k in range(n_lm):
        data[f"poseX{k}"] = x[:, k]
        data[f"poseY{k}"] = y[:, k]
    if indiv.likelihood is not None:
        for k in range(n_lm):
            data[f"poseP{k}"] = indiv.likelihood[:, k]

    return pd.DataFrame(data)


_FPS_DESCRIPTION = (
    "Frame rate used to derive time. Filled from the dataset's fps_default "
    "when the caller does not set it."
)


class DlcParams(TrackConvertParams):
    """Parameters for the DeepLabCut converter."""

    fps: Annotated[float, Declared(_FPS_DESCRIPTION)] = 30.0


@register_track_converter
class DlcConverter(TrackConverter[DlcParams]):
    """DeepLabCut ``.csv`` / ``.h5`` -> a ``mosaic_v1`` table."""

    src_format = "deeplabcut"
    # 0.2: see the SLEAP converter -- the same derived columns and the same
    # heading params are gone. Also serves Lightning Pose.
    version = "0.2"
    output_schema = "mosaic_v1"
    Params = DlcParams

    def convert(self, path: Path, params: DlcParams, hints: EntryHints) -> pd.DataFrame:
        individuals = load_dlc(path)
        group = norm_hint(hints.group) or ""
        sequence = norm_hint(hints.sequence) or path.stem

        frames = [
            _individual_to_trex_df(indiv, a, group, sequence, params.fps)
            for a, indiv in enumerate(individuals)
        ]
        out = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["frame", "time", "id", "group", "sequence"])
        )
        return out
