"""Tests for the generic DeepLabCut track converter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_library.deeplabcut import _dlc_converter, load_dlc

_BODYPARTS = ["snout", "midbody", "tailtip"]


def _write_single_animal_csv(path: Path, n_frames: int = 20) -> np.ndarray:
    """Write a single-animal DLC CSV (scorer/bodyparts/coords header).

    Returns the (n_frames, n_bodyparts, 3) array of [x, y, likelihood] written.
    """
    rng = np.random.default_rng(0)
    vals = rng.uniform(0, 100, size=(n_frames, len(_BODYPARTS), 3))
    vals[:, :, 2] = rng.uniform(0.5, 1.0, size=(n_frames, len(_BODYPARTS)))

    header_scorer = ["scorer"]
    header_bp = ["bodyparts"]
    header_coord = ["coords"]
    for bp in _BODYPARTS:
        header_scorer += ["DLC_model"] * 3
        header_bp += [bp] * 3
        header_coord += ["x", "y", "likelihood"]

    lines = [
        ",".join(header_scorer),
        ",".join(header_bp),
        ",".join(header_coord),
    ]
    for i in range(n_frames):
        row = [str(i)]
        for b in range(len(_BODYPARTS)):
            row += [f"{vals[i, b, c]:.6f}" for c in range(3)]
        lines.append(",".join(row))
    path.write_text("\n".join(lines))
    return vals


def _write_multi_animal_csv(path: Path, n_frames: int = 15) -> int:
    """Write a 2-individual maDLC CSV (scorer/individuals/bodyparts/coords)."""
    individuals = ["fish0", "fish1"]
    rng = np.random.default_rng(1)
    vals = rng.uniform(0, 100, size=(n_frames, len(individuals), len(_BODYPARTS), 3))

    rows: list[list[str]] = [
        ["scorer"],
        ["individuals"],
        ["bodyparts"],
        ["coords"],
    ]
    for ind in individuals:
        for bp in _BODYPARTS:
            rows[0] += ["DLC_model"] * 3
            rows[1] += [ind] * 3
            rows[2] += [bp] * 3
            rows[3] += ["x", "y", "likelihood"]
    lines = [",".join(r) for r in rows]
    for i in range(n_frames):
        row = [str(i)]
        for a in range(len(individuals)):
            for b in range(len(_BODYPARTS)):
                row += [f"{vals[i, a, b, c]:.6f}" for c in range(3)]
        lines.append(",".join(row))
    path.write_text("\n".join(lines))
    return len(individuals)


def test_load_dlc_single_animal(tmp_path: Path) -> None:
    csv = tmp_path / "single.csv"
    vals = _write_single_animal_csv(csv, n_frames=20)

    individuals = load_dlc(csv)
    assert len(individuals) == 1
    indiv = individuals[0]
    assert indiv.bodyparts == _BODYPARTS
    assert indiv.x.shape == (20, len(_BODYPARTS))
    assert indiv.likelihood is not None
    np.testing.assert_allclose(indiv.x[:, 0], vals[:, 0, 0], rtol=1e-5)
    np.testing.assert_allclose(indiv.likelihood[:, 1], vals[:, 1, 2], rtol=1e-5)


def test_dlc_converter_single_animal_schema(tmp_path: Path) -> None:
    csv = tmp_path / "single.csv"
    _write_single_animal_csv(csv, n_frames=20)

    df = _dlc_converter(csv, {"group": "g1", "sequence": "s1", "fps": 50.0})

    # Required trex_v1 columns + pose prefixes present.
    _, report = ensure_track_schema(df, "trex_v1", strict=True, source=str(csv))
    assert report["missing_required"] == []
    assert report["missing_prefixes"] == []

    assert len(df) == 20
    assert set(df["id"]) == {0}
    assert (df["group"] == "g1").all()
    assert (df["sequence"] == "s1").all()
    # 3 keypoints -> poseX0..2, poseY0..2, poseP0..2
    for k in range(len(_BODYPARTS)):
        assert f"poseX{k}" in df.columns
        assert f"poseY{k}" in df.columns
        assert f"poseP{k}" in df.columns
    # time derived from fps
    np.testing.assert_allclose(df["time"].to_numpy(), np.arange(20) / 50.0)
    # centroid equals mean of keypoint x's
    pose_x = df[[f"poseX{k}" for k in range(len(_BODYPARTS))]].to_numpy()
    np.testing.assert_allclose(
        df["X"].to_numpy(), np.nanmean(pose_x, axis=1), rtol=1e-6
    )


def test_dlc_converter_multi_animal(tmp_path: Path) -> None:
    csv = tmp_path / "multi.csv"
    n = _write_multi_animal_csv(csv, n_frames=15)

    individuals = load_dlc(csv)
    assert len(individuals) == n

    df = _dlc_converter(csv, {"group": "g", "sequence": "rec", "fps": 30.0})
    assert set(df["id"]) == set(range(n))
    # Each individual contributes n_frames rows.
    assert len(df) == 15 * n


def test_dlc_converter_csv_h5_roundtrip(tmp_path: Path) -> None:
    """A DLC export saved as HDF5 yields the same poses as the CSV form."""
    pytest.importorskip("tables")
    csv = tmp_path / "single.csv"
    _write_single_animal_csv(csv, n_frames=12)

    indiv = load_dlc(csv)[0]
    # Build the equivalent maDLC-less HDF5 with a (scorer, bodypart, coord) index.
    cols = pd.MultiIndex.from_tuples(
        [("DLC_model", bp, c) for bp in _BODYPARTS for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    flat = np.empty((12, len(_BODYPARTS) * 3))
    for b in range(len(_BODYPARTS)):
        flat[:, b * 3 + 0] = indiv.x[:, b]
        flat[:, b * 3 + 1] = indiv.y[:, b]
        assert indiv.likelihood is not None
        flat[:, b * 3 + 2] = indiv.likelihood[:, b]
    h5 = tmp_path / "single.h5"
    pd.DataFrame(flat, columns=cols).to_hdf(h5, key="df", mode="w")

    df_csv = _dlc_converter(csv, {"fps": 30.0})
    df_h5 = _dlc_converter(h5, {"fps": 30.0})
    np.testing.assert_allclose(
        df_csv[["poseX0", "poseY1", "poseP2"]].to_numpy(),
        df_h5[["poseX0", "poseY1", "poseP2"]].to_numpy(),
        rtol=1e-6,
    )
