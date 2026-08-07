"""Centimetres are a claim about the world, so they are a recorded step.

A pixel is what the camera measured. A centimetre is a pixel plus a scale, and
the scale is knowledge about the rig that no probe recovers. Keeping it out of
the track table is what stops a table being silently wrong about its own units --
the failure this whole arrangement removed, where one tracker wrote centimetres
into the columns every other tracker filled with pixels.

So the conversion is a feature: the scale enters a run identifier, and two
calibrations are two addressable results rather than one column that changed
meaning. And an uncalibrated dataset refuses, because ``1.0`` is a scale rather
than an absence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.scale_to_cm import ScaleToCm, scalable_columns
from mosaic.core.dataset import Dataset, new_dataset_manifest

from .conftest import add_media_sequence


def _table(sequence: str = "seq_a", n: int = 4) -> pd.DataFrame:
    position = np.linspace(0.0, 10.0, n)
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "time": np.arange(n, dtype=float) / 30.0,
            "id": np.zeros(n, dtype=np.int64),
            "group": [""] * n,
            "sequence": [sequence] * n,
            "X": position,
            "Y": position,
            "poseX0": position,
            "poseY0": position,
            "poseP0": np.full(n, 0.9),
            "ANGLE": np.full(n, 0.5),
        }
    )


# --- which columns carry a length --------------------------------------------


def test_positions_and_keypoints_scale() -> None:
    chosen = scalable_columns(["X", "Y", "X#wcentroid", "poseX0", "poseY3", "bbox_x1"])
    assert set(chosen) == {"X", "Y", "X#wcentroid", "poseX0", "poseY3", "bbox_x1"}


def test_angles_counts_and_identifiers_do_not_scale() -> None:
    """Multiplying a heading by a distance ratio yields a plausible wrong number."""
    chosen = scalable_columns(
        ["ANGLE", "ANGULAR_V", "frame", "time", "id", "group", "sequence", "det_cls"]
    )
    assert chosen == []


def test_a_keypoint_confidence_does_not_scale() -> None:
    """``poseP`` shares its prefix with the coordinates and carries no length."""
    assert scalable_columns(["poseP0", "poseP1"]) == []


# --- the conversion ----------------------------------------------------------


def test_an_explicit_scale_converts_every_length_column() -> None:
    out = ScaleToCm(params={"cm_per_pixel": 0.25}).apply(_table())

    assert out["X_cm"].to_numpy() == pytest.approx(np.linspace(0.0, 2.5, 4))
    assert out["poseX0_cm"].to_numpy() == pytest.approx(np.linspace(0.0, 2.5, 4))
    # Untouched, and not copied through: the source table stays the authority on
    # what was measured.
    assert "ANGLE_cm" not in out.columns
    assert "X" not in out.columns


def test_the_suffix_is_configurable() -> None:
    out = ScaleToCm(params={"cm_per_pixel": 2.0, "suffix": "_mm"}).apply(_table())
    assert "X_mm" in out.columns


def test_naming_a_column_the_table_lacks_refuses() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        _ = ScaleToCm(params={"cm_per_pixel": 1.0, "columns": ["nope"]}).apply(_table())


def test_a_table_with_nothing_to_scale_refuses() -> None:
    frame = _table()[["frame", "time", "id", "group", "sequence", "ANGLE"]]
    with pytest.raises(ValueError, match="no length-bearing column"):
        _ = ScaleToCm(params={"cm_per_pixel": 1.0}).apply(frame)


def test_a_non_positive_scale_refuses() -> None:
    with pytest.raises(ValueError, match="not a usable scale"):
        _ = ScaleToCm(params={"cm_per_pixel": 0.0}).apply(_table())


def test_the_scale_is_part_of_the_run_identity() -> None:
    """Two calibrations are two results, not one column that quietly changed."""
    a = ScaleToCm(params={"cm_per_pixel": 0.25})
    b = ScaleToCm(params={"cm_per_pixel": 0.5})
    assert a.params.identity_dump() != b.params.identity_dump()


# --- reading the scale from the dataset --------------------------------------


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="scale", base_dir=tmp_path / "dataset")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    add_media_sequence(ds, "seq_a", videos=("a.mp4",), frames=3)
    return ds


def test_the_scale_comes_from_the_media_row(tmp_path: Path) -> None:
    """Where a calibration belongs: beside the video it describes."""
    ds = _dataset(tmp_path)
    _ = ds.set_media_calibration(0.25)

    feature = ScaleToCm()
    feature.bind_dataset(ds)
    out = feature.apply(_table())

    assert out["X_cm"].to_numpy() == pytest.approx(np.linspace(0.0, 2.5, 4))


def test_an_uncalibrated_sequence_refuses_rather_than_assuming_one(
    tmp_path: Path,
) -> None:
    """The refusal this feature exists to make. ``1.0`` is a scale, not silence."""
    ds = _dataset(tmp_path)
    feature = ScaleToCm()
    feature.bind_dataset(ds)

    with pytest.raises(ValueError, match="no recorded cm_per_pixel"):
        _ = feature.apply(_table())


def test_an_explicit_scale_overrides_the_recorded_one(tmp_path: Path) -> None:
    """A caller who states a scale has said something the dataset does not know."""
    ds = _dataset(tmp_path)
    _ = ds.set_media_calibration(0.25)

    feature = ScaleToCm(params={"cm_per_pixel": 1.0})
    feature.bind_dataset(ds)
    out = feature.apply(_table())

    assert out["X_cm"].to_numpy() == pytest.approx(np.linspace(0.0, 10.0, 4))


def test_running_unbound_without_a_scale_says_so() -> None:
    with pytest.raises(ValueError, match="not bound to a dataset"):
        _ = ScaleToCm().apply(_table())


def test_a_frame_spanning_two_sequences_refuses() -> None:
    """A calibration is per-recording, so a mixed frame has no single scale."""
    mixed = pd.concat([_table("seq_a"), _table("seq_b")], ignore_index=True)
    feature = ScaleToCm()
    feature.bind_dataset(object())
    with pytest.raises(ValueError, match="one sequence per call"):
        _ = feature.apply(mixed)
