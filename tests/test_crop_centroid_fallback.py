"""A crop needs a centre, and a centroid-only tracker reports one.

``egocentric-crop`` averaged ``poseX0..poseX{pose_n-1}`` and took each column
only if present, so a table with no keypoints contributed nothing, every centre
came out NaN, and ``_extract_egocentric_crop`` turned each non-finite centre into
a solid ``background_color`` rectangle. Nothing warned: the run reported success,
the parquet and the crop files were written, and every crop was a blank frame.

That mattered because ``egocentric-crop`` is the sole input to all three identity
models, so a TREx-tracked dataset trained on blank images with no error anywhere
along the way.

The fix was blocked on the units question -- ``X``/``Y`` used to be centimetres
on TREx and pixels everywhere else. It is not any more: tracks are pixels, and
the one family that is not (``trex_v1``) is refused by name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
from mosaic.behavior.visualization_library.helpers import require_pixel_positions
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import write_tracks_row


def _centroid_table(n: int = 5) -> pd.DataFrame:
    """A schema-valid ``mosaic_v1`` table with no keypoints at all."""
    frame = np.arange(n, dtype=np.int64)
    return pd.DataFrame(
        {
            "frame": frame,
            "time": frame / 30.0,
            "id": np.zeros(n, dtype=np.int64),
            "group": [""] * n,
            "sequence": ["seq"] * n,
            "X": 10.0 + frame,
            "Y": 20.0 + frame,
            "ANGLE": np.full(n, 0.1),
        }
    )


def _posed_table(n: int = 5, n_keypoints: int = 7) -> pd.DataFrame:
    table = _centroid_table(n)
    for k in range(n_keypoints):
        table[f"poseX{k}"] = table["X"] + k
        table[f"poseY{k}"] = table["Y"] + k
    return table


def test_a_table_without_keypoints_centres_on_the_body_centre() -> None:
    """The blank-crop defect: these used to come back all-NaN."""
    table = _centroid_table()
    _, cx, cy = EgocentricCrop(params={"angle_col": "ANGLE"})._precompute_geometry(
        table, False
    )
    assert np.isfinite(cx).all()
    assert np.allclose(cx, table["X"])
    assert np.allclose(cy, table["Y"])


def test_keypoints_still_win_where_they_exist() -> None:
    """The counter-test: the fallback is reached only when there is nothing else.

    Seven keypoints at ``X + 0 .. X + 6`` average to ``X + 3``, so a centre equal
    to ``X`` would mean the fallback had fired over real keypoints.
    """
    table = _posed_table()
    _, cx, _ = EgocentricCrop(params={"angle_col": "ANGLE"})._precompute_geometry(
        table, False
    )
    assert np.allclose(cx, table["X"] + 3.0)


def test_an_explicit_pose_index_refuses_by_name() -> None:
    """It used to be a bare ``KeyError: 'poseX0'`` from the column lookup."""
    with pytest.raises(ValueError, match="center_mode"):
        _ = EgocentricCrop(params={"center_mode": "pose0"})._precompute_geometry(
            _centroid_table(), False
        )


def test_no_derivable_centre_at_all_refuses() -> None:
    """Never a full run of background-filled frames in silence.

    A per-frame dropout is ordinary and still yields a blank frame for that
    frame. Every frame blank is not a dropout, it is the wrong table.
    """
    table = _centroid_table()
    table["X"] = np.nan
    table["Y"] = np.nan
    with pytest.raises(ValueError, match="could not derive a centre"):
        _ = EgocentricCrop(params={"angle_col": "ANGLE"})._precompute_geometry(
            table, False
        )


def _dataset_recording(base: Path, std_format: str) -> Dataset:
    """A dataset holding one entry whose table records *std_format*."""
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={"tracks": str(base / "tracks"), "tracks_raw": str(base / "tracks_raw")},
    )
    ds.ensure_roots()
    ds.save()
    out = ds.get_root("tracks") / "seq.parquet"
    _centroid_table().to_parquet(out)
    write_tracks_row(
        ds,
        run_id="convert-x.0.1-aaaaaaaaaa",
        group="",
        sequence="seq",
        out_path=out,
        producer="convert-x",
        std_format=std_format,
        n_rows=5,
    )
    return ds


def test_a_pixel_table_permits_the_fallback(tmp_path: Path) -> None:
    require_pixel_positions(_dataset_recording(tmp_path, "mosaic_v1"), "", "seq", "who")


def test_a_centimetre_table_refuses_and_names_the_fix(tmp_path: Path) -> None:
    """Cropping at centimetre coordinates lands elsewhere in the frame, silently."""
    ds = _dataset_recording(tmp_path, "trex_v1")
    with pytest.raises(ValueError, match="upgrade-tracks"):
        require_pixel_positions(ds, "", "seq", "who")


def test_an_unrecorded_schema_is_read_as_the_legacy_one(tmp_path: Path) -> None:
    """Matching the scope check: a keypoint-less table from that era is TREx."""
    ds = _dataset_recording(tmp_path, "")
    with pytest.raises(ValueError, match="upgrade-tracks"):
        require_pixel_positions(ds, "", "seq", "who")
