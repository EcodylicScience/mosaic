"""A crop needs a centre, and a centroid-only tracker reports one.

``egocentric-crop`` averaged ``poseX0..poseX{pose_n-1}`` and took each column
only if present, so a table with no keypoints contributed nothing, every centre
came out NaN, and ``extract_egocentric_crop`` turned each non-finite center into
a solid ``background_color`` rectangle. Nothing warned: the run reported success,
the parquet and the crop files were written, and every crop was a blank frame.

That mattered because ``egocentric-crop`` is the sole input to all three identity
models, so a TREx-tracked dataset trained on blank images with no error anywhere
along the way.

The fix was blocked on the units question -- ``X``/``Y`` used to be centimetres
on TREx and pixels everywhere else. It is not any more: every schema family
declares its space, and a crop or an overlay refuses a table known to be in
centimetres (``trex_v1``, ``mosaic_cm_v1``). A schema whose space cannot be
read -- one declaring none, or one registered only by code another process ran --
is drawn as pixels with a warning, because refusing would stop a legitimate
render over a fact the rendering process cannot see.

``center_mode="xy"`` is that fallback asked for outright: a request for the body
centre that never consults pose even where pose exists, and that therefore
carries the units refusal unconditionally rather than only where keypoints are
absent. It exists because averaging *whichever* keypoints happened to be
detected on a row moves the crop centre between rows, which reads downstream as
appearance noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
from mosaic.behavior.visualization_library.helpers import require_pixel_positions
from mosaic.behavior.visualization_library.interaction_crop import (
    InteractionCropPipeline,
)
from mosaic.behavior.visualization_library.playback import build_overlay
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.tracks_index import write_tracks_row
from mosaic.core.schema import TRACK_SCHEMAS, TrackSchema, register_track_schema


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


def _dataset_recording(
    base: Path, std_format: str, table: pd.DataFrame | None = None
) -> Dataset:
    """A dataset holding one entry whose table records *std_format*."""
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={"tracks": str(base / "tracks"), "tracks_raw": str(base / "tracks_raw")},
    )
    ds.ensure_roots()
    ds.save()
    out = ds.get_root("tracks") / "seq.parquet"
    (_centroid_table() if table is None else table).to_parquet(out)
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


def test_a_centimetre_table_without_a_pixel_factor_refuses(tmp_path: Path) -> None:
    """``mosaic_cm_v1`` used to pass: the guard only knew the ``trex_v1`` family."""
    ds = _dataset_recording(tmp_path, "mosaic_cm_v1")
    with pytest.raises(ValueError, match="centimetres"):
        require_pixel_positions(ds, "", "seq", "who")


def test_an_unregistered_schema_warns_and_is_drawn_as_pixels(tmp_path: Path) -> None:
    """The process that renders is often not the one that registered the schema."""
    ds = _dataset_recording(tmp_path, "never_registered_v1")
    with pytest.warns(UserWarning, match="never_registered_v1"):
        require_pixel_positions(ds, "", "seq", "who")


def test_a_schema_declaring_no_space_warns_and_is_drawn_as_pixels(
    tmp_path: Path,
) -> None:
    register_track_schema(TrackSchema(name="spaceless_root", required={"frame"}))
    try:
        ds = _dataset_recording(tmp_path, "spaceless_root")
        with pytest.warns(UserWarning, match="spaceless_root"):
            require_pixel_positions(ds, "", "seq", "who")
    finally:
        _ = TRACK_SCHEMAS.pop("spaceless_root", None)


def test_the_schema_of_the_table_in_hand_wins_over_the_index(tmp_path: Path) -> None:
    """A feature knows which table it was handed; the index row may be another."""
    ds = _dataset_recording(tmp_path, "trex_v1")
    require_pixel_positions(ds, "", "seq", "who", schema="mosaic_v1")
    with pytest.raises(ValueError, match="centimetres"):
        require_pixel_positions(ds, "", "seq", "who", schema="mosaic_cm_v1")


def _add_pixel_variant(ds: Dataset) -> str:
    """A second variant of the entry, in pixels; returns its run id."""
    run_id = "convert-y.0.1-bbbbbbbbbb"
    out = ds.get_root("tracks") / "seq_pixels.parquet"
    _centroid_table().to_parquet(out)
    write_tracks_row(
        ds,
        run_id=run_id,
        group="",
        sequence="seq",
        out_path=out,
        producer="convert-y",
        std_format="mosaic_v1",
        n_rows=5,
    )
    return run_id


def test_the_guard_judges_the_variant_that_was_read(tmp_path: Path) -> None:
    """Two variants of one entry: the first index row is not necessarily the one."""
    ds = _dataset_recording(tmp_path, "mosaic_cm_v1")
    pixels = _add_pixel_variant(ds)
    require_pixel_positions(ds, "", "seq", "who", tracks_run_id=pixels)
    with pytest.raises(ValueError, match="centimetres"):
        require_pixel_positions(
            ds, "", "seq", "who", tracks_run_id="convert-x.0.1-aaaaaaaaaa"
        )
    with pytest.raises(ValueError):
        require_pixel_positions(ds, "", "seq", "who")


@pytest.mark.parametrize(
    ("crop_type", "run"),
    [(EgocentricCrop, "transform"), (InteractionCropPipeline, "apply")],
)
def test_a_posed_centimetre_table_refuses_in_the_default_mode(
    tmp_path: Path,
    crop_type: type[EgocentricCrop] | type[InteractionCropPipeline],
    run: str,
) -> None:
    """Keypoints are no safer than ``X``: in ``trex_v1`` both are centimetres.

    The gate used to run only without keypoints or under ``"xy"``, so a
    ``trex_v1`` table -- which always has keypoints -- was cropped around its
    centimetre keypoints with nothing raising.
    """
    ds = _dataset_recording(tmp_path, "trex_v1", table=_posed_table())
    crop = crop_type(params={"angle_col": "ANGLE"})
    crop.bind_dataset(ds)
    with pytest.raises(ValueError, match="upgrade-tracks"):
        _ = getattr(crop, run)(_posed_table())


@pytest.mark.parametrize(
    ("crop_type", "run"),
    [(EgocentricCrop, "transform"), (InteractionCropPipeline, "apply")],
)
def test_a_crop_reads_the_schema_off_its_scope(
    tmp_path: Path,
    crop_type: type[EgocentricCrop] | type[InteractionCropPipeline],
    run: str,
) -> None:
    """The scope names the schema of the table the run resolved for the entry."""
    ds = _dataset_recording(tmp_path, "mosaic_v1", table=_posed_table())
    crop = crop_type(params={"angle_col": "ANGLE"})
    crop.bind_dataset(ds)
    crop.set_scope(ResolvedScope(tracks_schemas={("", "seq"): "mosaic_cm_v1"}))
    with pytest.raises(ValueError, match="centimetres"):
        _ = getattr(crop, run)(_posed_table())


@pytest.mark.parametrize("std_format", ["trex_v1", "mosaic_cm_v1"])
def test_the_overlay_refuses_a_posed_centimetre_table(
    tmp_path: Path, std_format: str
) -> None:
    """The overlay's gate also ran only for keypoint-less tables."""
    ds = _dataset_recording(tmp_path, std_format, table=_posed_table())
    with pytest.raises(ValueError, match="centimetres"):
        _ = build_overlay(ds, "", "seq", {}, label_kind=None)


def test_the_overlay_judges_the_pinned_variant(tmp_path: Path) -> None:
    """A pinned pixel variant draws although another variant of the entry is cm."""
    ds = _dataset_recording(tmp_path, "mosaic_cm_v1")
    pixels = _add_pixel_variant(ds)
    _ = build_overlay(ds, "", "seq", {}, label_kind=None, tracks_run_id=pixels)
    with pytest.raises(ValueError, match="centimetres"):
        _ = build_overlay(
            ds,
            "",
            "seq",
            {},
            label_kind=None,
            tracks_run_id="convert-x.0.1-aaaaaaaaaa",
        )


# ---------------------------------------------------------------------------
# center_mode="xy" -- the fallback asked for by name
# ---------------------------------------------------------------------------


def test_xy_never_consults_keypoints() -> None:
    """The whole point of "xy": pose is present and deliberately ignored.

    Seven keypoints average to ``X + 3`` -- that is what ``"default"`` returns
    (see ``test_keypoints_still_win_where_they_exist``). ``"xy"`` must return
    ``X``. Asserting against a keypoint-less table instead would pass even if
    ``"xy"`` were silently aliased to ``"default"``.
    """
    table = _posed_table()
    _, cx, cy = EgocentricCrop(
        params={"angle_col": "ANGLE", "center_mode": "xy"}
    )._precompute_geometry(table, False)
    assert np.allclose(cx, table["X"])
    assert np.allclose(cy, table["Y"])


def test_xy_without_a_body_centre_refuses_by_name() -> None:
    """Not a bare ``KeyError`` out of the column lookup."""
    table = _posed_table().drop(columns=["X", "Y"])
    with pytest.raises(ValueError, match="Keypoints are never consulted"):
        _ = EgocentricCrop(
            params={"angle_col": "ANGLE", "center_mode": "xy"}
        )._precompute_geometry(table, False)


def test_xy_does_not_quietly_fall_back_to_the_pose_mean() -> None:
    """The same table succeeds under ``"default"``; the pair is what discriminates."""
    table = _posed_table()
    table["X"] = np.nan
    table["Y"] = np.nan
    with pytest.raises(ValueError, match="could not derive a centre"):
        _ = EgocentricCrop(
            params={"angle_col": "ANGLE", "center_mode": "xy"}
        )._precompute_geometry(table, False)


def test_the_refusal_names_every_accepted_mode() -> None:
    """Catches the half-change: branch added, enumeration left stale."""
    with pytest.raises(ValueError, match="xy"):
        _ = EgocentricCrop(
            params={"angle_col": "ANGLE", "center_mode": "banana"}
        )._precompute_geometry(_posed_table(), False)


def test_xy_carries_the_units_refusal_onto_a_posed_table(tmp_path: Path) -> None:
    """The units gate used to fire only for keypoint-less tables.

    ``trex_v1`` *requires* pose columns, so a trex_v1 entry always had them and
    always skipped the gate. Under ``"xy"`` the crop reads ``X``/``Y``
    regardless, so centimetres would be read as pixels with nothing raising.
    """
    ds = _dataset_recording(tmp_path, "trex_v1", table=_posed_table())
    crop = EgocentricCrop(params={"angle_col": "ANGLE", "center_mode": "xy"})
    crop.bind_dataset(ds)
    with pytest.raises(ValueError, match="upgrade-tracks"):
        _ = crop.transform(_posed_table())


def test_interaction_crop_xy_never_consults_keypoints() -> None:
    """The sibling feature's ``center_mode`` is a verbatim copy; keep them equal."""
    table = _posed_table()
    _, cx, cy = InteractionCropPipeline(
        params={"angle_col": "ANGLE", "center_mode": "xy"}
    )._precompute_geometry(table, False)
    assert np.allclose(cx, table["X"])
    assert np.allclose(cy, table["Y"])
