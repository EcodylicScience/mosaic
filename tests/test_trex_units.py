"""TRex tables are centimetres and a head position; mosaic tables are neither.

Two conversions happen on the way out of a TRex export, and both used to not
happen at all.

**Units.** TRex scales its positional output by ``cm_per_pixel``, so ``X`` and
``SPEED`` arrive in centimetres while the keypoints beside them are pixels. One
table, two coordinate systems, and nothing recording which column is which. It
went unnoticed because ``cm_per_pixel`` defaults to 1, where the two are the
same number -- the error appears the first time somebody calibrates, and then
every distance and threshold in the dataset is wrong by a constant nobody wrote
down.

**Landmark.** In TRex the ``#`` suffix names the *source* of a value and the
bare name means ``#head``. So TRex's ``X`` is a head position, present only
where posture was calculated, while every other tracker mosaic supports puts a
body centre in ``X``. A feature reading ``X`` across trackers was comparing a
head to a centroid.

The factor is read from the file rather than from mosaic's own parameter,
because they are not the same number: TRex substitutes
``meta_real_width / video_width`` when the setting is unset, and extra settings
can override it without reaching mosaic. What TRex writes is what TRex applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.core.track_converter import EntryHints, TrackConvertParams
from mosaic.core.track_library.trex import (
    CALIBRATION_COLUMN,
    MissingBodyCentreError,
    MissingTrexCalibrationError,
    TrexCalibrationConflictError,
    TrexNpzCmConverter,
    TrexNpzConverter,
    TrexNpzScaledConverter,
    TrexScaledNpzParams,
    UnknownTrexUnitsError,
    load_npz_to_df,
    calibration_from_frame,
    unscale_to_pixels,
)

from .conftest import write_trex_npz


def _convert(path: Path) -> pd.DataFrame:
    return TrexNpzConverter().convert(
        path, TrackConvertParams(), EntryHints(group="", sequence="seq")
    )


# --- the calibration survives the flatten ------------------------------------


def test_a_one_element_field_reaches_the_frame_nan_padded(tmp_path: Path) -> None:
    """The behavior every recovery path here depends on, pinned before it moves.

    TRex writes ``cm_per_pixel`` as a one-element array rather than a scalar, so
    the flattener pads it to full length with NaN instead of broadcasting it.
    The value therefore sits in the first row and nowhere else -- which is why
    it is read by dropping NaN, and why an already-converted parquet still
    carries the factor that converted it.

    This is undocumented upstream behavior of ``load_npz_to_df``. A tidy-up
    there -- narrowing to schema columns, dropping short arrays -- would destroy
    the only record of the calibration that does not depend on mosaic having
    been the thing that ran TRex.
    """
    path = tmp_path / "seq_fish0.npz"
    write_trex_npz(path, n=5, cm_per_pixel=0.25)

    frame = load_npz_to_df(path)
    values = frame[CALIBRATION_COLUMN].to_list()
    assert values[0] == pytest.approx(0.25)
    assert all(np.isnan(v) for v in values[1:])


def test_the_calibration_is_read_despite_the_padding(tmp_path: Path) -> None:
    path = tmp_path / "seq_fish0.npz"
    write_trex_npz(path, n=5, cm_per_pixel=0.25)
    assert calibration_from_frame(load_npz_to_df(path)) == pytest.approx(0.25)


def test_a_frame_without_the_field_reports_none_rather_than_one() -> None:
    """ "Does not say" and "says one" are different, and only one is safe."""
    assert calibration_from_frame(pd.DataFrame({"X": [1.0]})) is None


def test_disagreeing_individuals_refuse_rather_than_average() -> None:
    """A merged sequence exported under two calibrations has no single answer."""
    frame = pd.DataFrame({CALIBRATION_COLUMN: [0.5, np.nan, 0.25, np.nan]})
    with pytest.raises(ValueError, match="different"):
        _ = calibration_from_frame(frame)


# --- the unscale -------------------------------------------------------------


def test_unscaling_at_one_changes_nothing() -> None:
    frame = pd.DataFrame({"X": [1.0, 2.0], "SPEED": [3.0, 4.0]})
    assert unscale_to_pixels(frame, 1.0).equals(frame)


def test_length_columns_divide_and_others_do_not() -> None:
    frame = pd.DataFrame(
        {
            "X": [1.0],
            "X#wcentroid": [2.0],
            "SPEED": [4.0],
            "midline_x": [8.0],
            "midline_segment_length": [16.0],
            # Verified against TRex itself: this one's centimetre conversion is
            # commented out upstream, so it is already pixels despite the name.
            "midline_length": [32.0],
            # An angle, despite reading like an offset.
            "MIDLINE_OFFSET": [0.5],
            "ANGLE": [0.25],
            "poseX0": [64.0],
            "num_pixels": [128.0],
        }
    )
    out = unscale_to_pixels(frame, 0.5)

    assert out["X"].to_list() == [2.0]
    assert out["X#wcentroid"].to_list() == [4.0]
    assert out["SPEED"].to_list() == [8.0]
    assert out["midline_x"].to_list() == [16.0]
    assert out["midline_segment_length"].to_list() == [32.0]

    assert out["midline_length"].to_list() == [32.0]
    assert out["MIDLINE_OFFSET"].to_list() == [0.5]
    assert out["ANGLE"].to_list() == [0.25]
    assert out["poseX0"].to_list() == [64.0]
    assert out["num_pixels"].to_list() == [128.0]


def test_an_unclassified_numeric_column_refuses_when_calibrated() -> None:
    """``output_fields`` is open, so guessing a unit would be a silent error."""
    frame = pd.DataFrame({"X": [1.0], "some_future_field": [2.0]})
    with pytest.raises(UnknownTrexUnitsError, match="some_future_field"):
        _ = unscale_to_pixels(frame, 0.5)


def test_an_unclassified_column_is_fine_when_uncalibrated() -> None:
    """At one, no column is touched, so no unit can be got wrong."""
    frame = pd.DataFrame({"X": [1.0], "some_future_field": [2.0]})
    assert unscale_to_pixels(frame, 1.0).equals(frame)


def test_a_non_numeric_column_is_never_unclassified() -> None:
    """A label carries no unit, whatever it is called."""
    frame = pd.DataFrame({"X": [1.0], "camera": ["cam0"], "missing": [True]})
    out = unscale_to_pixels(frame, 0.5)
    assert out["camera"].to_list() == ["cam0"]


# --- the whole conversion ----------------------------------------------------


def test_a_calibrated_export_converts_to_the_same_pixels_as_an_uncalibrated_one(
    tmp_path: Path,
) -> None:
    """The round trip that is the whole point: cm in, the original pixels out.

    The same underlying motion exported at two calibrations must convert to one
    table, because pixels are what the camera saw and the calibration is only
    how TRex chose to report it.
    """
    plain = tmp_path / "a_fish0.npz"
    scaled = tmp_path / "b_fish0.npz"
    pixels_x = np.linspace(0.0, 100.0, 6)
    pixels_y = np.linspace(100.0, 0.0, 6)

    write_trex_npz(
        plain,
        n=6,
        cm_per_pixel=1.0,
        **{"X#wcentroid": pixels_x, "Y#wcentroid": pixels_y},
    )
    write_trex_npz(
        scaled,
        n=6,
        cm_per_pixel=0.25,
        **{"X#wcentroid": pixels_x * 0.25, "Y#wcentroid": pixels_y * 0.25},
    )

    assert _convert(scaled)["X"].to_numpy() == pytest.approx(
        _convert(plain)["X"].to_numpy()
    )
    assert _convert(scaled)["X"].to_numpy() == pytest.approx(pixels_x)


def test_x_becomes_the_body_centre_and_the_head_is_preserved(tmp_path: Path) -> None:
    """TRex's bare pair is the head; every other tracker's X is a body centre."""
    path = tmp_path / "seq_fish0.npz"
    head_x = np.full(6, 7.0)
    centre_x = np.full(6, 3.0)
    write_trex_npz(
        path,
        n=6,
        X=head_x,
        Y=head_x,
        **{"X#wcentroid": centre_x, "Y#wcentroid": centre_x},
    )

    out = _convert(path)
    assert out["X"].to_numpy() == pytest.approx(centre_x)
    assert out["X#head"].to_numpy() == pytest.approx(head_x)
    # Kept as well, so TRex-aware code reading it keeps resolving.
    assert out["X#wcentroid"].to_numpy() == pytest.approx(centre_x)


def test_an_export_without_a_body_centre_refuses(tmp_path: Path) -> None:
    """The bare pair is not a substitute -- promoting it would put a head in X."""
    path = tmp_path / "seq_fish0.npz"
    np.savez(
        path,
        frame=np.arange(4),
        time=np.arange(4) / 30.0,
        id=np.array([0]),
        cm_per_pixel=np.array([1.0]),
        X=np.zeros(4),
        Y=np.zeros(4),
        poseX0=np.zeros(4),
        poseY0=np.zeros(4),
    )
    with pytest.raises(MissingBodyCentreError, match="wcentroid"):
        _ = _convert(path)


def test_an_export_without_a_calibration_refuses(tmp_path: Path) -> None:
    """Assuming 1.0 would read a calibrated file's centimetres as pixels."""
    path = tmp_path / "seq_fish0.npz"
    np.savez(
        path,
        frame=np.arange(4),
        time=np.arange(4) / 30.0,
        id=np.array([0]),
        X=np.zeros(4),
        Y=np.zeros(4),
        poseX0=np.zeros(4),
        poseY0=np.zeros(4),
        **{"X#wcentroid": np.zeros(4), "Y#wcentroid": np.zeros(4)},
    )
    with pytest.raises(MissingTrexCalibrationError, match=CALIBRATION_COLUMN):
        _ = _convert(path)


def test_the_converter_declares_the_trex_superset_schema() -> None:
    """It emits SPEED and ANGLE, which the tracker-neutral base forbids."""
    assert TrexNpzConverter.output_schema == "trex_v2"


# --- the per-tracklet exports are a different axis ---------------------------


def _with_tracklet_arrays(path: Path, *, n: int = 8) -> None:
    """A real export's tracklet trio, at the shapes TRex actually writes.

    Taken from a measured file: ``tracklets`` is ``(n_tracklets, 2)`` and
    ``tracklet_vxys`` ``(n_frames_in_tracklets, 4)``, both shorter than the frame
    axis, while ``tracklet_id`` is one value per frame.
    """
    write_trex_npz(
        path,
        n=n,
        cm_per_pixel=0.03,
        tracklet_id=np.full(n, 1.266e14),
        tracklets=np.array([[0, 3], [4, 7]], dtype=np.uint32),
        tracklet_vxys=np.array(
            [[1.0, -60.0, -120.0, 134.164], [2.0, -109.9, -300.0, 319.504]],
            dtype=np.float32,
        ),
    )


def test_a_calibrated_export_with_tracklet_arrays_converts(tmp_path: Path) -> None:
    """The failure a real run hit: four NPZ refused, nothing published.

    ``UnknownTrexUnitsError`` fired on ``tracklets_0`` and ``tracklet_vxys_0..3``
    -- names no classification list had, because the flattener invents them from
    an ND array while the lists were written against the NPZ key. The run still
    reported ``finished``, because the bridge logs a conversion failure and
    returns None, so the whole session's tracks were silently absent.
    """
    path = tmp_path / "seq_fish0.npz"
    _with_tracklet_arrays(path)

    table = _convert(path)

    assert "tracklet_id" in table.columns, "the per-frame tracklet key must survive"
    leaked = [c for c in table.columns if c.startswith(("tracklets", "tracklet_vxys"))]
    assert not leaked, f"off-axis per-tracklet arrays reached the table: {leaked}"


def test_the_dropped_arrays_are_the_off_axis_ones_only(tmp_path: Path) -> None:
    """Dropping must not become a licence to drop the per-frame column too."""
    path = tmp_path / "seq_fish0.npz"
    _with_tracklet_arrays(path)

    table = _convert(path)

    assert table["tracklet_id"].notna().all()
    assert len(table) == 8, "the frame axis must be unchanged by the drop"


def test_a_flattened_field_is_classified_under_its_base_name() -> None:
    """The general defect behind the specific one.

    ``load_npz_to_df`` names an ND array's components ``<key>_<i>``, so any field
    classified under its NPZ key was invisible to the unit guard once it arrived
    two-dimensional. Reducing the index is what makes a classification apply to
    the components it was written for -- and is what stops the *next* ND field
    from failing a whole session the same way.
    """
    from mosaic.core.track_library.trex import base_field

    assert base_field("tracklet_vxys_2") == "tracklet_vxys"
    assert base_field("tracklets_0") == "tracklets"
    # A source suffix and an index reduce together, and neither eats a name that
    # merely ends in a word or a digit-bearing keypoint column.
    assert base_field("SPEED#wcentroid") == "SPEED"
    assert base_field("midline_x") == "midline_x"
    assert base_field("poseX0") == "poseX0"


def test_an_unclassified_field_still_refuses_a_calibrated_table(
    tmp_path: Path,
) -> None:
    """The guard must keep its teeth: dropping two fields is not disarming it."""
    path = tmp_path / "seq_fish0.npz"
    write_trex_npz(
        path, n=4, cm_per_pixel=0.03, some_new_trex_field=np.arange(4, dtype=float)
    )
    with pytest.raises(UnknownTrexUnitsError, match="some_new_trex_field"):
        _ = _convert(path)


# --- the reader that is told the factor --------------------------------------
#
# TRex has written `cm_per_pixel` into every export since 2025-02-18. An older
# export is centimetres with no record of by how much, and `MissingTrexCalibration
# Error` names the remedy: "converted by a reader that is told the factor". These
# are the two halves of that promise.


def _write_uncalibrated(path: Path, *, n: int = 8, **columns: np.ndarray) -> None:
    """A pre-2025 export: everything TRex writes today, minus the factor."""
    write_trex_npz(path, n=n, **columns)
    kept = {k: v for k, v in np.load(path).items() if k != CALIBRATION_COLUMN}
    np.savez(path, **kept)


def _convert_scaled(path: Path, factor: float) -> pd.DataFrame:
    return TrexNpzScaledConverter().convert(
        path,
        TrexScaledNpzParams(cm_per_pixel=factor),
        EntryHints(group="", sequence="seq"),
    )


def test_a_file_with_no_factor_refuses_under_trex_npz_and_converts_told(
    tmp_path: Path,
) -> None:
    """The refusal names a remedy; this is the remedy existing."""
    path = tmp_path / "seq_fish0.npz"
    _write_uncalibrated(path, n=4)

    with pytest.raises(MissingTrexCalibrationError, match=CALIBRATION_COLUMN):
        _ = _convert(path)

    out = _convert_scaled(path, 0.25)
    assert not out.empty


def test_the_stated_factor_is_the_one_applied(tmp_path: Path) -> None:
    path = tmp_path / "seq_fish0.npz"
    _write_uncalibrated(path, n=4)
    raw = np.load(path)

    out = _convert_scaled(path, 0.25)

    assert out["X#wcentroid"].to_numpy() == pytest.approx(raw["X#wcentroid"] / 0.25)
    assert out["Y#wcentroid"].to_numpy() == pytest.approx(raw["Y#wcentroid"] / 0.25)


def test_x_is_the_body_centre_and_the_head_survives(tmp_path: Path) -> None:
    """Same landmark rule as the calibrated reader: bare X is the *head*."""
    path = tmp_path / "seq_fish0.npz"
    head = np.linspace(5.0, 6.0, 4)
    _write_uncalibrated(path, n=4, X=head, Y=head)
    raw = np.load(path)

    out = _convert_scaled(path, 0.5)

    assert out["X"].to_numpy() == pytest.approx(raw["X#wcentroid"] / 0.5)
    assert out["X#head"].to_numpy() == pytest.approx(head / 0.5)
    assert out["Y#head"].to_numpy() == pytest.approx(head / 0.5)


def test_a_recorded_factor_that_disagrees_refuses(tmp_path: Path) -> None:
    """One of the two is wrong about this recording, and neither is a default."""
    path = tmp_path / "seq_fish0.npz"
    write_trex_npz(path, n=4, cm_per_pixel=0.25)

    with pytest.raises(TrexCalibrationConflictError, match="0.25"):
        _ = _convert_scaled(path, 0.5)


def test_a_recorded_factor_that_agrees_converts_identically(tmp_path: Path) -> None:
    """Being told what the file already says is not a conflict, and not a difference."""
    path = tmp_path / "seq_fish0.npz"
    write_trex_npz(path, n=4, cm_per_pixel=0.25)

    told = _convert_scaled(path, 0.25)
    read = _convert(path)

    pd.testing.assert_frame_equal(told, read)


def test_the_two_readers_are_two_variants(tmp_path: Path) -> None:
    """A variant names one producer, so a reconstructed factor is addressable apart."""
    from mosaic.core.pipeline.tracks_identity import converter_op

    assert converter_op(TrexNpzScaledConverter.src_format) == "convert-trex_npz_scaled"
    assert converter_op(TrexNpzConverter.src_format) == "convert-trex_npz"


def test_two_factors_are_two_recipes() -> None:
    """The factor is a claim about the world, so it belongs in the identity."""
    one = TrexScaledNpzParams(cm_per_pixel=0.25).identity_dump()
    other = TrexScaledNpzParams(cm_per_pixel=0.5).identity_dump()
    assert one != other


def test_the_factor_is_required_rather_than_defaulted() -> None:
    """`1.0` is a scale, not an absence -- and this reader exists for the absence."""
    with pytest.raises(ValidationError):
        _ = TrexScaledNpzParams()


# --- the reader that keeps the centimetres -----------------------------------
#
# TREx scaled its output long before it recorded the factor, so a pre-2.0.0
# export is centimetres with no record of by how much. Nothing can divide that
# back out. Refusing the data over a number its owner may never need would be
# refusing an analysis that is perfectly well defined in centimetres.


def _convert_cm(path: Path) -> pd.DataFrame:
    return TrexNpzCmConverter().convert(
        path, TrackConvertParams(), EntryHints(group="", sequence="seq")
    )


def test_a_file_with_no_factor_converts_in_centimetres_with_no_parameter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seq_fish0.npz"
    _write_uncalibrated(path, n=4)

    out = _convert_cm(path)

    assert TrexNpzCmConverter.output_schema == "mosaic_cm_v1"
    assert not out.empty


def test_the_centimetres_are_left_exactly_as_trex_wrote_them(tmp_path: Path) -> None:
    """It converts nothing. That is the whole point, and the thing to pin."""
    path = tmp_path / "seq_fish0.npz"
    _write_uncalibrated(path, n=4)
    raw = np.load(path)

    out = _convert_cm(path)

    assert out["X#wcentroid"].to_numpy() == pytest.approx(raw["X#wcentroid"])
    assert out["Y#wcentroid"].to_numpy() == pytest.approx(raw["Y#wcentroid"])


def test_the_landmark_is_corrected_even_though_the_unit_is_not(
    tmp_path: Path,
) -> None:
    """Head-versus-centre is not a unit question and does not ride along with one."""
    path = tmp_path / "seq_fish0.npz"
    head = np.linspace(5.0, 6.0, 4)
    _write_uncalibrated(path, n=4, X=head, Y=head)
    raw = np.load(path)

    out = _convert_cm(path)

    assert out["X"].to_numpy() == pytest.approx(raw["X#wcentroid"])
    assert out["X#head"].to_numpy() == pytest.approx(head)


def test_a_recorded_factor_is_not_a_reason_to_refuse(tmp_path: Path) -> None:
    """Keeping centimetres is a choice, not a fallback -- so a factor is ignored."""
    path = tmp_path / "seq_fish0.npz"
    write_trex_npz(path, n=4, cm_per_pixel=0.25)
    raw = np.load(path)

    out = _convert_cm(path)

    assert out["X#wcentroid"].to_numpy() == pytest.approx(raw["X#wcentroid"])


def test_the_three_readers_are_three_variants() -> None:
    """One recipe each: the unit a table is in is part of what produced it."""
    from mosaic.core.pipeline.tracks_identity import converter_op

    ops = {
        converter_op(c.src_format)
        for c in (TrexNpzConverter, TrexNpzScaledConverter, TrexNpzCmConverter)
    }
    assert ops == {
        "convert-trex_npz",
        "convert-trex_npz_scaled",
        "convert-trex_npz_cm",
    }


def test_the_refusal_names_all_three_routes(tmp_path: Path) -> None:
    """An error naming a remedy that does not exist is a defect."""
    path = tmp_path / "seq_fish0.npz"
    _write_uncalibrated(path, n=4)

    with pytest.raises(MissingTrexCalibrationError) as excinfo:
        _ = _convert(path)

    message = str(excinfo.value)
    assert "re-export" in message.lower()
    assert "trex_npz_scaled" in message
    assert "trex_npz_cm" in message


# --- a real TREx 1.x export, by its own field names ---------------------------


_TREX_1X_KEYS: tuple[str, ...] = (
    "ACCELERATION#pcentroid",
    "ACCELERATION#wcentroid",
    "ANGLE",
    "ANGULAR_A#centroid",
    "ANGULAR_V#centroid",
    "AX",
    "AY",
    "BORDER_DISTANCE#pcentroid",
    "MIDLINE_OFFSET",
    "SPEED",
    "SPEED#pcentroid",
    "SPEED#smooth#wcentroid",
    "SPEED#wcentroid",
    "VX",
    "VY",
    "midline_length",
    "midline_x",
    "midline_y",
    "missing",
    "normalized_midline",
    "num_pixels",
    "segment_length",
    "timestamp",
)
"""Every per-frame field of a real TREx 1.x export, verbatim.

Taken from an archived 2019-2020 dataset of 720 files. TREx renamed three of
these on the way to 2.x -- ``frame_segments`` -> ``tracklets``, ``segment_vxys``
-> ``tracklet_vxys``, ``segment_length`` -> ``midline_segment_length`` -- and a
classifier that knew only the later spellings refused every one of those files.
"""


def _write_trex_1x(path: Path, *, n: int = 6) -> None:
    """A TREx 1.x export: no calibration, the 1.x spellings, no keypoints."""
    ramp = np.linspace(0.0, 1.0, n)
    fields: dict[str, np.ndarray] = {k: ramp.copy() for k in _TREX_1X_KEYS}
    fields["frame"] = np.arange(n, dtype=np.int64)
    fields["time"] = np.arange(n, dtype=float) / 30.0
    fields["X"] = ramp.copy()
    fields["Y"] = ramp.copy()
    fields["X#wcentroid"] = ramp.copy()
    fields["Y#wcentroid"] = ramp.copy()
    # Per-tracklet, not per-frame, under the 1.x names.
    fields["frame_segments"] = np.zeros((3, 2), dtype=np.int32)
    fields["segment_vxys"] = np.zeros((5, 4), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)


def test_a_trex_1x_export_converts_in_centimetres(tmp_path: Path) -> None:
    path = tmp_path / "seq_fish0.npz"
    _write_trex_1x(path)

    out = _convert_cm(path)

    assert {"X", "Y", "frame", "time", "id", "group", "sequence"} <= set(out.columns)


def test_the_1x_per_tracklet_fields_are_dropped_not_padded(tmp_path: Path) -> None:
    """Padded onto the frame axis they would put a value on a row that denies it."""
    path = tmp_path / "seq_fish0.npz"
    _write_trex_1x(path)

    out = _convert_cm(path)

    stray = sorted(
        c
        for c in map(str, out.columns)
        if c.startswith(("frame_segments", "segment_vxys"))
    )
    assert stray == []


def test_a_trex_1x_export_unscales_when_told_the_factor(tmp_path: Path) -> None:
    """The same file down the pixel route: every 1.x field has to classify."""
    path = tmp_path / "seq_fish0.npz"
    _write_trex_1x(path)

    out = _convert_scaled(path, 0.5)

    # `segment_length` is 1.x's `midline_segment_length`, and TREx scales both.
    assert out["segment_length"].to_numpy() == pytest.approx(
        np.linspace(0.0, 1.0, 6) / 0.5
    )
    # `midline_length` is the one TREx leaves in pixels.
    assert out["midline_length"].to_numpy() == pytest.approx(np.linspace(0.0, 1.0, 6))
