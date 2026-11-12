"""Placing tracks on a uniform frame rate, and the ways that goes silently wrong.

Every test here guards a way of re-gridding that produces a plausible table
holding numbers nobody measured -- which is the whole hazard of this operation.
Interpolating an angle linearly writes a full body rotation into the table at
every wrap; bridging a gap invents positions across a dropout; averaging a
tracklet id yields an id that never existed; and losing the padded
``cm_per_pixel`` row takes the calibration with it, which is unrecoverable once
the raw export is swept.

The identity case is the sharpest of them: re-gridding a table onto the rate it
already carries must return it unchanged, including where a neighbouring sample
is missing. Without the exact-hit carve-out in ``_both_ends_finite`` a NaN would
spread to both of its neighbours on every no-op run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.ops import IdentityDeferred, run_op
from mosaic.core.pipeline.resample_tracks import resample_entry_table
from mosaic.core.timebase import measure_time_axis
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import read_tracks_index, write_tracks_row
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.scope import Scope
from mosaic.tracking.ops.resample import ResampleTracksOp, ResampleTracksParams

from tests.helpers import (
    ANGULAR_VELOCITY,
    GAP_SECONDS,
    VELOCITY,
    dropped_frame_times,
    dropped_frame_tracks,
    make_dataset,
)

SOURCE_VARIANT = "convert-trex_npz.0.2-aaaaaaaaaa"


def _table(
    fps: float,
    n: int = 600,
    ids: tuple[int, ...] = (0, 1),
    *,
    sequence: str = "s1",
    carry_rate: bool = True,
) -> pd.DataFrame:
    """A TRex-shaped table: two individuals, one calibration row each."""
    blocks: list[pd.DataFrame] = []
    for individual in ids:
        frames = np.arange(n)
        calibration = np.full(n, np.nan)
        calibration[0] = 0.05
        columns: dict[str, object] = {
            "frame": frames.astype(np.int64),
            # float32 on purpose: it is what TRex writes, and what makes a
            # median-of-differences rate estimate snap to a binary fraction.
            "time": (frames / fps).astype(np.float32),
            "id": np.full(n, individual, dtype=np.int64),
            "group": [""] * n,
            "sequence": [sequence] * n,
            "X": frames * 0.2 + individual,
            "Y": np.sin(frames / 25.0) * 10.0 + individual,
            "ANGLE": ((frames * 0.3 + np.pi) % (2 * np.pi)) - np.pi,
            "cm_per_pixel": calibration,
            "tracklet_id": (frames // 100).astype(np.int64),
        }
        if carry_rate:
            columns["frame_rate"] = np.full(n, float(fps))
        blocks.append(pd.DataFrame(columns))
    return pd.concat(blocks, ignore_index=True)


# --- the time a table is on ------------------------------------------------


def _rate(frame: pd.DataFrame) -> float:
    return measure_time_axis(frame, "test").rate


def test_positions_are_placed_by_time_across_a_gap() -> None:
    """The table dropped frames, so ``frame`` over a rate misplaces what follows.

    Each individual moves at a constant velocity, so the position at any target
    time has one right answer, including the targets that fall inside a gap.
    """
    resampled = resample_entry_table(dropped_frame_tracks(), 30.0)

    seconds = resampled["time"].to_numpy()
    np.testing.assert_allclose(resampled["frame"], np.rint(seconds * 30.0))
    np.testing.assert_allclose(
        resampled["X"], VELOCITY[0] * seconds + 10.0 * resampled["id"], atol=1e-9
    )
    np.testing.assert_allclose(resampled["Y"], VELOCITY[1] * seconds, atol=1e-9)
    # An angle is placed on the circle, so compare it modulo a full turn. Across a
    # gap the chord of the unit circle is shorter than the arc, so only the targets
    # outside every gap are exact to the chord's error.
    native = dropped_frame_times()
    inside_gap = np.zeros(seconds.size, dtype=bool)
    for row in GAP_SECONDS:
        inside_gap |= (seconds > native[row - 1]) & (seconds < native[row])
    turned = np.angle(np.exp(1j * (resampled["ANGLE"] - ANGULAR_VELOCITY * seconds)))
    np.testing.assert_allclose(turned[~inside_gap], 0.0, atol=1e-4)


def test_a_table_that_dropped_frames_leaves_on_an_even_grid() -> None:
    resampled = resample_entry_table(dropped_frame_tracks(), 30.0)

    assert _rate(resampled) == pytest.approx(30.0)
    for _, block in resampled.groupby("id"):
        assert np.all(np.diff(block["frame"].to_numpy()) == 1)


def test_a_stated_rate_that_disagrees_with_time_is_overwritten_not_trusted() -> None:
    frame = _table(30.0, n=300)
    frame["frame_rate"] = 60.0

    resampled = resample_entry_table(frame, 25.0)

    assert resampled["frame_rate"].unique().tolist() == [25.0]
    assert _rate(resampled) == pytest.approx(25.0)


def test_a_float32_clock_three_hours_in_resamples_onto_its_own_rate_unchanged() -> None:
    """At 10,800 s a float32 time is only good to a millisecond, which is 3% of a step.

    The snap tolerance follows that resolution, so a no-op resample stays one and a
    missing sample does not spread to its neighbors.
    """
    rate = 30.0
    start = int(10_800 * rate) - 600
    frames = np.arange(start, start + 1200)
    frame = pd.DataFrame(
        {
            "frame": frames,
            "time": (frames / rate).astype(np.float32),
            "id": np.zeros(frames.size, dtype=np.int64),
            "group": [""] * frames.size,
            "sequence": ["s"] * frames.size,
            "X": frames * 0.01,
            "Y": np.ones(frames.size),
        }
    )
    frame.loc[600, "X"] = np.nan

    resampled = resample_entry_table(frame, rate)

    assert int(resampled["X"].isna().sum()) == 1
    np.testing.assert_array_equal(resampled["frame"], frame["frame"])
    pd.testing.assert_series_equal(
        resampled["X"].reset_index(drop=True),
        frame["X"].reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
    )


def test_the_rate_is_measured_from_the_span_when_time_is_float32() -> None:
    """Endpoints, not consecutive differences, so float32 ``time`` does not bias it."""
    assert _rate(_table(34.679, carry_rate=False)) == pytest.approx(34.679, rel=1e-4)


def test_a_table_without_time_is_refused_by_name() -> None:
    frame = _table(30.0, carry_rate=False).drop(columns=["time"])
    with pytest.raises(ValueError, match="no 'time' column"):
        _ = resample_entry_table(frame, 25.0, source="g/s")


def test_a_time_that_runs_backwards_is_refused() -> None:
    frame = _table(30.0, n=100, ids=(0,))
    frame.loc[frame.index[50], "time"] = frame.loc[frame.index[10], "time"]
    with pytest.raises(ValueError, match="does not increase"):
        _ = resample_entry_table(frame, 25.0)


# --- the grid ---------------------------------------------------------------


def test_the_output_is_on_the_target_rate_and_its_frames_are_contiguous() -> None:
    resampled = resample_entry_table(_table(34.679, n=1000), 30.0)

    assert _rate(resampled) == pytest.approx(30.0)
    for _, block in resampled.groupby("id"):
        frames = block["frame"].to_numpy()
        assert np.all(np.diff(frames) == 1)
    # 30/34.679 of the rows, to within the inward rounding at each end.
    assert len(resampled) == pytest.approx(
        len(_table(34.679, n=1000)) * 30 / 34.679, abs=4
    )


def test_every_individual_lands_on_one_shared_grid() -> None:
    """The frame axis stays a key two individuals can be joined on.

    Each is interpolated over its own coverage, so a naive implementation gives
    each a grid starting at its own first frame -- and then frame 40 names a
    different moment for each of them, which is exactly what makes a per-video
    lag unusable in a pooled analysis.
    """
    frame = _table(34.679, n=400)
    late = frame["id"] == 1
    frame.loc[late, "frame"] = frame.loc[late, "frame"] + 137
    frame.loc[late, "time"] = (frame.loc[late, "frame"] / 34.679).astype(np.float32)

    resampled = resample_entry_table(frame, 30.0)
    times = {
        int(individual): dict(zip(block["frame"], block["time"], strict=True))
        for individual, block in resampled.groupby("id")
    }
    shared = set(times[0]) & set(times[1])
    assert shared, "the two individuals overlap in time and must share frame numbers"
    for number in shared:
        assert times[0][number] == pytest.approx(times[1][number])


def test_resampling_onto_the_rate_a_table_already_carries_is_a_no_op() -> None:
    frame = _table(30.0, n=300)
    resampled = resample_entry_table(frame, 30.0)

    assert list(resampled.columns) == list(frame.columns)
    # ``check_dtype=False`` for one column and one reason: ``time`` is recomputed
    # as ``k / target_fps`` in float64, where the tracker wrote float32. That is
    # the axis becoming exact rather than the table changing -- three hours into
    # a float32 time column the resolution is about a millisecond, which is what
    # makes a median-of-differences rate estimate wrong in the first place.
    pd.testing.assert_frame_equal(
        resampled.reset_index(drop=True),
        frame.reset_index(drop=True),
        check_dtype=False,
    )


def test_a_no_op_does_not_spread_a_missing_sample_to_its_neighbours() -> None:
    """The exact-hit carve-out, which is why ``_both_ends_finite`` is not ``lo & hi``.

    At weight 0 the answer is the left sample alone. Requiring both brackets
    unconditionally would erase a measured value because the *next* one is
    missing, turning every no-op run into a slow erosion of the track.
    """
    frame = _table(30.0, n=300)
    frame.loc[frame.index[100], "X"] = np.nan

    resampled = resample_entry_table(frame, 30.0)

    assert int(resampled["X"].isna().sum()) == 1


# --- what must not be invented ----------------------------------------------


def test_an_angle_is_interpolated_on_the_circle_and_not_across_the_wrap() -> None:
    """A linear interpolation across +pi/-pi writes a full rotation into the table.

    It reads downstream as a real turn -- the same artefact that made ``heading``
    a feature with a declared method rather than something a converter computed.
    """
    turn_per_frame = 0.3
    resampled = resample_entry_table(_table(30.0, n=400, ids=(0,)), 23.0)

    steps = np.abs(np.diff(np.unwrap(resampled["ANGLE"].to_numpy())))
    assert steps.max() < np.pi
    assert steps.max() == pytest.approx(turn_per_frame * 30.0 / 23.0, rel=0.05)


def test_a_gap_is_not_bridged() -> None:
    """Filling holes belongs to ``trajectory-smooth``, under limits it declares."""
    frame = _table(30.0, n=400, ids=(0,))
    frame.loc[100:139, "X"] = np.nan

    resampled = resample_entry_table(frame, 21.0)

    missing = int(resampled["X"].isna().sum())
    assert missing == pytest.approx(40 * 21.0 / 30.0, abs=2)


def test_an_integer_column_is_gathered_rather_than_averaged() -> None:
    """Interpolating a tracklet id yields an id that was never observed."""
    resampled = resample_entry_table(_table(34.679, n=600), 30.0)

    assert resampled["tracklet_id"].dtype == np.int64
    assert set(np.unique(resampled["tracklet_id"])) <= {0, 1, 2, 3, 4, 5}


def test_a_nullable_tracklet_key_is_gathered_and_keeps_its_gaps() -> None:
    """The TRex converter's ``tracklet_start``: Int64, NA where undetected.

    Carried by the same rule as any integer column, and with its dtype intact,
    so a gap stays NA rather than becoming a float the next resample would
    interpolate.
    """
    frame = _table(34.679, n=600)
    starts = (frame["frame"] // 100) * 100
    frame["tracklet_start"] = pd.array(starts, dtype="Int64")
    frame.loc[frame["frame"].between(250, 299), "tracklet_start"] = pd.NA

    resampled = resample_entry_table(frame, 30.0)

    assert resampled["tracklet_start"].dtype == "Int64"
    observed = set(resampled["tracklet_start"].dropna().astype(int))
    assert observed <= {0, 100, 200, 300, 400, 500}
    assert resampled["tracklet_start"].isna().any(), "the gap must survive"


def test_the_padded_calibration_row_survives() -> None:
    """``cm_per_pixel`` is a scalar about the recording, not a series along it.

    The flattener pads it from a one-element array, so it is finite on one row
    per individual and NaN below. Interpolated like a measurement it has no
    bracket and would vanish -- taking with it the only record of the factor TRex
    applied, which ``mosaic upgrade-tracks`` exists because nothing can recover.
    """
    resampled = resample_entry_table(_table(34.679, n=600), 30.0)

    for _, block in resampled.groupby("id"):
        stated = block["cm_per_pixel"].dropna().unique()
        assert list(stated) == [0.05]


def test_a_repeated_frame_for_one_individual_is_refused() -> None:
    """Two rows for one moment do not describe an axis to interpolate along."""
    frame = _table(30.0, n=100, ids=(0,))
    frame.loc[frame.index[50], "frame"] = 49
    with pytest.raises(
        ValueError, match=r"^this table: frame 49 of id 0 appears in more than one row$"
    ):
        _ = resample_entry_table(frame, 25.0)


def test_a_non_positive_target_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="not a usable frame rate"):
        _ = resample_entry_table(_table(30.0, n=50), 0.0)


# --- the prefilter ----------------------------------------------------------


def test_the_prefilter_drops_a_spike_before_it_is_blended() -> None:
    """The clean-then-resample ordering, in one number stated per second.

    Off, the spike is attenuated by interpolation but still lands in the output;
    on, the sample it came from is removed and the gap propagates.
    """
    frame = _table(30.0, n=400, ids=(0,))
    frame.loc[frame.index[200], "X"] = 5000.0

    blended = resample_entry_table(frame, 25.0)
    gated = resample_entry_table(frame, 25.0, prefilter=40.0)

    assert blended["X"].max() > 1000.0
    assert int(blended["X"].isna().sum()) == 0
    assert np.nanmax(gated["X"].to_numpy()) < 1000.0
    assert int(gated["X"].isna().sum()) > 0


def test_the_prefilter_threshold_is_per_second_not_per_frame() -> None:
    """One number is the same physical threshold at every recording rate.

    Stating it per frame is the defect this module exists to fix, so reproducing
    it in the module's own gate would be a poor joke. The same displacement per
    *second* must be rejected identically whatever rate it was sampled at.
    """
    step = 3.0  # units per frame at 30 fps == 90 units/s
    for fps in (30.0, 34.679):
        frame = _table(fps, n=300, ids=(0,))
        frame["X"] = frame["frame"] * (step * 30.0 / fps)
        # 90 units/s, so a 100 units/s gate passes and an 80 units/s gate does not.
        assert (
            int(resample_entry_table(frame, 30.0, prefilter=100.0)["X"].isna().sum())
            == 0
        )
        assert (
            int(resample_entry_table(frame, 30.0, prefilter=80.0)["X"].isna().sum()) > 0
        )


# --- the op -----------------------------------------------------------------


def _seed_variant(
    ds: Dataset,
    sequences: tuple[str, ...],
    fps: float,
    *,
    carries_tracklets: bool | None = None,
) -> None:
    """A source tracks variant on disk, one table per sequence."""
    root = tracks_variant_root(ds.get_root("tracks"), SOURCE_VARIANT)
    for sequence in sequences:
        frame = _table(fps, n=400, sequence=sequence)
        out_path = root / f"{sequence}.parquet"
        _ = write_parquet_atomic(frame, out_path)
        write_tracks_row(
            ds,
            run_id=SOURCE_VARIANT,
            group="",
            sequence=sequence,
            out_path=out_path,
            producer="convert-trex_npz",
            std_format="trex_v2",
            n_rows=len(frame),
            consumed_source_roots=("tracks_raw",),
            carries_tracklets=carries_tracklets,
        )


def test_the_op_writes_a_second_variant_and_leaves_the_first_alone(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_variant(ds, ("s1", "s2"), 34.679)

    run_id = run_op(ds, "resample-tracks", {"target_fps": 30.0}, track=False)

    index = read_tracks_index(ds)
    variants = sorted(set(index["run_id"]))
    assert len(variants) == 2
    assert SOURCE_VARIANT in variants
    produced = next(name for name in variants if name != SOURCE_VARIANT)
    assert produced.startswith("resample-tracks.0.2-")

    rows = index[index["run_id"] == produced]
    assert sorted(rows["sequence"]) == ["s1", "s2"]
    assert set(rows["producer"]) == {"resample-tracks"}
    assert set(rows["producer_run_id"]) == {run_id}
    # It read a table and never opened video; claiming otherwise would put this
    # run into the drift reports on the next re-transcode.
    assert all(not str(value) for value in rows["consumed_media_composition"])
    for path in rows["abs_path"]:
        assert _rate(pd.read_parquet(ds.resolve_path(str(path)))) == (
            pytest.approx(30.0)
        )


@pytest.mark.parametrize(
    ("answer", "cell"), [(True, "true"), (False, "false"), (None, "")]
)
def test_the_op_copies_the_sources_tracklets_answer(
    tmp_path: Path, answer: bool | None, cell: str
) -> None:
    """Re-gridding changes no id, so the table follows animals as its source does."""
    ds = make_dataset(tmp_path / "ds")
    _seed_variant(ds, ("s1",), 34.679, carries_tracklets=answer)

    _ = run_op(ds, "resample-tracks", {"target_fps": 30.0}, track=False)

    index = read_tracks_index(ds)
    produced = index[index["producer"] == "resample-tracks"]
    assert [str(value) for value in produced["carries_tracklets"]] == [cell]


def test_the_op_records_the_new_frame_extent(tmp_path: Path) -> None:
    """The axis is measured from the parquet, which is why this is not a feature.

    ``FeatureIndexRow`` records no frame extent at all, so a re-gridded table
    offered as a feature ``Result`` would carry its new axis nowhere.
    """
    ds = make_dataset(tmp_path / "ds")
    _seed_variant(ds, ("s1",), 34.679)
    _ = run_op(ds, "resample-tracks", {"target_fps": 30.0}, track=False)

    index = read_tracks_index(ds)
    produced = index[index["run_id"] != SOURCE_VARIANT].iloc[0]
    source = index[index["run_id"] == SOURCE_VARIANT].iloc[0]
    assert int(produced["frame_max"]) < int(source["frame_max"])
    assert int(produced["frame_max"]) == pytest.approx(399 * 30 / 34.679, abs=1)


def test_the_variant_chains_from_the_source_it_read(tmp_path: Path) -> None:
    """One recipe over two sources is two variants, via the ``upstream`` term."""
    first = make_dataset(tmp_path / "a")
    _seed_variant(first, ("s1",), 34.679)
    second = make_dataset(tmp_path / "b")
    root = tracks_variant_root(second.get_root("tracks"), "trex.0.1-bbbbbbbbbb")
    frame = _table(34.679, n=400, sequence="s1")
    out_path = root / "s1.parquet"
    _ = write_parquet_atomic(frame, out_path)
    write_tracks_row(
        second,
        run_id="trex.0.1-bbbbbbbbbb",
        group="",
        sequence="s1",
        out_path=out_path,
        producer="trex",
        std_format="trex_v2",
        n_rows=len(frame),
        consumed_source_roots=("tracks_raw",),
        carries_tracklets=None,
    )

    params = ResampleTracksParams(target_fps=30.0)
    op = ResampleTracksOp()
    assert (
        op.plan_identity(first, params, ResolvedScope()).tracks_variant
        != op.plan_identity(second, params, ResolvedScope()).tracks_variant
    )


def test_two_source_variants_in_scope_are_refused(tmp_path: Path) -> None:
    """Its identity carries one upstream, so it cannot span two."""
    ds = make_dataset(tmp_path / "ds")
    _seed_variant(ds, ("s1",), 34.679)
    root = tracks_variant_root(ds.get_root("tracks"), "trex.0.1-bbbbbbbbbb")
    frame = _table(30.0, n=100, sequence="s2")
    out_path = root / "s2.parquet"
    _ = write_parquet_atomic(frame, out_path)
    write_tracks_row(
        ds,
        run_id="trex.0.1-bbbbbbbbbb",
        group="",
        sequence="s2",
        out_path=out_path,
        producer="trex",
        std_format="trex_v2",
        n_rows=len(frame),
        consumed_source_roots=("tracks_raw",),
        carries_tracklets=None,
    )

    with pytest.raises(ValueError, match="reads one tracks variant"):
        _ = ResampleTracksOp().plan_identity(
            ds, ResampleTracksParams(target_fps=30.0), ResolvedScope()
        )


def test_an_empty_scope_defers_its_identity_rather_than_guessing(
    tmp_path: Path,
) -> None:
    """A graph step whose source is an earlier step has nothing to name yet."""
    ds = make_dataset(tmp_path / "ds")
    with pytest.raises(IdentityDeferred):
        _ = ResampleTracksOp().plan_identity(
            ds, ResampleTracksParams(target_fps=30.0), ResolvedScope()
        )


def test_rerunning_is_a_no_op_and_keeps_the_identifier(tmp_path: Path) -> None:
    """A resampler does not read its own output.

    Without that exclusion the second run resolves two variants per entry and
    refuses, which would make the op runnable exactly once per dataset -- and
    would leave a run that failed part way through unable to resume, because its
    own partial output is what it would trip over.
    """
    ds = make_dataset(tmp_path / "ds")
    _seed_variant(ds, ("s1", "s2"), 34.679)

    first = run_op(ds, "resample-tracks", {"target_fps": 30.0}, track=False)
    second = run_op(ds, "resample-tracks", {"target_fps": 30.0}, track=False)

    assert first == second
    assert len(read_tracks_index(ds)) == 4


def test_a_scoped_run_reads_only_the_entries_it_names(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_variant(ds, ("s1", "s2"), 34.679)

    run_id = run_op(
        ds,
        "resample-tracks",
        {"target_fps": 30.0},
        track=False,
        scope=Scope(entries=[("", "s1")]),
    )

    index = read_tracks_index(ds)
    written = index[index["producer_run_id"] == run_id]
    assert list(written["sequence"]) == ["s1"]


def test_a_second_target_rate_is_a_second_variant_beside_the_first(
    tmp_path: Path,
) -> None:
    """And the first is still not read as an input by the second."""
    ds = make_dataset(tmp_path / "ds")
    _seed_variant(ds, ("s1",), 34.679)

    slow = run_op(ds, "resample-tracks", {"target_fps": 30.0}, track=False)
    fast = run_op(ds, "resample-tracks", {"target_fps": 25.0}, track=False)

    assert slow != fast
    index = read_tracks_index(ds)
    assert len(set(index["run_id"])) == 3
    rates = {
        float(_rate(pd.read_parquet(ds.resolve_path(str(path)))))
        for path, producer in zip(index["abs_path"], index["producer"], strict=True)
        if str(producer) == "resample-tracks"
    }
    assert rates == {30.0, 25.0}


def test_a_recipe_can_wire_a_tracks_reference_to_this_op() -> None:
    """``writes_tracks`` is declared, not inferred from the tracking-roots table.

    That table is the one a producer must appear in to bridge *from a tracker run
    root*, which is true of every tracks producer there was and false for one
    that reads a tracks table. Inferring from it would leave a downstream step
    unable to reference this op's output at all -- and ``reads_media`` must not
    follow, because this one never opens video.
    """
    from mosaic.core.pipeline.graph.resolve import declaration_catalog

    catalog = declaration_catalog()
    declaration = catalog.get("resample-tracks")
    assert declaration is not None
    assert declaration.produces.writes_tracks is True
    assert declaration.consumes.reads_media is False
    # Unchanged for a producer that does bridge from a run root.
    trex = catalog.get("trex")
    assert trex is not None
    assert trex.produces.writes_tracks is True
    assert trex.consumes.reads_media is True
