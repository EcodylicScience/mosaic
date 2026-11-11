"""Tests for the EventResponse feature: one row per individual and event."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.event_response import EventResponse
from mosaic.behavior.feature_library.individual_aggregate import IndividualAggregate
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import OverlapUnsupported, run_feature
from mosaic.core.pipeline.types import Result
from tests.helpers import add_track_sequences, make_dataset

GROUP = "G"
SEQUENCE = "S1"


def _speed(times: np.ndarray, *, onset: float, dip: float, back: float) -> np.ndarray:
    """A speed of 10 that falls to 2 at *onset* and returns to 10 at *back*.

    The sample at *back* is 9, the first value at or above 0.9 times baseline.
    """
    value = np.full(len(times), 10.0)
    value[(times >= onset) & (times < back)] = dip
    value[times == back] = 9.0
    return value


def _table(
    times: np.ndarray,
    values: np.ndarray,
    *,
    identity: int = 0,
    **columns: str | np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "group": GROUP,
            "sequence": SEQUENCE,
            "id": identity,
            "frame": np.arange(len(times)),
            "time": times,
            "speed": values,
        }
    )
    for name, column in columns.items():
        frame[name] = column
    return frame


def _recovering(identity: int = 0, **columns: str | np.ndarray) -> pd.DataFrame:
    """One sample a second for 40 s. Speed 10, 2 from t=20 to t=23, 9 at t=24."""
    times = np.arange(40, dtype=float)
    return _table(
        times,
        _speed(times, onset=20.0, dip=2.0, back=24.0),
        identity=identity,
        **columns,
    )


def _params(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "baseline_s": 10.0,
        "response_s": 15.0,
        "recover_fraction": 0.9,
    }
    base.update(overrides)
    return base


def _apply(df: pd.DataFrame, **overrides: object) -> pd.DataFrame:
    return EventResponse(params=_params(**overrides)).apply(df)


def test_a_known_recovery_latency_and_the_window_statistics() -> None:
    out = _apply(_recovering(), onset_times=[20.0])

    assert list(out.columns) == [
        "group",
        "sequence",
        "id",
        "event",
        "onset_time",
        "baseline_mean",
        "response_min",
        "response_mean",
        "recovery_latency_s",
        "recovered",
        "n_baseline",
        "n_response",
    ]
    row = out.iloc[0]
    assert len(out) == 1
    assert (row["group"], row["sequence"], row["id"]) == (GROUP, SEQUENCE, 0)
    assert (row["event"], row["onset_time"]) == (0, 20.0)
    assert row["baseline_mean"] == 10.0
    assert row["response_min"] == 2.0
    # t=20..23 are 2, t=24 is 9, t=25..34 are 10: (8 + 9 + 100) / 15.
    assert row["response_mean"] == pytest.approx(117.0 / 15.0)
    assert row["recovery_latency_s"] == 4.0
    assert out["recovered"].tolist() == [True]
    assert (row["n_baseline"], row["n_response"]) == (10, 15)


def test_a_response_that_never_recovers_has_nan_latency_and_a_flag() -> None:
    times = np.arange(40, dtype=float)
    values = np.where(times >= 20.0, 3.0, 10.0)

    out = _apply(_table(times, values), onset_times=[20.0])

    row = out.iloc[0]
    assert row["response_min"] == 3.0
    assert math.isnan(row["recovery_latency_s"])
    assert out["recovered"].tolist() == [False]


def test_a_value_that_never_falls_below_the_recovery_level_recovers_at_once() -> None:
    times = np.arange(40, dtype=float)

    out = _apply(_table(times, np.full(40, 10.0)), onset_times=[20.0])

    assert out.iloc[0]["recovery_latency_s"] == 0.0
    assert out["recovered"].tolist() == [True]


def test_recovery_is_measured_from_the_trough_not_from_an_early_dip() -> None:
    """One low sample, one high sample, a long trough, then the recovery."""
    times = np.arange(40, dtype=float)
    values = np.full(40, 10.0)
    values[20] = 2.0
    values[22:28] = 1.0
    values[28] = 9.0

    out = _apply(_table(times, values), onset_times=[20.0])

    row = out.iloc[0]
    assert row["response_min"] == 1.0
    assert row["recovery_latency_s"] == 8.0
    assert out["recovered"].tolist() == [True]


def test_a_value_that_recovers_only_before_its_trough_has_not_recovered() -> None:
    times = np.arange(40, dtype=float)
    values = np.full(40, 10.0)
    values[20] = 2.0
    values[22:] = 1.0

    out = _apply(_table(times, values), onset_times=[20.0])

    assert math.isnan(out.iloc[0]["recovery_latency_s"])
    assert out["recovered"].tolist() == [False]


def test_recover_fraction_scales_the_recovery_level() -> None:
    times = np.arange(40, dtype=float)
    values = np.full(40, 10.0)
    values[20:24] = 2.0
    values[24:28] = 6.0

    half = _apply(_table(times, values), onset_times=[20.0], recover_fraction=0.5)
    near_full = _apply(_table(times, values), onset_times=[20.0], recover_fraction=0.95)

    assert half.iloc[0]["recovery_latency_s"] == 4.0
    assert near_full.iloc[0]["recovery_latency_s"] == 8.0


def test_onsets_from_the_rising_edge_of_a_numeric_column() -> None:
    stimulus = (np.arange(40) >= 20).astype(float)

    out = _apply(_recovering(stim=stimulus), onset_column="stim")

    assert out["onset_time"].tolist() == [20.0]
    assert out.iloc[0]["recovery_latency_s"] == 4.0


def test_onsets_from_the_rising_edges_of_a_string_column() -> None:
    """Tracktor writes ``stimuli`` as ``Y`` or ``N``."""
    states = np.array(["N"] * 40, dtype=object)
    states[20:25] = "Y"
    states[30:32] = "Y"

    out = _apply(
        _recovering(stimuli=states),
        onset_column="stimuli",
        onset_value="Y",
        response_s=8.0,
    )

    assert out["event"].tolist() == [0, 1]
    assert out["onset_time"].tolist() == [20.0, 30.0]


def test_a_column_that_starts_active_has_no_edge_at_its_first_row() -> None:
    stimulus = np.zeros(40)
    stimulus[:5] = 1.0
    stimulus[20:25] = 1.0

    out = _apply(_recovering(stim=stimulus), onset_column="stim")

    assert out["onset_time"].tolist() == [20.0]


def test_a_nan_in_the_column_is_not_active() -> None:
    stimulus = np.full(40, np.nan)
    stimulus[20:25] = 1.0

    out = _apply(_recovering(stim=stimulus), onset_column="stim")

    assert out["onset_time"].tolist() == [20.0]


def test_a_string_column_needs_onset_value() -> None:
    states = np.array(["N"] * 40, dtype=object)
    states[20:25] = "Y"

    with pytest.raises(ValueError, match="onset_value"):
        _ = _apply(_recovering(stimuli=states), onset_column="stimuli")


def test_explicit_times_number_the_events_in_time_order() -> None:
    out = _apply(_recovering(), onset_times=[30.0, 20.0], response_s=5.0)

    assert out["event"].tolist() == [0, 1]
    assert out["onset_time"].tolist() == [20.0, 30.0]
    assert out["recovery_latency_s"].tolist() == [4.0, 0.0]


def test_every_individual_has_a_row_for_every_explicit_time() -> None:
    table = pd.concat([_recovering(0), _recovering(1)], ignore_index=True)

    out = _apply(table, onset_times=[20.0, 500.0])

    assert out[["id", "event"]].values.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    beyond = out[out["event"] == 1].iloc[0]
    assert beyond["n_baseline"] == 0
    assert beyond["n_response"] == 0
    assert math.isnan(beyond["baseline_mean"])
    assert math.isnan(beyond["recovery_latency_s"])
    assert beyond["recovered"] is pd.NA


def test_a_window_without_a_baseline_has_no_recovery_to_report() -> None:
    out = _apply(_recovering(), onset_times=[0.0])

    row = out.iloc[0]
    assert row["n_baseline"] == 0
    assert math.isnan(row["baseline_mean"])
    assert row["response_min"] == 10.0
    assert math.isnan(row["recovery_latency_s"])
    assert row["recovered"] is pd.NA


def test_a_window_without_a_response_has_no_recovery_to_report() -> None:
    """Onset 45 s lies past the last row, so only the baseline holds rows."""
    out = _apply(_recovering(), onset_times=[45.0])

    row = out.iloc[0]
    assert row["n_baseline"] == 5
    assert row["n_response"] == 0
    assert row["baseline_mean"] == 10.0
    assert math.isnan(row["response_min"])
    assert math.isnan(row["recovery_latency_s"])
    assert row["recovered"] is pd.NA


def test_recovered_is_a_nullable_boolean_column() -> None:
    out = _apply(_recovering(), onset_times=[20.0, 500.0])

    assert str(out["recovered"].dtype) == "boolean"
    assert out["recovered"].tolist() == [True, pd.NA]


def test_individuals_are_measured_alone() -> None:
    times = np.arange(40, dtype=float)
    slow = _table(times, _speed(times, onset=20.0, dip=2.0, back=28.0), identity=1)
    table = pd.concat([slow, _recovering(0)], ignore_index=True)

    out = _apply(table, onset_times=[20.0])

    assert out["id"].tolist() == [0, 1]
    assert out["recovery_latency_s"].tolist() == [4.0, 8.0]


def test_an_arena_column_keys_each_individual_by_its_arena() -> None:
    first = _recovering(0, arena="a")
    second = _table(
        np.arange(40, dtype=float),
        _speed(np.arange(40, dtype=float), onset=20.0, dip=2.0, back=28.0),
        identity=0,
        arena="b",
    )

    out = _apply(pd.concat([first, second], ignore_index=True), onset_times=[20.0])

    assert list(out.columns[:4]) == ["group", "sequence", "id", "arena"]
    assert out["arena"].tolist() == ["a", "b"]
    assert out["recovery_latency_s"].tolist() == [4.0, 8.0]


def test_windows_follow_the_time_column_when_frames_were_dropped() -> None:
    """Samples every 0.5 s with some missing: seconds select rows, not counts."""
    times = np.arange(0.0, 40.0, 0.5)
    values = np.full(len(times), 10.0)
    values[(times >= 20.0) & (times < 24.5)] = 2.0
    values[times == 24.5] = 9.0
    # Drop 12.0..14.5 from the baseline window and 21.0..22.5 from the dip.
    dropped = ((times >= 12.0) & (times < 15.0)) | ((times >= 21.0) & (times < 23.0))
    table = _table(times[~dropped], values[~dropped])

    out = _apply(table, onset_times=[20.0])

    row = out.iloc[0]
    # [10, 20) holds 20 half-second samples less the 6 dropped.
    assert row["n_baseline"] == 14
    # [20, 35) holds 30 samples less the 4 dropped.
    assert row["n_response"] == 26
    assert row["baseline_mean"] == 10.0
    assert row["recovery_latency_s"] == 4.5


def test_an_irregular_time_column_is_never_read_as_a_frame_rate() -> None:
    """Samples at 0, 1, 2, 4, 8, ...: a window holds the rows inside its seconds."""
    times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 17.0, 18.0, 30.0, 31.0])
    values = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 2.0, 2.0, 9.0, 10.0, 10.0])

    out = _apply(
        _table(times, values), onset_times=[16.0], baseline_s=10.0, response_s=5.0
    )

    row = out.iloc[0]
    assert row["n_baseline"] == 1
    assert row["n_response"] == 3
    assert row["response_min"] == 2.0
    assert row["recovery_latency_s"] == 2.0


def _dense_labels(path: Path, labels: np.ndarray) -> Path:
    np.savez(path, labels=labels)
    return path


def _label_feature(
    tmp_path: Path, labels: np.ndarray, **overrides: object
) -> EventResponse:
    path = _dense_labels(tmp_path / "labels.npz", labels)
    feature = EventResponse(
        params=_params(labels={"kind": "behavior"}, onset_classes=[1], **overrides)
    )
    _ = feature.load_state(tmp_path, {}, {"labels": {(GROUP, SEQUENCE): path}})
    return feature


def test_onsets_from_a_label_series(tmp_path: Path) -> None:
    labels = np.zeros(40, dtype=np.int64)
    labels[20:25] = 1
    labels[30:32] = 2
    feature = _label_feature(tmp_path, labels)

    out = feature.apply(_recovering())

    assert out["onset_time"].tolist() == [20.0]
    assert out.iloc[0]["recovery_latency_s"] == 4.0


def test_each_run_of_a_listed_class_is_an_event(tmp_path: Path) -> None:
    labels = np.zeros(40, dtype=np.int64)
    labels[20:25] = 1
    labels[30:32] = 1
    feature = _label_feature(tmp_path, labels, response_s=8.0)

    out = feature.apply(_recovering())

    assert out["event"].tolist() == [0, 1]
    assert out["onset_time"].tolist() == [20.0, 30.0]


def test_a_label_run_in_the_first_row_has_no_onset(tmp_path: Path) -> None:
    labels = np.zeros(40, dtype=np.int64)
    labels[:3] = 1
    labels[20:25] = 1
    feature = _label_feature(tmp_path, labels)

    assert feature.apply(_recovering())["onset_time"].tolist() == [20.0]


def test_an_entry_without_a_label_file_is_refused(tmp_path: Path) -> None:
    feature = EventResponse(
        params=_params(labels={"kind": "behavior"}, onset_classes=[1])
    )
    _ = feature.load_state(tmp_path, {}, {"labels": {}})

    with pytest.raises(ValueError, match="no label file"):
        _ = feature.apply(_recovering())


def test_a_table_with_no_events_gives_no_rows() -> None:
    stimulus = np.zeros(40)

    out = _apply(_recovering(stim=stimulus), onset_column="stim")

    assert out.empty
    assert list(out.columns)[:5] == ["group", "sequence", "id", "event", "onset_time"]


def test_empty_input_gives_no_output() -> None:
    assert _apply(pd.DataFrame(), onset_times=[1.0]).empty


def test_another_value_column_is_measured() -> None:
    table = _recovering().rename(columns={"speed": "distance"})

    out = _apply(table, onset_times=[20.0], value_column="distance")

    assert out.iloc[0]["recovery_latency_s"] == 4.0


def test_a_missing_column_is_named() -> None:
    with pytest.raises(ValueError, match="speed_smooth"):
        _ = _apply(_recovering(), onset_times=[20.0], value_column="speed_smooth")


def test_exactly_one_source_of_onsets_is_required() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _ = EventResponse(params=_params())
    with pytest.raises(ValueError, match="exactly one"):
        _ = EventResponse(params=_params(onset_times=[1.0], onset_column="stim"))


def test_a_label_source_needs_its_classes_and_the_reverse() -> None:
    with pytest.raises(ValueError, match="onset_classes"):
        _ = EventResponse(params=_params(labels={"kind": "behavior"}))
    with pytest.raises(ValueError, match="onset_classes"):
        _ = EventResponse(params=_params(onset_times=[1.0], onset_classes=[1]))


def test_onset_value_needs_a_column() -> None:
    with pytest.raises(ValueError, match="onset_value"):
        _ = EventResponse(params=_params(onset_times=[1.0], onset_value="Y"))


def test_overlap_is_refused(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    add_track_sequences(dataset, ("g", "s1"), n_rows=10, n_ids=2)
    feature = EventResponse(params=_params(onset_times=[1.0], value_column="X"))

    with pytest.raises(OverlapUnsupported, match="accepts_overlap = False"):
        _ = run_feature(dataset, feature, overlap_frames=3)


def test_a_run_writes_one_row_per_individual_and_event(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    add_track_sequences(dataset, ("g", "s1"), n_rows=90, n_ids=2)
    feature = EventResponse(
        params=_params(
            onset_times=[1.0, 2.0], value_column="X", baseline_s=0.5, response_s=0.5
        )
    )

    result = run_feature(dataset, feature)

    assert result.failed_entries == ()
    assert result.entries_written == 1
    root = feature_run_root(dataset, result.feature, str(result.run_id))
    (path,) = root.glob("*.parquet")
    out = pd.read_parquet(path)
    assert out[["id", "event", "onset_time"]].values.tolist() == [
        [0, 0, 1.0],
        [0, 1, 2.0],
        [1, 0, 1.0],
        [1, 1, 2.0],
    ]
    # Half a second at 30 frames a second is 15 rows of each window.
    assert out["n_baseline"].tolist() == [15, 15, 15, 15]
    assert out["n_response"].tolist() == [15, 15, 15, 15]


def test_a_run_reads_a_stimulus_column_through_the_two_input_route(
    tmp_path: Path,
) -> None:
    """``speed-angvel`` carries the speed and the tracks carry ``stimuli``."""
    dataset = make_dataset(tmp_path / "ds")
    add_track_sequences(dataset, ("g", "s1"), n_rows=300, n_ids=2)
    tracks = dataset.get_root("tracks") / f"{make_entry_key('g', 's1')}.parquet"
    table = pd.read_parquet(tracks)
    # Frames run at 30 a second, so frame 150 is 5 s.
    table["stimuli"] = np.where(
        (table["frame"] >= 150) & (table["frame"] < 200), "Y", "N"
    )
    table.to_parquet(tracks)
    speed = run_feature(dataset, SpeedAngvel())
    feature = EventResponse(
        inputs=EventResponse.Inputs(
            (Result(feature=speed.feature, run_id=str(speed.run_id)), "tracks")
        ),
        params={
            "onset_column": "stimuli",
            "onset_value": "Y",
            "baseline_s": 1.0,
            "response_s": 2.0,
        },
    )

    result = run_feature(dataset, feature)

    assert result.failed_entries == ()
    assert result.entries_written == 1
    root = feature_run_root(dataset, result.feature, str(result.run_id))
    (path,) = root.glob("*.parquet")
    out = pd.read_parquet(path)
    assert out[["id", "event"]].values.tolist() == [[0, 0], [1, 0]]
    assert out["onset_time"].tolist() == [5.0, 5.0]
    assert (out["n_baseline"] > 0).all()
    assert (out["n_response"] > 0).all()


def test_a_null_recovered_survives_the_parquet_round_trip(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    add_track_sequences(dataset, ("g", "s1"), n_rows=90, n_ids=1)
    feature = EventResponse(
        params=_params(
            onset_times=[1.0, 500.0], value_column="X", baseline_s=0.5, response_s=0.5
        )
    )

    result = run_feature(dataset, feature)

    assert result.failed_entries == ()
    root = feature_run_root(dataset, result.feature, str(result.run_id))
    (path,) = root.glob("*.parquet")
    out = pd.read_parquet(path)
    assert out["onset_time"].tolist() == [1.0, 500.0]
    assert out["recovered"].isna().tolist() == [False, True]


def test_individual_aggregate_reduces_a_run_with_a_null_recovered(
    tmp_path: Path,
) -> None:
    """One individual's last event has no response rows, so its ``recovered`` is null."""
    dataset = make_dataset(tmp_path / "ds")
    add_track_sequences(dataset, ("g", "s1"), n_rows=90, n_ids=2)
    tracks = dataset.get_root("tracks") / f"{make_entry_key('g', 's1')}.parquet"
    table = pd.read_parquet(tracks)
    table["X"] = 10.0
    table = table[~((table["id"] == 1) & (table["time"] > 0.9))]
    table.to_parquet(tracks)
    responses = run_feature(
        dataset,
        EventResponse(
            params={
                "onset_times": [0.6, 1.0],
                "value_column": "X",
                "baseline_s": 0.3,
                "response_s": 0.3,
            }
        ),
    )
    per_individual = IndividualAggregate(
        inputs=IndividualAggregate.Inputs(
            (Result(feature=responses.feature, run_id=str(responses.run_id)),)
        ),
        params={"columns": ["recovered"], "aggs": ["sum", "mean"]},
    )

    result = run_feature(dataset, per_individual)

    assert result.failed_entries == ()
    root = feature_run_root(dataset, result.feature, str(result.run_id))
    (path,) = root.glob("*.parquet")
    out = pd.read_parquet(path)
    assert out["id"].tolist() == [0, 1]
    assert out["recovered_sum"].tolist() == [2.0, 1.0]
    assert out["recovered_mean"].tolist() == [1.0, 1.0]
