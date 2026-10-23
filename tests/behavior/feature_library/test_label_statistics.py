"""``label-statistics`` reduces a label sequence to a per-individual fingerprint.

Every case is a sequence short enough that its statistics are worked out by hand
in the test: usages, bout counts and durations, transition probabilities and both
entropies. The ones that matter most are the ones a per-frame array would get
wrong without complaint -- an overlap kept from annotated segments, so usages sum
past coverage; a nested bout counted as its own bout; a state an entry never
shows reported as zero rather than as a missing column.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.label_statistics import (
    LabelStatistics,
    state_suffix,
)
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.label_intervals import LabelSegments
from mosaic.core.pipeline._utils import atomic_savez
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.labels_identity import labels_variant_root
from mosaic.core.pipeline.labels_index import write_labels_row
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import write_tracks_row
from mosaic.core.params import Params
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    ResultColumn,
    TrackInput,
)
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.schema import ensure_track_schema

from tests.helpers import make_dataset

FPS = 10.0
ENTRY = ("", "s")

# label_ids 0, 1, 2 named none, rest, groom -- "none" the background class.
LABEL_IDS = np.array([0, 1, 2], dtype=np.int32)
LABEL_NAMES = np.array(["none", "rest", "groom"], dtype=object)


# --- builders ------------------------------------------------------------------


def _column_table(
    states: Mapping[int, Sequence[object]], frames: dict[int, list[int]] | None = None
) -> pd.DataFrame:
    """One row per (frame, id), carrying a categorical ``state`` column."""
    parts: list[pd.DataFrame] = []
    for ident, values in states.items():
        frame = (frames or {}).get(ident, list(range(len(values))))
        parts.append(
            pd.DataFrame(
                {
                    "frame": frame,
                    "time": [f / FPS for f in frame],
                    "id": ident,
                    "group": ENTRY[0],
                    "sequence": ENTRY[1],
                    "state": values,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _tracks(frames: dict[int, list[int]], sequence: str = ENTRY[1]) -> pd.DataFrame:
    """A labels-free table with frame, time and id: what tracks supply."""
    parts = [
        pd.DataFrame(
            {
                "frame": frame,
                "time": [f / FPS for f in frame],
                "id": ident,
                "group": "",
                "sequence": sequence,
            }
        )
        for ident, frame in frames.items()
    ]
    return pd.concat(parts, ignore_index=True)


def _column_feature(**params: object) -> LabelStatistics:
    feature = LabelStatistics(
        params={"labels": ResultColumn(column="state"), "fps": FPS, **params}
    )
    _ = feature.load_state(Path(), {}, {})
    return feature


def _ground_truth_feature(path: Path, **params: object) -> LabelStatistics:
    feature = LabelStatistics(params={"fps": FPS, **params})
    _ = feature.load_state(Path(), {}, {"labels": {ENTRY: path}})
    return feature


def _row(out: pd.DataFrame, ident: int) -> pd.Series:
    rows = out[out["id"] == ident]
    assert len(rows) == 1
    return rows.iloc[0]


def _bits(*weights: float) -> float:
    p = np.asarray(weights, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def _segments() -> LabelSegments:
    """rest [0, 10) with grooming [3, 5) nested inside it, and grooming [12, 14)."""
    return LabelSegments(
        start_frame=np.array([0, 3, 12]),
        end_frame=np.array([10, 5, 14]),
        label=np.array([1, 2, 2]),
    )


def _label_file(
    path: Path, segments: LabelSegments | None, dense: list[int] | None = None
) -> Path:
    payload: dict[str, object] = {
        "label_format": "dense",
        "label_ids": LABEL_IDS,
        "label_names": LABEL_NAMES,
    }
    if segments is not None:
        payload["labels"] = segments.to_dense(20, policy="shortest")
        payload.update(segments.to_payload())
    else:
        payload["labels"] = np.asarray(dense, dtype=np.int32)
    atomic_savez(path, **payload)
    return path


# --- a categorical column: runs ------------------------------------------------


def test_runs_give_usage_bouts_durations_transitions_and_entropies() -> None:
    """id 0: 0 0 0 1 1 0 0 2 2 2 at 10 frames per second.

    Runs 0x3, 1x2, 0x2, 2x3. Usage 0.5 / 0.2 / 0.3. Transitions 0->1, 1->0,
    0->2: from 0 half each way, from 1 always back to 0, from 2 none. The
    transition entropy weights row 0 (1 bit) by 2/3 and row 1 (0 bits) by 1/3.
    """
    table = _column_table({0: [0, 0, 0, 1, 1, 0, 0, 2, 2, 2], 1: [1, 1, 1, 1, 1]})
    out = _column_feature(states=[0, 1, 2]).apply(table)
    row = _row(out, 0)

    assert row["n_frames"] == 10
    assert row["coverage"] == pytest.approx(1.0)
    assert row["n_bouts"] == 4
    assert row["usage__0"] == pytest.approx(0.5)
    assert row["usage__1"] == pytest.approx(0.2)
    assert row["usage__2"] == pytest.approx(0.3)
    assert (row["bouts__0"], row["bouts__1"], row["bouts__2"]) == (2, 1, 1)
    # One minute holds 600 frames; 10 frames is 1/60 of one.
    assert row["rate_per_min__0"] == pytest.approx(120.0)
    assert row["rate_per_min__1"] == pytest.approx(60.0)
    assert row["bout_mean_s__0"] == pytest.approx(0.25)
    assert row["bout_median_s__0"] == pytest.approx(0.25)
    assert row["bout_mean_s__2"] == pytest.approx(0.3)
    assert row["p__0__1"] == pytest.approx(0.5)
    assert row["p__0__2"] == pytest.approx(0.5)
    assert row["p__1__0"] == pytest.approx(1.0)
    assert row["p__1__2"] == pytest.approx(0.0)
    assert np.isnan(row["p__2__0"]) and np.isnan(row["p__2__1"])
    assert row["entropy"] == pytest.approx(_bits(0.5, 0.2, 0.3), rel=1e-6)
    assert row["transition_entropy"] == pytest.approx(2.0 / 3.0, rel=1e-6)

    other = _row(out, 1)
    assert other["n_frames"] == 5
    assert other["usage__1"] == pytest.approx(1.0)
    assert other["entropy"] == 0.0
    assert np.isnan(other["transition_entropy"])
    assert np.isnan(other["bout_mean_s__0"])
    assert other["bout_mean_s__1"] == pytest.approx(0.5)


def test_the_output_is_one_typed_row_per_individual() -> None:
    table = _column_table({3: [0, 1], 1: [1, 0]})
    out = _column_feature(states=[0, 1]).apply(table)
    assert out.columns.tolist() == [
        "group",
        "sequence",
        "id",
        "n_frames",
        "coverage",
        "n_bouts",
        "entropy",
        "transition_entropy",
        "usage__0",
        "usage__1",
        "bouts__0",
        "bouts__1",
        "rate_per_min__0",
        "rate_per_min__1",
        "bout_mean_s__0",
        "bout_mean_s__1",
        "bout_median_s__0",
        "bout_median_s__1",
        "p__0__1",
        "p__1__0",
    ]
    assert out["id"].tolist() == [1, 3]
    assert out["sequence"].tolist() == ["s", "s"]
    assert out["n_frames"].dtype == np.int64
    assert out["bouts__0"].dtype == np.int64
    assert out["usage__0"].dtype == np.float32
    assert out["p__0__1"].dtype == np.float32
    assert "frame" not in out.columns


def test_short_runs_are_dropped_before_anything_is_counted() -> None:
    """min_bout_frames=3 keeps 0x3 and 2x3; the 1x2 and 0x2 runs vanish."""
    table = _column_table({0: [0, 0, 0, 1, 1, 0, 0, 2, 2, 2]})
    row = _row(_column_feature(states=[0, 1, 2], min_bout_frames=3).apply(table), 0)
    assert row["usage__0"] == pytest.approx(0.3)
    assert row["usage__1"] == 0.0
    assert row["usage__2"] == pytest.approx(0.3)
    assert row["coverage"] == pytest.approx(0.6)
    assert row["n_bouts"] == 2
    assert row["p__0__2"] == pytest.approx(1.0)
    assert row["p__0__1"] == 0.0
    assert row["transition_entropy"] == 0.0


def test_an_ignored_state_counts_toward_frames_and_nothing_else() -> None:
    """With 1 ignored the bouts are 0, 0, 2: 0 -> 0 is no transition."""
    table = _column_table({0: [0, 0, 0, 1, 1, 0, 0, 2, 2, 2]})
    row = _row(_column_feature(states=[0, 2], ignore=[1]).apply(table), 0)
    assert "usage__1" not in row.index
    assert row["n_frames"] == 10
    assert row["usage__0"] == pytest.approx(0.5)
    assert row["coverage"] == pytest.approx(0.8)
    assert row["bouts__0"] == 2
    assert row["n_bouts"] == 3
    assert row["p__0__2"] == pytest.approx(1.0)
    assert np.isnan(row["p__2__0"])


def test_a_fixed_vocabulary_reports_an_absent_state_as_zero() -> None:
    table = _column_table({0: [0, 0, 1, 1]})
    row = _row(_column_feature(states=[0, 1, 7]).apply(table), 0)
    assert row["usage__7"] == 0.0
    assert row["bouts__7"] == 0
    assert row["rate_per_min__7"] == 0.0
    assert np.isnan(row["bout_mean_s__7"])
    assert np.isnan(row["p__7__0"])
    assert row["p__0__7"] == 0.0
    assert row["p__0__1"] == pytest.approx(1.0)


def test_a_missing_frame_ends_a_run() -> None:
    table = _column_table({0: [0] * 5}, frames={0: [0, 1, 2, 5, 6]})
    row = _row(_column_feature(states=[0]).apply(table), 0)
    assert row["n_frames"] == 5
    assert row["bouts__0"] == 2
    assert row["bout_mean_s__0"] == pytest.approx(0.25)


def test_a_table_holding_every_other_frame_is_not_cut_at_every_row() -> None:
    """A table tracked on a variant decimated by 2 holds source frames 0, 2, 4, ...

    id 0 is state 0 on frames 0-18 and state 1 on 20-38, at 30 fps from the
    time column: two bouts of ten rows, each row two frames, so 20/30 s each,
    and one bout of 0 in 40/30 s of presence -- 45 a minute. id 1 misses frame
    10, a whole sample, which ends its run of 0 there: bouts of five and four
    rows, 0.3 s on average.
    """
    full = list(range(0, 40, 2))
    table = _column_table(
        {0: [0] * 10 + [1] * 10, 1: [0] * 9 + [1] * 10},
        frames={0: full, 1: [f for f in full if f != 10]},
    )
    table["time"] = table["frame"] / 30.0
    feature = LabelStatistics(
        params={"labels": ResultColumn(column="state"), "states": [0, 1]}
    )
    _ = feature.load_state(Path(), {}, {})
    out = feature.apply(table)

    first = _row(out, 0)
    assert first["n_frames"] == 20
    assert (first["n_bouts"], first["bouts__0"], first["bouts__1"]) == (2, 1, 1)
    assert first["bout_mean_s__0"] == pytest.approx(20 / 30)
    assert first["rate_per_min__0"] == pytest.approx(45.0)
    assert first["p__0__1"] == pytest.approx(1.0)

    second = _row(out, 1)
    assert second["bouts__0"] == 2
    assert second["bout_mean_s__0"] == pytest.approx(4.5 * 2 / 30)
    assert second["usage__0"] == pytest.approx(9 / 19)


def test_min_bout_frames_counts_frames_not_rows() -> None:
    """At a step of 2, a 5-row bout is 10 frames long and a 4-row one 8."""
    table = _column_table(
        {1: [0] * 9 + [1] * 10},
        frames={1: [f for f in range(0, 40, 2) if f != 10]},
    )
    feature = _column_feature(states=[0, 1], min_bout_frames=10)
    row = _row(feature.apply(table), 1)
    assert row["bouts__0"] == 1
    assert row["usage__0"] == pytest.approx(5 / 19)
    assert row["bout_mean_s__0"] == pytest.approx(10 / FPS)


def test_the_median_bout_is_not_the_mean() -> None:
    """0 1 00 1 000000: bouts of 0 last 1, 2 and 6 frames -- mean 3, median 2."""
    table = _column_table({0: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]})
    row = _row(_column_feature(states=[0, 1]).apply(table), 0)
    assert row["bout_mean_s__0"] == pytest.approx(0.3)
    assert row["bout_median_s__0"] == pytest.approx(0.2)
    assert row["bout_median_s__1"] == pytest.approx(0.1)


def test_a_null_value_is_unlabelled() -> None:
    table = _column_table({0: [0.0, 0.0, np.nan, 1.0]})
    row = _row(_column_feature(states=[0, 1]).apply(table), 0)
    assert row["coverage"] == pytest.approx(0.75)
    assert row["usage__0"] == pytest.approx(0.5)


def test_text_values_are_matched_by_name() -> None:
    table = _column_table({0: ["walk", "walk", "rest up", "walk"]})
    row = _row(_column_feature(states=["walk", "rest up"]).apply(table), 0)
    assert row["usage__walk"] == pytest.approx(0.75)
    assert row["bouts__walk"] == 2
    assert row["p__rest_up__walk"] == pytest.approx(1.0)


def test_rows_in_any_order_give_the_same_rows() -> None:
    """Interleaved ids and shuffled rows: bouts are cut per id, in frame order.

    id 4 is 0 0 1 1 0 on frames 10-14, id 2 is 1 1 1 0 on frames 3, 4, 7, 8 -- a
    gap after frame 4, so its 1s are two bouts and its transition is 1 -> 0.
    """
    table = _column_table(
        {4: [0, 0, 1, 1, 0], 2: [1, 1, 1, 0]},
        frames={4: [10, 11, 12, 13, 14], 2: [3, 4, 7, 8]},
    )
    feature = _column_feature(states=[0, 1])
    ordered = feature.apply(table)
    shuffled = feature.apply(table.sample(frac=1.0, random_state=1))
    pd.testing.assert_frame_equal(ordered, shuffled)

    four, two = _row(ordered, 4), _row(ordered, 2)
    assert (four["bouts__0"], four["bouts__1"]) == (2, 1)
    assert four["p__0__1"] == pytest.approx(1.0)
    assert four["p__1__0"] == pytest.approx(1.0)
    assert (two["bouts__0"], two["bouts__1"]) == (1, 2)
    assert two["bout_mean_s__1"] == pytest.approx(0.15)
    assert two["p__1__0"] == pytest.approx(1.0)
    assert np.isnan(two["p__0__1"])


def test_an_individual_with_no_labels_is_undefined_not_zero() -> None:
    """Every frame of id 1 is null: present, so a row, and nothing to measure."""
    table = _column_table({0: [0.0, 1.0], 1: [np.nan] * 3})
    row = _row(_column_feature(states=[0, 1]).apply(table), 1)
    assert row["n_frames"] == 3
    assert (row["coverage"], row["n_bouts"], row["usage__0"]) == (0.0, 0, 0.0)
    assert row["rate_per_min__0"] == 0.0
    for column in ("entropy", "transition_entropy", "bout_mean_s__0", "p__0__1"):
        assert np.isnan(row[column]), column


def test_two_rows_on_one_frame_with_different_labels_are_refused() -> None:
    table = _column_table({0: [0, 1]}, frames={0: [5, 5]})
    with pytest.raises(ValueError, match="two rows on frame 5"):
        _ = _column_feature(states=[0, 1]).apply(table)


@pytest.mark.parametrize("column", ["group", "sequence"])
def test_an_input_that_does_not_say_whose_entry_it_is_is_refused(
    column: str,
) -> None:
    """Defaulting a missing entry column to '' would put the row under another."""
    table = _column_table({0: [0, 1]}).drop(columns=[column])
    with pytest.raises(ValueError, match=rf"no '{column}' column"):
        _ = _column_feature(states=[0, 1]).apply(table)


def test_a_value_in_neither_states_nor_ignore_is_refused() -> None:
    table = _column_table({0: [0, 1, 5]})
    with pytest.raises(ValueError, match=r"5 is in neither states \[0, 1\] nor ignore"):
        _ = _column_feature(states=[0, 1]).apply(table)


def test_the_frame_rate_is_read_from_the_time_column() -> None:
    table = _column_table({0: [0, 0, 1, 1]})
    table["time"] = table["frame"] / 25.0
    feature = LabelStatistics(
        params={"labels": ResultColumn(column="state"), "states": [0, 1]}
    )
    _ = feature.load_state(Path(), {}, {})
    row = _row(feature.apply(table), 0)
    assert row["bout_mean_s__0"] == pytest.approx(2 / 25.0)


def test_no_frame_rate_and_no_time_column_is_refused() -> None:
    table = _column_table({0: [0, 0, 1, 1]}).drop(columns=["time"])
    feature = LabelStatistics(
        params={"labels": ResultColumn(column="state"), "states": [0, 1]}
    )
    _ = feature.load_state(Path(), {}, {})
    with pytest.raises(ValueError, match="needs a frame rate"):
        _ = feature.apply(table)


# --- construction ----------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        {"kind": "segments"},
        {"feature": "arhmm", "run_id": "0.1-abc", "column": "syllable"},
    ],
)
def test_the_params_survive_the_dict_a_worker_rebuilds_them_from(
    labels: dict[str, object],
) -> None:
    """A process worker and a recipe both hand the labels source in as a dict.

    Merged into a default ground-truth source, a ResultColumn's keys would
    validate as neither member of the union.
    """
    feature = LabelStatistics(params={"labels": labels, "states": [1, "a"]})
    dumped = feature.params.model_dump()
    rebuilt = LabelStatistics(params=dumped)
    assert rebuilt.params.model_dump() == dumped
    assert type(rebuilt.params.labels) is type(feature.params.labels)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"labels": ResultColumn(column="state")}, "needs states"),
        ({"states": [0, 1], "ignore": [1]}, "both states and ignore"),
        ({"states": ["rear up", "rear_up"]}, "both be spelled"),
        ({"states": [0, 0]}, "repeats"),
        ({"states": []}, "at least one"),
        ({"min_bout_frames": 0}, "greater than or equal"),
        # An infinite rate passes gt=0 and makes every presence zero minutes long.
        ({"fps": float("inf")}, "finite number"),
        ({"fps": float("nan")}, "finite number"),
        # The resolver skips an empty kind, which would fail every entry later.
        ({"labels": {"kind": ""}}, "at least 1 character"),
    ],
)
def test_an_unworkable_vocabulary_is_refused_at_construction(
    params: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _ = LabelStatistics(params=params)


def test_state_suffixes_are_identifier_safe_and_unambiguous() -> None:
    assert state_suffix("rear up") == "rear_up"
    assert state_suffix("a__b") == "a_b"
    assert state_suffix("  Walk! ") == "Walk"
    assert state_suffix(3) == "3"
    assert state_suffix(-1) == "neg1"
    with pytest.raises(ValueError, match="no letters or digits"):
        _ = state_suffix("--")


# --- ground truth ---------------------------------------------------------------


def test_a_dense_label_file_is_scene_level_over_each_individuals_frames(
    tmp_path: Path,
) -> None:
    """Labels none none rest rest rest groom groom none none none, none ignored.

    id 0 is present on all ten frames, id 1 on frames 2-6 only, so the two share
    one sequence and differ in what they were there for.
    """
    path = _label_file(tmp_path / "s.npz", None, dense=[0, 0, 1, 1, 1, 2, 2, 0, 0, 0])
    feature = _ground_truth_feature(path, ignore=["none"])
    out = feature.apply(_tracks({0: list(range(10)), 1: [2, 3, 4, 5, 6]}))

    first = _row(out, 0)
    assert "usage__none" not in first.index
    assert first["n_frames"] == 10
    assert first["usage__rest"] == pytest.approx(0.3)
    assert first["usage__groom"] == pytest.approx(0.2)
    assert first["coverage"] == pytest.approx(0.5)
    assert first["n_bouts"] == 2
    assert first["p__rest__groom"] == pytest.approx(1.0)

    second = _row(out, 1)
    assert second["n_frames"] == 5
    assert second["usage__rest"] == pytest.approx(0.6)
    assert second["usage__groom"] == pytest.approx(0.4)
    assert second["coverage"] == pytest.approx(1.0)


def test_label_names_are_the_default_vocabulary(tmp_path: Path) -> None:
    path = _label_file(tmp_path / "s.npz", None, dense=[1, 1, 2, 2])
    out = _ground_truth_feature(path).apply(_tracks({0: [0, 1, 2, 3]}))
    row = _row(out, 0)
    assert row["usage__none"] == 0.0
    assert row["usage__rest"] == pytest.approx(0.5)


def test_states_may_name_labels_by_id(tmp_path: Path) -> None:
    path = _label_file(tmp_path / "s.npz", None, dense=[1, 1, 2, 2])
    out = _ground_truth_feature(path, states=[2, 1], ignore=[0]).apply(
        _tracks({0: [0, 1, 2, 3]})
    )
    row = _row(out, 0)
    assert row["usage__2"] == pytest.approx(0.5)
    assert row["p__1__2"] == pytest.approx(1.0)
    # Grooming is the last bout, so no transition leaves it.
    assert np.isnan(row["p__2__1"])


def test_segments_keep_their_overlap(tmp_path: Path) -> None:
    """rest [0, 10), grooming [3, 5) inside it, grooming [12, 14); 20 frames.

    Painted onto frames the nested grooming would cut rest into two bouts and
    usages would sum to coverage. Read as segments, rest is one ten-frame bout,
    grooming two two-frame bouts, and usage 0.5 + 0.2 exceeds coverage 0.6.
    """
    path = _label_file(tmp_path / "s.npz", _segments())
    row = _row(
        _ground_truth_feature(path, ignore=["none"]).apply(
            _tracks({0: list(range(20))})
        ),
        0,
    )
    assert row["usage__rest"] == pytest.approx(0.5)
    assert row["usage__groom"] == pytest.approx(0.2)
    assert row["coverage"] == pytest.approx(0.6)
    assert row["usage__rest"] + row["usage__groom"] > row["coverage"]
    assert (row["n_bouts"], row["bouts__rest"], row["bouts__groom"]) == (3, 1, 2)
    assert row["bout_mean_s__rest"] == pytest.approx(1.0)
    assert row["bout_median_s__groom"] == pytest.approx(0.2)
    # One minute holds 600 frames; 20 frames is 1/30 of one.
    assert row["rate_per_min__groom"] == pytest.approx(60.0)
    # Onset order rest, groom, groom: one transition, and groom -> groom is none.
    assert row["p__rest__groom"] == pytest.approx(1.0)
    assert np.isnan(row["p__groom__rest"])
    assert row["transition_entropy"] == 0.0
    assert row["entropy"] == pytest.approx(_bits(0.5, 0.2), rel=1e-6)


def test_segments_on_every_other_frame_last_as_long_as_on_every_frame(
    tmp_path: Path,
) -> None:
    """The same segments over tracks holding even frames only: rest [0, 10)
    covers five rows of two frames each, so it still lasts one second, and
    every fraction is what the full table gives."""
    path = _label_file(tmp_path / "s.npz", _segments())
    row = _row(
        _ground_truth_feature(path, ignore=["none"]).apply(
            _tracks({0: list(range(0, 20, 2))})
        ),
        0,
    )
    assert row["n_frames"] == 10
    assert row["usage__rest"] == pytest.approx(0.5)
    assert row["usage__groom"] == pytest.approx(0.2)
    assert row["bout_mean_s__rest"] == pytest.approx(1.0)
    assert row["bout_mean_s__groom"] == pytest.approx(0.2)
    assert row["rate_per_min__groom"] == pytest.approx(60.0)


def test_a_segment_counts_only_the_frames_its_individual_was_present_in(
    tmp_path: Path,
) -> None:
    path = _label_file(tmp_path / "s.npz", _segments())
    row = _row(
        _ground_truth_feature(path, ignore=["none"]).apply(_tracks({0: [0, 1, 2, 3]})),
        0,
    )
    assert row["n_frames"] == 4
    assert row["usage__rest"] == pytest.approx(1.0)
    assert row["usage__groom"] == pytest.approx(0.25)
    assert row["bouts__groom"] == 1
    assert row["bout_mean_s__groom"] == pytest.approx(0.1)


def test_a_segment_naming_an_individual_counts_for_it_alone(tmp_path: Path) -> None:
    segments = LabelSegments(
        start_frame=np.array([0, 3, 12]),
        end_frame=np.array([10, 5, 14]),
        label=np.array([1, 2, 2]),
        individual=np.array([-1, 0, 1]),
    )
    path = _label_file(tmp_path / "s.npz", segments)
    out = _ground_truth_feature(path, ignore=["none"]).apply(
        _tracks({0: list(range(20)), 1: list(range(20))})
    )
    first, second = _row(out, 0), _row(out, 1)
    assert first["usage__groom"] == pytest.approx(0.1)
    assert first["coverage"] == pytest.approx(0.5)
    assert second["usage__groom"] == pytest.approx(0.1)
    assert second["coverage"] == pytest.approx(0.6)
    assert first["usage__rest"] == second["usage__rest"] == pytest.approx(0.5)


def test_min_bout_frames_drops_short_segments(tmp_path: Path) -> None:
    path = _label_file(tmp_path / "s.npz", _segments())
    row = _row(
        _ground_truth_feature(path, ignore=["none"], min_bout_frames=3).apply(
            _tracks({0: list(range(20))})
        ),
        0,
    )
    assert row["bouts__groom"] == 0
    assert row["usage__groom"] == 0.0
    assert row["coverage"] == pytest.approx(0.5)
    assert np.isnan(row["transition_entropy"])


def _event_file(path: Path) -> Path:
    """BORIS-shaped events, listed in this order:

    rest on frames 0-2 for subject 0, groom on frames 5-6 for subject 1, rest on
    frame 8 for the scene, and -- listed last -- groom on frame 2 for subject 0.
    """
    atomic_savez(
        path,
        label_format="individual_pair_v1",
        frames=np.array([0, 1, 2, 5, 6, 8, 2]),
        labels=np.array([1, 1, 1, 2, 2, 1, 2]),
        individual_ids=np.array(
            [[0, -1], [0, -1], [0, -1], [1, -1], [1, -1], [-1, -1], [0, -1]]
        ),
        label_ids=LABEL_IDS,
        label_names=LABEL_NAMES,
    )
    return path


def test_an_event_counts_for_its_subject_and_a_scene_event_for_everyone(
    tmp_path: Path,
) -> None:
    """id 0: rest rest groom . . . . . rest . -- groom on frame 2 was listed last.
    id 1: . . . . . groom groom . rest .

    A reader that dropped the individual-level events, as a symmetric
    deduplication does, would see only the scene's frame 8 for both.
    """
    path = _event_file(tmp_path / "s.npz")
    out = _ground_truth_feature(path, ignore=["none"]).apply(
        _tracks({0: list(range(10)), 1: list(range(10))})
    )
    first, second = _row(out, 0), _row(out, 1)
    assert first["usage__rest"] == pytest.approx(0.3)
    assert first["usage__groom"] == pytest.approx(0.1)
    assert (first["n_bouts"], first["bouts__rest"]) == (3, 2)
    assert first["p__rest__groom"] == pytest.approx(1.0)
    assert first["p__groom__rest"] == pytest.approx(1.0)
    assert second["usage__groom"] == pytest.approx(0.2)
    assert second["usage__rest"] == pytest.approx(0.1)
    assert second["bout_mean_s__groom"] == pytest.approx(0.2)
    assert second["p__groom__rest"] == pytest.approx(1.0)
    assert np.isnan(second["p__rest__groom"])


def test_a_label_id_of_minus_one_is_a_label_not_a_gap(tmp_path: Path) -> None:
    path = tmp_path / "s.npz"
    atomic_savez(
        path,
        labels=np.array([-1, -1, 1, 1]),
        label_ids=np.array([-1, 1]),
        label_names=np.array(["unsure", "rest"], dtype=object),
    )
    out = _ground_truth_feature(path).apply(_tracks({0: [0, 1, 2, 3, 4]}))
    row = _row(out, 0)
    assert row["usage__unsure"] == pytest.approx(0.4)
    # Frame 4 is past the dense array, and only that frame is unlabelled.
    assert row["coverage"] == pytest.approx(0.8)
    ignored = _row(
        _ground_truth_feature(path, ignore=["unsure"]).apply(_tracks({0: [0, 1]})), 0
    )
    assert ignored["coverage"] == 0.0


def test_a_label_file_without_a_vocabulary_needs_states(tmp_path: Path) -> None:
    """Defaulting to the labels present would give entries different columns."""
    path = tmp_path / "s.npz"
    atomic_savez(path, labels=np.array([3, 3, 4, 4]))
    with pytest.raises(ValueError, match="declares no label_ids"):
        _ = _ground_truth_feature(path).apply(_tracks({0: [0, 1, 2, 3]}))
    row = _row(
        _ground_truth_feature(path, states=[3, 4, 5]).apply(_tracks({0: [0, 1, 2, 3]})),
        0,
    )
    assert row["usage__3"] == pytest.approx(0.5)
    assert row["usage__5"] == 0.0


def test_ignoring_every_label_is_refused(tmp_path: Path) -> None:
    """Default states less ignore leaving nothing would emit a row of NaNs."""
    path = _label_file(tmp_path / "s.npz", None, dense=[1, 1, 2, 2])
    feature = _ground_truth_feature(path, ignore=["none", "rest", 2])
    with pytest.raises(ValueError, match="no state left"):
        _ = feature.apply(_tracks({0: [0, 1, 2, 3]}))


def test_an_entry_without_a_label_file_is_refused(tmp_path: Path) -> None:
    path = _label_file(tmp_path / "s.npz", None, dense=[0, 1])
    feature = _ground_truth_feature(path)
    with pytest.raises(ValueError, match="no labels/behavior file"):
        _ = feature.apply(_tracks({0: [0, 1]}, sequence="other"))


# --- a column of another feature's output ---------------------------------------


def test_a_column_named_by_feature_is_joined_on_frame_and_id(tmp_path: Path) -> None:
    """The upstream run covers frames 0-5; frames 6 and 7 are unlabelled."""
    upstream = pd.DataFrame(
        {
            "frame": list(range(6)) * 2,
            "id": [0] * 6 + [1] * 6,
            "syllable": [3, 3, 4, 4, 4, 3] + [4] * 6,
        }
    )
    path = tmp_path / "s.parquet"
    _ = write_parquet_atomic(upstream, path)
    feature = LabelStatistics(
        params={
            "labels": ResultColumn(feature="arhmm", run_id="0.1-x", column="syllable"),
            "states": [3, 4],
            "fps": FPS,
        }
    )
    _ = feature.load_state(Path(), {}, {"labels": {ENTRY: path}})
    out = feature.apply(_tracks({0: list(range(8)), 1: list(range(8))}))

    first = _row(out, 0)
    assert first["n_frames"] == 8
    assert first["usage__3"] == pytest.approx(3 / 8)
    assert first["usage__4"] == pytest.approx(3 / 8)
    assert first["coverage"] == pytest.approx(0.75)
    assert first["p__3__4"] == pytest.approx(1.0)
    assert first["p__4__3"] == pytest.approx(1.0)
    assert _row(out, 1)["usage__4"] == pytest.approx(0.75)


def test_a_column_without_ids_is_scene_level(tmp_path: Path) -> None:
    upstream = pd.DataFrame({"frame": [0, 1, 2, 3], "cluster": [0, 0, 1, 1]})
    path = tmp_path / "s.parquet"
    _ = write_parquet_atomic(upstream, path)
    feature = LabelStatistics(
        params={
            "labels": ResultColumn(feature="global-kmeans", column="cluster"),
            "states": [0, 1],
            "fps": FPS,
        }
    )
    _ = feature.load_state(Path(), {}, {"labels": {ENTRY: path}})
    out = feature.apply(_tracks({0: [0, 1, 2, 3], 1: [2, 3]}))
    assert _row(out, 0)["usage__1"] == pytest.approx(0.5)
    assert _row(out, 1)["usage__1"] == pytest.approx(1.0)


# --- through run_feature --------------------------------------------------------


def _dataset(tmp_path: Path, kind: str = "behavior") -> Dataset:
    """One ``mosaic_v1`` entry ``seq`` with two individuals over 20 frames,
    and a label file of *kind* carrying the nested segments, per individual."""
    dataset = make_dataset(tmp_path / "dataset", name="labels")
    frames = np.arange(20)
    table = pd.DataFrame(
        {
            "frame": np.concatenate([frames, frames]),
            "time": np.concatenate([frames, frames]) / FPS,
            "id": np.repeat([0, 1], 20),
            "group": "",
            "sequence": "seq",
            "X": 10.0,
            "Y": 20.0,
        }
    )
    _ = ensure_track_schema(table, "mosaic_v1", strict=True)
    tracks_run = "convert-pixels.0.1-aaaaaaaaaa"
    tracks_root = tracks_variant_root(dataset.get_root("tracks"), tracks_run)
    tracks_path = tracks_root / f"{make_entry_key('', 'seq')}.parquet"
    _ = write_parquet_atomic(table, tracks_path)
    write_tracks_row(
        dataset,
        run_id=tracks_run,
        group="",
        sequence="seq",
        out_path=tracks_path,
        producer="convert-pixels",
        std_format="mosaic_v1",
        n_rows=len(table),
    )

    labels_run = "convert-labels-segments.0.1-bbbbbbbbbb"
    labels_root = labels_variant_root(dataset.get_root("labels") / kind, labels_run)
    labels_root.mkdir(parents=True, exist_ok=True)
    segments = LabelSegments(
        start_frame=np.array([0, 3, 12]),
        end_frame=np.array([10, 5, 14]),
        label=np.array([1, 2, 2]),
        individual=np.array([-1, 0, 1]),
    )
    labels_path = _label_file(
        labels_root / f"{make_entry_key('', 'seq')}.npz", segments
    )
    write_labels_row(
        dataset,
        run_id=labels_run,
        group="",
        sequence="seq",
        out_path=labels_path,
        producer="convert-labels-segments",
        label_kind=kind,
        label_format="dense",
        n_frames=20,
    )
    return dataset


def test_it_runs_through_run_feature(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = run_feature(
        dataset, LabelStatistics(params={"fps": FPS, "ignore": ["none"]})
    )
    assert result.run_id
    assert not result.failed_entries
    (parquet,) = feature_run_root(dataset, result.feature, result.run_id).glob(
        "*.parquet"
    )
    out = pd.read_parquet(parquet)
    assert out["sequence"].tolist() == ["seq", "seq"]
    first, second = _row(out, 0), _row(out, 1)
    assert first["usage__rest"] == pytest.approx(0.5)
    assert first["usage__groom"] == pytest.approx(0.1)
    assert first["coverage"] == pytest.approx(0.5)
    assert second["coverage"] == pytest.approx(0.6)
    assert second["p__rest__groom"] == pytest.approx(1.0)


def test_a_label_kind_other_than_behavior_resolves_through_its_own_index(
    tmp_path: Path,
) -> None:
    """Labels converted under ``labels/segments/`` are found by naming the kind,
    and are not what the default ``behavior`` kind resolves to."""
    dataset = _dataset(tmp_path, kind="segments")
    result = run_feature(
        dataset,
        LabelStatistics(
            params={"labels": {"kind": "segments"}, "fps": FPS, "ignore": ["none"]}
        ),
    )
    assert result.run_id
    assert not result.failed_entries
    (parquet,) = feature_run_root(dataset, result.feature, result.run_id).glob(
        "*.parquet"
    )
    assert _row(pd.read_parquet(parquet), 1)["coverage"] == pytest.approx(0.6)
    with pytest.raises(FileNotFoundError, match="kind='behavior'"):
        _ = run_feature(dataset, LabelStatistics(params={"fps": FPS}))


def test_a_repeat_is_the_same_run_and_another_setting_is_not(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = run_feature(dataset, LabelStatistics(params={"fps": FPS}))
    again = run_feature(dataset, LabelStatistics(params={"fps": FPS}))
    assert first.run_id == again.run_id
    other = run_feature(
        dataset, LabelStatistics(params={"fps": FPS, "min_bout_frames": 2})
    )
    assert other.run_id != first.run_id


class _Syllables:
    """A stand-in upstream: a per-frame categorical column, as ``arhmm`` writes.

    State 3 on frames 0-4 and 4 after, for every individual.
    """

    name = "test-syllables"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self) -> None:
        self._inputs = self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(None)

    @property
    def inputs(self) -> _Syllables.Inputs:
        return self._inputs

    @property
    def params(self) -> _Syllables.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df["frame"].to_numpy()
        return pd.DataFrame(
            {"frame": frame, "id": df["id"].to_numpy(), "syllable": 3 + (frame >= 5)}
        )


def test_a_column_of_an_upstream_run_is_read_through_its_lookup(
    tmp_path: Path,
) -> None:
    """The ResultColumn names a run; run_feature resolves it to each entry's file."""
    dataset = _dataset(tmp_path)
    upstream = run_feature(dataset, _Syllables())
    column = ResultColumn(column="syllable").from_result(upstream)
    result = run_feature(
        dataset,
        LabelStatistics(params={"labels": column, "states": [3, 4], "fps": FPS}),
    )
    assert result.run_id
    assert not result.failed_entries
    (parquet,) = feature_run_root(dataset, result.feature, result.run_id).glob(
        "*.parquet"
    )
    row = _row(pd.read_parquet(parquet), 1)
    assert row["usage__3"] == pytest.approx(0.25)
    assert row["usage__4"] == pytest.approx(0.75)
    assert row["p__3__4"] == pytest.approx(1.0)
    assert row["bout_mean_s__4"] == pytest.approx(1.5)
