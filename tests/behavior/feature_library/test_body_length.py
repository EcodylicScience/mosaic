"""Body length, and the conversion into it, for comparing animals of different sizes.

``body-length`` measures a length along a chain of keypoints, or reads one a
tracker measured, and reports each individual's median over the sequence.
``scale-to-bl`` divides every length column by one length per sequence, so a
table of animals of different ages can be compared in body lengths.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.body_length import BodyLength
from mosaic.behavior.feature_library.scale_to_bl import ScaleToBodyLength
from mosaic.core.pipeline.column_contract import (
    MissingInputColumns,
    check_input_columns,
)


def _fish(lengths: dict[int, float], n: int = 5) -> pd.DataFrame:
    """Straight fish along +x, three keypoints from nose to tail, per id."""
    rows: list[dict[str, object]] = []
    for fid, length in lengths.items():
        for frame in range(n):
            nose = (10.0 * fid + frame, 5.0)
            rows.append(
                {
                    "frame": frame,
                    "time": frame / 10.0,
                    "id": fid,
                    "group": "g",
                    "sequence": "s",
                    "X": nose[0] - length / 2,
                    "Y": nose[1],
                    # nose, mid-body, tail: two segments that sum to the length
                    "poseX0": nose[0],
                    "poseY0": nose[1],
                    "poseX1": nose[0] - length / 2,
                    "poseY1": nose[1],
                    "poseX2": nose[0] - length,
                    "poseY2": nose[1],
                }
            )
    return pd.DataFrame(rows)


# --- body-length ----------------------------------------------------------------


def test_length_along_the_whole_chain() -> None:
    out = BodyLength().apply(_fish({0: 4.0, 1: 12.0}))
    per_id = out.groupby("id")["body_length"].first()
    assert per_id.to_dict() == pytest.approx({0: 4.0, 1: 12.0})
    assert out["body_length_frame"].to_numpy() == pytest.approx(
        out["body_length"].to_numpy()
    )


def test_a_bent_body_is_measured_along_the_bend() -> None:
    df = _fish({0: 4.0}, n=1)
    # Fold the tail 90 degrees at the mid-body: the polyline keeps its length,
    # the nose-to-tail chord shrinks to sqrt(2) * 2.
    df.loc[0, "poseX2"] = df.loc[0, "poseX1"]
    df.loc[0, "poseY2"] = df.loc[0, "poseY1"] + 2.0
    out = BodyLength().apply(df)
    assert out["body_length"].iloc[0] == pytest.approx(4.0)
    chord = BodyLength(params={"keypoints": [0, 2]}).apply(df)
    assert chord["body_length"].iloc[0] == pytest.approx(2.0 * np.sqrt(2.0))


def test_keypoints_by_name() -> None:
    out = BodyLength(
        params={
            "keypoints": ["nose", "tail"],
            "keypoint_names": ["nose", "mid", "tail"],
        }
    ).apply(_fish({0: 6.0}))
    assert out["body_length"].iloc[0] == pytest.approx(6.0)


def test_a_missing_point_leaves_the_frame_unmeasured_not_shorter() -> None:
    df = _fish({0: 4.0}, n=3)
    df.loc[1, "poseX2"] = np.nan
    out = BodyLength().apply(df)
    assert out["frame"].tolist() == [0, 2]
    assert out["body_length"].tolist() == pytest.approx([4.0, 4.0])


def test_the_reported_length_is_the_median_over_frames() -> None:
    df = _fish({0: 4.0}, n=5)
    # One frame with a mistracked tail at twice the length.
    df.loc[2, "poseX2"] = df.loc[2, "poseX0"] - 8.0
    out = BodyLength(params={"keypoints": [0, 2]}).apply(df)
    assert out["body_length_frame"].max() == pytest.approx(8.0)
    assert out["body_length"].to_numpy() == pytest.approx(np.full(5, 4.0))


def test_a_length_column_the_tracker_measured() -> None:
    df = _fish({0: 4.0}, n=3).drop(columns=["poseX0", "poseY0"])
    df["body_length"] = [70.0, 77.0, 80.0]
    out = BodyLength(params={"source": "column"}).apply(df)
    assert out["body_length"].iloc[0] == pytest.approx(77.0)


def test_a_centroid_only_table_refuses_with_a_way_out() -> None:
    df = _fish({0: 4.0}).drop(columns=[c for c in _fish({0: 4.0}) if "pose" in c])
    with pytest.raises(ValueError, match="source='column'"):
        BodyLength().apply(df)


def test_a_chain_of_one_keypoint_refuses_at_construction() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        BodyLength(params={"keypoints": [0]})


def test_a_name_without_names_refuses_at_construction() -> None:
    with pytest.raises(ValueError, match="keypoint_names"):
        BodyLength(params={"keypoints": ["nose", "tail"]})


def test_an_index_past_the_table_refuses() -> None:
    with pytest.raises(ValueError, match="nose|range|index|0..2"):
        BodyLength(params={"keypoints": [0, 7]}).apply(_fish({0: 4.0}))


# --- body-length options -------------------------------------------------------

_KEYPOINTS = 4


def _wiggly(
    frames: int = 9, *, three_d: bool = False, ids: tuple[int, ...] = (0, 1)
) -> pd.DataFrame:
    """Seeded keypoint chains whose per-frame lengths differ, per id."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for identity in ids:
        scale = 3.0 + 2.0 * identity
        for frame in range(frames):
            row: dict[str, object] = {
                "frame": frame,
                "time": frame / 10.0,
                "id": identity,
                "group": "g",
                "sequence": "s",
                "X": 0.0,
                "Y": 0.0,
            }
            if three_d:
                row["Z"] = 0.0
            for k in range(_KEYPOINTS):
                row[f"poseX{k}"] = scale * k + rng.normal()
                row[f"poseY{k}"] = rng.normal()
                if three_d:
                    row[f"poseZ{k}"] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def _coordinates(df: pd.DataFrame, identity: int, axes: str) -> np.ndarray:
    """The ``(frames, keypoints, len(axes))`` positions of one id."""
    rows = df[df["id"] == identity]
    return np.stack(
        [
            rows[[f"pose{axis}{k}" for k in range(_KEYPOINTS)]].to_numpy(dtype=float)
            for axis in axes
        ],
        axis=-1,
    )


def _segment_lengths(df: pd.DataFrame, identity: int, axes: str) -> np.ndarray:
    """The ``(frames, keypoints - 1)`` segment lengths of one id."""
    steps = np.diff(_coordinates(df, identity, axes), axis=1)
    if len(axes) == 2:
        return np.hypot(steps[..., 0], steps[..., 1])
    return np.linalg.norm(steps, axis=-1)


def _chain_lengths(df: pd.DataFrame, identity: int, axes: str = "XY") -> np.ndarray:
    """The per-frame chain length of one id, NaN where any segment is."""
    segments = _segment_lengths(df, identity, axes)
    return np.where(
        np.isfinite(segments).all(axis=1), np.nansum(segments, axis=1), np.nan
    )


def _in_row_order(per_id: Sequence[np.ndarray]) -> np.ndarray:
    """Per-id arrays of equal length in the output's row order, frame then id."""
    stacked = np.stack(per_id, axis=1)
    return stacked.reshape(-1, *stacked.shape[2:])


def _median(values: np.ndarray) -> float:
    return float(np.median(values))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _percentile(q: float) -> Callable[[np.ndarray], float]:
    def reference(values: np.ndarray) -> float:
        return float(np.percentile(values, q, method="linear"))

    return reference


def _sum_of_segment_statistics(
    segments: np.ndarray, reference: Callable[[np.ndarray], float]
) -> float:
    """Each segment's statistic over its finite rows, summed in chain order."""
    total = 0.0
    for segment in range(segments.shape[1]):
        column = segments[:, segment]
        total += float(reference(column[np.isfinite(column)]))
    return total


@pytest.mark.parametrize(
    ("params", "reference"),
    [
        ({}, _median),
        ({"statistic": "median"}, _median),
        ({"statistic": "mean"}, _mean),
        (
            {"statistic": "percentile", "percentile": 25.0},
            _percentile(25.0),
        ),
        (
            {"statistic": "percentile", "percentile": 100.0},
            _percentile(100.0),
        ),
    ],
    ids=["default", "median", "mean", "percentile-25", "percentile-100"],
)
def test_each_statistic_pools_the_rows_of_one_id(
    params: dict[str, object], reference: Callable[[np.ndarray], float]
) -> None:
    df = _wiggly()
    out = BodyLength(params=params).apply(df)
    for identity in (0, 1):
        lengths = _chain_lengths(df, identity)
        rows = out[out["id"] == identity]
        np.testing.assert_array_equal(rows["body_length_frame"].to_numpy(), lengths)
        np.testing.assert_array_equal(
            rows["body_length"].to_numpy(), np.full(len(lengths), reference(lengths))
        )


def test_per_sequence_pools_every_row_of_the_entry() -> None:
    df = _wiggly()
    out = BodyLength(params={"per": "sequence", "statistic": "mean"}).apply(df)
    everything = _in_row_order([_chain_lengths(df, 0), _chain_lengths(df, 1)])
    np.testing.assert_array_equal(
        out["body_length"].to_numpy(), np.full(len(out), np.mean(everything))
    )
    per_id = BodyLength(params={"statistic": "mean"}).apply(df)
    assert not np.array_equal(
        out["body_length"].to_numpy(), per_id["body_length"].to_numpy()
    )


def test_per_sequence_percentile_reads_the_pooled_rows() -> None:
    df = _wiggly()
    out = BodyLength(
        params={"per": "sequence", "statistic": "percentile", "percentile": 10.0}
    ).apply(df)
    everything = np.concatenate([_chain_lengths(df, 0), _chain_lengths(df, 1)])
    np.testing.assert_array_equal(
        out["body_length"].to_numpy(),
        np.full(len(out), np.percentile(everything, 10.0, method="linear")),
    )


def test_a_pooled_statistic_skips_rows_without_a_length() -> None:
    df = _wiggly()
    df.loc[2, "poseX3"] = np.nan
    out = BodyLength(params={"per": "sequence", "statistic": "mean"}).apply(df)
    finite = _in_row_order([_chain_lengths(df, 0), _chain_lengths(df, 1)])
    finite = finite[np.isfinite(finite)]
    assert len(out) == len(finite)
    np.testing.assert_array_equal(
        out["body_length"].to_numpy(), np.full(len(out), np.mean(finite))
    )


def test_a_column_source_takes_the_statistic_and_the_pooling_unit() -> None:
    df = _wiggly().drop(columns=[f"pose{a}{k}" for a in "XY" for k in range(4)])
    df["body_length"] = np.arange(len(df), dtype=float) ** 1.5 + 1.0
    out = BodyLength(
        params={"source": "column", "statistic": "mean", "per": "sequence"}
    ).apply(df)
    in_row_order = np.asarray(
        df.sort_values(["frame", "id"])["body_length"], dtype=float
    )
    np.testing.assert_array_equal(
        out["body_length"].to_numpy(), np.full(len(out), np.mean(in_row_order))
    )


def test_a_three_dimensional_chain_is_the_norm_of_its_steps() -> None:
    df = _wiggly(three_d=True)
    out = BodyLength(params={"dimensions": 3}).apply(df)
    for identity in (0, 1):
        lengths = _chain_lengths(df, identity, "XYZ")
        rows = out[out["id"] == identity]
        np.testing.assert_array_equal(rows["body_length_frame"].to_numpy(), lengths)
        np.testing.assert_array_equal(
            rows["body_length"].to_numpy(),
            np.full(len(lengths), np.median(lengths)),
        )
    planar = df.drop(columns=["Z", *[f"poseZ{k}" for k in range(_KEYPOINTS)]])
    flat = BodyLength().apply(planar)
    assert not np.array_equal(
        flat["body_length_frame"].to_numpy(), out["body_length_frame"].to_numpy()
    )


def _with_gaps() -> pd.DataFrame:
    """Two ids where the tail keypoint is missing on some rows."""
    df = _wiggly(frames=10)
    df.loc[[1, 4], "poseY3"] = np.nan
    df.loc[[12, 13, 17], "poseX3"] = np.nan
    return df


@pytest.mark.parametrize(
    ("params", "reference"),
    [
        ({"segment_statistic": True}, _median),
        ({"segment_statistic": True, "statistic": "mean"}, _mean),
        (
            {
                "segment_statistic": True,
                "statistic": "percentile",
                "percentile": 75.0,
            },
            _percentile(75.0),
        ),
    ],
    ids=["median", "mean", "percentile-75"],
)
@pytest.mark.parametrize("per", ["id", "sequence"])
def test_a_segment_statistic_reads_each_segment_where_it_is_finite(
    params: dict[str, object], reference: Callable[[np.ndarray], float], per: str
) -> None:
    df = _with_gaps()
    out = BodyLength(params={**params, "per": per}).apply(df)

    segments = {identity: _segment_lengths(df, identity, "XY") for identity in (0, 1)}
    chains = {identity: _chain_lengths(df, identity) for identity in (0, 1)}
    pooled = _sum_of_segment_statistics(
        _in_row_order([segments[0], segments[1]]), reference
    )
    for identity in (0, 1):
        expected = (
            _sum_of_segment_statistics(segments[identity], reference)
            if per == "id"
            else pooled
        )
        rows = out[out["id"] == identity]
        finite = np.isfinite(chains[identity])
        assert len(rows) == int(finite.sum())
        np.testing.assert_array_equal(
            rows["body_length_frame"].to_numpy(), chains[identity][finite]
        )
        np.testing.assert_array_equal(
            rows["body_length"].to_numpy(), np.full(len(rows), expected)
        )


def test_a_segment_statistic_differs_from_the_statistic_of_the_chain() -> None:
    df = _with_gaps()
    by_segment = BodyLength(params={"segment_statistic": True}).apply(df)
    by_chain = BodyLength().apply(df)
    assert not np.array_equal(
        by_segment["body_length"].to_numpy(), by_chain["body_length"].to_numpy()
    )


def test_a_three_dimensional_segment_statistic_uses_the_norm() -> None:
    df = _wiggly(three_d=True)
    out = BodyLength(
        params={"dimensions": 3, "per": "sequence", "segment_statistic": True}
    ).apply(df)
    stacked = _in_row_order(
        [_segment_lengths(df, identity, "XYZ") for identity in (0, 1)]
    )
    np.testing.assert_array_equal(
        out["body_length"].to_numpy(),
        np.full(len(out), _sum_of_segment_statistics(stacked, _median)),
    )


def test_a_world_frame_table_under_two_dimensions_is_refused_naming_three() -> None:
    with pytest.raises(ValueError, match="dimensions: 3"):
        BodyLength().apply(_wiggly(three_d=True))


def test_a_planar_table_under_three_dimensions_is_refused() -> None:
    with pytest.raises(ValueError, match="no Z column"):
        BodyLength(params={"dimensions": 3}).apply(_wiggly())


def test_a_chain_keypoint_without_its_height_is_refused_under_three_dimensions() -> (
    None
):
    df = _wiggly(three_d=True).drop(columns=["poseZ2"])
    with pytest.raises(ValueError, match="poseZ2"):
        BodyLength(params={"dimensions": 3}).apply(df)
    out = BodyLength(params={"dimensions": 3, "keypoints": [0, 1, 3]}).apply(df)
    assert len(out) == len(df)


def test_the_height_columns_are_required_only_under_three_dimensions() -> None:
    columns = ["frame", "id", "poseX0", "poseY0", "poseX1", "poseY1"]
    check_input_columns(BodyLength(), columns)
    with pytest.raises(MissingInputColumns, match="poseZ"):
        check_input_columns(BodyLength(params={"dimensions": 3}), columns)
    check_input_columns(
        BodyLength(params={"dimensions": 3}), [*columns, "poseZ0", "poseZ1"]
    )


def test_the_class_computes_in_every_track_space() -> None:
    assert BodyLength.track_spaces == frozenset({"image_px", "plane_cm", "world_m"})


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"statistic": "percentile"}, "statistic='percentile' needs percentile"),
        (
            {"statistic": "median", "percentile": 50.0},
            "percentile is read only with statistic='percentile'",
        ),
        (
            {"statistic": "mean", "percentile": 50.0},
            "percentile is read only with statistic='percentile'",
        ),
        (
            {"statistic": "percentile", "percentile": -1.0},
            "greater than or equal to 0",
        ),
        (
            {"statistic": "percentile", "percentile": 100.5},
            "less than or equal to 100",
        ),
    ],
)
def test_the_percentile_pairs_with_its_statistic_at_construction(
    params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BodyLength(params=params)


@pytest.mark.parametrize(
    ("option", "value"), [("dimensions", 3), ("segment_statistic", True)]
)
def test_a_keypoint_option_is_refused_under_a_column_source(
    option: str, value: object
) -> None:
    with pytest.raises(ValueError, match=f"{option}.*reads keypoints"):
        BodyLength(params={"source": "column", option: value})
    BodyLength(params={"source": "keypoints", option: value})


def test_the_defaults_of_the_keypoint_options_pass_under_a_column_source() -> None:
    BodyLength(params={"source": "column", "dimensions": 2, "segment_statistic": False})


def test_the_new_params_enter_the_identity_only_when_set() -> None:
    default = BodyLength().params.identity_dump()
    for name in ("statistic", "percentile", "per", "dimensions", "segment_statistic"):
        assert name not in default
    explicit = BodyLength(
        params={
            "statistic": "median",
            "per": "id",
            "dimensions": 2,
            "segment_statistic": False,
        }
    ).params.identity_dump()
    assert explicit == default
    moved: list[dict[str, object]] = [
        {"statistic": "mean"},
        {"statistic": "percentile", "percentile": 50.0},
        {"per": "sequence"},
        {"dimensions": 3},
        {"segment_statistic": True},
    ]
    dumps = [BodyLength(params=p).params.identity_dump() for p in moved]
    assert all(dump != default for dump in dumps)
    assert len({repr(sorted(dump.items())) for dump in dumps}) == len(dumps)


# --- scale-to-bl ----------------------------------------------------------------


def test_an_empty_columns_list_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        _ = ScaleToBodyLength(params={"length": 4.0, "columns": []})


def test_a_named_column_the_input_lacks_is_refused_before_apply() -> None:
    feature = ScaleToBodyLength(params={"length": 4.0, "columns": ["X", "nope"]})
    with pytest.raises(MissingInputColumns, match="lacks 'nope'"):
        check_input_columns(feature, ["X", "Y"])
    check_input_columns(feature, ["X", "nope"])


def test_an_explicit_length_divides_every_length_column() -> None:
    df = _fish({0: 4.0})
    out = ScaleToBodyLength(params={"length": 4.0}).apply(df)
    assert out["X_bl"].to_numpy() == pytest.approx(df["X"].to_numpy() / 4.0)
    assert out["poseX2_bl"].to_numpy() == pytest.approx(df["poseX2"].to_numpy() / 4.0)
    assert "time_bl" not in out.columns
    assert "frame_bl" not in out.columns


def test_convert_mode_is_track_shaped() -> None:
    df = _fish({0: 4.0})
    out = ScaleToBodyLength(params={"length": 2.0, "mode": "convert"}).apply(df)
    assert list(out.columns) == list(df.columns)
    assert out["Y"].to_numpy() == pytest.approx(df["Y"].to_numpy() / 2.0)
    assert out["time"].equals(df["time"])


def test_a_run_needs_a_length() -> None:
    with pytest.raises(ValueError, match="needs a length"):
        ScaleToBodyLength()


def test_a_run_takes_one_length() -> None:
    with pytest.raises(ValueError, match="both set"):
        ScaleToBodyLength(
            params={"length": 1.0, "body_length": {"feature": "body-length"}}
        )


def test_a_non_positive_length_refuses() -> None:
    with pytest.raises(ValueError, match="not a usable body length"):
        ScaleToBodyLength(params={"length": 0.0}).apply(_fish({0: 4.0}))


def test_the_length_is_one_per_sequence_from_body_length(tmp_path: Path) -> None:
    """Every individual is divided by the median of the individuals' lengths."""
    df = _fish({0: 4.0, 1: 6.0, 2: 20.0})
    lengths = BodyLength().apply(df)
    path = tmp_path / "s.parquet"
    lengths.to_parquet(path)

    feat = ScaleToBodyLength(params={"body_length": {"feature": "body-length"}})
    assert feat.load_state(tmp_path, {}, {"body_length": {("g", "s"): path}})
    out = feat.apply(df)
    # median(4, 6, 20) = 6 for every individual, so relative positions hold.
    assert out["X_bl"].to_numpy() == pytest.approx(df["X"].to_numpy() / 6.0)


def test_a_missing_body_length_run_refuses_once_at_load(tmp_path: Path) -> None:
    feat = ScaleToBodyLength(params={"body_length": {"feature": "body-length"}})
    with pytest.raises(ValueError, match="resolved to no entries"):
        feat.load_state(tmp_path, {}, {})


def test_an_entry_body_length_did_not_measure_refuses(tmp_path: Path) -> None:
    feat = ScaleToBodyLength(params={"body_length": {"feature": "body-length"}})
    feat.load_state(tmp_path, {}, {"body_length": {("g", "other"): tmp_path / "x"}})
    with pytest.raises(ValueError, match="no output for"):
        feat.apply(_fish({0: 4.0}))
