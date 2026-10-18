"""``body-scale`` and ``orientation-rel`` against the per-row loops they replaced.

Both features once looped in Python over ``groupby([frame, id])``, with a ``pdist``
and a Series lookup per row, and ``orientation-rel`` rebuilt the ids of a frame by
scanning every pose it had cached, once per frame. A 750,000-frame entry took
minutes, so a 89-entry run took half an hour. They now compute every row at once.

The loops are kept here as they were, as the reference. The vectorised output has to
equal theirs to float rounding, and exactly in the frame, the ids and the perspective,
wherever the two could differ: a missing keypoint, an individual with too few of them,
a frame with one individual, rows out of order, a scale that does not resolve, a
quantile list that is empty, and a ``nearest_k`` beyond the keypoint count.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from mosaic.behavior.feature_library import helpers as feature_helpers
from mosaic.behavior.feature_library.body_scale import BodyScaleFeature
from mosaic.behavior.feature_library.helpers import entry_identity, wrap_angle
from mosaic.behavior.feature_library.orientation_relative import (
    OrientationRelativeFeature,
    _ordered_pairs,  # pyright: ignore[reportPrivateUsage]
)
from mosaic.core.pipeline.run import process_apply_worker
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pose_columns import pose_column_pairs
from mosaic.core.track_library.calms21 import calms21_to_trex_df
from tests.helpers import write_calms21_npy

N_KEYPOINTS = 7


# --- the loops, as they were -------------------------------------------------


def _reference_body_scale(df: pd.DataFrame) -> pd.DataFrame:
    """``BodyScaleFeature.apply`` before it was vectorised."""
    pose_pairs = pose_column_pairs(df.columns)
    group, sequence = entry_identity(df)
    rows: list[dict[str, int | str | float]] = []
    x_cols = [x for x, _ in pose_pairs]
    y_cols = [y for _, y in pose_pairs]
    for _, sub in df.groupby([C.frame_col, C.id_col], sort=True):
        row = sub.iloc[0]
        xs = row[x_cols].to_numpy(dtype=float)
        ys = row[y_cols].to_numpy(dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < 2:
            continue
        pts = np.column_stack((xs[valid], ys[valid]))
        dists = pdist(pts)
        rows.append(
            {
                C.frame_col: row[C.frame_col],
                C.id_col: row[C.id_col],
                "scale": float(np.median(dists)),
                C.seq_col: sequence,
                C.group_col: group,
            }
        )
    return pd.DataFrame(rows)


def _reference_rotation(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def _reference_local_scale(pts: np.ndarray) -> float | None:
    if pts.shape[0] < 2:
        return None
    dists = pdist(pts)
    val = float(np.median(dists))
    return val if np.isfinite(val) and val > 0 else None


def _reference_orientation_rel(
    df: pd.DataFrame,
    *,
    global_scale: float | None = None,
    quantiles: list[float] | None = None,
    nearest_k: int = 3,
) -> pd.DataFrame:
    """``OrientationRelativeFeature.apply`` before it was vectorised.

    *global_scale* is what ``_scale_for`` resolved. The loop took it as it came, so
    a value it did not check there is checked here, as it was.
    """
    quantiles = [0.25, 0.5, 0.75] if quantiles is None else quantiles
    pose_pairs = pose_column_pairs(df.columns)
    group, sequence = entry_identity(df)
    rows: list[dict[str, object]] = []

    x_cols = [x for x, _ in pose_pairs]
    y_cols = [y for _, y in pose_pairs]

    grouped = df.groupby([C.frame_col, C.id_col], sort=True)
    pose_cache: dict[tuple[int, Any], dict[str, Any]] = {}
    for key, sub in grouped:
        frame_val, id_val = cast("tuple[Any, Any]", key)
        row = sub.iloc[0]
        xs = row[x_cols].to_numpy(dtype=float)
        ys = row[y_cols].to_numpy(dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < 2:
            continue
        arr = np.column_stack((xs[valid], ys[valid]))
        centroid = arr.mean(axis=0)
        angle = float(row.get(C.orientation_col, 0.0))
        pose_cache[(int(frame_val), id_val)] = {
            "pts": arr,
            "centroid": centroid,
            "angle": angle,
        }

    frames = sorted({f for f, _ in pose_cache})
    for f in frames:
        ids_here = [i for (fv, i) in pose_cache if fv == f]
        if len(ids_here) < 2:
            continue
        for id_a in ids_here:
            for id_b in ids_here:
                if id_a == id_b:
                    continue
                cache_a = pose_cache[(f, id_a)]
                cache_b = pose_cache[(f, id_b)]
                pts_b = cache_b["pts"]
                centroid_a = cache_a["centroid"]
                centroid_b = cache_b["centroid"]
                angle_a = cache_a["angle"]
                angle_b = cache_b["angle"]
                scale = global_scale
                if scale is None or not np.isfinite(scale) or scale <= 0:
                    scale = _reference_local_scale(cache_a["pts"])
                if scale is None or scale <= 0:
                    continue
                delta = pts_b - centroid_a
                rot = _reference_rotation(-angle_a)
                rel = (rot @ delta.T).T / scale
                rel_centroid = (rot @ (centroid_b - centroid_a)) / scale
                dtheta = wrap_angle(np.array(angle_b - angle_a))

                dists = np.sqrt((rel**2).sum(axis=1))
                q_vals: list[float] = (
                    np.quantile(dists, quantiles).tolist() if quantiles else []
                )
                d_near = np.sort(dists)[:nearest_k].tolist()
                d_near += [np.nan] * (nearest_k - len(d_near))

                x_vals = rel[:, 0]
                y_vals = rel[:, 1]
                feats: dict[str, object] = {
                    C.frame_col: int(f),
                    "id1": id_a,
                    "id2": id_b,
                    "perspective": 0 if id_a < id_b else 1,
                    C.seq_col: sequence,
                    C.group_col: group,
                    "dx": float(rel_centroid[0]),
                    "dy": float(rel_centroid[1]),
                    "dtheta": float(dtheta),
                    "dist_min": float(np.min(dists)),
                    "dist_median": float(np.median(dists)),
                    "dist_max": float(np.max(dists)),
                }
                for idx, qv in enumerate(q_vals):
                    feats[f"dist_q{int(quantiles[idx] * 100):02d}"] = float(qv)
                feats["x_min"] = float(np.nanmin(x_vals))
                feats["x_median"] = float(np.nanmedian(x_vals))
                feats["x_max"] = float(np.nanmax(x_vals))
                feats["y_min"] = float(np.nanmin(y_vals))
                feats["y_median"] = float(np.nanmedian(y_vals))
                feats["y_max"] = float(np.nanmax(y_vals))
                for i, val in enumerate(d_near):
                    feats[f"near_{i}"] = float(val) if np.isfinite(val) else np.nan
                rows.append(feats)

    return pd.DataFrame(rows)


def _reference_with_warnings_muted(
    reference: Callable[..., pd.DataFrame], *args: object, **kwargs: object
) -> pd.DataFrame:
    """Run a loop whose ``nanmin`` of an all-NaN slice warns, as it always did."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return reference(*args, **kwargs)


# --- tables -------------------------------------------------------------------


def _pose_table(
    n_frames: int = 14,
    ids: tuple[int, ...] = (0, 1),
    *,
    n_keypoints: int = N_KEYPOINTS,
    seed: int = 0,
) -> pd.DataFrame:
    """A frame-major table of *ids* walking about, each with a heading and keypoints."""
    rng = np.random.default_rng(seed)
    frame = np.repeat(np.arange(n_frames, dtype=np.int64), len(ids))
    identity = np.tile(np.asarray(ids, dtype=np.int64), n_frames)
    n = len(frame)
    centre = 200.0 + np.cumsum(rng.normal(0.0, 3.0, (n_frames, len(ids), 2)), axis=0)
    centre = centre.reshape(n, 2)
    offsets = rng.normal(0.0, 8.0, (n, n_keypoints, 2))
    columns: dict[str, object] = {
        "frame": frame,
        "time": frame / 30.0,
        "id": identity,
        "group": "g",
        "sequence": "s",
        "X": centre[:, 0],
        "Y": centre[:, 1],
        "ANGLE": rng.uniform(-np.pi, np.pi, n),
    }
    for k in range(n_keypoints):
        columns[f"poseX{k}"] = centre[:, 0] + offsets[:, k, 0]
        columns[f"poseY{k}"] = centre[:, 1] + offsets[:, k, 1]
    return pd.DataFrame(columns)


def _pose_cells(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith(("poseX", "poseY"))]


def _drop_cells(table: pd.DataFrame, fraction: float, seed: int = 1) -> pd.DataFrame:
    """Blank a *fraction* of the coordinates, an x or a y alone as often as a pair."""
    rng = np.random.default_rng(seed)
    out = table.copy()
    for column in _pose_cells(out):
        out.loc[rng.random(len(out)) < fraction, column] = np.nan
    return out


def _keep_one_keypoint(table: pd.DataFrame, rows: list[int]) -> pd.DataFrame:
    """Leave the rows *rows* one keypoint each: too few to measure a spread."""
    out = table.copy()
    for column in _pose_cells(out):
        if column not in ("poseX0", "poseY0"):
            out.loc[rows, column] = np.nan
    return out


def _unsorted(table: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    return table.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _same(new: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Equal in order, dtype and value; floats to rounding, everything else exactly."""
    pd.testing.assert_frame_equal(
        new, reference, check_dtype=True, check_exact=False, rtol=1e-12, atol=1e-12
    )


def _orientation(
    table: pd.DataFrame,
    *,
    global_scale: float | None = None,
    quantiles: list[float] | None = None,
    nearest_k: int = 3,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The feature's output and the loop's, for one scale the lookup resolved."""
    params: dict[str, object] = {"nearest_k": nearest_k}
    if quantiles is not None:
        params["quantiles"] = quantiles
    feature = OrientationRelativeFeature(params=params)
    if monkeypatch is not None:

        def resolved(group: str, sequence: str) -> float | None:
            return global_scale

        monkeypatch.setattr(feature, "_scale_for", resolved)
    new = feature.apply(table)
    reference = _reference_with_warnings_muted(
        _reference_orientation_rel,
        table,
        global_scale=global_scale,
        quantiles=quantiles,
        nearest_k=nearest_k,
    )
    return new, reference


# --- the row statistics -------------------------------------------------------


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.333, 0.5, 0.9, 1.0])
def test_the_row_statistics_equal_numpy_over_the_entries_that_are_not_nan(
    q: float,
) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(300, 9))
    values[rng.random(values.shape) < 0.4] = np.nan
    values[0] = np.nan  # a row with nothing to measure
    values[1, 1:] = np.nan  # a row with one entry

    ordered, count = feature_helpers.sort_nan_last(values)
    median = feature_helpers.row_median(ordered, count)
    largest = feature_helpers.row_max(ordered, count)
    quantile = feature_helpers.row_quantile(ordered, count, q)

    assert count.tolist() == (~np.isnan(values)).sum(axis=1).tolist()
    for row, present in enumerate(~np.isnan(values)):
        if not present.any():
            assert np.isnan([median[row], largest[row], quantile[row]]).all()
            continue
        kept = values[row][present]
        assert median[row] == np.median(kept)
        assert largest[row] == kept.max()
        assert quantile[row] == np.quantile(kept, q)


# --- body-scale ---------------------------------------------------------------


@pytest.mark.parametrize("ids", [(0,), (0, 1), (0, 1, 2), (3, 7, 11)])
def test_body_scale_matches_the_loop_for_any_number_of_individuals(
    ids: tuple[int, ...],
) -> None:
    table = _pose_table(ids=ids)

    new = BodyScaleFeature().apply(table)

    assert len(new) == len(table)
    _same(new, _reference_body_scale(table))


def test_body_scale_matches_the_loop_with_missing_keypoints() -> None:
    table = _drop_cells(_pose_table(n_frames=40, ids=(0, 1, 2)), 0.35)

    new = BodyScaleFeature().apply(table)

    _same(new, _reference_body_scale(table))
    assert len(new) < len(table)


def test_body_scale_skips_a_row_with_fewer_than_two_valid_keypoints() -> None:
    table = _keep_one_keypoint(_pose_table(), rows=[0, 5, 9])
    # A coordinate pair is valid only when both of its coordinates are.
    table.loc[11, "poseX1"] = np.nan
    table.loc[11, ["poseX0", "poseX2", "poseX3", "poseX4", "poseX5", "poseX6"]] = np.nan

    new = BodyScaleFeature().apply(table)

    _same(new, _reference_body_scale(table))
    assert len(new) == len(table) - 4


def test_body_scale_of_a_table_with_no_measurable_row_is_empty() -> None:
    table = _keep_one_keypoint(_pose_table(n_frames=3), rows=list(range(6)))

    new = BodyScaleFeature().apply(table)

    _same(new, _reference_body_scale(table))
    assert new.empty


def test_body_scale_of_an_empty_table_is_empty() -> None:
    assert BodyScaleFeature().apply(_pose_table().iloc[0:0]).empty


def test_body_scale_matches_the_loop_on_unsorted_rows() -> None:
    table = _unsorted(_drop_cells(_pose_table(n_frames=30, ids=(0, 1, 2)), 0.2))

    new = BodyScaleFeature().apply(table)

    _same(new, _reference_body_scale(table))
    assert new[[C.frame_col, C.id_col]].equals(
        new[[C.frame_col, C.id_col]].sort_values([C.frame_col, C.id_col])
    )


def test_body_scale_takes_the_first_of_a_repeated_frame_and_id() -> None:
    table = _pose_table(n_frames=6)
    repeat = table.iloc[[1, 4]].copy()
    repeat["poseX3"] += 50.0
    table = _unsorted(pd.concat([table, repeat], ignore_index=True), seed=5)

    new = BodyScaleFeature().apply(table)

    _same(new, _reference_body_scale(table))
    assert not new.duplicated([C.frame_col, C.id_col]).any()


def test_body_scale_matches_the_loop_with_gaps_in_the_frames() -> None:
    table = _pose_table(n_frames=20)
    table = table[~table["frame"].isin([3, 4, 5, 11])].reset_index(drop=True)
    table["frame"] = table["frame"] * 7 + 100

    _same(BodyScaleFeature().apply(table), _reference_body_scale(table))


def test_body_scale_has_the_columns_and_dtypes_it_always_had() -> None:
    new = BodyScaleFeature().apply(_pose_table())

    assert list(new.columns) == ["frame", "id", "scale", "sequence", "group"]
    assert new["frame"].dtype == np.int64
    assert new["id"].dtype == np.int64
    assert new["scale"].dtype == np.float64
    assert set(new["sequence"]) == {"s"}
    assert set(new["group"]) == {"g"}
    assert isinstance(new.index, pd.RangeIndex)


def test_body_scale_keeps_the_input_dtype_of_the_frame_and_the_id() -> None:
    """A table with no text column read as float, because the loop took a mixed Series.

    ``row = sub.iloc[0]`` is one dtype when every column is numeric, so the frame and
    the id came out as floats and a join on them missed. The key now keeps the dtype
    it arrived in.
    """
    table = _pose_table().drop(columns=["group", "sequence"])

    new = BodyScaleFeature().apply(table)

    assert new["frame"].dtype == np.int64
    assert new["id"].dtype == np.int64
    assert new["sequence"].tolist() == [""] * len(new)
    assert new["group"].tolist() == [""] * len(new)
    reference = _reference_body_scale(table)
    assert reference["frame"].dtype == np.float64  # what the loop did
    _same(new.astype({"frame": float, "id": float}), reference)


def test_body_scale_matches_the_loop_across_a_block_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_helpers, "_BLOCK_CELLS", 100)
    table = _drop_cells(_pose_table(n_frames=40, ids=(0, 1, 2)), 0.3)

    _same(BodyScaleFeature().apply(table), _reference_body_scale(table))


# --- orientation-rel ----------------------------------------------------------


@pytest.mark.parametrize("ids", [(0, 1), (0, 1, 2), (4, 9, 2, 6, 7, 1)])
def test_orientation_rel_matches_the_loop_for_any_number_of_individuals(
    ids: tuple[int, ...],
) -> None:
    table = _pose_table(n_frames=8, ids=ids)

    new, reference = _orientation(table)

    assert len(new) == 8 * len(ids) * (len(ids) - 1)
    _same(new, reference)


def test_orientation_rel_of_one_individual_is_empty() -> None:
    table = _pose_table(ids=(0,))

    new, reference = _orientation(table)

    assert new.empty
    _same(new, reference)


def test_orientation_rel_of_a_table_with_no_measurable_row_is_empty() -> None:
    table = _keep_one_keypoint(_pose_table(n_frames=3), rows=list(range(6)))

    new, reference = _orientation(table)

    assert new.empty
    _same(new, reference)


def test_orientation_rel_of_an_empty_table_is_empty() -> None:
    assert OrientationRelativeFeature().apply(_pose_table().iloc[0:0]).empty


def test_orientation_rel_matches_the_loop_with_missing_keypoints() -> None:
    table = _drop_cells(_pose_table(n_frames=40, ids=(0, 1, 2)), 0.35)

    new, reference = _orientation(table)

    assert not new.empty
    _same(new, reference)


def test_orientation_rel_matches_the_loop_when_most_keypoints_are_missing() -> None:
    table = _drop_cells(_pose_table(n_frames=120, ids=(0, 1, 2)), 0.55, seed=9)

    new, reference = _orientation(table, nearest_k=5)

    assert not new.empty
    _same(new, reference)


def test_orientation_rel_drops_an_individual_with_fewer_than_two_valid_keypoints() -> (
    None
):
    table = _keep_one_keypoint(_pose_table(ids=(0, 1, 2)), rows=[0, 4, 5, 30])

    new, reference = _orientation(table)

    _same(new, reference)
    # Frame 0 lost id 0, so two remain and make one pair each way. Frame 1 lost ids
    # 1 and 2, so it holds no pair, and frame 2 lost nothing.
    assert len(new[new["frame"] == 0]) == 2
    assert len(new[new["frame"] == 1]) == 0
    assert len(new[new["frame"] == 2]) == 6


def test_orientation_rel_skips_a_frame_with_one_individual() -> None:
    table = _pose_table(n_frames=10, ids=(0, 1))
    alone = table[~((table["frame"].isin([2, 3, 7])) & (table["id"] == 1))]

    new, reference = _orientation(alone.reset_index(drop=True))

    _same(new, reference)
    assert set(new["frame"]) == set(range(10)) - {2, 3, 7}


def test_orientation_rel_matches_the_loop_on_unsorted_rows() -> None:
    table = _unsorted(_drop_cells(_pose_table(n_frames=30, ids=(0, 1, 2)), 0.2))

    new, reference = _orientation(table)

    _same(new, reference)


def test_orientation_rel_takes_the_first_of_a_repeated_frame_and_id() -> None:
    table = _pose_table(n_frames=6)
    repeat = table.iloc[[1, 4]].copy()
    repeat["poseY2"] -= 40.0
    repeat["ANGLE"] += 1.0
    table = _unsorted(pd.concat([table, repeat], ignore_index=True), seed=5)

    new, reference = _orientation(table)

    _same(new, reference)
    assert not new.duplicated(["frame", "id1", "id2"]).any()


def test_orientation_rel_matches_the_loop_with_gaps_in_the_frames() -> None:
    table = _pose_table(n_frames=20, ids=(0, 1, 2))
    table = table[~table["frame"].isin([3, 4, 5, 11])].reset_index(drop=True)
    table["frame"] = table["frame"] * 7 + 100

    new, reference = _orientation(table)

    _same(new, reference)


def test_orientation_rel_with_a_missing_heading_is_nan_for_the_row_and_no_other() -> (
    None
):
    table = _pose_table(n_frames=8, ids=(0, 1, 2))
    table.loc[[3, 10], "ANGLE"] = np.nan

    new, reference = _orientation(table)

    _same(new, reference)
    assert new["x_min"].isna().any()
    assert new["x_min"].notna().any()


@pytest.mark.parametrize(
    "global_scale",
    [
        pytest.param(None, id="none"),
        pytest.param(12.5, id="finite"),
        pytest.param(0.001, id="tiny"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="inf"),
        pytest.param(0.0, id="zero"),
        pytest.param(-3.0, id="negative"),
    ],
)
def test_orientation_rel_matches_the_loop_for_any_global_scale(
    global_scale: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _drop_cells(_pose_table(n_frames=25, ids=(0, 1, 2)), 0.2)

    new, reference = _orientation(
        table, global_scale=global_scale, monkeypatch=monkeypatch
    )

    assert not new.empty
    _same(new, reference)


def test_orientation_rel_skips_a_pair_whose_focal_has_no_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All keypoints of the focal at one point: the local scale is 0, so no pair."""
    table = _pose_table(n_frames=6, ids=(0, 1, 2))
    focal = (table["id"] == 1) & (table["frame"].isin([1, 2]))
    for k in range(N_KEYPOINTS):
        table.loc[focal, f"poseX{k}"] = 5.0
        table.loc[focal, f"poseY{k}"] = 6.0

    new, reference = _orientation(table, global_scale=None, monkeypatch=monkeypatch)

    _same(new, reference)
    assert not ((new["frame"] == 1) & (new["id1"] == 1)).any()
    assert ((new["frame"] == 1) & (new["id2"] == 1)).any()


@pytest.mark.parametrize(
    "quantiles",
    [
        pytest.param(None, id="default"),
        pytest.param([], id="none"),
        pytest.param([0.0, 1.0], id="ends"),
        pytest.param([0.1, 0.9, 0.333], id="odd"),
        pytest.param([0.5], id="median"),
        pytest.param([0.25, 0.255], id="two-quantiles-one-name"),
        pytest.param([0.29, 0.07], id="float-truncation"),
    ],
)
def test_orientation_rel_matches_the_loop_for_any_quantiles(
    quantiles: list[float] | None,
) -> None:
    table = _drop_cells(_pose_table(n_frames=25, ids=(0, 1)), 0.3)

    new, reference = _orientation(table, quantiles=quantiles)

    _same(new, reference)


def test_orientation_rel_refuses_a_quantile_outside_zero_and_one() -> None:
    feature = OrientationRelativeFeature(params={"quantiles": [0.5, 1.5]})

    with pytest.raises(ValueError, match="Quantiles must be in the range"):
        _ = feature.apply(_pose_table())


@pytest.mark.parametrize("nearest_k", [1, 3, 7, 8, 12])
def test_orientation_rel_matches_the_loop_for_any_nearest_k(nearest_k: int) -> None:
    table = _drop_cells(_pose_table(n_frames=25, ids=(0, 1)), 0.3)

    new, reference = _orientation(table, nearest_k=nearest_k)

    _same(new, reference)
    assert [c for c in new.columns if c.startswith("near_")] == [
        f"near_{i}" for i in range(nearest_k)
    ]
    # Past the keypoints there is nothing to be nearest, whatever was missing.
    if nearest_k > N_KEYPOINTS:
        assert new[f"near_{nearest_k - 1}"].isna().all()


def test_orientation_rel_has_the_columns_and_dtypes_it_always_had() -> None:
    new = OrientationRelativeFeature().apply(_pose_table(n_frames=4, ids=(0, 1, 2)))

    assert list(new.columns) == [
        "frame",
        "id1",
        "id2",
        "perspective",
        "sequence",
        "group",
        "dx",
        "dy",
        "dtheta",
        "dist_min",
        "dist_median",
        "dist_max",
        "dist_q25",
        "dist_q50",
        "dist_q75",
        "x_min",
        "x_median",
        "x_max",
        "y_min",
        "y_median",
        "y_max",
        "near_0",
        "near_1",
        "near_2",
    ]
    for column in ("frame", "id1", "id2", "perspective"):
        assert new[column].dtype == np.int64, column
    for column in new.columns.drop(
        ["frame", "id1", "id2", "perspective", "sequence", "group"]
    ):
        assert new[column].dtype == np.float64, column
    assert set(new["sequence"]) == {"s"}
    assert set(new["group"]) == {"g"}
    assert isinstance(new.index, pd.RangeIndex)


def test_orientation_rel_orders_its_rows_by_frame_then_focal_then_other() -> None:
    new = OrientationRelativeFeature().apply(
        _unsorted(_pose_table(n_frames=9, ids=(5, 2, 8)))
    )

    assert new[["frame", "id1", "id2"]].equals(
        new[["frame", "id1", "id2"]].sort_values(["frame", "id1", "id2"])
    )
    lower_first = new["id1"] < new["id2"]
    assert (new["perspective"] == (~lower_first).astype(np.int64)).all()


def test_orientation_rel_without_pose_columns_still_refuses() -> None:
    table = _pose_table().drop(columns=_pose_cells(_pose_table()))

    with pytest.raises(ValueError, match="pose keypoint columns"):
        _ = OrientationRelativeFeature().apply(table)


def test_orientation_rel_matches_the_loop_across_a_block_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_helpers, "_BLOCK_CELLS", 40)
    table = _drop_cells(_pose_table(n_frames=30, ids=(0, 1, 2)), 0.3)

    new, reference = _orientation(table)

    _same(new, reference)


def test_both_features_match_the_loops_when_the_ids_are_text() -> None:
    table = _drop_cells(_pose_table(n_frames=20, ids=(0, 1, 2)), 0.2)
    table["id"] = table["id"].map({0: "mouse-b", 1: "mouse-a", 2: "mouse-c"})

    _same(BodyScaleFeature().apply(table), _reference_body_scale(table))
    new, reference = _orientation(table)
    _same(new, reference)
    assert set(new["id1"]) == {"mouse-a", "mouse-b", "mouse-c"}


# --- the pairs of a frame, and what apply must leave alone --------------------


@pytest.mark.parametrize("seed", range(6))
def test_the_ordered_pairs_are_every_pair_of_a_frame_by_focal_then_other(
    seed: int,
) -> None:
    """The index arithmetic, against a nested loop, over frames of any size."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 9, size=40)
    frames = np.repeat(np.arange(lengths.size) * 3, lengths)

    focal, other = _ordered_pairs(frames)

    expected: list[tuple[int, int]] = []
    for a in range(frames.size):
        for b in range(frames.size):
            if a != b and frames[a] == frames[b]:
                expected.append((a, b))
    assert list(zip(focal.tolist(), other.tolist(), strict=True)) == expected


def test_the_ordered_pairs_of_no_rows_or_of_single_rows_are_none() -> None:
    for frames in (np.array([], dtype=np.int64), np.array([4, 5, 9], dtype=np.int64)):
        focal, other = _ordered_pairs(frames)
        assert focal.size == other.size == 0


def test_orientation_rel_matches_the_loop_when_individuals_come_and_go() -> None:
    """Frames of every size from empty to seven, the ids in each chosen at random."""
    table = _pose_table(n_frames=60, ids=tuple(range(7)), seed=11)
    keep = np.random.default_rng(12).random(len(table)) < 0.55
    table = _drop_cells(table[keep].reset_index(drop=True), 0.2, seed=13)

    new, reference = _orientation(table, nearest_k=4)

    assert new["frame"].nunique() > 20
    _same(new, reference)


@pytest.mark.parametrize(
    "feature_class",
    [BodyScaleFeature, OrientationRelativeFeature],
    ids=lambda c: c.name,
)
def test_apply_leaves_the_table_it_was_given_as_it_was(
    feature_class: type[BodyScaleFeature] | type[OrientationRelativeFeature],
) -> None:
    table = _unsorted(_drop_cells(_pose_table(n_frames=20, ids=(0, 1, 2)), 0.3))
    repeat = table.iloc[[2, 7]].copy()
    repeat["poseX1"] += 9.0
    table = pd.concat([table, repeat])  # a repeated row, and a repeated index label
    before = table.copy(deep=True)

    _ = feature_class().apply(table)

    pd.testing.assert_frame_equal(table, before, check_exact=True)


@pytest.mark.parametrize("feature", ["body-scale", "orientation-rel"])
def test_a_table_whose_index_repeats_gives_the_output_of_one_that_does_not(
    feature: str,
) -> None:
    table = _drop_cells(_pose_table(n_frames=15, ids=(0, 1, 2)), 0.25)
    repeating = table.copy()
    repeating.index = pd.Index(np.arange(len(table)) % 5)
    run: Callable[[pd.DataFrame], pd.DataFrame] = (
        BodyScaleFeature().apply
        if feature == "body-scale"
        else OrientationRelativeFeature().apply
    )

    pd.testing.assert_frame_equal(run(repeating), run(table), check_exact=True)


def test_an_infinite_coordinate_is_a_missing_keypoint_in_both_features() -> None:
    table = _pose_table(n_frames=12, ids=(0, 1))
    table.loc[[1, 4, 9], "poseX2"] = np.inf
    table.loc[[2, 4], "poseY5"] = -np.inf

    _same(BodyScaleFeature().apply(table), _reference_body_scale(table))
    new, reference = _orientation(table)
    _same(new, reference)
    assert np.isfinite(new[["dist_max", "x_max", "y_max"]].to_numpy()).all()


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_both_features_match_the_loops_for_a_frame_stored_as_it_arrived(
    dtype: type,
) -> None:
    table = _drop_cells(_pose_table(n_frames=15, ids=(0, 1, 2)), 0.2)
    table["frame"] = table["frame"].astype(dtype)

    _same(BodyScaleFeature().apply(table), _reference_body_scale(table))
    new, reference = _orientation(table)
    _same(new, reference)


# --- the scale a body-scale run supplies -------------------------------------


def _scale_run(tmp_path: Path, scales: list[float]) -> dict[tuple[str, str], Path]:
    path = tmp_path / "scale.parquet"
    pd.DataFrame({"scale": scales}).to_parquet(path)
    return {("g", "s"): path}


@pytest.mark.parametrize(
    ("scales", "resolved"),
    [
        pytest.param([8.0, 12.0, np.nan], 10.0, id="mean-of-the-finite"),
        pytest.param([np.nan, np.nan], None, id="all-missing"),
        pytest.param([0.0, 0.0], None, id="zero"),
        pytest.param([-1.0, -3.0], None, id="negative"),
    ],
)
def test_orientation_rel_normalises_by_the_scale_a_body_scale_run_resolves(
    tmp_path: Path, scales: list[float], resolved: float | None
) -> None:
    table = _drop_cells(_pose_table(n_frames=20, ids=(0, 1, 2)), 0.2)
    feature = OrientationRelativeFeature()
    assert feature.load_state(tmp_path, {}, {"scale": _scale_run(tmp_path, scales)})

    new = feature.apply(table)

    reference = _reference_with_warnings_muted(
        _reference_orientation_rel, table, global_scale=resolved
    )
    _same(new, reference)


def test_orientation_rel_falls_back_to_the_local_scale_for_a_sequence_without_one(
    tmp_path: Path,
) -> None:
    table = _pose_table(n_frames=10)
    table["sequence"] = "other"
    feature = OrientationRelativeFeature()
    assert feature.load_state(tmp_path, {}, {"scale": _scale_run(tmp_path, [10.0])})

    new = feature.apply(table)

    _same(new, _reference_orientation_rel(table, global_scale=None))


def test_a_body_scale_run_feeds_orientation_rel(tmp_path: Path) -> None:
    """The two features together, as a recipe chains them."""
    table = _drop_cells(_pose_table(n_frames=30, ids=(0, 1)), 0.2)
    body_scale = BodyScaleFeature().apply(table)
    feature = OrientationRelativeFeature()
    assert feature.load_state(
        tmp_path, {}, {"scale": _scale_run(tmp_path, body_scale["scale"].tolist())}
    )

    new = feature.apply(table)

    mean_scale = float(body_scale["scale"].mean())
    _same(new, _reference_orientation_rel(table, global_scale=mean_scale))


# --- a table a converter wrote -----------------------------------------------


def _calms21_table(tmp_path: Path, n_frames: int, seed: int) -> pd.DataFrame:
    """Two mice of seven keypoints, as the CalMS21 converter writes them, with a heading."""
    npy = tmp_path / f"calms{seed}.npy"
    write_calms21_npy(npy, {"train": {"mouse": n_frames}}, seed=seed)
    table = calms21_to_trex_df(npy)
    table["ANGLE"] = np.arctan2(
        table["poseY0"] - table["poseY6"], table["poseX0"] - table["poseX6"]
    )
    return table


def test_both_features_match_the_loops_on_a_converted_calms21_table(
    tmp_path: Path,
) -> None:
    table = _calms21_table(tmp_path, n_frames=400, seed=4)

    _same(BodyScaleFeature().apply(table), _reference_body_scale(table))
    new, reference = _orientation(table)
    assert len(new) == 800
    _same(new, reference)


def test_both_features_match_the_loops_on_a_converted_table_with_missing_keypoints(
    tmp_path: Path,
) -> None:
    table = _drop_cells(_calms21_table(tmp_path, n_frames=400, seed=5), 0.4)
    table.loc[table["frame"] % 37 == 0, "ANGLE"] = np.nan

    _same(BodyScaleFeature().apply(table), _reference_body_scale(table))
    new, reference = _orientation(table)
    _same(new, reference)
    assert new["dist_min"].isna().any()


# --- throughput and the process mode -----------------------------------------

THROUGHPUT_FRAMES = 20_000
SECONDS_FOR_THOSE_FRAMES = 5.0


@pytest.mark.parametrize(
    "feature_class",
    [BodyScaleFeature, OrientationRelativeFeature],
    ids=lambda c: c.name,
)
def test_an_entry_of_twenty_thousand_frames_takes_seconds_not_a_minute(
    feature_class: type[BodyScaleFeature] | type[OrientationRelativeFeature],
) -> None:
    """The loops took 16 s (body-scale) and 32 s (orientation-rel) for this table.

    The vectorised code takes a tenth of a second. The bound sits between the two,
    a factor of three under the faster loop and fifty over the new code, so a loaded
    machine does not trip it and a return to a per-row loop does.
    """
    table = _pose_table(n_frames=THROUGHPUT_FRAMES, ids=(0, 1))
    feature = feature_class()

    started = time.perf_counter()
    result = feature.apply(table)
    elapsed = time.perf_counter() - started

    assert len(result) == len(table)
    assert elapsed < SECONDS_FOR_THOSE_FRAMES, (
        f"{feature_class.name} took {elapsed:.1f} s for {len(table)} rows"
    )


def test_orientation_rel_applies_in_a_worker_process_with_its_scale_loaded_there(
    tmp_path: Path,
) -> None:
    """``parallel_mode='process'`` rebuilds the feature per entry and calls ``load_state``."""
    table = _drop_cells(_pose_table(n_frames=30, ids=(0, 1, 2)), 0.2)
    lookup = _scale_run(tmp_path, [9.0, 11.0])
    feature = OrientationRelativeFeature(params={"nearest_k": 4})

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        result = pool.submit(
            process_apply_worker,
            feature.__module__,
            type(feature).__name__,
            feature.inputs.model_dump(),
            feature.params.model_dump(),
            str(tmp_path),
            {},
            {"scale": lookup},
            table,
        ).result(timeout=120)

    in_process = OrientationRelativeFeature(params={"nearest_k": 4})
    assert in_process.load_state(tmp_path, {}, {"scale": lookup})
    _same(result, in_process.apply(table))
    _same(
        result,
        _reference_with_warnings_muted(
            _reference_orientation_rel, table, global_scale=10.0, nearest_k=4
        ),
    )


def test_one_instance_applied_to_two_entries_carries_nothing_between_them() -> None:
    first = _drop_cells(_pose_table(n_frames=15, ids=(0, 1, 2), seed=1), 0.2)
    second = _pose_table(n_frames=15, ids=(0, 1), seed=2)
    feature = OrientationRelativeFeature()

    _ = feature.apply(first)
    after_another = feature.apply(second)

    _same(after_another, OrientationRelativeFeature().apply(second))
