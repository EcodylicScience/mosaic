"""Regression tests: vectorized FFGroups.transform vs reference implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.ffgroups import FFGroups


def _make_track_data(
    n_frames: int = 200,
    n_ids: int = 8,
    *,
    missing_fraction: float = 0.0,
    inf_fraction: float = 0.0,
    duplicate_fraction: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic tracking data."""
    rng = np.random.RandomState(seed)

    frames = np.repeat(np.arange(n_frames), n_ids)
    ids = np.tile(np.arange(n_ids), n_frames)
    x = rng.uniform(0, 500, size=len(frames)).astype(np.float32)
    y = rng.uniform(0, 500, size=len(frames)).astype(np.float32)

    df = pd.DataFrame(
        {
            "frame": frames,
            "id": ids,
            "X": x,
            "Y": y,
            "group": "test_group",
            "sequence": "test_seq",
        }
    )

    n_total = len(df)

    # Introduce missing values (drop some rows)
    if missing_fraction > 0:
        drop_mask = rng.random(n_total) < missing_fraction
        df = df[~drop_mask].reset_index(drop=True)

    # Introduce inf values
    if inf_fraction > 0:
        n_inf = int(len(df) * inf_fraction)
        inf_idx = rng.choice(len(df), size=n_inf, replace=False)
        df.loc[inf_idx[:n_inf // 2], "X"] = np.inf
        df.loc[inf_idx[n_inf // 2:], "Y"] = -np.inf

    # Introduce duplicate (frame, id) rows
    if duplicate_fraction > 0:
        n_dup = int(len(df) * duplicate_fraction)
        dup_idx = rng.choice(len(df), size=n_dup, replace=False)
        dups = df.iloc[dup_idx].copy()
        # Slightly different coordinates for duplicates
        dups["X"] = dups["X"] + rng.uniform(-1, 1, size=len(dups)).astype(np.float32)
        dups["Y"] = dups["Y"] + rng.uniform(-1, 1, size=len(dups)).astype(np.float32)
        df = pd.concat([df, dups], ignore_index=True)

    return df


@pytest.fixture
def feature():
    return FFGroups(
        {"distance_cutoff": 50, "window_size": 5, "min_event_duration": 3}
    )


class TestFFGroupsVectorizedEquivalence:
    """Verify vectorized transform matches the original reference implementation."""

    def test_clean_data(self, feature):
        """All IDs present in every frame, no missing/inf/duplicates."""
        df = _make_track_data(n_frames=100, n_ids=8)
        result = feature.transform(df)
        expected = feature._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_missing_individuals(self, feature):
        """Some (frame, id) pairs missing."""
        df = _make_track_data(n_frames=100, n_ids=8, missing_fraction=0.1)
        result = feature.transform(df)
        expected = feature._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_inf_values(self, feature):
        """Some coordinates are +/- inf."""
        df = _make_track_data(n_frames=100, n_ids=8, inf_fraction=0.05)
        result = feature.transform(df)
        expected = feature._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_duplicate_rows(self, feature):
        """Duplicate (frame, id) pairs with different coordinates."""
        df = _make_track_data(n_frames=100, n_ids=8, duplicate_fraction=0.05)
        result = feature.transform(df)
        expected = feature._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_combined_issues(self, feature):
        """Missing + inf + duplicates combined."""
        df = _make_track_data(
            n_frames=200,
            n_ids=8,
            missing_fraction=0.1,
            inf_fraction=0.03,
            duplicate_fraction=0.03,
        )
        result = feature.transform(df)
        expected = feature._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_window_size_1(self):
        """No smoothing (window_size=1)."""
        feat = FFGroups(
            {"distance_cutoff": 50, "window_size": 1, "min_event_duration": 1}
        )
        df = _make_track_data(n_frames=50, n_ids=4, missing_fraction=0.05)
        result = feat.transform(df)
        expected = feat._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_two_ids(self):
        """Minimal number of individuals."""
        feat = FFGroups(
            {"distance_cutoff": 100, "window_size": 3, "min_event_duration": 1}
        )
        df = _make_track_data(n_frames=50, n_ids=2, missing_fraction=0.1)
        result = feat.transform(df)
        expected = feat._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_single_frame(self, feature):
        """Edge case: only one frame."""
        df = _make_track_data(n_frames=1, n_ids=4)
        result = feature.transform(df)
        expected = feature._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self, feature):
        """Edge case: empty input."""
        df = pd.DataFrame()
        result = feature.transform(df)
        expected = feature._transform_reference(df)
        pd.testing.assert_frame_equal(result, expected)
