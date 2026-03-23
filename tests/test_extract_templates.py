"""Tests for ExtractTemplates feature."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.extract_templates import ExtractTemplates
from mosaic.behavior.feature_library.spec import Result

InputFactory = Callable[[], Iterator[tuple[str, pd.DataFrame]]]


# --- Helpers ---


def _make_feature(
    inputs: ExtractTemplates.Inputs | None = None,
    params: dict[str, object] | None = None,
) -> ExtractTemplates:
    if inputs is None:
        inputs = ExtractTemplates.Inputs((Result(feature="speed-angvel"),))
    return ExtractTemplates(inputs=inputs, params=params)


def _make_df(
    n_rows: int,
    n_features: int = 3,
    sequence: str = "seq_a",
    group: str = "grp_a",
    offset: float = 0.0,
) -> pd.DataFrame:
    """Build a DataFrame with metadata columns and n_features numeric columns."""
    rng = np.random.default_rng(hash((sequence, n_rows)) % 2**32)
    data: dict[str, object] = {
        "frame": np.arange(n_rows),
        "time": np.arange(n_rows, dtype=float) / 30.0,
        "id": np.zeros(n_rows, dtype=int),
        "group": [group] * n_rows,
        "sequence": [sequence] * n_rows,
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows) + offset
    return pd.DataFrame(data)


def _make_factory(
    entries: list[tuple[str, pd.DataFrame]],
) -> InputFactory:
    """Wrap a list of (key, df) pairs into an iterator factory."""

    def factory() -> Iterator[tuple[str, pd.DataFrame]]:
        yield from entries

    return factory


def _fit_and_save(
    feat: ExtractTemplates,
    factory: InputFactory,
    run_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit, save, and read back both artifacts."""
    feat.fit(factory)
    feat.save_state(run_root)

    templates = pd.read_parquet(run_root / "templates.parquet")
    provenance = pd.read_parquet(run_root / "template_provenance.parquet")
    return templates, provenance


# --- Exact allocation ---


class TestExactAllocation:
    """Tests for _fit_exact via allocation='exact'."""

    def test_exact_provenance_proportional(self, tmp_path: Path) -> None:
        """Provenance counts should be proportional to entry sizes."""
        df_a = _make_df(100, sequence="seq_a", group="grp_a")
        df_b = _make_df(200, sequence="seq_b", group="grp_b")
        df_c = _make_df(300, sequence="seq_c", group="grp_c")

        pool_size = 60
        feat = _make_feature(
            params={
                "n_templates": pool_size,
                "pool": {"allocation": "exact", "size": pool_size},
            }
        )

        templates, provenance = _fit_and_save(
            feat,
            _make_factory([("a", df_a), ("b", df_b), ("c", df_c)]),
            tmp_path,
        )

        assert len(templates) == pool_size

        prov = provenance.set_index("sequence")
        # 100/600 * 60 = 10, 200/600 * 60 = 20, 300/600 * 60 = 30
        assert prov.loc["seq_a", "count"] == 10
        assert prov.loc["seq_b", "count"] == 20
        assert prov.loc["seq_c", "count"] == 30

    def test_exact_provenance_proportions_sum_to_one(self, tmp_path: Path) -> None:
        df_a = _make_df(50, sequence="seq_a")
        df_b = _make_df(150, sequence="seq_b")

        feat = _make_feature(
            params={
                "n_templates": 40,
                "pool": {"allocation": "exact", "size": 40},
            }
        )
        _, provenance = _fit_and_save(
            feat,
            _make_factory([("a", df_a), ("b", df_b)]),
            tmp_path,
        )

        assert abs(provenance["proportion"].sum() - 1.0) < 1e-9

    def test_exact_small_entry_gets_all_rows(self, tmp_path: Path) -> None:
        """If an entry has fewer rows than its quota, all rows go in."""
        # 3/30 * 30 = 3 quota, entry has exactly 3 rows -> all included
        df_small = _make_df(3, sequence="seq_small")
        df_big = _make_df(27, sequence="seq_big")

        feat = _make_feature(
            params={
                "n_templates": 30,
                "pool": {"allocation": "exact", "size": 30},
            }
        )
        _, provenance = _fit_and_save(
            feat,
            _make_factory([("s", df_small), ("b", df_big)]),
            tmp_path,
        )

        prov = provenance.set_index("sequence")
        assert prov.loc["seq_small", "count"] == 3

    def test_exact_tiny_entry_gets_zero_quota(self, tmp_path: Path) -> None:
        """An entry too small for a proportional share gets 0 templates
        but still appears in provenance."""
        # 5/505 * 50 = 0.49 -> int truncates to 0
        df_tiny = _make_df(5, sequence="seq_tiny")
        df_big = _make_df(500, sequence="seq_big")

        feat = _make_feature(
            params={
                "n_templates": 50,
                "pool": {"allocation": "exact", "size": 50},
            }
        )
        _, provenance = _fit_and_save(
            feat,
            _make_factory([("t", df_tiny), ("b", df_big)]),
            tmp_path,
        )

        prov = provenance.set_index("sequence")
        assert prov.loc["seq_tiny", "count"] == 0
        assert prov.loc["seq_tiny", "proportion"] == 0.0

    def test_exact_preserves_feature_columns(self, tmp_path: Path) -> None:
        df = _make_df(100, n_features=5, sequence="seq_a")

        feat = _make_feature(
            params={
                "n_templates": 20,
                "pool": {"allocation": "exact"},
            }
        )
        templates, _ = _fit_and_save(
            feat,
            _make_factory([("a", df)]),
            tmp_path,
        )

        assert len(templates.columns) == 5
        assert templates.shape == (20, 5)

    def test_exact_with_max_entry_fraction(self, tmp_path: Path) -> None:
        """Cap should limit large entries."""
        df_big = _make_df(900, sequence="seq_big")
        df_small = _make_df(100, sequence="seq_small")

        feat = _make_feature(
            params={
                "n_templates": 100,
                "pool": {
                    "allocation": "exact",
                    "size": 100,
                    "max_entry_fraction": 0.5,
                },
            }
        )
        _, provenance = _fit_and_save(
            feat,
            _make_factory([("big", df_big), ("small", df_small)]),
            tmp_path,
        )

        big_count = provenance.loc[provenance["sequence"] == "seq_big", "count"].item()
        assert big_count <= 50


# --- Reservoir allocation ---


class TestReservoirAllocation:
    """Tests for _fit_reservoir (default allocation)."""

    def test_reservoir_all_data_fits(self, tmp_path: Path) -> None:
        """When total rows < pool_size, all data goes in unsampled."""
        df_a = _make_df(10, sequence="seq_a")
        df_b = _make_df(15, sequence="seq_b")

        feat = _make_feature(
            params={
                "n_templates": 100,
                "pool": {"size": 100},
            }
        )
        templates, provenance = _fit_and_save(
            feat,
            _make_factory([("a", df_a), ("b", df_b)]),
            tmp_path,
        )

        assert len(templates) == 25  # 10 + 15

        prov = provenance.set_index("sequence")
        assert prov.loc["seq_a", "count"] == 10
        assert prov.loc["seq_b", "count"] == 15

    def test_reservoir_output_shape(self, tmp_path: Path) -> None:
        """When total rows > pool_size, output has exactly pool_size rows."""
        entries = [(f"e{i}", _make_df(100, sequence=f"seq_{i}")) for i in range(10)]
        pool_size = 200
        feat = _make_feature(
            params={
                "n_templates": pool_size,
                "pool": {"size": pool_size},
            }
        )
        templates, _ = _fit_and_save(
            feat,
            _make_factory(entries),
            tmp_path,
        )

        assert len(templates) == pool_size

    def test_reservoir_all_entries_represented(self, tmp_path: Path) -> None:
        """With enough pool, all entries should appear in provenance."""
        entries = [(f"e{i}", _make_df(50, sequence=f"seq_{i}")) for i in range(5)]
        feat = _make_feature(
            params={
                "n_templates": 200,
                "pool": {"size": 200},
            }
        )
        _, provenance = _fit_and_save(
            feat,
            _make_factory(entries),
            tmp_path,
        )

        assert len(provenance) == 5

    def test_reservoir_deterministic_with_seed(self, tmp_path: Path) -> None:
        """Same seed should produce identical results."""
        entries = [(f"e{i}", _make_df(200, sequence=f"seq_{i}")) for i in range(5)]

        results: list[pd.DataFrame] = []
        for run in range(2):
            run_root = tmp_path / f"run_{run}"
            feat = _make_feature(
                params={
                    "n_templates": 100,
                    "pool": {"size": 100},
                    "random_state": 123,
                }
            )
            templates, _ = _fit_and_save(
                feat,
                _make_factory(entries),
                run_root,
            )
            results.append(templates)

        pd.testing.assert_frame_equal(results[0], results[1])

    def test_reservoir_default_pool_size_equals_n_templates(
        self, tmp_path: Path
    ) -> None:
        """When pool.size is None, pool defaults to n_templates."""
        df = _make_df(50, sequence="seq_a")

        feat = _make_feature(params={"n_templates": 20})
        templates, _ = _fit_and_save(
            feat,
            _make_factory([("a", df)]),
            tmp_path,
        )

        assert len(templates) == 20


# --- Farthest-first selection ---


class TestFarthestFirst:
    """Tests for farthest-first template selection."""

    def test_farthest_first_picks_spread_points(self, tmp_path: Path) -> None:
        """Given corners of a square, farthest-first should pick all 4."""
        # Build a pool with 4 corners + 6 near-center points
        corners = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        center_noise = np.random.default_rng(0).normal(0.5, 0.01, (6, 2))
        pool = np.vstack([corners, center_noise])

        n = pool.shape[0]
        data: dict[str, object] = {
            "frame": np.arange(n),
            "time": np.arange(n, dtype=float),
            "id": np.zeros(n, dtype=int),
            "group": ["grp"] * n,
            "sequence": ["seq"] * n,
            "feat_0": pool[:, 0],
            "feat_1": pool[:, 1],
        }
        df = pd.DataFrame(data)

        feat = _make_feature(
            params={
                "n_templates": 4,
                "strategy": "farthest_first",
                "pool": {"size": n},
            }
        )
        templates, _ = _fit_and_save(
            feat,
            _make_factory([("a", df)]),
            tmp_path,
        )

        assert templates.shape == (4, 2)

        selected = templates.to_numpy()
        for corner in corners:
            dists = np.linalg.norm(selected - corner, axis=1)
            assert dists.min() < 0.05, f"corner {corner} not selected"

    def test_farthest_first_with_pool_larger_than_n_templates(
        self, tmp_path: Path
    ) -> None:
        """Pool > n_templates: reservoir fills pool, then ff selects."""
        entries = [
            (f"e{i}", _make_df(100, sequence=f"seq_{i}", offset=i * 10.0))
            for i in range(5)
        ]
        feat = _make_feature(
            params={
                "n_templates": 50,
                "strategy": "farthest_first",
                "pool": {"size": 200},
            }
        )
        templates, _ = _fit_and_save(
            feat,
            _make_factory(entries),
            tmp_path,
        )

        assert len(templates) == 50

    def test_farthest_first_provenance_tracks_selection(self, tmp_path: Path) -> None:
        """Provenance should reflect counts after ff selection, not pool."""
        df_a = _make_df(50, sequence="seq_a", offset=0.0)
        df_b = _make_df(50, sequence="seq_b", offset=100.0)

        feat = _make_feature(
            params={
                "n_templates": 10,
                "strategy": "farthest_first",
                "pool": {"size": 100},
            }
        )
        _, provenance = _fit_and_save(
            feat,
            _make_factory([("a", df_a), ("b", df_b)]),
            tmp_path,
        )

        assert provenance["count"].sum() == 10


# --- Save/load round-trip ---


class TestSaveLoad:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        df = _make_df(100, n_features=4, sequence="seq_a")

        feat = _make_feature(
            params={
                "n_templates": 20,
                "pool": {"allocation": "exact"},
            }
        )
        templates_orig, _ = _fit_and_save(
            feat,
            _make_factory([("a", df)]),
            tmp_path,
        )

        # Load into a fresh instance
        feat2 = _make_feature(params={"n_templates": 20})
        loaded = feat2.load_state(tmp_path, {}, {})

        assert loaded is True
        feat2.save_state(tmp_path / "reload")
        templates_reloaded = pd.read_parquet(tmp_path / "reload" / "templates.parquet")

        pd.testing.assert_frame_equal(templates_orig, templates_reloaded)

    def test_load_state_returns_false_when_missing(self, tmp_path: Path) -> None:
        feat = _make_feature(params={"n_templates": 10})
        assert feat.load_state(tmp_path, {}, {}) is False


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_inputs_raises(self) -> None:
        feat = _make_feature(params={"n_templates": 10})
        with pytest.raises(RuntimeError, match="No data"):
            feat.fit(lambda: iter([]))

    def test_single_entry(self, tmp_path: Path) -> None:
        df = _make_df(50, sequence="seq_only")

        feat = _make_feature(
            params={
                "n_templates": 20,
                "pool": {"allocation": "exact"},
            }
        )
        _, provenance = _fit_and_save(
            feat,
            _make_factory([("only", df)]),
            tmp_path,
        )

        assert len(provenance) == 1
        assert provenance.iloc[0]["count"] == 20

    def test_apply_is_passthrough(self) -> None:
        df = _make_df(10, sequence="seq_a")
        feat = _make_feature(params={"n_templates": 5})
        result = feat.apply(df)
        pd.testing.assert_frame_equal(result, df)
