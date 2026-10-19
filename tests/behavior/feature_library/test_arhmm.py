"""Feature-level tests for the AR-HMM feature (fit/apply, backends, caching)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.arhmm import ArHmmFeature
from mosaic.behavior.feature_library.fit_label_map import (
    LABEL_MAP_FILE,
    FitLabelMap,
    read_label_map,
)
from mosaic.behavior.feature_library.helpers import entry_identity
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.graph import (
    Recipe,
    declaration_catalog,
    resolved_columns,
    resolved_levels,
)
from mosaic.core.pipeline.loading import load_entry_data
from mosaic.core.pipeline.types import InputStream, ParquetLoadSpec, Result
from tests.helpers import run_id_digest, write_sparse_labels

_INPUTS = ArHmmFeature.Inputs((Result(feature="speed-angvel"),))


def _make_sequence_df(
    sequence: str = "s1",
    n: int = 400,
    seed: int = 0,
    n_features: int = 3,
) -> pd.DataFrame:
    """Two separable motion regimes so discrete states are recoverable."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = np.cumsum(rng.standard_normal((half, n_features)) * 0.1, axis=0)
    b = np.cumsum(rng.standard_normal((n - half, n_features)) * 0.1, axis=0) + 5.0
    X = np.vstack([a, b])
    data: dict[str, object] = {
        "frame": np.arange(n),
        "time": np.arange(n, dtype=float) / 30.0,
        "id": np.zeros(n, dtype=int),
        "group": "g",
        "sequence": sequence,
    }
    for i in range(n_features):
        data[f"f{i}"] = X[:, i]
    return pd.DataFrame(data)


def _make_pair_df(
    sequence: str = "s1", n: int = 400, seed: int = 0, n_features: int = 3
) -> pd.DataFrame:
    """One pair's rows, the two perspectives interleaved frame by frame.

    Each perspective is its own two-regime walk, and perspective 1 sits 20 above
    perspective 0, so a model that reads a row after the other animal's decodes
    something other than either walk.
    """
    blocks: list[pd.DataFrame] = []
    for perspective in (0, 1):
        walk = _make_sequence_df(
            sequence, n=n, seed=seed + 10 * perspective, n_features=n_features
        ).drop(columns="id")
        for i in range(n_features):
            walk[f"f{i}"] += 20.0 * perspective
        blocks.append(
            walk.assign(id1=perspective, id2=1 - perspective, perspective=perspective)
        )
    rows = pd.concat(blocks, ignore_index=True)
    return rows.sort_values(["frame", "perspective"], kind="stable").reset_index(
        drop=True
    )


def _make_feature(backend: str = "auto", **params: object) -> ArHmmFeature:
    p: dict[str, object] = {
        "n_states": 6,
        "n_lags": 1,
        "n_iter": 30,
        "backend": backend,
    }
    p.update(params)
    return ArHmmFeature(inputs=_INPUTS, params=p)


def _fit(feat: ArHmmFeature, dfs: list[pd.DataFrame]) -> ArHmmFeature:
    stream = [(f"g__{df['sequence'].iloc[0]}", df) for df in dfs]
    feat.fit(lambda: iter(stream))
    return feat


# --------------------------------------------------------------------------
# Fit + apply
# --------------------------------------------------------------------------


class TestFitApply:
    def test_apply_output_schema(self) -> None:
        feat = _fit(_make_feature("numba"), [_make_sequence_df("s1", seed=1)])
        df = _make_sequence_df("s1", seed=1)
        result = feat.apply(df)

        assert list(result.columns) == ["frame", "syllable", "id", "group", "sequence"]
        assert result["syllable"].dtype == np.int32
        assert len(result) == len(df)
        pd.testing.assert_series_equal(result["frame"], df["frame"], check_names=False)

    def test_apply_raises_before_fit(self) -> None:
        feat = _make_feature("numba")
        with pytest.raises(RuntimeError, match="not fitted"):
            feat.apply(_make_sequence_df())

    def test_fit_raises_without_sequences(self) -> None:
        feat = _make_feature("numba")
        with pytest.raises(RuntimeError, match="No valid sequences"):
            feat.fit(lambda: iter([]))

    def test_jax_backend_not_implemented(self) -> None:
        feat = _make_feature("jax")
        with pytest.raises(NotImplementedError, match="jax"):
            feat.fit(lambda: iter([("g__s1", _make_sequence_df())]))


# --------------------------------------------------------------------------
# Backend equivalence
# --------------------------------------------------------------------------


class TestBackends:
    def test_numba_matches_numpy(self) -> None:
        dfs = [_make_sequence_df("s1", seed=1), _make_sequence_df("s2", seed=2)]
        f_nb = _fit(
            _make_feature("numba", prune_threshold=0.0), [d.copy() for d in dfs]
        )
        f_np = _fit(
            _make_feature("numpy", prune_threshold=0.0), [d.copy() for d in dfs]
        )

        held = _make_sequence_df("s1", seed=1)
        agree = (
            f_nb.apply(held)["syllable"].values == f_np.apply(held)["syllable"].values
        ).mean()
        assert agree >= 0.99

    def test_auto_selects_numba(self) -> None:
        assert _make_feature("auto")._select_backend() == "numba"
        assert _make_feature("numpy")._select_backend() == "numpy"


# --------------------------------------------------------------------------
# Serialization + caching
# --------------------------------------------------------------------------


class TestSerialization:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        feat = _fit(_make_feature("numba"), [_make_sequence_df("s1", seed=1)])
        feat.save_state(tmp_path)
        assert (tmp_path / "arhmm_model.joblib").exists()

        df = _make_sequence_df("s1", seed=1)
        expected = feat.apply(df)

        feat2 = _make_feature("numba")
        assert feat2.load_state(tmp_path, {}, {}) is True
        pd.testing.assert_frame_equal(feat2.apply(df), expected)

    def test_load_state_false_when_missing(self, tmp_path: Path) -> None:
        assert _make_feature().load_state(tmp_path, {}, {}) is False

    def test_numba_model_loads_and_applies_with_numpy(self, tmp_path: Path) -> None:
        """A numba-fit model must decode identically via the numpy backend."""
        feat = _fit(_make_feature("numba"), [_make_sequence_df("s1", seed=3)])
        feat.save_state(tmp_path)

        df = _make_sequence_df("s1", seed=3)
        via_numba = feat.apply(df)

        feat_np = _make_feature("numpy")
        feat_np.load_state(tmp_path, {}, {})
        pd.testing.assert_frame_equal(feat_np.apply(df), via_numba)


def _fitted_columns(run_root: Path) -> list[str]:
    """The columns that the model saved in *run_root* was fitted on."""
    bundle = ArHmmFeature.ModelArtifact().from_path(run_root / "arhmm_model.joblib")
    return bundle["feature_columns"]


class TestColumns:
    """``columns`` names the payload columns that the model is fitted on."""

    def test_the_model_reads_only_the_named_columns(self, tmp_path: Path) -> None:
        """An unnamed column changes neither the fit nor the decoded states."""
        df = _make_sequence_df("s1", seed=4)
        feat = _fit(_make_feature("numpy", columns=["f2", "f0"]), [df])
        feat.save_state(tmp_path)
        noisy = df.assign(f1=np.random.default_rng(5).standard_normal(len(df)))

        assert _fitted_columns(tmp_path) == ["f0", "f2"]
        reference = _fit(_make_feature("numpy"), [df.drop(columns=["f1"])])
        pd.testing.assert_frame_equal(feat.apply(noisy), reference.apply(df))

    def test_unset_columns_fits_every_payload_column(self, tmp_path: Path) -> None:
        feat = _fit(_make_feature("numpy"), [_make_sequence_df("s1", seed=4)])
        feat.save_state(tmp_path)

        assert _fitted_columns(tmp_path) == ["f0", "f1", "f2"]

    @pytest.mark.parametrize(
        ("named", "reason"),
        [
            (["f0", "id"], "Identity or bookkeeping columns: ['id']."),
            (["f0", "f9"], "Not a numeric column of the input: ['f9']."),
        ],
        ids=["identity", "absent"],
    )
    def test_a_name_outside_the_payload_is_refused(
        self, named: list[str], reason: str
    ) -> None:
        feat = _make_feature("numpy", columns=named)

        with pytest.raises(ValueError, match="columns may name only payload") as raised:
            _ = _fit(feat, [_make_sequence_df("s1")])

        assert reason in str(raised.value)
        payload = "The input's payload columns: ['f0', 'f1', 'f2']."
        assert payload in str(raised.value)

    def test_the_order_and_repeats_of_the_names_leave_the_run_id(self) -> None:
        def digest(columns: list[str]) -> str:
            return run_id_digest(_make_feature(columns=columns).params)

        assert digest(["f2", "f0", "f2"]) == digest(["f0", "f2"])
        assert digest(["f0"]) != digest(["f0", "f2"])


class TestPairTable:
    """A pair table is read as one series per ``(id1, id2, perspective)``."""

    _KEY = ["frame", "id1", "id2", "perspective"]

    def test_the_output_carries_the_pair_key(self) -> None:
        df = _make_pair_df()
        feat = _fit(_make_feature("numba"), [df])

        result = feat.apply(df)

        assert list(result.columns) == [
            "frame",
            "syllable",
            *self._KEY[1:],
            "group",
            "sequence",
        ]
        assert len(result) == len(df)
        assert not result.duplicated(self._KEY).any()
        joined = df[self._KEY].merge(
            result, on=self._KEY, how="left", validate="one_to_one"
        )
        assert joined["syllable"].notna().all()

    def test_the_states_merge_one_to_one_with_the_features_they_came_from(
        self, tmp_path: Path
    ) -> None:
        """A step that reads both gets each state on the row it describes."""
        df = _make_pair_df()
        states = _fit(_make_feature("numba"), [df]).apply(df)
        df.to_parquet(tmp_path / "features.parquet")
        states.to_parquet(tmp_path / "states.parquet")

        merged = load_entry_data(
            [
                (tmp_path / "features.parquet", ParquetLoadSpec()),
                (tmp_path / "states.parquet", ParquetLoadSpec()),
            ]
        )

        assert merged is not None
        assert len(merged) == len(df)
        assert merged["syllable"].notna().all()

    def test_each_perspective_decodes_as_its_own_series(self) -> None:
        """A perspective's states do not depend on the rows of the other."""
        df = _make_pair_df()
        feat = _fit(_make_feature("numba"), [df])

        both = feat.apply(df)

        for perspective in (0, 1):
            alone = feat.apply(df[df["perspective"] == perspective])
            decoded = both[both["perspective"] == perspective].reset_index(drop=True)
            pd.testing.assert_frame_equal(decoded, alone.reset_index(drop=True))

    def test_a_series_decodes_as_an_individual_does(self) -> None:
        """The same rows without the pair columns, as one individual, decode alike."""
        df = _make_pair_df()
        feat = _fit(_make_feature("numba"), [df])
        first = df[df["perspective"] == 0]
        as_individual = first.drop(columns=["id1", "id2", "perspective"]).assign(id=0)

        decoded = feat.apply(df).query("perspective == 0")["syllable"]

        np.testing.assert_array_equal(
            decoded.to_numpy(), feat.apply(as_individual)["syllable"].to_numpy()
        )

    def test_the_order_of_the_rows_changes_neither_the_fit_nor_the_output(self) -> None:
        df = _make_pair_df()
        shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)

        reference = _fit(_make_feature("numba"), [df]).apply(df)
        result = _fit(_make_feature("numba"), [shuffled]).apply(shuffled)

        pd.testing.assert_frame_equal(result, reference)

    def test_an_even_downsample_keeps_both_perspectives(self, tmp_path: Path) -> None:
        """Rows alternate perspectives, so an even stride over the table keeps one."""
        df = _make_pair_df()
        feat = _fit(_make_feature("numba", downsample_rate=2), [df])
        feat.save_state(tmp_path)

        bundle = ArHmmFeature.ModelArtifact().from_path(tmp_path / "arhmm_model.joblib")
        result = feat.apply(df)

        # Perspective 1 is 20 above perspective 0, whose walks stay below 6.
        scaler_mean = bundle["scaler_mean"]
        assert scaler_mean is not None
        assert (scaler_mean > 10).all()
        assert len(result) == len(df)
        assert result.groupby("perspective").size().tolist() == [400, 400]


class TestRepeatedRows:
    """A series that repeats a frame is refused, as ``vame`` refuses it."""

    def test_a_pair_series_repeating_a_frame_is_refused_in_fit(self) -> None:
        df = _make_pair_df()
        repeated = pd.concat([df, df.iloc[[5]]], ignore_index=True)

        with pytest.raises(
            ValueError,
            match=r"^arhmm: frame 2 of id1 1, id2 0, perspective 1 appears in more",
        ):
            _fit(_make_feature("numba"), [repeated])

    def test_an_individual_repeating_a_frame_is_refused_in_apply(self) -> None:
        df = _make_sequence_df()
        feat = _fit(_make_feature("numba"), [df])
        repeated = pd.concat([df, df.iloc[[7]]], ignore_index=True)

        with pytest.raises(
            ValueError, match=r"^arhmm: frame 7 of id 0 appears in more than one row$"
        ):
            feat.apply(repeated)


class TestNonFinite:
    """A NaN or an infinity is a missing value, and standardizing does not count it."""

    @staticmethod
    def _with_gaps() -> pd.DataFrame:
        """Two regimes with their first and last 30 frames missing in two columns."""
        df = _make_sequence_df("s1", seed=6)
        df.loc[:29, ["f0", "f2"]] = np.nan
        df.loc[370:, ["f0", "f2"]] = np.inf
        return df

    @staticmethod
    def _scaler(run_root: Path) -> tuple[np.ndarray, np.ndarray]:
        bundle = ArHmmFeature.ModelArtifact().from_path(run_root / "arhmm_model.joblib")
        assert bundle["scaler_mean"] is not None
        assert bundle["scaler_std"] is not None
        return bundle["scaler_mean"], bundle["scaler_std"]

    def test_the_scaler_reads_the_finite_values_only(self, tmp_path: Path) -> None:
        df = self._with_gaps()
        feat = _fit(_make_feature("numpy"), [df])
        feat.save_state(tmp_path)

        values = df[["f0", "f1", "f2"]].to_numpy()
        finite = np.where(np.isfinite(values), values, np.nan)
        mean, std = self._scaler(tmp_path)

        np.testing.assert_allclose(mean, np.nanmean(finite, axis=0))
        np.testing.assert_allclose(std, np.nanstd(finite, axis=0))

    def test_a_missing_value_reads_as_the_column_mean(self, tmp_path: Path) -> None:
        feat = _fit(_make_feature("numpy"), [self._with_gaps()])
        feat.save_state(tmp_path)
        mean, _ = self._scaler(tmp_path)
        gappy = _make_sequence_df("s1", seed=6)
        gappy.loc[100:119, "f0"] = np.nan
        gappy.loc[200:209, "f2"] = np.inf
        filled = gappy.copy()
        filled.loc[100:119, "f0"] = mean[0]
        filled.loc[200:209, "f2"] = mean[2]

        pd.testing.assert_frame_equal(feat.apply(gappy), feat.apply(filled))

    def test_an_unstandardized_missing_value_reads_as_zero(self) -> None:
        feat = _fit(
            _make_feature("numpy", standardize=False), [_make_sequence_df("s1")]
        )
        gappy = _make_sequence_df("s1")
        gappy.loc[100:119, "f1"] = np.nan
        zeroed = gappy.copy()
        zeroed.loc[100:119, "f1"] = 0.0

        pd.testing.assert_frame_equal(feat.apply(gappy), feat.apply(zeroed))

    def test_a_column_without_a_finite_value_is_left_unscaled(
        self, tmp_path: Path
    ) -> None:
        df = _make_sequence_df("s1")
        df["f1"] = np.nan
        feat = _fit(_make_feature("numpy"), [df])
        feat.save_state(tmp_path)

        mean, std = self._scaler(tmp_path)

        assert (mean[1], std[1]) == (0.0, 1.0)
        assert len(feat.apply(df)) == len(df)


class TestShortSeries:
    """A series with no more rows than ``n_lags`` has no state to decode."""

    @staticmethod
    def _with_short_pair(df: pd.DataFrame, rows: int) -> pd.DataFrame:
        """*df* and the first *rows* frames of perspective 0 for the pair (5, 6)."""
        short = df[df["perspective"] == 0].iloc[:rows].assign(id1=5, id2=6)
        return pd.concat([df, short], ignore_index=True)

    def test_a_series_too_short_to_decode_is_left_out(self) -> None:
        df = _make_pair_df()
        feat = _fit(_make_feature("numba", n_lags=2), [df])

        result = feat.apply(self._with_short_pair(df, rows=2))

        pd.testing.assert_frame_equal(result, feat.apply(df))

    def test_a_series_one_row_longer_than_the_lags_is_decoded(self) -> None:
        df = _make_pair_df()
        feat = _fit(_make_feature("numba", n_lags=2), [df])

        result = feat.apply(self._with_short_pair(df, rows=3))

        assert len(result) == len(df) + 3

    def test_the_length_that_counts_is_the_downsampled_one(self) -> None:
        df = _make_pair_df()
        feat = _fit(_make_feature("numba", n_lags=1, downsample_rate=2), [df])

        left_out = feat.apply(self._with_short_pair(df, rows=2))
        decoded = feat.apply(self._with_short_pair(df, rows=3))

        assert len(left_out) == len(df)
        assert len(decoded) == len(df) + 3

    def test_an_entry_of_short_series_alone_gives_an_empty_table(self) -> None:
        df = _make_pair_df()
        feat = _fit(_make_feature("numba"), [df])

        result = feat.apply(df.iloc[:1])

        assert result.empty
        assert list(result.columns) == [
            "frame",
            "syllable",
            "id1",
            "id2",
            "perspective",
            "group",
            "sequence",
        ]

    def test_an_empty_entry_concatenates_with_the_others_without_changing_a_dtype(
        self,
    ) -> None:
        """A reader that stacks the entries of a run keeps integer columns."""
        df = _make_pair_df()
        feat = _fit(_make_feature("numba"), [df])
        decoded = feat.apply(df)

        stacked = pd.concat([decoded, feat.apply(df.iloc[:1])], ignore_index=True)

        assert stacked.dtypes["frame"] == decoded.dtypes["frame"]
        assert stacked.dtypes["id1"] == decoded.dtypes["id1"]
        assert pd.api.types.is_integer_dtype(stacked["syllable"])


class TestEntryIdentity:
    """The output names its entry, as the other fitters' outputs do.

    A step that scores states against labels finds an entry's label file by the
    ``group`` and ``sequence`` of its first row, so a table without them finds none.
    """

    def test_an_individual_output_carries_the_entry_name(self) -> None:
        df = _make_sequence_df("s1")
        result = _fit(_make_feature("numba"), [df]).apply(df)

        assert (result["group"] == "g").all()
        assert (result["sequence"] == "s1").all()
        assert entry_identity(result) == ("g", "s1")

    def test_a_pair_output_carries_the_entry_name(self) -> None:
        df = _make_pair_df("s2")
        result = _fit(_make_feature("numba"), [df]).apply(df)

        assert (result["group"] == "g").all()
        assert (result["sequence"] == "s2").all()
        assert entry_identity(result) == ("g", "s2")

    def test_a_table_without_the_names_gets_none(self) -> None:
        """Only the identity the input carries is kept, as for the series key."""
        df = _make_sequence_df().drop(columns=["group", "sequence"])
        feat = _make_feature("numba")
        feat.fit(lambda: iter([("g__s1", df)]))
        result = feat.apply(df)

        assert list(result.columns) == ["frame", "syllable", "id"]

    def test_an_entry_of_short_series_keeps_the_names_of_its_dtype(self) -> None:
        """An empty entry stacks with the others without turning a column to object."""
        df = _make_pair_df()
        feat = _fit(_make_feature("numba"), [df])
        decoded = feat.apply(df)

        stacked = pd.concat([decoded, feat.apply(df.iloc[:1])], ignore_index=True)

        assert stacked.dtypes["group"] == decoded.dtypes["group"]
        assert stacked.dtypes["sequence"] == decoded.dtypes["sequence"]

    @pytest.mark.parametrize("pair", [False, True])
    def test_the_states_can_be_scored_against_labels(
        self, tmp_path: Path, pair: bool
    ) -> None:
        """``fit-label-map`` reads the states of an ``arhmm`` run, pair or individual."""
        df = _make_pair_df() if pair else _make_sequence_df()
        states = _fit(_make_feature("numba"), [df]).apply(df)
        labels = write_sparse_labels(
            tmp_path / "labels.npz",
            [(frame, 0, 1, 1 if frame < 200 else 0) for frame in range(400)],
        )
        mapper = FitLabelMap(
            FitLabelMap.Inputs((Result(feature="arhmm"),)),
            {"prediction_column": "syllable", "perspective": 0 if pair else None},
        )
        run_root = tmp_path / "run"
        assert not mapper.load_state(run_root, {}, {"labels": {("g", "s1"): labels}})

        mapper.fit(
            InputStream(
                lambda: iter([(make_entry_key("g", "s1"), states)]), n_entries=1
            )
        )
        mapper.save_state(run_root)

        assert read_label_map(run_root / LABEL_MAP_FILE)


class TestDeclaration:
    """The catalog predicts what a step writes, from the level of its input."""

    @staticmethod
    def _recipe(*steps: tuple[str, str, list[object]]) -> Recipe:
        return Recipe.model_validate(
            {
                "steps": [
                    {"id": name, "type": "feature", "feature": feature, "inputs": reads}
                    for name, feature, reads in steps
                ]
            }
        )

    def test_a_pair_input_gives_a_pair_output(self) -> None:
        recipe = self._recipe(
            ("pca", "pair-posedistance-pca", ["tracks"]),
            ("wave", "pair-wavelet", [{"step": "pca"}]),
            ("hmm", "arhmm", [{"step": "wave"}]),
        )

        assert resolved_levels(recipe, declaration_catalog())["hmm"] == "pair"
        written = resolved_columns(recipe)["hmm"]
        assert written.guaranteed == {
            "frame",
            "syllable",
            "id1",
            "id2",
            "perspective",
            "group",
            "sequence",
        }

    def test_an_individual_input_gives_an_individual_output(self) -> None:
        recipe = self._recipe(
            ("speed", "speed-angvel", ["tracks"]),
            ("hmm", "arhmm", [{"step": "speed"}]),
        )

        assert resolved_levels(recipe, declaration_catalog())["hmm"] == "individual"
        assert resolved_columns(recipe)["hmm"].guaranteed == {
            "frame",
            "syllable",
            "id",
            "group",
            "sequence",
        }


class TestCacheKey:
    def test_backend_excluded_from_run_id_hash(self) -> None:
        """`backend` is HASH_EXCLUDE: numpy and numba runs share a cache entry."""
        dump_np = _make_feature("numpy").params.identity_dump()
        dump_nb = _make_feature("numba").params.identity_dump()
        assert dump_np == dump_nb
        assert "backend" not in dump_np
        # ...but it still round-trips through the full model dump.
        assert "backend" in _make_feature("numba").params.model_dump()
