"""`lightning-action` trains on runs of consecutive frames and applies one series at a time.

A temporal model reads one animal's frames in time order. `fit` therefore writes each
run of consecutive labeled frames of one series as a file of its own, from the entries
that the templates run assigned to `train`, standardized by statistics stored with the
model. `apply` writes each series of an entry, an individual or an ordered pair, as a
file of its own and returns the predictions on the rows it was given.

The extra it needs does not resolve on a machine without CUDA -- `lightning-action`
requires `nvidia-dali-cuda110` with no environment marker -- so here the model is
stubbed rather than skipped, and these tests run in every job. The stub keeps the
package's data contract: it reads each CSV with `index_col=0`, needs a label file for
every feature file, cuts whole windows from a file's first row, refuses a file shorter
than a window and its context at predict, and writes the row index first, one column
per class name, and NaN past the last whole window. A stub that wrote CSVs its own way
once hid a fit that raised on every real call. `test_lightning_action_real.py` fits the
real package on CPU in CI's `lightning-action` job.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library.classifier_common import FIT_SCOPE_REMEDY
from mosaic.behavior.feature_library.lightning_action_feature import (
    LightningActionFeature,
    LightningActionModelArtifact,
    LightningActionModelBundle,
)
from mosaic.core.pipeline.types import InputStream, Result
from tests.helpers import (
    LA_CLASS_NAMES,
    LA_CLASSES,
    LA_PAD,
    LA_WINDOW,
    Event,
    LightningActionRecorder,
    StudyEntry,
    TemplatesStudy,
    fit_lightning_action,
    install_lightning_action_stub,
    lightning_action_params,
    stub_probabilities,
    write_templates_study,
)

_CLASSES = LA_CLASSES
_NAMES = LA_CLASS_NAMES
_FEATURES = ["feat_0", "feat_1", "feat_2"]
_WINDOW = LA_WINDOW
Entry = StudyEntry
_PAD = LA_PAD
_DEFAULT = 99  # a class the stub never predicts


@pytest.fixture(autouse=True)
def package(monkeypatch: pytest.MonkeyPatch) -> LightningActionRecorder:
    """Satisfy the package imports with the stub, and return what it records."""
    return install_lightning_action_stub(monkeypatch)


# --- Tables and labels ---------------------------------------------------------


def _table(
    entry: Entry,
    frames: Sequence[int],
    *,
    code: float,
    pair: bool = True,
    nan: tuple[int, str] | None = None,
) -> pd.DataFrame:
    """One entry's rows: `feat_0` is the frame, `feat_1` the entry, `feat_2` the perspective.

    `feat_0` also carries a thousandth of the perspective, so that the two series of a
    pair differ in the one column that the stub predicts from.

    A pair table holds two rows per frame, perspective 0 first, each block in frame
    order. An individual table holds one row per frame of `id` 0.
    """
    blocks: list[pd.DataFrame] = []
    for perspective in (0, 1) if pair else (0,):
        n = len(frames)
        block = pd.DataFrame(
            {
                "frame": np.asarray(frames, dtype=np.int64),
                "group": [entry[0]] * n,
                "sequence": [entry[1]] * n,
            }
        )
        if pair:
            block["id1"] = perspective
            block["id2"] = 1 - perspective
            block["perspective"] = perspective
        else:
            block["id"] = 0
        block["feat_0"] = (
            np.asarray(frames, dtype=np.float64) / 100.0 + perspective / 1000.0
        )
        block["feat_1"] = code
        block["feat_2"] = float(perspective)
        blocks.append(block)
    table = pd.concat(blocks, ignore_index=True)
    if nan is not None:
        row, column = nan
        table.loc[row, column] = np.nan
    return table


def _events(
    frames: Sequence[int],
    label_of: Callable[[int], int | None],
    *,
    pair: bool = True,
    both_directions: bool = False,
) -> list[Event]:
    """The label events of the frames that `label_of` labels."""
    events: list[Event] = []
    for frame in frames:
        label = label_of(frame)
        if label is None:
            continue
        events.append((frame, 0, 1 if pair else -1, label))
        if pair and both_directions:
            events.append((frame, 1, 0, label))
    return events


def _cycling(frame: int) -> int:
    return _CLASSES[(frame // 3) % len(_CLASSES)]


def _study(
    root: Path,
    tables: dict[Entry, tuple[pd.DataFrame, list[Event]]],
    *,
    train: Sequence[Entry],
    test: Sequence[Entry] = (),
) -> TemplatesStudy:
    events = {entry: entry_events for entry, (_, entry_events) in tables.items()}
    return write_templates_study(root, events, train=train, test=test)


def _fit_params(**overrides: object) -> dict[str, object]:
    return lightning_action_params(default_class=_DEFAULT, **overrides)


def _fit(
    study: TemplatesStudy,
    tables: dict[Entry, tuple[pd.DataFrame, list[Event]]],
    **overrides: object,
) -> LightningActionFeature:
    streamed = {entry: table for entry, (table, _) in tables.items()}
    return fit_lightning_action(study, streamed, default_class=_DEFAULT, **overrides)


def _saved(feature: LightningActionFeature, root: Path) -> LightningActionModelBundle:
    feature.save_state(root / "saved")
    return LightningActionModelArtifact().from_path(
        root / "saved" / "lightning_action_model.joblib"
    )


@dataclass(frozen=True)
class _Run:
    """One training file, read back into the values the table held."""

    frames: list[int]
    codes: set[float]
    perspectives: set[float]
    labels: list[int]


def _statistics(
    bundle: LightningActionModelBundle,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    mean = bundle.get("feature_mean")
    std = bundle.get("feature_std")
    assert mean is not None
    assert std is not None
    return np.asarray(mean), np.asarray(std)


def _runs(
    package: LightningActionRecorder, bundle: LightningActionModelBundle
) -> list[_Run]:
    mean, std = _statistics(bundle)
    runs: list[_Run] = []
    for stem in sorted(package.trained):
        features, one_hot = package.trained[stem]
        raw = features.to_numpy(dtype=np.float64) * std + mean
        hot = np.argmax(one_hot.to_numpy(dtype=np.int64), axis=1)
        runs.append(
            _Run(
                frames=[int(round(v * 100.0)) for v in raw[:, 0]],
                codes={round(float(v), 3) for v in raw[:, 1]},
                perspectives={round(float(v), 3) for v in raw[:, 2]},
                labels=[_CLASSES[int(i)] for i in hot],
            )
        )
    return runs


A: Entry = ("train", "a")
B: Entry = ("train", "b")
C: Entry = ("test", "c")


# --- Training ------------------------------------------------------------------


def test_each_training_file_is_one_entitys_consecutive_frames_with_their_labels(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = list(range(23))
    tables = {
        entry: (
            _table(entry, frames, code=code),
            _events(frames, _cycling, both_directions=True),
        )
        for entry, code in ((A, 1.0), (B, 2.0), (C, 3.0))
    }
    study = _study(tmp_path, tables, train=[A, B], test=[C])

    bundle = _saved(_fit(study, tables), tmp_path)

    runs = _runs(package, bundle)
    # Two entries, two perspectives each. A run keeps whole windows from its first
    # frame, so the last three of the 23 frames are not in a file.
    assert len(runs) == 4
    for run in runs:
        assert run.frames == list(range(20))
        assert run.labels == [_cycling(f) for f in range(20)]
        assert len(run.codes) == 1
    assert {code for run in runs for code in run.codes} == {1.0, 2.0}
    assert sorted(tuple(run.perspectives) for run in runs) == [
        (0.0,),
        (0.0,),
        (1.0,),
        (1.0,),
    ]


def test_an_entry_the_templates_run_holds_out_or_does_not_list_never_trains(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = list(range(20))
    unlisted: Entry = ("train", "unlisted")
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0), (C, 3.0), (unlisted, 4.0))
    }
    study = _study(tmp_path, tables, train=[A, B], test=[C])

    bundle = _saved(_fit(study, tables), tmp_path)

    assert {code for run in _runs(package, bundle) for code in run.codes} == {1.0, 2.0}


def test_the_unlabeled_perspective_of_a_pair_writes_no_file(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    """A CalMS21 label covers the resident-focal row, and the intruder-focal row is unlabeled."""
    frames = list(range(20))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0))
    }
    study = _study(tmp_path, tables, train=[A, B])

    bundle = _saved(_fit(study, tables), tmp_path)

    runs = _runs(package, bundle)
    assert len(runs) == 2
    assert all(run.perspectives == {0.0} for run in runs)


def test_a_run_ends_at_a_frame_gap_and_at_an_unlabeled_frame(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = [*range(0, 30), *range(40, 70)]

    def label(frame: int) -> int | None:
        return None if 50 <= frame < 53 else _cycling(frame)

    tables = {
        A: (
            _table(A, frames, code=1.0, pair=False),
            _events(frames, label, pair=False),
        ),
        B: (
            _table(B, frames, code=2.0, pair=False),
            _events(frames, label, pair=False),
        ),
    }
    study = _study(tmp_path, tables, train=[A, B])

    bundle = _saved(_fit(study, tables), tmp_path)

    per_entry = [run.frames for run in _runs(package, bundle) if run.codes == {1.0}]
    # 0-29 is three windows, 40-49 one, and 53-69 holds 17 frames of which 10 train.
    assert per_entry == [
        list(range(30)),
        list(range(40, 50)),
        list(range(53, 63)),
    ]


def test_a_run_shorter_than_a_window_does_not_train(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    short = list(range(_WINDOW - 1))
    long = list(range(_WINDOW))
    tables = {
        A: (
            _table(A, short, code=1.0, pair=False),
            _events(short, _cycling, pair=False),
        ),
        B: (_table(B, long, code=2.0, pair=False), _events(long, _cycling, pair=False)),
    }
    study = _study(tmp_path, tables, train=[A, B])

    bundle = _saved(_fit(study, tables), tmp_path)

    assert [run.codes for run in _runs(package, bundle)] == [{2.0}]


def test_a_fit_with_no_run_of_a_whole_window_says_why(tmp_path: Path) -> None:
    short = list(range(_WINDOW - 1))
    tables = {
        A: (_table(A, short, code=1.0), _events(short, _cycling)),
        B: (_table(B, short, code=2.0), _events(short, _cycling)),
    }
    study = _study(tmp_path, tables, train=[A, B])

    with pytest.raises(RuntimeError, match=r"2 runs are shorter than sequence_length"):
        _fit(study, tables)


def test_a_fit_over_one_class_is_refused(tmp_path: Path) -> None:
    frames = list(range(20))
    tables = {
        A: (_table(A, frames, code=1.0), _events(frames, lambda _: 2)),
        B: (_table(B, frames, code=2.0), _events(frames, lambda _: 2)),
    }
    study = _study(tmp_path, tables, train=[A, B])

    with pytest.raises(RuntimeError, match=r"hold 1 class"):
        _fit(study, tables)


def test_an_entry_without_a_label_file_is_skipped(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = list(range(20))
    bare: Entry = ("train", "bare")
    no_events: list[Event] = []
    tables = {
        A: (_table(A, frames, code=1.0), _events(frames, _cycling)),
        B: (_table(B, frames, code=2.0), _events(frames, _cycling)),
        bare: (_table(bare, frames, code=3.0), no_events),
    }
    study = _study(tmp_path, tables, train=[A, B, bare])

    bundle = _saved(_fit(study, tables), tmp_path)

    assert {code for run in _runs(package, bundle) for code in run.codes} == {1.0, 2.0}


def test_a_train_entry_outside_the_steps_scope_is_refused(tmp_path: Path) -> None:
    """Fitting on fewer entries than the templates name would be a silent narrowing."""
    frames = list(range(20))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0))
    }
    study = _study(tmp_path, tables, train=[A, B])
    narrower = {A: tables[A]}

    with pytest.raises(
        ValueError, match=r"1 train entries .* not in this step's scope"
    ) as refused:
        _fit(study, narrower)

    assert FIT_SCOPE_REMEDY in str(refused.value)


def test_a_templates_run_without_provenance_is_refused(tmp_path: Path) -> None:
    (tmp_path / "templates").mkdir()
    feature = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params=_fit_params(),
    )

    with pytest.raises(ValueError, match=r"template_provenance\.parquet"):
        feature.load_state(
            tmp_path / "run",
            {"templates": tmp_path / "templates" / "templates.parquet"},
            {},
        )


def test_a_window_that_the_context_would_empty_is_refused(tmp_path: Path) -> None:
    frames = list(range(40))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0))
    }
    study = _study(tmp_path, tables, train=[A, B])

    # Six layers of four lags read 504 frames on each side of a frame.
    with pytest.raises(ValueError, match=r"must exceed twice the 504 frames"):
        _fit(study, tables, num_layers=6, num_lags=4)


def test_the_config_puts_the_optimizer_where_the_package_reads_it(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = list(range(20))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0))
    }
    study = _study(tmp_path, tables, train=[A, B])

    _fit(
        study,
        tables,
        optimizer="AdamW",
        learning_rate=0.01,
        weight_decay=0.0005,
        batch_size=4,
    )

    # The package reads `config["optimizer"]` and its weight decay as `wd`. Under
    # `training` the block is ignored without a warning, and its defaults apply.
    assert package.config["optimizer"] == {"type": "AdamW", "lr": 0.01, "wd": 0.0005}
    assert package.config["training"] == {
        "batch_size": 4,
        "num_epochs": 1,
        "device": "cpu",
        "sequence_length": _WINDOW,
        "seed": 42,
        "train_probability": 0.9,
        "val_probability": 0.1,
    }
    # The frames are standardized before they are written, so the package must not
    # standardize each file again.
    assert package.config_value("data", "transforms") == []
    assert package.config_value("model", "input_size") == len(_FEATURES)
    assert package.config_value("model", "output_size") == len(_CLASSES)


# --- Standardization -----------------------------------------------------------


def test_the_statistics_are_those_of_the_training_frames(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = list(range(23))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 5.0), (C, 50.0))
    }
    study = _study(tmp_path, tables, train=[A, B], test=[C])

    bundle = _saved(_fit(study, tables), tmp_path)

    # The training frames are the resident-focal rows of A and B, whole windows only.
    kept = np.array([[f / 100.0, code, 0.0] for code in (1.0, 5.0) for f in range(20)])
    mean, std = _statistics(bundle)
    np.testing.assert_allclose(mean, kept.mean(axis=0))
    np.testing.assert_allclose(
        std, np.where(kept.std(axis=0) > 0, kept.std(axis=0), 1.0)
    )
    assert bundle["feature_columns"] == _FEATURES
    assert bundle.get("sequence_length") == _WINDOW
    assert bundle.get("sequence_pad") == _PAD
    assert bundle["classes"] == _CLASSES


def test_a_missing_value_reads_as_the_column_mean_in_the_files_and_the_statistics(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = list(range(20))
    tables = {
        A: (_table(A, frames, code=1.0, nan=(3, "feat_0")), _events(frames, _cycling)),
        B: (_table(B, frames, code=2.0), _events(frames, _cycling)),
    }
    study = _study(tmp_path, tables, train=[A, B])

    bundle = _saved(_fit(study, tables), tmp_path)

    finite = np.array(
        [f / 100.0 for f in range(20) if f != 3] + [f / 100.0 for f in range(20)]
    )
    assert _statistics(bundle)[0][0] == pytest.approx(finite.mean())
    first = package.trained["run_000000"][0]
    assert first["feat_0"].iloc[3] == 0.0
    assert np.isfinite(first.to_numpy()).all()


# --- Apply ---------------------------------------------------------------------


def _write_bundle(
    run_root: Path,
    *,
    mean: Sequence[float] = (0.0, 0.0, 0.0),
    std: Sequence[float] = (1.0, 1.0, 1.0),
    version: str = "0.5",
    stored: bool = True,
) -> None:
    run_root.mkdir(parents=True)
    bundle: LightningActionModelBundle = {
        "model_dir": "lightning_action_model",
        "feature_columns": list(_FEATURES),
        "classes": list(_CLASSES),
        "class_names": list(_NAMES),
        "config": {},
        "version": version,
    }
    if stored:
        bundle["feature_mean"] = list(mean)
        bundle["feature_std"] = list(std)
        bundle["sequence_length"] = _WINDOW
        bundle["sequence_pad"] = _PAD
    joblib.dump(bundle, run_root / "lightning_action_model.joblib")


def _applier(
    tmp_path: Path,
    *,
    mean: Sequence[float] = (0.0, 0.0, 0.0),
    std: Sequence[float] = (1.0, 1.0, 1.0),
    threshold: float | None = None,
) -> LightningActionFeature:
    """A feature that applies a fitted model, the way a step given `model` does."""
    _write_bundle(tmp_path / "model", mean=mean, std=std)
    params: dict[str, object] = {
        "model": {"feature": "lightning-action"},
        "default_class": _DEFAULT,
        "device": "cpu",
        "decision_threshold": threshold,
    }
    feature = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params=params,
    )
    assert feature.load_state(tmp_path / "model", {}, {})
    return feature


def test_a_pair_table_is_applied_one_ordered_pair_at_a_time(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    n = 23
    df = _table(C, list(range(n)), code=3.0)

    _applier(tmp_path).apply(df)

    [calls] = package.predicted
    assert len(calls) == 2
    for perspective, (_, features) in enumerate(sorted(calls.items())):
        # A whole number of windows, and at least two, whose leading rows are zeros
        # and whose last row is the series' last frame.
        assert len(features) % _WINDOW == 0
        lead = len(features) - n
        assert 0 <= lead < _WINDOW
        assert (features.iloc[:lead].to_numpy() == 0.0).all()
        body = features.iloc[lead:]
        np.testing.assert_allclose(
            body["feat_0"], np.arange(n) / 100.0 + perspective / 1000.0
        )
        assert set(body["feat_2"]) == {float(perspective)}


def test_each_row_takes_the_prediction_of_its_own_frame_and_stays_in_place(
    tmp_path: Path,
) -> None:
    n = 23
    df = _table(C, list(range(n)), code=3.0)
    shuffled = df.sample(frac=1.0, random_state=3)

    out = _applier(tmp_path).apply(shuffled)

    assert list(out.index) == list(shuffled.index)
    for column in ("frame", "group", "sequence", "id1", "id2", "perspective"):
        assert list(out[column]) == list(shuffled[column])
    expected = stub_probabilities(
        shuffled["feat_0"].to_numpy(dtype=np.float64), len(_CLASSES)
    )
    for position, cls in enumerate(_CLASSES):
        np.testing.assert_allclose(out[f"prob_{cls}"], expected[:, position])
    assert list(out["predicted_label"]) == [
        _CLASSES[i] for i in expected.argmax(axis=1)
    ]


@pytest.mark.parametrize("n", [3, 10, 23, 40])
def test_no_row_falls_to_the_default_class_for_lying_past_the_last_window(
    tmp_path: Path, n: int
) -> None:
    """A series shorter than a window, and one that ends inside a window, are predicted."""
    out = _applier(tmp_path).apply(_table(C, list(range(n)), code=3.0))

    prob_columns = [f"prob_{c}" for c in _CLASSES]
    assert len(out) == 2 * n
    assert not out[prob_columns].isna().any(axis=None)
    assert _DEFAULT not in set(out["predicted_label"])


def test_a_row_with_a_missing_key_has_no_series_and_takes_the_default_class(
    tmp_path: Path,
) -> None:
    df = _table(C, list(range(12)), code=3.0)
    df["id1"] = df["id1"].astype(float)
    df.loc[5, "id1"] = np.nan

    out = _applier(tmp_path).apply(df)

    assert out["predicted_label"].iloc[5] == _DEFAULT
    assert np.isnan(
        out.loc[5, ["prob_1", "prob_2", "prob_3"]].to_numpy(dtype=float)
    ).all()
    assert (out.drop(index=5)["predicted_label"] != _DEFAULT).to_numpy().all()


def test_a_frame_that_a_series_holds_twice_is_refused(tmp_path: Path) -> None:
    df = _table(C, list(range(12)), code=3.0, pair=False)
    df.loc[4, "frame"] = 3

    with pytest.raises(ValueError, match=r"frame 3 .* appears in more than one row"):
        _applier(tmp_path).apply(df)


def test_an_individual_table_and_a_table_with_no_identity_are_each_one_series(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    individual = _table(C, list(range(15)), code=3.0, pair=False)
    bare = individual.drop(columns=["id"])

    feature = _applier(tmp_path)
    feature.apply(individual)
    feature.apply(bare)

    assert [len(calls) for calls in package.predicted] == [1, 1]


def test_the_stored_statistics_standardize_a_frame_whatever_its_entry_holds(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    df = _table(C, list(range(12)), code=3.0, nan=(2, "feat_1"))

    _applier(tmp_path, mean=(0.0, 1.0, 0.0), std=(1.0, 2.0, 1.0)).apply(df)

    [calls] = package.predicted
    features = next(iter(calls.values())).iloc[-12:]
    np.testing.assert_allclose(
        features["feat_1"].to_numpy()[[0, 1, 3]], (3.0 - 1.0) / 2.0
    )
    assert features["feat_1"].iloc[2] == 0.0  # the missing value reads as the mean


def test_a_loaded_model_reads_its_prediction_batches_in_this_process(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    """The package starts four worker processes for each series it predicts, by default."""
    _applier(tmp_path).apply(_table(C, list(range(12)), code=3.0))

    assert package.predict_workers == [0]


def test_a_model_that_was_just_fitted_predicts_in_process_and_trained_with_the_defaults(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    frames = list(range(20))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0))
    }
    study = _study(tmp_path, tables, train=[A, B])
    fitted = _fit(study, tables)

    fitted.apply(_table(C, list(range(12)), code=3.0))

    assert package.predict_workers == [0]
    # Training reads its workers from the config it is given, which names none.
    training = package.config_value("training")
    assert isinstance(training, dict)
    assert "num_workers" not in training


def test_a_decision_threshold_leaves_the_rows_no_class_clears_at_the_default_class(
    tmp_path: Path,
) -> None:
    df = _table(C, list(range(12)), code=3.0)

    out = _applier(tmp_path, threshold=0.99).apply(df)

    assert set(out["predicted_label"]) == {_DEFAULT}


def test_a_model_fitted_before_0_5_is_refused_with_the_reason(tmp_path: Path) -> None:
    _write_bundle(tmp_path / "model", version="0.4", stored=False)
    feature = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params={
            "model": {"feature": "lightning-action"},
            "default_class": _DEFAULT,
        },
    )

    with pytest.raises(ValueError, match=r"fitted by version 0\.4, .* Fit it again"):
        feature.load_state(tmp_path / "model", {}, {})


def test_a_fitted_model_applies_the_same_after_it_is_saved_and_loaded(
    tmp_path: Path,
) -> None:
    frames = list(range(23))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0))
    }
    study = _study(tmp_path, tables, train=[A, B])
    fitted = _fit(study, tables)
    df = _table(C, list(range(17)), code=3.0)
    before = fitted.apply(df)
    fitted.save_state(tmp_path / "run")

    reloaded = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params=_fit_params(),
    )
    assert reloaded.load_state(
        tmp_path / "run", {"templates": study.templates}, {"labels": study.label_files}
    )

    pd.testing.assert_frame_equal(reloaded.apply(df), before)


def test_a_bundle_that_fails_to_write_leaves_no_cached_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = list(range(23))
    tables = {
        entry: (_table(entry, frames, code=code), _events(frames, _cycling))
        for entry, code in ((A, 1.0), (B, 2.0))
    }
    fitted = _fit(_study(tmp_path, tables, train=[A, B]), tables)

    def fail(value: object, filename: object) -> None:
        Path(str(filename)).write_bytes(b"partial")
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr(joblib, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        fitted.save_state(tmp_path / "run")

    # `load_state` takes a file at this path for a finished model.
    assert not (tmp_path / "run" / "lightning_action_model.joblib").exists()
    assert not [p for p in (tmp_path / "run").iterdir() if p.suffix == ".tmp"]


# --- Declaration ---------------------------------------------------------------


def test_templates_come_with_labels_and_a_model_with_none() -> None:
    inputs = LightningActionFeature.Inputs((Result(feature="temporal-stack"),))
    templates = {"feature": "extract-labeled-templates"}
    model = {"feature": "lightning-action"}

    with pytest.raises(ValueError, match=r"labels is required with templates"):
        LightningActionFeature(
            inputs, params={"templates": templates, "default_class": 0}
        )
    with pytest.raises(ValueError, match=r"a run given a model fits nothing"):
        LightningActionFeature(
            inputs,
            params={"model": model, "labels": {"kind": "behavior"}, "default_class": 0},
        )


def test_no_run_hashes_the_scope_and_only_a_fit_fits() -> None:
    inputs = LightningActionFeature.Inputs((Result(feature="temporal-stack"),))

    fit = LightningActionFeature(inputs, params=_fit_params())
    apply_only = LightningActionFeature(
        inputs,
        params={"model": {"feature": "lightning-action"}, "default_class": 0},
    )

    assert LightningActionFeature.fit_set_pinned_by == "templates"
    assert fit.scope_dependent is False
    assert apply_only.scope_dependent is False
    assert fit.fits is True
    assert apply_only.fits is False


def test_a_model_that_did_not_resolve_is_not_replaced_by_a_fit(tmp_path: Path) -> None:
    inputs = LightningActionFeature.Inputs((Result(feature="temporal-stack"),))
    feature = LightningActionFeature(
        inputs,
        params={"model": {"feature": "lightning-action"}, "default_class": 0},
    )

    assert feature.load_state(tmp_path, {}, {}) is False
    with pytest.raises(RuntimeError, match="did not resolve to a stored model"):
        feature.fit(InputStream(lambda: iter(()), n_entries=0))


def test_the_step_is_offered_to_a_gpu_worker_and_trains_there_by_default() -> None:
    overrides = _fit_params()
    del overrides["device"]

    assert LightningActionFeature.resource_class == "gpu"
    assert LightningActionFeature.Params.from_overrides(overrides).device == "gpu"
