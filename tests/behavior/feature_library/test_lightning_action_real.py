"""`lightning-action` fits and applies with the real package, on CPU.

The stubbed tests in `test_lightning_action_feature.py` mirror the package's data
contract, but only a real fit proves it: through 0.3 the feature wrote its CSVs without
the index column the package reads with `index_col=0`, so the package dropped the first
feature column and every fit raised a channel-count error. Through 0.4 the optimizer
block sat where the package does not read it, so every fit trained at the package's own
weight decay and ignored the learning rate and the optimizer that the params named.

The `lightning-action` extra does not install without CUDA, because the package
requires a DALI build that only builds with CUDA. CI's `lightning-action` job
installs the package with `--no-deps` beside its pure-Python requirements and sets
MOSAIC_CI_LIGHTNING_ACTION, which makes a missing package an error rather than a
skip.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.lightning_action_feature import (
    LightningActionFeature,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, run_pipeline
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    Event,
    StudyEntry,
    calms21_dataset,
    fit_lightning_action,
    write_templates_study,
)

_HAS_LIGHTNING_ACTION = importlib.util.find_spec("lightning_action") is not None

_CLASSES = [0, 1, 2]
_FEATURES = [f"feat_{i}" for i in range(5)]
_WINDOW = 50
_BOUT = 25
_LEARNING_RATE = 0.01
_WEIGHT_DECAY = 0.0005
TRAIN: list[StudyEntry] = [("train", "a"), ("train", "b")]

pytestmark = [
    pytest.mark.lightning_action,
    pytest.mark.skipif(
        not _HAS_LIGHTNING_ACTION, reason="requires the lightning-action package"
    ),
    # Lightning's advice for a tiny CPU run, and deprecations inside the packages.
    pytest.mark.filterwarnings("ignore:GPU available but not used:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not have many workers:UserWarning"),
    pytest.mark.filterwarnings("ignore:The number of training batches:UserWarning"),
    pytest.mark.filterwarnings("ignore:To copy construct from a tensor:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:`isinstance\\(treespec, LeafSpec\\)`:FutureWarning"
    ),
]


@pytest.fixture(scope="module")
def in_process_loaders() -> Iterator[None]:
    """Keep the package's data loaders in this process.

    They default to four persistent worker processes. On macOS those are spawned,
    and under pytest's output capture they hang. The package's config passes
    `num_workers` and no `persistent_workers`, which then refuses zero workers, so
    both are set on the data module itself.
    """
    from lightning_action.data import DataModule

    init = DataModule.__init__

    def in_process(self: DataModule, **kwargs: object) -> None:
        kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
        init(self, **kwargs)  # pyright: ignore[reportArgumentType]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DataModule, "__init__", in_process)
        yield


def _label(frame: int) -> int:
    return _CLASSES[(frame // _BOUT) % len(_CLASSES)]


def _pair_table(entry: StudyEntry, n_frames: int, seed: int) -> pd.DataFrame:
    """A pair table whose resident-focal rows show their class in one feature.

    Each class shifts its own feature of perspective 0. Perspective 1 is noise,
    and its rows are unlabeled.
    """
    rng = np.random.default_rng(seed)
    blocks: list[pd.DataFrame] = []
    for perspective in (0, 1):
        features = rng.standard_normal((n_frames, len(_FEATURES)))
        if perspective == 0:
            for frame in range(n_frames):
                features[frame, _label(frame)] += 3.0
        block = pd.DataFrame(features, columns=_FEATURES)
        block.insert(0, "frame", np.arange(n_frames))
        block["group"] = entry[0]
        block["sequence"] = entry[1]
        block["id1"] = perspective
        block["id2"] = 1 - perspective
        block["perspective"] = perspective
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _events(n_frames: int) -> list[Event]:
    return [(frame, 0, 1, _label(frame)) for frame in range(n_frames)]


@pytest.fixture(scope="module")
def fitted(
    in_process_loaders: None, tmp_path_factory: pytest.TempPathFactory
) -> tuple[LightningActionFeature, Path]:
    """A feature fitted on two 400-frame entries, and the run root it was saved to."""
    root = tmp_path_factory.mktemp("la_real")
    tables = {entry: _pair_table(entry, 400, seed) for seed, entry in enumerate(TRAIN)}
    study = write_templates_study(
        root, {entry: _events(400) for entry in TRAIN}, train=TRAIN
    )
    feature = fit_lightning_action(
        study,
        tables,
        default_class=3,
        num_epochs=60,
        sequence_length=_WINDOW,
        batch_size=4,
        num_hid_units=16,
        num_layers=2,
        num_lags=4,
        learning_rate=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
    )
    feature.save_state(root / "saved")
    return feature, root / "saved"


def test_a_fitted_model_applies_to_a_pair_table_after_it_is_saved_and_loaded(
    fitted: tuple[LightningActionFeature, Path],
) -> None:
    feature, saved = fitted
    entry: StudyEntry = ("test", "c")
    # 333 frames end inside a window, and every row must still be predicted.
    df = _pair_table(entry, 333, seed=9)
    reloaded = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params={
            "model": {"feature": "lightning-action"},
            "default_class": 3,
            "device": "cpu",
        },
    )
    assert reloaded.load_state(saved, {}, {})

    out = reloaded.apply(df)

    prob_cols = [f"prob_{c}" for c in _CLASSES]
    assert len(out) == len(df)
    for column in ("frame", "group", "sequence", "id1", "id2", "perspective"):
        assert list(out[column]) == list(df[column])
    assert not out[prob_cols].isna().any(axis=None)
    np.testing.assert_allclose(out[prob_cols].sum(axis=1), 1.0, rtol=1e-4)
    assert 3 not in set(out["predicted_label"])
    resident = out["perspective"] == 0
    truth = [_label(frame) for frame in out.loc[resident, "frame"]]
    accuracy = float(np.mean(out.loc[resident, "predicted_label"] == truth))
    assert accuracy > 0.85
    # What was fitted and what was loaded predict alike.
    np.testing.assert_allclose(
        feature.apply(df)[prob_cols], out[prob_cols], rtol=1e-4, atol=1e-5
    )


def test_the_package_builds_the_optimizer_that_the_params_name(
    fitted: tuple[LightningActionFeature, Path],
) -> None:
    """The package reads its optimizer from the top level of the config it was saved with."""
    from lightning_action.api.model import Model

    _, saved = fitted

    model = Model.from_dir(str(saved / "lightning_action_model"))
    group = model.model.configure_optimizers()["optimizer"].param_groups[0]

    assert group["lr"] == pytest.approx(_LEARNING_RATE)
    assert group["weight_decay"] == pytest.approx(_WEIGHT_DECAY)


TRAIN_GROUP, BOTH_GROUPS = Scope(groups=["train"]), Scope(groups=["train", "test"])


def _dataset_and_recipe(base: Path) -> tuple[Dataset, Recipe]:
    """Six training entries and two test entries, and the chain that fits on them."""
    ds = calms21_dataset(
        base,
        {
            # Eighteen windows of the training entries. The package validates on a
            # tenth of its windows, and warns when it has fewer than ten.
            "train": {"a": {f"s{i}": 150 for i in range(1, 7)}},
            "test": {"a": {"t1": 150, "t2": 83}},
        },
        seed=11,
        n_classes=3,
    )
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "sequence-classifier",
            "steps": [
                CONVERT_TRACKS_STEP,
                CONVERT_LABELS_STEP,
                {
                    "id": "heading",
                    "type": "feature",
                    "feature": "heading",
                    "inputs": ["tracks"],
                    "tracks": {"step": "tracks"},
                    "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
                },
                {
                    "id": "pairs",
                    "type": "feature",
                    "feature": "pair-position",
                    "inputs": ["tracks", {"step": "heading"}],
                    "tracks": {"step": "tracks"},
                },
                {
                    "id": "labeled",
                    "type": "feature",
                    "feature": "extract-labeled-templates",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "n_total": 300,
                        "test_fraction": 0.0,
                    },
                },
                {
                    "id": "classifier",
                    "type": "feature",
                    "feature": "lightning-action",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        "templates": {"step": "labeled"},
                        "labels": {"step": "labels"},
                        "default_class": 3,
                        "device": "cpu",
                        "num_epochs": 2,
                        "sequence_length": _WINDOW,
                        "batch_size": 4,
                        "num_hid_units": 8,
                    },
                },
            ],
        }
    )
    return ds, recipe


def test_the_scoped_chain_runs_with_the_real_package(
    in_process_loaders: None, tmp_path: Path
) -> None:
    ds, recipe = _dataset_and_recipe(tmp_path / "ds")

    run = run_pipeline(
        ds,
        recipe,
        step_scopes={"labeled": TRAIN_GROUP, "classifier": BOTH_GROUPS},
    )

    assert {outcome.state for outcome in run.outcomes} == {"ran"}
    [run_id] = [str(o.run_id) for o in run.outcomes if o.step_id == "classifier"]
    paths = list(
        ds.get_root("features").glob(f"lightning-action__*/{run_id}/*.parquet")
    )
    assert len(paths) == 8
    for path in paths:
        table = pd.read_parquet(path)
        assert not table.filter(like="prob_").isna().any(axis=None)
        assert set(table["perspective"]) == {0, 1}


def test_widening_the_scope_applies_the_trained_model_and_trains_nothing_again(
    in_process_loaders: None, tmp_path: Path
) -> None:
    ds, recipe = _dataset_and_recipe(tmp_path / "ds")
    narrow = run_pipeline(
        ds, recipe, step_scopes={"labeled": TRAIN_GROUP, "classifier": TRAIN_GROUP}
    )
    [run_id] = [str(o.run_id) for o in narrow.outcomes if o.step_id == "classifier"]
    [root] = ds.get_root("features").glob(f"lightning-action__*/{run_id}")
    model = (root / "lightning_action_model.joblib").read_bytes()
    before = {path.name: pd.read_parquet(path) for path in root.glob("*.parquet")}
    assert len(before) == 6

    wide = run_pipeline(
        ds, recipe, step_scopes={"labeled": TRAIN_GROUP, "classifier": BOTH_GROUPS}
    )

    [again] = [str(o.run_id) for o in wide.outcomes if o.step_id == "classifier"]
    assert again == run_id
    assert (root / "lightning_action_model.joblib").read_bytes() == model
    after = {path.name: pd.read_parquet(path) for path in root.glob("*.parquet")}
    assert len(after) == 8
    for name, table in before.items():
        pd.testing.assert_frame_equal(after[name], table)
