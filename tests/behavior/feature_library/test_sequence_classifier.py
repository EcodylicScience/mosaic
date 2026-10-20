"""``sequence-classifier``: fitting heads on runs of frames, and applying them to series.

These tests train real networks on the CPU, a few parameters wide and a few dozen
steps long, so that they run in seconds. CI's ``identity`` job runs them, because it
installs torch. The registry, the parameters and the batches are in
``test_sequence_heads.py`` and run without torch.

The loop is checked against a copy of the agent's script that reached the numbers
this step reproduces: the same networks, batches and optimizer, written with
nothing of mosaic's, and a mosaic fit must give the same network.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from mosaic.behavior.feature_library import sequence_classifier as module
from mosaic.behavior.feature_library.sequence_classifier import (
    SequenceClassifier,
    SequenceClassifierBundle,
    SequenceClassifierModelArtifact,
)
from mosaic.behavior.feature_library.sequence_heads import (
    HEADS,
    GruConfig,
    HeadConfig,
    HeadSpec,
    TcnConfig,
    assemble_network,
    register_head,
)
from mosaic.behavior.feature_library.sequence_training import (
    Sampler,
    advance,
    backpropagate,
    class_weights,
    deterministic_torch,
    seed_torch,
    train_network,
)
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.progress import NullProgressCallback
from mosaic.core.pipeline.types import InputStream, Result
from tests.helpers import Event, StudyEntry, TemplatesStudy, write_templates_study

pytestmark = pytest.mark.identity

Entry = StudyEntry
CLASSES = [0, 1, 2]
FEATURES = ["f0", "f1", "f2", "f3"]
FRAMES = 60
A: Entry = ("train", "a")
B: Entry = ("train", "b")
C: Entry = ("test", "c")

HEADS_SMALL: dict[str, dict[str, object]] = {
    "tcn": {"hidden": 8, "blocks": 2},
    "gru": {"hidden": 8, "layers": 1},
    "mlp": {"hidden": 8, "layers": 1},
}


# --- Tables, labels and fits ------------------------------------------------------


def _label(frame: int, offset: int) -> int:
    return CLASSES[(frame // 6 + offset) % 3]


def _table(
    entry: Entry,
    frames: Sequence[int],
    *,
    offset: int = 0,
    pair: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """One entry's rows: three columns that mark the label of the frame, and noise.

    A pair table holds two series, and only perspective 0 is labeled.
    """
    rng = np.random.default_rng(seed + zlib.crc32("/".join(entry).encode()))
    blocks: list[pd.DataFrame] = []
    for perspective in (0, 1) if pair else (0,):
        labels = np.array([_label(f, offset) for f in frames])
        block = pd.DataFrame(
            {
                "frame": np.asarray(frames, dtype=np.int64),
                "group": entry[0],
                "sequence": entry[1],
            }
        )
        if pair:
            block["id1"] = perspective
            block["id2"] = 1 - perspective
            block["perspective"] = perspective
        else:
            block["id"] = 0
        for index in range(3):
            block[f"f{index}"] = (labels == CLASSES[index]) + rng.normal(
                0, 0.1, len(frames)
            )
        block["f3"] = rng.normal(0, 1, len(frames))
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _events(frames: Sequence[int], offset: int, *, pair: bool = True) -> list[Event]:
    return [(f, 0, 1 if pair else -1, _label(f, offset)) for f in frames]


def _study(
    root: Path,
    offsets: dict[Entry, int],
    *,
    train: Sequence[Entry],
    test: Sequence[Entry] = (),
    frames: Mapping[Entry, Sequence[int]] | None = None,
) -> tuple[TemplatesStudy, dict[Entry, pd.DataFrame]]:
    frames = frames or {}
    tables = {
        entry: _table(entry, frames.get(entry, range(FRAMES)), offset=offset)
        for entry, offset in offsets.items()
    }
    events = {
        entry: _events(frames.get(entry, range(FRAMES)), offset)
        for entry, offset in offsets.items()
    }
    return write_templates_study(root, events, train=train, test=test), tables


def _params(**overrides: object) -> dict[str, object]:
    head = str(overrides.get("head", "gru"))
    return {
        "templates": {"feature": "extract-labeled-templates"},
        "labels": {"kind": "behavior"},
        "default_class": 99,
        "device": "cpu",
        "head": head,
        "head_config": HEADS_SMALL.get(head, {}),
        "steps": 80,
        "batch_size": 4,
        "crop_length": 30,
        "min_run": 8,
        "learning_rate": 0.02,
        "random_state": 3,
        **overrides,
    }


def _feature(**overrides: object) -> SequenceClassifier:
    return SequenceClassifier(
        SequenceClassifier.Inputs((Result(feature="temporal-stack"),)),
        params=_params(**overrides),
    )


def _stream(tables: dict[Entry, pd.DataFrame]) -> InputStream:
    items = [(f"{g}__{s}", table) for (g, s), table in sorted(tables.items())]
    return InputStream(lambda: iter(items), n_entries=len(items))


def _fit(
    study: TemplatesStudy,
    tables: dict[Entry, pd.DataFrame],
    run: str = "run",
    **overrides: object,
) -> SequenceClassifier:
    feature = _feature(**overrides)
    assert not feature.load_state(
        study.root / run, {"templates": study.templates}, {"labels": study.label_files}
    )
    feature.fit(_stream(tables))
    return feature


def _saved(feature: SequenceClassifier, root: Path) -> SequenceClassifierBundle:
    feature.save_state(root)
    return SequenceClassifierModelArtifact().from_path(
        root / "sequence_classifier_model.joblib"
    )


def _record_training_runs(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record the lengths of the runs each network trains on, and train as usual."""
    recorded: list[list[int]] = []
    original = module.train_network

    def spy(
        network: nn.Module,
        frames: npt.NDArray[np.float32],
        classes: npt.NDArray[np.intp],
        sampler: Sampler,
        weights: npt.NDArray[np.float64],
        *,
        steps: int,
        learning_rate: float,
        weight_decay: float,
        pct_start: float,
        grad_clip: float,
        seed: int,
        device: str,
        check: Callable[[], None],
    ) -> None:
        recorded.append(sorted(sampler.lengths.tolist()))
        original(
            network,
            frames,
            classes,
            sampler,
            weights,
            steps=steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pct_start=pct_start,
            grad_clip=grad_clip,
            seed=seed,
            device=device,
            check=check,
        )

    monkeypatch.setattr(module, "train_network", spy)
    return recorded


def _two_entries(tmp_path: Path) -> tuple[TemplatesStudy, dict[Entry, pd.DataFrame]]:
    return _study(tmp_path, {A: 0, B: 1, C: 2}, train=[A, B], test=[C])


# --- The loop against the agent's script ------------------------------------------


class _ReferenceTcn(nn.Module):
    """Non-causal dilated residual convolutions, as ``run4_seq.py`` defines them."""

    def __init__(
        self, n_in: int, hidden: int, layers: int, dropout: float, in_dropout: float
    ):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(in_dropout), nn.Conv1d(n_in, hidden, 1))
        self.blocks = nn.ModuleList()
        for i in range(layers):
            d = 2**i
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, 1),
                )
            )
        self.out = nn.Conv1d(hidden, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))
        for block in self.blocks:
            h = h + block(h)
        return self.out(functional.relu(h)).transpose(1, 2)


class _ReferenceGru(nn.Module):
    """The bidirectional GRU of ``run4_seq.py``."""

    def __init__(
        self, n_in: int, hidden: int, layers: int, dropout: float, in_dropout: float
    ):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Dropout(in_dropout), nn.Linear(n_in, hidden), nn.ReLU()
        )
        self.rnn = nn.GRU(
            hidden,
            hidden,
            num_layers=min(layers, 2),
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.inp(x))
        return self.out(h)


def _reference_train(
    model: nn.Module,
    frames: npt.NDArray[np.float32],
    labels: npt.NDArray[np.intp],
    bounds: list[tuple[int, int]],
    *,
    seed: int,
    steps: int,
    lr: float,
    wd: float,
    crop: int,
    batch: int,
    weight: npt.NDArray[np.float64],
) -> nn.Module:
    """The body of ``train_model`` in the agent's script, with its call order."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=steps, pct_start=0.1
    )
    w = torch.as_tensor(weight.astype(np.float32))
    x_all = torch.as_tensor(frames)
    y_all = torch.as_tensor(labels.astype(np.int64))
    lens = np.array([b - a for a, b in bounds])
    p = lens / lens.sum()
    model.train()
    for _ in range(steps):
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        chosen: npt.NDArray[np.int64] = rng.choice(len(bounds), batch, p=p)
        for i in chosen:
            a, b = bounds[int(i)]
            length = min(crop, b - a)
            start = a + int(rng.integers(0, b - a - length + 1))
            xs.append(x_all[start : start + length])
            ys.append(y_all[start : start + length])
        shortest = min(x.shape[0] for x in xs)
        xb = torch.stack([x[:shortest] for x in xs])
        yb = torch.stack([v[:shortest] for v in ys])
        loss = functional.cross_entropy(
            model(xb).reshape(-1, 3), yb.reshape(-1), weight=w
        )
        optimizer.zero_grad(set_to_none=True)
        backpropagate(loss)
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        advance(optimizer)
        schedule.step()
    return model


@pytest.mark.parametrize("head", ["tcn", "gru"])
def test_a_fit_gives_the_network_the_agents_script_gives(head: str) -> None:
    rng = np.random.default_rng(11)
    lengths = np.array([70, 120, 45], dtype=np.intp)
    frames = rng.normal(size=(int(lengths.sum()), 4)).astype(np.float32)
    labels = rng.integers(0, 3, size=int(lengths.sum())).astype(np.intp)
    ends = np.cumsum(lengths)
    bounds = [(int(e - n), int(e)) for e, n in zip(ends, lengths)]
    steps, crop, batch, seed = 25, 32, 4, 9
    weights = class_weights(np.bincount(labels, minlength=3).astype(np.intp), "sqrt")
    options = HEADS_SMALL[head]

    with deterministic_torch(device="cpu", n_threads=1):
        seed_torch(seed, device="cpu")
        if head == "tcn":
            reference: nn.Module = _ReferenceTcn(4, 8, 2, 0.3, 0.1)
        else:
            reference = _ReferenceGru(4, 8, 2, 0.3, 0.1)
        reference = _reference_train(
            reference,
            frames,
            labels,
            bounds,
            seed=seed,
            steps=steps,
            lr=0.02,
            wd=1e-3,
            crop=crop,
            batch=batch,
            weight=weights,
        )
        seed_torch(seed, device="cpu")
        config = (
            TcnConfig.model_validate(options)
            if head == "tcn"
            else GruConfig(hidden=8, layers=2)
        )
        network = assemble_network(
            head, config, input_dropout=0.1, n_features=4, n_classes=3
        )
        train_network(
            network,
            frames,
            labels,
            Sampler.of(lengths, crop, batch),
            weights,
            steps=steps,
            learning_rate=0.02,
            weight_decay=1e-3,
            pct_start=0.1,
            grad_clip=1.0,
            seed=seed,
            device="cpu",
            check=lambda: None,
        )
        probe = torch.as_tensor(frames[:60])[None]
        reference.eval()
        network.eval()
        with torch.no_grad():
            expected = reference(probe)
            actual = network(probe)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# --- Determinism ---------------------------------------------------------------


def test_two_fits_of_one_seed_are_equal_and_two_seeds_differ(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)

    first = _saved(_fit(study, tables, "one"), tmp_path / "one")
    again = _saved(_fit(study, tables, "two"), tmp_path / "two")
    other = _saved(_fit(study, tables, "three", random_state=4), tmp_path / "three")

    for name, weights in first["models"][-1]["weights"].items():
        np.testing.assert_array_equal(weights, again["models"][-1]["weights"][name])
    assert any(
        not np.array_equal(w, other["models"][-1]["weights"][name])
        for name, w in first["models"][-1]["weights"].items()
        if w.dtype.kind == "f"
    )


def test_a_fit_leaves_the_torch_settings_as_it_found_them() -> None:
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.get_num_threads(),
    )

    with deterministic_torch(device="cpu", n_threads=2):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == 2

    assert before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.get_num_threads(),
    )


# --- Applying ------------------------------------------------------------------


def test_apply_returns_a_row_for_every_row_in_order_with_the_identity(
    tmp_path: Path,
) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _fit(study, tables)
    df = tables[C]

    result = feature.apply(df)

    assert len(result) == len(df)
    assert result["frame"].tolist() == df["frame"].tolist()
    assert result["perspective"].tolist() == df["perspective"].tolist()
    probs = result[["prob_0", "prob_1", "prob_2"]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert result["predicted_label"].tolist() == list(
        np.array(CLASSES)[probs.argmax(axis=1)]
    )


def test_a_fit_learns_the_signal_it_was_given(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _fit(study, tables, steps=200)
    df = tables[C]

    result = feature.apply(df)

    labeled = df["perspective"] == 0
    truth = [_label(int(f), 2) for f in df.loc[labeled, "frame"].tolist()]
    predicted: list[int] = result.loc[labeled, "predicted_label"].tolist()
    hits = sum(1 for guess, label in zip(predicted, truth) if guess == label)
    assert hits / len(truth) > 0.8


def test_a_gap_in_the_frames_splits_a_series_so_each_side_is_predicted_alone(
    tmp_path: Path,
) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _fit(study, tables)
    frames = [*range(0, 30), *range(40, 70)]
    whole = _table(C, frames, offset=2)
    second = whole[whole["frame"] >= 40].reset_index(drop=True)

    together = feature.apply(whole)
    alone = feature.apply(second)

    after = together[together["frame"] >= 40].reset_index(drop=True)
    np.testing.assert_allclose(
        after[["prob_0", "prob_1", "prob_2"]].to_numpy(),
        alone[["prob_0", "prob_1", "prob_2"]].to_numpy(),
        atol=1e-6,
    )


def test_the_convolutions_of_a_tcn_read_only_their_receptive_field(
    tmp_path: Path,
) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _fit(study, tables, head="tcn")
    base = _table(C, range(FRAMES), offset=2, pair=False)
    far, near = base.copy(), base.copy()
    far.loc[far["frame"] == 50, FEATURES] += 5.0
    near.loc[near["frame"] == 33, FEATURES] += 5.0
    target = base["frame"] == 30

    probabilities = [
        feature.apply(table).loc[target, ["prob_0", "prob_1", "prob_2"]].to_numpy()
        for table in (base, far, near)
    ]

    # Two blocks of kernel 3 read 1 + 2 * 3 = 7 frames: three on each side.
    np.testing.assert_allclose(probabilities[1], probabilities[0], atol=1e-6)
    assert not np.allclose(probabilities[2], probabilities[0], atol=1e-6)


def test_an_mlp_scores_each_frame_by_itself(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _fit(study, tables, head="mlp")
    base = _table(C, range(FRAMES), offset=2, pair=False)
    changed = base.copy()
    changed.loc[changed["frame"] == 31, FEATURES] += 5.0

    before = feature.apply(base)
    after = feature.apply(changed)

    others = base["frame"] != 31
    np.testing.assert_allclose(
        after.loc[others, ["prob_0", "prob_1", "prob_2"]].to_numpy(),
        before.loc[others, ["prob_0", "prob_1", "prob_2"]].to_numpy(),
        atol=1e-6,
    )


def test_a_row_of_no_series_takes_the_default_class(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _fit(study, tables, default_class=2)
    df = tables[C].copy()
    df["id1"] = df["id1"].astype("float64")
    df.loc[df.index[3], "id1"] = np.nan

    result = feature.apply(df)

    assert np.isnan(result.loc[3, ["prob_0", "prob_1", "prob_2"]].to_numpy()).all()
    assert result.loc[3, "predicted_label"] == 2


def test_a_frame_that_a_series_holds_twice_is_refused(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _fit(study, tables)
    df = pd.concat([tables[C], tables[C].iloc[:1]], ignore_index=True)

    with pytest.raises(ValueError, match="appears in more than one row"):
        feature.apply(df)


# --- Saving and loading -----------------------------------------------------------


def test_a_fitted_model_applies_the_same_after_it_is_saved_and_loaded(
    tmp_path: Path,
) -> None:
    study, tables = _two_entries(tmp_path)
    fitted = _fit(study, tables)
    before = fitted.apply(tables[C])
    bundle = _saved(fitted, tmp_path / "saved")

    reloaded = SequenceClassifier(
        SequenceClassifier.Inputs((Result(feature="temporal-stack"),)),
        params={"model": {"feature": "sequence-classifier"}, "default_class": 99},
    )
    assert reloaded.load_state(
        tmp_path / "elsewhere",
        {"model": tmp_path / "saved" / "sequence_classifier_model.joblib"},
        {},
    )

    pd.testing.assert_frame_equal(reloaded.apply(tables[C]), before)
    assert bundle["environment"]["torch"] == torch.__version__
    assert bundle["environment"]["device"] == "cpu"
    assert bundle["classes"] == CLASSES
    assert bundle["columns"] == FEATURES


def test_a_cached_run_loads_without_a_fit(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)
    fitted = _fit(study, tables)
    fitted.save_state(tmp_path / "run")
    again = _feature()

    assert again.load_state(
        tmp_path / "run", {"templates": study.templates}, {"labels": study.label_files}
    )
    pd.testing.assert_frame_equal(again.apply(tables[C]), fitted.apply(tables[C]))


def test_a_bundle_that_fails_to_write_leaves_no_cached_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, tables = _two_entries(tmp_path)
    fitted = _fit(study, tables)

    def fail(value: object, filename: object) -> None:
        Path(str(filename)).write_bytes(b"partial")
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr(joblib, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        fitted.save_state(tmp_path / "run")

    assert not (tmp_path / "run" / "sequence_classifier_model.joblib").exists()


# --- Fitting -------------------------------------------------------------------------


def test_only_the_train_entries_the_templates_run_names_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, tables = _two_entries(tmp_path)
    seen = _record_training_runs(monkeypatch)

    _ = _fit(study, tables)

    # A and B are 60 frames each; C is a test entry and trains nothing.
    assert seen == [[FRAMES, FRAMES]]


def test_a_run_ends_at_a_frame_gap_and_a_short_run_does_not_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames: dict[Entry, Sequence[int]] = {
        A: [*range(0, 30), *range(40, 47), *range(60, 100)]
    }
    study, tables = _study(tmp_path, {A: 0, B: 1}, train=[A, B], frames=frames)
    seen = _record_training_runs(monkeypatch)

    _ = _fit(study, tables)

    # A holds runs of 30, 7 (under min_run 8) and 40 frames; B one run of 60.
    assert seen == [[30, 40, 60]]


def test_a_fit_over_one_class_is_refused(tmp_path: Path) -> None:
    events = {
        A: [(f, 0, 1, 0) for f in range(FRAMES)],
        B: [(f, 0, 1, 0) for f in range(FRAMES)],
    }
    study = write_templates_study(tmp_path, events, train=[A, B])
    tables = {A: _table(A, range(FRAMES)), B: _table(B, range(FRAMES))}
    feature = _feature()
    assert not feature.load_state(
        tmp_path / "run", {"templates": study.templates}, {"labels": study.label_files}
    )

    with pytest.raises(RuntimeError, match="a classifier needs two or more"):
        feature.fit(_stream(tables))


def test_a_train_entry_outside_the_steps_scope_is_refused(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)
    feature = _feature()
    assert not feature.load_state(
        tmp_path / "run", {"templates": study.templates}, {"labels": study.label_files}
    )
    narrower = {entry: table for entry, table in tables.items() if entry != B}

    with pytest.raises(
        ValueError, match=r"1 train entries .* not in this step's scope"
    ):
        feature.fit(_stream(narrower))


def test_the_columns_a_network_reads_are_the_ones_named(tmp_path: Path) -> None:
    study, tables = _two_entries(tmp_path)

    bundle = _saved(_fit(study, tables, columns=["f2", "f0"]), tmp_path / "saved")

    assert bundle["columns"] == ["f2", "f0"]
    assert len(bundle["models"][0]["mean"]) == 2


def test_a_missing_value_standardizes_to_the_mean_and_the_fit_finishes(
    tmp_path: Path,
) -> None:
    study, tables = _two_entries(tmp_path)
    tables[A].loc[tables[A].index[4], "f1"] = np.nan

    feature = _fit(study, tables)

    assert np.isfinite(
        feature.apply(tables[C])[["prob_0", "prob_1", "prob_2"]].to_numpy()
    ).all()


# --- Cross-validation ------------------------------------------------------------------


def test_every_fold_model_trains_without_its_fold_and_the_full_model_on_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [("train", f"s{i}") for i in range(4)]
    frames = {entry: range(40 + 10 * index) for index, entry in enumerate(entries)}
    study, tables = _study(
        tmp_path,
        {entry: i for i, entry in enumerate(entries)},
        train=entries,
        frames=frames,
    )
    runs = _record_training_runs(monkeypatch)

    feature = _fit(study, tables, cv_folds=2, cv_seed=1, steps=20)

    bundle = _saved(feature, tmp_path / "saved")
    folds = pd.read_parquet(tmp_path / "saved" / "cv_folds.parquet")
    lengths = {entry: 40 + 10 * index for index, entry in enumerate(entries)}
    fold_of = {
        (g, s): int(f)
        for g, s, f in zip(folds["group"], folds["sequence"], folds["fold"])
    }
    assert sorted(fold_of) == sorted(entries)
    assert sorted(fold_of.values()) == [0, 0, 1, 1]
    for fold in range(2):
        outside = sorted(n for entry, n in lengths.items() if fold_of[entry] != fold)
        assert runs[fold] == outside
    assert runs[2] == sorted(lengths.values())
    assert [m["fold"] for m in bundle["models"]] == [0, 1, None]
    assert folds["n_rows"].tolist() == [
        lengths[(g, s)] for g, s in zip(folds["group"], folds["sequence"])
    ]


def _scored_by(
    bundle: SequenceClassifierBundle, model: int, df: pd.DataFrame, directory: Path
) -> npt.NDArray[np.float64]:
    """The probabilities that model *model* of *bundle* gives *df*, as a model of its own."""
    alone = {key: value for key, value in bundle.items() if key not in ("cv", "models")}
    alone["models"] = [bundle["models"][model]]
    directory.mkdir(parents=True)
    joblib.dump(alone, directory / "sequence_classifier_model.joblib")
    probe = SequenceClassifier(
        SequenceClassifier.Inputs((Result(feature="temporal-stack"),)),
        params={"model": {"feature": "sequence-classifier"}, "default_class": 99},
    )
    assert probe.load_state(
        directory / "run", {"model": directory / "sequence_classifier_model.joblib"}, {}
    )
    return probe.apply(df)[["prob_0", "prob_1", "prob_2"]].to_numpy()


def test_a_training_entry_is_scored_by_the_model_that_did_not_see_it(
    tmp_path: Path,
) -> None:
    entries = [("train", f"s{i}") for i in range(4)]
    study, tables = _study(
        tmp_path,
        {entry: i for i, entry in enumerate(entries)} | {C: 2},
        train=entries,
        test=[C],
    )
    feature = _fit(study, tables, cv_folds=2, steps=20)
    bundle = _saved(feature, tmp_path / "saved")
    cv = bundle.get("cv")
    assert cv is not None
    fold_of = {(g, s): f for g, s, f in cv["fold_of"]}
    full = len(bundle["models"]) - 1
    target = entries[0]

    scored = feature.apply(tables[target])[["prob_0", "prob_1", "prob_2"]].to_numpy()

    held_out = _scored_by(bundle, fold_of[target], tables[target], tmp_path / "fold")
    in_sample = _scored_by(bundle, full, tables[target], tmp_path / "full")
    np.testing.assert_allclose(scored, held_out, atol=1e-6)
    assert not np.allclose(scored, in_sample, atol=1e-6)
    # A test entry is scored by the model fitted on every training run.
    test_scored = feature.apply(tables[C])[["prob_0", "prob_1", "prob_2"]].to_numpy()
    np.testing.assert_allclose(
        test_scored,
        _scored_by(bundle, full, tables[C], tmp_path / "full-test"),
        atol=1e-6,
    )


def test_a_run_without_folds_fits_the_model_of_the_run_with_them(
    tmp_path: Path,
) -> None:
    entries = [("train", f"s{i}") for i in range(4)]
    study, tables = _study(
        tmp_path, {e: i for i, e in enumerate(entries)}, train=entries
    )

    plain = _saved(_fit(study, tables, "plain", steps=20), tmp_path / "plain")
    folded = _saved(
        _fit(study, tables, "folded", cv_folds=2, steps=20), tmp_path / "folded"
    )

    for name, weights in plain["models"][-1]["weights"].items():
        np.testing.assert_array_equal(weights, folded["models"][-1]["weights"][name])


def test_a_cached_run_with_folds_applies_the_same(tmp_path: Path) -> None:
    entries = [("train", f"s{i}") for i in range(4)]
    study, tables = _study(
        tmp_path, {e: i for i, e in enumerate(entries)}, train=entries
    )
    fitted = _fit(study, tables, cv_folds=2, steps=20)
    fitted.save_state(tmp_path / "run")
    again = _feature(cv_folds=2, steps=20)

    assert again.load_state(
        tmp_path / "run", {"templates": study.templates}, {"labels": study.label_files}
    )
    pd.testing.assert_frame_equal(
        again.apply(tables[entries[1]]), fitted.apply(tables[entries[1]])
    )


# --- Progress and cancel ---------------------------------------------------------------


class _Recorder(NullProgressCallback):
    def __init__(self) -> None:
        self.phases: list[str] = []

    def on_phase(self, phase: str, message: str) -> None:
        self.phases.append(message)


def test_a_fit_names_each_model_and_a_cancel_stops_before_the_next(
    tmp_path: Path,
) -> None:
    entries = [("train", f"s{i}") for i in range(4)]
    study, tables = _study(
        tmp_path, {e: i for i, e in enumerate(entries)}, train=entries
    )
    feature = _feature(cv_folds=2, steps=20)
    recorder = _Recorder()
    feature.progress_callback = recorder
    assert not feature.load_state(
        tmp_path / "run", {"templates": study.templates}, {"labels": study.label_files}
    )

    feature.fit(_stream(tables))

    assert recorder.phases == ["fold 1 of 2", "fold 2 of 2", "all runs"]

    token = CancelToken()
    stopped = _feature(cv_folds=2, steps=20)
    stopped.cancel_token = token
    token.cancel()
    assert not stopped.load_state(
        tmp_path / "run2", {"templates": study.templates}, {"labels": study.label_files}
    )
    with pytest.raises(Cancelled):
        stopped.fit(_stream(tables))
    stopped.save_state(tmp_path / "run2")
    assert not (tmp_path / "run2" / "sequence_classifier_model.joblib").exists()


# --- A head of your own -----------------------------------------------------------------


class _LinearConfig(HeadConfig):
    """The options of the head that these tests register."""


def _linear_build(config: _LinearConfig, n_features: int, n_classes: int) -> nn.Module:
    return nn.Linear(n_features, n_classes)


def _linear_with_bias_build(
    config: _LinearConfig, n_features: int, n_classes: int
) -> nn.Module:
    return nn.Linear(n_features, n_classes, bias=False)


def test_a_head_registered_in_python_fits_and_applies(
    tmp_path: Path, head_registry: None
) -> None:
    _ = register_head(
        HeadSpec(
            name="linear",
            config=_LinearConfig,
            build=_linear_build,
            description="One linear layer per frame.",
        )
    )
    study, tables = _two_entries(tmp_path)

    feature = _fit(study, tables, head="linear", head_config={})

    result = feature.apply(tables[C])
    assert len(result) == len(tables[C])
    assert _saved(feature, tmp_path / "saved")["head"] == "linear"


def test_a_model_warns_when_the_code_of_its_head_changed(
    tmp_path: Path,
    head_registry: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _ = register_head(
        HeadSpec(
            name="linear",
            config=_LinearConfig,
            build=_linear_build,
            description="One layer.",
        )
    )
    study, tables = _two_entries(tmp_path)
    _ = _saved(_fit(study, tables, head="linear", head_config={}), tmp_path / "run")
    monkeypatch.setitem(
        HEADS,
        "linear",
        HeadSpec(
            name="linear",
            config=_LinearConfig,
            build=_linear_with_bias_build,
            description="One layer.",
        ),
    )
    caplog.set_level(logging.WARNING)

    again = _feature(head="linear", head_config={})
    assert again.load_state(
        tmp_path / "run", {"templates": study.templates}, {"labels": study.label_files}
    )

    assert (
        "the code of head 'linear' changed since this model was fitted" in caplog.text
    )
