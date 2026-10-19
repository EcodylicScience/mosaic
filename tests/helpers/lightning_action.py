"""A stand-in for the lightning-action package, with the package's data contract.

The extra does not resolve on a machine without CUDA -- `lightning-action` requires
`nvidia-dali-cuda110` with no environment marker -- so a test that is not about the
network stubs the package rather than skips, and runs in every job.

The stub keeps what the package does with the files a feature hands it, because a stub
that wrote or read CSVs its own way once hid a fit that raised on every real call:

- It reads each CSV with `index_col=0`, and a training feature file needs a label file
  of the same length, with one column per class.
- A model cuts whole windows from a file's first row, so a file shorter than a window
  and the context on each side of it cannot be predicted, and the rows past the last
  whole window come back NaN.
- The predictions have the row index first and one column per class name.

Its probabilities are a fixed function of each row's first feature, so a test can tell
which row a prediction was made for. `LightningActionRecorder` keeps what the stub was
given: the config, the files it trained on, how many times it trained and the files it
predicted.
"""

from __future__ import annotations

import copy
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tests.helpers.labels import Event, write_sparse_labels

if TYPE_CHECKING:
    from mosaic.behavior.feature_library.lightning_action_feature import (
        LightningActionFeature,
    )

LA_CLASSES = [1, 2, 3]
"""The classes a stub model predicts until a fit has shown it others."""

LA_CLASS_NAMES = [str(c) for c in LA_CLASSES]

LA_WINDOW = 10
"""The window of the stub, and so the `sequence_length` that a test of it passes."""

LA_PAD = 2
"""The context of a dilated head of one layer and one lag, on each side of a frame."""


def stub_probabilities(
    value: npt.NDArray[np.float64], n_classes: int
) -> npt.NDArray[np.float64]:
    """The class probabilities the stub gives a row whose first feature is *value*."""
    centered = np.arange(n_classes) - (n_classes - 1) / 2
    logits = value[:, None] * centered[None, :]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


@runtime_checkable
class _Subscriptable(Protocol):
    def __getitem__(self, key: str, /) -> object: ...


@dataclass
class LightningActionRecorder:
    """What the stub package saw of a feature's calls."""

    config: dict[str, object] = field(default_factory=dict[str, object])
    trained: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(
        default_factory=dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    )
    predicted: list[dict[str, pd.DataFrame]] = field(
        default_factory=list[dict[str, pd.DataFrame]]
    )
    class_names: list[str] = field(default_factory=lambda: list(LA_CLASS_NAMES))
    trainings: int = 0
    predict_workers: list[object] = field(default_factory=list[object])

    def config_value(self, *keys: str) -> object:
        """Read *keys* down the config of the last `from_config`, which nests by name."""
        value: object = self.config
        for key in keys:
            assert isinstance(value, _Subscriptable)
            value = value[key]
        return value


class _StubModel:
    """Stands in for `lightning_action.api.model.Model`."""

    recorder: ClassVar[LightningActionRecorder]

    def __init__(self) -> None:
        self.config: dict[str, dict[str, object]] = {"training": {}}

    @classmethod
    def from_config(cls, config: dict[str, object], /) -> _StubModel:
        cls.recorder.config = copy.deepcopy(config)
        return cls()

    @classmethod
    def from_dir(cls, path: str, /) -> _StubModel:
        return cls()

    def train(self, output_dir: str, post_inference: bool) -> None:
        recorder = self.recorder
        recorder.trainings += 1
        data = Path(str(recorder.config_value("data", "data_path")))
        features = {p.stem: p for p in sorted((data / "features").glob("*.csv"))}
        labels = {p.stem: p for p in sorted((data / "labels").glob("*.csv"))}
        for stem, path in features.items():
            # The package raises on a feature file that has no label file, and on
            # files of unequal length.
            frame = pd.read_csv(path, index_col=0)
            one_hot = pd.read_csv(labels[stem], index_col=0)
            assert len(frame) == len(one_hot), stem
            recorder.class_names = list(one_hot.columns)
            recorder.trained[stem] = (frame, one_hot)
        # With no file of a whole window the package has nothing to train on.
        assert any(len(frame) >= LA_WINDOW for frame, _ in recorder.trained.values())
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def predict(self, *, data_path: str, input_dir: str, output_dir: str) -> None:
        recorder = self.recorder
        recorder.predict_workers.append(self.config["training"].get("num_workers"))
        calls: dict[str, pd.DataFrame] = {}
        for path in sorted((Path(data_path) / input_dir).glob("*.csv")):
            features = pd.read_csv(path, index_col=0)
            n = len(features)
            if n < LA_WINDOW + LA_PAD:
                msg = f"could not cut a window from {n} rows"
                raise ValueError(msg)
            calls[path.stem] = features
            probs = stub_probabilities(
                features.iloc[:, 0].to_numpy(dtype=np.float64),
                len(recorder.class_names),
            )
            probs[n - n % LA_WINDOW :] = np.nan
            pd.DataFrame(probs, columns=recorder.class_names).to_csv(
                Path(output_dir) / f"{path.stem}_predictions.csv"
            )
        recorder.predicted.append(calls)


def install_lightning_action_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> LightningActionRecorder:
    """Satisfy the package imports with the stub, and return what it records."""
    recorder = LightningActionRecorder()
    monkeypatch.setattr(_StubModel, "recorder", recorder, raising=False)

    def compute_sequence_pad(
        model_type: str, default: int | None = None, **params: int
    ) -> int:
        return sum(2 * 2**n * params["num_lags"] for n in range(params["num_layers"]))

    modules = {
        "lightning_action": types.ModuleType("lightning_action"),
        "lightning_action.api": types.ModuleType("lightning_action.api"),
        "lightning_action.api.model": types.ModuleType("lightning_action.api.model"),
        "lightning_action.data": types.ModuleType("lightning_action.data"),
        "lightning_action.data.utils": types.ModuleType("lightning_action.data.utils"),
    }
    monkeypatch.setattr(
        modules["lightning_action.api.model"], "Model", _StubModel, raising=False
    )
    monkeypatch.setattr(
        modules["lightning_action.data.utils"],
        "compute_sequence_pad",
        compute_sequence_pad,
        raising=False,
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return recorder


StudyEntry = tuple[str, str]
"""The ``(group, sequence)`` of an entry."""


@dataclass(frozen=True)
class TemplatesStudy:
    """The files a fit reads that `extract-labeled-templates` and the labels leave."""

    root: Path
    label_files: dict[StudyEntry, Path]

    @property
    def templates(self) -> Path:
        """Where the templates run's ``templates.parquet`` would be."""
        return self.root / "templates" / "templates.parquet"


def write_templates_study(
    root: Path,
    events: Mapping[StudyEntry, Sequence[Event]],
    *,
    train: Sequence[StudyEntry],
    test: Sequence[StudyEntry] = (),
) -> TemplatesStudy:
    """Write the label files of *events* and the provenance of a templates run.

    An entry with no events has no label file. The provenance lists *train* and
    *test* entries, and an entry in neither is one the templates run never saw.
    """
    label_files: dict[StudyEntry, Path] = {}
    for entry, entry_events in events.items():
        if entry_events:
            label_files[entry] = write_sparse_labels(
                root / "labels" / f"{entry[0]}__{entry[1]}.npz", list(entry_events)
            )
    listed = [*train, *test]
    provenance = pd.DataFrame(
        {
            "group": [g for g, _ in listed],
            "sequence": [s for _, s in listed],
            "split": ["train"] * len(train) + ["test"] * len(test),
            "count": 1,
            "proportion": 1.0 / len(listed) if listed else 0.0,
        }
    )
    (root / "templates").mkdir(parents=True)
    provenance.to_parquet(root / "templates" / "template_provenance.parquet")
    return TemplatesStudy(root, label_files)


def lightning_action_params(**overrides: object) -> dict[str, object]:
    """The params of a `lightning-action` fit that a stub trains in a moment."""
    return {
        "templates": {"feature": "extract-labeled-templates"},
        "labels": {"kind": "behavior"},
        "default_class": 99,
        "device": "cpu",
        "num_epochs": 1,
        "sequence_length": LA_WINDOW,
        "num_layers": 1,
        "num_lags": 1,
        **overrides,
    }


def fit_lightning_action(
    study: TemplatesStudy,
    tables: Mapping[StudyEntry, pd.DataFrame],
    **overrides: object,
) -> LightningActionFeature:
    """Fit a `lightning-action` feature on the stream of *tables*.

    The entries stream in the order of the pipeline's manifest, sorted by
    ``(group, sequence)``.
    """
    from mosaic.behavior.feature_library.lightning_action_feature import (
        LightningActionFeature,
    )
    from mosaic.core.pipeline.types import InputStream, Result

    feature = LightningActionFeature(
        LightningActionFeature.Inputs((Result(feature="temporal-stack"),)),
        params=lightning_action_params(**overrides),
    )
    assert not feature.load_state(
        study.root / "run",
        {"templates": study.templates},
        {"labels": study.label_files},
    )
    streamed = [(f"{g}__{s}", table) for (g, s), table in sorted(tables.items())]
    feature.fit(InputStream(lambda: iter(streamed), n_entries=len(streamed)))
    return feature
