"""Tests for FeralFeature, end to end.

They cover the FERAL compat helpers and the import guard, the labels dict and
the validation split, the crop and sequence video sources, ``fit`` and
``train_feral`` through FERAL's training pieces, loading a checkpoint, and
``apply`` with its output columns. None needs a GPU, and the ``no_gpu`` fixture
hides any from torch. The tests marked ``feral`` need the ``feral`` package and
are skipped when it is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library import feral_feature
from mosaic.behavior.feature_library.feral_feature import (
    FeralAnswer,
    FeralConfig,
    FeralFeature,
    FeralLabels,
    FeralLabelsJson,
    FeralModelArtifact,
    FeralTrainingConfig,
    FeralVideo,
    _check_feral,
    chunk_probabilities,
    _evaluate_partition,
    _load_checkpoint_state_dict,
    build_feral_labels,
    checkpoint_classes,
    crop_videos,
    imported_network,
    ensembled_predictions,
    feral_config,
    feral_input_mode,
    feral_setup_check,
    refuse_crop_without_videos,
    sequence_video,
    train_feral,
    validation_entries,
)
from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
from mosaic.behavior.visualization_library.interaction_crop import (
    InteractionCropPipeline,
)
from mosaic.core.dataset import (
    Dataset,
    ResolvedMedia,
    UnplacedSequence,
    new_dataset_manifest,
)
from mosaic.core.media.store_division import StoreWindow
from mosaic.core.pipeline.graph import ColumnSet
from mosaic.core.pipeline.graph.columns import measured_columns, output_columns
from mosaic.core.pipeline.progress import NullProgressCallback
from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.core.pipeline.types import GroundTruthLabelsSource, InputStream, Result
from mosaic_media import MediaProbeError
from tests.helpers import Event, clip_facts, hide_package, write_sparse_labels

if TYPE_CHECKING:
    import torch

_HAS_FERAL = importlib.util.find_spec("feral") is not None


class TestLoadCheckpointStateDict:
    """`_load_checkpoint_state_dict` normalizes every checkpoint shape."""

    def test_new_dict_format_unwraps_and_returns_metadata(self) -> None:
        raw = {
            "state_dict": {"backbone.model.enc": 1, "head.weight": 2},
            "class_names": {"0": "none", "1": "troph"},
            "is_multilabel": False,
            "cfg": {"backbone": "vjepa2_vitl_diving48"},
            "class_values": [2, 5],
        }
        sd, meta = _load_checkpoint_state_dict(raw)
        assert sd == {"backbone.model.enc": 1, "head.weight": 2}
        assert meta == {
            "class_names": {"0": "none", "1": "troph"},
            "is_multilabel": False,
            "cfg": {"backbone": "vjepa2_vitl_diving48"},
            "class_values": [2, 5],
        }

    def test_bare_new_layout_passthrough(self) -> None:
        raw = {"backbone.model.enc": 1, "clip_projector.x_q": 2, "head.weight": 3}
        sd, meta = _load_checkpoint_state_dict(raw)
        assert sd == raw
        assert meta is None

    def test_old_hfmodel_keys_are_remapped(self) -> None:
        raw = {
            "model.encoder.layer.0.w": 1,
            "model.embeddings.w": 2,
            "clip_projector.x_q": 3,
            "fc_norm.weight": 4,
            "head.weight": 5,
        }
        sd, meta = _load_checkpoint_state_dict(raw)
        assert meta is None
        # encoder keys get the backbone. prefix; classifier keys untouched.
        assert sd == {
            "backbone.model.encoder.layer.0.w": 1,
            "backbone.model.embeddings.w": 2,
            "clip_projector.x_q": 3,
            "fc_norm.weight": 4,
            "head.weight": 5,
        }

    def test_new_dict_wrapping_old_layout_is_remapped(self) -> None:
        # Defensive: a dict-format wrapper whose inner state_dict is old-layout.
        raw = {"state_dict": {"model.enc": 1, "head.weight": 2}}
        sd, _meta = _load_checkpoint_state_dict(raw)
        assert sd == {"backbone.model.enc": 1, "head.weight": 2}


class TestImportGuard:
    """FeralFeature is registered and constructed even without feral; fit and apply refuse."""

    def test_feature_is_registered(self) -> None:
        # The module imports (light deps only), so the feature registers
        # regardless of whether feral/torch are installed.
        assert "FeralFeature" in FEATURES

    @pytest.mark.skipif(_HAS_FERAL, reason="feral is installed")
    def test_check_feral_raises_helpful_error(self) -> None:
        with pytest.raises(
            ToolUnavailable, match=r"mosaic-behavior\[feral\]"
        ) as raised:
            _check_feral()

        assert raised.value.detail == {"tool": "feral", "cause": "environment"}

    @pytest.mark.skipif(_HAS_FERAL, reason="feral is installed")
    def test_construction_succeeds_and_fit_and_apply_refuse_without_feral(
        self,
    ) -> None:
        feature = FeralFeature(
            FeralFeature.Inputs((Result(feature="upstream"),)),
            {"model_dir": "/tmp/does-not-matter"},
        )

        with pytest.raises(ToolUnavailable, match=r"mosaic-behavior\[feral\]"):
            feature.fit(InputStream(lambda: iter([]), 0))
        with pytest.raises(ToolUnavailable, match=r"mosaic-behavior\[feral\]"):
            _ = feature.apply(pd.DataFrame({"frame": [0]}))

    def test_loading_a_model_without_feral_is_a_tool_unavailable_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``load_state`` finds a checkpoint to apply and needs the package for it."""
        hide_package(monkeypatch, "feral")
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_best.pt").write_bytes(b"")
        feature = FeralFeature(
            FeralFeature.Inputs((Result(feature="upstream"),)),
            {"model_dir": str(model_dir)},
        )

        with pytest.raises(
            ToolUnavailable, match=r"mosaic-behavior\[feral\]"
        ) as raised:
            _ = feature.load_state(tmp_path / "run", {}, {})

        assert raised.value.detail == {"tool": "feral", "cause": "environment"}
        assert isinstance(raised.value.__cause__, ImportError)

    def _refuses_before_reading_the_bundle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        feature: FeralFeature,
        run_root: Path,
        artifact_paths: dict[str, Path],
    ) -> None:
        """``load_state`` refuses without the package, and unpickles nothing."""
        hide_package(monkeypatch, "feral")
        read: list[Path] = []

        def from_path(self: FeralModelArtifact, path: Path) -> object:
            read.append(path)
            return {}

        monkeypatch.setattr(FeralModelArtifact, "from_path", from_path)

        with pytest.raises(ToolUnavailable) as raised:
            _ = feature.load_state(run_root, artifact_paths, {})

        assert raised.value.detail == {"tool": "feral", "cause": "environment"}
        assert isinstance(raised.value.__cause__, ImportError)
        assert read == []

    def test_a_cached_run_without_the_package_is_refused_before_its_bundle_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir()
        _ = (run_root / "feral_model.joblib").write_bytes(b"")
        _ = (run_root / "feral_config.json").write_text("{}")
        feature = _feature({"model": _TRAINED_FERAL})

        self._refuses_before_reading_the_bundle(monkeypatch, feature, run_root, {})

    def test_a_named_model_without_the_package_is_refused_before_its_bundle_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        named = tmp_path / "feral_model.joblib"
        _ = named.write_bytes(b"")
        feature = _feature({"model": _TRAINED_FERAL})

        self._refuses_before_reading_the_bundle(
            monkeypatch, feature, tmp_path / "run", {"model": named}
        )


_EXTRACT_LABELED_TEMPLATES = {
    "feature": "extract-labeled-templates",
    "run_id": "0.1-abc",
}
_TRAINED_FERAL = {"feature": "feral", "run_id": "0.3-abc"}
_TRAINING_SOURCE: dict[str, object] = {
    "templates": _EXTRACT_LABELED_TEMPLATES,
    "labels": {},
    "training": {},
}
_UPSTREAM_INPUTS = FeralFeature.Inputs((Result(feature="upstream"),))


def _feature(params: Mapping[str, object], inputs: Result | None = None):
    items = _UPSTREAM_INPUTS if inputs is None else FeralFeature.Inputs((inputs,))
    return FeralFeature(items, dict(params))


class TestParamsMode:
    """Exactly one of ``templates``, ``model`` and ``model_dir`` chooses the mode."""

    @pytest.mark.parametrize(
        "params",
        [
            _TRAINING_SOURCE,
            {"model": _TRAINED_FERAL},
            {"model_dir": "models/feral/0.1-abc"},
        ],
        ids=["templates", "model", "model_dir"],
    )
    def test_each_single_source_validates(self, params: Mapping[str, object]) -> None:
        _ = FeralFeature.Params.model_validate(dict(params))

    @pytest.mark.parametrize(
        "params",
        [
            {**_TRAINING_SOURCE, "model": _TRAINED_FERAL},
            {**_TRAINING_SOURCE, "model_dir": "models/x"},
            {"model": _TRAINED_FERAL, "model_dir": "models/x"},
            {},
        ],
        ids=["templates+model", "templates+model_dir", "model+model_dir", "none"],
    )
    def test_two_sources_or_none_are_refused(
        self, params: Mapping[str, object]
    ) -> None:
        with pytest.raises(ValidationError, match="Exactly one of templates"):
            _ = FeralFeature.Params.model_validate(dict(params))

    @pytest.mark.parametrize("field", ["labels", "training"])
    def test_a_fit_only_param_is_refused_without_templates(self, field: str) -> None:
        value: dict[str, object] = {}
        with pytest.raises(ValidationError, match=f"{field} is read when training"):
            _ = FeralFeature.Params.model_validate(
                {"model_dir": "models/x", field: value}
            )

    def test_unlabeled_as_is_refused_without_templates(self) -> None:
        with pytest.raises(ValidationError, match="unlabeled_as is read when training"):
            _ = FeralFeature.Params.model_validate(
                {"model": _TRAINED_FERAL, "unlabeled_as": 0}
            )

    @pytest.mark.parametrize("missing", ["labels", "training"])
    def test_templates_need_labels_and_training(self, missing: str) -> None:
        params = {k: v for k, v in _TRAINING_SOURCE.items() if k != missing}

        with pytest.raises(
            ValidationError, match=f"{missing} is required with templates"
        ):
            _ = FeralFeature.Params.model_validate(params)

    @pytest.mark.parametrize(
        "params",
        [
            {**_TRAINING_SOURCE, "unlabeled_as": 0},
            {"model": _TRAINED_FERAL},
            {"model_dir": "models/x"},
        ],
        ids=["templates", "model", "model_dir"],
    )
    def test_a_dump_validates_again(self, params: Mapping[str, object]) -> None:
        """The dump writes each unset field as ``null``, which counts as unset."""
        dumped = FeralFeature.Params.model_validate(dict(params)).model_dump()

        again = FeralFeature.Params.model_validate(dumped)

        assert again == FeralFeature.Params.model_validate(dict(params))

    def test_the_payload_the_api_validates_leaves_model_dir_unset(self) -> None:
        params = FeralFeature.Params.from_overrides(
            {
                "templates": _EXTRACT_LABELED_TEMPLATES,
                "labels": {"kind": "behavior"},
                "training": {"epochs": 3},
            }
        )

        assert params.model_dir is None
        assert params.model is None

    @pytest.mark.parametrize("value", [-0.1, 1.0])
    def test_a_validation_fraction_outside_zero_to_one_is_refused(
        self, value: float
    ) -> None:
        with pytest.raises(ValidationError):
            _ = FeralTrainingConfig.model_validate({"val_fraction": value})

    def test_the_validation_fraction_defaults_to_a_fifth_and_accepts_zero(self) -> None:
        assert FeralTrainingConfig().val_fraction == 0.2
        assert FeralTrainingConfig.model_validate({"val_fraction": 0}).val_fraction == 0


class TestFeralDeclaration:
    """What the class declares about fitting and about the roots it reads."""

    def test_a_run_given_templates_fits(self) -> None:
        assert _feature(_TRAINING_SOURCE).fits is True

    @pytest.mark.parametrize(
        "params", [{"model": _TRAINED_FERAL}, {"model_dir": "models/x"}]
    )
    def test_a_run_given_a_model_fits_nothing(
        self, params: Mapping[str, object]
    ) -> None:
        assert _feature(params).fits is False

    def test_the_class_keeps_the_declarations_of_a_pinned_fitter(self) -> None:
        assert FeralFeature.fits is True
        assert FeralFeature.scope_dependent is False
        assert FeralFeature.fit_set_pinned_by == "templates"

    def test_it_reads_media_through_resolve_media(self) -> None:
        assert "media_raw" in FeralFeature.consumed_roots

    def test_the_model_artifact_is_the_bundle_of_a_feral_run(self) -> None:
        assert FeralFeature.ModelArtifact is FeralModelArtifact
        assert FeralModelArtifact().feature == "feral"
        assert FeralModelArtifact().pattern == "feral_model.joblib"


def _bundle_file(
    run_root: Path, class_values: Sequence[int] = (2, 5), *, config: bool = True
) -> dict[str, object]:
    """Write a ``feral_model.joblib`` of plain values into *run_root*."""
    import joblib

    bundle: dict[str, object] = {
        "state_dict": {},
        "class_values": list(class_values),
        "class_names": {value: _NAMES[value] for value in class_values},
        "is_multilabel": False,
        "cfg": {
            "backbone": "vjepa2_vitl_diving48",
            "predict_per_item": 8,
            "model": {"fc_drop_rate": 0.5, "freeze_encoder_layers": 2},
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    _ = joblib.dump(bundle, run_root / "feral_model.joblib")
    if config:
        _ = (run_root / "feral_config.json").write_text("{}")
    return bundle


class _LoadedModels:
    """What ``_load_model`` was handed, recorded in place of loading."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, Path | None]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def record(
            feature: FeralFeature, source: object, config_path: Path | None = None
        ) -> None:
            self.calls.append((source, config_path))

        def package_present() -> None:
            return None

        monkeypatch.setattr(feral_feature, "_check_feral", package_present)
        monkeypatch.setattr(FeralFeature, "_load_model", record)


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> _LoadedModels:
    recorder = _LoadedModels()
    recorder.install(monkeypatch)
    return recorder


def _templates_run(
    root: Path, train: Sequence[str], test: Sequence[str], *, provenance: bool = True
) -> Path:
    """The ``templates.parquet`` of a run with its provenance, as ``artifact_paths`` names it."""
    root.mkdir(parents=True, exist_ok=True)
    templates = root / "templates.parquet"
    _ = templates.write_bytes(b"")
    if provenance:
        pd.DataFrame(
            {
                "group": "g",
                "sequence": [*train, *test],
                "split": ["train"] * len(train) + ["test"] * len(test),
            }
        ).to_parquet(root / "template_provenance.parquet")
    return templates


class TestLoadStateOrder:
    """``load_state`` tries a cached run, a ``model``, a ``model_dir``, then ``templates``."""

    def test_a_cached_run_wins_over_the_model_named(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        cached = _bundle_file(tmp_path / "run", class_values=(2, 5))
        named = tmp_path / "named"
        _ = _bundle_file(named, class_values=(0, 1))
        feature = _feature({"model": _TRAINED_FERAL})

        state_ready = feature.load_state(
            tmp_path / "run", {"model": named / "feral_model.joblib"}, {}
        )

        assert state_ready is True
        assert [call[0] for call in loaded.calls] == [cached]

    def test_the_model_named_is_loaded_without_a_cached_run(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        named = tmp_path / "named"
        bundle = _bundle_file(named, class_values=(0, 1))
        feature = _feature({"model": _TRAINED_FERAL})

        state_ready = feature.load_state(
            tmp_path / "run", {"model": named / "feral_model.joblib"}, {}
        )

        assert state_ready is True
        assert [call[0] for call in loaded.calls] == [bundle]

    def test_a_cached_model_file_without_its_config_is_not_whole(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        _ = _bundle_file(tmp_path / "run", config=False)
        feature = _feature({"model": _TRAINED_FERAL})

        with pytest.raises(FileNotFoundError):
            _ = feature.load_state(tmp_path / "run", {}, {})

        assert loaded.calls == []

    def test_a_model_that_did_not_resolve_raises_and_fits_nothing(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        feature = _feature({"model": _TRAINED_FERAL})

        with pytest.raises(FileNotFoundError, match="0.3-abc"):
            _ = feature.load_state(tmp_path / "run", {}, {})

        assert loaded.calls == []

    def test_a_relative_model_dir_without_a_checkpoint_names_its_resolved_path(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        ds = _dataset(tmp_path)
        resolved = ds.get_root("models") / "feral" / "0.1-abc"
        resolved.mkdir(parents=True)
        feature = _feature({"model_dir": "models/feral/0.1-abc"})
        feature.bind_dataset(ds)

        with pytest.raises(FileNotFoundError, match="model_best.pt") as raised:
            _ = feature.load_state(tmp_path / "run", {}, {})

        assert str(resolved) in str(raised.value)
        assert loaded.calls == []

    def test_a_model_dir_loads_its_checkpoint_and_optional_config(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        ds = _dataset(tmp_path)
        model_dir = ds.get_root("models") / "feral" / "0.1-abc"
        model_dir.mkdir(parents=True)
        _ = (model_dir / "model_best.pt").write_bytes(b"")
        feature = _feature({"model_dir": "models/feral/0.1-abc"})
        feature.bind_dataset(ds)

        assert feature.load_state(tmp_path / "run", {}, {}) is True
        _ = (model_dir / "config.json").write_text("{}")
        assert feature.load_state(tmp_path / "run", {}, {}) is True

        assert loaded.calls == [
            (model_dir / "model_best.pt", None),
            (model_dir / "model_best.pt", model_dir / "config.json"),
        ]

    def test_templates_record_the_entries_and_ask_for_a_fit(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        templates = _templates_run(tmp_path / "templates", ["a", "b"], ["c"])
        labels = {("g", "a"): tmp_path / "a.npz"}
        feature = _feature(_TRAINING_SOURCE)

        state_ready = feature.load_state(
            tmp_path / "run", {"templates": templates}, {"labels": labels}
        )

        assert state_ready is False
        assert loaded.calls == []

    def test_templates_without_provenance_are_refused(self, tmp_path: Path) -> None:
        templates = _templates_run(tmp_path / "templates", [], [], provenance=False)
        feature = _feature(_TRAINING_SOURCE)

        with pytest.raises(ValueError, match="template_provenance.parquet"):
            _ = feature.load_state(tmp_path / "run", {"templates": templates}, {})


_CROP_RESULT = Result(
    feature="interaction-crop-pipeline__from__tracks", run_id="0.1-abc"
)


class TestLoadStateCropInput:
    """A crop input names a run that wrote videos, on every branch of ``load_state``."""

    def _crop_lookup(self, run_root: Path) -> dict[str, dict[tuple[str, str], Path]]:
        return {"_input_0": {("g", "s"): run_root / "table.parquet"}}

    @pytest.mark.parametrize(
        ("crop_params", "reason"),
        [
            ({"output_root": "/elsewhere"}, "output_root"),
            ({"output_mode": "frames"}, "frames"),
        ],
        ids=["output_root", "frames"],
    )
    def test_a_model_run_refuses_a_crop_input_that_wrote_no_video(
        self, tmp_path: Path, crop_params: Mapping[str, object], reason: str
    ) -> None:
        crop_root = _crop_run(tmp_path, _params_json(crop_params))
        feature = _feature({"model": _TRAINED_FERAL}, _CROP_RESULT)

        with pytest.raises(ValueError, match=reason) as raised:
            _ = feature.load_state(tmp_path / "run", {}, self._crop_lookup(crop_root))

        assert crop_root.name in str(raised.value)

    def test_a_cached_run_refuses_it_too(self, tmp_path: Path) -> None:
        crop_root = _crop_run(tmp_path, _params_json({"output_root": "/elsewhere"}))
        _ = _bundle_file(tmp_path / "run")
        feature = _feature({"model": _TRAINED_FERAL}, _CROP_RESULT)

        with pytest.raises(ValueError, match="output_root"):
            _ = feature.load_state(tmp_path / "run", {}, self._crop_lookup(crop_root))

    def test_a_crop_input_with_no_lookup_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        feature = _feature({"model": _TRAINED_FERAL}, _CROP_RESULT)

        with pytest.raises(ValueError, match="interaction-crop-pipeline__from__tracks"):
            _ = feature.load_state(tmp_path / "run", {}, {})

    def test_the_input_run_root_is_the_first_inputs_not_a_params_lookup(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        """A labels lookup listed first would name a directory with no params.json."""
        crop_root = _crop_run(tmp_path, _params_json({"output_root": "/elsewhere"}))
        labels_root = tmp_path / "labels"
        labels_root.mkdir()
        _ = _bundle_file(tmp_path / "run")
        feature = _feature({"model": _TRAINED_FERAL}, _CROP_RESULT)
        lookups = {
            "labels": {("g", "s"): labels_root / "g__s.npz"},
            **self._crop_lookup(crop_root),
        }

        with pytest.raises(ValueError, match="output_root") as raised:
            _ = feature.load_state(tmp_path / "run", {}, lookups)

        assert crop_root.name in str(raised.value)

    def test_a_sequence_input_needs_no_run_root(
        self, tmp_path: Path, loaded: _LoadedModels
    ) -> None:
        _ = _bundle_file(tmp_path / "run")
        feature = _feature({"model": _TRAINED_FERAL}, Result(feature="speed-angvel"))

        assert feature.load_state(tmp_path / "run", {}, {}) is True


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestResolveBackboneKey:
    """`_resolve_backbone_key` accepts a BACKBONES key or a HuggingFace slug."""

    def test_key_passthrough(self) -> None:
        from mosaic.behavior.feature_library.feral_feature import _resolve_backbone_key

        assert _resolve_backbone_key("vjepa2_vitl_diving48") == "vjepa2_vitl_diving48"

    def test_slug_maps_to_key(self) -> None:
        from mosaic.behavior.feature_library.feral_feature import _resolve_backbone_key

        assert (
            _resolve_backbone_key("facebook/vjepa2-vitl-fpc32-256-diving48")
            == "vjepa2_vitl_diving48"
        )

    def test_unknown_raises(self) -> None:
        from mosaic.behavior.feature_library.feral_feature import _resolve_backbone_key

        with pytest.raises(ValueError, match="Unknown FERAL backbone"):
            _resolve_backbone_key("not-a-real-backbone")


def test_the_training_config_refuses_wandb_project() -> None:
    with pytest.raises(ValidationError, match="wandb_project"):
        _ = FeralTrainingConfig.model_validate({"wandb_project": "behavior"})


_NAMES: tuple[str, ...] = ("zero", "one", "walk", "three", "four", "run")
"""A vocabulary in which the values 2 and 5 name ``walk`` and ``run``."""

_INTERACTION_IDS: dict[str, int | str] = {
    "id1": 0,
    "id2": 1,
    "perspective": 0,
    "interaction_id": 0,
}


def _entry_files(
    tmp_path: Path, events_by_entry: Mapping[tuple[str, str], Sequence[Event]]
) -> dict[tuple[str, str], Path]:
    return {
        entry: write_sparse_labels(
            tmp_path / f"{entry[0]}__{entry[1]}.npz", events, names=_NAMES
        )
        for entry, events in events_by_entry.items()
    }


def _video(
    tmp_path: Path,
    entry: tuple[str, str],
    frames: Sequence[int],
    ids: Mapping[str, int | str] | None = None,
) -> FeralVideo:
    filename = f"{entry[0]}__{entry[1]}.mp4"
    return FeralVideo(
        entry=entry,
        filename=filename,
        path=tmp_path / filename,
        frames=tuple(frames),
        ids=_INTERACTION_IDS if ids is None else ids,
    )


def _build(
    train: Sequence[FeralVideo],
    label_files: Mapping[tuple[str, str], Path],
    *,
    val: Sequence[FeralVideo] = (),
    test: Sequence[FeralVideo] = (),
    unlabeled_as: int | None = None,
    min_frames: int = 1,
    labels: GroundTruthLabelsSource | None = None,
):
    return build_feral_labels(
        {"train": list(train), "val": list(val), "test": list(test)},
        label_files,
        labels if labels is not None else GroundTruthLabelsSource(),
        unlabeled_as=unlabeled_as,
        min_frames=min_frames,
    )


def _pair_events(first: int, second: int, count: int = 2) -> list[Event]:
    """``count`` frames of ``first`` then ``count`` of ``second``, pair (0, 1)."""
    return [(f, 0, 1, first if f < count else second) for f in range(2 * count)]


class TestBuildFeralLabels:
    """`build_feral_labels` labels each video's frames and shapes FERAL's dict."""

    def test_an_interaction_video_labels_its_frames(self, tmp_path: Path) -> None:
        entry = ("g", "pair")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4))

        result = _build([video], files)

        assert result.labels_json["labels"] == {video.filename: [0, 0, 1, 1]}
        assert result.labels_json["splits"] == {
            "train": [video.filename],
            "val": [],
            "test": [],
        }

    def test_an_interaction_video_takes_the_label_of_its_own_pair(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "pair")
        events: list[Event] = [(0, 0, 1, 2), (0, 1, 0, 5), (1, 0, 1, 2), (1, 1, 0, 5)]
        files = _entry_files(tmp_path, {entry: events})
        forward = _video(tmp_path, entry, [0, 1])
        reverse = FeralVideo(
            entry=entry,
            filename="reverse.mp4",
            path=tmp_path / "reverse.mp4",
            frames=(0, 1),
            ids={**_INTERACTION_IDS, "id1": 1, "id2": 0},
        )

        result = _build([forward, reverse], files)

        assert result.labels_json["labels"] == {
            forward.filename: [0, 0],
            "reverse.mp4": [1, 1],
        }

    def test_an_egocentric_video_labels_its_frames(self, tmp_path: Path) -> None:
        entry = ("g", "ego")
        events: list[Event] = [(f, 3, -1, 2 if f < 2 else 5) for f in range(4)]
        files = _entry_files(tmp_path, {entry: events})
        video = _video(tmp_path, entry, range(4), {"id": 3})

        result = _build([video], files)

        assert result.labels_json["labels"] == {video.filename: [0, 0, 1, 1]}

    def test_an_egocentric_video_with_a_gap_labels_the_frames_it_holds(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "ego")
        by_frame = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 5, 6: 5}
        events: list[Event] = [(f, 3, -1, label) for f, label in by_frame.items()]
        files = _entry_files(tmp_path, {entry: events})
        video = _video(tmp_path, entry, [0, 1, 5, 6], {"id": 3})

        result = _build([video], files)

        assert result.labels_json["labels"] == {video.filename: [0, 0, 1, 1]}

    def test_a_sequence_video_labels_its_frames(self, tmp_path: Path) -> None:
        entry = ("g", "seq")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4), {})

        result = _build([video], files)

        assert result.labels_json["labels"] == {video.filename: [0, 0, 1, 1]}

    def test_a_sequence_video_aligns_by_frame_alone_when_several_events_cover_a_frame(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "seq")
        events: list[Event] = [
            (0, 0, 1, 2),
            (0, 2, 3, 5),
            (1, 0, 1, 2),
            (1, 2, 3, 2),
            (2, 0, 1, 5),
        ]
        files = _entry_files(tmp_path, {entry: events})
        video = _video(tmp_path, entry, [0, 1, 2], {})

        result = _build([video], files)

        assert result.labels_json["labels"] == {video.filename: [1, 0, 1]}

    def test_unlabeled_frames_take_unlabeled_as(self, tmp_path: Path) -> None:
        entry = ("g", "seq")
        events: list[Event] = [(0, 0, 1, 2), (1, 0, 1, 5)]
        files = _entry_files(tmp_path, {entry: events})
        video = _video(tmp_path, entry, range(4), {})

        result = _build([video], files, unlabeled_as=2)

        assert result.labels_json["labels"] == {video.filename: [0, 1, 0, 0]}
        assert result.class_values == (2, 5)
        assert result.dropped["unlabeled_frames"] == 0

    def test_a_video_with_unlabeled_frames_leaves_its_split_without_unlabeled_as(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        full = ("g", "full")
        partial = ("g", "partial")
        files = _entry_files(
            tmp_path,
            {full: _pair_events(2, 5), partial: [(0, 0, 1, 2), (1, 0, 1, 5)]},
        )
        kept = _video(tmp_path, full, range(4), {})
        dropped = _video(tmp_path, partial, range(4), {})

        with caplog.at_level(logging.WARNING):
            result = _build([kept, dropped], files)

        assert result.labels_json["splits"]["train"] == [kept.filename]
        assert dropped.filename not in result.labels_json["labels"]
        assert result.dropped["unlabeled_frames"] == 1
        assert sum("unlabeled" in r.getMessage() for r in caplog.records) == 1

    def test_a_masked_class_reads_as_unlabeled(self, tmp_path: Path) -> None:
        masked_entry = ("g", "masked")
        full_entry = ("g", "full")
        events: list[Event] = [(0, 0, 1, 2), (1, 0, 1, 3), (2, 0, 1, 5)]
        files = _entry_files(
            tmp_path, {masked_entry: events, full_entry: _pair_events(2, 5)}
        )
        video = _video(tmp_path, masked_entry, range(3), {})
        full = _video(tmp_path, full_entry, range(4), {})
        masked = GroundTruthLabelsSource(mask=[3])

        refused = _build([full, video], files, labels=masked)
        filled = _build([full, video], files, labels=masked, unlabeled_as=5)

        assert refused.dropped["unlabeled_frames"] == 1
        assert refused.labels_json["splits"]["train"] == [full.filename]
        assert filled.dropped["unlabeled_frames"] == 0
        assert filled.labels_json["labels"][video.filename] == [0, 1, 1]

    def test_a_video_shorter_than_min_frames_leaves_every_split(
        self, tmp_path: Path
    ) -> None:
        long_entry = ("g", "long")
        short_entry = ("g", "short")
        files = _entry_files(
            tmp_path,
            {long_entry: _pair_events(2, 5), short_entry: _pair_events(2, 5)},
        )
        long_video = _video(tmp_path, long_entry, range(4), {})
        short_train = _video(tmp_path, short_entry, range(3), {})
        short_test = FeralVideo(
            entry=short_entry,
            filename="short_test.mp4",
            path=tmp_path / "short_test.mp4",
            frames=(0, 1, 2),
            ids={},
        )

        result = _build(
            [long_video, short_train], files, test=[short_test], min_frames=4
        )

        assert result.labels_json["splits"] == {
            "train": [long_video.filename],
            "val": [],
            "test": [],
        }
        assert result.dropped["too_short"] == 2

    def test_a_video_without_a_label_file_leaves_its_split(
        self, tmp_path: Path
    ) -> None:
        labeled = ("g", "labeled")
        bare = ("g", "bare")
        files = _entry_files(tmp_path, {labeled: _pair_events(2, 5)})
        kept = _video(tmp_path, labeled, range(4), {})
        lost = _video(tmp_path, bare, range(4), {})

        result = _build([kept], files, val=[lost])

        assert result.labels_json["splits"]["val"] == []
        assert result.dropped["no_label_file"] == 1

    def test_values_map_to_dense_indices_named_by_the_vocabulary(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "seq")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4), {})

        result = _build([video], files)

        assert result.class_values == (2, 5)
        assert result.labels_json["class_names"] == {"0": "walk", "1": "run"}
        assert result.labels_json["is_multilabel"] is False

    def test_a_swapping_remap_names_each_class_after_the_frames_it_holds(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "seq")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4), {})

        result = _build(
            [video], files, labels=GroundTruthLabelsSource(remap={2: 5, 5: 2})
        )

        assert result.class_values == (2, 5)
        assert result.labels_json["labels"] == {video.filename: [1, 1, 0, 0]}
        assert result.labels_json["class_names"] == {"0": "run", "1": "walk"}

    def test_ids_that_a_remap_joins_share_the_joined_name(self, tmp_path: Path) -> None:
        entry = ("g", "seq")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4), {})

        result = _build([video], files, labels=GroundTruthLabelsSource(remap={5: 2}))

        assert result.class_values == (2,)
        assert result.labels_json["class_names"] == {"0": "walk+run"}

    def test_a_masked_id_gives_its_name_to_no_class(self, tmp_path: Path) -> None:
        entry = ("g", "seq")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4), {})

        result = _build(
            [video],
            files,
            labels=GroundTruthLabelsSource(mask=[5]),
            unlabeled_as=2,
        )

        assert result.class_values == (2,)
        assert result.labels_json["class_names"] == {"0": "walk"}

    def test_a_value_without_a_vocabulary_is_named_by_its_spelling(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "seq")
        path = tmp_path / "bare.npz"
        events = _pair_events(2, 5)
        np.savez(
            path,
            frames=np.array([e[0] for e in events], dtype=np.int64),
            individual_ids=np.array([[e[1], e[2]] for e in events], dtype=np.int64),
            labels=np.array([e[3] for e in events], dtype=np.int64),
            background_id=np.array(-1, dtype=np.int32),
        )
        video = _video(tmp_path, entry, range(4), {})

        result = _build([video], {entry: path})

        assert result.labels_json["class_names"] == {"0": "2", "1": "5"}

    def test_a_test_video_holding_a_value_absent_from_train_is_dropped(
        self, tmp_path: Path
    ) -> None:
        train_entry = ("g", "train")
        test_entry = ("g", "test")
        files = _entry_files(
            tmp_path,
            {
                train_entry: _pair_events(2, 5),
                test_entry: _pair_events(2, 3),
                ("g", "ok"): _pair_events(5, 2),
            },
        )
        train = _video(tmp_path, train_entry, range(4), {})
        bad = _video(tmp_path, test_entry, range(4), {})
        good = _video(tmp_path, ("g", "ok"), range(4), {})

        result = _build([train], files, val=[good], test=[bad])

        assert result.labels_json["splits"]["test"] == []
        assert result.labels_json["splits"]["val"] == [good.filename]
        assert result.dropped["unseen_class"] == 1
        remapped = [v for row in result.labels_json["labels"].values() for v in row]
        assert set(remapped) <= {0, 1}

    def test_every_label_is_a_python_int_and_the_dict_is_json(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "seq")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4), {})

        result = _build([video], files)

        for row in result.labels_json["labels"].values():
            assert {type(value) for value in row} == {int}
        assert json.loads(json.dumps(result.labels_json)) == result.labels_json

    def test_no_train_video_left_raises_naming_the_feature(
        self, tmp_path: Path
    ) -> None:
        entry = ("g", "seq")
        files = _entry_files(tmp_path, {entry: _pair_events(2, 5)})
        video = _video(tmp_path, entry, range(4), {})

        with pytest.raises(ValueError, match="feral"):
            _ = _build([video], files, min_frames=10)


class TestValidationEntries:
    """`validation_entries` draws whole train entries with a seed."""

    ENTRIES = [("g", f"s{i}") for i in range(10)]

    def test_the_same_seed_draws_the_same_entries_for_shuffled_input(self) -> None:
        shuffled = [self.ENTRIES[i] for i in (7, 2, 9, 0, 5, 1, 8, 3, 6, 4)]

        first = validation_entries(self.ENTRIES, 0.3, 11)
        second = validation_entries(shuffled, 0.3, 11)

        assert first == second

    def test_the_count_is_the_rounded_fraction_of_the_entries(self) -> None:
        drawn = validation_entries(self.ENTRIES, 0.25, 0)

        assert len(drawn) == round(0.25 * 10)
        assert set(drawn) <= set(self.ENTRIES)
        assert len(set(drawn)) == len(drawn)

    def test_a_fraction_leaving_no_train_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="train"):
            _ = validation_entries(self.ENTRIES[:2], 0.9, 0)

    def test_a_fraction_rounding_to_zero_draws_none(self) -> None:
        assert validation_entries(self.ENTRIES, 0.04, 0) == []

    def test_different_seeds_draw_different_entries(self) -> None:
        assert validation_entries(self.ENTRIES, 0.3, 0) != validation_entries(
            self.ENTRIES, 0.3, 1
        )

    def test_the_draw_is_not_the_leading_entries(self) -> None:
        drawn = validation_entries(self.ENTRIES, 0.3, 0)

        assert drawn != self.ENTRIES[: len(drawn)]


class TestFeralInputMode:
    """The input's feature slug picks where FERAL's videos come from."""

    def test_each_crop_slug_selects_its_mode(self) -> None:
        assert feral_input_mode(InteractionCropPipeline.name) == "interaction"
        assert feral_input_mode(EgocentricCrop.name) == "egocentric"

    def test_the_crop_slugs_are_the_registered_names(self) -> None:
        assert InteractionCropPipeline.name == "interaction-crop-pipeline"
        assert EgocentricCrop.name == "egocentric-crop"

    def test_a_crop_stored_from_its_input_keeps_its_mode(self) -> None:
        assert feral_input_mode("egocentric-crop__from__tracks") == "egocentric"
        assert (
            feral_input_mode("interaction-crop-pipeline__from__tracks") == "interaction"
        )

    def test_a_shortened_storage_name_keeps_its_mode(self) -> None:
        from mosaic.core.pipeline.storage_names import derive_storage_name

        shortened = derive_storage_name("egocentric-crop", "tracks+" + "x" * 300)

        assert "~" in shortened
        assert feral_input_mode(shortened) == "egocentric"

    def test_another_feature_reads_the_sequence_video(self) -> None:
        assert feral_input_mode("speed-angvel__from__tracks") == "sequence"

    def test_a_track_input_reads_the_sequence_video(self) -> None:
        assert feral_input_mode("tracks") == "sequence"

    def test_a_slug_containing_a_crop_slug_is_not_a_crop(self) -> None:
        assert feral_input_mode("egocentric-crop-stats__from__tracks") == "sequence"


def _crop_run(tmp_path: Path, params_json: str | None) -> Path:
    run_root = tmp_path / "features" / "egocentric-crop__from__tracks" / "0.1-abc"
    run_root.mkdir(parents=True)
    if params_json is not None:
        _ = (run_root / "params.json").write_text(params_json)
    return run_root


def _params_json(params: Mapping[str, object]) -> str:
    return json.dumps({"_params": dict(params)})


class TestRefuseCropWithoutVideos:
    """A crop run that wrote no video under its run root is refused by name."""

    def test_a_run_with_an_output_root_is_refused(self, tmp_path: Path) -> None:
        run_root = _crop_run(tmp_path, _params_json({"output_root": "/elsewhere"}))

        with pytest.raises(ValueError, match="output_root") as raised:
            refuse_crop_without_videos(run_root)

        assert run_root.name in str(raised.value)

    def test_a_frames_only_run_is_refused(self, tmp_path: Path) -> None:
        run_root = _crop_run(tmp_path, _params_json({"output_mode": "frames"}))

        with pytest.raises(ValueError, match="frames") as raised:
            refuse_crop_without_videos(run_root)

        assert run_root.name in str(raised.value)

    def test_a_run_without_params_is_refused_as_unknown(self, tmp_path: Path) -> None:
        run_root = _crop_run(tmp_path, None)

        with pytest.raises(ValueError, match="params.json is missing") as raised:
            refuse_crop_without_videos(run_root)

        assert run_root.name in str(raised.value)

    def test_a_run_whose_params_are_not_a_mapping_is_refused(
        self, tmp_path: Path
    ) -> None:
        run_root = _crop_run(tmp_path, "[1, 2]")

        with pytest.raises(ValueError, match="params.json is unreadable"):
            refuse_crop_without_videos(run_root)

    @pytest.mark.parametrize("mode", ["video", "both"])
    def test_a_run_that_wrote_videos_passes(self, tmp_path: Path, mode: str) -> None:
        run_root = _crop_run(tmp_path, _params_json({"output_mode": mode}))

        refuse_crop_without_videos(run_root)

    def test_a_run_with_default_params_passes(self, tmp_path: Path) -> None:
        run_root = _crop_run(tmp_path, _params_json({"output_root": None}))

        refuse_crop_without_videos(run_root)


def _interaction_rows(*clips: tuple[str, int, int, int, int]) -> pd.DataFrame:
    """One row per ``(video_path, start_frame, end_frame, n_frames, interaction_id)``."""
    return pd.DataFrame(
        [
            {
                "group": "g",
                "sequence": "s",
                "id1": 0,
                "id2": 1,
                "perspective": 0,
                "interaction_id": interaction_id,
                "start_frame": start,
                "end_frame": end,
                "n_frames": count,
                "video_path": video_path,
            }
            for video_path, start, end, count, interaction_id in clips
        ]
    )


class TestCropVideos:
    """`crop_videos` turns one entry's crop rows into FERAL videos."""

    def test_an_interaction_row_is_one_video_over_its_frame_range(
        self, tmp_path: Path
    ) -> None:
        input_root = tmp_path / "features" / "crop" / "run"
        rows = _interaction_rows(("clip0.mp4", 10, 12, 3, 4))

        videos, dropped = crop_videos("interaction", input_root, rows, tmp_path)

        assert dropped == 0
        assert videos == [
            FeralVideo(
                entry=("g", "s"),
                filename="features/crop/run/clip0.mp4",
                path=input_root / "clip0.mp4",
                frames=(10, 11, 12),
                ids={"id1": 0, "id2": 1, "perspective": 0, "interaction_id": 4},
            )
        ]

    def test_an_interaction_row_with_a_short_frame_count_is_dropped_and_counted(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        rows = _interaction_rows(
            ("whole.mp4", 0, 4, 5, 0),
            ("gappy.mp4", 10, 19, 8, 1),
            ("also_gappy.mp4", 30, 34, 4, 2),
        )

        with caplog.at_level(logging.WARNING):
            videos, dropped = crop_videos("interaction", tmp_path, rows, tmp_path)

        assert [video.filename for video in videos] == ["whole.mp4"]
        assert dropped == 2
        assert any(
            record.getMessage().startswith("[feral] 2 interaction clip(s)")
            for record in caplog.records
        )

    def test_an_egocentric_id_is_one_video_over_the_frames_of_its_rows(
        self, tmp_path: Path
    ) -> None:
        input_root = tmp_path / "run"
        rows = pd.DataFrame(
            {
                "group": "g",
                "sequence": "s",
                "id": [1, 0, 0, 1, 0, 1],
                "frame": [7, 5, 0, 8, 1, 2],
            }
        )

        videos, dropped = crop_videos("egocentric", input_root, rows, tmp_path)

        assert dropped == 0
        assert [video.ids for video in videos] == [{"id": 0}, {"id": 1}]
        assert [video.frames for video in videos] == [(0, 1, 5), (2, 7, 8)]
        assert [video.path for video in videos] == [
            input_root / "g__s" / "egocentric_id0.mp4",
            input_root / "g__s" / "egocentric_id1.mp4",
        ]
        assert [video.filename for video in videos] == [
            "run/g__s/egocentric_id0.mp4",
            "run/g__s/egocentric_id1.mp4",
        ]
        assert {video.entry for video in videos} == {("g", "s")}

    def test_a_crop_outside_the_dataset_keeps_its_absolute_filename(
        self, tmp_path: Path
    ) -> None:
        dataset_root = tmp_path / "ds"
        dataset_root.mkdir()
        input_root = tmp_path / "elsewhere"
        rows = _interaction_rows(("clip0.mp4", 0, 1, 2, 0))

        videos, _ = crop_videos("interaction", input_root, rows, dataset_root)

        assert videos[0].filename == str(input_root / "clip0.mp4")

    def test_no_rows_give_no_videos(self, tmp_path: Path) -> None:
        rows = _interaction_rows()

        assert crop_videos("interaction", tmp_path, rows, tmp_path) == ([], 0)
        assert crop_videos("egocentric", tmp_path, rows, tmp_path) == ([], 0)


def _media(
    paths: Sequence[Path],
    *,
    frame_counts: Sequence[int] = (5,),
    windows: Sequence[StoreWindow | None] | None = None,
    placement: int | UnplacedSequence | None = None,
) -> ResolvedMedia:
    return ResolvedMedia(
        paths=list(paths),
        facts=[clip_facts(frame_count=count) for count in frame_counts],
        windows=list(windows) if windows is not None else [None] * len(paths),
        placement=placement,
    )


def _counting(count: int | None) -> Callable[[str], int | None]:
    """A frame counter that returns *count* for the string path it is given."""

    def frame_count(path: str) -> int | None:
        assert isinstance(path, str)
        return count

    return frame_count


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("f", base_dir=tmp_path)
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _resolves(
    monkeypatch: pytest.MonkeyPatch, outcome: ResolvedMedia | Exception
) -> None:
    """Make every ``Dataset.resolve_media`` call return or raise *outcome*."""

    def resolve(
        self: Dataset,
        group: str,
        sequence: str,
        camera: str | None = None,
        index_filename: str = "index.csv",
    ) -> ResolvedMedia:
        assert (group, sequence) == ("g", "s1")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(Dataset, "resolve_media", resolve)


ENTRY_TEXT = re.escape("('g', 's1')")


class TestSequenceVideo:
    """`sequence_video` turns an entry's own media into one FERAL video."""

    def test_one_plain_file_is_one_video_over_its_counted_frames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        path = ds.base_dir / "media_raw" / "a.mp4"
        _resolves(monkeypatch, _media([path]))

        video = sequence_video(ds, "g", "s1", _counting(5))

        assert video == FeralVideo(
            entry=("g", "s1"),
            filename="media_raw/a.mp4",
            path=path,
            frames=(0, 1, 2, 3, 4),
            ids={},
        )

    def test_a_placed_sequence_offsets_its_frames_by_its_first_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        path = ds.base_dir / "media_raw" / "a.mp4"
        _resolves(monkeypatch, _media([path], placement=100, windows=[None]))

        video = sequence_video(ds, "g", "s1", _counting(5))

        assert video.frames == (100, 101, 102, 103, 104)

    def test_a_file_outside_the_dataset_keeps_its_absolute_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path / "ds")
        path = tmp_path / "elsewhere" / "a.mp4"
        _resolves(monkeypatch, _media([path]))

        video = sequence_video(ds, "g", "s1", _counting(5))

        assert video.filename == str(path)
        assert video.path == path

    def test_several_files_are_refused_by_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        paths = [ds.base_dir / "a.mp4", ds.base_dir / "b.mp4"]
        _resolves(monkeypatch, _media(paths, frame_counts=(5, 5)))

        with pytest.raises(ValueError, match=ENTRY_TEXT) as raised:
            _ = sequence_video(ds, "g", "s1", _counting(5))

        assert "2 media files" in str(raised.value)

    def test_a_store_window_is_refused_by_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        media = _media([ds.base_dir / "store"], windows=[StoreWindow(0, 1)])
        _resolves(monkeypatch, media)

        with pytest.raises(ValueError, match=ENTRY_TEXT) as raised:
            _ = sequence_video(ds, "g", "s1", _counting(5))

        assert "store" in str(raised.value)

    def test_a_media_error_from_resolving_is_refused_with_its_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        original = MediaProbeError("the sequence spans 2 cameras")
        _resolves(monkeypatch, original)

        with pytest.raises(ValueError, match=ENTRY_TEXT) as raised:
            _ = sequence_video(ds, "g", "s1", _counting(5))

        assert "the sequence spans 2 cameras" in str(raised.value)
        assert raised.value.__cause__ is original

    def test_an_unplaced_sequence_is_refused_with_its_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        media = _media(
            [ds.base_dir / "a.mp4"],
            placement=UnplacedSequence("it follows a sequence of unknown length"),
        )
        _resolves(monkeypatch, media)

        with pytest.raises(ValueError, match=ENTRY_TEXT) as raised:
            _ = sequence_video(ds, "g", "s1", _counting(5))

        assert "it follows a sequence of unknown length" in str(raised.value)
        assert isinstance(raised.value.__cause__, MediaProbeError)

    def test_a_count_that_disagrees_with_the_stored_count_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        _resolves(monkeypatch, _media([ds.base_dir / "a.mp4"], frame_counts=(300,)))

        with pytest.raises(ValueError, match=ENTRY_TEXT) as raised:
            _ = sequence_video(ds, "g", "s1", _counting(299))

        assert "299" in str(raised.value)
        assert "300" in str(raised.value)

    def test_a_stored_count_of_none_skips_the_comparison(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(tmp_path)
        media = _media([ds.base_dir / "a.mp4"], frame_counts=(0,))
        assert media.frame_count is None
        _resolves(monkeypatch, media)

        video = sequence_video(ds, "g", "s1", _counting(7))

        assert video.frames == tuple(range(7))

    def test_a_missing_file_is_refused_by_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FERAL's own frame count raises ``FileNotFoundError``, which names no entry."""
        ds = _dataset(tmp_path)
        _resolves(monkeypatch, _media([tmp_path / "gone.mp4"]))

        def frame_count(path: str) -> int | None:
            raise FileNotFoundError(f"Video not found: {path}")

        with pytest.raises(ValueError, match=ENTRY_TEXT) as raised:
            _ = sequence_video(ds, "g", "s1", frame_count)

        assert "missing" in str(raised.value)
        assert str(tmp_path / "gone.mp4") in str(raised.value)

    @pytest.mark.parametrize("stored", [0, 5])
    def test_an_uncountable_video_is_refused_as_unreadable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stored: int
    ) -> None:
        ds = _dataset(tmp_path)
        _resolves(monkeypatch, _media([ds.base_dir / "a.mp4"], frame_counts=(stored,)))

        with pytest.raises(ValueError, match=ENTRY_TEXT) as raised:
            _ = sequence_video(ds, "g", "s1", _counting(None))

        assert "unreadable" in str(raised.value)


def _params(
    training: Mapping[str, object] | None = None, **overrides: object
) -> FeralFeature.Params:
    """Training params of a feature that fits on a templates run."""
    return FeralFeature.Params.model_validate(
        {
            "templates": _EXTRACT_LABELED_TEMPLATES,
            "labels": {},
            "training": {} if training is None else dict(training),
            **overrides,
        }
    )


class TestFeralConfig:
    """`feral_config` is the whole nested config FERAL reads, built from params."""

    def test_each_param_sits_under_its_nested_key_beside_the_fixed_values(
        self, tmp_path: Path
    ) -> None:
        params = _params(
            {
                "epochs": 3,
                "train_bs": 2,
                "val_bs": 5,
                "num_workers": 1,
                "lr": 1e-3,
                "weight_decay": 0.2,
                "label_smoothing": 0.05,
                "fc_drop_rate": 0.3,
                "freeze_encoder_layers": 3,
                "class_weights": "inv_freq",
                "ema_decay": 0.9,
                "mixup_alpha": 0.6,
                "part_warmup": 0.1,
                "patience": 4,
                "compile": False,
                "do_aa": False,
                "seed": 7,
                "part_sample": 0.5,
                "multilabel_threshold": 0.7,
            },
            predict_per_item=16,
            chunk_length=12,
            chunk_shift=5,
            chunk_step=2,
            resize_to=48,
            device="cpu",
        )

        config = feral_config(
            params, backbone="vjepa2_vitl_diving48", dataset_root=tmp_path
        )

        assert config == {
            "run_name": "mosaic",
            "backbone": "vjepa2_vitl_diving48",
            "predict_per_item": 16,
            "model": {
                "fc_drop_rate": 0.3,
                "class_weights": "inv_freq",
                "freeze_encoder_layers": 3,
                "gradient_checkpointing": False,
                "max_class_weight": None,
            },
            "data": {
                "prefix": str(tmp_path),
                "chunk_length": 12,
                "chunk_shift": 5,
                "chunk_step": 2,
                "eval_chunk_shift": None,
                "eval_smoothing_window": None,
                "resize_to": 48,
                "resize_style": "square",
                "do_aa": False,
                "part_sample": 0.5,
                "subsample_keep_rare_threshold": None,
            },
            "training": {
                "epochs": 3,
                "train_bs": 2,
                "val_bs": 5,
                "num_workers": 1,
                "part_warmup": 0.1,
                "lr": 1e-3,
                "weight_decay": 0.2,
                "label_smoothing": 0.05,
                "compile": False,
                "patience": 4,
                "grad_clip_norm": None,
                "log_grad_norm": False,
                "heavy_log_every": None,
            },
            "mixup_alpha": 0.6,
            "ema_decay": 0.9,
            "seed": 7,
            "device": "cpu",
            "starting_checkpoint": None,
            "max_batches": None,
            "multilabel_threshold": 0.7,
        }

    def test_the_dataset_root_is_the_prefix_as_text(self, tmp_path: Path) -> None:
        config = feral_config(_params(), backbone="b", dataset_root=tmp_path)

        assert config["data"]["prefix"] == str(tmp_path)

    def test_params_without_training_are_refused(self, tmp_path: Path) -> None:
        params = FeralFeature.Params.model_validate({"model_dir": "models/x"})

        with pytest.raises(ValueError, match="training"):
            _ = feral_config(params, backbone="b", dataset_root=tmp_path)


def _interaction_stream(
    entries: Sequence[str], frame_counts: Mapping[str, int] | None = None
) -> InputStream:
    """The crop rows of each entry, as ``fit`` streams them.

    Each entry holds one clip of four frames, or of the frames *frame_counts* names.
    """
    counts = frame_counts or {}
    tables = [
        (
            f"g/{entry}",
            _interaction_rows(
                (f"{entry}.mp4", 0, counts.get(entry, 4) - 1, counts.get(entry, 4), 0)
            ).assign(sequence=entry),
        )
        for entry in entries
    ]
    return InputStream(lambda: iter(tables), len(tables))


class _WeightlessModel:
    """A trained model that has no weights, which ``save_state`` can still write."""

    def state_dict(self) -> dict[str, object]:
        return {}


class _FitCalls:
    """What ``fit`` handed to FERAL, recorded by stand-ins for its pieces."""

    def __init__(self) -> None:
        self.validated: list[tuple[dict[str, object], Path]] = []
        self.trained: list[tuple[FeralConfig, FeralLabelsJson, Path]] = []
        self.class_values: list[tuple[int, ...]] = []
        self.evaluated: list[str] = []


def _stand_in_feral(monkeypatch: pytest.MonkeyPatch) -> _FitCalls:
    """Replace each FERAL-facing function ``fit`` calls, so it imports no FERAL."""
    calls = _FitCalls()

    def validate(labels_json: FeralLabelsJson, dataset_root: Path) -> None:
        calls.validated.append((dict(labels_json), dataset_root))

    def train(
        cfg: FeralConfig,
        labels_json: FeralLabelsJson,
        *,
        class_values: Sequence[int],
        run_root: Path,
        progress_callback: object,
    ) -> tuple[_WeightlessModel, float, list[dict[str, float]]]:
        calls.trained.append((cfg, labels_json, run_root))
        calls.class_values.append(tuple(class_values))
        return _WeightlessModel(), 0.5, [{"epoch": 0, "train_loss": 1.0}]

    def evaluate(
        model: object,
        cfg: FeralConfig,
        labels_json: FeralLabelsJson,
        partition: str,
        device: object,
    ) -> dict[str, float]:
        calls.evaluated.append(partition)
        return {f"{partition}_frame_level_map": 0.75}

    monkeypatch.setattr(feral_feature, "_check_feral", lambda: None)

    def backbone(value: str) -> str:
        return "backbone"

    monkeypatch.setattr(feral_feature, "_resolve_backbone_key", backbone)
    monkeypatch.setattr(feral_feature, "_validate_feral_labels", validate)
    monkeypatch.setattr(feral_feature, "train_feral", train)
    monkeypatch.setattr(feral_feature, "_evaluate_partition", evaluate)
    return calls


class _Fit:
    """A feature loaded for a fit over crop inputs, and what its entries hold."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        train: Sequence[str],
        test: Sequence[str],
        labeled: Sequence[str],
        val_fraction: float = 0.5,
        bind: bool = True,
        tables: Mapping[str, int] | None = None,
        events: Mapping[str, Sequence[Event]] | None = None,
        **params: object,
    ) -> None:
        self.tmp_path = tmp_path
        self.ds = _dataset(tmp_path)
        self.crop_root = (
            tmp_path
            / "features"
            / "interaction-crop-pipeline__from__tracks"
            / "0.1-abc"
        )
        self.crop_root.mkdir(parents=True)
        _ = (self.crop_root / "params.json").write_text(_params_json({}))
        self.run_root = tmp_path / "run"
        self.train = tuple(train)
        label_files = _entry_files(
            tmp_path / "labels",
            {
                ("g", entry): (events or {}).get(entry, _pair_events(2, 5, count=2))
                for entry in labeled
            },
        )
        self.feature = FeralFeature(
            FeralFeature.Inputs((_CROP_RESULT,)),
            {
                **_TRAINING_SOURCE,
                "training": {"val_fraction": val_fraction, "seed": 3},
                "chunk_length": 2,
                "chunk_shift": 1,
                "predict_per_item": 2,
                "device": "cpu",
                **params,
            },
        )
        if bind:
            self.feature.bind_dataset(self.ds)
        templates = _templates_run(tmp_path / "templates", train, test)
        crop_tables = {("g", "a"): self.crop_root / "table.parquet"}
        for entry, row_count in (tables or {}).items():
            crop_tables[("g", entry)] = self._write_table(entry, row_count)
        ready = self.feature.load_state(
            self.run_root,
            {"templates": templates},
            {"labels": label_files, "_input_0": crop_tables},
        )
        assert ready is False

    def _write_table(self, entry: str, row_count: int) -> Path:
        """The crop table of *entry* in the crop run, with *row_count* rows."""
        path = self.crop_root / f"{entry}.parquet"
        clips = [(f"{entry}.mp4", 0, 3, 4, index) for index in range(max(row_count, 1))]
        _interaction_rows(*clips).iloc[:row_count].to_parquet(path)
        return path

    def filename(self, entry: str) -> str:
        return f"features/interaction-crop-pipeline__from__tracks/0.1-abc/{entry}.mp4"

    def drawn(self, fraction: float = 0.5) -> list[str]:
        """The train entries the validation draw takes, by sequence."""
        return [
            sequence
            for _, sequence in validation_entries(
                [("g", entry) for entry in sorted(self.train)], fraction, 3
            )
        ]


class TestFit:
    """``fit`` reads each entry's videos, splits and labels them, and trains."""

    def test_the_labels_split_the_entries_into_train_val_and_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b", "c", "d"], test=["e"], labeled="abcde")
        drawn = fit.drawn()

        fit.feature.fit(_interaction_stream(["e", "d", "c", "b", "a", "z"]))

        ((cfg, labels_json, run_root),) = calls.trained
        assert run_root == fit.run_root
        assert labels_json["splits"] == {
            "train": [fit.filename(e) for e in "abcd" if e not in drawn],
            "val": [fit.filename(e) for e in "abcd" if e in drawn],
            "test": [fit.filename("e")],
        }
        assert len(drawn) == 2
        assert labels_json["class_names"] == {"0": "walk", "1": "run"}
        assert labels_json["labels"][fit.filename("a")] == [0, 0, 1, 1]
        assert cfg["data"]["prefix"] == str(tmp_path)
        assert cfg["backbone"] == "backbone"

    def test_the_class_values_of_the_labels_dict_reach_the_training_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=[], labeled="ab", val_fraction=0)

        fit.feature.fit(_interaction_stream(["a", "b"]))

        assert calls.class_values == [(2, 5)]

    def test_the_labels_dict_and_the_config_are_written_to_the_run_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=["e"], labeled="abe")

        fit.feature.fit(_interaction_stream(["a", "b", "e"]))

        ((cfg, labels_json, _),) = calls.trained
        assert (
            json.loads((fit.run_root / "feral_labels.json").read_text()) == labels_json
        )
        assert json.loads((fit.run_root / "config.json").read_text()) == cfg

    def test_feral_validates_the_labels_against_the_dataset_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=["e"], labeled="abe")

        fit.feature.fit(_interaction_stream(["a", "b", "e"]))

        ((validated, root),) = calls.validated
        ((_, labels_json, _),) = calls.trained
        assert validated == dict(labels_json)
        assert root == tmp_path

    def test_the_test_split_is_evaluated_and_the_classes_are_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=["e"], labeled="abe")

        fit.feature.fit(_interaction_stream(["a", "b", "e"]))

        assert calls.evaluated == ["test"]
        fit.feature.save_state(fit.run_root)
        saved = json.loads((fit.run_root / "feral_config.json").read_text())
        assert saved["class_values"] == [2, 5]
        assert saved["class_names"] == {"2": "walk", "5": "run"}
        assert saved["is_multilabel"] is False
        reports = json.loads((fit.run_root / "reports.json").read_text())
        assert reports["test_metrics"] == {"test_frame_level_map": 0.75}
        assert reports["best_val_map"] == 0.5

    def test_the_class_names_param_renames_the_classes_a_fit_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=["e"],
            labeled="abe",
            class_names={"2": "stroll"},
        )

        fit.feature.fit(_interaction_stream(["a", "b", "e"]))

        fit.feature.save_state(fit.run_root)
        saved = json.loads((fit.run_root / "feral_config.json").read_text())
        assert saved["class_names"] == {"2": "stroll", "5": "run"}
        reports = json.loads((fit.run_root / "reports.json").read_text())
        assert reports["class_names"] == {"2": "stroll", "5": "run"}

    def test_a_class_name_for_a_value_no_train_video_holds_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=[],
            labeled="ab",
            class_names={"7": "x"},
        )

        with pytest.raises(ValueError, match="class_names names the values"):
            fit.feature.fit(_interaction_stream(["a", "b"]))

        assert calls.trained == []

    def test_a_templates_run_without_test_entries_skips_the_test_evaluation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=[], labeled="ab")

        fit.feature.fit(_interaction_stream(["a", "b"]))

        assert len(calls.trained) == 1
        assert calls.evaluated == []

    def test_an_entry_the_templates_did_not_assign_is_left_out_though_it_is_labeled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=["e"], labeled="abez")

        fit.feature.fit(_interaction_stream(["a", "b", "e", "z"]))

        ((_, labels_json, _),) = calls.trained
        assert fit.filename("z") not in labels_json["labels"]
        assert [
            fit.filename("z") in names for names in labels_json["splits"].values()
        ] == [False] * 3
        assert labels_json["splits"]["test"] == [fit.filename("e")]

    def test_the_draw_is_of_the_templates_train_entries_not_the_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Train entry ``e`` is labeled by no file and streamed by no scope."""
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a", "b", "c", "d", "e"],
            test=[],
            labeled="abcd",
            val_fraction=0.4,
        )
        whole_draw = fit.drawn(0.4)
        scope_draw = _Fit(
            tmp_path / "scope", train=list("abcd"), test=[], labeled="abcd"
        ).drawn(0.4)
        assert [name for name in whole_draw if name != "e"] != scope_draw

        fit.feature.fit(_interaction_stream(["d", "c", "b", "a"]))

        ((_, labels_json, _),) = calls.trained
        val = [
            name.rsplit("/", 1)[1].removesuffix(".mp4")
            for name in labels_json["splits"]["val"]
        ]
        assert val == [name for name in whole_draw if name != "e"]

    def test_a_scope_missing_a_train_entry_is_refused_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=[], labeled="ab")

        with pytest.raises(
            ValueError, match=r"train entries of the templates run"
        ) as raised:
            fit.feature.fit(_interaction_stream(["a"]))

        assert "g/b" in str(raised.value)
        assert calls.trained == []

    def test_a_scope_missing_a_labeled_test_entry_is_refused_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=["e"], labeled="abe")

        with pytest.raises(
            ValueError, match=r"test entries of the templates run"
        ) as raised:
            fit.feature.fit(_interaction_stream(["a", "b"]))

        assert "g/e" in str(raised.value)
        assert calls.trained == []

    def test_a_validation_fraction_that_leaves_no_train_entry_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a"], test=[], labeled="a", val_fraction=0.9)

        with pytest.raises(ValueError, match="leaves no train entry"):
            fit.feature.fit(_interaction_stream(["a"]))

        assert calls.trained == []

    def test_a_val_split_that_loses_every_video_trains_without_validation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        entries = "abcd"
        drawn = _Fit(
            tmp_path / "probe", train=list(entries), test=[], labeled=""
        ).drawn()
        labeled = [entry for entry in entries if entry not in drawn]
        fit = _Fit(tmp_path, train=list(entries), test=[], labeled=labeled)

        with caplog.at_level(logging.WARNING):
            fit.feature.fit(_interaction_stream(list(entries)))

        ((_, labels_json, _),) = calls.trained
        assert labels_json["splits"]["val"] == []
        assert len(labels_json["splits"]["train"]) == 2
        assert "none of the 2 validation video(s)" in caplog.text
        assert "2 no label file" in caplog.text

    def test_a_validation_fraction_of_zero_trains_without_a_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=[], labeled="ab", val_fraction=0)

        with caplog.at_level(logging.WARNING):
            fit.feature.fit(_interaction_stream(["a", "b"]))

        ((_, labels_json, _),) = calls.trained
        assert labels_json["splits"]["val"] == []
        assert "validation" not in caplog.text

    def test_a_test_split_that_loses_every_video_is_refused_with_the_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=["e", "f"], labeled="ab")

        with pytest.raises(ValueError, match=r"none of the 2 test entries") as raised:
            fit.feature.fit(_interaction_stream(["a", "b", "e", "f"]))

        assert "2 test videos" in str(raised.value)
        assert "2 no label file" in str(raised.value)
        assert calls.trained == []
        assert calls.validated == []

    def test_a_video_shorter_than_the_chunk_span_leaves_every_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        # Two chunk frames three apart span (2 - 1) * 3 + 1 = 4 frames.
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=[],
            labeled="ab",
            val_fraction=0,
            chunk_length=2,
            chunk_step=3,
        )

        fit.feature.fit(_interaction_stream(["a", "b"], {"a": 3, "b": 4}))

        ((_, labels_json, _),) = calls.trained
        assert labels_json["splits"] == {
            "train": [fit.filename("b")],
            "val": [],
            "test": [],
        }
        assert fit.filename("a") not in labels_json["labels"]

    def test_unlabeled_as_labels_the_frames_no_event_covers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        partial: list[Event] = [(0, 0, 1, 2), (1, 0, 1, 5)]
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=[],
            labeled="ab",
            val_fraction=0,
            events={"b": partial},
            unlabeled_as=2,
        )

        fit.feature.fit(_interaction_stream(["a", "b"]))

        ((_, labels_json, _),) = calls.trained
        assert labels_json["splits"]["train"] == [fit.filename("a"), fit.filename("b")]
        assert labels_json["labels"][fit.filename("b")] == [0, 1, 0, 0]

    def test_a_train_entry_with_an_empty_crop_table_trains_with_no_video(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=["e"],
            labeled="abe",
            val_fraction=0,
            tables={"b": 0},
        )

        with caplog.at_level(logging.WARNING):
            fit.feature.fit(_interaction_stream(["a", "e"]))

        ((_, labels_json, _),) = calls.trained
        assert labels_json["splits"]["train"] == [fit.filename("a")]
        assert "1 entry(ies) have no crop video" in caplog.text

    def test_every_train_entry_with_an_empty_crop_table_names_that_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a"],
            test=[],
            labeled="a",
            val_fraction=0,
            tables={"a": 0},
        )

        with pytest.raises(ValueError, match="no training video is left") as raised:
            fit.feature.fit(_interaction_stream([]))

        assert "1 no crop video" in str(raised.value)
        assert "scope" not in str(raised.value)
        assert calls.trained == []

    def test_a_test_entry_with_an_empty_crop_table_is_evaluated_without_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=["e", "f"],
            labeled="abef",
            val_fraction=0,
            tables={"f": 0},
        )

        fit.feature.fit(_interaction_stream(["a", "b", "e"]))

        ((_, labels_json, _),) = calls.trained
        assert labels_json["splits"]["test"] == [fit.filename("e")]
        assert calls.evaluated == ["test"]

    def test_every_test_entry_with_an_empty_crop_table_names_that_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=["e"],
            labeled="abe",
            val_fraction=0,
            tables={"e": 0},
        )

        with pytest.raises(ValueError, match=r"none of the 1 test entries") as raised:
            fit.feature.fit(_interaction_stream(["a", "b"]))

        assert "1 no crop video" in str(raised.value)
        assert "scope" not in str(raised.value)
        assert calls.trained == []

    def test_an_entry_with_crop_rows_that_the_scope_lacks_is_still_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _stand_in_feral(monkeypatch)
        fit = _Fit(
            tmp_path,
            train=["a", "b"],
            test=[],
            labeled="ab",
            val_fraction=0,
            tables={"b": 2},
        )

        with pytest.raises(ValueError, match="scope"):
            fit.feature.fit(_interaction_stream(["a"]))

    def test_fit_needs_a_bound_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _stand_in_feral(monkeypatch)
        fit = _Fit(tmp_path, train=["a", "b"], test=[], labeled="ab", bind=False)

        with pytest.raises(RuntimeError, match="bind_dataset"):
            fit.feature.fit(_interaction_stream(["a", "b"]))

    def test_fit_without_loaded_templates_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _stand_in_feral(monkeypatch)
        feature = _feature(_TRAINING_SOURCE)
        feature.bind_dataset(_dataset(tmp_path))

        with pytest.raises(RuntimeError, match="no templates were loaded"):
            feature.fit(InputStream(lambda: iter([]), 0))


class TestCheckpointClasses:
    """The classes of a checkpoint trained outside mosaic."""

    _META: Mapping[str, object] = {
        "class_names": {"0": "rest", "1": "walk"},
        "is_multilabel": True,
    }
    _LEGACY: Mapping[str, object] = {
        "class_names": {"0": "old rest", "1": "old walk"},
        "is_multilabel": False,
    }

    def test_the_param_supplies_the_classes(self) -> None:
        names, multilabel = checkpoint_classes({"2": "walk", "5": "run"}, None, None)

        assert names == {2: "walk", 5: "run"}
        assert multilabel is False

    def test_an_older_config_supplies_the_classes_and_the_flag_without_the_param(
        self,
    ) -> None:
        names, multilabel = checkpoint_classes(None, None, self._LEGACY)

        assert names == {0: "old rest", 1: "old walk"}
        assert multilabel is False

    def test_an_older_config_that_is_multilabel_says_so(self) -> None:
        legacy = {**self._LEGACY, "is_multilabel": True}

        assert checkpoint_classes(None, None, legacy)[1] is True

    def test_the_checkpoint_metadata_wins_over_the_config(self) -> None:
        names, multilabel = checkpoint_classes(None, self._META, self._LEGACY)

        assert names == {0: "rest", 1: "walk"}
        assert multilabel is True

    def test_recorded_class_values_put_the_dense_names_under_their_label_values(
        self,
    ) -> None:
        meta = {**self._META, "class_values": [2, 5]}

        names, _ = checkpoint_classes(None, meta, self._LEGACY)

        assert names == {2: "rest", 5: "walk"}

    def test_class_values_that_do_not_match_the_names_are_refused(self) -> None:
        meta = {**self._META, "class_values": [2, 5, 7]}

        with pytest.raises(ValueError, match="class_values"):
            _ = checkpoint_classes(None, meta, None)

    def test_the_param_names_the_label_values_whatever_the_recorded_class_values(
        self,
    ) -> None:
        meta = {**self._META, "class_values": [2, 5]}

        names, _ = checkpoint_classes({"5": "mine"}, meta, None)

        assert names == {5: "mine"}

    def test_the_param_wins_over_both_for_the_names(self) -> None:
        names, multilabel = checkpoint_classes({"0": "mine"}, self._META, self._LEGACY)

        assert names == {0: "mine"}
        assert multilabel is True

    def test_a_checkpoint_naming_no_classes_is_refused_by_the_param(self) -> None:
        with pytest.raises(ValueError, match="class_names param"):
            _ = checkpoint_classes(None, {"cfg": {}}, {"model_name": "x"})

    def test_a_checkpoint_recording_no_flag_is_read_as_single_label_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _, multilabel = checkpoint_classes({"0": "a"}, {"cfg": {}}, None)

        assert multilabel is False
        assert "single-label" in caplog.text


_NESTED_NETWORK = {
    "backbone": "vjepa2_vitl_diving48",
    "predict_per_item": 16,
    "model": {"fc_drop_rate": 0.3, "freeze_encoder_layers": 4, "class_weights": "x"},
    "data": {"chunk_length": 8},
}


class TestImportedNetwork:
    """The network of an imported checkpoint comes from the first config that records it."""

    def test_a_nested_config_is_taken_as_it_is(self) -> None:
        network = imported_network(
            [_NESTED_NETWORK, None], model_name="param", predict_per_item=64
        )

        assert network["backbone"] == "vjepa2_vitl_diving48"
        assert network["predict_per_item"] == 16
        assert network["model"]["fc_drop_rate"] == 0.3
        assert network["model"]["freeze_encoder_layers"] == 4

    def test_the_config_json_wins_over_the_checkpoints_own_config(self) -> None:
        other = {**_NESTED_NETWORK, "predict_per_item": 8}

        network = imported_network(
            [_NESTED_NETWORK, other], model_name="param", predict_per_item=64
        )

        assert network["predict_per_item"] == 16

    def test_the_checkpoints_nested_config_is_read_without_a_config_json(self) -> None:
        network = imported_network(
            [None, _NESTED_NETWORK], model_name="param", predict_per_item=64
        )

        assert network["backbone"] == "vjepa2_vitl_diving48"

    def test_a_flat_config_of_an_older_mosaic_run_becomes_a_nested_one(self) -> None:
        flat = {
            "model_name": "facebook/vjepa2-vitl-fpc32-256-diving48",
            "predict_per_item": 32,
            "fc_drop_rate": 0.25,
            "freeze_encoder_layers": 6,
            "epochs": 10,
        }

        network = imported_network(
            [flat, None], model_name="param", predict_per_item=64
        )

        assert network == {
            "backbone": "facebook/vjepa2-vitl-fpc32-256-diving48",
            "predict_per_item": 32,
            "model": {"fc_drop_rate": 0.25, "freeze_encoder_layers": 6},
        }

    def test_a_nested_config_beats_a_flat_one(self) -> None:
        flat = {"model_name": "flat", "predict_per_item": 32}

        network = imported_network(
            [flat, _NESTED_NETWORK], model_name="param", predict_per_item=64
        )

        assert network["backbone"] == "vjepa2_vitl_diving48"

    def test_a_checkpoint_recording_nothing_takes_the_steps_params(self) -> None:
        network = imported_network(
            [None, None], model_name="param", predict_per_item=64
        )

        assert network["backbone"] == "param"
        assert network["predict_per_item"] == 64
        defaults = FeralTrainingConfig()
        assert network["model"] == {
            "fc_drop_rate": defaults.fc_drop_rate,
            "freeze_encoder_layers": 0,
        }

    def test_a_flat_config_without_a_freeze_freezes_no_layer(self) -> None:
        flat = {"model_name": "vjepa2_vitb_384", "predict_per_item": 32}

        network = imported_network(
            [flat, None], model_name="param", predict_per_item=64
        )

        assert network["model"]["freeze_encoder_layers"] == 0


# --- FERAL's own pieces ------------------------------------------------------

_BACKBONE = "vjepa2_vitl_diving48"


def _feral_module(name: str) -> ModuleType:
    """The FERAL submodule *name*, imported when a test needs it."""
    return importlib.import_module(f"feral.{name}")


def _write_video(path: Path, frames: int = 16, size: int = 48) -> None:
    """Write *frames* noise frames of *size* pixels to the mp4 at *path*."""
    import cv2

    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (size, size)
    )
    for _ in range(frames):
        _ = writer.write(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    writer.release()


def _written_labels(
    tmp_path: Path,
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str] = (),
) -> FeralLabels:
    """Sequence videos of 16 frames under *tmp_path* and the labels dict they get."""
    entries = {name: ("g", name) for name in (*train, *val, *test)}
    files = _entry_files(
        tmp_path, {entry: _pair_events(2, 5, count=8) for entry in entries.values()}
    )
    videos: dict[str, FeralVideo] = {}
    for name, entry in entries.items():
        videos[name] = _video(tmp_path, entry, range(16), {})
        _write_video(videos[name].path)
    return build_feral_labels(
        {
            "train": [videos[name] for name in train],
            "val": [videos[name] for name in val],
            "test": [videos[name] for name in test],
        },
        files,
        GroundTruthLabelsSource(),
        unlabeled_as=None,
        min_frames=1,
    )


def _synthetic_answers(labels_json: FeralLabelsJson, split: str) -> list[FeralAnswer]:
    """Chunk answers of 4 frames, shifted by 2, over each video of *split*."""
    rng = np.random.default_rng(0)
    answers: list[FeralAnswer] = []
    for filename in labels_json["splits"][split]:
        video_labels = labels_json["labels"][filename]
        for start in range(0, len(video_labels) - 3, 2):
            for position in range(4):
                frame = start + position
                logits = rng.normal(size=2)
                probs = np.exp(logits) / np.exp(logits).sum()
                target = np.eye(2)[video_labels[frame]]
                answers.append(
                    ((filename, frame, position), probs.tolist(), target.tolist())
                )
    return answers


class TestComputeMetricsAreFloats:
    """A metric is a Python float, which a resume checkpoint can hold."""

    def test_numpy_scalars_from_feral_come_back_as_floats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def ensembled(
            answers: Sequence[FeralAnswer], labels_json: Mapping[str, object], part: str
        ) -> dict[str, object]:
            return {}

        def multiclass(
            answers: Sequence[FeralAnswer], names: Mapping[int, str], prefix: str = ""
        ) -> dict[str, np.float64]:
            return {f"{prefix}_accuracy": np.float64(0.5)}

        def frame_map(
            predictions: object, labels_json: Mapping[str, object], partition: str
        ) -> np.float64:
            return np.float64(0.25)

        def f1(
            predictions: object,
            labels_json: Mapping[str, object],
            partition: str,
            is_multilabel: bool,
            prefix: str,
            threshold: float,
        ) -> dict[str, np.float64]:
            return {f"{prefix}/f1_walk": np.float64(0.75)}

        monkeypatch.setattr(feral_feature, "ensembled_predictions", ensembled)
        monkeypatch.setattr(
            feral_feature, "_feral_calculate_multiclass_metrics", lambda: multiclass
        )
        monkeypatch.setattr(
            feral_feature, "_feral_calc_frame_level_map", lambda: frame_map
        )
        monkeypatch.setattr(feral_feature, "_feral_calculate_f1_metrics", lambda: f1)
        answers: list[FeralAnswer] = [(("a.mp4", 0, 0), [0.5, 0.5], [1.0, 0.0])]

        measured = FeralFeature.compute_metrics(
            answers, {"class_names": {"0": "walk", "1": "run"}}, "val"
        )

        assert measured == {
            "val_accuracy": 0.5,
            "val_frame_level_map": 0.25,
            "val/f1_walk": 0.75,
        }
        assert {type(value) for value in measured.values()} == {float}


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestComputeMetrics:
    """`compute_metrics` ensembles the chunk answers before it measures."""

    _LABELS: FeralLabelsJson = {
        "class_names": {"0": "walk", "1": "run"},
        "is_multilabel": False,
        "labels": {
            "a.mp4": [0, 0, 0, 1, 1, 0, 1, 1],
            "b.mp4": [1, 1, 0, 0, 0, 1, 1, 0],
        },
        "splits": {"train": [], "val": [], "test": ["a.mp4", "b.mp4"]},
    }

    def test_the_metrics_equal_FERALs_on_the_ensembled_predictions(self) -> None:
        metrics = _feral_module("metrics")
        answers = _synthetic_answers(self._LABELS, "test")
        ensembled = metrics.postprocess_predictions(
            metrics.ensemble_predictions(
                answers, metrics.generate_empty_logits(self._LABELS, "test")
            )
        )
        expected = {
            **metrics.calculate_multiclass_metrics(
                answers, {0: "walk", 1: "run"}, "test"
            ),
            "test_frame_level_map": metrics.calc_frame_level_map(
                ensembled, self._LABELS, "test"
            ),
            **metrics.calculate_f1_metrics(
                ensembled, self._LABELS, "test", False, "test", 0.85
            ),
        }

        measured = FeralFeature.compute_metrics(answers, dict(self._LABELS), "test")

        assert measured == expected

    def test_the_prefix_names_every_key(self) -> None:
        answers = _synthetic_answers(self._LABELS, "test")

        measured = FeralFeature.compute_metrics(
            answers, dict(self._LABELS), "test", prefix="held_out"
        )

        assert "held_out_frame_level_map" in measured
        assert all(key.startswith("held_out") for key in measured)

    def test_no_answers_give_no_metrics(self) -> None:
        assert FeralFeature.compute_metrics([], dict(self._LABELS), "test") == {}


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestFeralAcceptsMosaicsInputs:
    """FERAL's own checks take what mosaic builds."""

    def test_validate_labels_json_accepts_a_built_labels_dict(
        self, tmp_path: Path
    ) -> None:
        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])

        _feral_module("utils").validate_labels_json(dict(built.labels_json), tmp_path)

    def test_every_key_of_the_default_config_is_a_key_of_the_built_config(
        self, tmp_path: Path
    ) -> None:
        config = feral_config(_params(), backbone=_BACKBONE, dataset_root=tmp_path)

        default = importlib.import_module("feral").load_default_config()

        default_keys = set(pd.json_normalize(default).columns)
        built_keys = set(pd.json_normalize(dict(config)).columns)
        assert default_keys <= built_keys


def _tiny_module(num_classes: int, rows_per_chunk: int) -> torch.nn.Module:
    """A trainable stand-in for FERAL's model: a linear map of pooled pixels."""
    import torch

    class TinyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pooled = x.mean(dim=(3, 4))
            assert pooled.shape[1] == rows_per_chunk
            return self.linear(pooled.reshape(-1, 3))

    return TinyModule()


def _linear_weight(module: torch.nn.Module) -> torch.Tensor:
    """A copy of the tiny module's weight, as it is now."""
    return module.state_dict()["linear.weight"].clone()


_SMOKE_TRAINING: dict[str, object] = {
    "epochs": 1,
    "ema_decay": None,
    "train_bs": 1,
    "val_bs": 1,
    "num_workers": 0,
    "compile": False,
    "seed": 123,
}
"""Training settings that keep a CPU run to one epoch of whole chunks."""


def _smoke_config(tmp_path: Path, **training: object) -> FeralConfig:
    params = _params(
        {**_SMOKE_TRAINING, **training},
        device="cpu",
        chunk_length=8,
        chunk_shift=4,
        chunk_step=1,
        predict_per_item=8,
        resize_to=32,
    )
    return feral_config(params, backbone=_BACKBONE, dataset_root=tmp_path)


def _epochs_config(tmp_path: Path, **training: object) -> FeralConfig:
    """The smoke config with a learning rate that moves the tiny weights."""
    return _smoke_config(tmp_path, lr=1e-2, mixup_alpha=None, **training)


class _RecordedEpochs(NullProgressCallback):
    """A progress callback that keeps what each epoch reported."""

    def __init__(self) -> None:
        self.epochs: list[tuple[int, int, dict[str, float]]] = []

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        self.epochs.append((epoch, total_epochs, dict(metrics)))


class _WeightsAtEpochEnd(_RecordedEpochs):
    """A recorder that also copies the first built module's weight each epoch."""

    def __init__(self, built: list[torch.nn.Module]) -> None:
        super().__init__()
        self._built = built
        self.weights: list[torch.Tensor] = []

    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        super().on_epoch_end(epoch, total_epochs, metrics)
        self.weights.append(_linear_weight(self._built[0]))


@pytest.fixture(scope="module", autouse=True)
def no_gpu() -> Iterator[None]:
    """Hide every GPU from the file's tests, so none of them initializes CUDA.

    FERAL's epoch functions open a CUDA autocast, and ``evaluate`` asks torch
    whether CUDA is available. Either asks the driver about the GPU when there
    is one, and the driver reads the device list once per process. On a machine
    whose GPU the installed torch has no kernels for, the first such query
    raises, in whichever test of the worker makes it first. The variable is set
    before the file's first test, so no query precedes it.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield


@pytest.fixture
def tiny_model(monkeypatch: pytest.MonkeyPatch) -> list[torch.nn.Module]:
    """Replace FERAL's ``build_model`` with one that builds a tiny module.

    The fixture's value lists the modules built, in order.
    """
    built: list[torch.nn.Module] = []

    def build_model(
        cfg: FeralConfig,
        num_classes: int,
        device: str | torch.device,
        *,
        with_ema: bool = True,
    ) -> tuple[torch.nn.Module, None]:
        module = _tiny_module(num_classes, cfg["predict_per_item"]).to(device)
        built.append(module)
        return module, None

    monkeypatch.setattr(_feral_module("modeling"), "build_model", build_model)
    return built


class _Ema(Protocol):
    """The exponential-moving-average wrapper timm builds around a model."""

    ema: torch.nn.Module


class _TimmModelEma(Protocol):
    def __call__(
        self, model: torch.nn.Module, *, decay: float, device: str
    ) -> _Ema: ...


def _timm_model_ema() -> _TimmModelEma:
    return importlib.import_module("timm.utils.model_ema").ModelEma


def _validation_maps(monkeypatch: pytest.MonkeyPatch, maps: Sequence[float]) -> None:
    """Make each validation pass report the next of *maps* as its mAP."""
    remaining = iter(maps)

    def compute_metrics(
        answers: Sequence[FeralAnswer],
        labels_json: dict[str, object],
        partition: str,
        **_: object,
    ) -> dict[str, float]:
        return {"val_frame_level_map": next(remaining)}

    monkeypatch.setattr(FeralFeature, "compute_metrics", staticmethod(compute_metrics))


# Each filter below is raised by FERAL's loop, loaders or metrics on a CPU run.
_CPU_RUN_WARNINGS = [
    # `train_one_epoch` and `evaluate` open `autocast(device_type="cuda")`, which
    # warns and disables itself when no GPU is present. The message differs
    # between torch releases and always says "Disabling".
    "ignore:.*Disabling:UserWarning",
    # The loaders FERAL builds set `pin_memory=True`, which warns without an
    # accelerator.
    "ignore:'pin_memory' argument is set as true:UserWarning",
    # A one-epoch, one-video validation split can lack a positive sample of a
    # class, and sklearn says so.
    "ignore:No positive samples:UserWarning",
    "ignore:No positive class found:UserWarning",
    # An untrained model after one epoch can predict a single class, which leaves
    # the other class without a predicted sample for FERAL's precision.
    "ignore:Precision is ill-defined:sklearn.exceptions.UndefinedMetricWarning",
]


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
@pytest.mark.filterwarnings(*_CPU_RUN_WARNINGS)
class TestTrainFeral:
    """`train_feral` runs FERAL's loop pieces for the epochs of one config."""

    def test_train_and_val_write_the_best_model_and_the_summary(
        self, tmp_path: Path, tiny_model: list[torch.nn.Module]
    ) -> None:
        import torch

        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])
        run_root = tmp_path / "run"
        run_root.mkdir()
        progress = _RecordedEpochs()

        model, best_map, metrics = train_feral(
            _smoke_config(tmp_path),
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=progress,
        )

        assert isinstance(model, torch.nn.Module)
        assert len(metrics) == 1
        assert metrics[0]["val_frame_level_map"] == best_map
        assert best_map > 0
        assert (run_root / "model_best.pt").is_file()
        assert (run_root / "summary.csv").is_file()
        assert not (run_root / "training_checkpoint.pt").exists()
        assert [(e, n) for e, n, _ in progress.epochs] == [(0, 1)]
        assert progress.epochs[0][2] == metrics[0]

    def test_the_best_model_file_holds_the_returned_weights_and_the_classes(
        self, tmp_path: Path, tiny_model: list[torch.nn.Module]
    ) -> None:
        import torch

        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])
        run_root = tmp_path / "run"
        run_root.mkdir()
        config = _smoke_config(tmp_path)

        model, _, _ = train_feral(
            config,
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=None,
        )

        saved = torch.load(run_root / "model_best.pt", map_location="cpu")
        assert saved["class_names"] == built.labels_json["class_names"]
        assert saved["class_values"] == list(built.class_values)
        assert saved["is_multilabel"] is False
        assert saved["cfg"] == config
        for name, tensor in model.state_dict().items():
            assert torch.equal(saved["state_dict"][name], tensor.cpu())

    def test_train_alone_writes_the_model_and_the_summary_without_validation_metrics(
        self, tmp_path: Path, tiny_model: list[torch.nn.Module]
    ) -> None:
        import torch

        built = _written_labels(tmp_path, train=["a", "b"], val=[])
        run_root = tmp_path / "run"
        run_root.mkdir()

        model, best_map, metrics = train_feral(
            _smoke_config(tmp_path),
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=None,
        )

        assert isinstance(model, torch.nn.Module)
        assert len(metrics) == 1
        assert "val_frame_level_map" not in metrics[0]
        assert best_map == -1.0
        assert (run_root / "model_best.pt").is_file()
        assert (run_root / "summary.csv").is_file()

    def test_the_random_states_are_seeded_from_the_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import random

        import torch

        class Stop(Exception):
            pass

        seen: list[tuple[int, float, float]] = []

        def build_model(*_args: object, **_kwargs: object) -> tuple[None, None]:
            seen.append(
                (torch.initial_seed(), random.random(), float(np.random.random()))
            )
            raise Stop

        monkeypatch.setattr(_feral_module("modeling"), "build_model", build_model)
        built = _written_labels(tmp_path, train=["a", "b"], val=[])
        run_root = tmp_path / "run"
        run_root.mkdir()
        config = _smoke_config(tmp_path)

        with pytest.raises(Stop):
            _ = train_feral(
                config,
                built.labels_json,
                class_values=built.class_values,
                run_root=run_root,
                progress_callback=None,
            )

        random.seed(123)
        np.random.seed(123)
        assert seen == [(123, random.random(), float(np.random.random()))]


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
@pytest.mark.filterwarnings(*_CPU_RUN_WARNINGS)
class TestTrainFeralEpochs:
    """`train_feral` keeps the best epoch, stops on patience and resumes."""

    def test_the_returned_model_holds_the_best_epochs_weights(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: list[torch.nn.Module],
    ) -> None:
        import torch

        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])
        run_root = tmp_path / "run"
        run_root.mkdir()
        _validation_maps(monkeypatch, [0.9, 0.1, 0.1])
        progress = _WeightsAtEpochEnd(tiny_model)

        model, best_map, metrics = train_feral(
            _epochs_config(tmp_path, epochs=3),
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=progress,
        )

        weights = progress.weights
        assert len(metrics) == 3
        assert best_map == 0.9
        assert not torch.equal(weights[0], weights[2])
        assert torch.equal(_linear_weight(model), weights[0])
        saved = torch.load(run_root / "model_best.pt", map_location="cpu")
        assert torch.equal(saved["state_dict"]["linear.weight"], weights[0])

    def test_train_alone_keeps_the_last_epoch(
        self, tmp_path: Path, tiny_model: list[torch.nn.Module]
    ) -> None:
        import torch

        built = _written_labels(tmp_path, train=["a", "b"], val=[])
        run_root = tmp_path / "run"
        run_root.mkdir()
        progress = _WeightsAtEpochEnd(tiny_model)

        model, _, metrics = train_feral(
            _epochs_config(tmp_path, epochs=2),
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=progress,
        )

        weights = progress.weights
        assert len(metrics) == 2
        assert not torch.equal(weights[0], weights[1])
        assert torch.equal(_linear_weight(model), weights[1])
        saved = torch.load(run_root / "model_best.pt", map_location="cpu")
        assert torch.equal(saved["state_dict"]["linear.weight"], weights[1])

    def test_an_ema_model_that_validates_better_is_the_one_kept(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: list[torch.nn.Module],
    ) -> None:
        import torch

        averaged: list[_Ema] = []
        evaluated: list[tuple[torch.nn.Module, list[FeralAnswer]]] = []
        loops = _feral_module("loops")
        evaluate = loops.evaluate

        def spying_evaluate(
            model: torch.nn.Module, *args: object, **kwargs: object
        ) -> tuple[list[FeralAnswer], float | None]:
            answers, loss = evaluate(model, *args, **kwargs)
            evaluated.append((model, answers))
            return answers, loss

        def build_model(
            cfg: FeralConfig,
            num_classes: int,
            device: str | torch.device,
            *,
            with_ema: bool = True,
        ) -> tuple[torch.nn.Module, _Ema]:
            module = _tiny_module(num_classes, cfg["predict_per_item"]).to(device)
            tiny_model.append(module)
            averaged.append(_timm_model_ema()(module, decay=0.9, device=str(device)))
            return module, averaged[0]

        written: list[str] = []
        atomic_write = feral_feature.atomic_write

        def recording_atomic_write(
            final_path: Path, write_fn: Callable[[Path], object]
        ) -> None:
            written.append(final_path.name)
            atomic_write(final_path, write_fn)

        monkeypatch.setattr(feral_feature, "atomic_write", recording_atomic_write)
        monkeypatch.setattr(loops, "evaluate", spying_evaluate)
        monkeypatch.setattr(_feral_module("modeling"), "build_model", build_model)
        _validation_maps(monkeypatch, [-0.5])
        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])
        run_root = tmp_path / "run"
        run_root.mkdir()
        progress = _WeightsAtEpochEnd(tiny_model)

        model, best_map, metrics = train_feral(
            _epochs_config(tmp_path, epochs=1, ema_decay=0.9),
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=progress,
        )

        ema_answers = [
            answers
            for evaluated_model, answers in evaluated
            if evaluated_model is averaged[0].ema
        ]
        assert len(ema_answers) == 1
        expected = _feral_module("metrics").calc_frame_level_map(
            ensembled_predictions(ema_answers[0], built.labels_json, "val"),
            built.labels_json,
            "val",
        )
        assert expected > 0
        assert metrics[0]["ema_val_frame_level_map"] == expected
        assert best_map == expected
        saved = torch.load(run_root / "model_best.pt", map_location="cpu")
        assert not torch.equal(
            saved["state_dict"]["linear.weight"], progress.weights[0]
        )
        assert torch.equal(_linear_weight(model), saved["state_dict"]["linear.weight"])
        assert {"model_best.pt", "model_ema.pt", "training_checkpoint.pt"} <= set(
            written
        )
        ema_file = torch.load(run_root / "model_ema.pt", map_location="cpu")
        assert torch.equal(
            ema_file["state_dict"]["linear.weight"], _linear_weight(averaged[0].ema)
        )

    def test_patience_stops_after_that_many_epochs_without_improvement(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: list[torch.nn.Module],
    ) -> None:
        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])
        run_root = tmp_path / "run"
        run_root.mkdir()
        _validation_maps(monkeypatch, [0.5] * 5)

        _, best_map, metrics = train_feral(
            _epochs_config(tmp_path, epochs=5, patience=2),
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=None,
        )

        assert best_map == 0.5
        assert [m["epoch"] for m in metrics] == [0, 1, 2]

    def test_an_interrupted_run_resumes_from_its_saved_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tiny_model: list[torch.nn.Module],
    ) -> None:
        import torch

        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])
        run_root = tmp_path / "run"
        run_root.mkdir()
        config = _epochs_config(tmp_path, epochs=2)
        # Epoch 0 validates best. The interrupted epoch 1 and the resumed
        # epoch 1 both validate worse, so only a restored best mAP keeps the
        # epoch 0 model in `model_best.pt`.
        _validation_maps(monkeypatch, [0.9, 0.1, 0.1])

        class Interrupted(Exception):
            pass

        class InterruptAfterSecondEpoch(_WeightsAtEpochEnd):
            def on_epoch_end(
                self, epoch: int, total_epochs: int, metrics: dict[str, float]
            ) -> None:
                super().on_epoch_end(epoch, total_epochs, metrics)
                if epoch == 1:
                    raise Interrupted

        first = InterruptAfterSecondEpoch(tiny_model)
        with pytest.raises(Interrupted):
            _ = train_feral(
                config,
                built.labels_json,
                class_values=built.class_values,
                run_root=run_root,
                progress_callback=first,
            )
        assert (run_root / "training_checkpoint.pt").is_file()

        loops = _feral_module("loops")
        train_one_epoch = loops.train_one_epoch
        entered: list[tuple[torch.Tensor, int]] = []

        def spying_train_one_epoch(
            model: torch.nn.Module,
            loader: object,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            *args: object,
            **kwargs: object,
        ) -> tuple[list[FeralAnswer], float]:
            entered.append(
                (_linear_weight(model), len(optimizer.state_dict()["state"]))
            )
            return train_one_epoch(model, loader, criterion, optimizer, *args, **kwargs)

        monkeypatch.setattr(loops, "train_one_epoch", spying_train_one_epoch)
        resumed = _RecordedEpochs()

        _, best_map, metrics = train_feral(
            config,
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=resumed,
        )

        assert [epoch for epoch, _, _ in resumed.epochs] == [1]
        assert [m["epoch"] for m in metrics] == [0, 1]
        assert len(entered) == 1
        assert torch.equal(entered[0][0], first.weights[0])
        assert entered[0][1] > 0
        assert best_map == 0.9
        saved = torch.load(run_root / "model_best.pt", map_location="cpu")
        assert torch.equal(saved["state_dict"]["linear.weight"], first.weights[0])
        assert not (run_root / "training_checkpoint.pt").exists()

    def test_a_train_split_with_fewer_chunks_than_a_batch_is_refused(
        self, tmp_path: Path, tiny_model: list[torch.nn.Module]
    ) -> None:
        built = _written_labels(tmp_path, train=["a"], val=[])
        run_root = tmp_path / "run"

        # One 16-frame video makes 3 chunks, and a batch takes 4.
        with pytest.raises(ValueError, match=r"3 chunks.*train_bs of 4"):
            _ = train_feral(
                _smoke_config(tmp_path, train_bs=4),
                built.labels_json,
                class_values=built.class_values,
                run_root=run_root,
                progress_callback=None,
            )

        assert tiny_model == []
        assert not (run_root / "model_best.pt").exists()

    def test_a_run_interrupted_after_real_validation_metrics_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def build_model(
            cfg: FeralConfig,
            num_classes: int,
            device: str | torch.device,
            *,
            with_ema: bool = True,
        ) -> tuple[torch.nn.Module, _Ema]:
            module = _tiny_module(num_classes, cfg["predict_per_item"]).to(device)
            ema = _timm_model_ema()(module, decay=0.9, device=str(device))
            return module, ema

        loops = _feral_module("loops")
        metrics_module = _feral_module("metrics")
        train_one_epoch = loops.train_one_epoch
        evaluate = loops.evaluate
        frame_level_map = metrics_module.calc_frame_level_map

        # FERAL's losses and mAP come back as numpy scalars on some releases,
        # and the resume checkpoint has to hold plain floats all the same.
        def numpy_train_one_epoch(
            *args: object, **kwargs: object
        ) -> tuple[list[FeralAnswer], np.float64]:
            answers, loss = train_one_epoch(*args, **kwargs)
            return answers, np.float64(loss)

        def numpy_evaluate(
            *args: object, **kwargs: object
        ) -> tuple[list[FeralAnswer], np.float64 | None]:
            answers, loss = evaluate(*args, **kwargs)
            return answers, None if loss is None else np.float64(loss)

        def numpy_frame_level_map(*args: object, **kwargs: object) -> np.float64:
            return np.float64(frame_level_map(*args, **kwargs))

        monkeypatch.setattr(loops, "train_one_epoch", numpy_train_one_epoch)
        monkeypatch.setattr(loops, "evaluate", numpy_evaluate)
        monkeypatch.setattr(
            metrics_module, "calc_frame_level_map", numpy_frame_level_map
        )
        monkeypatch.setattr(_feral_module("modeling"), "build_model", build_model)
        built = _written_labels(tmp_path, train=["a", "b"], val=["c"])
        run_root = tmp_path / "run"
        run_root.mkdir()
        config = _epochs_config(tmp_path, epochs=2, ema_decay=0.9)

        class Interrupted(Exception):
            pass

        class InterruptAfterSecondEpoch(_RecordedEpochs):
            def on_epoch_end(
                self, epoch: int, total_epochs: int, metrics: dict[str, float]
            ) -> None:
                super().on_epoch_end(epoch, total_epochs, metrics)
                if epoch == 1:
                    raise Interrupted

        with pytest.raises(Interrupted):
            _ = train_feral(
                config,
                built.labels_json,
                class_values=built.class_values,
                run_root=run_root,
                progress_callback=InterruptAfterSecondEpoch(),
            )
        assert (run_root / "training_checkpoint.pt").is_file()
        resumed = _RecordedEpochs()

        _, _, metrics = train_feral(
            config,
            built.labels_json,
            class_values=built.class_values,
            run_root=run_root,
            progress_callback=resumed,
        )

        assert [epoch for epoch, _, _ in resumed.epochs] == [1]
        assert [m["epoch"] for m in metrics] == [0, 1]
        assert any(key.startswith("val/f1_") for key in metrics[0])
        assert "ema_val_frame_level_map" in metrics[0]
        assert not (run_root / "training_checkpoint.pt").exists()


# --- Loading, saving and fitting, against FERAL's real pieces --------------------


_NESTED_CONFIG: dict[str, object] = {
    "backbone": "vjepa2_vitl_diving48",
    "predict_per_item": 8,
    "model": {"fc_drop_rate": 0.5, "freeze_encoder_layers": 2},
}


def _head_module(num_classes: int) -> torch.nn.Module:
    """A stand-in for FERAL's model: one linear head, whose weights a checkpoint holds."""
    import torch

    class HeadModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(3, num_classes)

    return HeadModule()


def _built_with(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace ``FeralModel`` by a factory of head modules.

    Returns the list that records each call's kwargs.
    """
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> torch.nn.Module:
        calls.append(kwargs)
        num_classes = kwargs["num_classes"]
        assert isinstance(num_classes, int)
        return _head_module(num_classes)

    monkeypatch.setattr(_feral_module("model"), "FeralModel", factory)
    return calls


def _saved_config(feature: FeralFeature, run_root: Path) -> dict[str, object]:
    """What a loaded feature records of its model, as ``save_state`` writes it."""
    feature.save_state(run_root)
    return json.loads((run_root / "feral_config.json").read_text())


def _saved_head_bundle(
    run_root: Path, class_values: Sequence[int] = (2, 5)
) -> dict[str, torch.Tensor]:
    """A ``feral_model.joblib`` and ``feral_config.json`` of a head module's weights.

    Returns the weights written.
    """
    import joblib

    run_root.mkdir(parents=True, exist_ok=True)
    state_dict = _head_module(len(class_values)).state_dict()
    bundle = {
        "state_dict": state_dict,
        "class_values": list(class_values),
        "class_names": {value: _NAMES[value] for value in class_values},
        "is_multilabel": False,
        "cfg": _NESTED_CONFIG,
    }
    _ = joblib.dump(bundle, run_root / "feral_model.joblib")
    _ = (run_root / "feral_config.json").write_text("{}")
    return state_dict


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestLoadModel:
    """``_load_model`` builds FERAL's model from a bundle or an imported checkpoint."""

    def test_a_checkpoint_of_a_mosaic_run_is_imported_under_its_label_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        _ = _built_with(monkeypatch)
        model_dir = tmp_path / "imported"
        model_dir.mkdir()
        torch.save(
            {
                "state_dict": _head_module(2).state_dict(),
                "class_names": {"0": "walk", "1": "run"},
                "class_values": [2, 5],
                "is_multilabel": False,
                "cfg": _NESTED_CONFIG,
            },
            model_dir / "model_best.pt",
        )
        feature = _feature({"model_dir": str(model_dir), "device": "cpu"})

        assert feature.load_state(tmp_path / "run", {}, {}) is True

        saved = _saved_config(feature, tmp_path / "saved")
        assert saved["class_values"] == [2, 5]
        assert saved["class_names"] == {"2": "walk", "5": "run"}

    def test_a_bare_checkpoint_under_model_dir_takes_the_class_names_param(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        built = _built_with(monkeypatch)
        model_dir = tmp_path / "imported"
        model_dir.mkdir()
        torch.save(_head_module(2).state_dict(), model_dir / "model_best.pt")
        feature = _feature(
            {
                "model_dir": str(model_dir),
                "class_names": {"0": "rest", "1": "walk"},
                "device": "cpu",
            }
        )

        assert feature.load_state(tmp_path / "run", {}, {}) is True

        (kwargs,) = built
        assert kwargs["num_classes"] == 2
        assert kwargs["backbone"] == "vjepa2_vitl_diving48"
        assert kwargs["predict_per_item"] == 64
        assert kwargs["pretrained"] is False
        saved = _saved_config(feature, tmp_path / "saved")
        assert saved["class_values"] == [0, 1]
        assert saved["class_names"] == {"0": "rest", "1": "walk"}
        assert saved["is_multilabel"] is False

    def test_a_bundle_builds_the_model_from_its_nested_config_and_classes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = _built_with(monkeypatch)
        _saved_head_bundle(tmp_path / "run")
        feature = _feature({"model": _TRAINED_FERAL, "device": "cpu"})

        assert feature.load_state(tmp_path / "run", {}, {}) is True

        (kwargs,) = built
        assert kwargs == {
            "backbone": "vjepa2_vitl_diving48",
            "num_classes": 2,
            "predict_per_item": 8,
            "fc_drop_rate": 0.5,
            "freeze_encoder_layers": 2,
            "pretrained": False,
        }
        saved = _saved_config(feature, tmp_path / "saved")
        assert saved["class_values"] == [2, 5]
        assert saved["class_names"] == {"2": "walk", "5": "run"}
        assert saved["cfg"] == _NESTED_CONFIG

    def test_the_class_names_param_renames_the_classes_of_a_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _built_with(monkeypatch)
        _saved_head_bundle(tmp_path / "run")
        feature = _feature(
            {
                "model": _TRAINED_FERAL,
                "device": "cpu",
                "class_names": {"2": "stroll"},
            }
        )

        _ = feature.load_state(tmp_path / "run", {}, {})

        saved = _saved_config(feature, tmp_path / "saved")
        assert saved["class_values"] == [2, 5]
        assert saved["class_names"] == {"2": "stroll", "5": "run"}

    def test_a_class_name_for_a_value_the_model_lacks_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _built_with(monkeypatch)
        _saved_head_bundle(tmp_path / "run")
        feature = _feature(
            {"model": _TRAINED_FERAL, "device": "cpu", "class_names": {"3": "x"}}
        )

        with pytest.raises(ValueError, match="class_names names the values"):
            _ = feature.load_state(tmp_path / "run", {}, {})


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestSetupCheck:
    """``feral_setup_check`` builds the network a checkpoint records and loads it."""

    def test_the_network_comes_from_the_checkpoints_config_and_class_names(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import torch

        built = _built_with(monkeypatch)
        checkpoint = tmp_path / "model_best.pt"
        torch.save(
            {
                "state_dict": _head_module(2).state_dict(),
                "class_names": {"0": "walk", "1": "run"},
                "is_multilabel": False,
                "cfg": _NESTED_CONFIG,
            },
            checkpoint,
        )

        summary = feral_setup_check(checkpoint, build_model=True)

        (kwargs,) = built
        assert kwargs["num_classes"] == 2
        assert kwargs["predict_per_item"] == _NESTED_CONFIG["predict_per_item"]
        assert kwargs["pretrained"] is False
        assert summary["strict_load_ok"] is True
        assert (
            "strict load into FeralModel(num_classes=2) succeeded"
            in capsys.readouterr().out
        )

    def test_a_checkpoint_without_a_config_is_built_from_the_backbone_argument(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        built = _built_with(monkeypatch)
        checkpoint = tmp_path / "model_best.pt"
        torch.save(_head_module(3).state_dict(), checkpoint)

        summary = feral_setup_check(checkpoint, num_classes=3, build_model=True)

        (kwargs,) = built
        assert kwargs["backbone"] == "vjepa2_vitl_diving48"
        assert kwargs["num_classes"] == 3
        assert kwargs["freeze_encoder_layers"] == 0
        assert summary["strict_load_ok"] is True


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestSaveState:
    """``save_state`` writes the model file the run loads and applies from."""

    def _saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, Path]:
        """The run root ``save_state`` wrote, and the source run it loaded."""
        _ = _built_with(monkeypatch)
        source = tmp_path / "source"
        _saved_head_bundle(source)
        feature = _feature({"model": _TRAINED_FERAL, "device": "cpu"})
        _ = feature.load_state(source, {}, {})
        run_root = tmp_path / "run"

        feature.save_state(run_root)

        return run_root, source

    def test_the_artifact_reads_the_bundle_that_was_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        run_root, source = self._saved(tmp_path, monkeypatch)
        original = FeralModelArtifact().from_path(source / "feral_model.joblib")

        bundle = FeralModelArtifact().from_path(run_root / "feral_model.joblib")

        assert sorted(bundle) == [
            "cfg",
            "class_names",
            "class_values",
            "is_multilabel",
            "state_dict",
        ]
        assert bundle["class_values"] == [2, 5]
        assert bundle["class_names"] == {2: "walk", 5: "run"}
        assert bundle["is_multilabel"] is False
        assert bundle["cfg"] == _NESTED_CONFIG
        assert sorted(bundle["state_dict"]) == ["head.bias", "head.weight"]
        for name, weight in original["state_dict"].items():
            assert bundle["state_dict"][name].device.type == "cpu"
            assert torch.equal(bundle["state_dict"][name], weight)

    def test_the_config_holds_the_nested_config_and_no_flat_copy_of_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_root, _source = self._saved(tmp_path, monkeypatch)

        config = json.loads((run_root / "feral_config.json").read_text())

        assert config == {
            "cfg": _NESTED_CONFIG,
            "class_values": [2, 5],
            "class_names": {"2": "walk", "5": "run"},
            "is_multilabel": False,
            "inference_autocast": False,
            "version": FeralFeature.version,
        }

    def test_the_run_writes_no_checkpoint_of_the_old_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_root, _source = self._saved(tmp_path, monkeypatch)

        assert not (run_root / "feral_model.pt").exists()

    def test_the_saved_run_loads_as_a_cached_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_root, _source = self._saved(tmp_path, monkeypatch)
        feature = _feature({"model": _TRAINED_FERAL, "device": "cpu"})

        assert feature.load_state(run_root, {}, {}) is True

        saved = _saved_config(feature, tmp_path / "resaved")
        assert saved["class_values"] == [2, 5]


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestLoadTrainedModel:
    """``load_trained_model`` reads the model file, and ``model_best.pt`` without it."""

    def test_it_reads_the_model_file_of_a_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        built = _built_with(monkeypatch)
        weights = _saved_head_bundle(tmp_path)

        model = FeralFeature.load_trained_model(tmp_path, device="cpu")

        assert [kwargs["num_classes"] for kwargs in built] == [2]
        assert not model.training
        assert isinstance(model, torch.nn.Module)
        for name, weight in weights.items():
            assert torch.equal(model.state_dict()[name], weight)

    def test_it_falls_back_to_the_best_checkpoint_and_its_nested_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        built = _built_with(monkeypatch)
        reference = _head_module(2)
        torch.save(
            {
                "state_dict": reference.state_dict(),
                "class_names": {"0": "rest", "1": "walk"},
                "is_multilabel": False,
                "cfg": _NESTED_CONFIG,
            },
            tmp_path / "model_best.pt",
        )

        model = FeralFeature.load_trained_model(tmp_path, device="cpu")

        assert [kwargs["predict_per_item"] for kwargs in built] == [8]
        assert torch.equal(
            model.state_dict()["head.weight"], reference.state_dict()["head.weight"]
        )

    def test_a_directory_with_neither_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="feral_model.joblib"):
            _ = FeralFeature.load_trained_model(tmp_path, device="cpu")


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestEvaluate:
    """``evaluate`` reads the labels and config a trained run holds."""

    def _trained_run(self, tmp_path: Path) -> Path:
        built = _written_labels(tmp_path, train=["a", "b"], val=[], test=["c"])
        run_root = tmp_path / "run"
        _saved_head_bundle(run_root)
        _ = (run_root / "feral_labels.json").write_text(json.dumps(built.labels_json))
        trained_with = _smoke_config(tmp_path)
        # The run's dataset has moved: only the dataset_root argument finds it.
        trained_with["data"]["prefix"] = str(tmp_path / "moved")
        config = {
            "cfg": trained_with,
            "class_values": [2, 5],
            "class_names": {"2": "walk", "5": "run"},
            "is_multilabel": False,
            "inference_autocast": False,
            "version": FeralFeature.version,
        }
        _ = (run_root / "feral_config.json").write_text(json.dumps(config))
        return run_root

    def test_it_returns_the_answers_of_the_test_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _built_with(monkeypatch)
        run_root = self._trained_run(tmp_path)
        answers: list[FeralAnswer] = [(("c.mp4", 0, 0), [0.4, 0.6], [0.0, 1.0])]
        seen: list[tuple[int, bool, int]] = []

        def stub(
            model: torch.nn.Module,
            loader: torch.utils.data.DataLoader[object] | None,
            num_classes: int,
            is_multilabel: bool,
            device: object,
            **_: object,
        ) -> list[FeralAnswer]:
            assert loader is not None
            seen.append((num_classes, is_multilabel, len(loader)))
            return answers

        monkeypatch.setattr(feral_feature, "_run_inference_loop", stub)

        result = FeralFeature.evaluate(
            run_root, "test", dataset_root=tmp_path, device="cpu"
        )

        assert result == answers
        ((num_classes, is_multilabel, batches),) = seen
        assert (num_classes, is_multilabel) == (2, False)
        assert batches > 0

    def test_a_partition_without_videos_gives_no_answers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = _built_with(monkeypatch)
        run_root = self._trained_run(tmp_path)

        assert (
            FeralFeature.evaluate(run_root, "val", dataset_root=tmp_path, device="cpu")
            == []
        )


class _RecordedDataset:
    """A ``ClsDataset`` stand-in that keeps its keyword arguments."""

    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).calls.append(kwargs)

    def __len__(self) -> int:
        return 0


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestEvaluatePartition:
    """The held-out evaluation reads FERAL's chunking from the config it trained with."""

    def test_the_dataset_is_built_with_the_configs_chunking_and_the_dataset_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecordedDataset.calls = []
        monkeypatch.setattr(_feral_module("dataset"), "ClsDataset", _RecordedDataset)

        def no_answers(*args: object, **kwargs: object) -> list[FeralAnswer]:
            return []

        monkeypatch.setattr(feral_feature, "_run_inference_loop", no_answers)
        built = _written_labels(tmp_path, train=["a", "b"], val=[], test=["c"])
        params = _params(
            {**_SMOKE_TRAINING},
            device="cpu",
            chunk_length=12,
            chunk_shift=5,
            chunk_step=2,
            predict_per_item=8,
            resize_to=48,
        )
        cfg = feral_config(params, backbone=_BACKBONE, dataset_root=tmp_path)

        _ = _evaluate_partition(_head_module(2), cfg, built.labels_json, "test", "cpu")

        (kwargs,) = _RecordedDataset.calls
        assert kwargs["partition"] == "test"
        assert kwargs["chunk_length"] == 12
        assert kwargs["chunk_shift"] == 5
        assert kwargs["chunk_step"] == 2
        assert kwargs["resize_to"] == 48
        assert kwargs["prefix"] == str(tmp_path)
        assert kwargs["do_aa"] is False
        assert kwargs["num_classes"] == 2


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
@pytest.mark.filterwarnings(*_CPU_RUN_WARNINGS)
class TestFitOnCpu:
    """A whole ``fit`` over crop clips, with FERAL's loop and a tiny model."""

    def test_a_fit_writes_the_labels_the_config_and_the_model_file(
        self, tmp_path: Path, tiny_model: list[torch.nn.Module]
    ) -> None:
        ds = _dataset(tmp_path)
        crop_root = (
            tmp_path
            / "features"
            / "interaction-crop-pipeline__from__tracks"
            / "0.1-abc"
        )
        crop_root.mkdir(parents=True)
        _ = (crop_root / "params.json").write_text(_params_json({}))
        entries = ["a", "b", "c", "d"]
        for entry in entries:
            _write_video(crop_root / f"{entry}.mp4")
        label_files = _entry_files(
            tmp_path / "labels",
            {("g", entry): _pair_events(2, 5, count=8) for entry in entries},
        )
        templates = _templates_run(tmp_path / "templates", ["a", "b", "c"], ["d"])
        feature = FeralFeature(
            FeralFeature.Inputs((_CROP_RESULT,)),
            {
                **_TRAINING_SOURCE,
                "training": {**_SMOKE_TRAINING, "val_fraction": 0.34},
                "device": "cpu",
                "chunk_length": 8,
                "chunk_shift": 4,
                "chunk_step": 1,
                "predict_per_item": 8,
                "resize_to": 32,
            },
        )
        feature.bind_dataset(ds)
        run_root = tmp_path / "run"
        ready = feature.load_state(
            run_root,
            {"templates": templates},
            {
                "labels": label_files,
                "_input_0": {("g", "a"): crop_root / "table.parquet"},
            },
        )
        assert ready is False
        stream = InputStream(
            lambda: iter(
                [
                    (
                        f"g/{entry}",
                        _interaction_rows((f"{entry}.mp4", 0, 15, 16, 0)).assign(
                            sequence=entry
                        ),
                    )
                    for entry in entries
                ]
            ),
            len(entries),
        )

        feature.fit(stream)
        feature.save_state(run_root)

        labels_json = json.loads((run_root / "feral_labels.json").read_text())
        assert sorted(labels_json["splits"]) == ["test", "train", "val"]
        assert len(labels_json["splits"]["val"]) == 1
        assert len(labels_json["splits"]["test"]) == 1
        config = json.loads((run_root / "config.json").read_text())
        assert config["data"]["prefix"] == str(tmp_path)
        bundle = FeralModelArtifact().from_path(run_root / "feral_model.joblib")
        assert sorted(bundle) == [
            "cfg",
            "class_names",
            "class_values",
            "is_multilabel",
            "state_dict",
        ]
        assert bundle["class_values"] == [2, 5]
        assert bundle["class_names"] == {2: "walk", 5: "run"}
        assert bundle["cfg"] == config
        reports = json.loads((run_root / "reports.json").read_text())
        assert reports["test_metrics"]
        assert (run_root / "summary.csv").is_file()
        assert (run_root / "feral_config.json").is_file()


# --- apply ------------------------------------------------------------------

_APPLY_PARAMS: dict[str, object] = {
    "model": _TRAINED_FERAL,
    "device": "cpu",
    "chunk_length": 8,
    "chunk_shift": 8,
    "chunk_step": 1,
    "resize_to": 8,
    "infer_batch_size": 2,
}
_EGOCENTRIC_RESULT = Result(feature="egocentric-crop__from__tracks", run_id="0.1-abc")
_PAIR_COLUMNS = ("id1", "id2", "perspective", "interaction_id")


def _constant_logits(row: int) -> tuple[float, float]:
    """The same two logits on every row the stand-in model emits."""
    return (0.0, 2.0)


def _fixed_logits(
    logit_of_row: Callable[[int], Sequence[float]],
    tokens_per_chunk: int,
    batch_shapes: list[tuple[int, ...]] | None = None,
) -> torch.nn.Module:
    """A stand-in for FERAL's model that emits *tokens_per_chunk* rows per chunk.

    Row ``r`` of a batch holds the logits ``logit_of_row(r)``, so a test reads
    which row of the model's output each frame was given. The shape of each
    batch the model reads is appended to *batch_shapes*.
    """
    import torch

    class FixedLogits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            if batch_shapes is not None:
                batch_shapes.append(tuple(batch.shape))
            rows = int(batch.shape[0]) * tokens_per_chunk
            logits = torch.tensor([list(logit_of_row(row)) for row in range(rows)])
            return logits + self.anchor

    return FixedLogits()


_DECODED_SIZE = (8, 8)
"""The ``(width, height)`` of every frame the stand-in decoder returns."""


def _decoding(
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
    reads: list[tuple[str, list[int], int, int]],
) -> None:
    """Replace FERAL's frame counter and decoder.

    Each read is appended to *reads* as ``(path, frames, width, height)``.
    """
    import torch

    def count(path: str) -> int | None:
        return frame_count

    def read(
        path: str, frames: list[int], width: int = -1, height: int = -1
    ) -> torch.Tensor:
        reads.append((path, list(frames), width, height))
        return torch.zeros(
            len(frames), 3, _DECODED_SIZE[1], _DECODED_SIZE[0], dtype=torch.uint8
        )

    dataset = _feral_module("dataset")
    monkeypatch.setattr(dataset, "get_frame_count", count)
    monkeypatch.setattr(dataset, "read_range_video_decord", read)


def _applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: pd.DataFrame,
    inputs: FeralFeature.Inputs,
    *,
    lookups: dict[str, dict[tuple[str, str], Path]] | None = None,
    params: Mapping[str, object] | None = None,
    is_multilabel: bool = False,
    logit_of_row: Callable[[int], Sequence[float]] = _constant_logits,
    tokens_per_chunk: int = 8,
    frame_count: int = 16,
    reads: list[tuple[str, list[int], int, int]] | None = None,
    batch_shapes: list[tuple[int, ...]] | None = None,
) -> pd.DataFrame:
    """Load a cached run of a stand-in model, and apply it to *rows*.

    *reads* collects what the stand-in decoder is asked for, and *batch_shapes*
    the shape of each batch the model reads.
    """
    import joblib

    stand_in = _fixed_logits(logit_of_row, tokens_per_chunk, batch_shapes)

    def build(**kwargs: object) -> torch.nn.Module:
        return stand_in

    monkeypatch.setattr(_feral_module("model"), "FeralModel", build)
    _decoding(monkeypatch, frame_count, [] if reads is None else reads)
    run_root = tmp_path / "run"
    run_root.mkdir()
    bundle = {
        "state_dict": stand_in.state_dict(),
        "class_values": [2, 5],
        "class_names": {2: "walk", 5: "run"},
        "is_multilabel": is_multilabel,
        "cfg": {**_NESTED_CONFIG, "predict_per_item": tokens_per_chunk},
    }
    _ = joblib.dump(bundle, run_root / "feral_model.joblib")
    _ = (run_root / "feral_config.json").write_text("{}")
    feature = FeralFeature(inputs, {**_APPLY_PARAMS, **(params or {})})
    feature.bind_dataset(_dataset(tmp_path))
    assert feature.load_state(run_root, {}, lookups or {}) is True
    return feature.apply(rows)


def _interaction_clip(
    tmp_path: Path, first: int = 20
) -> tuple[pd.DataFrame, FeralFeature.Inputs, dict[str, dict[tuple[str, str], Path]]]:
    """One 16-frame interaction clip on disk, its row, its input and its lookup."""
    crop_root = _crop_run(tmp_path, _params_json({}))
    _ = (crop_root / "clip.mp4").write_bytes(b"")
    rows = _interaction_rows(("clip.mp4", first, first + 15, 16, 7))
    lookups = {"_input_0": {("g", "s"): crop_root / "table.parquet"}}
    return rows, FeralFeature.Inputs((_CROP_RESULT,)), lookups


def _sequence_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, first_frame: int
) -> None:
    """Make entry ``("g", "s1")`` resolve to one video file placed at *first_frame*."""
    video = tmp_path / "s1.mp4"
    _ = video.write_bytes(b"")
    _resolves(monkeypatch, _media([video], frame_counts=(16,), placement=first_frame))


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestApply:
    """``apply`` turns the model's output into the rows of its input's frames."""

    def test_a_single_label_models_probabilities_are_softmax_and_sum_to_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows, inputs, lookups = _interaction_clip(tmp_path)

        result = _applied(tmp_path, monkeypatch, rows, inputs, lookups=lookups)

        np.testing.assert_allclose(result["prob_2"] + result["prob_5"], 1.0, atol=1e-6)
        expected = float(np.exp(2.0) / (1.0 + np.exp(2.0)))
        np.testing.assert_allclose(result["prob_5"], expected, atol=1e-6)

    def test_a_multilabel_models_probabilities_are_sigmoids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cached run whose bundle says ``is_multilabel`` applies sigmoids."""
        rows, inputs, lookups = _interaction_clip(tmp_path)

        result = _applied(
            tmp_path, monkeypatch, rows, inputs, lookups=lookups, is_multilabel=True
        )

        np.testing.assert_allclose(result["prob_2"], 0.5, atol=1e-6)
        np.testing.assert_allclose(result["prob_5"], 1 / (1 + np.exp(-2.0)), atol=1e-6)

    def test_the_label_and_the_probability_columns_use_original_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows, inputs, lookups = _interaction_clip(tmp_path)

        result = _applied(tmp_path, monkeypatch, rows, inputs, lookups=lookups)

        assert [c for c in result.columns if c.startswith("prob_")] == [
            "prob_2",
            "prob_5",
        ]
        assert set(result["predicted_label"]) == {5}

    def test_an_interaction_clip_writes_one_row_per_frame_with_the_crops_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows, inputs, lookups = _interaction_clip(tmp_path, first=20)

        result = _applied(tmp_path, monkeypatch, rows, inputs, lookups=lookups)

        assert list(result["frame"]) == list(range(20, 36))
        for column, value in (
            ("id1", 0),
            ("id2", 1),
            ("perspective", 0),
            ("interaction_id", 7),
            ("group", "g"),
            ("sequence", "s"),
        ):
            assert set(result[column]) == {value}

    def test_an_egocentric_video_with_a_gap_writes_the_frames_of_its_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        crop_root = _crop_run(tmp_path, _params_json({}))
        (crop_root / "g__s").mkdir()
        _ = (crop_root / "g__s" / "egocentric_id3.mp4").write_bytes(b"")
        frames = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 20, 21, 22, 23]
        rows = pd.DataFrame({"group": "g", "sequence": "s", "id": 3, "frame": frames})
        lookups = {"_input_0": {("g", "s"): crop_root / "table.parquet"}}

        result = _applied(
            tmp_path,
            monkeypatch,
            rows,
            FeralFeature.Inputs((_EGOCENTRIC_RESULT,)),
            lookups=lookups,
        )

        assert list(result["frame"]) == frames
        assert set(result["id"]) == {3}

    def test_a_tracks_input_writes_a_row_per_individual_and_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sequence_of(tmp_path, monkeypatch, first_frame=10)
        rows = pd.DataFrame(
            {
                "group": "g",
                "sequence": "s1",
                "id": [0, 1] * 3,
                "frame": [10, 10, 11, 11, 12, 12],
                "speed": 1.0,
            }
        )

        result = _applied(tmp_path, monkeypatch, rows, FeralFeature.Inputs(("tracks",)))

        assert len(result) == 6
        assert sorted(zip(result["id"], result["frame"], strict=True)) == [
            (0, 10),
            (0, 11),
            (0, 12),
            (1, 10),
            (1, 11),
            (1, 12),
        ]
        assert "speed" not in result.columns
        assert set(result["predicted_label"]) == {5}
        assert set(result["group"]) == {"g"}
        assert set(result["sequence"]) == {"s1"}

    def test_the_frames_of_a_sequence_video_follow_its_first_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Video frame ``k`` is frame ``first_frame + k``, and a row beyond it has no prediction."""
        _sequence_of(tmp_path, monkeypatch, first_frame=10)
        rows = pd.DataFrame(
            {"group": "g", "sequence": "s1", "id": 0, "frame": [9, 10, 25, 26]}
        )

        result = _applied(tmp_path, monkeypatch, rows, FeralFeature.Inputs(("tracks",)))

        assert sorted(result["frame"]) == [10, 25]

    def test_a_pair_input_keeps_its_pair_columns_and_writes_no_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sequence_of(tmp_path, monkeypatch, first_frame=0)
        rows = pd.DataFrame(
            {
                "group": "g",
                "sequence": "s1",
                "id1": [0, 1],
                "id2": [1, 0],
                "perspective": [0, 1],
                "interaction_id": [4, 4],
                "frame": [3, 3],
            }
        )

        result = _applied(tmp_path, monkeypatch, rows, FeralFeature.Inputs(("tracks",)))

        assert len(result) == 2
        assert list(result["id1"]) == [0, 1]
        assert list(result["perspective"]) == [0, 1]
        assert list(result["interaction_id"]) == [4, 4]
        assert "id" not in result.columns

    def test_frames_are_decoded_at_the_model_size_and_not_resized_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows, inputs, lookups = _interaction_clip(tmp_path)
        reads: list[tuple[str, list[int], int, int]] = []
        batch_shapes: list[tuple[int, ...]] = []

        _ = _applied(
            tmp_path,
            monkeypatch,
            rows,
            inputs,
            lookups=lookups,
            params={"resize_to": 12},
            reads=reads,
            batch_shapes=batch_shapes,
        )

        assert reads
        assert {(width, height) for _, _, width, height in reads} == {(12, 12)}
        assert {shape[3:] for shape in batch_shapes} == {
            (_DECODED_SIZE[1], _DECODED_SIZE[0])
        }

    def test_the_threshold_falls_back_to_the_default_class_as_an_original_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows, inputs, lookups = _interaction_clip(tmp_path)

        def even(row: int) -> tuple[float, float]:
            return (0.0, 0.0)

        result = _applied(
            tmp_path,
            monkeypatch,
            rows,
            inputs,
            lookups=lookups,
            params={"decision_threshold": 0.9, "default_class": 5},
            logit_of_row=even,
        )

        assert set(result["predicted_label"]) == {5}

    def test_a_threshold_that_a_class_clears_picks_that_class_as_an_original_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows, inputs, lookups = _interaction_clip(tmp_path)

        def first_class(row: int) -> tuple[float, float]:
            return (3.0, 0.0)

        result = _applied(
            tmp_path,
            monkeypatch,
            rows,
            inputs,
            lookups=lookups,
            params={"decision_threshold": 0.9, "default_class": 5},
            logit_of_row=first_class,
        )

        assert set(result["predicted_label"]) == {2}

    def test_a_video_whose_counted_frames_differ_from_its_frame_numbers_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows, inputs, lookups = _interaction_clip(tmp_path)

        with pytest.raises(ValueError, match=r"clip\.mp4"):
            _ = _applied(
                tmp_path,
                monkeypatch,
                rows,
                inputs,
                lookups=lookups,
                frame_count=15,
            )

    def test_each_chunk_is_sliced_by_the_models_predictions_per_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model's config sets the output rows of a chunk.

        The step's ``predict_per_item`` is ignored.
        """
        rows, inputs, lookups = _interaction_clip(tmp_path, first=0)

        def by_row(row: int) -> tuple[float, float]:
            return (0.0, 0.1 * row)

        result = _applied(
            tmp_path,
            monkeypatch,
            rows,
            inputs,
            lookups=lookups,
            params={"predict_per_item": 4},
            logit_of_row=by_row,
            tokens_per_chunk=8,
        )

        assert list(result["frame"]) == list(range(16))
        expected = [1 / (1 + np.exp(-0.1 * frame)) for frame in range(16)]
        np.testing.assert_allclose(result["prob_5"], expected, atol=1e-5)


def _provided(columns: Sequence[str]) -> ColumnSet:
    """The columns FERAL's output declares over an input holding *columns*."""
    merged = measured_columns([columns])
    assert merged is not None
    return output_columns(FeralFeature.provides, merged, {})


class TestProvides:
    """The identity columns of the output follow the input's own."""

    def test_a_tracks_input_gives_id(self) -> None:
        provided = _provided(["group", "sequence", "frame", "id", "x", "y"])

        assert {"id", "frame", "predicted_label", "group", "sequence"} <= (
            provided.guaranteed
        )
        assert "prob_" in provided.guaranteed_prefixes
        assert not provided.admits("id1")
        assert not provided.admits("x")

    def test_a_pair_input_without_perspective_or_interaction_gives_the_pair_ids(
        self,
    ) -> None:
        provided = _provided(["group", "sequence", "frame", "id1", "id2"])

        assert {"id1", "id2"} <= provided.guaranteed
        assert not provided.admits("perspective")
        assert not provided.admits("interaction_id")
        assert not provided.admits("id")

    def test_a_pair_input_with_them_gives_all_four(self) -> None:
        provided = _provided(_PAIR_COLUMNS)

        assert set(_PAIR_COLUMNS) <= provided.guaranteed

    def test_an_input_with_only_a_perspective_keeps_that_column(self) -> None:
        provided = _provided(["frame", "id1", "id2", "perspective"])

        assert "perspective" in provided.guaranteed
        assert not provided.admits("interaction_id")


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestChunkProbabilities:
    """The probabilities of a chunk batch are float32 whatever the autocast dtype."""

    @pytest.mark.parametrize("is_multilabel", [False, True], ids=["softmax", "sigmoid"])
    def test_a_bfloat16_autocast_gives_float32_probabilities(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_multilabel: bool
    ) -> None:
        import torch

        def read(path: str, frames: list[int], width: int, height: int) -> torch.Tensor:
            return torch.zeros(len(frames), 4, dtype=torch.uint8)

        monkeypatch.setattr(_feral_module("dataset"), "read_range_video_decord", read)

        class Head(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                return self.linear(batch.reshape(-1, 4))

        def prepare(video: torch.Tensor) -> torch.Tensor:
            return video.float()

        probabilities = chunk_probabilities(
            Head().eval(),
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            tmp_path / "video.mp4",
            prepare,
            torch.device("cpu"),
            torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16),
            decode_size=(4, 4),
            is_multilabel=is_multilabel,
            tokens_per_chunk=4,
        )

        assert [chunk.shape for chunk in probabilities] == [(4, 3), (4, 3)]
        assert {chunk.dtype for chunk in probabilities} == {np.dtype(np.float32)}
