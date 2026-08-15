"""Tests for FeralFeature's FERAL-package compatibility layer.

These cover the pure-Python compat helpers (checkpoint remap, backbone-key
resolution) and the optional-dependency import guard. They run without a GPU;
the parts that need the ``feral`` package are skipped when it isn't installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mosaic.behavior.feature_library.feral_feature import (
    FeralFeature,
    _check_feral,
    _load_checkpoint_state_dict,
    _resolve_dataset_path,
)
from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.types import Result

_HAS_FERAL = importlib.util.find_spec("feral") is not None


class TestLoadCheckpointStateDict:
    """`_load_checkpoint_state_dict` normalizes every checkpoint shape."""

    def test_new_dict_format_unwraps_and_returns_metadata(self) -> None:
        raw = {
            "state_dict": {"backbone.model.enc": 1, "head.weight": 2},
            "class_names": {"0": "none", "1": "troph"},
            "is_multilabel": False,
            "cfg": {"backbone": "vjepa2_vitl_diving48"},
        }
        sd, meta = _load_checkpoint_state_dict(raw)
        assert sd == {"backbone.model.enc": 1, "head.weight": 2}
        assert meta == {
            "class_names": {"0": "none", "1": "troph"},
            "is_multilabel": False,
            "cfg": {"backbone": "vjepa2_vitl_diving48"},
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
    """FeralFeature is registered even without feral; instantiating without it errors."""

    def test_feature_is_registered(self) -> None:
        # The module imports (light deps only), so the feature registers
        # regardless of whether feral/torch are installed.
        assert "FeralFeature" in FEATURES

    @pytest.mark.skipif(_HAS_FERAL, reason="feral is installed")
    def test_check_feral_raises_helpful_error(self) -> None:
        with pytest.raises(ImportError, match=r"mosaic-behavior\[feral\]"):
            _check_feral()

    @pytest.mark.skipif(_HAS_FERAL, reason="feral is installed")
    def test_instantiation_raises_without_feral(self) -> None:
        with pytest.raises(ImportError, match=r"mosaic-behavior\[feral\]"):
            FeralFeature(
                FeralFeature.Inputs((Result(feature="upstream"),)),
                {"model_dir": "/tmp/does-not-matter"},
            )


class TestPathResolution:
    """A dataset-relative ``model_dir`` resolves against the dataset, not the CWD.

    Exercised through the module-level helper rather than the class on purpose:
    ``FeralFeature.__init__`` raises without the optional ``feral`` package, which
    only the ``feral`` CI job installs, so a class-routed test would be skipped in
    every other one -- and this is a correctness fix that needs coverage
    everywhere.
    """

    @staticmethod
    def _dataset(tmp_path: Path) -> Dataset:
        manifest = new_dataset_manifest("f", base_dir=tmp_path)
        return Dataset(manifest_path=manifest).load(ensure_roots=True)

    def test_relative_path_resolves_under_the_dataset_root(
        self, tmp_path: Path
    ) -> None:
        ds = self._dataset(tmp_path)

        resolved = _resolve_dataset_path(ds, "models/feral/0.1-abc")

        assert resolved == ds.get_root("models") / "feral" / "0.1-abc"
        assert resolved is not None and resolved.is_absolute()

    def test_absolute_existing_path_is_returned_unchanged(self, tmp_path: Path) -> None:
        """The hash-neutrality argument, as code: an existing absolute is untouched."""
        ds = self._dataset(tmp_path)
        model_dir = tmp_path / "elsewhere" / "0.1-abc"
        model_dir.mkdir(parents=True)

        assert _resolve_dataset_path(ds, str(model_dir)) == model_dir

    def test_absolute_missing_path_is_returned_unchanged(self, tmp_path: Path) -> None:
        """Never raises -- load_state's branch 3 is what reports the miss."""
        ds = self._dataset(tmp_path)
        missing = Path("/media/otherbox/T9/models/feral/0.1-abc")

        assert _resolve_dataset_path(ds, missing) == missing

    def test_no_bound_dataset_falls_back_to_plain_path(self) -> None:
        """A direct load_state call, with no bind_dataset, behaves as it always did."""
        assert _resolve_dataset_path(None, "/abs/model") == Path("/abs/model")
        assert _resolve_dataset_path(None, "relative/model") == Path("relative/model")

    def test_a_path_is_accepted_as_well_as_a_string(self, tmp_path: Path) -> None:
        """``params.model_dir`` is a pydantic ``Path``; the config copies are ``str``."""
        ds = self._dataset(tmp_path)

        assert _resolve_dataset_path(ds, Path("models/feral/0.1-abc")) == (
            _resolve_dataset_path(ds, "models/feral/0.1-abc")
        )


@pytest.mark.feral
@pytest.mark.skipif(not _HAS_FERAL, reason="requires the feral package")
class TestLoadStateResolvesModelDir:
    """``load_state`` reaches the resolved directory, not a CWD-relative one."""

    def test_relative_model_dir_without_a_checkpoint_falls_through_to_fit(
        self, tmp_path: Path
    ) -> None:
        manifest = new_dataset_manifest("f", base_dir=tmp_path)
        ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
        (ds.get_root("models") / "feral" / "0.1-abc").mkdir(parents=True)

        feature = FeralFeature(
            FeralFeature.Inputs((Result(feature="upstream"),)),
            {"model_dir": "models/feral/0.1-abc"},
        )
        feature.bind_dataset(ds)

        run_root = tmp_path / "run"
        run_root.mkdir()
        # Branch 2 finds the resolved directory but no model_best.pt in it, so it
        # falls through to branch 3 -- reached without loading torch.
        assert feature.load_state(run_root, {}, {}) is False

    def test_relative_video_dir_is_resolved_against_the_dataset(
        self, tmp_path: Path
    ) -> None:
        manifest = new_dataset_manifest("f", base_dir=tmp_path)
        ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
        clips = ds.get_root("features") / "clips"
        clips.mkdir(parents=True)
        labels = tmp_path / "labels.json"
        _ = labels.write_text("{}")

        feature = FeralFeature(
            FeralFeature.Inputs((Result(feature="upstream"),)),
            {"video_dir": "features/clips", "label_json": str(labels)},
        )
        feature.bind_dataset(ds)

        run_root = tmp_path / "run"
        run_root.mkdir()
        _ = feature.load_state(run_root, {}, {})

        assert feature._video_dir == clips


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
