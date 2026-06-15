"""Tests for FeralFeature's FERAL-package compatibility layer.

These cover the pure-Python compat helpers (checkpoint remap, backbone-key
resolution) and the optional-dependency import guard. They run without a GPU;
the parts that need the ``feral`` package are skipped when it isn't installed.
"""

from __future__ import annotations

import importlib.util

import pytest

from mosaic.behavior.feature_library.feral_feature import (
    FeralFeature,
    _check_feral,
    _load_checkpoint_state_dict,
)
from mosaic.behavior.feature_library.registry import FEATURES
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
        with pytest.raises(ImportError, match=r"mosaic\[feral\]"):
            _check_feral()

    @pytest.mark.skipif(_HAS_FERAL, reason="feral is installed")
    def test_instantiation_raises_without_feral(self) -> None:
        with pytest.raises(ImportError, match=r"mosaic\[feral\]"):
            FeralFeature(
                FeralFeature.Inputs((Result(feature="upstream"),)),
                {"model_dir": "/tmp/does-not-matter"},
            )


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
