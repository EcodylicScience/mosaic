"""Tests for Pydantic feature parameter models."""
from __future__ import annotations

import json
import pytest
from pydantic import BaseModel as _PydanticBaseModel
from pydantic import Field, ValidationError

from mosaic.behavior.feature_library._param_bases import (
    FeatureParams,
    InterpolationMixin,
    PositionColumnsMixin,
    SamplingMixin,
)


# --- FeatureParams base ---


def test_defaults() -> None:
    p = FeatureParams()
    assert p.id_col == "id"
    assert p.seq_col == "sequence"
    assert p.group_col == "group"
    assert p.order_pref == ("frame", "time")


def test_getitem() -> None:
    p = FeatureParams()
    assert p["id_col"] == "id"
    with pytest.raises(KeyError):
        p["nonexistent"]


def test_get_with_default() -> None:
    p = FeatureParams()
    assert p.get("id_col") == "id"
    assert p.get("nonexistent", "fallback") == "fallback"
    assert p.get("nonexistent") is None


def test_contains() -> None:
    p = FeatureParams()
    assert "id_col" in p
    assert "nonexistent" not in p


def test_keys() -> None:
    p = FeatureParams()
    assert set(p.keys()) == {"id_col", "seq_col", "group_col", "order_pref"}


def test_dict_spread() -> None:
    p = FeatureParams()
    d = {**p}
    assert d == {"id_col": "id", "seq_col": "sequence", "group_col": "group",
                 "order_pref": ("frame", "time")}


def test_dict_spread_with_extra_key() -> None:
    p = FeatureParams()
    d = {**p, "_scope": None}
    assert "_scope" in d
    assert d["id_col"] == "id"


def test_from_overrides_empty() -> None:
    p = FeatureParams.from_overrides(None)
    assert p.id_col == "id"
    p2 = FeatureParams.from_overrides({})
    assert p2.id_col == "id"


def test_from_overrides_rejects_none_for_str_field() -> None:
    with pytest.raises(ValidationError):
        FeatureParams.from_overrides({"id_col": None, "seq_col": "seq"})


def test_from_overrides_applies_values() -> None:
    p = FeatureParams.from_overrides({"id_col": "animal_id"})
    assert p.id_col == "animal_id"


class _InnerModel(_PydanticBaseModel):
    a: int = 1
    b: int = 2


class _ParamsWithNested(FeatureParams):
    nested: _InnerModel = Field(default_factory=_InnerModel)


def test_from_overrides_partial_basemodel_merge() -> None:
    p = _ParamsWithNested.from_overrides({"nested": {"a": 99}})
    assert p.nested.a == 99
    assert p.nested.b == 2


def test_from_overrides_full_basemodel_override() -> None:
    p = _ParamsWithNested.from_overrides({"nested": {"a": 10, "b": 20}})
    assert p.nested.a == 10
    assert p.nested.b == 20


def test_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        FeatureParams(bogus="x")


# --- Mixins ---


class _MixedParams(FeatureParams, PositionColumnsMixin, InterpolationMixin, SamplingMixin):
    pass


def test_mixin_constraints() -> None:
    with pytest.raises(ValidationError):
        _MixedParams(linear_interp_limit=0)
    with pytest.raises(ValidationError):
        _MixedParams(max_missing_fraction=1.5)
    with pytest.raises(ValidationError):
        _MixedParams(fps_default=-1.0)


def test_mixin_keys_in_spread() -> None:
    p = _MixedParams()
    d = {**p}
    assert "x_col" in d
    assert "fps_default" in d
    assert "linear_interp_limit" in d


# --- Subclass override ---


class _OverrideParams(FeatureParams):
    group_col: str = "event"


def test_subclass_can_override_base_default() -> None:
    p = _OverrideParams()
    assert p.group_col == "event"


# --- dataset.py integration ---


def test_hash_params_with_model() -> None:
    from mosaic.core.dataset import _hash_params
    p = FeatureParams()
    d = {**p}
    assert _hash_params(p) == _hash_params(d)


def test_hash_params_deterministic() -> None:
    from mosaic.core.dataset import _hash_params
    p = FeatureParams(id_col="x")
    assert _hash_params(p) == _hash_params(p)


def test_json_ready_with_model() -> None:
    from mosaic.core.dataset import _json_ready
    p = FeatureParams()
    result = _json_ready(p)
    assert isinstance(result, dict)
    assert result["id_col"] == "id"
    # must be JSON-serializable
    json.dumps(result)


# --- Hash stability across dict/model ---


def test_hash_stability_all_converted_features() -> None:
    """Verify that Params model produces the same hash as the equivalent dict."""
    from mosaic.core.dataset import FEATURES, _hash_params
    for name, cls in FEATURES.items():
        params_cls = getattr(cls, "Params", None)
        if params_cls is None:
            continue
        try:
            model = params_cls()
        except ValidationError:
            # Some Params have required fields (e.g. GlobalKMeansClustering.Params.artifact)
            continue
        as_dict = {**model}
        assert _hash_params(model) == _hash_params(as_dict), (
            f"{name}: hash mismatch between model and dict"
        )


# --- Spec models ---


def test_npz_load_spec_requires_key() -> None:
    from mosaic.behavior.feature_library._param_bases import NpzLoadSpec
    with pytest.raises(ValidationError):
        NpzLoadSpec()
    s = NpzLoadSpec(key="templates")
    assert s.kind == "npz"
    assert s.key == "templates"


def test_parquet_load_spec_defaults() -> None:
    from mosaic.behavior.feature_library._param_bases import ParquetLoadSpec
    s = ParquetLoadSpec()
    assert s.kind == "parquet"
    assert s.columns is None
    assert s.numeric_only is True


def test_feature_ref_dict_like_access() -> None:
    from mosaic.behavior.feature_library._param_bases import FeatureRef
    ref = FeatureRef(feature="X")
    assert ref["feature"] == "X"
    assert ref.get("run_id") is None
    assert "feature" in ref


def test_nested_spec_model_dump_roundtrip() -> None:
    from mosaic.behavior.feature_library._param_bases import ArtifactSpec, NpzLoadSpec
    orig = ArtifactSpec(feature="test", load=NpzLoadSpec(key="X"))
    dumped = orig.model_dump()
    restored = ArtifactSpec.model_validate(dumped)
    assert restored == orig


# --- Validation behavior ---


def test_approach_avoidance_literal_validation() -> None:
    from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance
    with pytest.raises(ValidationError):
        ApproachAvoidance.Params(velocity_units="invalid")


def test_pair_egocentric_none_override_rejected() -> None:
    from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures
    with pytest.raises(ValidationError):
        PairEgocentricFeatures.Params.from_overrides({"neck_idx": None})


def test_temporal_stacking_pool_stats_normalization() -> None:
    from mosaic.behavior.feature_library.temporal_stacking import TemporalStackingFeature
    p = TemporalStackingFeature.Params.from_overrides({"pool_stats": "MEAN"})
    assert p.pool_stats == ("mean",)
    p2 = TemporalStackingFeature.Params.from_overrides({"pool_stats": ["Mean", "STD"]})
    assert p2.pool_stats == ("mean", "std")


# --- Deep merge on nested spec models ---


def test_global_ward_partial_artifact_override() -> None:
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering
    p = GlobalWardClustering.Params.from_overrides({"artifact": {"feature": "other"}})
    assert p.artifact.feature == "other"
    assert p.artifact.load.key == "templates"


# --- Mutable default isolation ---


def test_nn_delta_bins_mutable_default_isolation() -> None:
    from mosaic.behavior.feature_library.nn_delta_bins import NearestNeighborDeltaBins
    p1 = NearestNeighborDeltaBins.Params()
    p2 = NearestNeighborDeltaBins.Params()
    assert p1.category_specs is not p2.category_specs


def test_orientation_relative_mutable_default_isolation() -> None:
    from mosaic.behavior.feature_library.orientation_relative import OrientationRelativeFeature
    p1 = OrientationRelativeFeature.Params()
    p2 = OrientationRelativeFeature.Params()
    assert p1.quantiles is not p2.quantiles


def test_global_tsne_mutable_inputs_isolation() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE
    p1 = GlobalTSNE.Params()
    p2 = GlobalTSNE.Params()
    assert p1.inputs is not p2.inputs


# --- _inputs_overridden detection ---


def test_inputs_overridden_default() -> None:
    """No inputs override -> _inputs_overridden is False."""
    from mosaic.behavior.feature_library.ward_assign import WardAssignClustering
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE
    wa = WardAssignClustering()
    gt = GlobalTSNE()
    assert wa._inputs_overridden is False
    assert gt._inputs_overridden is False


def test_inputs_overridden_custom() -> None:
    """Custom inputs -> _inputs_overridden is True."""
    from mosaic.behavior.feature_library.ward_assign import WardAssignClustering
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE
    wa = WardAssignClustering({"inputs": [{"feature": "x", "pattern": "*.npz", "load": {"kind": "npz", "key": "y"}}]})
    gt = GlobalTSNE({"inputs": [{"feature": "x", "pattern": "*.npz", "load": {"kind": "npz", "key": "y"}}]})
    assert wa._inputs_overridden is True
    assert gt._inputs_overridden is True


def test_inputs_overridden_same_as_default() -> None:
    """Inputs identical to default -> _inputs_overridden is False (value equality)."""
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE
    default_inputs_dump = [inp.model_dump() for inp in GlobalTSNE.Params().inputs]
    gt = GlobalTSNE({"inputs": default_inputs_dump})
    assert gt._inputs_overridden is False
