"""Tests for Pydantic feature parameter models."""
from __future__ import annotations

import json
import pytest
from pydantic import BaseModel as _PydanticBaseModel
from pydantic import Field, ValidationError

from mosaic.behavior.feature_library._param_bases import (
    ColumnConfig,
    FeatureParams,
    InterpolationConfig,
    PositionColumns,
    SamplingConfig,
)


# --- FeatureParams base ---


def test_defaults() -> None:
    p = FeatureParams()
    assert p.columns.id_col == "id"
    assert p.columns.seq_col == "sequence"
    assert p.columns.group_col == "group"
    assert p.columns.order_pref == ("frame", "time")


def test_getitem() -> None:
    p = FeatureParams()
    assert p["columns"] == ColumnConfig()
    with pytest.raises(KeyError):
        p["nonexistent"]


def test_get_with_default() -> None:
    p = FeatureParams()
    assert p.get("columns") == ColumnConfig()
    assert p.get("nonexistent", "fallback") == "fallback"
    assert p.get("nonexistent") is None


def test_contains() -> None:
    p = FeatureParams()
    assert "columns" in p
    assert "id_col" not in p
    assert "nonexistent" not in p


def test_keys() -> None:
    p = FeatureParams()
    assert set(p.keys()) == {"columns"}


def test_dict_spread() -> None:
    p = FeatureParams()
    d = {**p}
    assert "columns" in d
    assert isinstance(d["columns"], ColumnConfig)


def test_dict_spread_with_extra_key() -> None:
    p = FeatureParams()
    d = {**p, "_scope": None}
    assert "_scope" in d
    assert "columns" in d


def test_from_overrides_empty() -> None:
    p = FeatureParams.from_overrides(None)
    assert p.columns.id_col == "id"
    p2 = FeatureParams.from_overrides({})
    assert p2.columns.id_col == "id"


def test_from_overrides_rejects_none_for_str_field() -> None:
    with pytest.raises(ValidationError):
        FeatureParams.from_overrides(
            {"columns": {"id_col": None, "seq_col": "seq"}}
        )


def test_from_overrides_applies_values() -> None:
    p = FeatureParams.from_overrides({"columns": {"id_col": "animal_id"}})
    assert p.columns.id_col == "animal_id"


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


# --- Composition ---


class _ComposedParams(FeatureParams):
    position: PositionColumns = Field(default_factory=PositionColumns)
    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


def test_group_constraints() -> None:
    with pytest.raises(ValidationError):
        _ComposedParams(interpolation=InterpolationConfig(linear_interp_limit=0))
    with pytest.raises(ValidationError):
        _ComposedParams(interpolation=InterpolationConfig(max_missing_fraction=1.5))
    with pytest.raises(ValidationError):
        _ComposedParams(sampling=SamplingConfig(fps_default=-1.0))


def test_group_keys_in_spread() -> None:
    p = _ComposedParams()
    d = {**p}
    assert "position" in d
    assert "sampling" in d
    assert "interpolation" in d
    assert "columns" in d


# --- Subclass override ---


class _OverrideParams(FeatureParams):
    columns: ColumnConfig = Field(
        default_factory=lambda: ColumnConfig(group_col="event")
    )


def test_subclass_can_override_base_default() -> None:
    p = _OverrideParams()
    assert p.columns.group_col == "event"


# --- dataset.py integration ---


def test_hash_params_with_model() -> None:
    from mosaic.core.dataset import _hash_params
    p = FeatureParams()
    d = p.model_dump()
    assert _hash_params(p) == _hash_params(d)


def test_hash_params_deterministic() -> None:
    from mosaic.core.dataset import _hash_params
    p = FeatureParams(columns=ColumnConfig(id_col="x"))
    assert _hash_params(p) == _hash_params(p)


def test_json_ready_with_model() -> None:
    from mosaic.core.dataset import _json_ready
    p = FeatureParams()
    result = _json_ready(p)
    assert isinstance(result, dict)
    assert isinstance(result["columns"], dict)
    assert result["columns"]["id_col"] == "id"
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


# --- Composition merge tests ---


def test_from_overrides_partial_group_merge() -> None:
    """Partial override of a group model merges with defaults."""
    p = _ComposedParams.from_overrides(
        {"position": {"x_col": "X#wcentroid"}}
    )
    assert p.position.x_col == "X#wcentroid"
    assert p.position.y_col == "Y"
    assert p.position.angle_col == "ANGLE"


def test_from_overrides_nested_group_merge() -> None:
    """Partial override of ColumnConfig merges with defaults."""
    p = FeatureParams.from_overrides(
        {"columns": {"id_col": "animal_id"}}
    )
    assert p.columns.id_col == "animal_id"
    assert p.columns.seq_col == "sequence"


def test_per_feature_default_override() -> None:
    """Feature can override individual group defaults via lambda factory."""

    class _CustomParams(FeatureParams):
        position: PositionColumns = Field(
            default_factory=lambda: PositionColumns(x_col="X#wcentroid")
        )

    p = _CustomParams()
    assert p.position.x_col == "X#wcentroid"
    assert p.position.y_col == "Y"

    # User can still override at runtime
    p2 = _CustomParams.from_overrides({"position": {"x_col": "custom"}})
    assert p2.position.x_col == "custom"
