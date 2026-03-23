"""Tests for Pydantic feature parameter models."""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel as _PydanticBaseModel
from pydantic import Field, ValidationError

from mosaic.behavior.feature_library.spec import (
    COLUMNS,
    Inputs,
    InterpolationConfig,
    Params,
    Result,
    SamplingConfig,
    TrackInput,
    resolve_order_col,
)

# --- COLUMNS global ---


def test_columns_defaults() -> None:
    assert COLUMNS.id_col == "id"
    assert COLUMNS.seq_col == "sequence"
    assert COLUMNS.group_col == "group"
    assert COLUMNS.frame_col == "frame"
    assert COLUMNS.time_col == "time"
    assert COLUMNS.order_by == "frames"
    assert COLUMNS.x_col == "X"
    assert COLUMNS.y_col == "Y"
    assert COLUMNS.orientation_col == "ANGLE"


def test_resolve_order_col_frames_first() -> None:
    import pandas as pd

    df = pd.DataFrame({"frame": [1, 2], "time": [0.0, 0.1]})
    assert resolve_order_col(df) == "frame"


def test_resolve_order_col_missing_both() -> None:
    import pandas as pd

    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Need"):
        resolve_order_col(df)


# --- Params base ---


def test_params_empty_by_default() -> None:
    p = Params()
    assert set(p.keys()) == set()


def test_params_getitem() -> None:
    p = Params()
    with pytest.raises(KeyError):
        p["nonexistent"]


def test_params_get_with_default() -> None:
    p = Params()
    assert p.get("nonexistent", "fallback") == "fallback"
    assert p.get("nonexistent") is None


def test_params_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        Params(bogus="x")


def test_from_overrides_empty() -> None:
    p = Params.from_overrides(None)
    p2 = Params.from_overrides({})
    assert p == p2


class _InnerModel(_PydanticBaseModel):
    a: int = 1
    b: int = 2


class _ParamsWithNested(Params):
    nested: _InnerModel = Field(default_factory=_InnerModel)


def test_from_overrides_partial_basemodel_merge() -> None:
    p = _ParamsWithNested.from_overrides({"nested": {"a": 99}})
    assert p.nested.a == 99
    assert p.nested.b == 2


def test_from_overrides_full_basemodel_override() -> None:
    p = _ParamsWithNested.from_overrides({"nested": {"a": 10, "b": 20}})
    assert p.nested.a == 10
    assert p.nested.b == 20


# --- Composition ---


class _ComposedParams(Params):
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
    assert "sampling" in d
    assert "interpolation" in d


# --- dataset.py integration ---


def test_hash_params_with_model() -> None:
    from mosaic.core.pipeline._utils import hash_params as _hash_params

    p = Params()
    d = p.model_dump()
    assert _hash_params(p) == _hash_params(d)


def test_hash_params_deterministic() -> None:
    from mosaic.core.pipeline._utils import hash_params as _hash_params

    p = _ComposedParams()
    assert _hash_params(p) == _hash_params(p)


def test_json_ready_with_model() -> None:
    from mosaic.core.pipeline._utils import json_ready as _json_ready

    p = Params()
    result = _json_ready(p)
    assert isinstance(result, dict)
    json.dumps(result)


# --- Hash stability across dict/model ---


def test_hash_stability_all_converted_features() -> None:
    """Verify that Params model produces the same hash as the equivalent dict."""
    from mosaic.behavior.feature_library.spec import FEATURES
    from mosaic.core.pipeline._utils import hash_params as _hash_params

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
    from mosaic.behavior.feature_library.spec import NpzLoadSpec

    with pytest.raises(ValidationError):
        NpzLoadSpec()
    s = NpzLoadSpec(key="templates")
    assert s.kind == "npz"
    assert s.key == "templates"


def test_parquet_load_spec_defaults() -> None:
    from mosaic.behavior.feature_library.spec import ParquetLoadSpec

    s = ParquetLoadSpec()
    assert s.kind == "parquet"
    assert s.columns is None
    assert s.numeric_only is True


def test_artifact_spec_dict_like_access() -> None:
    from mosaic.behavior.feature_library.spec import ArtifactSpec, NpzLoadSpec

    spec = ArtifactSpec(feature="X", load=NpzLoadSpec(key="Y"))
    assert spec["feature"] == "X"
    assert spec.get("run_id") is None
    assert "feature" in spec


def test_nested_spec_model_dump_roundtrip() -> None:
    from mosaic.behavior.feature_library.spec import ArtifactSpec, NpzLoadSpec

    orig = ArtifactSpec(feature="test", load=NpzLoadSpec(key="X"))
    dumped = orig.model_dump()
    restored = ArtifactSpec.model_validate(dumped)
    assert restored == orig


def test_joblib_load_spec() -> None:
    from mosaic.behavior.feature_library.spec import JoblibLoadSpec

    s = JoblibLoadSpec()
    assert s.kind == "joblib"
    assert s.key is None
    s2 = JoblibLoadSpec(key="scaler")
    assert s2.key == "scaler"


def test_result_use_latest() -> None:
    r = Result(feature="nn", run_id="v1")
    latest = r.use_latest()
    assert latest.run_id is None
    assert r.run_id == "v1"


def test_nn_result_defaults() -> None:
    from mosaic.behavior.feature_library.spec import NNResult

    nn = NNResult()
    assert nn.feature == "nearest-neighbor"


def test_nn_result_rejects_wrong_feature() -> None:
    from mosaic.behavior.feature_library.spec import NNResult

    with pytest.raises(ValidationError):
        NNResult(feature="wrong")


def test_artifact_spec_auto_pattern() -> None:
    from mosaic.behavior.feature_library.spec import (
        ArtifactSpec,
        JoblibLoadSpec,
        NpzLoadSpec,
    )

    a = ArtifactSpec(feature="x", load=NpzLoadSpec(key="y"))
    assert a.pattern == "*.npz"
    b = ArtifactSpec(feature="x", load=JoblibLoadSpec())
    assert b.pattern == "*.joblib"


def test_artifact_spec_pattern_kind_mismatch() -> None:
    from mosaic.behavior.feature_library.spec import ArtifactSpec, NpzLoadSpec

    with pytest.raises(ValidationError):
        ArtifactSpec(feature="x", load=NpzLoadSpec(key="y"), pattern="foo.joblib")


def test_artifact_from_result() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    r = Result(feature="global-tsne", run_id="v1")
    art = GlobalTSNE.TemplatesArtifact.from_result(r)
    assert art.run_id == "v1"
    assert art.pattern == "global_templates_features.npz"


def test_artifact_from_result_wrong_feature() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    with pytest.raises(ValueError, match="expects feature"):
        GlobalTSNE.TemplatesArtifact.from_result(Result(feature="wrong"))


def test_artifact_from_result_chained() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    r = Result(feature="global-tsne", run_id="v1")
    art = GlobalTSNE.TemplatesArtifact.from_result(r.use_latest())
    assert art.run_id is None


def test_artifact_from_result_suffixed_feature() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    r = Result(
        feature="global-tsne__from__pair-wavelet__from__pair-egocentric",
        run_id="v2",
    )
    art = GlobalTSNE.TemplatesArtifact.from_result(r)
    assert art.feature == r.feature
    assert art.run_id == "v2"
    assert art.pattern == "global_templates_features.npz"


def test_pair_filter_on_params() -> None:
    from mosaic.behavior.feature_library.ward_assign import WardAssignClustering

    inputs = WardAssignClustering.Inputs((Result(feature="pair-wavelet"),))
    wa = WardAssignClustering(inputs=inputs)
    assert wa.params.pair_filter is None
    wa2 = WardAssignClustering(
        inputs=inputs,
        params={"pair_filter": {"feature": "nearest-neighbor"}},
    )
    assert wa2.params.pair_filter.feature == "nearest-neighbor"


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
    from mosaic.behavior.feature_library.temporal_stacking import (
        TemporalStackingFeature,
    )

    p = TemporalStackingFeature.Params.from_overrides({"pool_stats": "MEAN"})
    assert p.pool_stats == ("mean",)
    p2 = TemporalStackingFeature.Params.from_overrides({"pool_stats": ["Mean", "STD"]})
    assert p2.pool_stats == ("mean", "std")


# --- Deep merge on nested spec models ---


def test_global_ward_partial_artifact_override() -> None:
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering

    p = GlobalWardClustering.Params.from_overrides({"templates": {"feature": "other"}})
    assert p.templates.feature == "other"
    assert p.templates.load.key == "templates"


# --- Mutable default isolation ---


def test_nn_delta_bins_mutable_default_isolation() -> None:
    from mosaic.behavior.feature_library.nn_delta_bins import NearestNeighborDeltaBins

    p1 = NearestNeighborDeltaBins.Params()
    p2 = NearestNeighborDeltaBins.Params()
    assert p1.category_specs is not p2.category_specs


def test_orientation_relative_mutable_default_isolation() -> None:
    from mosaic.behavior.feature_library.orientation_relative import (
        OrientationRelativeFeature,
    )

    p1 = OrientationRelativeFeature.Params()
    p2 = OrientationRelativeFeature.Params()
    assert p1.quantiles is not p2.quantiles


# --- Result-based inputs (WardAssign, TemporalStacking, GlobalWard, GlobalKMeans) ---


def test_ward_assign_requires_inputs() -> None:
    """WardAssign constructor requires explicit inputs (no default)."""
    from mosaic.behavior.feature_library.ward_assign import WardAssignClustering

    with pytest.raises(TypeError):
        WardAssignClustering()


def test_temporal_stacking_requires_inputs() -> None:
    """TemporalStacking constructor requires explicit inputs (no default)."""
    from mosaic.behavior.feature_library.temporal_stacking import (
        TemporalStackingFeature,
    )

    with pytest.raises(TypeError):
        TemporalStackingFeature()


def test_global_ward_default_inputs() -> None:
    """GlobalWard defaults to empty inputs (artifact mode)."""
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering

    gw = GlobalWardClustering()
    assert len(gw.inputs.root) == 0
    assert gw.inputs.feature_inputs == ()


def test_global_kmeans_no_assign_dict() -> None:
    """GlobalKMeans no longer has assign dict; assign controlled via inputs."""
    from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering

    gk = GlobalKMeansClustering(
        params={
            "templates": {
                "feature": "global-tsne",
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates"},
            }
        }
    )
    assert not hasattr(gk.params, "assign")
    assert gk.inputs.feature_inputs == ()

    # With Result inputs -> assign mode
    inputs = GlobalKMeansClustering.Inputs(
        (
            Result(feature="pair-wavelet"),
            Result(feature="pair-ego-wavelet"),
        )
    )
    gk2 = GlobalKMeansClustering(
        inputs=inputs,
        params={
            "templates": {
                "feature": "global-tsne",
                "pattern": "global_templates_features.npz",
                "load": {"kind": "npz", "key": "templates"},
            }
        },
    )
    assert len(gk2.inputs.feature_inputs) == 2


# --- GlobalTSNE Result-based inputs ---


def test_global_tsne_requires_inputs() -> None:
    """GlobalTSNE constructor requires explicit inputs (no default)."""
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    with pytest.raises(TypeError):
        GlobalTSNE()


def test_global_tsne_result_inputs() -> None:
    """GlobalTSNE accepts Result-based inputs and computes correct storage_suffix."""
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    inputs = GlobalTSNE.Inputs(
        (
            Result(feature="pair-wavelet", run_id="0.1-abc"),
            Result(feature="pair-ego-wavelet"),
        )
    )
    gt = GlobalTSNE(inputs=inputs)
    assert gt.inputs.feature_inputs[0].feature == "pair-wavelet"
    assert gt.inputs.feature_inputs[0].run_id == "0.1-abc"
    assert gt.inputs.storage_suffix() == "pair-wavelet+pair-ego-wavelet"


# --- Inputs ---


def test_fi_tracks_only() -> None:
    i = Inputs(("tracks",))
    assert i.is_single_tracks
    assert not i.is_single_feature
    assert not i.is_multi
    assert i.has_tracks


def test_fi_single_feature() -> None:
    i = Inputs((Result(feature="speed-angvel"),))
    assert i.is_single_feature
    assert not i.is_single_tracks
    assert not i.is_multi
    assert not i.has_tracks


def test_result_str_delegates_to_repr() -> None:
    r = Result(feature="speed-angvel", run_id="0.1-abc")
    assert str(r) == repr(r)
    assert "Result(" in str(r)
    assert "speed-angvel" in str(r)


def test_fi_single_feature_with_run_id() -> None:
    i = Inputs((Result(feature="speed-angvel", run_id="v1"),))
    assert i.is_single_feature
    assert i.feature_inputs[0].run_id == "v1"


def test_fi_multi() -> None:
    i = Inputs(("tracks", Result(feature="nn")))
    assert i.is_multi
    assert not i.is_single_tracks
    assert not i.is_single_feature
    assert i.has_tracks
    assert len(i.feature_inputs) == 1


def test_fi_empty_raises() -> None:
    with pytest.raises(ValidationError):
        Inputs(())


def test_fi_empty_still_rejected_by_default() -> None:
    """Subclasses with default _require='nonempty' still reject empty inputs."""

    class StrictInputs(Inputs[Result]):
        pass

    with pytest.raises(ValidationError):
        StrictInputs(())


def test_fi_empty_allowed_with_opt_in() -> None:
    """Subclasses with _require='any' accept both empty and non-empty inputs."""
    from typing import ClassVar

    class EmptyOkInputs(Inputs[Result]):
        _require: ClassVar[str] = "any"

    i = EmptyOkInputs(())
    assert len(i.root) == 0
    assert not i.has_tracks
    assert i.feature_inputs == ()
    assert i.storage_suffix() is None


def test_fi_duplicate_raises() -> None:
    with pytest.raises(ValidationError):
        Inputs((Result(feature="nn"), Result(feature="nn")))


def test_fi_feature_inputs_property() -> None:
    i = Inputs(("tracks", Result(feature="a"), Result(feature="b")))
    feats = i.feature_inputs
    assert len(feats) == 2
    assert feats[0].feature == "a"
    assert feats[1].feature == "b"


def test_fi_storage_suffix_tracks_only() -> None:
    assert Inputs(("tracks",)).storage_suffix() is None


def test_fi_storage_suffix_single_feature() -> None:
    i = Inputs((Result(feature="speed-angvel"),))
    assert i.storage_suffix() == "speed-angvel"


def test_fi_storage_suffix_multi() -> None:
    i = Inputs((Result(feature="a"), Result(feature="b")))
    assert i.storage_suffix() == "a+b"


def test_fi_narrowed_accepts_correct() -> None:
    Narrowed = Inputs[Result[Literal["nn"]]]
    i = Narrowed((Result(feature="nn"),))
    assert i.is_single_feature


def test_fi_narrowed_rejects_wrong() -> None:
    Narrowed = Inputs[Result[Literal["nn"]]]
    with pytest.raises(ValidationError):
        Narrowed((Result(feature="wrong"),))


def test_fi_narrowed_tracks() -> None:
    Narrowed = Inputs[TrackInput]
    i = Narrowed(("tracks",))
    assert i.is_single_tracks


def test_fi_narrowed_with_run_id() -> None:
    Narrowed = Inputs[Result[Literal["nn"]]]
    i = Narrowed((Result(feature="nn", run_id="v1"),))
    assert i.feature_inputs[0].run_id == "v1"


def test_fi_roundtrip_tracks() -> None:
    orig = Inputs(("tracks",))
    dumped = orig.model_dump()
    restored = Inputs.model_validate(dumped)
    assert restored.root == orig.root


def test_fi_roundtrip_feature() -> None:
    orig = Inputs((Result(feature="nn", run_id="v1"),))
    dumped = orig.model_dump()
    restored = Inputs.model_validate(dumped)
    assert restored.root == orig.root
