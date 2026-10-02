"""``Params``, the base of every parameter model, and how a model is hashed.

Overrides, composition from config groups, the hash of a model against the hash
of its dump, and the fields ``HASH_EXCLUDE`` keeps out of a run's identity.
"""

from __future__ import annotations

import json
from typing import Annotated

import pytest
from pydantic import BaseModel as _PydanticBaseModel
from pydantic import Field, ValidationError

from mosaic.behavior.feature_library.types import InterpolationConfig, SamplingConfig
from mosaic.core.params import HASH_EXCLUDE, Params
from tests.helpers import run_id_digest


# --- Params base ---


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


# --- Hashing and serializing a model ---


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
    from mosaic.behavior.feature_library.registry import FEATURES
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
        as_dict = model.model_dump()
        assert _hash_params(model) == _hash_params(as_dict), (
            f"{name}: hash mismatch between model and dict"
        )


# --- HASH_EXCLUDE / identity_dump (run_id hash exclusion) ---


class _ThroughputParams(Params):
    """Params with a hash-excluded throughput knob alongside a real field."""

    real_field: int = 1
    batch_size: Annotated[int, HASH_EXCLUDE] = 4


def test_identity_dump_drops_marked_field_but_model_dump_keeps_it() -> None:
    p = _ThroughputParams(batch_size=8)
    # Excluded from the identity dump (run_id hash input)...
    assert "batch_size" not in p.identity_dump()
    assert "real_field" in p.identity_dump()
    # ...but still present in the full dump (params.json + worker propagation).
    assert p.model_dump()["batch_size"] == 8


def test_identity_dump_equals_model_dump_when_nothing_marked() -> None:
    """Backward-compat: unmarked Params hash exactly as before."""

    class _Plain(Params):
        a: int = 1
        b: str = "x"

    assert _Plain().identity_dump() == _Plain().model_dump()


def test_hash_excluded_field_does_not_change_run_id() -> None:
    assert run_id_digest(_ThroughputParams(batch_size=4)) == run_id_digest(
        _ThroughputParams(batch_size=8)
    )


def test_real_field_still_changes_run_id() -> None:
    assert run_id_digest(_ThroughputParams(real_field=1)) != run_id_digest(
        _ThroughputParams(real_field=2)
    )
