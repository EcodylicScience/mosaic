"""The trained model each op runs, as its ``model_reference`` declares it.

A caller choosing a model by run id reads the declaration, so it is pinned
whole, checked against each op's params, and checked against the training ops it
accepts.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from mosaic.core.pipeline.ops import OPS, ModelReference
from mosaic.tracking.model_refs import ModelReferenceRefusedError

from tests.helpers import make_dataset, minimal_op_params


_MODEL_REFERENCES = {
    "ultralytics": ModelReference(field="model_path", kinds=("train-pose",)),
    "trex": ModelReference(field="detect_model", kinds=("train-pose", "train-points")),
    "sleap": ModelReference(field="model_paths", kinds=("train-sleap",), many=True),
    "litpose": ModelReference(field="model_path", kinds=("train-litpose",)),
    "infer-pose": ModelReference(field="model", kinds=("train-pose",)),
    "infer-points": ModelReference(field="model", kinds=("train-points",)),
    "infer-localizer": ModelReference(field="model", kinds=("train-localizer",)),
}

_DECLARING_OPS = sorted(
    kind for kind, cls in OPS.items() if cls.model_reference is not None
)


def test_every_op_that_runs_a_trained_model_declares_it() -> None:
    """A caller choosing a model by run id reads this, so it is pinned whole."""
    declared = {kind: OPS[kind].model_reference for kind in _DECLARING_OPS}
    assert declared == _MODEL_REFERENCES


@pytest.mark.parametrize("kind", _DECLARING_OPS)
def test_a_declared_model_field_is_a_params_field_of_that_shape(kind: str) -> None:
    declared = OPS[kind].model_reference
    assert declared is not None
    fields = OPS[kind].Params.model_fields
    assert declared.field in fields, f"{kind} declares a field its Params lacks"
    is_list = typing.get_origin(fields[declared.field].annotation) is list
    assert is_list == declared.many, (
        f"{kind}.{declared.field} is {'' if is_list else 'not '}a list, and "
        f"the declaration says many={declared.many}"
    )


@pytest.mark.parametrize("kind", _DECLARING_OPS)
def test_every_declared_kind_is_a_registered_training_op(kind: str) -> None:
    declared = OPS[kind].model_reference
    assert declared is not None
    assert declared.kinds, f"{kind} accepts no training kind at all"
    for model_kind in declared.kinds:
        assert model_kind in OPS, f"{kind} accepts unregistered {model_kind!r}"
        assert OPS[model_kind].category == "train", (
            f"{kind} accepts {model_kind!r}, which trains nothing"
        )


@pytest.mark.parametrize("kind", _DECLARING_OPS)
def test_a_run_id_of_a_kind_the_op_does_not_accept_is_refused_at_plan(
    kind: str, tmp_path: Path
) -> None:
    """The planner refuses what the run would, before any weights are read."""
    declared = OPS[kind].model_reference
    assert declared is not None
    other = next(
        k for k in sorted(OPS) if OPS[k].category == "train" and k not in declared.kinds
    )
    ref = f"{other}.0.1-abcdef0123"
    params = OPS[kind].Params.model_validate(
        {**minimal_op_params(kind), declared.field: [ref] if declared.many else ref}
    )
    ds = make_dataset(tmp_path)

    with pytest.raises(ModelReferenceRefusedError) as caught:
        _ = OPS[kind]().plan_identity(ds, params, ds.resolve_scope(None))

    assert caught.value.reason == "kind_not_accepted"
    assert caught.value.op_kind == kind
    assert caught.value.reference == ref
    assert caught.value.accepted_kinds == declared.kinds


def test_resolve_model_moved_to_model_refs():
    from mosaic.tracking.model_refs import resolve_model

    assert callable(resolve_model)


@pytest.mark.parametrize("kind", _DECLARING_OPS)
def test_a_consumer_can_own_a_trackers_model_and_execution_knobs(kind: str) -> None:
    """A control plane sets the model and every execution knob of a tracker itself.

    mosaic-api removes those fields from a subclass of the op's params and rebuilds
    it. A validator naming a removed field without ``check_fields=False`` refuses
    that rebuild. Every such field must therefore leave the subclass buildable.
    """
    from pydantic import create_model

    from mosaic.core.params import HashExclude
    from mosaic.core.pipeline.ops import OPS

    op = OPS[kind]
    reference = op.model_reference
    assert reference is not None
    owned = {reference.field} | {
        name
        for name, field in op.Params.model_fields.items()
        if any(isinstance(meta, HashExclude) for meta in field.metadata)
    }

    exposed = create_model(f"Exposed{op.Params.__name__}", __base__=op.Params)
    for name in owned:
        _ = exposed.model_fields.pop(name)
    _ = exposed.model_rebuild(force=True)

    assert not owned & set(exposed.model_validate({}).model_dump())
