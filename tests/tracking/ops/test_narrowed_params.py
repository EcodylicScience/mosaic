"""The params model of every registered op, narrowed one field at a time.

A consumer of the toolkit builds a subclass of an op's params model that lacks
the fields the consumer owns, then completes the payload before the worker
validates it with the full model ("Validators on a narrowed model" in the
parameter-ownership spec). Every validator of the model must therefore give
the full model's answer, or no verdict, on a subclass that lacks any one field.

A field validator that names an absent field makes ``model_rebuild`` raise
``PydanticUserError``, so each one declares ``check_fields=False``. That
disables pydantic's misspelling check, which
``test_every_field_validator_names_a_declared_field`` restores.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from mosaic.core.pipeline.ops import OPS
from tests.helpers import narrowed_params, rule_reaching_op_params

RULE_REACHING_CASES = [
    pytest.param(kind, case, id=f"{kind}-{case}")
    for kind in sorted(OPS)
    for case in rule_reaching_op_params(kind)
]

NARROWED_CASES = [
    pytest.param(kind, case, field, id=f"{kind}-{case}-{field}")
    for kind in sorted(OPS)
    for case in rule_reaching_op_params(kind)
    for field in OPS[kind].Params.model_fields
]


def _undeclared_validator_fields(model: type[BaseModel]) -> list[str]:
    """The field names that a field validator of *model* names and *model* lacks."""
    return [
        name
        for decorator in model.__pydantic_decorators__.field_validators.values()
        for name in decorator.info.fields
        if name not in model.model_fields
    ]


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        pytest.param(kind, field, id=f"{kind}-{field}")
        for kind in sorted(OPS)
        for field in OPS[kind].Params.model_fields
    ],
)
def test_every_field_can_be_narrowed_out(kind: str, field: str):
    _ = narrowed_params(OPS[kind].Params, {field})


@pytest.mark.parametrize("kind", sorted(OPS))
def test_every_field_validator_names_a_declared_field(kind: str):
    assert _undeclared_validator_fields(OPS[kind].Params) == []


def test_the_check_finds_a_misnamed_field_validator():
    class Misnamed(BaseModel):
        present: int = 0

        @field_validator("absent", check_fields=False)
        @classmethod
        def _accept(cls, value: object) -> object:
            return value

    assert _undeclared_validator_fields(Misnamed) == ["absent"]


@pytest.mark.parametrize(("kind", "case"), RULE_REACHING_CASES)
def test_every_rule_reaching_payload_validates_on_the_full_model(kind: str, case: str):
    _ = OPS[kind].Params.model_validate(rule_reaching_op_params(kind)[case])


@pytest.mark.parametrize(("kind", "case", "field"), NARROWED_CASES)
def test_a_narrowed_model_answers_or_refuses(kind: str, case: str, field: str):
    """Any exception but ``ValidationError`` fails the test."""
    payload = rule_reaching_op_params(kind)[case]
    _ = payload.pop(field, None)
    narrowed = narrowed_params(OPS[kind].Params, {field})
    try:
        _ = narrowed.model_validate(payload)
    except ValidationError:
        pass
