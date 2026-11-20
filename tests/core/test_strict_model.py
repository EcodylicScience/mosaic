"""``terse`` and ``located`` render a validation failure for a person to read."""

from __future__ import annotations

import pytest
from pydantic import ValidationError, field_validator, model_validator

from mosaic.core.strict_model import StrictModel, located, terse


class Inner(StrictModel):
    size: int


class Outer(StrictModel):
    inner: Inner
    name: str = "x"

    @field_validator("name")
    @classmethod
    def _short(cls, value: str) -> str:
        if len(value) > 3:
            msg = "name is too long"
            raise ValueError(msg)
        return value


class Ordered(StrictModel):
    low: int
    high: int

    @model_validator(mode="after")
    def _ordered(self) -> Ordered:
        if self.low > self.high:
            msg = "low exceeds high"
            raise ValueError(msg)
        return self


def failure(model: type[StrictModel], data: dict[str, object]) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        _ = model.model_validate(data)
    return caught.value


def test_located_names_the_dotted_location_of_each_failure() -> None:
    rendered = located(failure(Outer, {"inner": {"size": "big", "extra": 1}}))

    assert "inner.size: Input should be a valid integer" in rendered
    assert "inner.extra: Extra inputs are not permitted" in rendered
    assert "; " in rendered


def test_located_strips_the_value_error_prefix() -> None:
    rendered = located(failure(Outer, {"inner": {"size": 1}, "name": "long name"}))

    assert rendered == "name: name is too long"


def test_located_leaves_a_failure_with_no_location_bare() -> None:
    assert located(failure(Ordered, {"low": 2, "high": 1})) == "low exceeds high"


def test_located_returns_another_exception_as_it_is() -> None:
    assert located(ValueError("plain")) == "plain"


def test_terse_still_drops_the_location() -> None:
    rendered = terse(failure(Outer, {"inner": {"size": "big"}}))

    assert not rendered.startswith("inner")
    assert "Input should be a valid integer" in rendered
