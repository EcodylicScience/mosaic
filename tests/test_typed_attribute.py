"""The typed-attribute grammar: six types, their constraints, their values.

These cases are the contract mosaic and mosaic-api both have to keep. The same
grammar is implemented in mosaic-api's ``db/queries/typed_attribute.py`` for the
tags a user attaches to a sequence or an individual, and here for the tags that
describe a dataset. A tag that validates in one place and not the other would
mean a manifest the control plane cannot read back, so the interesting tests are
the disagreements a careless port would introduce: bool-versus-int, NaN, an
option list with a duplicate, a constraint key that belongs to another type.
"""

from __future__ import annotations

import math

import pytest

from mosaic.core.typed_attribute import (
    TypedAttributeType,
    TypedAttributeValue,
    validate_constraints,
    validate_typed_value,
)

ALL_TYPES: tuple[TypedAttributeType, ...] = (
    "label",
    "text",
    "int",
    "float",
    "bool",
    "categorical",
)


class TestUnknownType:
    def test_an_unknown_type_is_refused_by_both_validators(self) -> None:
        with pytest.raises(ValueError, match="Unknown typed-attribute type"):
            validate_constraints("duration", {})  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="Unknown typed-attribute type"):
            validate_typed_value("duration", {}, 1)  # pyright: ignore[reportArgumentType]

    def test_the_message_lists_what_was_expected(self) -> None:
        with pytest.raises(ValueError, match="categorical"):
            validate_constraints("enum", {})  # pyright: ignore[reportArgumentType]


class TestConstraintsAreWellFormed:
    @pytest.mark.parametrize("type_", ["label", "bool"])
    def test_a_valueless_type_takes_no_constraints(
        self, type_: TypedAttributeType
    ) -> None:
        validate_constraints(type_, {})
        with pytest.raises(ValueError, match="has no constraints"):
            validate_constraints(type_, {"min": 1})

    def test_text_takes_only_a_positive_max_length(self) -> None:
        validate_constraints("text", {})
        validate_constraints("text", {"max_length": 200})
        with pytest.raises(ValueError, match="unknown keys"):
            validate_constraints("text", {"min": 1})
        for bad in (0, -1, 1.5, True, "10", None):
            with pytest.raises(ValueError, match="max_length must be a positive"):
                validate_constraints("text", {"max_length": bad})

    def test_int_bounds_must_be_real_integers_in_order(self) -> None:
        validate_constraints("int", {})
        validate_constraints("int", {"min": 1})
        validate_constraints("int", {"min": 1, "max": 1})
        with pytest.raises(ValueError, match="unknown keys"):
            validate_constraints("int", {"options": ["a"]})
        with pytest.raises(ValueError, match="min must be <= max"):
            validate_constraints("int", {"min": 5, "max": 4})
        for bad in (1.5, "3", None):
            with pytest.raises(ValueError, match="must be an integer"):
                validate_constraints("int", {"min": bad})

    def test_an_int_bound_rejects_a_bool(self) -> None:
        # isinstance(True, int) is true, so a port that forgets the guard turns
        # `min: true` into `min: 1` and silently narrows the attribute.
        with pytest.raises(ValueError, match="no bools"):
            validate_constraints("int", {"min": True})

    def test_float_bounds_must_be_finite_numbers_in_order(self) -> None:
        validate_constraints("float", {"min": -1.5, "max": 2})
        with pytest.raises(ValueError, match="min must be <= max"):
            validate_constraints("float", {"min": 2.0, "max": 1.0})
        with pytest.raises(ValueError, match="must not be NaN"):
            validate_constraints("float", {"min": math.nan})
        with pytest.raises(ValueError, match="must not be inf"):
            validate_constraints("float", {"max": math.inf})
        with pytest.raises(ValueError, match="no bools"):
            validate_constraints("float", {"max": False})

    def test_categorical_requires_a_non_empty_list_of_unique_strings(self) -> None:
        validate_constraints("categorical", {"options": ["a", "b"]})
        with pytest.raises(ValueError, match="options is required"):
            validate_constraints("categorical", {})
        with pytest.raises(ValueError, match="must be a list"):
            validate_constraints("categorical", {"options": "a"})
        with pytest.raises(ValueError, match="must be non-empty"):
            validate_constraints("categorical", {"options": []})
        with pytest.raises(ValueError, match="must be strings"):
            validate_constraints("categorical", {"options": ["a", 2]})
        with pytest.raises(ValueError, match="duplicate option"):
            validate_constraints("categorical", {"options": ["a", "a"]})
        with pytest.raises(ValueError, match="unknown keys"):
            validate_constraints("categorical", {"options": ["a"], "min": 1})

    def test_an_option_is_bounded_in_length(self) -> None:
        validate_constraints("categorical", {"options": ["x" * 64]})
        with pytest.raises(ValueError, match="length in"):
            validate_constraints("categorical", {"options": [""]})
        with pytest.raises(ValueError, match="length in"):
            validate_constraints("categorical", {"options": ["x" * 65]})

    def test_the_option_list_is_capped(self) -> None:
        validate_constraints("categorical", {"options": [str(i) for i in range(1024)]})
        with pytest.raises(ValueError, match="at most 1024"):
            validate_constraints(
                "categorical", {"options": [str(i) for i in range(1025)]}
            )


class TestValuesSatisfyConstraints:
    def test_a_label_carries_no_value(self) -> None:
        validate_typed_value("label", {}, None)
        for bad in ("", "x", 0, False):
            with pytest.raises(ValueError, match="label value must be null"):
                validate_typed_value("label", {}, bad)

    def test_text_is_a_string_within_max_length(self) -> None:
        validate_typed_value("text", {}, "anything at all")
        validate_typed_value("text", {"max_length": 3}, "abc")
        with pytest.raises(ValueError, match="exceeds max_length 3"):
            validate_typed_value("text", {"max_length": 3}, "abcd")
        for bad in (1, None, True):
            with pytest.raises(ValueError, match="text value must be a string"):
                validate_typed_value("text", {}, bad)

    def test_an_int_value_is_range_checked_at_both_bounds(self) -> None:
        bounds = {"min": 1, "max": 10}
        validate_typed_value("int", bounds, 1)
        validate_typed_value("int", bounds, 10)
        with pytest.raises(ValueError, match="below min 1"):
            validate_typed_value("int", bounds, 0)
        with pytest.raises(ValueError, match="above max 10"):
            validate_typed_value("int", bounds, 11)

    def test_an_int_value_is_neither_a_bool_nor_a_float(self) -> None:
        for bad in (True, False, 3.0, "3", None):
            with pytest.raises(ValueError, match="must be an integer"):
                validate_typed_value("int", {}, bad)

    def test_a_float_value_is_range_checked_and_finite(self) -> None:
        bounds = {"min": 0.0, "max": 1.0}
        validate_typed_value("float", bounds, 0.0)
        validate_typed_value("float", bounds, 1)
        with pytest.raises(ValueError, match="below min"):
            validate_typed_value("float", bounds, -0.1)
        with pytest.raises(ValueError, match="above max"):
            validate_typed_value("float", bounds, 1.1)
        with pytest.raises(ValueError, match="must not be NaN"):
            validate_typed_value("float", {}, math.nan)
        with pytest.raises(ValueError, match="must not be inf"):
            validate_typed_value("float", {}, math.inf)
        with pytest.raises(ValueError, match="must be a number"):
            validate_typed_value("float", {}, True)

    def test_a_bool_value_is_exactly_true_or_false(self) -> None:
        validate_typed_value("bool", {}, True)
        validate_typed_value("bool", {}, False)
        for bad in (0, 1, "true", None):
            with pytest.raises(ValueError, match="exactly True or False"):
                validate_typed_value("bool", {}, bad)

    def test_a_categorical_value_is_drawn_from_its_options(self) -> None:
        options = {"options": ["red", "green"]}
        validate_typed_value("categorical", options, "red")
        with pytest.raises(ValueError, match="not in options"):
            validate_typed_value("categorical", options, "blue")
        with pytest.raises(ValueError, match="must be a string"):
            validate_typed_value("categorical", options, 1)


class TestTheTwoValidatorsStaySeparate:
    def test_a_value_check_does_not_re_validate_its_constraints(self) -> None:
        """A malformed constraint must not be reported against a value.

        The value validator trusts what it is handed, so a caller that skipped
        the constraint check gets a wrong answer rather than a misattributed
        error. Here ``max`` is nonsense for text, and the value passes anyway --
        which is the documented division of labor, not a hole: the constraint
        would have been refused where it was declared.
        """
        validate_typed_value("text", {"max": 1}, "far longer than one")

    @pytest.mark.parametrize("type_", ALL_TYPES)
    def test_every_type_has_a_valid_empty_or_minimal_constraint_form(
        self, type_: TypedAttributeType
    ) -> None:
        constraints: dict[str, TypedAttributeValue | list[str]] = (
            {"options": ["only"]} if type_ == "categorical" else {}
        )
        validate_constraints(type_, constraints)
