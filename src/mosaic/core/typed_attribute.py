"""The typed-attribute grammar: six types, their constraints, and their values.

A typed attribute is a name carrying a declared ``type``, a ``type_constraints``
object whose legal keys depend on that type, and a ``value`` that must satisfy
both. mosaic-api already speaks this grammar for the tags a user attaches to a
sequence or an individual, and for the modifiers on a scored behavior event. The
dataset manifest speaks it for the tags that describe a dataset as a whole, so a
tag means the same thing wherever it is written.

Deliberately stdlib-only and free of dataset concepts, for the same reason
:mod:`mosaic.core.stored_paths` is: this is the *grammar*, while which attributes
exist and where they are stored belongs to whatever holds them. It is also why a
manifest read does not import pandas.

**Constraints and values are validated separately, and in that order.**
:func:`validate_constraints` answers "is this a well-formed constraint object for
this type", :func:`validate_typed_value` answers "does this value satisfy those
constraints". The second does not re-check the first -- a caller validates the
constraints once, when they are declared, and values as often as they change.

The grammar:

===========  ==============================================  =========================
type         constraints                                     value
===========  ==============================================  =========================
label        none allowed                                    must be ``None``
text         ``max_length`` (positive int)                    ``str``, length-checked
int          ``min``, ``max`` (ints, ``min <= max``)          ``int``, range-checked
float        ``min``, ``max`` (finite numbers)                number, range-checked
bool         none allowed                                    exactly ``True``/``False``
categorical  ``options`` (non-empty unique strings)           ``str`` drawn from them
===========  ==============================================  =========================

``label`` is presence-only: the attribute either is attached or is not, and a
value would be a second, contradictory claim.

Neither ``int`` nor ``bool`` accepts the other, even though ``isinstance(True,
int)`` is true in Python. A ``bool`` reaching an ``int`` attribute is a caller
mistake worth hearing about, not a zero or a one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Literal

from mosaic.core.json_value import JsonValue

__all__ = [
    "TypedAttributeType",
    "TypedAttributeValue",
    "validate_constraints",
    "validate_typed_value",
]


TypedAttributeType = Literal["label", "text", "int", "float", "bool", "categorical"]
"""The six declared types an attribute may carry."""

TypedAttributeValue = str | int | float | bool | None
"""Every value a typed attribute can hold, across all six types."""


_VALID_TYPES: Final[frozenset[str]] = frozenset(
    ("label", "text", "int", "float", "bool", "categorical")
)

# Option strings follow the same naming bounds as a user-visible attribute name.
_OPTION_MIN_LENGTH: Final = 1
_OPTION_MAX_LENGTH: Final = 64
# Cap on ``len(options)``. The validator walks the list more than once per call,
# so an unbounded one is a denial-of-service surface; 1024 leaves generous room
# for any realistic vocabulary.
_OPTIONS_MAX_COUNT: Final = 1024


def _ensure_known_type(type_: str) -> None:
    if type_ not in _VALID_TYPES:
        valid = sorted(_VALID_TYPES)
        msg = f"Unknown typed-attribute type {type_!r}; expected one of {valid}"
        raise ValueError(msg)


def _as_real_int(value: JsonValue) -> int | None:
    """*value* as an ``int``, or ``None`` when it is not one.

    A ``bool`` is not one. ``isinstance(True, int)`` is true, so without the
    explicit guard every ``int`` bound would silently accept ``True`` as 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_real_number(value: JsonValue) -> float | None:
    """*value* as a ``float``, or ``None`` when it is not a number.

    Ints widen to float so a downstream comparison has one type to reason about.
    A ``bool`` is excluded for the reason :func:`_as_real_int` gives.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _check_finite_float(name: str, value: float) -> None:
    if math.isnan(value):
        msg = f"{name} must not be NaN"
        raise ValueError(msg)
    if math.isinf(value):
        msg = f"{name} must not be inf or -inf"
        raise ValueError(msg)


def _validate_text_constraints(constraints: Mapping[str, JsonValue]) -> None:
    extra = set(constraints) - {"max_length"}
    if extra:
        msg = f"text constraints: unknown keys {sorted(extra)}"
        raise ValueError(msg)
    if "max_length" in constraints:
        max_length = _as_real_int(constraints["max_length"])
        if max_length is None or max_length < 1:
            msg = "text constraints: max_length must be a positive integer"
            raise ValueError(msg)


def _validate_int_constraints(constraints: Mapping[str, JsonValue]) -> None:
    extra = set(constraints) - {"min", "max"}
    if extra:
        msg = f"int constraints: unknown keys {sorted(extra)}"
        raise ValueError(msg)
    bounds: dict[str, int] = {}
    for key in ("min", "max"):
        if key in constraints:
            value = _as_real_int(constraints[key])
            if value is None:
                msg = (
                    f"int constraints: {key} must be an integer "
                    "(no bools, no fractional values)"
                )
                raise ValueError(msg)
            bounds[key] = value
    if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
        msg = "int constraints: min must be <= max"
        raise ValueError(msg)


def _validate_float_constraints(constraints: Mapping[str, JsonValue]) -> None:
    extra = set(constraints) - {"min", "max"}
    if extra:
        msg = f"float constraints: unknown keys {sorted(extra)}"
        raise ValueError(msg)
    bounds: dict[str, float] = {}
    for key in ("min", "max"):
        if key in constraints:
            value = _as_real_number(constraints[key])
            if value is None:
                msg = f"float constraints: {key} must be a number (no bools)"
                raise ValueError(msg)
            _check_finite_float(f"float constraints: {key}", value)
            bounds[key] = value
    if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
        msg = "float constraints: min must be <= max"
        raise ValueError(msg)


def _validate_categorical_constraints(constraints: Mapping[str, JsonValue]) -> None:
    extra = set(constraints) - {"options"}
    if extra:
        msg = f"categorical constraints: unknown keys {sorted(extra)}"
        raise ValueError(msg)
    if "options" not in constraints:
        msg = "categorical constraints: options is required"
        raise ValueError(msg)
    options = constraints["options"]
    if not isinstance(options, list):
        msg = "categorical constraints: options must be a list"
        raise ValueError(msg)
    if len(options) < 1:
        msg = "categorical constraints: options must be non-empty"
        raise ValueError(msg)
    if len(options) > _OPTIONS_MAX_COUNT:
        msg = (
            "categorical constraints: options must have at most "
            f"{_OPTIONS_MAX_COUNT} entries"
        )
        raise ValueError(msg)
    seen: set[str] = set()
    for option in options:
        if not isinstance(option, str):
            msg = "categorical constraints: options must be strings"
            raise ValueError(msg)
        if not _OPTION_MIN_LENGTH <= len(option) <= _OPTION_MAX_LENGTH:
            msg = (
                "categorical constraints: each option must have length in "
                f"[{_OPTION_MIN_LENGTH}, {_OPTION_MAX_LENGTH}]"
            )
            raise ValueError(msg)
        if option in seen:
            msg = f"categorical constraints: duplicate option {option!r}"
            raise ValueError(msg)
        seen.add(option)


def validate_constraints(
    type_: TypedAttributeType, constraints: Mapping[str, JsonValue]
) -> None:
    """Check that *constraints* is well formed for *type_*.

    Args:
        type_: The declared attribute type.
        constraints: The constraint object to check. An empty mapping is legal
            for every type except ``categorical``, which requires ``options``.

    Raises:
        ValueError: If *type_* is unknown, an unexpected key is present, or a
            bound is malformed. The message names the offending key.
    """
    _ensure_known_type(type_)

    if type_ in ("label", "bool"):
        if constraints:
            keys = sorted(constraints)
            msg = f"type {type_!r} has no constraints; got keys {keys}"
            raise ValueError(msg)
        return
    if type_ == "text":
        _validate_text_constraints(constraints)
        return
    if type_ == "int":
        _validate_int_constraints(constraints)
        return
    if type_ == "float":
        _validate_float_constraints(constraints)
        return
    _validate_categorical_constraints(constraints)


def validate_typed_value(
    type_: TypedAttributeType,
    constraints: Mapping[str, JsonValue],
    value: TypedAttributeValue,
) -> None:
    """Check that *value* satisfies *constraints* under *type_*.

    *constraints* is trusted, not re-checked -- run :func:`validate_constraints`
    over it once where it is declared. Checking it again on every value would
    turn a malformed constraint into an error blamed on whatever value arrived
    next.

    Args:
        type_: The declared attribute type.
        constraints: Already-validated constraints for *type_*.
        value: The value to check.

    Raises:
        ValueError: If *value* has the wrong Python type for *type_* or falls
            outside the constraints. The message says which.
    """
    _ensure_known_type(type_)

    if type_ == "label":
        if value is not None:
            msg = "label value must be null"
            raise ValueError(msg)
        return

    if type_ == "text":
        if not isinstance(value, str):
            msg = "text value must be a string"
            raise ValueError(msg)
        max_length = _as_real_int(constraints.get("max_length"))
        if max_length is not None and len(value) > max_length:
            msg = f"text value exceeds max_length {max_length}"
            raise ValueError(msg)
        return

    if type_ == "int":
        as_int = _as_real_int(value)
        if as_int is None:
            msg = "int value must be an integer (not bool, not float)"
            raise ValueError(msg)
        low = _as_real_int(constraints.get("min"))
        high = _as_real_int(constraints.get("max"))
        if low is not None and as_int < low:
            msg = f"int value {as_int} below min {low} (out of range)"
            raise ValueError(msg)
        if high is not None and as_int > high:
            msg = f"int value {as_int} above max {high} (out of range)"
            raise ValueError(msg)
        return

    if type_ == "float":
        as_float = _as_real_number(value)
        if as_float is None:
            msg = "float value must be a number (not bool)"
            raise ValueError(msg)
        _check_finite_float("float value", as_float)
        low_f = _as_real_number(constraints.get("min"))
        high_f = _as_real_number(constraints.get("max"))
        if low_f is not None and as_float < low_f:
            msg = f"float value {as_float} below min {low_f} (out of range)"
            raise ValueError(msg)
        if high_f is not None and as_float > high_f:
            msg = f"float value {as_float} above max {high_f} (out of range)"
            raise ValueError(msg)
        return

    if type_ == "bool":
        if not isinstance(value, bool):
            msg = "bool value must be exactly True or False"
            raise ValueError(msg)
        return

    if not isinstance(value, str):
        msg = "categorical value must be a string"
        raise ValueError(msg)
    options = constraints.get("options")
    if not isinstance(options, list) or value not in options:
        msg = f"categorical value {value!r} not in options"
        raise ValueError(msg)
