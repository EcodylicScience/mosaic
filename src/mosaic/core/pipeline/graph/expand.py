"""A recipe's ``repeat`` items, expanded into the items they stand for.

A recipe that tries several seeds, classes or folds repeats one step with one
value changed. ``repeat`` writes it once::

    {"id": "xgb_s{seed}", "type": "feature", "feature": "xgboost",
     "repeat": {"seed": [1, 2, 3]}, "params": {"random_state": "{seed}"}}

stands for three steps, ``xgb_s1``, ``xgb_s2`` and ``xgb_s3``. An item of a
step's ``inputs`` expands the same way, so an ensemble reads every seed with one
input, ``{"step": "xgb_s{seed}", "repeat": {"seed": [1, 2, 3]}}``.

The rules:

- ``repeat`` is read on a step and on an item of a step's ``inputs``, nowhere
  else, so a param that happens to be named ``repeat`` is left as written.
- A ``repeat`` maps names to non-empty lists of values. Several names expand to
  their cartesian product, the first name varying slowest, and the items take
  the place of the one that declared them, in that order.
- ``{name}`` is replaced in every string of the item, keys excepted. A string
  that is ``{name}`` alone becomes the value itself, typed as written, and a
  value inside a longer string must be a string, a number or a boolean. Braces
  around any other word are left as they are. An id is always text.
- A step's id names every one of its repeat's names, so each expanded step has
  an id of its own.
- An input of a repeated step may repeat again, over other names, and sees the
  step's values.
- A recipe expands to at most :data:`MAX_EXPANDED_ITEMS` steps and inputs.

Expansion happens when a recipe is parsed (``Recipe``'s ``before`` validator),
so the canonical form, its digest, the saved file and every run id are those of
the steps written out in full. A recipe without ``repeat`` is returned as it is.

Imports the standard library and :mod:`mosaic.core.narrowing` only.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from collections.abc import Mapping
from typing import Final

from mosaic.core.narrowing import is_object_list, is_str_mapping

__all__ = ["MAX_EXPANDED_ITEMS", "REPEAT_KEY", "RepeatInvalid", "expand_repeats"]

REPEAT_KEY: Final = "repeat"
"""The key of a step or an input that stands for several."""

MAX_EXPANDED_ITEMS: Final = 10_000
"""The most steps and inputs, together, that a recipe's repeats expand to."""

_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""A repeat name: an identifier."""

_PLACEHOLDER: Final = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
"""``{name}`` inside a string."""


class RepeatInvalid(ValueError):
    """A ``repeat`` that cannot be expanded.

    Attributes:
        step: The declared step it is in, as its id was written, or ``""``
            when the step has no id that is a string.
        where: The place of the fault in that step, such as ``repeat``,
            ``inputs[1].repeat`` or ``params.note``, led by ``steps[i].`` when
            the step has no id.
        message: What is wrong.
    """

    def __init__(self, step: str, where: str, message: str) -> None:
        at = f"step {step!r} {where}" if step else where
        super().__init__(f"{at}: {message}")
        self.step: str = step
        self.where: str = where
        self.message: str = message


class _Fault(Exception):
    """A fault inside one declared step, at *where* in it."""

    def __init__(self, where: str, message: str) -> None:
        super().__init__(message)
        self.where: str = where


class _Budget:
    """The steps and inputs a recipe may still expand to."""

    def __init__(self) -> None:
        self.left: int = MAX_EXPANDED_ITEMS

    def spend(self, count: int, where: str) -> None:
        self.left -= count
        if self.left < 0:
            raise _Fault(
                where,
                f"the recipe's repeats expand to more than {MAX_EXPANDED_ITEMS} "
                f"steps and inputs",
            )


def expand_repeats(document: object) -> object:
    """Return *document* with every ``repeat`` step and input expanded.

    Anything that is not a recipe object, and a recipe that holds no
    ``repeat`` on a step or an input, is returned unchanged, so the parser
    reports what is wrong with it.

    Raises:
        RepeatInvalid: A ``repeat`` is malformed, a step's id does not name each
            of its names, a placeholder takes a value it cannot hold, or the
            recipe expands to more than :data:`MAX_EXPANDED_ITEMS` steps and
            inputs.
    """
    if not is_str_mapping(document):
        return document
    steps = document.get("steps")
    if not is_object_list(steps) or not any(_repeats(step) for step in steps):
        return document
    budget = _Budget()
    expanded: list[object] = []
    for index, step in enumerate(steps):
        try:
            expanded.extend(_expand_step(step, budget))
        except _Fault as fault:
            declared = step.get("id") if is_str_mapping(step) else None
            name = declared if isinstance(declared, str) else ""
            where = fault.where if name else f"steps[{index}].{fault.where}"
            raise RepeatInvalid(name, where, str(fault)) from fault
    return {**document, "steps": expanded}


def _repeats(step: object) -> bool:
    """Whether *step*, or an item of its ``inputs``, carries ``repeat``."""
    if not is_str_mapping(step):
        return False
    inputs = step.get("inputs")
    return REPEAT_KEY in step or (
        is_object_list(inputs)
        and any(is_str_mapping(item) and REPEAT_KEY in item for item in inputs)
    )


def _expand_step(step: object, budget: _Budget) -> list[object]:
    """The steps that one declared step stands for."""
    if not is_str_mapping(step) or not _repeats(step):
        return [step]
    if REPEAT_KEY not in step:
        return [_bound_step(step, {}, budget)]
    names, values = _axes(step[REPEAT_KEY], {}, REPEAT_KEY)
    body = {key: value for key, value in step.items() if key != REPEAT_KEY}
    _check_id(body.get("id"), names)
    combinations = _product_size(values)
    budget.spend(combinations, REPEAT_KEY)
    return [
        _bound_step(body, dict(zip(names, chosen, strict=True)), budget)
        for chosen in itertools.product(*values)
    ]


def _bound_step(
    body: Mapping[str, object], bindings: Mapping[str, object], budget: _Budget
) -> dict[str, object]:
    """One step with *bindings* substituted and its ``inputs`` expanded."""
    bound: dict[str, object] = {}
    for key, value in body.items():
        if key == "id" and isinstance(value, str):
            bound[key] = _substitute_text(value, bindings, key)
        elif key == "inputs" and is_object_list(value):
            bound[key] = _expand_inputs(value, bindings, budget)
        else:
            bound[key] = _substitute_value(value, bindings, key)
    return bound


def _expand_inputs(
    inputs: list[object], bindings: Mapping[str, object], budget: _Budget
) -> list[object]:
    """A step's inputs with each ``repeat`` item expanded in its place."""
    expanded: list[object] = []
    for position, item in enumerate(inputs):
        where = f"inputs[{position}]"
        if not (is_str_mapping(item) and REPEAT_KEY in item):
            expanded.append(_substitute_value(item, bindings, where))
            continue
        names, values = _axes(item[REPEAT_KEY], bindings, f"{where}.{REPEAT_KEY}")
        body = {key: value for key, value in item.items() if key != REPEAT_KEY}
        budget.spend(_product_size(values), f"{where}.{REPEAT_KEY}")
        expanded.extend(
            _substitute_value(
                body, {**bindings, **dict(zip(names, chosen, strict=True))}, where
            )
            for chosen in itertools.product(*values)
        )
    return expanded


def _axes(
    declared: object, bindings: Mapping[str, object], where: str
) -> tuple[list[str], list[list[object]]]:
    """The names of a ``repeat`` and the values each takes, outer values substituted."""
    if not is_str_mapping(declared) or not declared:
        raise _Fault(
            where, 'repeat maps each name to a list of values, as {"seed": [1, 2]}'
        )
    names: list[str] = []
    values: list[list[object]] = []
    for name, options in declared.items():
        if not _NAME.fullmatch(name):
            raise _Fault(where, f"repeat name {name!r} is not an identifier")
        if name in bindings:
            raise _Fault(where, f"repeat name {name!r} is already repeated by its step")
        if not is_object_list(options) or not options:
            raise _Fault(where, f"repeat {name!r} takes a non-empty list of values")
        names.append(name)
        values.append(
            [_substitute_value(option, bindings, where) for option in options]
        )
    return names, values


def _product_size(values: list[list[object]]) -> int:
    """How many combinations the lists of *values* make."""
    size = 1
    for options in values:
        size *= len(options)
    return size


def _check_id(declared: object, names: list[str]) -> None:
    """Refuse a repeated step whose id would be the same for two of its items."""
    if not isinstance(declared, str):
        raise _Fault("id", "a repeated step needs an id naming each repeat name")
    used = set(_PLACEHOLDER.findall(declared))
    missing = [name for name in names if name not in used]
    if missing:
        named = ", ".join(f"{{{name}}}" for name in missing)
        raise _Fault(
            REPEAT_KEY,
            f"the id {declared!r} must name {named}, so each repeated step has "
            f"an id of its own",
        )


def _substitute_value(
    value: object, bindings: Mapping[str, object], where: str
) -> object:
    """*value* with each bound ``{name}`` replaced, at any depth."""
    if not bindings:
        return value
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value)
        if whole is not None and whole.group(1) in bindings:
            return copy.deepcopy(bindings[whole.group(1)])
        return _substitute_text(value, bindings, where)
    if is_object_list(value):
        return [
            _substitute_value(item, bindings, f"{where}[{position}]")
            for position, item in enumerate(value)
        ]
    if is_str_mapping(value):
        return {
            key: _substitute_value(item, bindings, f"{where}.{key}")
            for key, item in value.items()
        }
    return value


def _substitute_text(text: str, bindings: Mapping[str, object], where: str) -> str:
    """*text* with each bound ``{name}`` replaced by its value as text."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in bindings:
            return match.group(0)
        return _as_text(name, bindings[name], where)

    return _PLACEHOLDER.sub(replace, text)


def _as_text(name: str, value: object, where: str) -> str:
    """A repeat value as it reads inside a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float):
        return json.dumps(value)
    raise _Fault(
        where,
        f"{{{name}}} is part of a string here, so its value must be a string, a "
        f"number or a boolean, not {json.dumps(value)}",
    )
