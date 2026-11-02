"""A recipe's ``repeat`` items, expanded into the items they stand for.

A recipe that tries several seeds, classes or folds repeats one step with one
value changed. ``repeat`` writes it once::

    {"id": "xgb_s{seed}", "type": "feature", "feature": "xgboost",
     "repeat": {"seed": [1, 2, 3]}, "params": {"random_state": "{seed}"}}

stands for three steps, ``xgb_s1``, ``xgb_s2`` and ``xgb_s3``. An item of a
step's ``inputs`` expands the same way, so an ensemble reads every seed with one
input, ``{"step": "xgb_s{seed}", "repeat": {"seed": [1, 2, 3]}}``.

The rules:

- ``repeat`` and ``zip`` are read on a step and on an item of a step's
  ``inputs``, nowhere else, so a param that happens to be named ``repeat`` is
  left as written.
- A ``repeat`` maps names to non-empty lists of values. Several names expand to
  their cartesian product, the first name varying slowest, and the items take
  the place of the one that declared them, in that order.
- A ``zip``, written beside ``repeat`` or alone, binds its names in lockstep: the
  lists must have the same length, and the n-th value of each is taken
  together, as in ``"zip": {"behavior": ["attack", "mount"], "threshold":
  [0.3, 0.6]}``, which is two steps. Beside a ``repeat`` it is one more axis of
  the product, varying fastest.
- ``{name}`` is replaced in every string of the item, keys excepted. A string
  that is ``{name}`` alone becomes the value itself, typed as written, and a
  value inside a longer string must be a string, a number or a boolean. Braces
  around any other word are left as they are. An id is always text.
- A step's id names every one of its repeat's names, and at least one of its
  zip's, so each expanded step has an id of its own.
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
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

from mosaic.core.narrowing import is_object_list, is_str_mapping

__all__ = [
    "MAX_EXPANDED_ITEMS",
    "REPEAT_KEY",
    "ZIP_KEY",
    "RepeatInvalid",
    "expand_repeats",
]

REPEAT_KEY: Final = "repeat"
"""The key of a step or an input that stands for several."""

ZIP_KEY: Final = "zip"
"""The key that binds names in lockstep, beside or instead of :data:`REPEAT_KEY`."""

_EXPANDING_KEYS: Final = (REPEAT_KEY, ZIP_KEY)
"""The keys a step or an input expands by, and that expansion removes."""

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
    return _expands(step) or (
        is_object_list(inputs)
        and any(is_str_mapping(item) and _expands(item) for item in inputs)
    )


def _expands(item: Mapping[str, object]) -> bool:
    """Whether *item* carries ``repeat`` or ``zip``."""
    return any(key in item for key in _EXPANDING_KEYS)


def _without_expanding_keys(item: Mapping[str, object]) -> dict[str, object]:
    """*item* as it reads once expanded."""
    return {key: value for key, value in item.items() if key not in _EXPANDING_KEYS}


def _expand_step(step: object, budget: _Budget) -> list[object]:
    """The steps that one declared step stands for."""
    if not is_str_mapping(step) or not _repeats(step):
        return [step]
    if not _expands(step):
        return [_bound_step(step, {}, budget)]
    axes = _axes(step, {}, "")
    body = _without_expanding_keys(step)
    _check_id(body.get("id"), axes)
    budget.spend(axes.size, axes.where)
    return [_bound_step(body, bindings, budget) for bindings in axes.bindings()]


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
    """A step's inputs with each ``repeat`` or ``zip`` item expanded in its place."""
    expanded: list[object] = []
    for position, item in enumerate(inputs):
        where = f"inputs[{position}]"
        if not (is_str_mapping(item) and _expands(item)):
            expanded.append(_substitute_value(item, bindings, where))
            continue
        axes = _axes(item, bindings, f"{where}.")
        budget.spend(axes.size, axes.where)
        body = _without_expanding_keys(item)
        expanded.extend(
            _substitute_value(body, {**bindings, **chosen}, where)
            for chosen in axes.bindings()
        )
    return expanded


@dataclass(frozen=True)
class _Axes:
    """What one ``repeat`` and ``zip`` bind: independent axes of value tuples.

    A ``repeat`` name is an axis of its own and a ``zip`` is one axis for all its
    names, so the items are the product of the axes.
    """

    names: tuple[str, ...]
    """Every bound name, in the order the axes list their values."""

    axes: tuple[tuple[tuple[object, ...], ...], ...]
    where: str
    """The place of the first of ``repeat`` and ``zip`` the item carries."""

    zipped: tuple[str, ...] = ()
    """The names bound in lockstep."""

    zip_where: str = ZIP_KEY
    """The place of the ``zip``."""

    @property
    def size(self) -> int:
        """How many items the axes make."""
        return math.prod(len(axis) for axis in self.axes)

    def bindings(self) -> Iterator[dict[str, object]]:
        """Each item's name to value, the first axis varying slowest."""
        for combination in itertools.product(*self.axes):
            values = tuple(itertools.chain.from_iterable(combination))
            yield dict(zip(self.names, values, strict=True))


def _axes(
    item: Mapping[str, object], bindings: Mapping[str, object], prefix: str
) -> _Axes:
    """The axes of the ``repeat`` and ``zip`` of *item*, outer values substituted.

    *prefix* leads the place of a fault: ``""`` on a step, ``inputs[i].`` on an
    input.
    """
    names: list[str] = []
    axes: list[tuple[tuple[object, ...], ...]] = []
    zipped: list[str] = []
    for key in _EXPANDING_KEYS:
        if key not in item:
            continue
        where = f"{prefix}{key}"
        columns = _columns(key, item[key], bindings, names, where)
        names.extend(name for name, _ in columns)
        if key == ZIP_KEY:
            lengths = {len(values) for _, values in columns}
            if len(lengths) > 1:
                raise _Fault(
                    where,
                    f"zip takes lists of the same length, not "
                    f"{', '.join(str(len(values)) for _, values in columns)}",
                )
            zipped = [name for name, _ in columns]
            axes.append(tuple(zip(*(values for _, values in columns), strict=True)))
        else:
            axes.extend(tuple((value,) for value in values) for _, values in columns)
    first = next(key for key in _EXPANDING_KEYS if key in item)
    return _Axes(
        tuple(names),
        tuple(axes),
        f"{prefix}{first}",
        tuple(zipped),
        f"{prefix}{ZIP_KEY}",
    )


def _columns(
    key: str,
    declared: object,
    bindings: Mapping[str, object],
    taken: list[str],
    where: str,
) -> list[tuple[str, list[object]]]:
    """The names of a ``repeat`` or ``zip`` and the values each takes."""
    if not is_str_mapping(declared) or not declared:
        raise _Fault(
            where, f'{key} maps each name to a list of values, as {{"seed": [1, 2]}}'
        )
    columns: list[tuple[str, list[object]]] = []
    for name, options in declared.items():
        if not _NAME.fullmatch(name):
            raise _Fault(where, f"{key} name {name!r} is not an identifier")
        if name in bindings or name in taken:
            raise _Fault(where, f"{key} name {name!r} is already repeated by its step")
        if not is_object_list(options) or not options:
            raise _Fault(where, f"{key} {name!r} takes a non-empty list of values")
        columns.append(
            (name, [_substitute_value(option, bindings, where) for option in options])
        )
    return columns


def _check_id(declared: object, axes: _Axes) -> None:
    """Refuse a repeated step whose id would be the same for two of its items.

    The id names every ``repeat`` name and at least one ``zip`` name.
    """
    if not isinstance(declared, str):
        raise _Fault("id", "a repeated step needs an id naming each repeat name")
    used = set(_PLACEHOLDER.findall(declared))
    missing = [
        name for name in axes.names if name not in used and name not in axes.zipped
    ]
    if missing:
        named = ", ".join(f"{{{name}}}" for name in missing)
        raise _Fault(
            axes.where,
            f"the id {declared!r} must name {named}, so each repeated step has "
            f"an id of its own",
        )
    if axes.zipped and not used.intersection(axes.zipped):
        named = " or ".join(f"{{{name}}}" for name in axes.zipped)
        raise _Fault(
            axes.zip_where,
            f"the id {declared!r} must name {named}, so each zipped step has "
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
