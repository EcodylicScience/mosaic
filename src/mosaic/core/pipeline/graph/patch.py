"""A recipe changed by a list of edits, without sending it whole again.

A long recipe is changed one step or one param at a time while it is tuned. A
patch names the edits, :func:`patch_recipe` applies them to the expanded
recipe in order and parses the result, and the caller saves it under its new
digest. Four edits:

- ``add_step`` appends a step, whose ``repeat`` expands where it lands.
- ``remove_step`` removes a step. A step that still reads it makes the result
  refuse to parse, naming the reader, unless a later edit removes or changes
  that reader too.
- ``set_param`` sets one param of a step by a dotted path, creating the
  mappings on the way, or removes it when no ``value`` is given.
- ``replace_step`` replaces a step with another under the same id.

A step's scope belongs to a request, not to a recipe, so no edit sets one.

Imports no registry, and the edit models load nothing beyond pydantic, so a
tool contract can declare them. The patched recipe is checked against the step
types by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal

from pydantic import Field, TypeAdapter

from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_str_mapping

from ..error_kind import ClassifiedError, ErrorKind
from .digest import canonical_recipe
from .expand import REPEAT_KEY, RepeatInvalid, expand_repeats
from .model import GraphModel, Recipe

__all__ = [
    "AddStep",
    "PatchInvalid",
    "PatchOp",
    "RemoveStep",
    "ReplaceStep",
    "SetParam",
    "patch_recipe",
]


_JSON_LIST: TypeAdapter[list[JsonValue]] = TypeAdapter(list[JsonValue])
"""Validates the expanded steps back into JSON values."""


class AddStep(GraphModel):
    """Append a step, written as a recipe writes one; ``repeat`` expands."""

    op: Literal["add_step"]
    document: dict[str, JsonValue] = Field(
        description="The step, as a recipe writes it, with an id no step has."
    )


class RemoveStep(GraphModel):
    """Remove a step that no step reads once every edit is applied."""

    op: Literal["remove_step"]
    step: str = Field(description="The id of the step to remove.")


class SetParam(GraphModel):
    """Set one param of a step, or remove it when ``value`` is not given."""

    op: Literal["set_param"]
    step: str = Field(description="The id of the step.")
    path: str = Field(
        min_length=1,
        description="The param, as a dotted path into params, such as 'cv.folds'.",
    )
    value: JsonValue = Field(
        default=None,
        description="The value to set. Left out: the param is removed, to its default.",
    )


class ReplaceStep(GraphModel):
    """Replace a step with another under the same id."""

    op: Literal["replace_step"]
    step: str = Field(description="The id of the step to replace.")
    document: dict[str, JsonValue] = Field(
        description="The new step, as a recipe writes it. Its id, if given, is the same."
    )


type PatchOp = Annotated[
    AddStep | RemoveStep | SetParam | ReplaceStep, Field(discriminator="op")
]
"""One edit of a patch."""


class PatchInvalid(ClassifiedError, ValueError):
    """An edit that cannot be applied to the recipe as the earlier edits left it.

    Attributes:
        index: The edit's position in the patch.
        problem: What is wrong with it.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self, index: int, problem: str) -> None:
        super().__init__(f"ops[{index}]: {problem}")
        self.index: int = index
        self.problem: str = problem

    def error_reason(self) -> str:
        """Return ``patch_invalid``."""
        return "patch_invalid"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the edit's position and the problem."""
        return {"op": self.index, "problem": self.problem}


def patch_recipe(recipe: Recipe, ops: Sequence[PatchOp]) -> Recipe:
    """Return *recipe* with *ops* applied in order.

    Raises:
        PatchInvalid: An edit names a step the recipe does not have, a param
            path that crosses a value which is not a mapping, removes a param
            that is not set, or replaces a step under another id.
        RecipeInvalid: The patched recipe does not parse, for instance because
            a step reads one that was removed.
    """
    # Imported here: validation loads the run machinery, and the edit models
    # are read by the tool contract, which must not.
    from .validate import read_recipe

    document = canonical_recipe(recipe)
    found = document.get("steps")
    steps: list[JsonValue] = found if isinstance(found, list) else []
    for index, op in enumerate(ops):
        try:
            _apply(steps, op)
        except _Fault as fault:
            raise PatchInvalid(index, str(fault)) from fault
    return read_recipe({**document, "steps": steps})


class _Fault(Exception):
    """A fault of one edit, placed by the caller."""


def _apply(steps: list[JsonValue], op: PatchOp) -> None:
    """Apply *op* to *steps* in place."""
    if isinstance(op, AddStep):
        steps.extend(_expanded(op.document))
        return
    position = _position(steps, op.step)
    if isinstance(op, RemoveStep):
        del steps[position]
        return
    if isinstance(op, ReplaceStep):
        replacement = dict(op.document)
        declared = replacement.setdefault("id", op.step)
        if declared != op.step:
            raise _Fault(
                f"the new step's id {declared!r} is not {op.step!r}; remove the "
                f"step and add the new one to rename it"
            )
        if REPEAT_KEY in replacement:
            raise _Fault("a replacing step is one step and takes no repeat")
        steps[position] = _expanded(replacement)[0]
        return
    _set_param(steps[position], op)


def _expanded(step: dict[str, JsonValue]) -> list[JsonValue]:
    """The steps that one step document stands for, ``repeat`` expanded."""
    try:
        expanded = expand_repeats({"steps": [step]})
    except RepeatInvalid as invalid:
        raise _Fault(str(invalid)) from invalid
    # Expansion builds JSON from JSON, so the expanded steps are JSON values.
    found = _JSON_LIST.validate_python(
        expanded.get("steps") if is_str_mapping(expanded) else [step]
    )
    return found


def _position(steps: list[JsonValue], step_id: str) -> int:
    """Where the step *step_id* is in *steps*."""
    for position, step in enumerate(steps):
        if isinstance(step, dict) and step.get("id") == step_id:
            return position
    declared = [str(step.get("id")) for step in steps if isinstance(step, dict)]
    raise _Fault(f"no step {step_id!r} in the recipe. Declared: {declared}")


def _set_param(step: JsonValue, op: SetParam) -> None:
    """Set or remove the param at ``op.path`` of *step*."""
    if not isinstance(step, dict):
        raise _Fault(f"step {op.step!r} is not an object")
    parts = op.path.split(".")
    if not all(parts):
        raise _Fault(f"the path {op.path!r} has an empty part")
    setting = "value" in op.model_fields_set
    at = step.setdefault("params", {})
    for depth, part in enumerate(parts[:-1]):
        if not isinstance(at, dict):
            break
        below = at.get(part)
        if below is None and setting:
            below = at[part] = {}
        if below is None:
            raise _Fault(f"params.{op.path} of step {op.step!r} is not set")
        if not isinstance(below, dict):
            crossed = ".".join(parts[: depth + 1])
            raise _Fault(f"params.{crossed} of step {op.step!r} is not a mapping")
        at = below
    if not isinstance(at, dict):
        raise _Fault(f"params of step {op.step!r} is not a mapping")
    last = parts[-1]
    if setting:
        at[last] = op.value
    elif last in at:
        del at[last]
    else:
        raise _Fault(f"params.{op.path} of step {op.step!r} is not set")
