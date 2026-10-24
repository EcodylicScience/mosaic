"""How far in time one row of a step's output reads, and whether a CV gap covers it.

A feature that smooths, stacks or convolves over frames writes a row that depends on
the frames around it. A classifier cross-validated by blocks of frames holds a block
out of training and drops ``cv_gap_frames`` frames on each side of it. A training
frame just outside the gap still reads the held-out frames through every window
above the classifier unless the gap is at least as wide as the longest of them. This
module holds the two halves of that question. Both are pure: they read the facts a
plan already resolved and touch no dataset.

**The declaration** (:func:`feature_reach`). A feature may declare
``temporal_reach``: the frames on each side that one output row depends on, as a
class attribute or as a function of its validated params. ``0`` is a feature that
computes within a frame, and ``None`` is a reach no params can bound, such as an
interpolation over gaps of any length or a bidirectional recurrent network. A feature
that declares nothing is *undeclared*, which is a different answer from ``None``: the
author has not said, and the check cautions on it the same way.

**The check** (:func:`reach_findings`). The reach above a classifier is summed along a
path of the recipe's data edges and is the maximum over paths. Rows are taken to be
consecutive frames. The classifier's own reach is not part of it: a fold model
trains on pieces cut out of the runs, held-out blocks and gap removed, so its own head
never reads a held-out frame while it trains, and only the steps above it can carry
one in. A classifier planned for block cross-validation with a ``cv_gap_frames`` below
that reach is a problem when every step above it declares a reach. When some do not,
it is a caution, unless an explicit gap already covers the reach that is known: the
user has then chosen a gap with the unknown in view, and nothing more can be asked of
the plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from .compatibility import Reach
from .topo import edges
from .wire import Problem

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .model import Recipe
    from .plan import PlannedStep

__all__ = [
    "UNDECLARED",
    "UNKNOWN",
    "Reach",
    "ReachFindings",
    "class_reach",
    "feature_reach",
    "reach_findings",
    "recipe_reach",
]

UNKNOWN: Final = "unknown"
"""A declared reach that no params bound: the feature says it cannot say."""

UNDECLARED: Final = "undeclared"
"""A feature that declares no ``temporal_reach`` at all."""

_MISSING: Final = object()


def _as_reach(value: object) -> Reach:
    """A declared value as a :data:`Reach`: a non-negative int, else unknown."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return UNKNOWN


def _read(cls: type, params: object) -> Reach:
    """The reach *cls* declares under *params*, :data:`UNDECLARED` for none."""
    declared: object = getattr(cls, "temporal_reach", _MISSING)
    if declared is _MISSING:
        return UNDECLARED
    if callable(declared):
        try:
            return _as_reach(declared(params))
        except Exception:
            # A function that cannot read these params cannot bound them.
            return UNKNOWN
    return _as_reach(declared)


def feature_reach(feature: object) -> Reach:
    """The reach of a built feature, under its own validated params."""
    return _read(type(feature), getattr(feature, "params", None))


def class_reach(cls: type) -> Reach:
    """The reach of a feature class under its default params.

    This is what the catalog reports. A reach that depends on params reads as it
    does for a step that sets none, and a plan reads the step's own params instead.
    """
    if not callable(getattr(cls, "temporal_reach", None)):
        return _read(cls, None)
    params_cls: object = getattr(cls, "Params", None)
    build: object = getattr(params_cls, "from_overrides", None)
    if not callable(build):
        return UNKNOWN
    try:
        params: object = build(None)
    except Exception:
        # A reach that depends on params, and params with no default to read it by.
        return UNKNOWN
    return _read(cls, params)


def recipe_reach(cls: type, params: Mapping[str, object]) -> Reach:
    """The reach of a feature class under the params a recipe step writes.

    For a step whose feature cannot be built yet, because an identity above it is
    unresolved. A field holding a step reference is left out, so the rest is read
    as the step wrote it: a window in ``half`` counts, where the default window
    would not. Params that do not validate without the reference are built unchecked
    from what the step wrote, and a reach that cannot be read from those is unknown.
    """
    if not callable(getattr(cls, "temporal_reach", None)):
        return _read(cls, None)
    plain: dict[str, object] = {
        name: value
        for name, value in params.items()
        if not (isinstance(value, Mapping) and "step" in value)
    }
    params_cls: object = getattr(cls, "Params", None)
    build: object = getattr(params_cls, "from_overrides", None)
    if callable(build):
        try:
            return _read(cls, build(plain))
        except Exception:
            # Invalid beside the reference it lacks: read the fields as written.
            pass
    unchecked: object = getattr(params_cls, "model_construct", None)
    if callable(unchecked):
        try:
            return _read(cls, unchecked(**plain))
        except Exception:
            return UNKNOWN
    return UNKNOWN


@dataclass(frozen=True, slots=True)
class ReachFindings:
    """What the gap check found in one plan.

    Attributes:
        problems: A classifier whose gap is shorter than a reach every step above
            it declares.
        cautions: A classifier with a step above it whose reach is unknown or
            undeclared, and whose gap is zero or shorter than the reach known, so
            that nothing shows the gap was chosen with that in view.
    """

    problems: tuple[Problem, ...] = ()
    cautions: tuple[Problem, ...] = ()


@dataclass(frozen=True, slots=True)
class _Above:
    """What reaches one step's output: the longest known path, and what is unbounded."""

    frames: int = 0
    """The most frames on each side, summed along the longest path of known reach."""
    path: tuple[tuple[str, int], ...] = ()
    """The steps of that path with a reach above 0, with each one's reach."""
    unknown: frozenset[str] = frozenset()
    """The steps above, this one included, that declare a reach no params bound."""
    undeclared: frozenset[str] = frozenset()
    """The steps above, this one included, that declare no reach."""


def _through(parents: Iterable[_Above], step_id: str, own: Reach) -> _Above:
    """The :class:`_Above` of a step, from its parents' and its own reach."""
    best = _Above()
    unknown: frozenset[str] = frozenset()
    undeclared: frozenset[str] = frozenset()
    for parent in parents:
        if parent.frames > best.frames:
            best = parent
        unknown |= parent.unknown
        undeclared |= parent.undeclared
    if own == UNKNOWN:
        unknown |= {step_id}
    elif own == UNDECLARED:
        undeclared |= {step_id}
    if isinstance(own, int) and own > 0:
        return _Above(
            best.frames + own, (*best.path, (step_id, own)), unknown, undeclared
        )
    return _Above(best.frames, best.path, unknown, undeclared)


def _block_gap(planned: PlannedStep) -> int | None:
    """The ``cv_gap_frames`` of a step that fits under block cross-validation.

    ``None`` for any other step, which is not checked: a step bound to a run or
    given a model fits nothing, and the sequence unit holds out whole sequences.
    """
    if planned.kind != "feature" or not planned.fits:
        return None
    params = planned.spec.params
    if params.get("cv_unit") != "block":
        return None
    return _frames(params.get("cv_gap_frames", 0))


def _frames(value: object) -> int:
    """A ``cv_gap_frames`` as the params model reads it, 0 for what it would refuse.

    The recipe holds the raw value, and the params model accepts a whole number
    written as a string or a float (``"11"``, ``11.0``), so a gap read as an
    integer alone would count as 0 and ask for a gap already given.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else 0
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _chain(path: Sequence[tuple[str, int]]) -> str:
    """The steps of a path with their reach, for a message."""
    return ", ".join(f"{step!r} {reach}" for step, reach in path)


def _named(steps: Iterable[str]) -> str:
    return ", ".join(repr(step) for step in sorted(steps))


def reach_findings(
    recipe: Recipe,
    steps: Sequence[PlannedStep],
    *,
    skip: Iterable[str] = (),
) -> ReachFindings:
    """Check the gap of each classifier planned for block cross-validation.

    Args:
        recipe: The recipe, whose data edges (``inputs``, ``tracks`` and params
            references, not ``after``) carry the reach upward.
        steps: The planned steps, in topological order. A feature step's
            ``temporal_reach`` is read off them, and an op passes the reach above it
            through unchanged: it writes the recording's tracks or media, and no
            window of it spans the training and held-out frames.
        skip: Step ids that are not executed, such as the ones bound to a run.

    Returns:
        A problem for each checked step whose ``cv_gap_frames`` is below the reach
        above it when every step above it declares one, and a caution for each
        whose reach is not fully known and whose gap is zero or below the reach
        that is known. A positive gap that covers the known reach silences the
        caution. The step's own reach is not counted.
    """
    skipped = frozenset(skip)
    producers: dict[str, list[str]] = {}
    for edge in edges(recipe):
        if edge.site != "after":
            producers.setdefault(edge.consumer, []).append(edge.producer)
    above: dict[str, _Above] = {}
    problems: list[Problem] = []
    cautions: list[Problem] = []
    for planned in steps:
        parents = [above[parent] for parent in producers.get(planned.step_id, ())]
        own: Reach = planned.temporal_reach if planned.kind == "feature" else 0
        above[planned.step_id] = _through(parents, planned.step_id, own)
        gap = None if planned.step_id in skipped else _block_gap(planned)
        if gap is None:
            continue
        # Without the step's own reach: only what is above it can leak.
        reading = _through(parents, planned.step_id, 0)
        problem = _gap_finding(planned, reading, gap)
        if problem is None:
            continue
        (problems if not (reading.unknown or reading.undeclared) else cautions).append(
            problem
        )
    return ReachFindings(tuple(problems), tuple(cautions))


def _gap_finding(planned: PlannedStep, found: _Above, gap: int) -> Problem | None:
    """The finding for one checked step, or ``None`` when its gap is enough.

    *found* is the reach above the step, its own left out.
    """
    step = planned.step_id
    where = "params.cv_gap_frames"
    unbounded = found.unknown | found.undeclared
    if gap >= found.frames and (not unbounded or gap > 0):
        # Covers what is known, and a positive gap beside steps that say nothing
        # of their reach is a gap the user chose with that in view.
        return None
    if not unbounded:
        return Problem(
            step,
            where,
            f"{planned.runs} cross-validates by blocks of frames and drops "
            f"{gap} frames around each held-out block, but a row it trains on "
            f"reads {found.frames} frames on each side ({_chain(found.path)}). "
            "Training rows just outside the gap would carry the held-out "
            f"frames in their inputs. Set cv_gap_frames to {found.frames} or more",
        )
    lead = (
        f"at least {found.frames} frames on each side ({_chain(found.path)}) and "
        if found.frames
        else ""
    )
    causes: list[str] = []
    if found.unknown:
        causes.append(f"{_named(found.unknown)} read a span no params bound")
    if found.undeclared:
        causes.append(f"{_named(found.undeclared)} declare no temporal reach")
    return Problem(
        step,
        where,
        f"{planned.runs} cross-validates by blocks of frames and drops {gap} frames "
        f"around each held-out block. A row it trains on reads {lead}an unknown "
        f"span more: {'; '.join(causes)}. The gap cannot be checked, so set "
        "cv_gap_frames to the longest window above the classifier",
    )
