"""Whether a recipe is a recipe, decided before a dataset is opened.

A pipeline written by hand is a document, and most of what can be wrong with one
is wrong whatever dataset it is pointed at: a slug that names nothing, params
that do not validate, a cycle, a reference in a field that cannot hold one, two
steps wired together whose outputs cannot be aligned. Discovering those at run
time means discovering them one step at a time, after the expensive steps have
already run -- so they are decided here, against the real ``Params`` and
``Inputs`` models and the real declarations, with no dataset in hand.

**Every problem, not the first.** An author fixing a recipe wants the list, so
the walk records what it finds and carries on wherever carrying on is
meaningful. Where it is not -- a step whose slug names nothing has no params
model to validate against, and a step below it would only restate that fault
under another name -- the step is marked and its dependents skip the checks that
depend on it.

**An ``after`` edge between a media writer and a media reader is permitted.** It
was the one hazard here that produced a wrong answer rather than waste: a
tracker's identity has no term for the media it read, so re-transcoding left
every run below it reading as complete over different pixels. That is closed one
level down -- a bridged tracks row records the media composition it consumed and
the inventory compares it -- so such a graph now reads as drifted rather than as
done, and refusing the edge would refuse the ordinary ``transcode -> trex`` shape
for a gap that no longer exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from .compatibility import (
    TRACKS_DECLARATION,
    Declaration,
    EntityLevel,
    ProducerDecl,
    can_join,
    resolve_emits,
)
from .model import (
    FeatureStepSpec,
    OpStepSpec,
    Recipe,
    Step,
    StepRef,
    params_step_refs,
    references_of,
)
from .resolve import (
    ResolvedStep,
    StepBuildError,
    build_step_feature,
    build_step_op_params,
    declaration_catalog,
    params_reference_site,
    resolve_step_spec,
)
from .topo import RecipeCycle, topological_order

if TYPE_CHECKING:
    from .compatibility import DeclarationCatalog

__all__ = [
    "EXCLUDED_KINDS",
    "OVERWRITE_PARAM",
    "Problem",
    "RecipeInvalid",
    "check_recipe",
    "reject_unless_valid",
]

OVERWRITE_PARAM: Final = "overwrite"
"""The one params key a recipe may not carry, whatever its value.

It mutates content under a stable address, so a concurrent downstream reader gets
a mixed read that its own ``run_id`` records nothing about -- the address says one
thing and the bytes say another, and nothing on disk records which. The format has
no field for it and this is what keeps it that way: refused on presence rather
than on truth, because ``overwrite: false`` in a file is an author expecting the
key to mean something.
"""

EXCLUDED_KINDS: Final[dict[str, str]] = {
    "extract-frames": (
        "mosaic-api owns frame extraction through its own registration request, "
        "and embeds this op's frozen identifier mid-string in the image paths of "
        "annotated frames. Two owners of one piece of work is worse than a "
        "missing step type"
    ),
}
"""Ops a graph may not run, and why -- ownership rather than capability."""

_PLACEHOLDER_IDENTIFIER: Final = "0.0-0000000000"
"""Stands in for an identifier only a dataset can supply.

Validation asks whether a step is *constructible*, which needs its references
substituted with something of the right shape rather than with the right value.
An ordinary-looking identifier keeps a params model that requires a non-empty one
from failing for something the recipe is not responsible for.
"""

_PROBE_ENTRIES: Final[tuple[tuple[str, str], ...]] = (("", "__probe__"),)
"""A scope of one, standing in for the one a plan supplies.

An op step's entries come from the plan rather than from the recipe, and
``TranscodeOp`` refuses params that name no scope -- correctly, because a
transcode with none re-encodes a whole corpus. Validating a recipe with an empty
scope would therefore refuse the ordinary transcode step for the one thing a
recipe is *not* supposed to carry. So the question asked here is the honest one:
would these params be valid once a plan says what to run them over.
"""


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a recipe, at the place it is wrong.

    Attributes:
        step: The step id, or ``""`` for a fault of the graph as a whole.
        where: Which part of the step -- ``feature``, ``kind``, ``inputs``,
            ``tracks``, ``params.templates``, ``steps``.
        message: What is wrong, phrased so it reads on its own.
    """

    step: str
    where: str
    message: str

    def __str__(self) -> str:
        at = f"step {self.step!r} " if self.step else ""
        return f"{at}{self.where}: {self.message}"


class RecipeInvalid(ValueError):
    """A recipe cannot be run as written, carrying every fault found."""

    def __init__(self, problems: tuple[Problem, ...]) -> None:
        self.problems: tuple[Problem, ...] = problems
        listed = "\n".join(f"  - {problem}" for problem in problems)
        plural = "" if len(problems) == 1 else "s"
        super().__init__(
            f"this recipe cannot be run: {len(problems)} problem{plural}\n{listed}"
        )


def _runs(step: Step) -> tuple[str, str]:
    """What this step runs, and which field names it."""
    if isinstance(step, FeatureStepSpec):
        return step.feature, "feature"
    return step.kind, "kind"


def _inputs_of(step: Step) -> tuple[str | StepRef, ...]:
    """A feature step's inputs; an op takes none, and declares what it reads."""
    return tuple(step.inputs) if isinstance(step, FeatureStepSpec) else ()


@dataclass
class _Walk:
    """What the topological walk has established about the steps behind it.

    Held rather than recomputed because three checks need it: an edge needs the
    upstream's declaration and its *resolved* entity level, a ``tracks``
    reference needs to know what the referenced step runs, and construction needs
    the storage names the substitution reads.
    """

    catalog: DeclarationCatalog
    runs: dict[str, str] = field(default_factory=dict)
    levels: dict[str, EntityLevel] = field(default_factory=dict)
    resolved: dict[str, ResolvedStep] = field(default_factory=dict)
    broken: set[str] = field(default_factory=set)

    def declaration_behind(self, step_id: str) -> Declaration | None:
        """What the step with this id runs, as a declaration."""
        return self.catalog.get(self.runs.get(step_id, ""))

    def level_behind(self, step_id: str) -> EntityLevel:
        """The resolved entity level of that step's output.

        ``individual`` when the walk has nothing recorded, which is what reading
        raw tracks gives a step and the only thing a step with no resolvable
        upstream could be reading.
        """
        return self.levels.get(step_id, "individual")

    def record(self, step: Step, declared: Declaration, storage_name: str) -> None:
        """Take down what a step that checked out will offer to the ones below."""
        self.levels[step.id] = resolve_emits(
            declared.emits,
            [
                TRACKS_DECLARATION.produces.level
                if isinstance(item, str)
                else self.level_behind(item.step)
                for item in _inputs_of(step)
            ],
        )
        self.resolved[step.id] = ResolvedStep(
            step_id=step.id,
            storage_name=storage_name,
            run_id=_PLACEHOLDER_IDENTIFIER,
            tracks_variant=(
                _PLACEHOLDER_IDENTIFIER if declared.produces.writes_tracks else ""
            ),
            model_run_id=(
                _PLACEHOLDER_IDENTIFIER if declared.category == "train" else ""
            ),
        )


def check_recipe(
    recipe: Recipe, catalog: DeclarationCatalog | None = None
) -> tuple[Problem, ...]:
    """Every reason *recipe* cannot be run, in topological order.

    Args:
        recipe: The graph to check. Read only, and no dataset is opened.
        catalog: The declarations to check against. Built from the registries
            when not given, which is what costs the feature-library import -- a
            caller already holding one (an API serving it to a canvas) passes it
            and pays nothing.

    Returns:
        The problems found, empty when the recipe is well formed.
    """
    try:
        ordered = topological_order(recipe)
    except RecipeCycle as exc:
        # Nothing else can be checked: every remaining question is asked of a
        # step's upstreams, and a cycle is exactly the state in which "upstream"
        # has no meaning.
        return (Problem("", "steps", str(exc)),)

    walk = _Walk(catalog=declaration_catalog() if catalog is None else catalog)
    problems: list[Problem] = []
    for step in ordered:
        problems.extend(_check_step(step, recipe, walk))
    return tuple(problems)


def _check_step(step: Step, recipe: Recipe, walk: _Walk) -> list[Problem]:
    """Check one step, and record what it offers the steps below it."""
    name, where = _runs(step)

    if isinstance(step, OpStepSpec) and step.kind in EXCLUDED_KINDS:
        walk.broken.add(step.id)
        return [
            Problem(
                step.id,
                where,
                f"{step.kind!r} may not run as a graph step: "
                f"{EXCLUDED_KINDS[step.kind]}",
            )
        ]

    declared = walk.catalog.get(name)
    expected_kind = "feature" if isinstance(step, FeatureStepSpec) else "op"
    if declared is None or declared.produces.kind != expected_kind:
        walk.broken.add(step.id)
        return [
            Problem(step.id, where, f"no registered {expected_kind} is named {name!r}")
        ]
    walk.runs[step.id] = name

    problems: list[Problem] = []
    overwrites = OVERWRITE_PARAM in step.params
    if overwrites:
        problems.append(
            Problem(
                step.id,
                f"params.{OVERWRITE_PARAM}",
                "a graph step may not overwrite: it mutates content under a "
                "stable address, so a downstream reader gets a mixed read its own "
                "run_id records nothing about. Change the params instead, which "
                "gives the new work its own address",
            )
        )

    unsubstitutable = _reference_site_problems(step, name, expected_kind)
    problems.extend(unsubstitutable)
    problems.extend(_tracks_reference_problems(step, walk))

    if unsubstitutable or any(
        referenced in walk.broken for referenced, _ in references_of(step)
    ):
        # Either this step's references cannot be substituted at all, or one of
        # them names a step that did not check out -- so anything derived from
        # the substitution would report a fault the author has already been told
        # about, under a name that hides it.
        walk.broken.add(step.id)
        return problems

    edge = _edge_problem(step, declared, walk)
    if edge is not None:
        problems.append(edge)

    spec = resolve_step_spec(recipe, step.id, walk.resolved, entries=_PROBE_ENTRIES)
    if overwrites:
        # Already reported, and reported for a reason the params model does not
        # know: on a feature there is no such field at all, so leaving it in
        # would add "extra inputs are not permitted" beside a message that
        # already says what is wrong and why.
        spec = replace(
            spec,
            params={
                key: value
                for key, value in spec.params.items()
                if key != OVERWRITE_PARAM
            },
        )
    try:
        if isinstance(step, FeatureStepSpec):
            _ = build_step_feature(spec)
        else:
            _ = build_step_op_params(spec)
    except StepBuildError as exc:
        walk.broken.add(step.id)
        problems.append(Problem(step.id, exc.stage, exc.detail))
        return problems

    walk.record(
        step, declared, spec.storage_name if isinstance(step, FeatureStepSpec) else ""
    )
    return problems


def _reference_site_problems(step: Step, name: str, kind: str) -> list[Problem]:
    """Params references landing in a field that cannot hold one."""
    found: list[Problem] = []
    for field_name in params_step_refs(step.params):
        site = params_reference_site(kind, name, field_name)
        if site == "unknown":
            found.append(
                Problem(
                    step.id,
                    f"params.{field_name}",
                    f"{name!r} declares no params field {field_name!r}",
                )
            )
        elif site == "unsupported":
            found.append(
                Problem(
                    step.id,
                    f"params.{field_name}",
                    f"{name}'s {field_name!r} holds a plain value rather than a "
                    f"run reference, so a step reference has nothing to "
                    f"substitute into. A reference resolves into a field declared "
                    f"as a Result or an ArtifactSpec, or as the bare identifier "
                    f"that names a model",
                )
            )
    return found


def _tracks_reference_problems(step: Step, walk: _Walk) -> list[Problem]:
    """A ``tracks`` reference must name a step that writes a tracks variant."""
    if not isinstance(step, FeatureStepSpec) or step.tracks is None:
        return []
    upstream = walk.declaration_behind(step.tracks.step)
    if upstream is None or upstream.produces.writes_tracks:
        return []
    return [
        Problem(
            step.id,
            "tracks",
            f"{step.tracks.step!r} runs {upstream.produces.name!r}, which writes "
            f"no tracks variant, so there is nothing for --tracks-run-id to name",
        )
    ]


def _edge_problem(step: Step, declared: Declaration, walk: _Walk) -> Problem | None:
    """Whether this step's inputs, together, may feed it.

    ``can_join`` rather than a loop of ``can_connect``: the refusal that matters
    is about the *set*, because two inputs at different entity levels share no
    identity column and merging them on ``frame`` alone is a per-frame cartesian
    product -- a plausible table nothing raises about.
    """
    producers: list[ProducerDecl] = []
    for item in _inputs_of(step):
        if isinstance(item, str):
            producers.append(TRACKS_DECLARATION.produces)
            continue
        upstream = walk.declaration_behind(item.step)
        if upstream is None:
            return None
        producers.append(replace(upstream.produces, level=walk.level_behind(item.step)))
    verdict = can_join(producers, declared.consumes)
    return None if verdict else Problem(step.id, "inputs", verdict.reason)


def reject_unless_valid(
    recipe: Recipe, catalog: DeclarationCatalog | None = None
) -> None:
    """Raise unless *recipe* is well formed.

    What ``plan_pipeline`` calls first, before it touches a dataset, so a
    malformed graph is refused as a document rather than half-planned against
    storage.

    Raises:
        RecipeInvalid: Carrying every problem found, not only the first.
    """
    problems = check_recipe(recipe, catalog)
    if problems:
        raise RecipeInvalid(problems)
