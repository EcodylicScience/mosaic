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

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, TypeGuard

from pydantic import ValidationError

from mosaic.core.json_value import JsonValue

from .compatibility import (
    TRACKS_DECLARATION,
    Declaration,
    EntityLevel,
    ProducerDecl,
    can_connect,
    can_join,
    resolve_emits,
)
from .digest import recipe_digest
from .model import (
    TRACKS_INPUT,
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
from .wire import AttachReport, Attachment, Problem, ValidationReport, render_location
from ..preprocess_layout import PREPROCESS_KIND
from mosaic.core.pipeline.types import ArtifactSpec
from mosaic.core.params import Params
from mosaic.core.scope import Scope

if TYPE_CHECKING:
    from .compatibility import DeclarationCatalog

__all__ = [
    "EXCLUDED_KINDS",
    "OVERWRITE_PARAM",
    "RecipeInvalid",
    "attach_candidates",
    "check_recipe",
    "read_recipe",
    "reject_unless_valid",
    "resolved_levels",
    "validation_report",
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

    known = declaration_catalog() if catalog is None else catalog
    walk = _Walk(catalog=known, levels=resolved_levels(recipe, known))
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
    problems.extend(_media_reference_problems(step, walk))

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

    spec = resolve_step_spec(
        recipe, step.id, walk.resolved, entries=Scope(entries=list(_PROBE_ENTRIES))
    )
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
            built = build_step_feature(spec).params
        else:
            built = build_step_op_params(spec)
    except StepBuildError as exc:
        walk.broken.add(step.id)
        problems.extend(_build_problems(step.id, exc))
        return problems
    except ImportError as exc:
        # A registered feature whose optional dependency is not installed here
        # raises on construction. Reported rather than raised, because the
        # recipe cannot be checked or run on this installation, and a validator
        # that crashes answers an editor with nothing.
        walk.broken.add(step.id)
        problems.append(
            Problem(
                step.id,
                where,
                f"{name!r} needs an optional dependency that is not installed "
                f"here: {exc}",
            )
        )
        return problems

    problems.extend(_glob_reference_problems(step, name, built))

    walk.record(
        step, declared, spec.storage_name if isinstance(step, FeatureStepSpec) else ""
    )
    return problems


def _build_problems(step_id: str, exc: StepBuildError) -> list[Problem]:
    """What a failed construction reports: one problem per params field refused.

    A params refusal is itemized at ``params.<field>`` so an editor can mark the
    field. An inputs or slug refusal stays one problem at its stage, and so does
    a params refusal no validator raised.
    """
    if exc.stage != "params" or not exc.errors:
        return [Problem(step_id, exc.stage, exc.detail)]
    return [
        Problem(step_id, f"params.{location}" if location else "params", message)
        for location, message in exc.errors
    ]


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


_GLOB_METACHARACTERS: Final = ("*", "?", "[")


def _glob_reference_problems(step: Step, name: str, built: Params) -> list[Problem]:
    """Artifact references that would resolve by glob rather than name a file.

    Asked of the *built* params, so the answer covers both halves of the question:
    a pattern the recipe wrote, and one the consumer's declared type defaulted to.
    A producer's run root holds one per-entry output parquet per sequence beside
    whatever it saved, so a glob resolves whichever file sorts first -- which is a
    per-entry table, and reads downstream as the artifact the author meant.

    Refused here rather than only at run time because this needs no dataset: the
    author hears it from ``mosaic pipeline validate``, before the producing step
    has run.
    """
    found: list[Problem] = []
    for field_name in params_step_refs(step.params):
        value: object = getattr(built, field_name, None)
        if not isinstance(value, ArtifactSpec):
            continue
        pattern = value.pattern
        if not any(char in pattern for char in _GLOB_METACHARACTERS):
            continue
        spelling = '{"step": ..., "pattern": "<filename>"}'
        found.append(
            Problem(
                step.id,
                f"params.{field_name}",
                f"{name}'s {field_name!r} would resolve by the glob {pattern!r}. A "
                f"producer's run directory holds one per-entry output parquet per "
                f"sequence beside its named artifacts, so a glob resolves whichever "
                f"file sorts first rather than the artifact meant. Name the file: "
                f"{spelling}",
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


_MEDIA_FIELD: Final = "media"
"""The params field that names the media variant that a step reads."""


def _media_reference_problems(step: Step, walk: _Walk) -> list[Problem]:
    """A ``media`` reference must name a step that writes a media variant.

    Every media op writes media, and only a ``preprocess`` run id names a variant
    that a consumer can read. A transcode's derivatives are routed by verdict and
    named by recipe. The validator refuses a transcode's run id in ``media``, which
    names a variant without rows.
    """
    reference = params_step_refs(step.params).get(_MEDIA_FIELD)
    if reference is None:
        return []
    upstream = walk.declaration_behind(reference.step)
    if upstream is None or (
        upstream.produces.kind == "op" and upstream.produces.name == PREPROCESS_KIND
    ):
        return []
    return [
        Problem(
            step.id,
            f"params.{_MEDIA_FIELD}",
            f"{reference.step!r} runs {upstream.produces.name!r}, which does not "
            f"write a media variant for {_MEDIA_FIELD!r} to name. A media "
            f"reference names a {PREPROCESS_KIND} step",
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


def read_recipe(document: object) -> Recipe:
    """Parse *document* as a recipe, reporting a malformed one as problems.

    The one parse every reader shares, so a structural fault reads the same from
    the CLI, from an API and from an editor. A pydantic error inside
    ``steps[i]`` is reported at that step's id when the document gives it one.

    Raises:
        RecipeInvalid: Carrying one problem per structural fault.
    """
    try:
        return Recipe.model_validate(document)
    except ValidationError as exc:
        problems = tuple(
            _structural_problem(document, err["loc"], err["type"], err["msg"])
            for err in exc.errors()
        )
        raise RecipeInvalid(problems) from exc


def _structural_problem(
    document: object, location: tuple[int | str, ...], kind: str, message: str
) -> Problem:
    """Place one pydantic error at the step it sits in, when that has an id."""
    if len(location) < 2 or location[0] != "steps" or not isinstance(location[1], int):
        return Problem("", render_location(location) or "recipe", message)
    index = location[1]
    rest = location[2:]
    raw_step = _step_document(document, index)
    tag = raw_step.get("type")
    # pydantic names the union member it tried after the index, which is the
    # step's own ``type`` and not a field of it.
    if rest and isinstance(tag, str) and rest[0] == tag:
        rest = rest[1:]
    where = render_location(rest)
    if not where and kind.startswith("union_tag"):
        where = "type"
    step_id = raw_step.get("id")
    if isinstance(step_id, str) and step_id.strip():
        return Problem(step_id, where or "step", message)
    at = render_location(location[:2])
    return Problem("", f"{at}.{where}" if where else at, message)


def _step_document(document: object, index: int) -> Mapping[str, object]:
    """The raw mapping at ``steps[index]``, or an empty one when there is none."""
    steps = document.get("steps") if _is_mapping(document) else None
    if not _is_list(steps) or not 0 <= index < len(steps):
        return {}
    raw = steps[index]
    return raw if _is_mapping(raw) else {}


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Whether *value* is a JSON object, read through string keys."""
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    """Whether *value* is a JSON array."""
    return isinstance(value, list)


def validation_report(
    document: object, catalog: DeclarationCatalog | None = None
) -> ValidationReport:
    """Everything wrong with *document*, which need not parse.

    Structural faults when it does not parse, and :func:`check_recipe`'s
    findings when it does. Never raises for a fault in the document, because an
    editor asks after every edit and a list of problems is the ordinary answer.
    """
    try:
        recipe = read_recipe(document)
    except RecipeInvalid as exc:
        return ValidationReport(valid=False, digest=None, problems=exc.problems)
    problems = check_recipe(recipe, catalog)
    return ValidationReport(
        valid=not problems, digest=recipe_digest(recipe), problems=problems
    )


def resolved_levels(
    recipe: Recipe, catalog: DeclarationCatalog
) -> dict[str, EntityLevel]:
    """The entity level each step's output resolves to, keyed by step id.

    A step's level depends only on its declaration and its inputs, so it is
    answered for every step whose name the catalog declares, including one
    whose params do not build. ``as-input`` resolves through the first input,
    and a step the catalog does not declare is left out; a reader of it gets
    ``individual``, which is what raw tracks give.

    Raises:
        RecipeCycle: The recipe has a cycle, so no step has an upstream.
    """
    levels: dict[str, EntityLevel] = {}
    for step in topological_order(recipe):
        declared = catalog.get(_runs(step)[0])
        if declared is None:
            continue
        levels[step.id] = resolve_emits(
            declared.emits,
            [
                TRACKS_DECLARATION.produces.level
                if isinstance(item, str)
                else levels.get(item.step, "individual")
                for item in _inputs_of(step)
            ],
        )
    return levels


def attach_candidates(
    recipe: Recipe, after: str | None, catalog: DeclarationCatalog | None = None
) -> AttachReport:
    """What may be added directly below *after*, and the step to add for each.

    Every step type a graph may run is answered, allowed or refused with a
    reason. An allowed one carries a template: the new step complete except for
    its ``id``, wired to *after* through ``inputs`` below a feature, through
    ``tracks`` for a feature below a tracks-writing op, and through ``after`` for
    an op below an op. At the top level (*after* ``None``) an op needs no
    reference and a feature reads the dataset's tracks.

    The recipe must parse and be acyclic. It need not validate: a params error
    elsewhere does not change what may follow a step.

    Raises:
        RecipeInvalid: The recipe has a cycle, or *after* names no step in it.
    """
    known = declaration_catalog() if catalog is None else catalog
    try:
        levels = resolved_levels(recipe, known)
    except RecipeCycle as exc:
        raise RecipeInvalid((Problem("", "steps", str(exc)),)) from exc
    above: Step | None = None
    if after is not None:
        above = next((step for step in recipe.steps if step.id == after), None)
        if above is None:
            message = f"no step {after!r} in this recipe"
            raise RecipeInvalid((Problem("", "after", message),))
    candidates: dict[str, Attachment] = {}
    for name in known.names():
        declared = known.get(name)
        if declared is None or name == TRACKS_INPUT or name in EXCLUDED_KINDS:
            continue
        candidates[name] = _attachment(name, declared, above, levels, known)
    return AttachReport(after=after, candidates=candidates)


def _attachment(
    name: str,
    declared: Declaration,
    above: Step | None,
    levels: dict[str, EntityLevel],
    catalog: DeclarationCatalog,
) -> Attachment:
    """Whether *name* may go directly below *above*, and wired how."""
    is_op = declared.produces.kind == "op"
    base: dict[str, JsonValue] = (
        {"type": "op", "kind": name} if is_op else {"type": "feature", "feature": name}
    )
    if above is None:
        if is_op:
            return _allowed(base)
        verdict = can_connect(TRACKS_DECLARATION.produces, declared.consumes)
        return (
            _allowed({**base, "inputs": [TRACKS_INPUT]})
            if verdict
            else _refused(verdict.reason)
        )

    upstream = catalog.get(_runs(above)[0])
    if upstream is None:
        return _refused(f"step {above.id!r} runs nothing this installation declares")
    if isinstance(above, FeatureStepSpec):
        producer = replace(upstream.produces, level=levels.get(above.id, "individual"))
        verdict = can_connect(producer, declared.consumes)
        if not verdict:
            return _refused(verdict.reason)
        return _allowed({**base, "inputs": [{"step": above.id}]})

    if is_op:
        return _allowed({**base, "after": [above.id]})
    if not upstream.produces.writes_tracks:
        return _refused(
            f"{above.id!r} runs {upstream.produces.name!r}, which writes no "
            f"tracks variant for a feature to read"
        )
    verdict = can_connect(TRACKS_DECLARATION.produces, declared.consumes)
    if not verdict:
        return _refused(verdict.reason)
    return _allowed({**base, "inputs": [TRACKS_INPUT], "tracks": {"step": above.id}})


def _allowed(template: dict[str, JsonValue]) -> Attachment:
    return Attachment(allowed=True, reason="", template=template)


def _refused(reason: str) -> Attachment:
    return Attachment(allowed=False, reason=reason, template=None)
