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

**Columns are checked against each feature's declared contract.** The walk
propagates the output columns of every feature step (:mod:`.columns`) and checks
each step's ``requires`` against its merged input. A required column that every
input lacks is a problem, and its message names the features that provide it. A
column is a *caution* when an input may contain it, or when an undeclared step
may produce it. Validation reports a caution, and a run is not refused for it. A
step whose feature does not declare ``requires`` is listed in the report's
``unchecked`` steps, and the columns that it reads are not checked. A plan checks
the columns again, with the columns of the tracks tables that each step reads
(``check_recipe(tracks=...)``). A column that the schemas allow and every table
lacks is then a problem.

**A feature that this installation cannot run is a problem at the step's
``feature``.** The problem's message is the sentence of the feature's unmet
prerequisites, :attr:`~.compatibility.Declaration.unavailable`, which the editor
catalog shows and an attach candidate is refused with. The rest of the step is
still checked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_object_list, is_str_mapping

from .._utils import ResolvedScope, hash_params
from ..column_contract import (
    NamedColumns,
    ParamValues,
    contract_fields,
    needs_of,
    param_values,
)
from ..error_kind import ClassifiedError, ErrorKind
from ..run import (
    MissingConsumedRootsDeclaration,
    MissingScopeDeclaration,
    compute_run_id,
)
from .columns import (
    UNKNOWN_COLUMNS,
    ColumnSet,
    merged_input,
    names_wire,
    need_status,
    output_columns,
    provider_phrase,
    providers_of,
    tracks_columns,
)
from .compatibility import (
    TRACKS_DECLARATION,
    Declaration,
    EntityLevel,
    ProducerDecl,
    can_connect,
    can_join,
    join_verdicts,
    resolve_emits,
)
from .digest import recipe_digest
from .expand import RepeatInvalid, expand_repeats
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
    StepSpec,
    build_step_feature,
    build_step_op_params,
    declaration_catalog,
    params_reference_site,
    resolve_step_spec,
)
from .topo import RecipeCycle, descendants_of, topological_order
from .wire import (
    AttachReport,
    Attachment,
    InputCandidate,
    InputCandidates,
    InsertReport,
    Problem,
    StillNeeded,
    UnmetStatus,
    ValidationReport,
    render_location,
)
from ..preprocess_layout import PREPROCESS_KIND
from mosaic.core.pipeline.types import ArtifactSpec
from mosaic.core.params import Params
from mosaic.core.scope import Scope

if TYPE_CHECKING:
    from mosaic.core.pipeline.types import Feature

    from .compatibility import DeclarationCatalog

__all__ = [
    "EXCLUDED_KINDS",
    "OVERWRITE_PARAM",
    "RecipeInvalid",
    "attach_candidates",
    "check_recipe",
    "input_candidates",
    "insert_candidates",
    "one_run_problems",
    "read_recipe",
    "reject_unless_valid",
    "resolved_columns",
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


def _stand_in(terms: object) -> str:
    """A stand-in for a step's identifier, which only a dataset can supply.

    Validation asks whether a step is *constructible*. Each reference is then
    substituted with a value of the right shape. An ordinary-looking identifier
    keeps a params model that requires a non-empty one from failing for something
    the recipe is not responsible for.

    The stand-in is a digest of the identity terms that the recipe decides, and
    two steps share one when those terms agree, and only then. Planning resolves such
    steps to one run id, and validation refuses them as two inputs of one step.
    Two steps running one feature over one input share a storage name, and with
    different params their stand-ins tell them apart.
    """
    return f"0.0-{hash_params(terms)}"


def _feature_stand_in(spec: StepSpec, feature: Feature) -> str:
    """The stand-in for a feature step: its ``run_id`` as far as the recipe decides it.

    :func:`~mosaic.core.pipeline.run.compute_run_id` hashes the step over a probe
    scope and the tracks variant that the step names. A time filter is hashed as
    written beside it, because converting it to frames takes the dataset's frame
    rate. A frame filter and the time filter equal to it therefore have different
    stand-ins, and a plan refuses the two inputs that resolve to one run.
    """
    variant = spec.tracks_run_id if feature.inputs.has_tracks else None
    probe = ResolvedScope(
        entries=set(_PROBE_ENTRIES),
        tracks_variants=() if variant is None else (variant,),
        # A labels reference pins the variant it reads, so two steps differing
        # only in the conversion they read are two runs.
        labels_variants=() if not spec.labels_run_id else (spec.labels_run_id,),
    )
    run_id, _ = compute_run_id(
        feature,
        spec.filter_start_frame,
        spec.filter_end_frame,
        probe,
        overlap_frames=spec.overlap_frames,
    )
    return _stand_in(
        {
            "feature": spec.feature,
            "run_id": run_id,
            "time": [spec.filter_start_time, spec.filter_end_time],
        }
    )


def _op_stand_in(spec: StepSpec) -> str:
    """The stand-in for an op step, from its kind and its params as written.

    An op's identity reads fields that ``identity_dump`` leaves out, such as
    ``media``. Every params field therefore counts.
    """
    return _stand_in({"kind": spec.op_kind, "params": dict(spec.params)})


_PROBE_ENTRIES: Final[tuple[tuple[str, str], ...]] = (("", "__probe__"),)
"""A scope of one, standing in for the one a plan supplies.

An op step's entries come from the plan rather than from the recipe, and
``TranscodeOp`` refuses params that name no scope -- correctly, because a
transcode with none re-encodes a whole corpus. Validating a recipe with an empty
scope would therefore refuse the ordinary transcode step for the one thing a
recipe is *not* supposed to carry. So the question asked here is the honest one:
would these params be valid once a plan says what to run them over.
"""


class RecipeInvalid(ClassifiedError, ValueError):
    """A recipe cannot be run as written, carrying every fault found."""

    error_kind: ClassVar[ErrorKind] = "invalid"

    lead: ClassVar[str] = "this recipe cannot be run"
    """How the message opens, before the count of problems."""

    def __init__(self, problems: tuple[Problem, ...]) -> None:
        self.problems: tuple[Problem, ...] = problems
        listed = "\n".join(f"  - {problem}" for problem in problems)
        plural = "" if len(problems) == 1 else "s"
        super().__init__(f"{self.lead}: {len(problems)} problem{plural}\n{listed}")

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the problems, each as its step, its place and its message."""
        return {"problems": [problem.as_json() for problem in self.problems]}


def _runs(step: Step) -> tuple[str, str]:
    """What this step runs, and which field names it."""
    if isinstance(step, FeatureStepSpec):
        return step.feature, "feature"
    return step.kind, "kind"


def _inputs_of(step: Step) -> tuple[str | StepRef, ...]:
    """A feature step's inputs; an op takes none, and declares what it reads."""
    return tuple(step.inputs) if isinstance(step, FeatureStepSpec) else ()


type _InputRun = str | tuple[str, str]
"""The run that one input of a step reads, keyed as a feature's inputs model keys it.

``tracks``, or an earlier step's storage name and stand-in.
"""


@dataclass
class _Walk:
    """What the topological walk has established about the steps behind it.

    Recorded once because four checks need it: an edge needs the
    upstream's declaration and its *resolved* entity level, a ``tracks``
    reference needs to know what the referenced step runs, construction needs
    the storage names the substitution reads, and a column check needs the
    columns of each upstream's output.
    """

    catalog: DeclarationCatalog
    runs: dict[str, str] = field(default_factory=dict)
    levels: dict[str, EntityLevel] = field(default_factory=dict)
    resolved: dict[str, ResolvedStep] = field(default_factory=dict)
    broken: set[str] = field(default_factory=set)
    columns: dict[str, ColumnSet] = field(default_factory=dict)
    """The output columns of each feature step that checked out."""
    values: dict[str, ParamValues] = field(default_factory=dict)
    """The built params of each such step, every field its contract reads."""
    cautions: list[Problem] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    """The feature steps whose feature does not declare ``requires``, in walk order."""
    measured_tracks: Mapping[str, ColumnSet] = field(
        default_factory=dict[str, ColumnSet]
    )
    """The columns of the tracks that a step reads, keyed by step id, where a plan
    has read them from the tables."""
    tracks: ColumnSet = field(init=False)
    """The columns of the dataset's tracks, as the declarations describe them."""

    def __post_init__(self) -> None:
        self.tracks = tracks_columns(self.catalog)

    def declaration_behind(self, step_id: str) -> Declaration | None:
        """What the step with this id runs, as a declaration."""
        return self.catalog.get(self.runs.get(step_id, ""))

    def tracks_of(self, step_id: str) -> ColumnSet:
        """The columns of the tracks that the step *step_id* reads.

        The columns read from its tables, where a plan has read them, and else
        the columns that the declarations describe.
        """
        return self.measured_tracks.get(step_id, self.tracks)

    def input_columns(self, item: str | StepRef) -> ColumnSet:
        """The columns of one input of a step: the tracks, or an earlier step's output.

        :data:`UNKNOWN_COLUMNS` for a step that did not check out. A column it
        supplies is then a caution.
        """
        if isinstance(item, str):
            return self.tracks
        return self.columns.get(item.step, UNKNOWN_COLUMNS)

    def merged_columns(self, step: FeatureStepSpec) -> ColumnSet:
        """The columns of *step*'s merged input, read with the tracks of *step*."""
        declared = self.catalog.get(step.feature)
        return merged_input(
            [
                self.tracks_of(step.id)
                if isinstance(item, str)
                else self.input_columns(item)
                for item in step.inputs
            ],
            stack=declared is not None and declared.consumes.stacks_inputs,
        )

    def level_behind(self, step_id: str) -> EntityLevel:
        """The resolved entity level of that step's output.

        ``individual`` when the walk has nothing recorded, which is what reading
        raw tracks gives a step and the only thing a step with no resolvable
        upstream could be reading.
        """
        return self.levels.get(step_id, "individual")

    def record(
        self, step: Step, declared: Declaration, storage_name: str, stand_in: str
    ) -> None:
        """Take down what a step that checked out will offer to the ones below."""
        writes_labels = declared.produces.writes_labels
        kind = step.params.get("kind", "behavior") if writes_labels else ""
        self.resolved[step.id] = ResolvedStep(
            step_id=step.id,
            storage_name=storage_name,
            run_id=stand_in,
            tracks_variant=stand_in if declared.produces.writes_tracks else "",
            model_run_id=stand_in if declared.category == "train" else "",
            labels_variant=stand_in if writes_labels else "",
            labels_kind=str(kind).lower() if isinstance(kind, str) else "",
        )

    def run_of(self, item: str | StepRef) -> _InputRun | None:
        """The run that *item* reads, or ``None`` when its step did not check out."""
        return _run_of(item, self.resolved)


def _run_of(
    item: str | StepRef, resolved: Mapping[str, ResolvedStep]
) -> _InputRun | None:
    """The run that *item* reads, or ``None`` when its step has no identity."""
    if isinstance(item, str):
        return item
    upstream = resolved.get(item.step)
    if upstream is None or upstream.run_id is None:
        return None
    return upstream.storage_name, upstream.run_id


def check_recipe(
    recipe: Recipe,
    catalog: DeclarationCatalog | None = None,
    *,
    tracks: Mapping[str, ColumnSet] | None = None,
) -> tuple[Problem, ...]:
    """Every reason *recipe* cannot be run, in topological order.

    Args:
        recipe: The graph to check. Read only, and no dataset is opened.
        catalog: The declarations to check against. Built from the registries
            when not given, which is what costs the feature-library import -- a
            caller already holding one (an API serving it to a canvas) passes it
            and pays nothing.
        tracks: The columns of the tracks that each step reads, keyed by step
            id, as a plan reads them from a dataset's tables. A step left out
            reads the tracks as the registered schemas describe them, and a
            column that they may contain is then a caution.

    Returns:
        The problems found, empty when the recipe is well formed.
    """
    return _examine(recipe, catalog, tracks).problems


@dataclass(frozen=True, slots=True)
class _Examined:
    """The problems that one walk of a recipe found, and the walk's record."""

    problems: tuple[Problem, ...]
    walk: _Walk
    cycle: RecipeCycle | None = None


def _examine(
    recipe: Recipe,
    catalog: DeclarationCatalog | None,
    tracks: Mapping[str, ColumnSet] | None = None,
) -> _Examined:
    """Walk *recipe* in topological order, checking every step."""
    known = declaration_catalog() if catalog is None else catalog
    try:
        ordered = topological_order(recipe)
    except RecipeCycle as exc:
        # Nothing else can be checked: every remaining question is asked of a
        # step's upstreams, and a cycle is exactly the state in which "upstream"
        # has no meaning.
        return _Examined((Problem("", "steps", str(exc)),), _Walk(catalog=known), exc)

    walk = _Walk(
        catalog=known,
        levels=resolved_levels(recipe, known),
        measured_tracks=tracks or {},
    )
    problems: list[Problem] = []
    for step in ordered:
        problems.extend(_check_step(step, recipe, walk))
    return _Examined(tuple(problems), walk)


def _walked(recipe: Recipe, catalog: DeclarationCatalog | None) -> _Walk:
    """The walk of *recipe*, for an answer that needs every step's columns.

    Raises:
        RecipeInvalid: The recipe has a cycle.
    """
    examined = _examine(recipe, catalog)
    if examined.cycle is not None:
        raise RecipeInvalid(examined.problems) from examined.cycle
    return examined.walk


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
    if isinstance(step, FeatureStepSpec) and declared.requires is None:
        walk.unchecked.append(step.id)

    problems: list[Problem] = []
    if declared.unavailable is not None:
        # The rest of the step is still checked, because its params and its
        # wiring are the author's to fix on any installation.
        problems.append(Problem(step.id, where, declared.unavailable))
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
    problems.extend(_labels_reference_problems(step, name, expected_kind, walk))
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
    empty_inputs = _empty_input_problems(step, recipe, walk)
    problems.extend(empty_inputs)

    one_run = one_run_problems(step, walk.resolved, walk.runs, alike=_SAME_TERMS)
    if one_run:
        # The inputs model refuses a run listed twice, and its message names the
        # stand-ins. These problems name the steps instead.
        walk.broken.add(step.id)
        problems.extend(one_run)
        return problems

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
            feature = build_step_feature(spec)
            built = feature.params
            stand_in = _feature_stand_in(spec, feature)
        else:
            built = build_step_op_params(spec)
            stand_in = _op_stand_in(spec)
    except StepBuildError as exc:
        walk.broken.add(step.id)
        problems.extend(_build_problems(step.id, exc))
        return problems
    except (MissingScopeDeclaration, MissingConsumedRootsDeclaration) as exc:
        # A feature registered without the declarations its identity reads. No
        # plan can name its run on any dataset.
        walk.broken.add(step.id)
        problems.append(Problem(step.id, where, str(exc)))
        return problems
    except ImportError as exc:
        # A feature may import its optional packages on construction. An unmet
        # declared prerequisite is reported above. An import that fails
        # otherwise is a dependency the feature does not declare. It is
        # reported as a problem. The recipe cannot be checked or run on this
        # installation, and a validator that crashes answers an editor with
        # nothing.
        walk.broken.add(step.id)
        if declared.unavailable is None:
            message = (
                f"{name!r} needs an optional dependency that is not installed "
                f"here: {exc}"
            )
            problems.append(Problem(step.id, where, message))
        return problems

    problems.extend(_glob_reference_problems(step, name, built))
    problems.extend(_artifact_feature_problems(step, name, built, walk))
    if isinstance(step, FeatureStepSpec):
        # An edge refusal or an empty input already says why these inputs
        # cannot feed this step. The columns they lack would restate it.
        checked = edge is None and not empty_inputs
        problems.extend(_column_problems(step, declared, built, walk, checked=checked))

    walk.record(
        step,
        declared,
        spec.storage_name if isinstance(step, FeatureStepSpec) else "",
        stand_in,
    )
    return problems


_SAME_TERMS: Final = "with the same params, inputs and run arguments"
"""Why two steps resolve to one run on any dataset."""


def one_run_problems(
    step: Step,
    resolved: Mapping[str, ResolvedStep],
    runs: Mapping[str, str],
    *,
    alike: str,
) -> list[Problem]:
    """A problem at *step*'s inputs for each run that several of them read.

    Args:
        step: The step whose inputs are compared.
        resolved: The identity of each step above it. An input whose step has
            none reads no run here.
        runs: What each step runs, by step id, for the message.
        alike: Why the steps behind one run resolve to it, said after what they
            run.

    Returns:
        One problem per run read more than once, empty when there is none.
    """
    readers: dict[_InputRun, list[str]] = {}
    for item in _inputs_of(step):
        run = _run_of(item, resolved)
        if run is not None:
            readers.setdefault(run, []).append(_input_label(item))
    return [
        Problem(step.id, "inputs", _one_run_reason(labels, runs, alike))
        for labels in readers.values()
        if len(labels) > 1
    ]


def _one_run_reason(labels: Sequence[str], runs: Mapping[str, str], alike: str) -> str:
    """Why a step may not read the inputs *labels* together: they are one run."""
    named = list(dict.fromkeys(labels))
    if len(named) == 1:
        return (
            f"{named[0]!r} appears {len(labels)} times among the inputs, and a "
            "step reads each input once"
        )
    listed = " and ".join(
        [", ".join(repr(label) for label in named[:-1]), repr(named[-1])]
    )
    return (
        f"{listed} run {runs[named[0]]} {alike}. "
        "They resolve to one run, and a step reads each run once"
    )


def _column_problems(
    step: FeatureStepSpec,
    declared: Declaration,
    built: Params,
    walk: _Walk,
    *,
    checked: bool,
) -> list[Problem]:
    """Record the columns of *step*'s output, and check its inputs against it.

    A required column that the merged input cannot contain is a problem. Columns
    it may contain, and columns that an undeclared step may produce, are
    cautions, one of each kind per step. A step whose feature does not declare
    ``requires`` is not checked, and :func:`_check_step` lists it as unchecked.
    """
    values = param_values(built, contract_fields(declared.requires, declared.provides))
    merged = walk.merged_columns(step)
    walk.values[step.id] = values
    walk.columns[step.id] = output_columns(declared.provides, merged, values)
    if not checked or declared.requires is None:
        return []

    problems: list[Problem] = []
    maybe: list[str] = []
    unchecked: list[str] = []
    for need in needs_of(declared.requires, values):
        status = need_status(need, merged)
        if status == "absent":
            problems.append(
                Problem(step.id, "inputs", _absent_message(step, need, walk))
            )
        elif status == "possible":
            maybe.append(need.label)
        elif status == "unknown":
            unchecked.append(need.label)
    if maybe:
        message = (
            f"{', '.join(maybe)} may be missing from its merged input. An input "
            "may contain it, and none guarantees it"
        )
        walk.cautions.append(Problem(step.id, "inputs", message))
    if unchecked:
        message = (
            f"{', '.join(unchecked)} cannot be checked. A step it reads from does "
            "not declare its columns, or a params value does not name a column"
        )
        walk.cautions.append(Problem(step.id, "inputs", message))
    return problems


def _empty_input_problems(step: Step, recipe: Recipe, walk: _Walk) -> list[Problem]:
    """A problem at *step*'s inputs for each input step with an empty declared output.

    Such a step writes only a sample (``Provides.is_empty``), and its table for
    each entry is empty. Each problem names the inputs of that step to read
    instead. It is raised even when another input meets *step*'s requirements.
    """
    problems: list[Problem] = []
    for item in _inputs_of(step):
        if isinstance(item, str):
            continue
        declared = walk.declaration_behind(item.step)
        if declared is None or declared.provides is None:
            continue
        if not declared.provides.is_empty:
            continue
        upstream = ", ".join(
            _input_label(each) for each in _inputs_of(recipe.step(item.step))
        )
        message = (
            f"{item.step!r} runs {declared.produces.name}, which writes only its "
            "sample and an empty table for each entry. Read its inputs "
            f"({upstream or 'none'}) here, and its sample through a params field"
        )
        problems.append(Problem(step.id, "inputs", message))
    return problems


def _input_label(item: str | StepRef) -> str:
    """One input as a message names it: ``tracks`` or the step's id."""
    return item if isinstance(item, str) else item.step


def _absent_message(step: FeatureStepSpec, need: NamedColumns, walk: _Walk) -> str:
    """The problem for a required column that every input lacks, naming its providers."""
    inputs = ", ".join(_input_label(item) for item in step.inputs) or "no inputs"
    return (
        f"needs {need.label}, and its merged input from {inputs} lacks it. "
        f"{_supplied_by(step, need, walk)}"
    )


def _supplied_by(step: FeatureStepSpec, need: NamedColumns, walk: _Walk) -> str:
    """Which inputs would supply *need*, as the end of a problem's message."""
    if need.payload:
        return "The tracks input and every feature that writes a measurement provide it"
    tracks = walk.tracks_of(step.id)
    declared = walk.catalog.get(step.feature)
    providers = providers_of(
        need,
        walk.catalog,
        tracks,
        besides=step.feature,
        consumer=None if declared is None else declared.consumes,
    )
    if providers:
        named = "; ".join(provider_phrase(provider) for provider in providers)
        sentence = f"{named} {'provides' if len(providers) == 1 else 'provide'} it"
        # The tracks lead the list, and their phrase is capitalized to open the
        # sentence. A slug keeps its case.
        if providers[0].feature == TRACKS_INPUT:
            return sentence[:1].upper() + sentence[1:]
        return sentence
    if need_status(need, tracks) == "possible":
        return "The tracks input may contain it"
    return "Neither the tracks nor a registered feature provides it"


def _build_problems(step_id: str, exc: StepBuildError) -> list[Problem]:
    """What a failed construction reports: one problem per params field refused.

    A params refusal is itemized at ``params.<field>`` so an editor can mark the
    field. An inputs or slug refusal stays one problem at its stage, and so does
    a params refusal raised outside a validator.
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
                    f"as a Result or an ArtifactSpec, as a labels source, or as "
                    f"the bare identifier that names a model",
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


def _artifact_feature_problems(
    step: Step, name: str, built: Params, walk: _Walk
) -> list[Problem]:
    """An artifact reference must name a step of the feature that writes the artifact.

    A field typed as one feature's artifact, such as the ``model`` of
    ``pair-posedistance-pca``, pins that feature. Several features write a
    ``model.joblib``, so a reference to the wrong step resolves to a file that
    loads and is read as the wrong model.
    """
    found: list[Problem] = []
    for field_name, reference in params_step_refs(step.params).items():
        value: object = getattr(built, field_name, None)
        if not isinstance(value, ArtifactSpec):
            continue
        pinned = value.pinned_feature()
        upstream = walk.runs.get(reference.step)
        if not pinned or upstream in (None, pinned):
            continue
        found.append(
            Problem(
                step.id,
                f"params.{field_name}",
                f"{name}'s {field_name!r} reads a run of {pinned!r}, and "
                f"{reference.step!r} runs {upstream!r}. Name a {pinned!r} step",
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


def _labels_reference_problems(
    step: Step, name: str, kind: str, walk: _Walk
) -> list[Problem]:
    """A labels reference must name one step, and that step must write labels.

    A labels field takes the kind of the conversion it names, and the run is
    pinned to the labels variant that conversion writes. A step runs over one
    labels variant, so two labels references must name one step.
    """
    found: list[Problem] = []
    named: dict[str, str] = {}
    for field_name, reference in params_step_refs(step.params).items():
        if params_reference_site(kind, name, field_name) != "labels":
            continue
        where = f"params.{field_name}"
        if isinstance(step, OpStepSpec):
            message = f"{name!r} is an op, and an op takes no labels reference"
            found.append(Problem(step.id, where, message))
            continue
        upstream = walk.declaration_behind(reference.step)
        if upstream is not None and not upstream.produces.writes_labels:
            message = (
                f"{reference.step!r} runs {upstream.produces.name!r}, which writes "
                f"no labels variant for {field_name!r} to name. A labels "
                f"reference names a convert-labels step"
            )
            found.append(Problem(step.id, where, message))
            continue
        named[field_name] = reference.step
    if len(set(named.values())) > 1:
        fields = ", ".join(sorted(named))
        message = (
            f"{fields} name {len(set(named.values()))} labels steps, and one run "
            f"reads one labels variant. Name one convert-labels step in each"
        )
        found.append(Problem(step.id, f"params.{sorted(named)[-1]}", message))
    return found


MEDIA_FIELD: Final = "media"
"""The params field that names the media variant that a step reads."""


def _media_reference_problems(step: Step, walk: _Walk) -> list[Problem]:
    """A ``media`` reference must name a step that writes a media variant.

    Every media op writes media, and only a ``preprocess`` run id names a variant
    that a consumer can read. A transcode's derivatives are routed by verdict and
    named by recipe. The validator refuses a transcode's run id in ``media``, which
    names a variant without rows.
    """
    reference = params_step_refs(step.params).get(MEDIA_FIELD)
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
            f"params.{MEDIA_FIELD}",
            f"{reference.step!r} runs {upstream.produces.name!r}, which does not "
            f"write a media variant for {MEDIA_FIELD!r} to name. A media "
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
        producer = _producer_of(item, walk)
        if producer is None:
            return None
        producers.append(producer)
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

    The one parse every reader shares. A structural fault reads the same from
    the CLI, from an API and from an editor. A pydantic error inside
    ``steps[i]`` is reported at that step's id when the document gives it one.
    The ``repeat`` items are expanded first (``expand.py``), so ``i`` counts
    the expanded steps and a fault in a repeated step names the step it became.
    A ``repeat`` that cannot be expanded names the step that declared it.

    Raises:
        RecipeInvalid: Carrying one problem per structural fault.
    """
    try:
        expanded = expand_repeats(document)
    except RepeatInvalid as exc:
        problem = Problem(exc.step, exc.where, exc.message)
        raise RecipeInvalid((problem,)) from exc
    try:
        return Recipe.model_validate(expanded)
    except ValidationError as exc:
        problems = tuple(
            _structural_problem(expanded, err["loc"], err["type"], err["msg"])
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
    # pydantic names the union member it tried after the index: the step's own
    # ``type``, not a field of it.
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
    steps = document.get("steps") if is_str_mapping(document) else None
    if not is_object_list(steps) or not 0 <= index < len(steps):
        return {}
    raw = steps[index]
    return raw if is_str_mapping(raw) else {}


def validation_report(
    document: object, catalog: DeclarationCatalog | None = None
) -> ValidationReport:
    """Everything wrong with *document*, which need not parse.

    Structural faults when it does not parse, and :func:`check_recipe`'s
    findings, the walk's cautions and its unchecked steps when it does. It does
    not raise for a fault in the document, because an editor asks after every
    edit and a list of problems is the ordinary answer.
    """
    try:
        recipe = read_recipe(document)
    except RecipeInvalid as exc:
        return ValidationReport(
            valid=False, digest=None, problems=exc.problems, cautions=(), unchecked=()
        )
    examined = _examine(recipe, catalog)
    return ValidationReport(
        valid=not examined.problems,
        digest=recipe_digest(recipe),
        problems=examined.problems,
        cautions=tuple(examined.walk.cautions),
        unchecked=tuple(examined.walk.unchecked),
    )


def resolved_levels(
    recipe: Recipe, catalog: DeclarationCatalog
) -> dict[str, EntityLevel]:
    """The entity level each step's output resolves to, keyed by step id.

    A step's level depends only on its declaration and its inputs. It is
    answered for every step whose name the catalog declares, including one
    whose params do not build. ``as-input`` resolves through the first input,
    and a step the catalog does not declare is left out; a reader of it gets
    ``individual``, the level of raw tracks.

    Raises:
        RecipeCycle: The recipe has a cycle, in which no step has an upstream.
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
    reference and a feature reads the dataset's tracks. A step type that this
    installation cannot run is refused wherever it would go, with its
    declaration's ``unavailable`` sentence.

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
    if declared.unavailable is not None:
        return _refused(declared.unavailable)
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
        reason = (
            f"{above.id!r} runs {upstream.produces.name!r}, which writes no "
            "tracks variant for a feature to read"
        )
        return _refused(reason)
    verdict = can_connect(TRACKS_DECLARATION.produces, declared.consumes)
    if not verdict:
        return _refused(verdict.reason)
    return _allowed({**base, "inputs": [TRACKS_INPUT], "tracks": {"step": above.id}})


def insert_candidates(
    recipe: Recipe, after: str | None, catalog: DeclarationCatalog | None = None
) -> InsertReport:
    """What may be inserted between *after* and the steps that read it first.

    A step type is answered as :func:`attach_candidates` answers it below
    *after*, with the same template. A step type that may be attached is still
    refused, naming the child, when a child reading it in place of *after*
    cannot read it beside its other inputs (:func:`~.compatibility.can_join`),
    or would then lack a column that one of the child's requirements names. The
    inserted output is read under the step type's default params. A requirement
    does not refuse it when the child's inputs already lack its column, or when
    the inserted output may contain the column or names unknown columns.

    The children are the feature steps whose first input reads *after*: a
    reference to it below a feature, ``tracks`` read from its variant below an
    op, and the dataset's tracks when *after* is ``None``. An editor's outline
    draws these steps below *after*. Each child reads the inserted step through
    its first input, and its other inputs stay as they are. A step that reads
    *after* through a later input only is not a child.

    A step type that every child can read is refused, naming the step, when
    the recipe with it inserted holds a problem at a step further below that
    the recipe without it does not hold. A child that inherits its input's
    entity level passes the inserted step's level down to its own readers.

    Raises:
        RecipeInvalid: The recipe has a cycle, or *after* is not a step of it.
    """
    attached = attach_candidates(recipe, after, catalog)
    walk = _walked(recipe, catalog)
    held = _problem_keys(recipe, catalog)
    above = next((step for step in recipe.steps if step.id == after), None)
    children = _children_of(recipe, above)
    if above is None or isinstance(above, OpStepSpec):
        source, level = walk.tracks, TRACKS_DECLARATION.produces.level
    else:
        source = walk.columns.get(above.id, UNKNOWN_COLUMNS)
        level = walk.level_behind(above.id)
    candidates: dict[str, Attachment] = {}
    for name, attachment in attached.candidates.items():
        declared = walk.catalog.get(name)
        if not attachment.allowed or not children or declared is None:
            candidates[name] = attachment
            continue
        inserted = _Inserted(
            name=name,
            producer=replace(
                declared.produces, level=resolve_emits(declared.emits, [level])
            ),
            columns=output_columns(
                declared.provides, merged_input([source]), declared.param_defaults
            ),
        )
        reason = next(
            (
                found
                for found in (
                    _insert_refusal(child, above, inserted, walk) for child in children
                )
                if found
            ),
            "",
        ) or _descendant_refusal(
            _with_inserted(recipe, children, attachment.template),
            held,
            catalog,
        )
        candidates[name] = _refused(reason) if reason else attachment
    return InsertReport(
        after=after,
        children=tuple(child.id for child in children),
        candidates=candidates,
    )


@dataclass(frozen=True, slots=True)
class _Inserted:
    """A step type inserted below a step, as its children would read it.

    Attributes:
        name: Its feature slug or op kind.
        producer: The producer that its children read, at its resolved level.
        columns: The columns of its output under its default params.
    """

    name: str
    producer: ProducerDecl
    columns: ColumnSet


_STEP: Final[TypeAdapter[Step]] = TypeAdapter(Step)

_INSERTED_ID: Final = "inserted"
"""The id that an inserted step takes, numbered when a step already has it."""


def _problem_keys(
    recipe: Recipe, catalog: DeclarationCatalog | None
) -> frozenset[tuple[str, str, str]]:
    """The problems that *recipe* holds, each as its step, location and message."""
    report = validation_report(recipe.model_dump(mode="json"), catalog)
    return frozenset(
        (problem.step, problem.where, problem.message) for problem in report.problems
    )


def _with_inserted(
    recipe: Recipe,
    children: Sequence[FeatureStepSpec],
    template: Mapping[str, JsonValue] | None,
) -> Recipe:
    """*recipe* with a step made from *template*, read by each of *children* first."""
    new_id = _INSERTED_ID
    number = 1
    while new_id in recipe.ids:
        new_id = f"{_INSERTED_ID}_{number}"
        number += 1
    readers = {child.id for child in children}
    steps: list[Step] = []
    for step in recipe.steps:
        if isinstance(step, FeatureStepSpec) and step.id in readers:
            inputs: list[str | StepRef] = [StepRef(step=new_id), *step.inputs[1:]]
            reads_tracks = any(isinstance(item, str) for item in inputs)
            step = step.model_copy(
                update={
                    "inputs": inputs,
                    "tracks": step.tracks if reads_tracks else None,
                }
            )
        steps.append(step)
    steps.append(_STEP.validate_python({**(template or {}), "id": new_id}))
    return recipe.model_copy(update={"steps": steps})


def _descendant_refusal(
    inserted: Recipe,
    held: frozenset[tuple[str, str, str]],
    catalog: DeclarationCatalog | None,
) -> str:
    """The first problem that *inserted* holds below the new step, ``""`` when none.

    A problem that the recipe held before the insertion, and a problem at the
    new step itself, are left out. The new step is the last of *inserted*.
    """
    new_id = inserted.steps[-1].id
    report = validation_report(inserted.model_dump(mode="json"), catalog)
    below = descendants_of(inserted, new_id) - {new_id}
    for problem in report.problems:
        key = (problem.step, problem.where, problem.message)
        if problem.step in below and key not in held:
            return f"{problem.step!r} would then be invalid: {problem.message}"
    return ""


def _reads_first(step: FeatureStepSpec, above: Step | None) -> bool:
    """Whether the first input of *step* reads the output of *above*."""
    if not step.inputs:
        return False
    first = step.inputs[0]
    if above is None:
        return isinstance(first, str) and step.tracks is None
    if isinstance(above, OpStepSpec):
        return (
            isinstance(first, str)
            and step.tracks is not None
            and step.tracks.step == above.id
        )
    return isinstance(first, StepRef) and first.step == above.id


def _children_of(recipe: Recipe, above: Step | None) -> list[FeatureStepSpec]:
    """The feature steps whose first input reads *above*, in topological order."""
    return [
        step
        for step in topological_order(recipe)
        if isinstance(step, FeatureStepSpec) and _reads_first(step, above)
    ]


def _insert_refusal(
    child: FeatureStepSpec, above: Step | None, inserted: _Inserted, walk: _Walk
) -> str:
    """Why *child* cannot read *inserted* first in place of *above*, ``""`` when it can."""
    declared = walk.catalog.get(child.feature)
    if declared is None:
        return ""
    producers: list[ProducerDecl] = []
    now: list[ColumnSet] = []
    then: list[ColumnSet] = []
    for position, item in enumerate(child.inputs):
        columns = (
            walk.tracks_of(child.id)
            if isinstance(item, str)
            else walk.input_columns(item)
        )
        now.append(columns)
        if position == 0:
            producers.append(inserted.producer)
            then.append(inserted.columns)
            continue
        producer = _producer_of(item, walk)
        if producer is None:
            return ""
        producers.append(producer)
        then.append(columns)
    verdict = can_join(producers, declared.consumes)
    if not verdict:
        return f"{child.id!r} cannot read {inserted.name}: {verdict.reason}"
    values = walk.values.get(child.id)
    if values is None:
        values = {**declared.param_defaults, **child.params}
    stack = declared.consumes.stacks_inputs
    merged_now = merged_input(now, stack=stack)
    merged_then = merged_input(then, stack=stack)
    lost = [
        need.label
        for need in needs_of(declared.requires or (), values)
        if need_status(need, merged_then) == "absent"
        and need_status(need, merged_now) != "absent"
    ]
    if not lost:
        return ""
    reading = above.id if above is not None else TRACKS_INPUT
    return (
        f"{child.id!r} would read {inserted.name} in place of {reading!r}, and its "
        f"merged input would then lack {'; '.join(lost)}"
    )


def resolved_columns(
    recipe: Recipe, catalog: DeclarationCatalog | None = None
) -> dict[str, ColumnSet]:
    """The output columns of each feature step that checked out, keyed by step id.

    They are the columns that validation checks each step's requirements
    against. An undeclared step's output is :data:`~.columns.UNKNOWN_COLUMNS`.
    A step that did not check out is left out, and the steps below it read its
    output as ``UNKNOWN_COLUMNS`` too.

    Raises:
        RecipeInvalid: The recipe has a cycle.
    """
    return dict(_walked(recipe, catalog).columns)


def input_candidates(
    recipe: Recipe, step: str, catalog: DeclarationCatalog | None = None
) -> InputCandidates:
    """The steps that may be added as an input of *step*, and its unmet requirements.

    An editor asks this while a step's inputs are being added. Each step outside
    *step*'s descendants and current inputs is answered, and ``tracks`` too, by
    :func:`~.compatibility.join_verdicts` beside the inputs already wired in.
    Each answer lists the columns of the candidate's output and the step's unmet
    requirements that it would meet.

    A requirement that the current inputs do not meet is reported as still
    needed. It is not a reason to refuse a candidate, because the next wire into
    a join may supply it. A candidate resolving to the run of a current input is
    refused, with the problem that validation reports for the wire.

    Raises:
        RecipeInvalid: The recipe has a cycle, *step* is not a step of it, or
            *step* does not run a registered feature.
    """
    walk = _walked(recipe, catalog)
    target = next((each for each in recipe.steps if each.id == step), None)
    if target is None:
        raise RecipeInvalid((Problem("", "step", f"no step {step!r} in this recipe"),))
    if not isinstance(target, FeatureStepSpec):
        message = f"{step!r} runs an op, and an op does not take pipeline inputs"
        raise RecipeInvalid((Problem(step, "kind", message),))
    declared = walk.catalog.get(target.feature)
    if declared is None or declared.produces.kind != "feature":
        message = f"no registered feature is named {target.feature!r}"
        raise RecipeInvalid((Problem(step, "feature", message),))

    values = walk.values.get(step)
    if values is None:
        # The step did not build. Its params as written, over the defaults,
        # stand in, and a value of the wrong shape resolves as unknown.
        values = {**declared.param_defaults, **target.params}
    current = [walk.input_columns(item) for item in target.inputs]
    stack = declared.consumes.stacks_inputs
    merged_now = merged_input(current, stack=stack)
    still: list[tuple[NamedColumns, UnmetStatus]] = []
    for need in needs_of(declared.requires or (), values):
        status = need_status(need, merged_now)
        if status != "guaranteed":
            still.append((need, status))

    existing = [
        producer
        for producer in (_producer_of(item, walk) for item in target.inputs)
        if producer is not None
    ]
    taken = {_input_label(item) for item in target.inputs}
    excluded = descendants_of(recipe, step)
    offered: dict[str, ProducerDecl] = {}
    supplied: dict[str, tuple[JsonValue, ColumnSet]] = {}
    if TRACKS_INPUT not in taken:
        offered[TRACKS_INPUT] = TRACKS_DECLARATION.produces
        supplied[TRACKS_INPUT] = (TRACKS_INPUT, walk.tracks)
    for other in topological_order(recipe):
        if other.id in excluded or other.id in taken:
            continue
        upstream = walk.catalog.get(_runs(other)[0])
        if upstream is None:
            continue
        offered[other.id] = replace(
            upstream.produces, level=walk.level_behind(other.id)
        )
        reference: JsonValue = {"step": other.id}
        supplied[other.id] = (reference, walk.columns.get(other.id, UNKNOWN_COLUMNS))

    reading: dict[_InputRun, str] = {}
    for current_item in target.inputs:
        run = walk.run_of(current_item)
        if run is not None:
            reading.setdefault(run, _input_label(current_item))

    candidates: dict[str, InputCandidate] = {}
    for key, verdict in join_verdicts(declared.consumes, offered, existing).items():
        item, columns = supplied[key]
        grown = merged_input([*current, columns], stack=stack)
        allowed, reason = verdict.allowed, verdict.reason
        run = None if key == TRACKS_INPUT else walk.run_of(StepRef(step=key))
        same = None if run is None else reading.get(run)
        if allowed and same is not None:
            # The one-run problem that validation would report for the wire.
            allowed, reason = (
                False,
                _one_run_reason([same, key], walk.runs, _SAME_TERMS),
            )
        candidates[key] = InputCandidate(
            allowed=allowed,
            reason=reason,
            input=item,
            supplies=tuple(sorted(columns.guaranteed)),
            may_supply=tuple(sorted(columns.possible - columns.guaranteed)),
            supplies_unknown=columns.unbounded,
            meets=tuple(
                need.label
                for need, _ in still
                if need_status(need, grown) == "guaranteed"
            ),
        )
    return InputCandidates(
        step=step,
        still_needs=tuple(
            StillNeeded(
                need=names_wire(need),
                status=status,
                providers=providers_of(
                    need,
                    walk.catalog,
                    walk.tracks,
                    besides=target.feature,
                    consumer=declared.consumes,
                ),
            )
            for need, status in still
        ),
        candidates=candidates,
    )


def _producer_of(item: str | StepRef, walk: _Walk) -> ProducerDecl | None:
    """What one input offers as a producer, at its resolved level.

    ``None`` for a step the catalog does not declare, which validation has
    already reported.
    """
    if isinstance(item, str):
        return TRACKS_DECLARATION.produces
    upstream = walk.declaration_behind(item.step)
    if upstream is None:
        return None
    return replace(upstream.produces, level=walk.level_behind(item.step))


def _allowed(template: dict[str, JsonValue]) -> Attachment:
    return Attachment(allowed=True, reason="", template=template)


def _refused(reason: str) -> Attachment:
    return Attachment(allowed=False, reason=reason, template=None)
