"""Registry of ops under the Job Contract.

Every long-running operation that rides the Job Contract -- frame extraction,
pose/point/localizer training, pose/point/localizer inference, annotation
conversion, TREx, media transcode, and future domains -- is an ``Op``: a class
carrying a ``kind`` (its ``runs.kind``), a ``domain`` (which package owns it),
a Pydantic ``Params`` model, and a ``run(ds, params, scope, overwrite, ctx)``
body that computes a content ``run_id``, does the work, writes an index row,
and returns the ``run_id``. What a run covers and whether it redoes the work
are arguments rather than params fields, because they describe an attempt and
not a recipe. Ops self-register via ``@register_op`` -- so a new op plugs in by
adding a module, with no edit to the runner, the CLI, or the API.

One generic entry point, :func:`run_op`, wraps *every* op in the Job Contract
(`core/pipeline/job.py`), so attempt-recording, progress, heartbeat, and
cooperative cancellation are written once. Because each op declares a Pydantic
``Params``, discovery is schema-driven -- ``op.Params.model_json_schema()``
gives a CLI / mosaic-api / MCP a full param spec exactly the way features are
discovered today, with zero per-op schema code.

**Registration must stay import-light.** Op modules import only their ``Params``
and light deps at module top; heavy backends (``ultralytics`` / ``torch`` /
POLO) are imported *inside* ``run()`` so importing an op package never fails
when an optional extra is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    TypeVar,
    get_args,
)

from mosaic.core.pipeline.job import CancelToken, JobContext, job_context
from mosaic.core.params import Params

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.progress import ProgressCallback
    from mosaic.core.scope import Scope


# ---------------------------------------------------------------------------
# Identity, resolvable before the work happens
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpIdentity:
    """What one op run will be called, worked out without running it.

    Every op mints its own identifier, and it used to do so **inside** ``run()``
    -- so the only way to learn what a run would be named was to perform it.
    Planning a graph needs the answer first: a step's identity is what its
    downstream steps hash, so a chain resolves in one topological walk or not at
    all.

    Attributes:
        run_id: The content-addressed op run identifier.
        tracks_variant: What names the ``tracks/`` tables this run produces, or
            ``""`` for an op that writes none. **A different identifier from**
            ``run_id``, even where the two coincide: the tracker variant payload
            is an unwrapped passthrough of the settings today, so they are
            byte-identical, and reading one as the other is a latent bug rather
            than a shortcut.
        model_run_id: What names the model a training op produces, or ``""``.
            Separate because a downstream inference step references the *model*,
            and a training run and its model need not always be one name.
    """

    run_id: str
    tracks_variant: str = ""
    model_run_id: str = ""


class IdentityDeferred(Exception):
    """This op's identity needs an artifact an upstream step has not written yet.

    Not a failure and not a bug: a few ops hash the *content* of what they read
    -- a training op fingerprints its dataset directory, so that two runs over
    changed annotations are two models -- and when that directory is itself
    produced by an earlier step in the same graph, there is nothing on disk to
    fingerprint at planning time.

    The honest answer is to say so. Guessing an identifier would produce one that
    execution then contradicts, and a plan is read as a preview: a wrong
    identifier there is a wrong answer, where an absent one is a stated
    limitation. Nothing downstream is blocked by it, because a resolved
    identifier is never load-bearing at execution -- every step resolves its own
    at its own start.
    """

    def __init__(self, kind: str, because: str) -> None:
        self.kind: str = kind
        self.because: str = because
        super().__init__(f"{kind}: identity is not resolvable yet -- {because}")


# ---------------------------------------------------------------------------
# Op spec + registry
# ---------------------------------------------------------------------------

P = TypeVar("P", bound=Params)


type ScopeTakes = Literal["none", "any", "at-least-one", "exactly-one"]
"""How much scope one op accepts.

- ``"none"`` -- reads a prepared directory or a training set, never a scope.
- ``"any"`` -- any scope, and an unset one covers every indexed entry.
- ``"at-least-one"`` -- refuses an unset selector and a resolution of zero.
- ``"exactly-one"`` -- the above, and refuses a resolution of more than one.
"""

SCOPE_TAKES_VALUES: Final[frozenset[str]] = frozenset(get_args(ScopeTakes.__value__))
"""The legal values, read off :data:`ScopeTakes` rather than restated.

A runtime check needs the same vocabulary the annotation declares. A second
literal set is free to drift from the first, and the direction it drifts in is
the harmful one: a value the checker rejects and the registry accepts.
"""


class ScopeRefused(ValueError):
    """A scope an op's declaration does not accept.

    A ``ValueError`` because that is what the command line already renders as a
    message. ``mosaic run`` and the three ``mosaic pipeline`` verbs catch it by
    name and print it beside the flags they offer. ``mosaic track`` does not,
    and needs no handler. Every tracker declares ``scope_takes = "any"``, for
    which this function returns before refusing anything.

    A graph step does **not** record it on the attempt. The failure record wraps
    the feature arm of ``execute_step`` only. A refusal from an op step
    propagates out uncaught. It is raised before any work starts, and the
    planner raises the same refusal earlier for the same step.

    The subclass exists so a caller that wants to tell a scope refusal apart
    from an invalid parameter can.
    """


_UNSCOPED_CONSEQUENCE: Final[dict[str, str]] = {
    "transcode": "re-encode every video in the dataset",
    "export-store": "export every imgstore in the dataset",
}
"""What an unscoped run of one op covers, in the words its refusal uses.

Named per op because the count is what a person weighs. An op absent from this
table falls back to a general phrasing rather than to silence.
"""


def check_scope_takes(
    kind: str, scope_takes: ScopeTakes, resolved: "ResolvedScope"
) -> None:
    """Raise unless *resolved* is a scope an op declaring *scope_takes* accepts.

    **The only place a scope refusal is raised.** An op implements no scope
    validation and writes no scope message.

    Reads the selector as well as the resolved entries. Three scopes resolve to
    zero entries and mean three things. An unset selector covers every indexed
    entry, an empty entry list names none, and a selector naming an absent group
    was looked up and missed. Each gets its own refusal, and only the first is
    what ``"at-least-one"`` exists to refuse. ``"none"`` reads the selector alone
    and opens no index.

    Args:
        kind: The op's registered kind, named in every message.
        scope_takes: The op's declaration. See :data:`ScopeTakes`.
        resolved: What the caller's scope resolved to.

    Raises:
        ScopeRefused: The scope is one this declaration does not accept.
    """
    selector = resolved.selector
    count = len(resolved.entries)

    if scope_takes == "none":
        if not selector.is_unset:
            message = (
                f"{kind} takes no entry scope. It reads a prepared directory "
                f"rather than a set of sequences. Pass Scope(), or omit the "
                f"scope argument."
            )
            raise ScopeRefused(message)
        return

    if scope_takes == "any":
        return

    if selector.is_unset:
        consequence = _UNSCOPED_CONSEQUENCE.get(kind, "cover every indexed entry")
        # A group selector is offered only where more than one entry is
        # acceptable. For an exactly-one op it resolves to several on any
        # populated dataset and meets the refusal below.
        if scope_takes == "exactly-one":
            remedy = "Pass Scope(entries=[(group, sequence)]) naming the one entry."
        else:
            remedy = (
                "Pass Scope(entries=[(group, sequence), ...]) or Scope(groups=[...])."
            )
        message = (
            f"{kind} must be told which entries to cover. An unset scope names "
            f"every indexed entry, and an unscoped run would {consequence}. "
            f"{remedy}"
        )
        raise ScopeRefused(message)
    if selector.entries is not None and not selector.entries:
        message = (
            f"{kind} was given an empty entry list, which names no entry to "
            f"cover. Name at least one, as "
            f"Scope(entries=[(group, sequence), ...])."
        )
        raise ScopeRefused(message)
    if count == 0:
        message = (
            f"{kind} was given {selector!r}, which matches no entry in the "
            f"media index. Check the names against 'mosaic sequences'."
        )
        raise ScopeRefused(message)
    if scope_takes == "exactly-one" and count > 1:
        listed = ", ".join(
            f"({group!r}, {sequence!r})" for group, sequence in sorted(resolved.entries)
        )
        message = (
            f"{kind} covers one entry and this scope resolves {count}: "
            f"{listed}. Narrow the scope to one entry, or run {kind} once per "
            f"entry."
        )
        raise ScopeRefused(message)


class Op(Generic[P]):
    """Base class for a registered op.

    Generic over its ``Params`` type so a subclass can narrow ``run``/``target``
    without an LSP-incompatible override. Subclasses set the class attributes,
    implement :meth:`run`, and are stateless (the registry stores the class and
    instantiates it per call).
    """

    kind: ClassVar[str]
    category: ClassVar[str]  # "extract" | "train" | "infer" | "convert" | "transcode"
    domain: ClassVar[str]  # "tracking" | "media"
    version: ClassVar[str] = "0.1"
    # Compute-placement hint for schedulers / the execution router ("gpu" | "heavy" |
    # "cpu"). Empty ("") derives it from ``category`` (train/infer -> gpu, else cpu) via
    # :func:`op_resource_class`; an op overrides it when category is misleading (e.g. TREx
    # is category "convert" but needs the GPU for YOLO detection, so it declares "gpu").
    resource_class: ClassVar[str] = ""
    # Whether this op writes into ``tracks/``, which is what lets a downstream
    # recipe step wire its ``tracks`` reference to this one. Declared rather than
    # inferred: it used to be read as ``kind in TRACKING_ROOTS``, which is the
    # table a producer must appear in to *bridge from a tracker run root* -- true
    # of every tracks producer there was, and false for one that reads a tracks
    # table and writes another. ``None`` keeps the old inference, so no existing
    # op declares anything and none of their declarations move.
    writes_tracks: ClassVar[bool | None] = None
    scope_takes: ClassVar[ScopeTakes]
    """How much scope this op accepts. See :data:`ScopeTakes`.

    Declared rather than inferred from the params model. Inference let four ops
    spell one question four ways: a required list, a singular field, a nullable
    default and an absence. One shared validator reads this and raises every
    refusal, and an op implements no scope checking of its own. A wrong value
    here either refuses a run the op can do or admits one it cannot.
    """

    scope_dependent: ClassVar[bool]
    """Whether the set of entries in scope decides what this run is named.

    The op-side twin of the feature declaration ``compute_run_id`` demands.
    ``True`` where :meth:`plan_identity` reads the scope: ``transcode`` and
    ``export-store`` hash the resolved source identities, and ``resample-tracks``
    resolves the tracks variant it chains from by filtering the index with the
    scope. A ``False`` written on an op whose identity does move mints one
    identifier for two different computations.
    """

    Params: ClassVar[type[Params]]

    def target(self, params: P, scope: "ResolvedScope") -> str:
        """A short human label for the ``runs.target`` column.

        Args:
            params: The validated params this run uses.
            scope: What this run covers, for a label that names its entries.

        Returns:
            The label recorded under ``runs.target``.
        """
        return self.kind

    def plan_identity(
        self,
        ds: "Dataset",
        params: P,
        scope: "ResolvedScope",
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this run will be called, without doing any of it.

        **The one place this op's identity is minted**, and ``run`` calls it
        rather than computing the same thing a second way. Two answers to "what
        will this run be named" is the shape of mistake that reports a cache hit
        over another run's output, and the second copy is always the one that
        gets forgotten when a payload changes.

        It may read the dataset -- a transcode identity covers the source videos'
        recorded identities, and a tracker's covers the content digest of the
        weights it was pointed at -- but it must not write, and it must not do
        the work.

        Args:
            ds: The dataset, for the recorded facts the identity covers.
            params: The validated params this run would use.
            scope: What this run would cover. An op whose identity moves with
                its coverage reads it, and ``scope_dependent`` declares which
                ops those are.
            require_data: Whether a training input that is not on disk defers
                the identity. A planner asks whether the run is nameable at all
                and must be told when it is not. An execution has its data
                already, and the refusal for a missing file belongs to the tool
                that reads it. An op whose identity reads a dataset fact it
                cannot obtain refuses regardless of this flag.

        Returns:
            The identifiers this run would produce.

        Raises:
            IdentityDeferred: The payload needs an artifact an upstream step has
                not written yet.
        """
        raise NotImplementedError(
            f"op {self.kind!r} ({type(self).__name__}) does not implement "
            f"plan_identity, so a graph cannot say what it will produce or "
            f"whether it has already run. Implement it, and have run() call it "
            f"rather than minting a second identifier of its own."
        )

    def run(
        self,
        ds: "Dataset",
        params: P,
        scope: "ResolvedScope",
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        """Do the work using *ctx* (progress/cancel/run_id) and return the run_id.

        Args:
            ds: The dataset the work is done against.
            params: The validated settings this run uses.
            scope: What this run covers.
            overwrite: Whether to recompute what is already there. Two attempts
                differing only in it produce the same run identifier.
            ctx: The Job-Contract context that reports progress, answers a
                cancellation and records the run identifier.

        Returns:
            The content ``run_id`` this run produced.
        """
        raise NotImplementedError


OPS: dict[str, type[Op[Any]]] = {}


def register_op(cls: type[Op[Any]]) -> type[Op[Any]]:
    """Class decorator: register *cls* under its ``kind``.

    Raises:
        ValueError: *cls* defines no non-empty ``kind``.
        TypeError: *cls* omits ``scope_takes`` or ``scope_dependent``, or
            declares a ``scope_takes`` outside :data:`SCOPE_TAKES_VALUES`.
    """
    if not getattr(cls, "kind", None):
        raise ValueError(f"{cls.__name__} must define a non-empty 'kind'")
    for name in ("scope_takes", "scope_dependent"):
        if not hasattr(cls, name):
            raise TypeError(
                f"{cls.__name__} declares no {name!r}. Every op states how much "
                f"scope it takes and whether coverage decides its output, and a "
                f"missing declaration cannot be guessed from the params model. "
                f"See mosaic.core.pipeline.ops.ScopeTakes."
            )
    if cls.scope_takes not in SCOPE_TAKES_VALUES:
        raise TypeError(
            f"{cls.__name__}.scope_takes is {cls.scope_takes!r}, which is not "
            f"one of {sorted(SCOPE_TAKES_VALUES)}."
        )
    OPS[cls.kind] = cls
    return cls


# ---------------------------------------------------------------------------
# Generic runner (the single Job-Contract wrapper for all ops)
# ---------------------------------------------------------------------------


def run_op(
    ds: "Dataset",
    kind: str,
    params: Params | dict[str, Any],
    *,
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: "ProgressCallback | None" = None,
    cancel_token: CancelToken | None = None,
    scope: "Scope | None" = None,
    overwrite: bool = False,
) -> str:
    """Run a registered op as a tracked Job-Contract attempt.

    *params* may be a validated ``Params`` instance or a plain dict (validated
    against the op's ``Params`` model). Returns the content ``run_id``.

    Args:
        ds: The dataset the op runs against.
        kind: The registered op kind.
        params: A validated ``Params`` instance, or a dict this validates.
        execution_id: An externally minted ULID to reuse.
        owner: Who to record the attempt under.
        track: Whether to write a run-log for the attempt.
        progress_callback: Where per-entry progress is reported.
        cancel_token: How a caller asks the run to stop.
        scope: What to cover. Resolved through
            :meth:`~mosaic.core.dataset.Dataset.resolve_scope`, checked against
            the op's ``scope_takes`` declaration, and handed to the op.
        overwrite: Whether the op recomputes what is already there. Every op
            reads it from here. Two attempts differing only in it produce the
            same run identifier.

    Returns:
        The content ``run_id`` the op produced.

    Raises:
        KeyError: *kind* names no registered op.
        ScopeRefused: *scope* is one the op's declaration does not accept.
        FileNotFoundError: *scope* names groups or sequences and the originals
            index does not exist.
    """
    op_cls = OPS.get(kind)
    if op_cls is None:
        raise KeyError(f"Unknown op '{kind}'. Registered: {sorted(OPS)}")
    op = op_cls()
    p = op.Params.model_validate(params) if isinstance(params, dict) else params
    # The seam every op reads its scope from, and the one place a scope is
    # refused. Resolved here so one enumeration answers for every op, and checked
    # here so no op implements an arity rule of its own.
    resolved = ds.resolve_scope(scope)
    check_scope_takes(kind, op.scope_takes, resolved)
    with job_context(
        ds,
        kind=kind,
        target=op.target(p, resolved),
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    ) as ctx:
        return op.run(ds, p, resolved, overwrite, ctx)


# ---------------------------------------------------------------------------
# Discovery (schema-driven; consumed by the CLI / mosaic-api / MCP)
# ---------------------------------------------------------------------------


def list_ops(
    category: str | None = None, domain: str | None = None
) -> list[dict[str, object]]:
    """Enumerate registered ops as one dict each.

    Each row carries ``kind``, ``domain``, ``category``, ``version`` and the two
    scope declarations. A client reads how much scope an op takes without
    running it, and without reading it out of the params model.
    """
    ops = sorted(OPS.values(), key=lambda c: c.kind)
    return [
        {
            "kind": c.kind,
            "domain": c.domain,
            "category": c.category,
            "version": c.version,
            "scope_takes": c.scope_takes,
            "scope_dependent": c.scope_dependent,
        }
        for c in ops
        if (category is None or c.category == category)
        and (domain is None or c.domain == domain)
    ]


_CATEGORY_RESOURCE_CLASS: dict[str, str] = {
    "train": "gpu",
    "infer": "gpu",
    "extract": "cpu",
    "convert": "cpu",
    "transcode": "cpu",
}


def op_resource_class(kind: str) -> str:
    """Return an op's compute-placement class (``"gpu"`` | ``"heavy"`` | ``"cpu"``).

    Prefers the op's explicit ``resource_class`` classvar; otherwise derives it from
    ``category`` (train/infer -> gpu, else cpu). Unknown kinds fall back to ``"cpu"``. Used by
    the execution router to send GPU work (training, inference, TREx) to a GPU lane / k8s
    without any per-op routing edits.
    """
    op_cls = OPS.get(kind)
    if op_cls is None:
        return "cpu"
    declared = getattr(op_cls, "resource_class", "")
    if declared:
        return declared
    return _CATEGORY_RESOURCE_CLASS.get(op_cls.category, "cpu")


def describe_op(kind: str) -> dict[str, object]:
    """Describe one op, as its identity, its declarations and its params schema.

    The two scope declarations sit beside ``kind`` and never inside
    ``params_schema``. They describe the op rather than its params, and a client
    drawing controls from that schema draws only fields a caller fills in.

    Raises:
        KeyError: *kind* names no registered op.
    """
    op_cls = OPS.get(kind)
    if op_cls is None:
        raise KeyError(f"Unknown op '{kind}'. Registered: {sorted(OPS)}")
    return {
        "kind": op_cls.kind,
        "domain": op_cls.domain,
        "category": op_cls.category,
        "version": op_cls.version,
        "scope_takes": op_cls.scope_takes,
        "scope_dependent": op_cls.scope_dependent,
        "params_schema": op_cls.Params.model_json_schema(),
    }
