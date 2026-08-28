"""Which steps may be wired to which, answered without a dataset.

Not every feature accepts every other as input, and the same is true of ops.
Today that is discovered at run time, or not at all -- the sharpest case being a
multi-input join of mismatched entity granularity, which produces a silent
per-frame cartesian product rather than an error. A recipe is a declarative file,
so the same question can be asked **before it runs**, and a canvas can ask it
constantly while a wire is being drawn.

**Dataset-independent, and that separation is deliberate.** A canvas has no
dataset selected while its user is connecting things, so folding this into
``plan_pipeline`` would mean no connection could be checked until one was chosen.
Everything here works on *declarations* -- the small frozen records below --
rather than on registry classes, which is also what keeps this module free of the
feature-library import. ``resolve.declaration_catalog`` builds them once from
``FEATURES`` and ``OPS``; the same catalog is what an API serves so a client can
answer these questions itself, with no round trip per candidate wire.

**Two questions, because they diverge the moment a consumer takes more than one
input.** :func:`can_connect` asks whether one producer is admissible as *an*
input to a consumer. :func:`can_join` asks whether a *set* of producers can be
aligned into one frame, and that is where the cartesian-product refusal lives --
the second wire into a join is refusable when the first was not. The enumeration
a palette wants is built on both.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from ..loading import alignment_verdict
from ..types.feature import EmitsLevel

__all__ = [
    "ConsumerDecl",
    "Declaration",
    "DeclarationCatalog",
    "EntityLevel",
    "ProducerDecl",
    "TRACKS_DECLARATION",
    "Verdict",
    "can_connect",
    "can_join",
    "columns_for_level",
    "compatible_consumers",
    "compatible_producers",
    "possible_connections",
    "resolve_emits",
]

type EntityLevel = Literal["individual", "pair", "global"]
"""The vocabulary ``entity_level_of`` answers in, and what a declaration resolves to.

``"global"`` is its name for *no identity column*, which is what an aggregate over
everyone present carries. The declaration side spells the same thing
``"unidentified"``, because on a feature "global" already means a fit-then-apply
feature and one word cannot carry both.
"""

_LEVEL_COLUMNS: Mapping[EntityLevel, frozenset[str]] = {
    "individual": frozenset({"id", "frame", "time"}),
    "pair": frozenset({"id1", "id2", "frame", "time"}),
    "global": frozenset({"frame", "time"}),
}
"""The identity columns each level implies, as ``alignment_verdict`` reads them.

Only these five names decide the question. ``ALIGN_COLS`` is
``{id, frame, time, id1, id2, perspective}`` and ``entity_level_of`` looks at the id
columns alone, so ``perspective`` narrows a pair-to-pair join at the merge but
answers nothing here: it says which ordering a row is, never whether two inputs
share an entity. Adding it to ``"pair"`` would also make the declaration lie about
any pair feature that does not write one. Everything else a feature emits is
payload, and payload cannot make a join legal or illegal.
"""


def columns_for_level(level: EntityLevel) -> frozenset[str]:
    """The identity columns *level* implies, for the alignment predicate."""
    return _LEVEL_COLUMNS[level]


@dataclass(frozen=True, slots=True)
class ProducerDecl:
    """What a step offers as an input to something else.

    Attributes:
        name: The slug, ``tracks``, or an op kind -- whatever a recipe would
            write to refer to it.
        kind: What sort of step it is. A canvas renders these differently and
            some consumers accept only one of them.
        level: The **resolved** entity level of its output. Never
            ``"as-input"``: a feature declaring that has its level resolved from
            its own upstream by :func:`resolve_emits` before it becomes a
            producer, because a passthrough with nothing upstream of it has no
            level to offer.
        writes_tracks: Whether it produces a ``tracks/`` variant, which is what
            a step-level ``tracks`` reference may point at.
        writes_media: Whether it writes or relinks media other steps read. What
            makes an ordering-only edge from it meaningful, and what
            ``validate`` pairs against ``reads_media``.
    """

    name: str
    kind: Literal["feature", "op", "tracks"]
    level: EntityLevel
    writes_tracks: bool = False
    writes_media: bool = False


@dataclass(frozen=True, slots=True)
class ConsumerDecl:
    """What a step accepts as input.

    Attributes:
        name: The slug or op kind.
        kind: What sort of step it is.
        accepts_tracks: Whether the raw ``tracks`` literal is a valid input.
        accepts_features: Whether another feature's output is a valid input.
        requires_track_shape: Whether an input must carry the track frame --
            positions and pose passed through. **Advisory, and deliberately not
            enforced here.** ``TrackInputs`` accepts a ``Result`` from a
            track-*producing* feature, and that producer set is open: mosaic
            documents it as inexpressible in the type and checks it at input
            resolution, by looking at whether the resolved output carries the
            position columns. Refusing every feature here would refuse
            ``trajectory-smooth -> speed-angvel``, which is an ordinary chain --
            a false refusal, which is worse than a missed one. A canvas may
            render it as a caution.
        takes_no_inputs: Whether it must have none at all, which is what an
            ``Inputs`` requiring empty means.
        cross_joins: Whether it deliberately joins across entity levels.
            ``interaction-crop-pipeline`` merges tracks against a pair-level
            filter and groups afterwards, so its fan-out costs memory rather
            than correctness -- the one sanctioned escape from the alignment
            refusal, and closed rather than a per-feature flag.
        reads_media: Whether it opens video, which is what makes an ordering-only
            edge from a media writer worth declaring.
    """

    name: str
    kind: Literal["feature", "op"]
    accepts_tracks: bool = False
    accepts_features: bool = False
    requires_track_shape: bool = False
    takes_no_inputs: bool = False
    cross_joins: bool = False
    reads_media: bool = False


@dataclass(frozen=True, slots=True)
class Declaration:
    """One step type, from both sides.

    Kept as one record because a canvas lists steps rather than roles, and every
    feature is both: it consumes and it produces.
    """

    produces: ProducerDecl
    consumes: ConsumerDecl
    emits: EmitsLevel
    """As declared, before resolution. ``"as-input"`` survives here and not on
    ``produces``, so a caller resolving a chain has the original to work from."""
    category: str = ""
    """What sort of work this is -- the feature's category, or the op's.

    Display, and one decision: a GPU op splits by category so fair-share can pool
    training separately from inference.
    """
    resource_class: str = "cpu"
    """The bottleneck this step contends for: ``gpu``, ``heavy`` or ``cpu``.

    Read from what the step declares, so a new heavy step routes correctly by
    declaring rather than by being added to a list. It rides on the declaration
    rather than being looked up where it is needed, because the lookup would
    otherwise need the feature registry -- and deciding a lane is one of the read
    paths that must not pay for it.
    """
    scope_takes: str = ""
    """How much scope an op accepts, or ``""`` for a step that is not an op.

    One of ``mosaic.core.pipeline.ops.ScopeTakes``. A canvas reads it to say
    whether a step needs an entry named before it can run. A feature declaration
    keeps the empty default: no feature refuses a scope, and a field with one
    legal value teaches a reader nothing.
    """
    scope_dependent: bool = False
    """Whether the entries in scope decide what an op run is named.

    ``False`` for a step that is not an op. The feature-side twin lives on the
    feature class and is read from there.
    """


@dataclass(frozen=True, slots=True)
class DeclarationCatalog:
    """Every step type this installation knows, keyed by the name a recipe writes.

    JSON-serializable by construction -- frozen dataclasses of strings and bools
    -- because the same object is what an API hands a browser so the canvas can
    answer connection questions without a round trip.
    """

    entries: Mapping[str, Declaration] = field(
        default_factory=lambda: cast("dict[str, Declaration]", {})
    )

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def get(self, name: str) -> Declaration | None:
        """The declaration for *name*, or ``None`` if nothing declares it."""
        return self.entries.get(name)

    def names(self) -> tuple[str, ...]:
        """Every declared name, sorted."""
        return tuple(sorted(self.entries))


TRACKS_DECLARATION: Declaration = Declaration(
    produces=ProducerDecl(name="tracks", kind="tracks", level="individual"),
    consumes=ConsumerDecl(name="tracks", kind="feature", takes_no_inputs=True),
    emits="individual",
)
"""The dataset's standardized tracks, as a producer.

Not a step and not in any registry, but a canvas has to offer it and every chain
starts from it. Individual-level: one row per ``(frame, id)``, which is what
``mosaic_v1`` requires.
"""


def resolve_emits(
    emits: EmitsLevel, upstream: Sequence[EntityLevel] = ()
) -> EntityLevel:
    """Turn a declared level into the level a producer actually offers.

    ``"as-input"`` is the only value needing anything: it means the output is
    keyed the way the input was, so it resolves to the first upstream level. A
    passthrough with no upstream resolves to ``"individual"``, which is what
    reading raw tracks gives it -- the only thing a feature with no feature
    inputs can be reading.

    Args:
        emits: What the feature declared.
        upstream: The resolved levels of its inputs, in order.

    Returns:
        A concrete level, never ``"as-input"``.
    """
    if emits == "as-input":
        return upstream[0] if upstream else "individual"
    if emits == "unidentified":
        return "global"
    return emits


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a connection is allowed, and why not when it is not.

    Truthy when allowed, so a caller wanting a plain boolean writes
    ``if can_connect(...)``. The reason is there because a greyed-out option a
    user cannot explain is worse than one that is simply absent.
    """

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


_ALLOWED: Verdict = Verdict(allowed=True)


def can_connect(producer: ProducerDecl, consumer: ConsumerDecl) -> Verdict:
    """May *producer*'s output be **an** input to *consumer*?

    One wire in isolation: the right sort of thing, an accepted input type, and a
    level the consumer can work with at all. Whether a *set* of wires into one
    consumer can be aligned is :func:`can_join`, and a canvas drawing the second
    wire into a join needs both.
    """
    if consumer.takes_no_inputs:
        return Verdict(
            False, f"{consumer.name} takes no pipeline inputs; it reads its own"
        )
    if producer.kind == "op":
        return Verdict(
            False,
            f"{producer.name} is an op, so it produces tracks rather than a "
            f"feature output; reference it from {consumer.name}'s tracks field "
            f"instead of its inputs",
        )
    if producer.kind == "tracks" and not consumer.accepts_tracks:
        return Verdict(False, f"{consumer.name} does not read tracks directly")
    if producer.kind == "feature" and not consumer.accepts_features:
        return Verdict(
            False, f"{consumer.name} reads tracks only, not another feature's output"
        )
    # ``requires_track_shape`` is deliberately not a refusal here; see the field.
    return _ALLOWED


def can_join(producers: Sequence[ProducerDecl], consumer: ConsumerDecl) -> Verdict:
    """May *producers* together be the inputs of *consumer*?

    Where the expensive refusal lives. Two inputs at different concrete entity
    levels share no identity column, so merging them on ``frame`` alone pairs
    every row of one with every row of the other -- a per-frame cartesian
    product that raises nothing and produces a plausible table.

    Decided by ``alignment_verdict``, the same rule the merge itself enforces, so
    a chain checked here cannot be refused at run time for a reason this did not
    see. It is name-based, so no data and no dataset are read.
    """
    for producer in producers:
        single = can_connect(producer, consumer)
        if not single:
            return single
    if len(producers) < 2:
        return _ALLOWED
    if consumer.cross_joins:
        return _ALLOWED
    verdict = alignment_verdict([columns_for_level(p.level) for p in producers])
    if verdict.compatible:
        return _ALLOWED
    named = ", ".join(f"{p.name} ({p.level})" for p in producers)
    return Verdict(False, f"{consumer.name} cannot align {named}: {verdict.reason}")


def possible_connections(
    consumer: ConsumerDecl,
    catalog: DeclarationCatalog,
    existing: Sequence[ProducerDecl] = (),
    candidates: Iterable[str] | None = None,
) -> dict[str, Verdict]:
    """What may be wired into *consumer* next, given what is already wired in.

    The question a canvas actually asks, and the reason it takes *existing*: the
    second wire into a join is refusable when the first was not, so an answer
    computed from the consumer alone would offer options that stop being valid
    the moment one is taken.

    Args:
        consumer: The step being wired into.
        catalog: Every declaration this installation knows.
        existing: The producers already feeding *consumer*, resolved.
        candidates: Which names to consider. ``None`` is every declared name.

    Returns:
        One verdict per candidate, keyed by name, so a caller can render the
        refused ones greyed out with their reason rather than hiding them.
    """
    wanted = sorted(catalog.names() if candidates is None else candidates)
    answers: dict[str, Verdict] = {}
    for name in wanted:
        declared = catalog.get(name)
        if declared is None:
            answers[name] = Verdict(False, f"nothing declares {name!r}")
            continue
        answers[name] = can_join([*existing, declared.produces], consumer)
    return answers


def compatible_producers(
    consumer: ConsumerDecl,
    catalog: DeclarationCatalog,
    candidates: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Which steps may feed *consumer* as its only input, sorted.

    The palette listing. For "what may I add to what is already there", use
    :func:`possible_connections`.
    """
    answers = possible_connections(consumer, catalog, candidates=candidates)
    return tuple(name for name, verdict in answers.items() if verdict)


def compatible_consumers(
    producer: ProducerDecl,
    catalog: DeclarationCatalog,
    candidates: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Which steps may read *producer*'s output, sorted.

    The forward listing, and the exact mirror of :func:`compatible_producers`:
    a producer appears in one iff the consumer appears in the other.
    """
    wanted = sorted(catalog.names() if candidates is None else candidates)
    found: list[str] = []
    for name in wanted:
        declared = catalog.get(name)
        if declared is not None and can_connect(producer, declared.consumes):
            found.append(name)
    return tuple(found)
