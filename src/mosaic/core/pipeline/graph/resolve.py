"""The one module in this package that reaches the registries.

Everything else here works on declarations -- small frozen records of strings and
booleans -- and this is where those come from. The split is not tidiness:
importing ``FEATURES`` imports every feature module and, through them, scipy,
sklearn and pywt, which is seconds of wall clock. Validating a recipe, ordering
it, listing a step's parents, deciding a lane, rendering a status view,
evaluating a release gate and cancelling must all work without paying it, and the
gate runs far more often than a submit does.

So the registry imports live **inside the functions**, not at module scope, and
``tests/test_graph_imports.py`` holds both halves of that line.

**Declarations are read off what a class already declares**, never off its name.
Whether a feature accepts tracks is answered by validating a probe payload
against its own ``Inputs`` model, so the answer cannot drift from what the
feature will actually accept; whether an op writes tracks is answered by the
tracking-roots registry, which is the table a producer has to fill in to exist at
all.
"""

from __future__ import annotations

from functools import lru_cache

from ..types.feature import EmitsLevel
from .compatibility import (
    TRACKS_DECLARATION,
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    ProducerDecl,
    resolve_emits,
)
from .model import TRACKS_INPUT

__all__ = ["declaration_catalog", "feature_class_for_slug"]

_EMITS_LEVELS: frozenset[str] = frozenset(
    {"individual", "pair", "unidentified", "as-input"}
)
"""The vocabulary a feature may declare, for narrowing what ``getattr`` returns."""

_PROBE_FEATURE: str = "__probe__"
"""A feature name no registry holds, used to ask an ``Inputs`` model a question.

Validating ``[{"feature": ...}]`` against a feature's own ``Inputs`` answers
"does this accept another feature's output" using the real validator rather than
by introspecting a generic parameter. The name never escapes this module.
"""


def feature_class_for_slug(slug: str) -> type | None:
    """The registered feature class whose ``name`` is *slug*, or ``None``.

    ``FEATURES`` is keyed by class name while a recipe and the CLI both name a
    feature by its ``name`` slug, so the lookup is a scan. One copy, here: it was
    spelled three times -- in the CLI, in this package's caller, and in
    mosaic-api's lane routing -- and a fourth would have been written the next
    time something needed to turn a slug into a class.

    Returns ``None`` rather than raising, because both callers want to phrase
    their own refusal: the CLI lists the available slugs, and recipe validation
    names the step the unknown slug sits in.
    """
    from mosaic.behavior.feature_library import FEATURES

    for cls in FEATURES.values():
        if getattr(cls, "name", None) == slug:
            return cls
    return None


def _accepts(inputs_cls: type, payload: list[object]) -> bool:
    """Would this ``Inputs`` model accept *payload*?

    Asked of the model rather than of its type parameter. A feature narrows its
    inputs by subclassing ``Inputs[...]`` and sometimes by a validator on top, so
    the specialization alone is not the whole answer -- and reading it back at
    run time means reading pydantic's generic metadata, which would be a second
    description of a rule the validator already holds.
    """
    validate = getattr(inputs_cls, "model_validate", None)
    if validate is None:
        return False
    try:
        _ = validate(payload)
    except Exception:
        return False
    return True


def _feature_declaration(cls: type) -> Declaration:
    """Read one feature class as a producer and a consumer."""
    from mosaic.core.pipeline.loading import CROSS_JOIN_FEATURES

    slug = str(getattr(cls, "name", cls.__name__))
    inputs_cls: type | None = getattr(cls, "Inputs", None)
    require = str(getattr(inputs_cls, "_require", "nonempty"))
    roots: tuple[str, ...] = tuple(
        str(root) for root in getattr(cls, "consumed_roots", ())
    )
    # Every registered feature declares this and a conformance test holds that,
    # so the fallback is for a class handed in from outside the registry.
    declared: object = getattr(cls, "emits", "as-input")
    emits: EmitsLevel = declared if declared in _EMITS_LEVELS else "as-input"  # pyright: ignore[reportAssignmentType]

    accepts_tracks = inputs_cls is not None and _accepts(inputs_cls, [TRACKS_INPUT])
    accepts_features = inputs_cls is not None and _accepts(
        inputs_cls, [{"feature": _PROBE_FEATURE}]
    )
    return Declaration(
        produces=ProducerDecl(
            name=slug,
            kind="feature",
            # Unresolved here on purpose: a chain resolves a passthrough from its
            # own upstream, and a producer offered to a canvas with nothing above
            # it reads as individual, which is what raw tracks give it.
            level=resolve_emits(emits),
            writes_tracks=False,
            writes_media=False,
        ),
        consumes=ConsumerDecl(
            name=slug,
            kind="feature",
            accepts_tracks=accepts_tracks,
            accepts_features=accepts_features,
            requires_track_shape=bool(getattr(inputs_cls, "_track_input", False)),
            takes_no_inputs=require == "empty",
            cross_joins=slug in CROSS_JOIN_FEATURES,
            reads_media="media_raw" in roots,
        ),
        emits=emits,
        category=str(getattr(cls, "category", "")),
        # Optional on a feature and defaulted to cpu, which is what every
        # feature but a declared-heavy one is.
        resource_class=str(getattr(cls, "resource_class", "") or "cpu"),
    )


def _op_declaration(kind: str, op_cls: type) -> Declaration:
    """Read one op as a producer and a consumer.

    An op is never a feature's *input* in the ``inputs`` sense -- what a feature
    reads from a tracker is the ``tracks/`` variant it produced, referenced from
    the step's ``tracks`` field -- so its producer record exists to say that, and
    to carry the two facts an ordering-only edge is checked against.

    ``writes_tracks`` comes from the tracking-roots registry, which is the table
    a producer must appear in to bridge into ``tracks/`` at all, and
    ``writes_media`` from the op's own declared category. Neither is a name list.
    """
    from mosaic.core.pipeline.ops import op_resource_class
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

    bridges_tracks = kind in TRACKING_ROOTS
    return Declaration(
        produces=ProducerDecl(
            name=kind,
            kind="op",
            level="individual",
            writes_tracks=bridges_tracks,
            writes_media=str(getattr(op_cls, "category", "")) == "transcode",
        ),
        consumes=ConsumerDecl(
            name=kind,
            kind="op",
            # An op declares its scope in its params rather than taking pipeline
            # inputs, so nothing may be wired into it. What it depends on is
            # expressed as a params reference or as an ordering-only edge.
            takes_no_inputs=True,
            # Every producer that bridges into tracks opens video to do it, which
            # is what makes an ordering-only edge from a media writer meaningful.
            reads_media=bridges_tracks,
        ),
        emits="individual",
        category=str(getattr(op_cls, "category", "")),
        resource_class=op_resource_class(kind),
    )


@lru_cache(maxsize=1)
def declaration_catalog() -> DeclarationCatalog:
    """Every step type this installation knows, as declarations.

    Cached for the process, which is safe because both registries are filled by
    import side effect and are fixed once the modules that populate them have
    run. Anything registering a feature after the first call would not appear,
    and nothing does that outside a test.

    The result is JSON-serializable by construction, so an API can hand it
    straight to a browser and the canvas answers connection questions locally
    rather than asking per candidate wire.
    """
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.ops import OPS
    from mosaic.tracking import register_ops

    # The generic registry holds no tracking kinds until their subpackages are
    # imported; this is idempotent and is what every other reader of OPS does.
    register_ops()

    entries: dict[str, Declaration] = {TRACKS_INPUT: TRACKS_DECLARATION}
    for cls in FEATURES.values():
        declared = _feature_declaration(cls)
        entries[declared.produces.name] = declared
    for kind, op_cls in OPS.items():
        entries[kind] = _op_declaration(kind, op_cls)
    return DeclarationCatalog(entries=entries)
