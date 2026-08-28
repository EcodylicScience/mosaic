from __future__ import annotations

from typing import Annotated, Generic, Self

from pydantic import model_validator
from typing_extensions import TypeVar

from mosaic.core.entry import Entry
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline.types.artifacts import JoblibArtifact, TemplatesRef

# ``JsonValue`` used to live here. It moved to ``mosaic.core.json_value`` -- a
# module with no imports at all -- because the dataset manifest needs the same
# type and importing this one drags the loader and artifact machinery, and
# through them pandas, into a manifest read. The package ``__init__`` re-exports
# it from its new home, so every existing import path is unchanged.


_ENTRIES_DESCRIPTION = (
    "Which (group, sequence) entries the run covers. Unset covers every entry "
    "the media index holds."
)

_OVERWRITE_DESCRIPTION = (
    "Recompute an entry whose output is already on disk, instead of keeping "
    "what is there and reporting it done."
)


class OpParams(Params):
    """The scope an op runs over, for every op that runs over a set of entries.

    An op that takes no scope at all -- one reading a labelled export, or a
    training set -- stays on :class:`Params`, and so does one whose arity is a
    single entry, because both fields here would then be accepted and never
    read.

    Both fields are ``HASH_EXCLUDE``. A selector names which entries a run
    covers rather than what comes out of any one of them, and an op whose
    coverage does change its output hashes the resolved entry identities in
    ``plan_identity`` instead -- which is also what keeps a sequence rename from
    moving a run identifier.

    One selector, and it enumerates. A ``groups`` field beside a ``sequences``
    one expresses a cross product, so it cannot name group A sequence 1 together
    with group B sequence 2 without also naming A/2 and B/1. ``mosaic track``
    takes the pair as flags and ``mosaic run --kind`` takes it inside
    ``--params``. Both resolve it through
    :meth:`~mosaic.core.dataset.Dataset.resolve_scope`, which enumerates the
    cross product against the media index, before an op's model validates.

    An op that refuses an unscoped run has a choice, and it is a trade rather
    than a constraint. Inheriting this field keeps the base and refuses ``None``
    and ``[]`` in a validator, at the cost of an ``entries`` still typed
    ``list[Entry] | None`` for every reader and an ``overwrite`` the op may not
    act on; declaring its own required ``entries`` on :class:`Params` buys the
    static type and gives up the shared base, because a subclass can neither
    drop an inherited default nor narrow an inherited type.
    :class:`~mosaic.core.pipeline.transcode.TranscodeParams` takes the second.
    """

    entries: Annotated[
        list[Entry] | None, HASH_EXCLUDE, Declared(_ENTRIES_DESCRIPTION)
    ] = None
    overwrite: Annotated[bool, HASH_EXCLUDE, Declared(_OVERWRITE_DESCRIPTION)] = False


M = TypeVar("M", bound=JoblibArtifact[object], default=JoblibArtifact[object])
T = TypeVar("T", bound=TemplatesRef, default=TemplatesRef)

_TEMPLATES_DESCRIPTION = (
    "The templates artifact to fit from. Exactly one of templates and "
    "model must be given."
)

_MODEL_DESCRIPTION = (
    "A pre-fitted model artifact to load. Exactly one of templates and "
    "model must be given."
)


class GlobalModelParams(Params, Generic[M, T]):
    """Base params for global features that fit on a templates artifact
    or load a pre-fitted model.

    Type parameter M is the model artifact type (must extend JoblibArtifact), and
    T the templates artifact type (must extend TemplatesRef). Exactly one of
    `templates` or `model` must be provided.

    **Both are pinned types rather than the bare aliases, and that is the whole
    defence of the artifact edge.** An ``ArtifactSpec`` with no pattern derives
    ``*.<load kind>``, and a producer's run root holds one per-entry output parquet
    per sequence beside its named artifacts -- so a generic ``ParquetArtifact`` here
    resolved whichever file sorted first, which is a per-entry table, and nothing
    downstream could tell. Naming the type puts the filename in the declaration,
    where it costs nothing at run time and cannot be forgotten by a recipe author.

    Neither field carries a default_factory. ``from_overrides`` merges a partial
    dict onto such a default, which would splice the *base* type's load spec into a
    payload destined for a narrowed one; with a plain ``None`` the payload validates
    straight against T and the pinned class defaults supply pattern and load.
    """

    templates: Annotated[T | None, Declared(_TEMPLATES_DESCRIPTION)] = None
    model: Annotated[M | None, Declared(_MODEL_DESCRIPTION)] = None

    @model_validator(mode="after")
    def _exclusive_source(self) -> Self:
        """Exactly one source, counted by what was *given* rather than named.

        Presence in ``model_fields_set`` alone is not the question, because this
        model's own ``model_dump`` writes both keys -- the one that was not
        provided as an explicit ``null``. Reading that dump back therefore set
        both, and every params file any global model feature ever wrote was
        rejected on reload. ``reconcile`` rebuilds a run's feature from exactly
        that file, so it could not confirm a single such run and reported them
        all unresolvable; the reader is `run_feature`'s own output.

        ``model_fields_set`` still has to be consulted for that reason alone:
        neither field carries a non-``None`` default any more, but a reloaded dump
        names both keys, and only the values tell which one was given.
        """
        has_templates = (
            "templates" in self.model_fields_set and self.templates is not None
        )
        has_model = "model" in self.model_fields_set and self.model is not None
        if has_templates == has_model:
            msg = "Exactly one of 'templates' or 'model' must be provided"
            raise ValueError(msg)
        if not has_templates:
            self.templates = None
        if not has_model:
            self.model = None
        return self
