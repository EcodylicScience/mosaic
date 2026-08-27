"""The label-converter contract: turn one raw label file into per-sequence labels.

The label sibling of :mod:`mosaic.core.track_converter`, and it exists for the
same two reasons. A label converter used to be a duck-typed class with no
``version``, no typed ``Params`` and no registered op name, whose ``convert``
both wrote the ``.npz`` files and returned the index rows -- so a re-conversion
overwrote in place and nothing about *which recipe* produced a kind reached any
identity. Item 9.3 gives labels the tracks treatment:

- A typed ``Params`` per converter, so validation and throughput knobs sit apart
  from the parameters that determine the labels, and a label variant identity is
  a function of the parameters alone.
- A declared ``version``, a visible segment of that identity.
- ``convert`` returns *data* (:class:`LabelEntry`); the Dataset owns writing the
  ``.npz`` into the variant directory and the typed index row -- the same split
  :class:`~mosaic.core.track_converter.TrackConverter` made, so a converter
  cannot write a file the index never learns about.

``group_from`` is entry policy -- which group string to assign -- not a property
of the labels, so it is excluded from identity exactly as it is kept off the
track-converter interface: two conversions differing only in ``group_from`` are
one recipe producing differently-assigned output, not two recipes.

This module imports nothing from :mod:`mosaic.core.dataset`: the registry moving
out is what closes the converter/dataset import cycle, as it did for tracks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Annotated, ClassVar, Generic

from pydantic import Field
from typing_extensions import TypeVar

from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)

__all__ = [
    "LABEL_CONVERTERS",
    "LabelConvertParams",
    "LabelConverter",
    "LabelEntry",
    "ensure_label_converters_registered",
    "get_label_converter",
    "register_label_converter",
    "registered_label_formats",
    "validate_label_format",
]

_STRICT_SCHEMA_DESCRIPTION = (
    "Whether a converted label set is rejected for failing its schema rather "
    "than reported on."
)

_STRICT_SCHEMA_UNWIRED = (
    "no label path validates a schema, so nothing consults it -- the "
    "same-named field on TrackConvertParams is the one that is read"
)

_GROUP_FROM_DESCRIPTION = (
    "Which group name a converter assigns to the output. infile assigns the "
    "group recorded inside the source file, filename assigns the raw "
    "file-level hint from the labels index, and both behaves the same as "
    "filename in every converter that reads it."
)


@dataclass(frozen=True, slots=True)
class LabelEntry:
    """One converted sequence's labels: the ``.npz`` payload and its index facts.

    A converter returns these; the Dataset writes each ``payload`` into the
    kind's variant directory and records the typed index row. Splitting *what the
    labels are* from *where they are written* is what lets a re-conversion mint a
    new variant beside the old rather than overwrite it, and what keeps a
    converter from writing a file the index never learns about.

    ``payload`` is the ``.npz`` contents (the arrays and scalars a reader loads).
    ``label_ids`` / ``label_names`` are the kind's vocabulary, recorded on the
    index row so a listing need not open every file.
    """

    group: str
    sequence: str
    payload: Mapping[str, object]
    n_frames: int
    label_ids: tuple[int, ...] = ()
    label_names: tuple[str, ...] = ()


class LabelConvertParams(Params):
    """Base for every label converter's parameters.

    ``strict_schema`` is validation strictness and ``group_from`` is entry policy;
    neither is a property of the labels, so both are excluded from identity --
    flipping either must not mint a second variant of the same labels.
    """

    strict_schema: Annotated[
        bool,
        HASH_EXCLUDE,
        Declared(_STRICT_SCHEMA_DESCRIPTION, unwired=_STRICT_SCHEMA_UNWIRED),
    ] = False
    group_from: Annotated[
        str,
        Field(examples=["infile", "filename", "both"]),
        HASH_EXCLUDE,
        Declared(_GROUP_FROM_DESCRIPTION),
    ] = "filename"


P = TypeVar("P", bound=LabelConvertParams, default=LabelConvertParams)


class LabelConverter(Generic[P]):
    """Convert one raw label file into per-sequence :class:`LabelEntry` records.

    Subclass, declare ``src_format`` / ``label_kind`` / ``label_format`` /
    ``Params``, implement ``convert``. The ``(src_format, label_kind)`` pair is
    the registry key: one raw format can feed two kinds.

    Class variables:
        src_format: The raw format this reads, matched against the
            ``labels_raw/index.csv`` ``src_format`` column.
        label_kind: The kind this produces (``"behavior"``, ``"id_tags"``, ...),
            i.e. the ``labels/<kind>/`` subdirectory.
        label_format: The on-disk payload format (e.g. ``individual_pair_v1``),
            recorded for a reader to dispatch on.
        version: Declared compatibility version, a visible segment of the label
            variant identity. Declared, never detected: bump it by hand when the
            output semantics change, so a bit-identical conversion keeps its
            identity across an unrelated release.
        Params: The parameter model. Everything in it determines the labels
            except fields marked ``HASH_EXCLUDE``.
    """

    src_format: ClassVar[str]
    label_kind: ClassVar[str]
    label_format: ClassVar[str] = ""
    version: ClassVar[str] = "0.1"
    Params: ClassVar[type[LabelConvertParams]] = LabelConvertParams

    def convert(
        self, src_path: Path, params: P, raw_row: Mapping[str, object]
    ) -> list[LabelEntry]:
        """Read *src_path* and return one :class:`LabelEntry` per sequence.

        Args:
            src_path: The raw label file.
            params: Typed conversion parameters. These, plus ``src_format``,
                ``label_kind`` and ``version``, are what a label variant identity
                is made of.
            raw_row: The ``labels_raw/index.csv`` row, carrying the group hint and
                the source ``md5``. Never hashed -- the label analog of a track
                converter's :class:`~mosaic.core.track_converter.EntryHints`.
        """
        raise NotImplementedError

    def get_metadata(self) -> dict[str, object]:
        """Format-specific metadata for ``dataset.meta['labels'][kind]`` (optional)."""
        return {}


LABEL_CONVERTERS: dict[tuple[str, str], type[LabelConverter[LabelConvertParams]]] = {}


def register_label_converter(
    cls: type[LabelConverter[LabelConvertParams]],
) -> type[LabelConverter[LabelConvertParams]]:
    """Register a converter under its ``(src_format, label_kind)`` pair.

    Keyed on the pair, not ``src_format`` alone, because one raw format can be a
    source for two kinds -- and, unlike a track converter, a label converter
    additionally names which kind it produces.
    """
    src_format = getattr(cls, "src_format", "")
    label_kind = getattr(cls, "label_kind", "")
    if not src_format or not label_kind:
        raise ValueError(
            f"{cls.__name__} must declare a non-empty src_format and label_kind"
        )
    LABEL_CONVERTERS[(src_format, label_kind)] = cls
    return cls


def ensure_label_converters_registered() -> None:
    """Fill :data:`LABEL_CONVERTERS` from the in-tree library when nothing has.

    A converter registers only as a side effect of importing the module that
    defines it, and the track registry gets that for free -- ``mosaic.core``
    imports ``track_library``. Nothing imports the label library, so a caller who
    reached a Dataset through ``mosaic.core`` alone held an empty registry and
    was told the format they named does not exist, when what was missing was an
    import they had no reason to know about.

    Guarded on emptiness, so a caller who registered converters of their own
    keeps exactly those and pays none of ``mosaic.behavior``'s import cost --
    which is real, because importing any part of it imports all of it.

    The import is deferred into the call rather than written at module scope:
    ``core`` is the layer ``behavior`` is built on, and the two would otherwise
    be a cycle. This is the same shape ``mosaic.core``'s own ``__getattr__``
    uses to reach the feature registry.
    """
    if LABEL_CONVERTERS:
        return
    _ = import_module("mosaic.behavior.label_library")


def registered_label_formats() -> frozenset[str]:
    """Every ``src_format`` some registered label converter claims.

    The format half of the registry's ``(src_format, label_kind)`` key, which is
    all a caller naming a *source* can be asked about -- one format may serve
    several kinds, and which kind is wanted is a question conversion asks, not
    indexing.
    """
    ensure_label_converters_registered()
    return frozenset(src_format for src_format, _kind in LABEL_CONVERTERS)


def validate_label_format(src_format: str) -> str:
    """Reject a ``src_format`` no registered label converter claims.

    Returns *src_format* unchanged, so it can wrap an assignment.

    Raises:
        ValueError: naming the formats that do exist. Checking only the format
            half is deliberate rather than a limitation: the pair is what a
            conversion resolves, and refusing at indexing on a kind nobody has
            named yet would reject a source that is going to convert fine.
    """
    known = registered_label_formats()
    if src_format not in known:
        listed = ", ".join(sorted(known)) or "(none registered)"
        raise ValueError(
            f"No label converter registered for src_format={src_format!r}. "
            f"Known: {listed}. Write one in label_library/ and register it "
            f"before indexing, or import the module that registers yours."
        )
    return src_format


def get_label_converter(
    src_format: str, label_kind: str
) -> LabelConverter[LabelConvertParams]:
    """Instantiate the converter registered for *(src_format, label_kind)*.

    Raises:
        ValueError: with the registered pairs listed, because "no converter for
            X" is almost always a typo or a missing import of the module that
            would have registered it. The same exception the track resolver
            raises, so that one kind of mistake has one kind of answer wherever
            a format is named.
    """
    ensure_label_converters_registered()
    cls = LABEL_CONVERTERS.get((src_format, label_kind))
    if cls is None:
        known = (
            ", ".join(f"{s}/{k}" for s, k in sorted(LABEL_CONVERTERS))
            or "(none registered)"
        )
        raise ValueError(
            f"No label converter registered for src_format={src_format!r}, "
            f"label_kind={label_kind!r}. Known: {known}. Write one in "
            f"label_library/ and register it, or import the module that "
            f"registers yours."
        )
    return cls()
