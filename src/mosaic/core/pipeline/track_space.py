"""Which track spaces a feature computes in, and the refusal when it is handed another.

A planar feature handed a world-frame table does not fail. It reads ``X`` and
``Y``, ignores ``Z``, and returns numbers that look like speeds and headings and
are the projection of the real ones onto one plane of the calibration. The schema
family already says what a table's coordinates are
(:data:`~mosaic.core.schema.TrackSpace`). This module is where a feature says what
it computes in, and where the two are compared before any work starts.

A feature that declares nothing computes in the two 2D spaces. That is the safe
direction for a default: every feature written before 3D tables existed is
planar, and the default refuses a world table to all of them without editing one.
A feature that handles 3D, or that reads no spatial column, declares
``track_spaces`` as a class attribute.

A track-shaped ``Result`` records no schema, so a feature reading one is shown
the spaces of the tracks behind it (:func:`upstream_track_spaces`), at plan and at
run. The walk reads output footers and ``params.json``, never the producer's
class, so it needs no feature registry.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Final, TypeIs

from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.manifest import tracks_table_schemas
from mosaic.core.pipeline.refusal import Refusal
from mosaic.core.pipeline.track_universe import run_is_track_shaped
from mosaic.core.pipeline.tracks_index import AmbiguousTracksVariant
from mosaic.core.schema import TRACK_SPACES, TrackSpace, schema_spaces
from mosaic.core.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.json_value import JsonValue

__all__ = [
    "DEFAULT_TRACK_SPACES",
    "TablesRead",
    "TrackSpaceRefused",
    "feature_track_spaces",
    "refuse_unsupported_track_space",
    "upstream_track_spaces",
    "variant_track_spaces",
]

type TablesRead = dict[tuple[str | None, frozenset[Entry]], Mapping[Entry, str]]
"""The schema of each table already read for a tracks variant over a set of entries."""

DEFAULT_TRACK_SPACES: Final[frozenset[TrackSpace]] = frozenset({"image_px", "plane_cm"})
"""What a feature that declares no ``track_spaces`` computes in: both 2D spaces."""


def _is_set(value: object) -> TypeIs[AbstractSet[object]]:
    return isinstance(value, AbstractSet)


def feature_track_spaces(feature: object) -> frozenset[TrackSpace]:
    """The track spaces *feature* declares, or :data:`DEFAULT_TRACK_SPACES`.

    Read off the instance, so a class attribute and an instance override both
    count. A declaration catalog reads it off the class.

    Args:
        feature: The feature instance, or its class.

    Returns:
        The declared spaces, or the default when it declares none.

    Raises:
        TypeError: If the declaration is a string, is not a set, or names a value
            outside :data:`~mosaic.core.schema.TRACK_SPACES`.
    """
    declared: object = getattr(feature, "track_spaces", None)
    if declared is None:
        return DEFAULT_TRACK_SPACES
    name = (feature if isinstance(feature, type) else type(feature)).__name__
    if isinstance(declared, str) or not _is_set(declared):
        raise TypeError(
            f"{name}.track_spaces must be a set of track spaces, got {declared!r}"
        )
    unknown = sorted(repr(member) for member in declared if member not in TRACK_SPACES)
    if unknown:
        raise TypeError(
            f"{name}.track_spaces names {', '.join(unknown)}, which are not track "
            f"spaces; the track spaces are {list(TRACK_SPACES)}"
        )
    return frozenset(space for space in TRACK_SPACES if space in declared)


class TrackSpaceRefused(Refusal, ValueError):
    """A feature was handed tracks in a space it does not compute in.

    A :class:`~mosaic.core.pipeline.refusal.Refusal` with reason
    ``schema_family_mismatch``, the existing member for "these tables mean
    something this step cannot read", so the wire set does not grow. Also a
    ``ValueError``, which is what a caller of ``run_feature`` already catches for
    the mixed-schema refusal beside it.

    The detail holds ``feature``, the sorted ``accepts`` and the sorted ``found``.
    """

    def __init__(
        self,
        feature_name: str,
        accepts: AbstractSet[TrackSpace],
        found: AbstractSet[TrackSpace],
    ) -> None:
        unsupported = sorted(found - accepts)
        accepted: list[JsonValue] = [*sorted(accepts)]
        resolved: list[JsonValue] = [*sorted(found)]
        if "world_m" in unsupported:
            explanation = (
                "A planar feature reading a world-frame table would compute on two "
                "of its three axes and return plausible numbers"
            )
        else:
            explanation = (
                "A feature that computes in 3D needs a world-frame table "
                "(mosaic_world_v1), and a planar table has no third axis"
            )
        super().__init__(
            f"Feature {feature_name!r} computes in {sorted(accepts)} and this scope's "
            f"tracks are in {unsupported}. {explanation}; run a feature that "
            "declares the space, or narrow the scope.",
            reason="schema_family_mismatch",
            detail={"feature": feature_name, "accepts": accepted, "found": resolved},
        )


def refuse_unsupported_track_space(
    accepts: AbstractSet[TrackSpace],
    feature_name: str,
    spaces_read: AbstractSet[TrackSpace],
) -> None:
    """Raise :class:`TrackSpaceRefused` if *spaces_read* holds a space not in *accepts*.

    A run passes ``feature_track_spaces(feature)`` and a plan passes the spaces of
    the step's declaration, so both refuse through one function and one message.

    Args:
        accepts: The spaces the feature computes in.
        feature_name: Its registered name, for the message.
        spaces_read: The spaces of the track-shaped tables the scope resolved,
            ``ResolvedScope.track_spaces``.

    Raises:
        TrackSpaceRefused: If a table is in a space the feature does not accept.
    """
    if spaces_read - accepts:
        raise TrackSpaceRefused(feature_name, accepts, spaces_read)


def _variant_track_schemas(
    ds: Dataset, variant: str | None, entries: frozenset[Entry], read: TablesRead
) -> Mapping[Entry, str]:
    """The schema of each tracks table of *variant* over *entries*, by entry.

    An entry without a table of *variant* is left out. *read* holds the answers
    already read, so callers sharing it read the tracks index once per variant
    and set of entries. An entry holding two variants under ``None`` makes the
    answer empty, because the reader's identity reports it.

    Args:
        ds: The dataset whose tracks index to read.
        variant: The tracks variant, ``""`` for the unlabeled tables, or
            ``None`` for whichever each entry has.
        entries: The entries to read.
        read: The answers already read, filled in place.
    """
    key = (variant, entries)
    if key not in read:
        try:
            read[key] = tracks_table_schemas(
                ds, variant, Scope(entries=sorted(entries))
            )
        except AmbiguousTracksVariant:
            read[key] = {}
    return read[key]


def variant_track_spaces(
    ds: Dataset, variant: str | None, entries: frozenset[Entry], read: TablesRead
) -> frozenset[TrackSpace]:
    """The spaces of the tracks tables of *variant* over *entries*.

    The schemas come from the tables, or from *read* where already answered.
    """
    return schema_spaces(_variant_track_schemas(ds, variant, entries, read).values())


def upstream_track_spaces(
    ds: Dataset,
    storage: str,
    run_id: str,
    entries: frozenset[Entry],
    read: TablesRead,
) -> frozenset[TrackSpace]:
    """The spaces of the tracks behind a stored track-shaped run, over *entries*.

    A run whose output is track-shaped (:func:`run_is_track_shaped`) is in the
    space of the tracks it was made from, and each of its feature inputs is
    followed by the same rule. A run that is not track-shaped, such as a run of
    embeddings, and a run without a readable ``params.json`` contribute nothing.

    A run that read ``tracks`` is read as the resolver chose its tables: each
    labeled variant on its tracks edge
    (:meth:`RunParams.consumed_tracks_variants`) over *entries*, then the
    unlabeled tables over the entries no labeled variant holds, since an
    unlabeled row loses to a labeled one. ``run_feature`` records labeled
    variants only, and records them before its scope narrows them, so the
    entries the edge does not cover are read from the unlabeled tables.

    Args:
        ds: The dataset holding the run.
        storage: The run's feature storage directory.
        run_id: The run.
        entries: The entries whose tables to read.
        read: The answers already read, as :func:`variant_track_spaces` takes.

    Returns:
        The spaces found, empty when nothing is known.
    """
    spaces: set[TrackSpace] = set()
    seen: set[tuple[str, str]] = set()
    pending: list[tuple[str, str]] = [(storage, run_id)]
    while pending:
        run = pending.pop()
        if run in seen or not run[1]:
            continue
        seen.add(run)
        if not run_is_track_shaped(ds, *run):
            continue
        params = read_run_params(feature_run_root(ds, *run)).params
        if params is None:
            continue
        sources = params.input_sources()
        if "tracks" in sources:
            spaces |= _tracks_read_spaces(
                ds, params.consumed_tracks_variants(), entries, read
            )
        pending.extend(source for source in sources if isinstance(source, tuple))
    return frozenset(spaces)


def _tracks_read_spaces(
    ds: Dataset,
    variants: frozenset[str],
    entries: frozenset[Entry],
    read: TablesRead,
) -> frozenset[TrackSpace]:
    """The spaces of the tables a run over *variants* read for *entries*."""
    schemas: list[str] = []
    covered: set[Entry] = set()
    for variant in sorted(variant for variant in variants if variant):
        tables = _variant_track_schemas(ds, variant, entries, read)
        schemas.extend(tables.values())
        covered.update(tables)
    remainder = entries - covered
    if remainder:
        schemas.extend(_variant_track_schemas(ds, "", remainder, read).values())
    return schema_spaces(schemas)
