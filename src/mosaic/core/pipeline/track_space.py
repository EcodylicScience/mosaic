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
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Final, TypeIs

from mosaic.core.pipeline.refusal import Refusal
from mosaic.core.schema import TRACK_SPACES, TrackSpace

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

__all__ = [
    "DEFAULT_TRACK_SPACES",
    "TrackSpaceRefused",
    "feature_track_spaces",
    "refuse_unsupported_track_space",
]

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
        super().__init__(
            f"Feature {feature_name!r} computes in {sorted(accepts)} and this scope's "
            f"tracks are in {unsupported}. A planar feature reading a world-frame "
            "table would compute on two of its three axes and return plausible "
            "numbers; run a feature that declares the space, or narrow the scope.",
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
