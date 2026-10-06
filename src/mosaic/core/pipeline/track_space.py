"""Which track spaces a feature computes in, and the refusal when it is handed another.

A planar feature handed a world-frame table does not fail. It reads ``X`` and
``Y``, ignores ``Z``, and returns numbers that look like speeds and headings and
are the projection of the real ones onto the floor -- or onto a wall, depending on
the calibration. The schema family already says what a table's coordinates are
(:data:`~mosaic.core.schema.TrackSpace`); this module is where a feature says what
it can compute in, and where the two are compared before any work starts.

**A feature that declares nothing computes in the two 2D spaces.** That is the
safe direction for a default, unlike ``emits`` or ``scope_dependent``: every
feature written before 3D tables existed is planar, and the default refuses a
world table to all of them without editing one. A feature that handles 3D says so
by declaring ``track_spaces`` as a class attribute.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Final

from mosaic.core.pipeline.refusal import Refusal
from mosaic.core.schema import TrackSpace

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


def feature_track_spaces(feature: object) -> frozenset[str]:
    """The track spaces *feature* declares, or :data:`DEFAULT_TRACK_SPACES`.

    Read off the instance, so a class attribute and an instance override both
    count. Never raises: the declaration is optional by design.
    """
    declared: object = getattr(feature, "track_spaces", None)
    if declared is None:
        return frozenset(DEFAULT_TRACK_SPACES)
    if isinstance(declared, str) or not isinstance(declared, AbstractSet):
        raise TypeError(
            f"{type(feature).__name__}.track_spaces must be a set of track spaces, "
            f"got {declared!r}"
        )
    return frozenset(str(space) for space in declared)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


class TrackSpaceRefused(Refusal, ValueError):
    """A feature was handed tracks in a space it does not compute in.

    A :class:`~mosaic.core.pipeline.refusal.Refusal` with reason
    ``schema_family_mismatch``, the existing member for "these tables mean
    something this step cannot read", so the wire set does not grow. Also a
    ``ValueError``, which is what a caller of ``run_feature`` already catches for
    the mixed-schema refusal beside it.
    """

    def __init__(
        self, feature_name: str, accepts: AbstractSet[str], found: AbstractSet[str]
    ) -> None:
        unsupported = sorted(set(found) - set(accepts))
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
    feature: object, feature_name: str, spaces_read: AbstractSet[str]
) -> None:
    """Raise :class:`TrackSpaceRefused` if *spaces_read* holds a space *feature* lacks.

    Args:
        feature: The feature instance, read for its ``track_spaces``.
        feature_name: Its registered name, for the message.
        spaces_read: The spaces of the track-shaped tables the run resolved,
            ``ResolvedScope.track_spaces``.
    """
    accepts = feature_track_spaces(feature)
    if set(spaces_read) - accepts:
        raise TrackSpaceRefused(feature_name, accepts, spaces_read)
