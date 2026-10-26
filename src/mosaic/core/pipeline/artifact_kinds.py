"""The kinds of artifact a dataset holds and the statuses the inventory gives them.

The inventory package defines its models over these names and re-exports them.
They are defined here, apart from it, because importing the inventory package
loads pandas, and a caller that only declares a field over the vocabulary, such
as the agent tool contract, imports the standard library alone.
"""

from __future__ import annotations

from typing import Final, Literal, TypeGuard

__all__ = [
    "ARTIFACT_KINDS",
    "ARTIFACT_STATUSES",
    "ArtifactKind",
    "ArtifactStatus",
    "is_artifact_kind",
    "is_artifact_status",
]

ArtifactKind = Literal[
    "feature",
    "tracks-variant",
    "labels-variant",
    "label-series",
    "tracker-run",
    "frame-run",
    "frame-predictions",
    "prediction-score",
    "frame-selection",
    "trained-model",
    "prepared-dataset",
    "media-derivative",
    "media-variant",
]
"""Every kind of artifact a dataset can hold, named once."""

ARTIFACT_KINDS: Final[tuple[ArtifactKind, ...]] = (
    "feature",
    "tracks-variant",
    "labels-variant",
    "label-series",
    "tracker-run",
    "frame-run",
    "frame-predictions",
    "prediction-score",
    "frame-selection",
    "trained-model",
    "prepared-dataset",
    "media-derivative",
    "media-variant",
)
"""The same vocabulary as a value, for anything that has to enumerate or check it.

Spelled twice rather than derived with ``get_args``, which returns an untyped
tuple and spreads that unknown into every caller. A test asserts the two agree,
so the duplication cannot drift.
"""


def is_artifact_kind(name: str) -> TypeGuard[ArtifactKind]:
    """Narrow a user-supplied string to a declared kind.

    A guard rather than a cast: the vocabulary is checked against the
    declaration once, instead of asserted past it at each call site.
    """
    return name in ARTIFACT_KINDS


ArtifactStatus = Literal[
    "absent",
    "partial",
    "complete",
    "complete-but-drifted",
    "inconsistent",
]
"""What an artifact's coverage and consistency add up to.

See :func:`~mosaic.core.pipeline.inventory.classify`.
"""

ARTIFACT_STATUSES: Final[tuple[ArtifactStatus, ...]] = (
    "absent",
    "partial",
    "complete",
    "complete-but-drifted",
    "inconsistent",
)
"""The statuses as a value, spelled twice for the reason :data:`ARTIFACT_KINDS` is."""


def is_artifact_status(name: str) -> TypeGuard[ArtifactStatus]:
    """Narrow a user-supplied string to a declared status."""
    return name in ARTIFACT_STATUSES
