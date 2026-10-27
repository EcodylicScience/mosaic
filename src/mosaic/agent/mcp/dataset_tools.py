"""The parts of ``manage_dataset`` that need no dataset: paths, fields, refusals.

A path an agent names is resolved, symbolic links and ``..`` included, and must
lie inside one of the roots the operator allowed. The source fields are mapped
onto :class:`~mosaic.core.source_fields.SourceFields`, the builder the CLI's
``sources add`` uses too. The manifest itself is written only through the
``Dataset`` methods that take its lock; nothing here touches ``splits``.

Imports the standard library, the error envelope and the tool contract only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.pipeline.error_envelope import ErrorEnvelope

from .spec import ConfirmationFinding, ManageDatasetInput

if TYPE_CHECKING:
    from mosaic.core.source_fields import SourceFields

__all__ = [
    "OUTSIDE_ROOTS",
    "ROOTS_NOT_CONFIGURED",
    "PathRefused",
    "confined_path",
    "create_findings",
    "missing_field",
    "refuse_escaping_globs",
    "source_fields_of",
    "source_findings",
]

OUTSIDE_ROOTS: Final = "outside_roots"
"""The ``reason`` of a path outside the roots the operator allowed."""

ROOTS_NOT_CONFIGURED: Final = "roots_not_configured"
"""The ``reason`` of an action whose roots the operator did not configure."""

_ROOTS_HELP: Final = {
    "dataset_roots": "create a dataset",
    "source_roots": "declare a source",
}


class PathRefused(Exception):
    """A path that the operator's roots do not allow, as its envelope."""

    def __init__(self, envelope: ErrorEnvelope) -> None:
        super().__init__(envelope.message)
        self.envelope: ErrorEnvelope = envelope


def confined_path(given: str, roots: Sequence[Path], setting: str) -> Path:
    """Return *given* resolved, when it lies inside one of *roots*.

    Args:
        given: The path as the call named it. Relative to the first root when
            it is relative.
        roots: The allowed roots, already resolved.
        setting: The configuration key that lists *roots*, for the message.

    Raises:
        PathRefused: There are no roots, or the resolved path is outside them.
    """
    if not roots:
        raise PathRefused(
            ErrorEnvelope(
                kind="unavailable",
                reason=ROOTS_NOT_CONFIGURED,
                message=(
                    f"this server lets no tool {_ROOTS_HELP[setting]}: its operator "
                    f"lists the directories that allow it under {setting} in the "
                    "server's configuration. Report this to the user."
                ),
                detail={"setting": setting},
            )
        )
    from mosaic.user_paths import user_path

    candidate = user_path(given)
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    resolved = candidate.resolve()
    if any(resolved == root or resolved.is_relative_to(root) for root in roots):
        return resolved
    listed = [str(root) for root in roots]
    raise PathRefused(
        ErrorEnvelope(
            kind="invalid",
            reason=OUTSIDE_ROOTS,
            message=(
                f"{given!r} resolves to {str(resolved)!r}, outside the {setting} "
                f"this server allows: {', '.join(listed)}"
            ),
            detail={"path": given, "resolved": str(resolved), setting: list(listed)},
        )
    )


def refuse_escaping_globs(args: ManageDatasetInput) -> None:
    """Refuse a glob that could match outside the source's path.

    A glob is walked from the path, and ``..`` or an absolute glob would reach
    files beside it, outside the roots the path was confined to.

    Raises:
        PathRefused: A pattern is absolute or has a ``..`` part.
    """
    for pattern in [*(args.patterns or ()), *(args.files or ())]:
        parts = pattern.replace("\\", "/").split("/")
        if pattern.startswith(("/", "\\", "~")) or ".." in parts:
            raise PathRefused(
                ErrorEnvelope(
                    kind="invalid",
                    reason=OUTSIDE_ROOTS,
                    message=(
                        f"{pattern!r} could match files outside the source's path; "
                        "a pattern or file is relative to path and has no '..'"
                    ),
                    detail={"pattern": pattern},
                )
            )


def missing_field(action: str, field: str) -> ErrorEnvelope:
    """Refuse an action called without a field it needs."""
    return ErrorEnvelope(
        kind="invalid",
        reason="field_required",
        message=f"manage_dataset {action} needs {field}",
        detail={"action": action, "field": field},
    )


def source_fields_of(args: ManageDatasetInput, path: Path) -> SourceFields:
    """The source that *args* describe, at the confined *path*."""
    from mosaic.core.source_fields import SourceFields

    assert args.kind is not None
    return SourceFields(
        kind=args.kind,
        path=str(path),
        id=args.source_id,
        files=tuple(args.files or ()),
        recursive=args.recursive,
        extensions=None if args.extensions is None else tuple(args.extensions),
        patterns=None if args.patterns is None else tuple(args.patterns),
        src_format=args.src_format,
        multi_sequences_per_file=args.multi_sequences_per_file,
        group_from=args.group_from,
        group_pattern=args.group_pattern,
        divide_frames=args.divide_frames,
        recording_identity=args.recording_by,
        recording_name=args.recording_name,
    )


def create_findings(name: str, path: Path) -> list[ConfirmationFinding]:
    """What a ``create`` call asks the user to confirm."""
    return [
        ConfirmationFinding(
            "create_dataset", f"creates the dataset {name!r} at {str(path)!r}"
        )
    ]


def source_findings(
    action: str, kind: str, source_id: str, path: str | None
) -> list[ConfirmationFinding]:
    """What an ``add_source`` or ``remove_source`` call asks the user to confirm."""
    if action == "add_source":
        message = f"declares the {kind} source {source_id!r} reading {path!r}"
    else:
        message = (
            f"removes the {kind} source {source_id!r}; the files it indexed stay "
            "indexed"
        )
    return [ConfirmationFinding("add_source", message)]
