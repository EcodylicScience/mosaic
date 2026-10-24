"""A scan source built from the fields a person or an agent gives, checked once.

``mosaic sources add`` and the MCP's ``manage_dataset`` both declare a source
from loose fields: a kind, a path, an id, a file list, globs and the options of
one kind. :func:`scan_source_from_fields` turns them into the manifest's model
with the same refusals for both, so the two cannot drift.

Imports :mod:`mosaic.core.manifest` and the standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from .strict_model import terse
from .manifest import (
    AnyScanSource,
    LabelsScanSource,
    MediaScanSource,
    SourceKind,
    TracksScanSource,
)

__all__ = ["SourceFields", "SourceFieldsInvalid", "scan_source_from_fields"]


class SourceFieldsInvalid(ValueError):
    """Fields that do not describe one scan source."""


@dataclass(frozen=True, slots=True)
class SourceFields:
    """The fields of one scan source, as given.

    ``None`` leaves a field to the model's default. Each attribute is the
    source field of the same name (``MediaScanSource``, ``TracksScanSource``,
    ``LabelsScanSource``), except:

    Attributes:
        kind: Which raw root the source feeds.
        path: Where the files are.
        id: The source's handle. ``None`` takes the directory's name.
        files: Paths relative to *path*. Non-empty makes a file source.
        series: Labels only: the versioned label series whose revisions it
            claims.
    """

    kind: SourceKind
    path: str
    id: str | None = None
    files: tuple[str, ...] = ()
    recursive: bool = True
    extensions: tuple[str, ...] | None = None
    layout: str | None = None
    match_mode: str | None = None
    patterns: tuple[str, ...] | None = None
    src_format: str | None = None
    exclude_patterns: tuple[str, ...] | None = None
    multi_sequences_per_file: bool = False
    group_from: str | None = None
    group_pattern: str | None = None
    md5: bool = True
    series: str | None = None


def scan_source_from_fields(fields: SourceFields) -> AnyScanSource:
    """Return the scan source *fields* describe.

    Raises:
        SourceFieldsInvalid: A field belongs to another kind, a file list is
            given beside globs, a series beside per-sequence fields, or the
            source model refuses a value.
    """
    walk = {
        "extensions": fields.extensions,
        "patterns": fields.patterns,
        "exclude_patterns": fields.exclude_patterns,
    }
    given_walk = [name for name, value in walk.items() if value is not None]
    if fields.files and given_walk:
        msg = (
            f"a file list and {', '.join(given_walk)} cannot both be given: a "
            "listed file is claimed whatever a glob says"
        )
        raise SourceFieldsInvalid(msg)
    if fields.kind == "media":
        _refuse_given(
            fields,
            "media",
            {
                "patterns": fields.patterns,
                "src_format": fields.src_format,
                "exclude_patterns": fields.exclude_patterns,
                "group_from": fields.group_from,
                "group_pattern": fields.group_pattern,
                "series": fields.series,
                "multi_sequences_per_file": fields.multi_sequences_per_file or None,
            },
        )
    else:
        _refuse_given(
            fields,
            fields.kind,
            {
                "extensions": fields.extensions,
                "layout": fields.layout,
                "match_mode": fields.match_mode,
            },
        )
    identifier = fields.id or PurePath(fields.path).name or "source"
    common: dict[str, object] = {"id": identifier, "path": fields.path}
    if fields.files:
        common["files"] = fields.files
    else:
        common["recursive"] = fields.recursive
    try:
        if fields.kind == "media":
            return MediaScanSource.model_validate(
                {
                    **common,
                    **_present(
                        extensions=fields.extensions,
                        layout=fields.layout,
                        match_mode=fields.match_mode,
                    ),
                }
            )
        if fields.series is not None:
            if fields.kind != "labels":
                msg = "series applies to labels: a series is a kind of label"
                raise SourceFieldsInvalid(msg)
            per_sequence = {
                "patterns": fields.patterns,
                "src_format": fields.src_format,
                "exclude_patterns": fields.exclude_patterns,
                "group_from": fields.group_from,
                "group_pattern": fields.group_pattern,
                "multi_sequences_per_file": fields.multi_sequences_per_file or None,
            }
            given = [name for name, value in per_sequence.items() if value is not None]
            if given:
                msg = (
                    f"series and {', '.join(given)} cannot both be given: a series "
                    "fixes what its files are"
                )
                raise SourceFieldsInvalid(msg)
            return LabelsScanSource.model_validate({**common, "series": fields.series})
        raw: dict[str, object] = {
            **common,
            "md5": fields.md5,
            **_present(
                patterns=fields.patterns,
                src_format=fields.src_format,
                exclude_patterns=fields.exclude_patterns,
                group_from=fields.group_from,
                group_pattern=fields.group_pattern,
            ),
        }
        if fields.multi_sequences_per_file:
            raw["multi_sequences_per_file"] = True
        if fields.kind == "tracks":
            return TracksScanSource.model_validate(raw)
        return LabelsScanSource.model_validate(raw)
    except SourceFieldsInvalid:
        raise
    except ValueError as error:
        raise SourceFieldsInvalid(terse(error)) from error


def _present(**values: object) -> dict[str, object]:
    """The *values* that were given."""
    return {name: value for name, value in values.items() if value is not None}


def _refuse_given(fields: SourceFields, kind: str, foreign: dict[str, object]) -> None:
    """Refuse a field of another kind of source."""
    given = [name for name, value in foreign.items() if value is not None]
    if given:
        msg = f"{', '.join(given)} does not apply to a {kind} source"
        raise SourceFieldsInvalid(msg)
