"""Track schema system for validating standardized track DataFrames."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace

import pandas as pd

_NO_COLUMNS: AbstractSet[str] = frozenset()
"""The empty column set, as a typed immutable default for the optional fields."""

__all__ = [
    "DERIVED_COLUMNS",
    "LEGACY_SCHEMA",
    "TRACK_SCHEMAS",
    "ForbiddenTrackColumnError",
    "TrackSchema",
    "TrackSchemaError",
    "UnknownTrackSchemaError",
    "ensure_track_schema",
    "register_track_schema",
    "schema_family",
]

LEGACY_SCHEMA = "trex_v1"
"""What a tracks row with no recorded schema is.

The column was added after ``trex_v1`` was the only schema there was, so a blank
cell is not an unknown -- it is that one, stated by omission. Every reader that
has to decide what an unrecorded schema means reads it from here, so the two
that ask -- the scope check in ``pipeline.manifest`` and the crop features'
units guard -- cannot drift into answering differently.
"""


@dataclass(frozen=True)
class TrackSchema:
    """What a standardized track table must contain to answer to a name.

    Attributes:
        name: The registry key, and the name a converter declares it emits.
        required: Exact column names that must exist.
        required_prefixes: Prefixes of which at least one column each must exist.
        recommended: Warn-only column names.
        forbidden: Column names that must **not** be present. Checked as a
            named, closed list -- an unlisted column is still accepted, so the
            promise that additive columns are back-compatible survives intact.
        extends: The schema this one builds on, resolved at registration. Its
            required, prefix, recommended and forbidden sets are merged in, so a
            superset declares only its additions rather than restating a base
            that would then drift.
        allows: Columns this schema lifts from the base's ``forbidden`` set. How
            a tracker that genuinely *measures* a quantity the base treats as
            derived says so.
        description: Human-readable summary, surfaced in validation output.

    The set fields are typed ``AbstractSet[str]`` rather than ``frozenset[str]``
    so a caller may pass a plain ``set`` literal, which is how every registration
    below reads. They were previously annotated ``Set[str] = None`` -- a default
    the annotation forbids, which basedpyright strict rejects on sight.

    ``recommended`` matches exact names while ``required_prefixes`` matches
    prefixes, so a prefix-shaped optional family (``poseP0``, ``poseP1``, ...)
    cannot be expressed as recommended: the literal ``"poseP"`` would match no
    column and report missing on every table. Such families are simply left
    undeclared rather than given a check that always fails.
    """

    name: str
    required: AbstractSet[str] = _NO_COLUMNS
    required_prefixes: AbstractSet[str] = _NO_COLUMNS
    recommended: AbstractSet[str] = _NO_COLUMNS
    forbidden: AbstractSet[str] = _NO_COLUMNS
    extends: str | None = None
    allows: AbstractSet[str] = _NO_COLUMNS
    description: str = ""


TRACK_SCHEMAS: dict[str, TrackSchema] = {}


def register_track_schema(schema: TrackSchema) -> None:
    """Register *schema* under its declared name, resolving ``extends`` now.

    Resolution is eager rather than walked at validation time, so a base that is
    not registered fails at import -- where the traceback names the schema doing
    the extending -- instead of on the first table validated against it.

    Raises:
        UnknownTrackSchemaError: If ``extends`` names a schema that is not
            registered yet. Registration order is therefore base-before-derived,
            which is also the order they read in.
    """
    if schema.extends is not None:
        base = TRACK_SCHEMAS.get(schema.extends)
        if base is None:
            known = ", ".join(sorted(TRACK_SCHEMAS)) or "(none registered)"
            raise UnknownTrackSchemaError(
                f"Schema {schema.name!r} extends {schema.extends!r}, which is not "
                f"registered. Known: {known}"
            )
        schema = replace(
            schema,
            required=base.required | schema.required,
            required_prefixes=base.required_prefixes | schema.required_prefixes,
            recommended=base.recommended | schema.recommended,
            forbidden=(base.forbidden | schema.forbidden) - schema.allows,
        )
    TRACK_SCHEMAS[schema.name] = schema


class TrackSchemaError(ValueError):
    """A table failed schema validation under ``strict=True``.

    Named rather than a bare ``ValueError`` so the conversion loops can let it
    through their "warn and keep going" handlers. Those exist so one broken file
    does not end a batch, which is right -- but they also swallowed the refusal a
    caller had explicitly asked for, turning ``strict_schema=True`` into a printed
    line and a silently smaller dataset. A ``ValueError`` subclass, so any caller
    already catching that keeps working.
    """


class UnknownTrackSchemaError(TrackSchemaError):
    """A caller named a schema that is not registered.

    A subclass of :class:`TrackSchemaError` for one reason: the conversion loops
    catch that class ahead of their bare ``except Exception`` and re-raise it, so
    naming a schema that does not exist aborts the batch. Were this a plain
    ``ValueError`` the warn-and-continue handler would swallow it per entry and
    the run would report success over a dataset nothing had validated.

    Distinct from ``TrackSchemaError`` because the two say different things: that
    one means the data is wrong, this one means the *request* is -- a typo, or a
    module whose registration was never imported.
    """


class ForbiddenTrackColumnError(TrackSchemaError):
    """A table carries a column its schema declares must not be present.

    **Raised regardless of ``strict``**, unlike a missing required column, and
    the asymmetry is deliberate. A missing column leaves a table that is merely
    incomplete: a reader asking for it gets a ``KeyError`` naming it. A
    *forbidden* column leaves a table that is confidently wrong -- a ``SPEED``
    in centimetres per second sitting in a pixel-native table reads as a
    perfectly good number, and every distance and threshold computed from it is
    off by a constant nobody recorded. Warning about that and continuing writes
    the bad table to disk, which is the failure this class exists to prevent.

    A converter whose tracker genuinely *measures* one of these declares a
    schema that ``allows`` it, which is what ``trex_v2`` does.
    """


def schema_family(schema_name: str) -> str:
    """The root of *schema_name*'s ``extends`` chain.

    Two schemas share a family when one guarantees everything the other's base
    does, which is the question a reader mixing tables has to answer: ``trex_v2``
    extends ``mosaic_v1``, so a feature reading the columns ``mosaic_v1``
    promises gets the same meaning from either. ``trex_v1`` is its own family --
    its spatial columns are centimetres and its ``X`` is a head position -- so
    mixing one with a ``mosaic_v1`` table is mixing units and landmarks.

    Deliberately total, never raising. An unregistered name (including the empty
    cell an index row written before schemas were recorded carries) is its own
    family: this classifies rows in order to *refuse* a bad mixture, and a
    classifier that throws on the unknown case cannot report the mixture it was
    called to find.
    """
    seen: set[str] = set()
    current = schema_name
    while current not in seen:
        seen.add(current)
        schema = TRACK_SCHEMAS.get(current)
        if schema is None or schema.extends is None:
            return current
        current = schema.extends
    return current


def ensure_track_schema(
    df: pd.DataFrame,
    schema_name: str,
    strict: bool = False,
    source: str = "",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Validate that *df* satisfies the schema registered as *schema_name*.

    Args:
        df: The table to validate. Never modified.
        schema_name: A registered schema name.
        strict: Raise :class:`TrackSchemaError` when a required column or prefix
            is missing, rather than printing a report.
        source: Optional identifier (file path or sequence key) included in log
            and error messages, so the offending file can be located when
            batch-converting.

    Returns:
        ``(df, report)`` where *report* holds ``missing_required``,
        ``missing_prefixes``, ``missing_recommended`` and ``forbidden_present``.

    Raises:
        UnknownTrackSchemaError: If *schema_name* is not registered. Unknown used
            to return an empty report, which validated nothing **and** silently
            disarmed ``strict=True``, because the strict check sits below that
            early return. A dataset naming a schema that does not exist got no
            validation and no warning, and the failure surfaced later as a
            missing column deep in a feature.
        ForbiddenTrackColumnError: If the table carries a column the schema
            forbids. Raised whatever *strict* says -- see the class docstring.
        TrackSchemaError: If *strict* and a required column or prefix is missing.
    """
    schema = TRACK_SCHEMAS.get(schema_name)
    if schema is None:
        known = ", ".join(sorted(TRACK_SCHEMAS)) or "(none registered)"
        src_tag = f" (validating {source})" if source else ""
        raise UnknownTrackSchemaError(
            f"No track schema registered as {schema_name!r}{src_tag}. Known: {known}"
        )

    # Stringified once, and every check reads this rather than ``df.columns``: a
    # non-string label (an integer column name survives a parquet round trip)
    # would silently match nothing under a ``startswith`` and compare unequal to
    # every required name.
    labels: list[object] = list(df.columns)
    column_names = frozenset(str(label) for label in labels)

    missing_required = sorted(c for c in schema.required if c not in column_names)
    missing_prefixes = sorted(
        prefix
        for prefix in schema.required_prefixes
        if not any(name.startswith(prefix) for name in column_names)
    )
    missing_recommended = sorted(c for c in schema.recommended if c not in column_names)
    forbidden_present = sorted(c for c in schema.forbidden if c in column_names)

    report = {
        "missing_required": missing_required,
        "missing_prefixes": missing_prefixes,
        "missing_recommended": missing_recommended,
        "forbidden_present": forbidden_present,
    }
    src_tag = f" {source}" if source else ""
    if forbidden_present:
        raise ForbiddenTrackColumnError(
            f"Schema '{schema_name}'{src_tag} forbids {forbidden_present}. These are "
            f"derived quantities a feature computes, not measurements a tracker "
            f"reports; a tracker that genuinely measures one declares a schema "
            f"that allows it."
        )
    if strict and (missing_required or missing_prefixes):
        raise TrackSchemaError(
            f"Schema '{schema_name}'{src_tag} validation failed: {report}"
        )
    if missing_required or missing_prefixes or missing_recommended:
        print(f"[schema:{schema_name}]{src_tag} Validation report -> {report}")
    return df, report


# Default T-Rex-like schema (flexible): must have these core columns; poseX/poseY
# are prefix-validated.
#
# Note on `group`: it is a *required column* but may be the empty string. `group`
# is an optional, coarse namespace that — together with `sequence` — forms the
# composite identity/filename key (`<group>__<seq>`, or just `<seq>` when group is
# empty). It is NOT the canonical way to categorize/group sequences for analysis:
# flexible, redefinable grouping is done with tags (owned by mosaic-api), and an
# arbitrary tag-resolved subset can be run via `run_feature(entries=[(group, seq), ...])`.
# `group` keeps a structural meaning only as a temporal-contiguity key for the
# future `continuous` dataset type (see core/pipeline/manifest.py).
register_track_schema(
    TrackSchema(
        name="trex_v1",
        required={
            "frame",
            "time",
            "id",
            "group",
            "sequence",
        },
        required_prefixes={"poseX", "poseY"},
        recommended={
            "X#wcentroid",
            "Y#wcentroid",
            "SPEED",
            "ANGLE",
        },
        description="Minimal T-Rex-like per-frame, per-id tracks with centroid/pose columns. "
        "`group` is required but may be empty (an optional namespace, not the "
        "canonical grouping — use tags / run_feature(entries=...) for that).",
    )
)


DERIVED_COLUMNS: AbstractSet[str] = frozenset(
    {
        "VX",
        "VY",
        "AX",
        "AY",
        "SPEED",
        "SPEED#centroid",
        "SPEED#pcentroid",
        "SPEED#wcentroid",
        "ANGLE",
        "ANGULAR_V",
        "ANGULAR_A",
        "ANGULAR_V#centroid",
        "ANGULAR_A#centroid",
        "X#wcentroid",
        "Y#wcentroid",
    }
)
"""Quantities a feature computes rather than a tracker measures.

Forbidden by ``mosaic_v1`` and allowed by ``trex_v2``, from this one list, so the
two cannot drift into disagreeing about which columns the distinction covers.

Every one of these was, at some point, written by a *converter* that derived it:
velocity by ``np.gradient``, speed by ``hypot``, heading by a principal-component
fit whose sign is arbitrary, and a ``#wcentroid`` pair that was a verbatim copy of
``X``/``Y``. Presented in the table they read as tracker output, which is what
made them worth forbidding rather than merely not writing -- the numbers are
plausible, so nothing downstream can tell a measurement from a guess.

A tracker that genuinely reports one of these is not the exception this list
overlooks; it is what ``allows`` is for.
"""


register_track_schema(
    TrackSchema(
        name="mosaic_v1",
        required={
            "frame",
            "time",
            "id",
            "group",
            "sequence",
            "X",
            "Y",
        },
        forbidden=DERIVED_COLUMNS,
        description=(
            "The tracker-neutral standard: per-frame, per-id tracks in video "
            "pixels. `X`/`Y` are the individual's body centre -- for a pose-only "
            "tracker the mean of that frame's keypoints, for one that measures a "
            "centroid its own. Keypoints are optional: a centroid-only tracker "
            "emits none rather than copying `X`/`Y` into a fabricated `poseX0`, "
            "and the features that need keypoints refuse without them. Every "
            "spatial column is pixels; a physical unit is a feature, not a "
            "column. Derived quantities (velocity, speed, heading) are "
            "forbidden: they belong to features, where the method is chosen and "
            "recorded. `group` is required but may be empty."
        ),
    )
)


register_track_schema(
    TrackSchema(
        name="trex_v2",
        extends="mosaic_v1",
        allows=DERIVED_COLUMNS,
        description=(
            "`mosaic_v1` plus what TREx genuinely measures, unscaled to pixels. "
            "TREx reports speed and heading itself, and a weighted centroid "
            "distinct from its blob centroid, so none of the derived set is "
            "forbidden here. `X`/`Y` carry the body centre (TREx's `#wcentroid`), "
            "and TREx's own bare `X`/`Y` -- which are the *head*, and present "
            "only where posture was calculated -- are kept as `X#head`/`Y#head`."
        ),
    )
)
