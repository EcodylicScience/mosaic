"""Track schema system for validating standardized track DataFrames."""

from __future__ import annotations

import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Final, Literal

import pandas as pd

_NO_COLUMNS: AbstractSet[str] = frozenset()
"""The empty column set, as a typed immutable default for the optional fields."""

__all__ = [
    "DERIVED_COLUMNS",
    "GAZE_COLUMNS",
    "HEAD_COLUMNS",
    "LEGACY_SCHEMA",
    "TRACK_SCHEMAS",
    "TRACK_SPACES",
    "ForbiddenTrackColumnError",
    "IncompleteTrackColumnsError",
    "TrackSchema",
    "TrackSchemaError",
    "TrackSpace",
    "UnknownTrackSchemaError",
    "ensure_track_schema",
    "register_track_schema",
    "schema_family",
    "schema_space",
]

LEGACY_SCHEMA = "trex_v1"
"""What a tracks row with no recorded schema is.

The column was added after ``trex_v1`` was the only schema there was, so a blank
cell is not an unknown -- it is that one, stated by omission. Every reader that
has to decide what an unrecorded schema means reads it from here, so the two
that ask -- the scope check in ``pipeline.manifest`` and the crop features'
units guard -- cannot drift into answering differently.
"""


type TrackSpace = Literal["image_px", "plane_cm", "world_m"]
"""What a table's spatial columns are measured in, and against what.

- ``"image_px"`` -- video pixels of the view the producer read. ``mosaic_v1`` and
  everything extending it. The only space a coordinate may be *drawn* in as it
  stands: an overlay or a crop places it on the frame directly.
- ``"plane_cm"`` -- centimetres in the image plane, for archived tables whose
  pixel factor cannot be recovered (``mosaic_cm_v1``, ``trex_v1``). Still two
  axes, so a planar feature computes in it; nothing draws it.
- ``"world_m"`` -- metres in a calibrated world frame, right-handed with +Z up,
  for a multi-view or fused producer (``mosaic_world_v1``). Three axes. It
  reaches a video only through a projection with the recording's calibration.

Declared on a schema family rather than inferred from columns, because the two
2D spaces are indistinguishable by their columns -- a centimetre table and a
pixel table both carry ``X`` and ``Y`` -- and the difference reads as a
plausible number either way.
"""

TRACK_SPACES: Final[tuple[TrackSpace, ...]] = ("image_px", "plane_cm", "world_m")
"""Every :data:`TrackSpace`, in a stable order for messages and iteration."""


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
        space: The :data:`TrackSpace` every spatial column is in. **Required on
            a root schema** and inherited through ``extends``, which may not
            change it: a family is one space, which is what makes refusing a
            mixture of families mean something.
        pose_axes: The axes every keypoint carries, as column-prefix suffixes
            (``("X", "Y", "Z")`` means ``poseX{k}``, ``poseY{k}``, ``poseZ{k}``).
            When set, a keypoint present on some of these axes but not all of
            them raises, whatever ``strict`` says. Empty checks nothing, which is
            what every 2D schema registered before this field declares.
        all_or_none: Column groups that must be present entirely or not at all,
            such as a gaze ray's origin and direction. A partial group raises,
            whatever ``strict`` says. Merged with the base's groups on
            ``extends``.
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
    space: TrackSpace | None = None
    pose_axes: tuple[str, ...] = ()
    all_or_none: tuple[AbstractSet[str], ...] = ()
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
        TrackSchemaError: If a root schema declares no ``space``, or an extension
            declares a ``space`` or ``pose_axes`` different from its base's. A
            root left without a space would be drawn as pixels or refused by
            every feature, and which of the two is not something to leave to a
            default.
    """
    if schema.extends is not None:
        base = TRACK_SCHEMAS.get(schema.extends)
        if base is None:
            known = ", ".join(sorted(TRACK_SCHEMAS)) or "(none registered)"
            raise UnknownTrackSchemaError(
                f"Schema {schema.name!r} extends {schema.extends!r}, which is not "
                f"registered. Known: {known}"
            )
        if schema.space is not None and schema.space != base.space:
            raise TrackSchemaError(
                f"Schema {schema.name!r} declares space {schema.space!r}, but it "
                f"extends {base.name!r}, whose space is {base.space!r}. A family is "
                f"one space: register a root schema for a different one."
            )
        if schema.pose_axes and schema.pose_axes != base.pose_axes:
            raise TrackSchemaError(
                f"Schema {schema.name!r} declares pose axes {schema.pose_axes}, but "
                f"it extends {base.name!r}, whose pose axes are {base.pose_axes}."
            )
        schema = replace(
            schema,
            required=base.required | schema.required,
            required_prefixes=base.required_prefixes | schema.required_prefixes,
            recommended=base.recommended | schema.recommended,
            forbidden=(base.forbidden | schema.forbidden) - schema.allows,
            space=base.space,
            pose_axes=base.pose_axes,
            all_or_none=base.all_or_none + schema.all_or_none,
        )
    elif schema.space is None:
        raise TrackSchemaError(
            f"Schema {schema.name!r} is a root schema and declares no space. Pass "
            f"space= one of {list(TRACK_SPACES)}, or extend the schema whose space "
            f"it shares (a pixel table extends 'mosaic_v1')."
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


class IncompleteTrackColumnsError(TrackSchemaError):
    """A table carries part of a column group its schema says is all or nothing.

    Raised regardless of ``strict``, for the reason a forbidden column is: the
    table is not merely incomplete but misleading. A keypoint with ``poseX``,
    ``poseY`` and no ``poseZ`` in a 3D table reads to every pair-shaped helper as
    a perfectly good 2D keypoint, and a gaze origin without its direction is a
    point nobody can tell from a measured one.
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


def schema_space(schema_name: str) -> TrackSpace | None:
    """The :data:`TrackSpace` of *schema_name*'s family, or ``None`` if unknown.

    A blank name is the legacy schema, the same reading ``schema_family``'s
    callers give an index row written before the column existed. Total, like
    ``schema_family``: an unregistered name answers ``None`` rather than raising,
    and a caller deciding whether to *draw* a table treats ``None`` as "not
    pixels".
    """
    schema = TRACK_SCHEMAS.get(schema_family(schema_name or LEGACY_SCHEMA))
    return None if schema is None else schema.space


_POSE_COLUMN = re.compile(r"^pose([A-Z])(\d+)$")


def _incomplete_keypoints(
    column_names: AbstractSet[str], pose_axes: tuple[str, ...]
) -> list[str]:
    """The keypoints present on some of *pose_axes* but not on all of them."""
    axes_by_index: dict[str, set[str]] = {}
    for name in column_names:
        match = _POSE_COLUMN.match(name)
        if match is not None and match.group(1) in pose_axes:
            axes_by_index.setdefault(match.group(2), set()).add(match.group(1))
    wanted = set(pose_axes)
    incomplete = sorted(
        (int(index) for index, axes in axes_by_index.items() if axes != wanted)
    )
    return [f"keypoint {index}" for index in incomplete]


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
        IncompleteTrackColumnsError: If a keypoint lacks one of the schema's
            ``pose_axes``, or an ``all_or_none`` group is partly present. Raised
            whatever *strict* says.
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
        derived = [c for c in forbidden_present if c in DERIVED_COLUMNS]
        other = [c for c in forbidden_present if c not in DERIVED_COLUMNS]
        reasons: list[str] = []
        if derived:
            reasons.append(
                f"{derived} are derived quantities a feature computes, not "
                f"measurements a tracker reports; a tracker that genuinely measures "
                f"one declares a schema that allows it."
            )
        if other:
            reasons.append(
                f"{other} cannot appear in this schema at all; its description "
                f"says why."
            )
        raise ForbiddenTrackColumnError(
            f"Schema '{schema_name}'{src_tag} forbids {forbidden_present}. "
            + " ".join(reasons)
        )
    if schema.pose_axes:
        incomplete = _incomplete_keypoints(column_names, schema.pose_axes)
        if incomplete:
            raise IncompleteTrackColumnsError(
                f"Schema '{schema_name}'{src_tag} keypoints carry every one of "
                f"{list(schema.pose_axes)} or none, and {incomplete[:5]}"
                f"{' (+more)' if len(incomplete) > 5 else ''} carry only some."
            )
    for group in schema.all_or_none:
        present = sorted(c for c in group if c in column_names)
        if present and len(present) != len(group):
            missing = sorted(c for c in group if c not in column_names)
            raise IncompleteTrackColumnsError(
                f"Schema '{schema_name}'{src_tag} carries {present} without "
                f"{missing}; those columns are present together or not at all."
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
# arbitrary tag-resolved subset can be run via
# `run_feature(scope=Scope(entries=[(group, seq), ...]))`.
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
        space="plane_cm",
        description="Minimal T-Rex-like per-frame, per-id tracks with centroid/pose columns. "
        "`group` is required but may be empty (an optional namespace, not the "
        "canonical grouping -- use tags / run_feature(scope=Scope(entries=...)) "
        "for that).",
    )
)


DERIVED_COLUMNS: AbstractSet[str] = frozenset(
    {
        "VX",
        "VY",
        "VZ",
        "AX",
        "AY",
        "AZ",
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


STANDARD_COLUMNS: AbstractSet[str] = frozenset(
    {
        "frame",
        "time",
        "id",
        "group",
        "sequence",
        "X",
        "Y",
    }
)
"""What every standard table must carry, whatever its unit.

Named once because two schema *families* require it -- ``mosaic_v1`` in pixels
and ``mosaic_cm_v1`` in centimetres. The second cannot inherit it, since
extending would put it in the pixel family and make the two mixable, which is the
one thing they must not be. So the set is shared here instead, and a column added
to the contract cannot reach one family and miss the other.
"""


register_track_schema(
    TrackSchema(
        name="mosaic_v1",
        required=STANDARD_COLUMNS,
        forbidden=DERIVED_COLUMNS,
        space="image_px",
        description=(
            "The tracker-neutral standard: per-frame, per-id tracks in video "
            "pixels. `X`/`Y` are the individual's body center -- for a pose-only "
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
            "`mosaic_v1` plus what TRex genuinely measures, unscaled to pixels. "
            "TRex reports speed and heading itself, and a weighted centroid "
            "distinct from its blob centroid, so none of the derived set is "
            "forbidden here. `X`/`Y` carry the body center (TRex's `#wcentroid`), "
            "and TRex's own bare `X`/`Y` -- which are the *head*, and present "
            "only where posture was calculated -- are kept as `X#head`/`Y#head`."
        ),
    )
)


register_track_schema(
    TrackSchema(
        name="mosaic_cm_v1",
        required=STANDARD_COLUMNS,
        allows=DERIVED_COLUMNS,
        space="plane_cm",
        description=(
            "The same contract as `trex_v2`, in centimeters rather than pixels, "
            "with `X`/`Y` still the body center. Use it for archived data whose "
            "`cm_per_pixel` factor cannot be recovered -- notably TRex exports "
            "predating TRex 2.0.0, which scaled their output without recording "
            "by how much. Its own schema family: a feature run whose scope "
            "resolves both this and a pixel schema is refused rather than mixed. "
            "For pixel data, `scale-to-cm` converts to centimeters as a recorded "
            "step instead."
        ),
    )
)


GAZE_COLUMNS: AbstractSet[str] = frozenset(
    {"gazeX", "gazeY", "gazeZ", "gazeDX", "gazeDY", "gazeDZ"}
)
"""A gaze ray in the table's frame: origin ``gazeX/Y/Z`` and unit direction ``gazeD*``.

A measurement of the tracked individual on the table's own frame axis -- an eye
tracker's ray, resampled by nothing -- which is why it is a tracks column and
not a separate stream. Present all together or not at all.
"""

HEAD_COLUMNS: AbstractSet[str] = frozenset(
    {"headX", "headY", "headZ", "headQW", "headQX", "headQY", "headQZ"}
)
"""A head pose in the table's frame: position ``headX/Y/Z`` and the head-to-world
rotation as a unit quaternion ``headQW/QX/QY/QZ`` (scalar first).

The head frame is +X forward (straight ahead of the face), +Y to the subject's
left, +Z up through the crown -- the body-frame convention, so a gaze direction
rotated into it reads as azimuth and elevation directly. A producer whose device
reports a camera pose (+X right, +Y down, +Z forward) converts it. Present all
together or not at all.
"""


register_track_schema(
    TrackSchema(
        name="mosaic_world_v1",
        required=STANDARD_COLUMNS | {"Z"},
        forbidden=DERIVED_COLUMNS | {"camera"},
        space="world_m",
        pose_axes=("X", "Y", "Z"),
        all_or_none=(GAZE_COLUMNS, HEAD_COLUMNS),
        description=(
            "The tracker-neutral 3D standard: per-frame, per-id tracks in metres "
            "in the recording's calibrated world frame, right-handed with +Z up. "
            "`X`/`Y`/`Z` are the individual's body center, the mean of its body "
            "keypoints for a pose producer. Keypoints are optional and, when "
            "present, carry all three axes (`poseX{k}`, `poseY{k}`, `poseZ{k}`, "
            "optionally `poseP{k}`). A gaze ray (`gaze*`) and a head pose "
            "(`head*`) may ride along, each complete or absent. A fused table "
            "spans every camera of the recording, so it carries no `camera` "
            "column; it reaches a video only through a projection with the "
            "recording's calibration. Derived quantities are forbidden, as in "
            "`mosaic_v1`. A producer converts its source's unit and axis "
            "convention into this frame and records the transform it applied."
        ),
    )
)
