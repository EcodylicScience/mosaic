"""What each Ultralytics tracker backend runs under, declared by mosaic.

A run identifier names the settings the run used, so these defaults are declared
here rather than read from the installed Ultralytics. Reading them would let an
upstream retune re-mint every identifier already on disk without a mosaic
change: the digest moves while the settings a user typed do not.

The models below are mosaic's declaration. The setting names are dictated by the
code that consumes them -- each backend reads its settings off the config object
-- and the values are transcribed from what Ultralytics 8.4.84 ships. No YAML file is
vendored, and ``tests/test_ultralytics_preflight.py`` diffs every model against
whatever Ultralytics the external environment holds, in both directions, so drift
is a named failure at upgrade time instead of a wrong number nobody looks at. See
``NOTICE`` at the repository root.

Every model states its own settings. Five of the six backends do share the six
ByteTrack association settings, and sharing them through a base class is refused
twice over. ``deepocsort`` raises two of those thresholds and ``tracktrack``
moves four and declares no ``fuse_score``, so a factored common row would state
a default two tables have to contradict -- and being a default, contradict
without raising. Pydantic also orders inherited fields ahead of a subclass's
own, which moves ``tracker_type`` out of the first position of the resolved
table, and narrowing an inherited ``tracker_type`` to one backend's ``Literal``
is an incompatible override.

``ConfigDict(strict=True)`` validates an override. It refuses an ``int`` for a
flag and a ``bool`` for a whole number or a threshold, and it widens an ``int``
to ``float`` on a threshold. Identity depends on that widening:
``json.dumps(1) != json.dumps(1.0)``, so ``match_thresh: 1`` and
``match_thresh: 1.0`` would otherwise be two identifiers for one recipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, ClassVar, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.core.params import (
    Declared,
    Params,
)

__all__ = [
    "CLOSED_KEYS",
    "GMC_BACKENDS",
    "TRACKER_CONFIG_MODELS",
    "TRACKER_NAMES",
    "TRACKER_TYPE_KEY",
    "BotsortConfig",
    "BytetrackConfig",
    "DeepocsortConfig",
    "FasttrackConfig",
    "GmcMethod",
    "OcsortConfig",
    "ReidCheckpoint",
    "TrackerConfig",
    "TrackerConfigError",
    "TrackerName",
    "TrackerSetting",
    "TracktrackConfig",
    "resolve_tracker_config",
    "validate_tracker_overrides",
]

TrackerName: TypeAlias = Literal[
    "botsort", "bytetrack", "deepocsort", "fasttrack", "ocsort", "tracktrack"
]
"""The selectable backends. Checked against the installed ``TRACKER_MAP`` before a
run starts, so an older Ultralytics is refused by name rather than by assertion.
"""

TRACKER_NAMES: Final[tuple[TrackerName, ...]] = (
    "botsort",
    "bytetrack",
    "deepocsort",
    "fasttrack",
    "ocsort",
    "tracktrack",
)

TrackerSetting: TypeAlias = bool | int | float | str
"""One resolved setting value. No ``None``: every setting below has a concrete
default, and a null would reach the backend as a missing attribute.
"""

GmcMethod: TypeAlias = Literal["sparseOptFlow", "orb", "sift", "ecc", "none"]
"""How camera motion is estimated between frames, where a backend estimates it."""

ReidCheckpoint: TypeAlias = Literal["auto"]
"""The one appearance-feature source mosaic accepts -- see :data:`CLOSED_KEYS`."""

TRACKER_TYPE_KEY: Final = "tracker_type"
"""The setting that selects the backend, and the tag the configuration union is
discriminated on. Overriding it would make the resolved table describe a
different tracker than the one the identifier names.
"""

CLOSED_KEYS: Final[frozenset[str]] = frozenset({"model"})
"""Settings a caller may not set, despite being real settings.

``model`` names a re-identification checkpoint, and the only value mosaic accepts
is ``"auto"`` (the detector's own features). A path cannot go into a hashed
payload -- a model is spelled as a content digest everywhere else in mosaic -- and
Ultralytics downloads a classifier without asking when re-identification is
requested on a head that cannot supply features, which is not something a queued
job should do. Supporting a real re-identification checkpoint means resolving it
through ``resolve_model`` and hashing its digest, which is a later change.
"""

GMC_BACKENDS: Final[frozenset[TrackerName]] = frozenset(
    {"botsort", "deepocsort", "tracktrack"}
)
"""The backends that declare a ``gmc_method``, and can therefore estimate camera
motion.

GMC is the reason the default backend is bytetrack: sparseOptFlow, orb and sift
run an unseeded RANSAC, so a run under one of them is not reproducible bit for
bit. Membership is not the same as running one -- ``deepocsort`` defaults to
``none`` -- so a caller reads a backend's own ``gmc_method`` for what a run
does, and this set for which backends offer the choice.
``tests/test_ultralytics_preflight.py`` checks it against the declared models.
"""


# --- the prose, declared once and reused where the meaning is the same -------

_TRACKER_TYPE_DESCRIPTION = (
    "Which backend this configuration is for. It is the tag the schema "
    "discriminates on, and a configuration naming a different backend than "
    "the run's tracker is refused."
)

_TRACK_HIGH_THRESH_DESCRIPTION = (
    "The confidence a detection must reach to enter the first association "
    "pass. Raising it produces cleaner tracks and keeps fewer detections."
)

_TRACK_LOW_THRESH_DESCRIPTION = (
    "The confidence below which a detection is discarded. Between this and "
    "track_high_thresh a detection is kept for the second association pass, "
    "which trades recovery against drift."
)

_OBSERVATION_CENTRIC_TRACK_LOW_THRESH_DESCRIPTION = (
    "The confidence below which a detection is discarded. Between this and "
    "track_high_thresh a detection is kept for the second association pass, "
    "which runs only under use_byte."
)

_NEW_TRACK_THRESH_DESCRIPTION = (
    "The confidence an unmatched detection must reach to start a new track. "
    "Raising it produces fewer spurious tracks."
)

_TRACK_BUFFER_DESCRIPTION = (
    "How long a lost track stays alive before it is removed. A longer buffer "
    "survives more occlusion and risks more identity switches."
)

_MATCH_THRESH_DESCRIPTION = (
    "The similarity a detection and a track must reach to be associated in the "
    "first pass. Tune it with the detector's quality."
)

_FUSE_SCORE_DESCRIPTION = (
    "Multiply the association similarity by the detection confidence before "
    "matching, which stabilizes weak detections."
)

_GMC_METHOD_DESCRIPTION = (
    "Which estimator cancels camera motion between frames. sparseOptFlow, orb "
    "and sift run an unseeded RANSAC, so a run under one of those is not "
    "reproducible bit for bit. none constructs no estimator."
)

_PROXIMITY_THRESH_DESCRIPTION = (
    "The overlap a detection and a track must share before their appearance "
    "embeddings are compared. Raising it makes re-identification stricter."
)

_APPEARANCE_THRESH_DESCRIPTION = (
    "The appearance similarity a re-identification match must reach. Raising "
    "it avoids identity swaps."
)

_WITH_REID_DESCRIPTION = (
    "Match on appearance embeddings beside motion and overlap, which costs a "
    "further model and further compute."
)

_REID_MODEL_DESCRIPTION = (
    "Which checkpoint supplies the appearance embeddings. Fixed at auto, the "
    "detector's own features: mosaic refuses every other value, because a "
    "checkpoint reaching a run identifier must be named by content digest."
)

_DELTA_T_DESCRIPTION = (
    "How far back the observation-centric velocity direction is measured. A "
    "longer window smooths more."
)

_INERTIA_DESCRIPTION = (
    "The weight the velocity-consistency term gets in the association cost. "
    "Raising it penalizes a direction change more."
)

_USE_BYTE_DESCRIPTION = (
    "Run ByteTrack's low-confidence second association pass as well."
)

_ALPHA_FIXED_EMB_DESCRIPTION = (
    "The base exponential-moving-average factor for a track's appearance "
    "embedding. Raising it updates the embedding more slowly."
)

_RESET_VELOCITY_OFFSET_OCC_DESCRIPTION = (
    "How far back the Kalman filter's velocity is restored from when an "
    "occlusion starts."
)

_RESET_POS_OFFSET_OCC_DESCRIPTION = (
    "How far back the Kalman filter's position is restored from when an "
    "occlusion starts."
)

_ENLARGE_BBOX_OCC_DESCRIPTION = (
    "The one-shot height scale applied to an occluded track's box, which "
    "widens the region searched for it."
)

_DAMPEN_MOTION_OCC_DESCRIPTION = (
    "How much of an occluded track's velocity is kept while it stays occluded, "
    "between 0 and 1."
)

_ACTIVE_OCC_TO_LOST_THRESH_DESCRIPTION = (
    "How long an active track survives continuous occlusion before it is marked lost."
)

_OCC_COVER_THRESH_DESCRIPTION = (
    "The fraction of a track's area another must cover for the track to count "
    "as occluded."
)

_OCC_REAPPEAR_WINDOW_DESCRIPTION = (
    "How long a recently occluded lost track stays re-findable."
)

_INIT_IOU_SUPPRESS_DESCRIPTION = (
    "The overlap with an active track at or above which a new track is not "
    "started. 1.0 starts every new track."
)

_TAI_NEW_TRACK_THRESH_DESCRIPTION = (
    "The score an unmatched detection must reach for track-aware "
    "initialization to start a new track from it."
)

_ITERATIVE_MATCH_THRESH_DESCRIPTION = (
    "The similarity the first iteration of the assignment accepts. Each later "
    "iteration lowers it by reduce_step."
)

_LOST_MATCH_THR_DESCRIPTION = (
    "The looser cost gate for rebinding a track that is still lost. 0.0 skips "
    "the relaxed second pass; a value above match_thresh suits long occlusions."
)

_IOU_WEIGHT_DESCRIPTION = "The weight the overlap distance gets in the cost matrix."

_REID_WEIGHT_DESCRIPTION = (
    "The weight the appearance cosine distance gets in the cost matrix. It "
    "falls back to the overlap distance where re-identification is off."
)

_CONF_WEIGHT_DESCRIPTION = "The weight the confidence distance gets in the cost matrix."

_ANGLE_WEIGHT_DESCRIPTION = (
    "The weight the corner-angle distance gets in the cost matrix."
)

_PENALTY_P_DESCRIPTION = (
    "The cost penalty a low-confidence detection is charged during iterative "
    "assignment."
)

_PENALTY_Q_DESCRIPTION = (
    "The cost penalty a deleted or recovered detection is charged during "
    "iterative assignment."
)

_REDUCE_STEP_DESCRIPTION = (
    "How much the matching threshold drops on each iteration of the assignment."
)

_TAI_THR_DESCRIPTION = (
    "The overlap at which track-aware initialization suppresses a new track."
)

_MIN_TRACK_LEN_DESCRIPTION = "How much history a track needs before it is confirmed."


# --- one model per backend ---------------------------------------------------


class _BackendConfig(Params):
    """The validation rules the six backend configurations share.

    ``strict=True`` is what refuses an override of the wrong shape, and the
    inherited ``extra="forbid"`` is what refuses a setting the backend does not
    have. No field is declared here: pydantic orders inherited fields ahead of a
    subclass's own, and ``tracker_type`` is first in every resolved table.

    :class:`~mosaic.core.params.Params` rather than ``StrictModel``,
    which would be enough for validation, because
    ``tests/test_params_declaration.py`` walks ``Params.__subclasses__()``: a
    setting declared without prose has to fail a test rather than depend on
    whoever adds it. ``BboxPolicy`` is a nested configuration model on the same
    base for the same reason.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)


class BytetrackConfig(_BackendConfig):
    """ByteTrack: association by overlap alone, and no motion estimator.

    The default backend, because it constructs no camera-motion estimator and
    is therefore reproducible bit for bit.
    """

    tracker_type: Annotated[
        Literal["bytetrack"], Declared(_TRACKER_TYPE_DESCRIPTION)
    ] = "bytetrack"
    track_high_thresh: Annotated[float, Declared(_TRACK_HIGH_THRESH_DESCRIPTION)] = 0.25
    track_low_thresh: Annotated[float, Declared(_TRACK_LOW_THRESH_DESCRIPTION)] = 0.1
    new_track_thresh: Annotated[float, Declared(_NEW_TRACK_THRESH_DESCRIPTION)] = 0.25
    track_buffer: Annotated[int, Declared(_TRACK_BUFFER_DESCRIPTION, unit="frames")] = (
        30
    )
    match_thresh: Annotated[float, Declared(_MATCH_THRESH_DESCRIPTION)] = 0.8
    fuse_score: Annotated[bool, Declared(_FUSE_SCORE_DESCRIPTION)] = True


class BotsortConfig(_BackendConfig):
    """BoT-SORT: ByteTrack association plus camera motion and appearance."""

    tracker_type: Annotated[Literal["botsort"], Declared(_TRACKER_TYPE_DESCRIPTION)] = (
        "botsort"
    )
    track_high_thresh: Annotated[float, Declared(_TRACK_HIGH_THRESH_DESCRIPTION)] = 0.25
    track_low_thresh: Annotated[float, Declared(_TRACK_LOW_THRESH_DESCRIPTION)] = 0.1
    new_track_thresh: Annotated[float, Declared(_NEW_TRACK_THRESH_DESCRIPTION)] = 0.25
    track_buffer: Annotated[int, Declared(_TRACK_BUFFER_DESCRIPTION, unit="frames")] = (
        30
    )
    match_thresh: Annotated[float, Declared(_MATCH_THRESH_DESCRIPTION)] = 0.8
    fuse_score: Annotated[bool, Declared(_FUSE_SCORE_DESCRIPTION)] = True
    gmc_method: Annotated[GmcMethod, Declared(_GMC_METHOD_DESCRIPTION)] = (
        "sparseOptFlow"
    )
    proximity_thresh: Annotated[float, Declared(_PROXIMITY_THRESH_DESCRIPTION)] = 0.5
    appearance_thresh: Annotated[float, Declared(_APPEARANCE_THRESH_DESCRIPTION)] = 0.8
    with_reid: Annotated[bool, Declared(_WITH_REID_DESCRIPTION)] = False
    model: Annotated[ReidCheckpoint, Declared(_REID_MODEL_DESCRIPTION)] = "auto"


class OcsortConfig(_BackendConfig):
    """OC-SORT: ByteTrack association plus observation-centric velocity."""

    tracker_type: Annotated[Literal["ocsort"], Declared(_TRACKER_TYPE_DESCRIPTION)] = (
        "ocsort"
    )
    track_high_thresh: Annotated[float, Declared(_TRACK_HIGH_THRESH_DESCRIPTION)] = 0.25
    track_low_thresh: Annotated[
        float, Declared(_OBSERVATION_CENTRIC_TRACK_LOW_THRESH_DESCRIPTION)
    ] = 0.1
    new_track_thresh: Annotated[float, Declared(_NEW_TRACK_THRESH_DESCRIPTION)] = 0.25
    track_buffer: Annotated[int, Declared(_TRACK_BUFFER_DESCRIPTION, unit="frames")] = (
        30
    )
    match_thresh: Annotated[float, Declared(_MATCH_THRESH_DESCRIPTION)] = 0.8
    fuse_score: Annotated[bool, Declared(_FUSE_SCORE_DESCRIPTION)] = True
    delta_t: Annotated[int, Declared(_DELTA_T_DESCRIPTION, unit="frames")] = 3
    inertia: Annotated[float, Declared(_INERTIA_DESCRIPTION)] = 0.2
    use_byte: Annotated[bool, Declared(_USE_BYTE_DESCRIPTION)] = False


class DeepocsortConfig(_BackendConfig):
    """Deep OC-SORT: OC-SORT plus appearance embeddings and motion compensation.

    Three thresholds sit higher here than in every other table -- the two
    detection thresholds and ``appearance_thresh`` -- which is why the shared
    settings are stated per backend rather than factored.
    """

    tracker_type: Annotated[
        Literal["deepocsort"], Declared(_TRACKER_TYPE_DESCRIPTION)
    ] = "deepocsort"
    track_high_thresh: Annotated[float, Declared(_TRACK_HIGH_THRESH_DESCRIPTION)] = 0.3
    track_low_thresh: Annotated[
        float, Declared(_OBSERVATION_CENTRIC_TRACK_LOW_THRESH_DESCRIPTION)
    ] = 0.1
    new_track_thresh: Annotated[float, Declared(_NEW_TRACK_THRESH_DESCRIPTION)] = 0.3
    track_buffer: Annotated[int, Declared(_TRACK_BUFFER_DESCRIPTION, unit="frames")] = (
        30
    )
    match_thresh: Annotated[float, Declared(_MATCH_THRESH_DESCRIPTION)] = 0.8
    fuse_score: Annotated[bool, Declared(_FUSE_SCORE_DESCRIPTION)] = True
    delta_t: Annotated[int, Declared(_DELTA_T_DESCRIPTION, unit="frames")] = 3
    inertia: Annotated[float, Declared(_INERTIA_DESCRIPTION)] = 0.2
    use_byte: Annotated[bool, Declared(_USE_BYTE_DESCRIPTION)] = False
    gmc_method: Annotated[GmcMethod, Declared(_GMC_METHOD_DESCRIPTION)] = "none"
    with_reid: Annotated[bool, Declared(_WITH_REID_DESCRIPTION)] = False
    model: Annotated[ReidCheckpoint, Declared(_REID_MODEL_DESCRIPTION)] = "auto"
    proximity_thresh: Annotated[float, Declared(_PROXIMITY_THRESH_DESCRIPTION)] = 0.5
    appearance_thresh: Annotated[float, Declared(_APPEARANCE_THRESH_DESCRIPTION)] = 0.9
    alpha_fixed_emb: Annotated[float, Declared(_ALPHA_FIXED_EMB_DESCRIPTION)] = 0.95


class FasttrackConfig(_BackendConfig):
    """FastTracker: ByteTrack association plus explicit occlusion handling."""

    tracker_type: Annotated[
        Literal["fasttrack"], Declared(_TRACKER_TYPE_DESCRIPTION)
    ] = "fasttrack"
    track_high_thresh: Annotated[float, Declared(_TRACK_HIGH_THRESH_DESCRIPTION)] = 0.25
    track_low_thresh: Annotated[float, Declared(_TRACK_LOW_THRESH_DESCRIPTION)] = 0.1
    new_track_thresh: Annotated[float, Declared(_NEW_TRACK_THRESH_DESCRIPTION)] = 0.25
    track_buffer: Annotated[int, Declared(_TRACK_BUFFER_DESCRIPTION, unit="frames")] = (
        30
    )
    match_thresh: Annotated[float, Declared(_MATCH_THRESH_DESCRIPTION)] = 0.8
    fuse_score: Annotated[bool, Declared(_FUSE_SCORE_DESCRIPTION)] = True
    reset_velocity_offset_occ: Annotated[
        int, Declared(_RESET_VELOCITY_OFFSET_OCC_DESCRIPTION, unit="frames")
    ] = 5
    reset_pos_offset_occ: Annotated[
        int, Declared(_RESET_POS_OFFSET_OCC_DESCRIPTION, unit="frames")
    ] = 3
    enlarge_bbox_occ: Annotated[float, Declared(_ENLARGE_BBOX_OCC_DESCRIPTION)] = 1.1
    dampen_motion_occ: Annotated[float, Declared(_DAMPEN_MOTION_OCC_DESCRIPTION)] = 0.5
    active_occ_to_lost_thresh: Annotated[
        int, Declared(_ACTIVE_OCC_TO_LOST_THRESH_DESCRIPTION, unit="frames")
    ] = 10
    occ_cover_thresh: Annotated[float, Declared(_OCC_COVER_THRESH_DESCRIPTION)] = 0.7
    occ_reappear_window: Annotated[
        int, Declared(_OCC_REAPPEAR_WINDOW_DESCRIPTION, unit="frames")
    ] = 40
    init_iou_suppress: Annotated[float, Declared(_INIT_IOU_SUPPRESS_DESCRIPTION)] = 0.7


class TracktrackConfig(_BackendConfig):
    """TrackTrack: a weighted multi-cue cost solved by iterative assignment.

    The one backend that is not ByteTrack-derived. Four of its thresholds
    differ from every other table's and it declares no ``fuse_score``, and its
    ``new_track_thresh`` and ``match_thresh`` mean something else here --
    track-aware initialization and the first iteration of the assignment.
    """

    tracker_type: Annotated[
        Literal["tracktrack"], Declared(_TRACKER_TYPE_DESCRIPTION)
    ] = "tracktrack"
    track_high_thresh: Annotated[float, Declared(_TRACK_HIGH_THRESH_DESCRIPTION)] = 0.6
    track_low_thresh: Annotated[float, Declared(_TRACK_LOW_THRESH_DESCRIPTION)] = 0.25
    new_track_thresh: Annotated[float, Declared(_TAI_NEW_TRACK_THRESH_DESCRIPTION)] = (
        0.7
    )
    track_buffer: Annotated[int, Declared(_TRACK_BUFFER_DESCRIPTION, unit="frames")] = (
        30
    )
    match_thresh: Annotated[float, Declared(_ITERATIVE_MATCH_THRESH_DESCRIPTION)] = 0.7
    lost_match_thr: Annotated[float, Declared(_LOST_MATCH_THR_DESCRIPTION)] = 0.0
    iou_weight: Annotated[float, Declared(_IOU_WEIGHT_DESCRIPTION)] = 0.5
    reid_weight: Annotated[float, Declared(_REID_WEIGHT_DESCRIPTION)] = 0.5
    conf_weight: Annotated[float, Declared(_CONF_WEIGHT_DESCRIPTION)] = 0.1
    angle_weight: Annotated[float, Declared(_ANGLE_WEIGHT_DESCRIPTION)] = 0.05
    penalty_p: Annotated[float, Declared(_PENALTY_P_DESCRIPTION)] = 0.2
    penalty_q: Annotated[float, Declared(_PENALTY_Q_DESCRIPTION)] = 0.4
    reduce_step: Annotated[float, Declared(_REDUCE_STEP_DESCRIPTION)] = 0.05
    tai_thr: Annotated[float, Declared(_TAI_THR_DESCRIPTION)] = 0.55
    min_track_len: Annotated[
        int, Declared(_MIN_TRACK_LEN_DESCRIPTION, unit="frames")
    ] = 3
    gmc_method: Annotated[GmcMethod, Declared(_GMC_METHOD_DESCRIPTION)] = (
        "sparseOptFlow"
    )
    with_reid: Annotated[bool, Declared(_WITH_REID_DESCRIPTION)] = False
    model: Annotated[ReidCheckpoint, Declared(_REID_MODEL_DESCRIPTION)] = "auto"


_ConfigModel: TypeAlias = (
    BotsortConfig
    | BytetrackConfig
    | DeepocsortConfig
    | FasttrackConfig
    | OcsortConfig
    | TracktrackConfig
)

TrackerConfig: TypeAlias = Annotated[
    _ConfigModel, Field(discriminator=TRACKER_TYPE_KEY)
]
"""One backend's whole configuration, tagged by ``tracker_type``.

Declared as a model field, this renders as a ``oneOf`` with a discriminator
mapping, so a schema reader draws the settings of the selected backend with
their types, defaults, choices and prose instead of a bare key/value editor.
"""

TRACKER_CONFIG_MODELS: Final[Mapping[TrackerName, type[_ConfigModel]]] = {
    "botsort": BotsortConfig,
    "bytetrack": BytetrackConfig,
    "deepocsort": DeepocsortConfig,
    "fasttrack": FasttrackConfig,
    "ocsort": OcsortConfig,
    "tracktrack": TracktrackConfig,
}


class TrackerConfigError(ValueError):
    """An override names a setting the chosen backend does not have, or cannot take."""


def _backends_offering(key: str) -> tuple[TrackerName, ...]:
    """Which backends do take *key*, so a refusal can point somewhere useful."""
    return tuple(
        name
        for name in TRACKER_NAMES
        if key in TRACKER_CONFIG_MODELS[name].model_fields
    )


def _fixed_value(tracker: TrackerName, key: str) -> object:
    """What *key* is fixed at on *tracker*, or on a backend that declares it.

    The fallback covers a backend that does not declare *key* at all, where the
    refusal still names the value mosaic fixes the setting at everywhere.
    """
    declared = TRACKER_CONFIG_MODELS[tracker].model_fields.get(key)
    if declared is not None:
        return declared.default
    offering = _backends_offering(key)
    return TRACKER_CONFIG_MODELS[offering[0]].model_fields[key].default


def _shape_refusal(tracker: TrackerName, error: ValidationError) -> str:
    """One line per refused override: the setting, what it takes, what it got.

    The rejected value is repeated with its type, because a params file is read
    back by whoever wrote it and the setting's name alone does not say which of
    several values in it was wrong.
    """
    return "; ".join(
        f"{tracker}.{'.'.join(str(part) for part in item['loc'])} "
        f"{item['msg'][0].lower()}{item['msg'][1:]}; got {item['input']!r} "
        f"({type(item['input']).__name__})."
        for item in error.errors()
    )


def validate_tracker_overrides(
    tracker: TrackerName, overrides: Mapping[str, JsonValue] | None = None
) -> TrackerConfig:
    """*tracker*'s configuration model, with *overrides* applied.

    The one place a raw override is validated, so a bad value is refused
    *before* anything is minted or a model is loaded -- rather than on the first
    frame of the first video, hours into a queued run.

    Args:
        tracker: Which backend the overrides are for.
        overrides: Settings to change from the declared defaults. ``None`` and
            an empty mapping both give the declared defaults.

    Returns:
        The backend's configuration model, with every setting resolved.

    Raises:
        TrackerConfigError: An override names the backend selector, a closed
            setting, a setting this backend does not have, or a value of the
            wrong shape.
    """
    model = TRACKER_CONFIG_MODELS[tracker]
    given = dict(overrides or {})
    for key, value in given.items():
        if key == TRACKER_TYPE_KEY:
            refused = (
                f"{TRACKER_TYPE_KEY!r} selects the backend and cannot be "
                f"overridden; pass tracker=... instead of overriding it to "
                f"{value!r}."
            )
            raise TrackerConfigError(refused)
        # The value a closed setting is already fixed at passes, so a caller
        # that reads a params file back submits what it was given. Only a value
        # that would change the setting is refused, and this branch supplies a
        # better message than the field's own one-value Literal.
        if key in CLOSED_KEYS and value != _fixed_value(tracker, key):
            refused = (
                f"{tracker}.{key} is fixed at {_fixed_value(tracker, key)!r} in "
                f"mosaic. A re-identification checkpoint would have to be named "
                f"by content digest rather than by a plain value in a tracker "
                f"config."
            )
            raise TrackerConfigError(refused)
        if key not in model.model_fields:
            elsewhere = _backends_offering(key)
            hint = (
                f" ({key!r} is a setting of {', '.join(elsewhere)}.)"
                if elsewhere
                else ""
            )
            settable = sorted(
                name for name in model.model_fields if name != TRACKER_TYPE_KEY
            )
            refused = (
                f"{tracker} has no setting {key!r}. It takes: "
                f"{', '.join(settable)}.{hint}"
            )
            raise TrackerConfigError(refused)
    try:
        return model.model_validate(given)
    except ValidationError as error:
        raise TrackerConfigError(_shape_refusal(tracker, error)) from error


def resolve_tracker_config(
    tracker: TrackerName,
    overrides: TrackerConfig | Mapping[str, JsonValue] | None = None,
) -> dict[str, TrackerSetting]:
    """Every setting *tracker* runs under, with *overrides* applied.

    Total: the result names ``tracker_type`` and every setting, whether or not
    the caller mentioned it. That is what makes it hashable into a run identifier
    -- a caller who restates a default and one who passes nothing mint the same
    identifier, and a caller who changes one setting mints a different one.

    Pure and idempotent. *overrides* is a configuration model, a raw mapping, or
    ``None``; a raw mapping is validated by
    :func:`validate_tracker_overrides`, and a model naming a different backend
    than *tracker* is refused, because the resolved table would otherwise
    describe a tracker the identifier does not name.
    """
    if isinstance(overrides, _BackendConfig):
        if overrides.tracker_type != tracker:
            refused = (
                f"a {overrides.tracker_type} configuration cannot configure "
                f"{tracker}; pass tracker={overrides.tracker_type!r} or the "
                f"{tracker} configuration."
            )
            raise TrackerConfigError(refused)
        config = overrides
    else:
        config = validate_tracker_overrides(tracker, overrides)
    resolved: dict[str, TrackerSetting] = config.model_dump()
    return resolved
