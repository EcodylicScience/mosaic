"""What each Ultralytics tracker backend is configured with, declared by mosaic.

A run identifier has to name the settings the run actually used, and the only way
to do that honestly is to hold the defaults here rather than read them from the
installed Ultralytics. Reading them would make an upstream retune re-mint every
identifier already on disk without a single mosaic change -- silently, because
the digest would move while the settings a user typed did not.

So the tables below are mosaic's own declaration. The setting *names* are
dictated by the code that consumes them (each backend reads its knobs straight
off the config object), and the values are the ones Ultralytics 8.4.63 shipped.
Nothing is copied: no YAML file is vendored, and
``tests/test_ultralytics_preflight.py`` diffs every table against whatever
Ultralytics is installed, so drift is a named failure at upgrade time instead of
a wrong number nobody looks at. See ``docs/licensing.md``.

There is deliberately **no shared base row**. Six of the seven common settings do
line up across five backends, but ``deepocsort`` raises three of the thresholds
and ``tracktrack`` moves four and drops ``fuse_score`` entirely -- a factored
"common seven" would be wrong in two tables and, being a default, wrong
invisibly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from mosaic.core.json_value import JsonValue

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
"""One resolved setting value. No ``None``: every knob below has a concrete
default, and a null would reach the backend as a missing attribute.
"""


@dataclass(frozen=True, slots=True)
class BoolKnob:
    """A flag. Strict: an ``int`` is not accepted, even though ``bool`` is one."""

    default: bool


@dataclass(frozen=True, slots=True)
class IntKnob:
    """A count of frames or iterations. Strict, and rejects ``bool``."""

    default: int


@dataclass(frozen=True, slots=True)
class FloatKnob:
    """A threshold or weight.

    Widens an ``int`` override to ``float``, because the run identifier hashes the
    resolved table and ``json.dumps(1) != json.dumps(1.0)`` -- left alone,
    ``match_thresh: 1`` and ``match_thresh: 1.0`` would be two identifiers for one
    recipe.
    """

    default: float


@dataclass(frozen=True, slots=True)
class StrKnob:
    """A named choice. *choices*, when given, is closed and checked before minting."""

    default: str
    choices: tuple[str, ...] = ()


Knob: TypeAlias = BoolKnob | IntKnob | FloatKnob | StrKnob

_GMC_METHODS: Final = ("sparseOptFlow", "orb", "sift", "ecc", "none")

# The seven settings every ByteTrack-derived backend reads. Spelled out per table
# rather than shared, for the reason in the module docstring.
_ASSOCIATION_KNOBS: Final[Mapping[str, Knob]] = {
    "track_high_thresh": FloatKnob(0.25),
    "track_low_thresh": FloatKnob(0.1),
    "new_track_thresh": FloatKnob(0.25),
    "track_buffer": IntKnob(30),
    "match_thresh": FloatKnob(0.8),
    "fuse_score": BoolKnob(True),
}

# Appearance re-identification, shared by the three backends that offer it.
# ``model`` is closed to "auto" -- see CLOSED_KEYS.
_REID_KNOBS: Final[Mapping[str, Knob]] = {
    "with_reid": BoolKnob(False),
    "model": StrKnob("auto", choices=("auto",)),
}

TRACKER_KNOBS: Final[Mapping[TrackerName, Mapping[str, Knob]]] = {
    "bytetrack": {**_ASSOCIATION_KNOBS},
    "botsort": {
        **_ASSOCIATION_KNOBS,
        "gmc_method": StrKnob("sparseOptFlow", choices=_GMC_METHODS),
        "proximity_thresh": FloatKnob(0.5),
        "appearance_thresh": FloatKnob(0.8),
        **_REID_KNOBS,
    },
    "ocsort": {
        **_ASSOCIATION_KNOBS,
        "delta_t": IntKnob(3),
        "inertia": FloatKnob(0.2),
        "use_byte": BoolKnob(False),
    },
    "deepocsort": {
        # Three thresholds sit higher here than in every other table.
        **_ASSOCIATION_KNOBS,
        "track_high_thresh": FloatKnob(0.3),
        "new_track_thresh": FloatKnob(0.3),
        "delta_t": IntKnob(3),
        "inertia": FloatKnob(0.2),
        "use_byte": BoolKnob(False),
        "gmc_method": StrKnob("none", choices=_GMC_METHODS),
        **_REID_KNOBS,
        "proximity_thresh": FloatKnob(0.5),
        "appearance_thresh": FloatKnob(0.9),
        "alpha_fixed_emb": FloatKnob(0.95),
    },
    "fasttrack": {
        **_ASSOCIATION_KNOBS,
        "reset_velocity_offset_occ": IntKnob(5),
        "reset_pos_offset_occ": IntKnob(3),
        "enlarge_bbox_occ": FloatKnob(1.1),
        "dampen_motion_occ": FloatKnob(0.5),
        "active_occ_to_lost_thresh": IntKnob(10),
        "occ_cover_thresh": FloatKnob(0.7),
        "occ_reappear_window": IntKnob(40),
        "init_iou_suppress": FloatKnob(0.7),
    },
    "tracktrack": {
        # Four thresholds differ and there is no ``fuse_score`` at all, so this
        # table does not build on the association block.
        "track_high_thresh": FloatKnob(0.6),
        "track_low_thresh": FloatKnob(0.25),
        "new_track_thresh": FloatKnob(0.7),
        "track_buffer": IntKnob(30),
        "match_thresh": FloatKnob(0.7),
        "lost_match_thr": FloatKnob(0.0),
        "iou_weight": FloatKnob(0.5),
        "reid_weight": FloatKnob(0.5),
        "conf_weight": FloatKnob(0.1),
        "angle_weight": FloatKnob(0.05),
        "penalty_p": FloatKnob(0.2),
        "penalty_q": FloatKnob(0.4),
        "reduce_step": FloatKnob(0.05),
        "tai_thr": FloatKnob(0.55),
        "min_track_len": IntKnob(3),
        "gmc_method": StrKnob("sparseOptFlow", choices=_GMC_METHODS),
        **_REID_KNOBS,
    },
}

TRACKER_TYPE_KEY: Final = "tracker_type"
"""The one setting that is not a knob: it selects the backend, so overriding it
would make the resolved table describe a different tracker than the one the
identifier names.
"""

CLOSED_KEYS: Final[frozenset[str]] = frozenset({"model"})
"""Knobs a caller may not set, despite being real settings.

``model`` names a re-identification checkpoint, and the only value mosaic accepts
is ``"auto"`` (the detector's own features). A path cannot go into a hashed
payload -- a model is spelled as a content digest everywhere else in mosaic -- and
Ultralytics silently *downloads* a classifier when re-identification is asked for
on a head that cannot supply features, which is not something a queued job should
do. Supporting a real re-identification checkpoint means resolving it through
``resolve_model`` and hashing its digest, which is a later change.
"""

# GMC is the reason the default backend is bytetrack: the optical-flow estimator
# runs an unseeded RANSAC, so a backend that constructs one is not bit-reproducible.
GMC_BACKENDS: Final[frozenset[TrackerName]] = frozenset(
    {"botsort", "deepocsort", "tracktrack"}
)


class TrackerConfigError(ValueError):
    """An override names a setting the chosen backend does not have, or cannot take."""


def _backends_offering(key: str) -> tuple[TrackerName, ...]:
    """Which backends do take *key*, so a refusal can point somewhere useful."""
    return tuple(n for n in TRACKER_NAMES if key in TRACKER_KNOBS[n])


def _coerce(
    tracker: TrackerName, key: str, knob: Knob, value: JsonValue
) -> TrackerSetting:
    """Validate one override against its knob, widening ``int`` to ``float`` only."""
    match knob:
        case BoolKnob():
            if not isinstance(value, bool):
                raise TrackerConfigError(
                    f"{tracker}.{key} is a flag; got {value!r} ({type(value).__name__})."
                )
            return value
        case IntKnob():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TrackerConfigError(
                    f"{tracker}.{key} is a whole number; got {value!r} "
                    f"({type(value).__name__})."
                )
            return value
        case FloatKnob():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrackerConfigError(
                    f"{tracker}.{key} is a number; got {value!r} "
                    f"({type(value).__name__})."
                )
            return float(value)
        case StrKnob(choices=choices):
            if not isinstance(value, str):
                raise TrackerConfigError(
                    f"{tracker}.{key} is a name; got {value!r} "
                    f"({type(value).__name__})."
                )
            if choices and value not in choices:
                raise TrackerConfigError(
                    f"{tracker}.{key} must be one of {', '.join(choices)}; got {value!r}."
                )
            return value


def resolve_tracker_config(
    tracker: TrackerName, overrides: Mapping[str, JsonValue] | None = None
) -> dict[str, TrackerSetting]:
    """Every setting *tracker* runs under, with *overrides* applied.

    Total: the result names ``tracker_type`` and every knob, whether or not the
    caller mentioned it. That is what makes it hashable into a run identifier --
    a caller who restates a default and one who passes nothing mint the same
    identifier, and a caller who changes one knob mints a different one.

    Pure and idempotent, and the only place an override is validated, so a bad
    value is refused *before* anything is minted or a model is loaded -- rather
    than on the first frame of the first video, hours into a queued run.
    """
    knobs = TRACKER_KNOBS[tracker]
    resolved: dict[str, TrackerSetting] = {TRACKER_TYPE_KEY: tracker}
    for key, knob in knobs.items():
        resolved[key] = knob.default

    for key, value in (overrides or {}).items():
        if key == TRACKER_TYPE_KEY:
            raise TrackerConfigError(
                f"{TRACKER_TYPE_KEY!r} selects the backend and cannot be overridden; "
                f"pass tracker=... instead of overriding it to {value!r}."
            )
        if key in CLOSED_KEYS:
            raise TrackerConfigError(
                f"{tracker}.{key} is fixed at {knobs[key].default!r} in mosaic. "
                "A re-identification checkpoint would have to be named by content "
                "digest rather than by the value a tracker config carries."
            )
        knob = knobs.get(key)
        if knob is None:
            elsewhere = _backends_offering(key)
            hint = (
                f" ({key!r} is a setting of {', '.join(elsewhere)}.)"
                if elsewhere
                else ""
            )
            raise TrackerConfigError(
                f"{tracker} has no setting {key!r}. It takes: "
                f"{', '.join(sorted(knobs))}.{hint}"
            )
        resolved[key] = _coerce(tracker, key, knob, value)

    return resolved
