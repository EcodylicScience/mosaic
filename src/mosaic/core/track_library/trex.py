"""TRex per-id .npz track converter.

Converts TRex-exported per-individual NPZ files to the standardized ``trex_v2``
parquet schema: pixels throughout, with ``X``/``Y`` naming the body centre.

**TRex reports centimetres, and mosaic tables are pixels.** TRex scales its
positional output by ``cm_per_pixel`` internally, so ``X``, ``SPEED`` and the
rest arrive in cm while the keypoints beside them are pixels -- one table, two
coordinate systems, and nothing recording which column is which. The conversion
divides the cm-bearing columns back out.

**The factor is read from the file, never from mosaic's own settings.** TRex
writes ``cm_per_pixel`` into every exported ``.npz`` (``Export.cpp``), and what
it writes is the value it *applied*. mosaic's op parameter is only what mosaic
*passed*: TRex substitutes ``meta_real_width / video_width`` when the setting is
zero or unset, and extra settings can override it without reaching mosaic at all.
Unscaling by the parameter would be a silent constant-factor error on any video
carrying a real-world width. Reading it from the file also keeps the conversion
deterministic in the only sense that matters here -- same bytes, same table --
without adding a parameter, so no existing tracks variant moves.

**``X`` is not TRex's ``X``.** In TRex the ``#`` suffix names the *source* of a
value, and the bare name means ``#head``: the head position, and only where
posture was calculated. ``#wcentroid`` is the body centre. Every other tracker
mosaic supports puts a body centre in ``X``, so a feature reading ``X`` across
trackers would be comparing a head to a centroid, and would find nothing on
TRex frames without posture. So ``X``/``Y`` are taken from ``#wcentroid``,
TRex's own bare pair is preserved as ``X#head``/``Y#head``, and ``#wcentroid``
is kept as well -- one body centre under both names, so TRex-aware code that
already reads ``X#wcentroid`` keeps working.
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

from mosaic.core.track_library.helpers import column_array, column_names
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)

__all__ = [
    "CALIBRATION_COLUMN",
    "MissingBodyCentreError",
    "MissingTrexCalibrationError",
    "TrexNpzConverter",
    "UnknownTrexUnitsError",
    "calibration_from_frame",
    "load_npz_to_df",
    "name_the_body_centre",
    "unscale_to_pixels",
]


# --- TRex per-id NPZ support ---
# Matches: _id0, _id1, _fish0, _fish1, _bee0, _bee1, etc. Read for two different
# questions -- which individual a file holds (below) and which sequence it
# belongs to (TrexNpzConverter.sequence_from_stem) -- so it stays module-level.
_TREX_ID_SUFFIX = re.compile(r"_(?:id|fish|bee|animal|ind)(\d+)$", re.IGNORECASE)

CALIBRATION_COLUMN = "cm_per_pixel"
"""The field TRex writes its applied pixel-to-centimetre factor into.

Written as a one-element array rather than a scalar, so the flattener pads it to
full length with NaN: the value sits in the first row of each individual's block
and nowhere else. That is why it is read by dropping NaN rather than by position.
"""


class MissingTrexCalibrationError(ValueError):
    """A TRex export does not record the factor it scaled its positions by.

    Fatal rather than assumed away. Defaulting to ``1.0`` would read a calibrated
    file's centimetres as pixels and be wrong by exactly the factor the file
    failed to mention -- and wrong quietly, since every number stays plausible.
    """


class MissingBodyCentreError(ValueError):
    """A TRex export carries no ``#wcentroid`` pair, so it records no body centre.

    Raised rather than falling back to the bare ``X``/``Y``, which are the *head*
    position under a name that looks interchangeable. See the module docstring.
    """


class UnknownTrexUnitsError(ValueError):
    """A calibrated file carries a column whose units are not classified.

    ``output_fields`` is user-configurable, so the set of columns a TRex export
    holds is open. Guessing wrong is a silent constant-factor error, so an
    unrecognized numeric column refuses -- but only when there is a conversion to
    get wrong. At ``cm_per_pixel == 1`` centimetres and pixels are the same
    number, no column is touched, and any field set is accepted.
    """


# Base field names TRex reports in centimetres, verified against its own
# ``output_annotations`` map and the functions behind the unannotated ones.
# Read after stripping the ``#`` suffix, so ``X``, ``X#wcentroid`` and
# ``X#pcentroid`` classify together.
#
# Two of these are counter-intuitive enough to be worth stating: ``midline_x``
# and ``midline_y`` *are* multiplied by ``cm_per_pixel``, and so is
# ``midline_segment_length`` -- but ``midline_length`` is not (its conversion is
# commented out in ``OutputLibrary.cpp``), so it stays in pixels despite sitting
# beside them under a name that reads like a length.
_LENGTH_FIELDS: frozenset[str] = frozenset(
    {
        "X",
        "Y",
        "VX",
        "VY",
        "AX",
        "AY",
        "SPEED",
        "SPEED_SMOOTH",
        "SPEED_OLD",
        "ACCELERATION",
        "ACCELERATION_SMOOTH",
        "BORDER_DISTANCE",
        "NEIGHBOR_DISTANCE",
        "midline_x",
        "midline_y",
        "midline_segment_length",
    }
)

# Base field names that carry no length dimension: angles, counts, identifiers,
# probabilities, timings, and the pixel-valued fields TRex never scales.
# ``MIDLINE_OFFSET`` is here because it returns an *angle* despite its name, and
# ``midline_length`` because its centimetre conversion is disabled upstream.
_DIMENSIONLESS_FIELDS: frozenset[str] = frozenset(
    {
        "ANGLE",
        "ORIENTATION",
        "ANGULAR_V",
        "ANGULAR_A",
        "MIDLINE_OFFSET",
        "midline_length",
        "midline_lengths",
        "normalized_midline",
        "num_pixels",
        "outline_size",
        "variance",
        "time",
        "timestamp",
        "frame",
        "frames",
        "missing",
        "id",
        "qr_id",
        "detection_p",
        "visual_identification_p",
        "tracklet_id",
        "tracklets",
        "tracklet_vxys",
        "blobid",
        "blob_id",
        "frame_rate",
        "video_size",
        "detect_type",
        "detect_format",
        "group",
        "sequence",
        CALIBRATION_COLUMN,
    }
)

# Keypoint columns are already pixels -- TRex passes through what the pose model
# reported, unscaled -- and are prefix-shaped rather than named, so they are
# matched separately from the two sets above.
_PIXEL_PREFIXES: tuple[str, ...] = ("poseX", "poseY", "poseP", "pose")


def base_field(column: str) -> str:
    """A column name without its TRex source suffix (``X#wcentroid`` -> ``X``)."""
    return column.split("#", 1)[0]


def calibration_from_frame(frame: pd.DataFrame) -> float | None:
    """The ``cm_per_pixel`` TRex applied, or ``None`` when the file records none.

    Reads the value TRex itself wrote, which is the only trustworthy source --
    see the module docstring. Returns ``None`` rather than assuming ``1.0``: a
    file that does not say is not a file that says one, and the caller decides
    whether that is fatal.

    Raises:
        ValueError: If the frame carries more than one distinct factor. That
            means several individuals were exported under different calibrations
            and merged, and there is no defensible single answer -- averaging
            them would put every column off by a different amount.
    """
    if CALIBRATION_COLUMN not in column_names(frame):
        return None
    raw = column_array(frame, CALIBRATION_COLUMN).astype("float64", copy=False)
    present = raw[~np.isnan(raw)]
    distinct = sorted({float(value) for value in present.tolist()})
    if not distinct:
        return None
    if len(distinct) > 1:
        raise ValueError(
            f"This table records {len(distinct)} different {CALIBRATION_COLUMN} "
            f"values ({distinct}). Its individuals were exported under different "
            "calibrations, so no single factor converts it."
        )
    return distinct[0]


def unscale_to_pixels(frame: pd.DataFrame, cm_per_pixel: float) -> pd.DataFrame:
    """Return *frame* with its centimetre columns divided back into pixels.

    A no-op at ``cm_per_pixel == 1``, where the two units are the same number.
    That is also the only case in which an unclassified column is tolerated:
    with nothing to convert there is nothing to get wrong.

    Raises:
        UnknownTrexUnitsError: If a calibrated frame carries a numeric column
            whose units are not classified.
    """
    if cm_per_pixel == 1.0:
        return frame

    if cm_per_pixel <= 0.0:
        raise ValueError(
            f"{CALIBRATION_COLUMN}={cm_per_pixel} is not a usable scale factor."
        )

    unknown: list[str] = []
    scaled: dict[str, np.ndarray] = {}
    for name in column_names(frame):
        base = base_field(name)
        if base in _LENGTH_FIELDS:
            values = column_array(frame, name).astype("float64", copy=False)
            scaled[name] = values / cm_per_pixel
            continue
        if base in _DIMENSIONLESS_FIELDS or name.startswith(_PIXEL_PREFIXES):
            continue
        # Only a numeric column can carry a unit to get wrong; a label or a flag
        # is passed through whatever it is called. ``kind`` is "f" for floats and
        # "iu" for integers, and deliberately not "b": a boolean is a flag, so
        # dividing one would be meaningless rather than merely unscaled.
        if column_array(frame, name).dtype.kind in "fiu":
            unknown.append(name)

    if unknown:
        raise UnknownTrexUnitsError(
            f"Cannot convert this TRex table to pixels: {sorted(unknown)} "
            f"{'are' if len(unknown) > 1 else 'is'} not classified as a length or as "
            f"dimensionless, and {CALIBRATION_COLUMN}={cm_per_pixel} means guessing "
            "would scale them wrongly. TRex's `output_fields` is configurable, so a "
            "field mosaic has not seen is expected; classify it in "
            "mosaic.core.track_library.trex."
        )

    out = frame.copy()
    for name, values in scaled.items():
        out[name] = values
    return out


def load_npz_to_df(filepath: Path) -> pd.DataFrame:
    """
    Flatten a TRex-like NPZ (per-id) into a DataFrame, robust to arrays with
    slightly different lengths in the same file. We pick the most common length
    across arrays as the target 'n', and truncate longer arrays to fit.
    """
    data = np.load(filepath, allow_pickle=True)
    keys = list(data.files)

    skip_keys = {}  # including all for now.  Could make this a parameter to pass

    # Determine candidate lengths per key
    lens = []
    for k in keys:
        if k in skip_keys:
            continue
        v = data[k]
        if getattr(v, "ndim", 0) > 0:
            lens.append(int(v.shape[0]))
    if not lens:
        raise ValueError(f"No array-like keys with length found in NPZ: {filepath}")

    # Prefer 'time' length if present and 1D; else use the mode length
    if "time" in data.files and getattr(data["time"], "ndim", 0) == 1:
        n = int(data["time"].shape[0])
    else:
        # mode (most common) length among arrays
        vals, counts = np.unique(np.array(lens), return_counts=True)
        n = int(vals[np.argmax(counts)])

    cols: dict[str, np.ndarray] = {}

    for k in sorted(keys):
        if k in skip_keys:
            continue
        v = data[k]

        if np.ndim(v) == 0:
            # scalar -> broadcast
            cols[k] = np.repeat(v.item(), n)
            continue

        # 1D or ND: align first dimension to n by truncation (or pad if you prefer)
        if v.shape[0] < n:
            # If you want to pad instead of skip, do it here; for now, we take the shorter n
            # but to keep a consistent table width we'll just truncate n down for this column.
            vi = v  # keep as-is; we will align by slicing [:vi.shape[0]] and then pad
            # simple pad with NaN to length n
            pad = (
                np.full((n - vi.shape[0],), np.nan, dtype=float)
                if v.ndim == 1
                else np.full((n - vi.shape[0],) + v.shape[1:], np.nan, dtype=float)
            )
            vi = np.concatenate([vi, pad], axis=0)
        else:
            vi = v[:n]

        if vi.ndim == 1:
            cols[k] = vi
        else:
            flat = vi.reshape(n, -1)
            for i in range(flat.shape[1]):
                cols[f"{k}_{i}"] = flat[:, i]

    df = pd.DataFrame(cols)

    # TRex uses inf for undetected keypoints — standardize to NaN
    float_cols = df.select_dtypes(include=[np.floating]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].replace([np.inf, -np.inf], np.nan)

    # id
    if "id" in data.files and np.ndim(data["id"]) > 0:
        try:
            id_val = int(np.array(data["id"]).ravel()[0])
        except Exception:
            # fallback for weird dtypes
            id_val = int(np.array(data["id"]).ravel()[0].astype(np.int64))
    else:
        # derive from filename suffix _id<digits>
        m = _TREX_ID_SUFFIX.search(filepath.stem)
        id_val = int(m.group(1)) if m else 0
    df["id"] = id_val

    # frame/time
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df), dtype=int)
    if "time" not in df.columns:
        if "frame_rate" in data.files:
            fr = float(np.array(data["frame_rate"]).ravel()[0])
            fr = fr if fr > 0 else 1.0
            df["time"] = df["frame"] / fr
        else:
            df["time"] = df["frame"].astype(float)

    return df


def name_the_body_centre(frame: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    """Move TRex's ``#wcentroid`` pair into ``X``/``Y``, preserving its head pair.

    See the module docstring for why. The order matters: the bare pair is moved
    aside to ``#head`` first, so that assigning ``X`` from ``#wcentroid`` cannot
    overwrite the head position before it has been preserved.

    Raises:
        MissingBodyCentreError: If the file carries no ``#wcentroid`` pair. TRex's
            default ``output_fields`` exports one, so this means a trimmed
            configuration -- and the bare pair is not a substitute, because it is
            the head. Silently promoting it would put a head position in the
            column every other tracker fills with a body centre.
    """
    if not {"X#wcentroid", "Y#wcentroid"} <= set(column_names(frame)):
        raise MissingBodyCentreError(
            f"{source} carries no X#wcentroid/Y#wcentroid pair, so it records no body "
            "centre. TRex's bare X/Y are the *head* and cannot stand in for one. Add "
            "WCENTROID to the X and Y entries of TRex's `output_fields` and re-export."
        )

    out = frame.copy()
    present = set(column_names(frame))
    for axis in ("X", "Y"):
        if axis in present:
            out[f"{axis}#head"] = column_array(frame, axis)
        out[axis] = column_array(frame, f"{axis}#wcentroid")
    return out


@register_track_converter
class TrexNpzConverter(TrackConverter[TrackConvertParams]):
    """Per-id TRex NPZ -> the standard T-Rex-like table.

    Carries no parameters of its own: the conversion is fully determined by the
    file. What it declares instead is the shape of TRex's output -- one file per
    individual, so several files make one sequence. ``merges_per_sequence`` is
    how ``core`` learns that without comparing ``src_format`` to a literal, and
    ``sequence_from_stem`` is the rule that recognizes those files as one entry.
    Both describe TRex's own naming, so neither is a knob and neither is a
    parameter.
    """

    src_format = "trex_npz"
    # 0.2: tables are pixels rather than centimetres, and ``X``/``Y`` name the
    # body centre (TRex's ``#wcentroid``) rather than its head. Both change what
    # the numbers mean, so they are a new recipe rather than the same one.
    version = "0.2"
    output_schema = "trex_v2"
    merges_per_sequence = True
    Params = TrackConvertParams

    def sequence_from_stem(self, stem: str) -> str:
        """The stem with its trailing individual-id suffix removed, if present.

        Handles ``_id0``, ``_fish2``, ``_bee1``, ``_animal3``, ``_ind0``.

        Examples:
            ``hex_7_fish2`` -> ``hex_7``, ``OCI_1_fish0`` -> ``OCI_1``,
            ``video1_id3`` -> ``video1``.
        """
        match = _TREX_ID_SUFFIX.search(stem)
        return stem[: match.start()] if match else stem

    def convert(
        self, path: Path, params: TrackConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        df = load_npz_to_df(path)
        cm_per_pixel = calibration_from_frame(df)
        if cm_per_pixel is None:
            raise MissingTrexCalibrationError(
                f"{path} records no {CALIBRATION_COLUMN}, so there is no way to tell "
                "whether its positions are centimetres or pixels, or by how much they "
                "differ. TRex has written this field into every export since "
                "2025-02-18; an older file has to be re-exported from its .results, or "
                "converted by a reader that is told the factor."
            )
        df = unscale_to_pixels(df, cm_per_pixel)
        df = name_the_body_centre(df, source=path)
        df["group"] = hints.group
        df["sequence"] = hints.sequence or self.sequence_from_stem(path.stem)
        return df
