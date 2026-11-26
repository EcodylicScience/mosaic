"""Converter for the ``.mat`` files of the social s-DANNCE rat dataset.

Converts the per-session ``.mat`` that s-DANNCE (Klibaite et al. 2025, Harvard
Dataverse dataset ``socialDANNCE_data``) writes to the ``mosaic_world_v1`` schema:
meters in a right-handed world frame with +Z up.

The file layout
---------------
A file is a MATLAB v5 MAT-file (``save -v7``) with one variable, ``sdannce``, a 1x1
struct. A MATLAB v7.3 file is HDF5 and is refused with the command that resaves it.
The converter reads these fields and no others:

- ``m1``: the animal that the file names (``ratid``, ``ratint``), a float array
  ``(frames, 3, 23)`` of frame, axis (x, y, z) and keypoint, in millimeters.
- ``m2``: its partner (``ratp_id``, ``ratp_int``), the same shape, or empty in a lone
  file.
- ``ratint`` and ``ratp_int``: the two animals' integer numbers.
- ``issoc``: 1 for a social file and 0 for a lone one, checked against ``m2`` when
  the file has it.

The dataset's ``Documentation/README.txt`` describes ``m1`` as the focal animal and
``m2`` as its partner. A dyad session is published once for each focal animal, with
``m1`` and ``m2`` swapped, and some cohorts add ``_S-1`` files. The converter
converts each file it is given as it stands. The cluster labels, the embeddings and
the recording metadata (``ratgroup``, ``ratdate``, ``ratgen`` and the rest) are not
read.

The 23 keypoints are the rat skeleton of the paper, in file order
(``RAT23_KEYPOINT_NAMES``), and a file whose keypoint axis has another length is
refused. A track table names a keypoint by its index, so a feature takes the names
through ``keypoint_names``. The converter reads the rat cohorts. The dataset also
lists a mouse cohort whose files are unchecked, and one with another keypoint count
is refused by the shape check.

Identity and rows
-----------------
``id`` is the animal's ``ratint`` number. An array holds one named animal for the
whole file, so the ids follow animals and the converter declares
``carries_tracklets``. A file whose ``issoc`` disagrees with ``m2``, whose full ``m2``
has no ``ratp_int``, or whose two numbers are equal is refused.

Each animal has a row on every frame of its array. An animal-frame whose keypoints
are all NaN keeps its row with NaN keypoints, and one with some NaN keypoints keeps
its row with those keypoints NaN. ``X``/``Y``/``Z`` is the mean of the keypoints
that have all three axes on that row, and NaN where none does. Every NaN stays NaN.

The frame and the rate
----------------------
``meters_per_unit`` scales every value (s-DANNCE writes millimeters), and ``axes``
then names which source axis becomes world X, Y and Z, each with an optional
leading ``-``. It must be a proper rotation, so a reflection is refused. The default
is the file's own frame, which has +Z up and the floor near 0.

A file records no frame rate, so ``time`` is ``frame / rate`` through
``conversion_rate``: ``fps``, else the entry's measured media rate, else the
dataset's ``fps_default`` (the published files run at 50 Hz), else a refusal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Final, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from mosaic.core.narrowing import is_object_mapping
from mosaic.core.params import Declared
from mosaic.core.track_converter import (
    EntryHints,
    TimedTrackConvertParams,
    TrackConverter,
    conversion_rate,
    register_track_converter,
)
from mosaic.core.track_library.helpers import norm_hint
from mosaic.core.track_library.world_frame import (
    WorldAxes,
    animal_columns,
    stacked_table,
    to_world,
)

RAT23_KEYPOINT_NAMES: Final = (
    "Snout",
    "EarL",
    "EarR",
    "SpineF",
    "SpineM",
    "SpineL",
    "TailBase",
    "ShoulderL",
    "ElbowL",
    "WristL",
    "HandL",
    "ShoulderR",
    "ElbowR",
    "WristR",
    "HandR",
    "HipL",
    "KneeL",
    "AnkleL",
    "FootL",
    "HipR",
    "KneeR",
    "AnkleR",
    "FootR",
)
"""The keypoints of a rat file, in the order of its last axis."""

_VARIABLE: Final = "sdannce"
_N_KEYPOINTS: Final = len(RAT23_KEYPOINT_NAMES)
_EXPECTED_SHAPE: Final = f"(frames, 3, {_N_KEYPOINTS})"

_METERS_PER_UNIT_DESCRIPTION = (
    "Meters in one unit of the file's coordinates. s-DANNCE writes millimeters. "
    "The default is 0.001."
)
_AXES_DESCRIPTION = (
    "The source axis that becomes world X, Y and Z, each one of x, y or z with an "
    "optional leading minus. ('x', '-z', 'y') takes world Y from minus source z "
    "and world Z from source y. It must be a proper rotation, so a reflection is "
    "refused. The default is the file's frame, which has +Z up."
)


class SdannceMatParams(TimedTrackConvertParams):
    """Everything that determines the converter's output besides the file."""

    meters_per_unit: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False),
        Declared(_METERS_PER_UNIT_DESCRIPTION),
    ] = 0.001
    axes: Annotated[WorldAxes, Declared(_AXES_DESCRIPTION)] = ("x", "y", "z")


def _read_struct(path: Path) -> Mapping[object, object]:
    """The ``sdannce`` struct of the MAT-file at *path*, as a mapping.

    Raises:
        ValueError: If the file is not a MATLAB v5 or v7 MAT-file, or has no
            ``sdannce`` struct.
    """
    from scipy.io.matlab import loadmat, matfile_version

    try:
        major, _ = matfile_version(path, appendmat=False)
    except ValueError as error:
        msg = f"{path}: not a MATLAB v5/v7 MAT-file ({error})"
        raise ValueError(msg) from error
    if major == 2:
        msg = (
            f"{path}: a MATLAB v7.3 (HDF5) file, which this converter does not "
            "read. Save it again with save(..., '-v7')"
        )
        raise ValueError(msg)
    if major != 1:
        msg = f"{path}: not a MATLAB v5/v7 MAT-file (version {major})"
        raise ValueError(msg)
    loaded = loadmat(
        path,
        appendmat=False,
        spmatrix=False,
        simplify_cells=True,
        variable_names=[_VARIABLE],
    )
    struct = loaded.get(_VARIABLE)
    if not is_object_mapping(struct):
        msg = f"{path}: the file has no {_VARIABLE!r} variable holding a struct"
        raise ValueError(msg)
    return struct


def _field(path: Path, struct: Mapping[object, object], name: str) -> object:
    """The field *name* of *struct*.

    Raises:
        ValueError: If the struct has no such field.
    """
    if name not in struct:
        msg = f"{path}: the {_VARIABLE!r} struct has no field {name!r}"
        raise ValueError(msg)
    return struct[name]


def _float_array(path: Path, name: str, value: object) -> npt.NDArray[np.float64]:
    """*value* as a float array.

    Raises:
        ValueError: If it is not a floating-point array.
    """
    array = np.asarray(value)
    if array.dtype.kind != "f":
        msg = f"{path}: {name} must hold floating-point values, not {array.dtype}"
        raise ValueError(msg)
    return array.astype(np.float64, copy=False)


def _with_frame_axis(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """*array*, with the frame axis back when ``simplify_cells`` squeezed it.

    ``simplify_cells`` implies ``squeeze_me``, which loads a one-frame array of
    ``(1, 3, 23)`` as ``(3, 23)``.
    """
    if array.shape == (3, _N_KEYPOINTS):
        return array[np.newaxis]
    return array


def _read_focal(path: Path, value: object) -> npt.NDArray[np.float64]:
    """The ``m1`` array, ``(frames, 3, 23)`` with at least one frame.

    Raises:
        ValueError: If it is not a float array of that shape.
    """
    array = _float_array(path, "m1", value)
    shape = array.shape
    array = _with_frame_axis(array)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1:] != (3, _N_KEYPOINTS):
        msg = (
            f"{path}: m1 has shape {shape}, expected {_EXPECTED_SHAPE} with at "
            "least one frame"
        )
        raise ValueError(msg)
    return array


def _read_partner(
    path: Path, value: object, focal: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | None:
    """The ``m2`` array, or None when it is empty.

    Raises:
        ValueError: If a full ``m2`` is not a float array shaped like ``m1``.
    """
    if np.asarray(value).size == 0:
        return None
    array = _float_array(path, "m2", value)
    shape = array.shape
    array = _with_frame_axis(array)
    if array.shape != focal.shape:
        msg = (
            f"{path}: m2 has shape {shape}, expected {focal.shape}, the shape of "
            "m1, or no elements"
        )
        raise ValueError(msg)
    return array


def _integer_scalar(path: Path, name: str, value: object) -> int:
    """*value* as an integer.

    Raises:
        ValueError: If it is not a finite, integer-valued number.
    """
    raw = np.asarray(value)
    if raw.size != 1 or raw.dtype.kind not in "iuf":
        msg = f"{path}: {name} must be an integer-valued scalar, not {value!r}"
        raise ValueError(msg)
    number = float(raw.astype(np.float64).reshape(-1)[0])
    if not math.isfinite(number) or number != math.floor(number):
        msg = f"{path}: {name} must be an integer-valued scalar, not {value!r}"
        raise ValueError(msg)
    return int(number)


def _animal_ids(
    path: Path, struct: Mapping[object, object], *, social: bool
) -> tuple[int, int | None]:
    """The ids of the focal animal and, for a social file, its partner.

    Raises:
        ValueError: If the struct's ids or ``issoc`` contradict ``m2``.
    """
    focal = _integer_scalar(path, "ratint", _field(path, struct, "ratint"))
    if "issoc" in struct:
        flag = _integer_scalar(path, "issoc", struct["issoc"])
        if flag not in (0, 1) or bool(flag) != social:
            msg = (
                f"{path}: issoc is {flag}, but m2 is "
                f"{'full' if social else 'empty'}. A social file has issoc 1 and a "
                "full m2, and a lone file issoc 0 and an empty m2"
            )
            raise ValueError(msg)
    if not social:
        return focal, None
    partner_value = _field(path, struct, "ratp_int")
    if np.asarray(partner_value).size == 0:
        msg = f"{path}: m2 is full but ratp_int is empty, so its animal has no id"
        raise ValueError(msg)
    partner = _integer_scalar(path, "ratp_int", partner_value)
    if partner == focal:
        msg = f"{path}: ratint and ratp_int are both {focal}, one id for two animals"
        raise ValueError(msg)
    return focal, partner


@register_track_converter
class SdannceMatConverter(TrackConverter[SdannceMatParams]):
    """An s-DANNCE ``.mat`` file -> a ``mosaic_world_v1`` table."""

    src_format = "sdannce_mat"
    version = "0.1"
    output_schema = "mosaic_world_v1"
    carries_tracklets = True
    enumerable = False
    merges_per_sequence = False
    Params = SdannceMatParams

    @override
    def convert(
        self, path: Path, params: SdannceMatParams, hints: EntryHints
    ) -> pd.DataFrame:
        struct = _read_struct(path)
        focal = _read_focal(path, _field(path, struct, "m1"))
        partner = _read_partner(path, _field(path, struct, "m2"), focal)
        focal_id, partner_id = _animal_ids(path, struct, social=partner is not None)
        rate = conversion_rate(params.fps, params.fallback_fps, hints, str(path))

        group = norm_hint(hints.group) or ""
        sequence = norm_hint(hints.sequence) or path.stem
        frames = np.arange(focal.shape[0], dtype=np.int64)

        animals = [(focal_id, focal)]
        if partner is not None and partner_id is not None:
            animals.append((partner_id, partner))
        per_animal = [
            animal_columns(
                animal_id,
                frames,
                to_world(array.transpose(0, 2, 1), params.meters_per_unit, params.axes),
                rate,
                group,
                sequence,
            )
            for animal_id, array in animals
        ]
        return stacked_table(per_animal)
