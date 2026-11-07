"""Shared utilities for track format converters."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
import json
import numpy as np

from mosaic.core.helpers import text_cell
from mosaic.core.narrowing import is_object_mapping
from mosaic.core.track_library.safe_load import load_pickled_npy_dict

if TYPE_CHECKING:
    import pandas as pd


def column_names(frame: "pd.DataFrame") -> list[str]:
    """The frame's column labels as plain strings.

    A pandas ``Index`` is keyed by an untyped label type, so iterating it
    directly leaves every name unknown to the type checker and every use of one
    unknown in turn. Materializing the labels once, here, keeps that confined to
    a single expression instead of spreading through every caller's loop.
    """
    return [str(name) for name in frame.columns.tolist()]


def column_array(frame: "pd.DataFrame", name: str) -> np.ndarray:
    """One column as a numpy array, for the same reason as :func:`column_names`.

    numpy's dtypes are typed where pandas' element access is not, so a caller
    that needs to ask what a column *holds* -- rather than merely carry it --
    gets a typed answer by going through this.
    """
    return np.asarray(frame[name])


type Calms21Nested = dict[str, dict[str, Mapping[str, object]]]
"""A CalMS21 file's contents: group -> sequence id -> the sequence's fields."""


def _nested_calms21(path: Path, loaded: object) -> Calms21Nested:
    """*loaded*, the contents of the CalMS21 file at *path*, as nested mappings.

    Raises:
        ValueError: If a level of *loaded* is not a mapping.
    """
    if not is_object_mapping(loaded):
        raise ValueError(
            f"{path}: expected a CalMS21 dict, got {type(loaded).__name__}"
        )
    nested: Calms21Nested = {}
    for group, sequences in loaded.items():
        if not is_object_mapping(sequences):
            raise ValueError(f"{path}: CalMS21 group {group!r} is not a dict")
        nested[str(group)] = {}
        for sequence_id, fields in sequences.items():
            if not is_object_mapping(fields):
                raise ValueError(
                    f"{path}: CalMS21 sequence {sequence_id!r} is not a dict"
                )
            nested[str(group)][str(sequence_id)] = {
                str(name): value for name, value in fields.items()
            }
    return nested


def load_calms21(path: Path | str) -> Calms21Nested:
    """Load a single CalMS21 file: either .npy (dict) or the original .json.

    A CalMS21 ``.npy`` is a pickled dict, so it loads by unpickling, which can run
    code. It loads only when ``MOSAIC_TRUST_PICKLED_FILES=1`` says the file is
    trusted. The original ``.json`` always loads.

    Args:
        path: The ``.npy`` or ``.json`` file.

    Returns:
        A nested dict: group -> seq_id -> dict(...)

    Raises:
        PickledFormatNotTrustedError: If *path* is a ``.npy`` and pickled files are
            not trusted.
        ValueError: If *path* is neither a ``.npy`` nor a ``.json``, or its
            contents are not nested dicts.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        loaded = load_pickled_npy_dict(p, "CalMS21 .npy", "the original CalMS21 .json")
        return _nested_calms21(p, loaded)
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return _nested_calms21(p, json.load(f))
    else:
        raise ValueError(f"Unsupported CalMS21 path (expect .npy or .json): {p}")


def norm_hint(x: object) -> str | None:
    """A group/sequence hint, absent spellings collapsed to ``None``.

    One delegation rather than a second implementation of what counts as absent:
    a converter reading a hint and the index writer recording what it produced
    have to agree, and two copies of that rule are free to drift. ``None`` here
    rather than ``""`` because the callers spell their fallback as ``or``.
    """
    return text_cell(x) or None
