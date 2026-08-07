"""Shared utilities for track format converters."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import json
import numpy as np

from mosaic.core.helpers import text_cell

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


def load_calms21(path: Path | str):
    """
    Load a single CalMS21 file: either .npy (dict) or the original .json.
    Returns a nested dict: group -> seq_id -> dict(...)
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p, allow_pickle=True).item()
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f)
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
