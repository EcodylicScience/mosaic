"""Whether the imgstore recordings of one entry share a frame rate.

A store records a timestamp for each frame and no rate. mosaic estimates the rate
as ``(frames - 1) / (last timestamp - first timestamp)``, and reads a store's
frames by index. Nothing joins stores, so the stores of one entry are read as one
axis only when they have one rate, and the question is whether two estimates
measure one rate.

The plain multi-video reader asks whether one rate places every frame of every
clip within half a frame, which is a tolerance on the product of the rate
difference and the length. For stores that refuses one rate measured twice once
the stores are long enough: 30.0 and 30.002 fps pass at 7,400 frames and fail at
7,600. The error of an estimate does not grow with the store, so the rule here is
a relative difference of the two rates, :data:`STORE_RATE_TOLERANCE`.

The reader of a store sequence and the check that refuses one before any work
starts (:func:`~mosaic.tracking.common.scope.refuse_unjoinable`) both apply
:func:`store_rate_mismatch`, so for a consumer that reads the stores itself, a
sequence that one accepts the other accepts. A consumer that hands its tool a
path refuses every sequence of several stores, since no file joins them. The
timeline that times a table read from stores, directly or through a media
variant (:func:`~mosaic.core.media.timeline.concatenated_timeline`), classifies
their rate by the same rule, so stores that the reader reads as one rate keep the
columns that one rate computes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

__all__ = ["STORE_RATE_TOLERANCE", "store_rate_mismatch"]

STORE_RATE_TOLERANCE: Final = 5e-4
"""The largest relative difference between two estimates of one store rate.

It sits between what one rate measured twice can differ by and what two rates
do differ by:

- Two recording machines' clocks drift apart by tens of parts per million, and
  timestamps a few milliseconds off at the two ends of a store of a minute or more
  move its estimate by less than 1e-4. 30.0 and 30.002 fps differ by 6.7e-5.
- The two closest rates that cameras are set to, 30 and 29.97 (30000/1001) fps,
  differ by 1/1001, twice this. 30 and 31 fps differ by 3.2 percent.

A store of a few seconds can mismeasure its rate by more than this, and is refused
as a store at another rate is.
"""


def store_rate_mismatch(rates: Sequence[float]) -> int | None:
    """Return the position of the first store at another rate, or ``None``.

    A rate differs when it differs from the first store's by more than
    :data:`STORE_RATE_TOLERANCE` of the first, and a rate that is not positive
    differs from every rate. ``None`` for fewer than two stores.
    """
    if len(rates) < 2:
        return None
    reference = rates[0]
    for position, rate in enumerate(rates[1:], start=1):
        if reference <= 0.0 or rate <= 0.0:
            return position
        if abs(rate - reference) / reference > STORE_RATE_TOLERANCE:
            return position
    return None
