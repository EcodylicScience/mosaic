"""T-Rex CLI integration for automated video conversion and tracking.

This module wraps the T-Rex command-line interface, enabling Mosaic to
convert raw videos into T-Rex .pv format and run tracking headlessly.

Requires:
    The ``trex`` binary must be installed and available on ``$PATH``.
    See https://trex.run for installation instructions.

Usage
-----
>>> from mosaic.tracking.trex import run_trex_convert, run_trex_track, run_trex_batch
>>>
>>> result = run_trex_convert("video.mp4", "output/", detect_model="model.pt")
>>> result = run_trex_track("video.pv", "output/", track_max_individuals=4)
>>> results = run_trex_batch(["v1.mp4", "v2.mp4"], "output/", detect_model="model.pt")
"""

from mosaic.tracking.trex.run import (
    TRexConvertResult,
    TRexError,
    TRexNotFoundError,
    TRexTrackResult,
    run_trex_batch,
    run_trex_convert,
    run_trex_track,
)
from mosaic.tracking.trex.settings import generate_settings_file

__all__ = [
    "TRexConvertResult",
    "TRexError",
    "TRexNotFoundError",
    "TRexTrackResult",
    "generate_settings_file",
    "run_trex_batch",
    "run_trex_convert",
    "run_trex_track",
]
