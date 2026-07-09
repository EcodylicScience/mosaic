"""T-Rex CLI integration for automated video conversion and tracking.

This module wraps the T-Rex command-line interface, enabling Mosaic to
convert raw videos into T-Rex .pv format and run tracking headlessly.

Requires:
    The ``trex`` binary (https://trex.run). TRex pins ``python=3.11`` /
    ``numpy=1.26``, so it usually lives in its **own** conda env; point the
    wrappers at it with ``trex_conda_env=`` / ``MOSAIC_TREX_CONDA_ENV`` (or
    ``trex_bin=`` / ``MOSAIC_TREX_BIN``), else ``trex`` is found on ``$PATH``.
    TRex needs a display even headless — run one persistent ``Xvfb`` and pass
    ``display=`` (see :mod:`mosaic.tracking.trex.run`).

Usage
-----
>>> from mosaic.tracking.trex import run_trex_convert, run_trex_track, run_trex_batch
>>>
>>> result = run_trex_convert("video.mp4", "output/", detect_model="model.pt")
>>> result = run_trex_track("video.pv", "output/", track_max_individuals=4)
>>> results = run_trex_batch(["v1.mp4", "v2.mp4"], "output/", detect_model="model.pt")

When TRex lives in its own conda env (the usual case), drive it cross-env and
give it a headless display (one persistent ``Xvfb :99`` running):

>>> result = run_trex_convert(
...     "video.mp4", "output/", detect_model="model.pt",
...     trex_conda_env="track", display=":99",
... )

Equivalently set ``MOSAIC_TREX_CONDA_ENV=track`` and ``DISPLAY=:99`` once.
"""

from mosaic.tracking.trex.dataset_runs import (
    TRexIndexRow,
    list_trex_runs,
    run_trex,
)
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
    "TRexIndexRow",
    "TRexNotFoundError",
    "TRexTrackResult",
    "generate_settings_file",
    "list_trex_runs",
    "run_trex",
    "run_trex_batch",
    "run_trex_convert",
    "run_trex_track",
]
