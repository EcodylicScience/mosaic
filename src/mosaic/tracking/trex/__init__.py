"""T-Rex CLI integration for automated video conversion and tracking.

This module wraps the T-Rex command-line interface, enabling Mosaic to
convert raw videos into T-Rex .pv format and run tracking headlessly.

Requires:
    The ``trex`` binary (https://trex.run). TRex pins ``python=3.11`` /
    ``numpy=1.26``, so it usually lives in its **own** conda env. Point the
    wrappers at it with a :class:`~mosaic.tracking.common.toolenv.ToolEnv` built
    from :data:`~mosaic.tracking.trex.run.TREX_ENV`, or with
    ``MOSAIC_TREX_CONDA_ENV`` / ``MOSAIC_TREX_BIN``; otherwise ``trex`` is found
    on ``$PATH``. TRex needs a display even headless -- run one persistent
    ``Xvfb`` and name it in the same ``ToolEnv`` (see
    :mod:`mosaic.tracking.trex.run`).

Usage
-----
>>> from pathlib import Path
>>> from mosaic.tracking.trex import run_trex_convert, run_trex_track, run_trex_batch
>>> from mosaic.tracking.trex.params import TrexParams
>>>
>>> params = TrexParams(detect_type="yolo", track_max_individuals=4)
>>> result = run_trex_convert(
...     "video.mp4", "output/", params=params, detect_model_path=Path("model.pt")
... )
>>> result = run_trex_track(result.pv_path, "output/", params=params)
>>> results = run_trex_batch(
...     ["v1.mp4", "v2.mp4"], "output/", params=params,
...     detect_model_path=Path("model.pt"),
... )

When TRex lives in its own conda env (the usual case), drive it cross-env and
give it a headless display (one persistent ``Xvfb :99`` running):

>>> from dataclasses import replace
>>> from mosaic.tracking.trex.run import TREX_ENV
>>>
>>> result = run_trex_convert(
...     "video.mp4", "output/", params=params,
...     env=replace(TREX_ENV, conda_env="track", display=":99"),
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
