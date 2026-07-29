"""SLEAP CLI integration for pose inference + identity tracking.

This module wraps the SLEAP command-line interface, enabling mosaic to run
``sleap-track`` inference + tracking headlessly against a pre-trained SLEAP model
and bridge the results into standardized tracks.

Requires:
    The ``sleap-track`` / ``sleap-convert`` console scripts (https://sleap.ai).
    SLEAP 1.6 is heavy (PyTorch + Qt), so it usually lives in its **own**
    environment; point the wrappers at it with ``sleap_conda_env=`` /
    ``MOSAIC_SLEAP_CONDA_ENV`` (or ``sleap_bin=`` / ``MOSAIC_SLEAP_BIN``), else
    the console scripts are found on ``$PATH`` (the ``uv tool install`` case).
    Unlike TRex, SLEAP inference is headless and needs no ``DISPLAY``.

Usage
-----
>>> from mosaic.tracking.sleap import run_sleap_track, run_sleap_convert
>>>
>>> result = run_sleap_track(
...     "video.mp4", "out/video.predictions.slp", model_paths=["models/centroid"],
... )
>>> analysis = run_sleap_convert(result.slp_path, "out/video.analysis.h5")

The dataset-level entry point runs it as a tracked, tracks-integrated job:

>>> from mosaic.tracking.sleap import run_sleap
>>> run_sleap(ds, model_paths=["models/td_centroid", "models/td_instance"])

Equivalently set ``MOSAIC_SLEAP_CONDA_ENV=sleap`` once and drive it from the
``mosaic run --kind sleap`` op.
"""

from mosaic.tracking.sleap.dataset_runs import (
    ResolvedSleapModels,
    SleapIndexRow,
    list_sleap_runs,
    resolve_sleap_models,
    run_sleap,
)
from mosaic.tracking.sleap.run import (
    SleapConvertResult,
    SleapError,
    SleapNotFoundError,
    SleapTrackResult,
    run_sleap_convert,
    run_sleap_track,
)

__all__ = [
    "ResolvedSleapModels",
    "SleapConvertResult",
    "SleapError",
    "SleapIndexRow",
    "SleapNotFoundError",
    "SleapTrackResult",
    "list_sleap_runs",
    "resolve_sleap_models",
    "run_sleap",
    "run_sleap_convert",
    "run_sleap_track",
]
