"""SLEAP CLI integration for pose inference + identity tracking.

This module wraps the SLEAP command-line interface, enabling mosaic to run
``sleap-nn track`` inference + tracking headlessly against a pre-trained SLEAP
model, or ``sleap-nn predict`` appearance tracking when the last model is an
embedding model, and bridge the results into standardized tracks.

Requires:
    An environment with ``sleap-nn`` and its ``sleap-io`` (https://sleap.ai).
    The ``sleap-nn`` console script infers, and the environment's ``python``
    writes the analysis export through ``sleap-io``. SLEAP is heavy (PyTorch),
    so it usually lives in its **own** environment; point the wrappers at
    it with ``sleap_conda_env=`` / ``MOSAIC_SLEAP_CONDA_ENV`` (or ``sleap_bin=``
    / ``MOSAIC_SLEAP_BIN``), else the environment is found through the ``sleap``
    distribution's ``sleap-convert`` on ``$PATH`` (the ``uv tool install
    "sleap[nn]"`` case).
    Unlike TRex, SLEAP inference is headless and needs no ``DISPLAY``.

Usage
-----
>>> from mosaic.tracking.sleap import run_sleap_track, run_sleap_analysis_export
>>>
>>> result = run_sleap_track(
...     "video.mp4", "out/video.predictions.slp", model_paths=["models/centroid"],
... )
>>> analysis = run_sleap_analysis_export(result.slp_path, "out/video.analysis.h5")

The dataset-level entry point runs it as a tracked, tracks-integrated job:

>>> from mosaic.tracking.sleap import SleapParams, run_sleap
>>> run_sleap(
...     ds, SleapParams(model_paths=["models/td_centroid", "models/td_instance"]),
... )

Equivalently set ``MOSAIC_SLEAP_CONDA_ENV=sleap`` once and drive it from the
``mosaic run --kind sleap`` op.
"""

from mosaic.tracking.sleap.dataset_runs import (
    SleapIndexRow,
    list_sleap_runs,
    run_sleap,
)
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.sleap.run import (
    SleapAnalysisExportResult,
    SleapError,
    SleapNotFoundError,
    SleapTrackResult,
    run_sleap_analysis_export,
    run_sleap_track,
)

__all__ = [
    "SleapAnalysisExportResult",
    "SleapError",
    "SleapIndexRow",
    "SleapNotFoundError",
    "SleapParams",
    "SleapTrackResult",
    "list_sleap_runs",
    "run_sleap",
    "run_sleap_analysis_export",
    "run_sleap_track",
]
