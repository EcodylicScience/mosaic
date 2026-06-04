"""Backward-compatibility shim: ``mosaic.media`` moved to :mod:`mosaic.core.media`.

Media I/O is foundational infrastructure (read/decode/encode frames), so it now
lives under ``core`` alongside the pipeline engine. Importing from
``mosaic.media`` — including submodules like ``mosaic.media.video_io`` — still
works, but is deprecated; prefer ``mosaic.core.media``.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

from mosaic.core.media import *  # noqa: F401,F403

# Preserve old deep-import paths, e.g. ``from mosaic.media.video_io import X``, by
# aliasing the real modules under the legacy ``mosaic.media.*`` names. (Frame
# extraction/sampling moved to ``mosaic.tracking.frame_extraction``, so only the
# low-level I/O modules are aliased here.)
for _name in ("video_io", "imgstore_io"):
    _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(
        f"mosaic.core.media.{_name}"
    )
