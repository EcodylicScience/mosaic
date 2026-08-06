"""The Ultralytics tracking integration's kind and compatibility version.

A leaf module with no imports, so ``ops/ultralytics.py`` can read both at module
scope while every heavy import stays inside ``run()``.

The directory is ``ultralytics_track`` while the kind is ``ultralytics``. Naming
the package after the library would be legal -- Python 3 imports are absolute, so
``from ultralytics import YOLO`` inside it still reaches the installed
distribution -- but ``mosaic.tracking`` binds its subpackages as attributes, and a
reader meeting ``ultralytics.trackers`` beside ``mosaic.tracking.ultralytics``
has to work out which is which every time. ``frame_extraction/`` already holds an
op named ``extract-frames``, so a directory that does not restate its kind is the
existing habit rather than a new one.
"""

from __future__ import annotations

from typing import Final

ULTRALYTICS_KIND: Final = "ultralytics"

TRAIN_POSE_KIND: Final = "train-pose"
"""Where a model reference that names a run rather than a path is looked up.

A ``run_id`` resolves against ``models/<kind>/index.csv``, and the kind naming
that index belongs to the *training* op, never to this tracker -- mosaic has no
``train-ultralytics``. This is the fallback for a reference that parses as no run
id at all; a reference that does parse names its own kind, because both
``train-pose`` and ``train-points`` produce weights this tracker can run.
"""

ULTRALYTICS_VERSION: Final = "8.4"
"""The *integration's* compatibility version, not the installed library's.

Declared, never detected. Reading ``ultralytics.__version__`` would re-mint every
tracks variant on every upstream patch release, which is exactly what a variant
identifier exists to avoid. Seeded at the release line this integration targets:
8.4.63 is where the last four tracker backends landed, so ``8.4`` is the oldest
line on which all six selectable trackers exist.

Bump it by hand when a vendored default table is re-transcribed, or when the
detect-synthesis, frame-addressing or output-column rules change -- anything that
makes the same settings mean a different table.
"""
