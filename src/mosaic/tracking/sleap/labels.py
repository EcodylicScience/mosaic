"""Writing a mosaic annotation set out as a SLEAP ``.slp`` labels file.

``.slp`` is not a documented file format so much as ``sleap-io``'s serialization
of its own object graph, so writing one means running that library. mosaic does
not depend on it, and should not: it is a SLEAP concern living in the SLEAP
environment, and pulling it into the mosaic environment to write one file would
put the heavy stack in the type checker and in everybody's install.

So mosaic writes what it already knows how to write -- COCO keypoints, which is
the interchange format both sides speak -- and a static program run by the SLEAP
interpreter turns that into a ``.slp``. The bridge is ``sleap_io.load_coco``
followed by ``save_slp``, both of which that library ships.

**The snippet takes only argv.** No path is interpolated into its source, so
there is nothing to quote and nothing a filename can do to it, and mosaic never
imports ``sleap_io``. That is the same arrangement Lightning Pose inference
already uses, for the same reasons.

**The output is published atomically.** Whether a ``.slp`` exists is how a caller
decides the export succeeded, and a killed process that had opened the real path
for writing would leave a truncated file that reads as complete. It is built
beside the target and moved into place.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final

from mosaic.core.annotations.model import AnnotationSet
from mosaic.core.annotations.writers import write_coco_keypoints
from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.pipeline.toolenv import (
    subprocess_env,
    tool_invocation,
)
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapError

logger = logging.getLogger(__name__)

__all__ = [
    "SLEAP_VALIDATION_SUFFIX",
    "validation_labels_beside",
    "validation_labels_path",
    "write_slp",
]

SLEAP_VALIDATION_SUFFIX: Final = ".val.slp"
"""The suffix of the validation file written beside a ``.slp`` of training labels."""

# Run by the SLEAP interpreter. Its only inputs are argv -- the COCO json, the
# directory its file_name values are relative to, and where to write -- so no
# path is ever interpolated into this source. sleap-io's COCO reader names the
# track of each track_id value "track_<value>". When every track carries such a
# name, each is renamed to its value, which is what a SLEAP identity model then
# reports. A reader that names tracks otherwise is left as it is.
_TO_SLP_SNIPPET: str = """
import json
import sys
import sleap_io

_json, _root, _out = sys.argv[1], sys.argv[2], sys.argv[3]
_labels = sleap_io.load_coco(_json, dataset_root=_root)
with open(_json, encoding="utf-8") as _file:
    _annotations = json.load(_file)["annotations"]
_values = {
    f"track_{_each['track_id']}": str(_each["track_id"])
    for _each in _annotations
    if "track_id" in _each
}
if {_track.name for _track in _labels.tracks} <= set(_values):
    for _track in _labels.tracks:
        _track.name = _values[_track.name]
sleap_io.save_slp(_labels, _out)
print(f"wrote {len(_labels.labeled_frames)} labeled frames to {_out}")
"""


def validation_labels_path(labels_path: Path) -> Path:
    """Return the path of the validation file beside *labels_path*.

    The validation file of ``labels.slp`` is ``labels.val.slp``.
    """
    return labels_path.with_name(labels_path.stem + SLEAP_VALIDATION_SUFFIX)


def validation_labels_beside(labels_path: Path) -> Path | None:
    """Return the validation file beside *labels_path* when it exists, else ``None``."""
    validation = validation_labels_path(labels_path)
    return validation if validation.is_file() else None


def write_slp(
    annotations: AnnotationSet,
    out_path: str | Path,
    *,
    images_dir: str | Path | None = None,
    sleap_conda_env: str | None = None,
    sleap_bin: str | Path | None = None,
    idle_timeout: float = 300,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> Path:
    """Write *annotations* as a ``.slp``, via COCO and the SLEAP environment.

    **Identity survives.** An object carrying a ``track_id`` arrives as a SLEAP
    ``Track``, which is what the ``multi_class_topdown`` and
    ``multi_class_bottomup`` heads classify against; a set carrying none produces
    a ``.slp`` with no tracks, and those two heads then have nothing to learn.
    Each track is named by its ``track_id``, so a model trained on the file
    reports those names.

    Args:
        annotations: What to write. Frames with no instances are written too --
            a labelled-empty frame is a negative example, and SLEAP can use it.
        out_path: Destination ``.slp``.
        images_dir: What the written COCO file's ``file_name`` values resolve
            against. Defaults to the set's own ``image_root``, which is right
            whenever the images have not moved since it was read.
        sleap_conda_env: Run in this conda env, overriding the environment
            variable.
        sleap_bin: A SLEAP console script whose directory names the install,
            overriding the environment variable.
        idle_timeout: Kill the subprocess after this long with no output.
        max_runtime: Optional absolute ceiling.

    Returns:
        The path written.

    Raises:
        ValueError: The set has no frames, or *images_dir* cannot be resolved.
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: The conversion exited non-zero.
        FileNotFoundError: The subprocess reported success but wrote nothing.
    """
    out_path = Path(out_path)
    if not annotations.frames:
        raise ValueError("cannot write a .slp from an annotation set with no frames")

    root = Path(images_dir) if images_dir is not None else annotations.image_root
    if root is None:
        raise ValueError(
            "images_dir is required when the annotation set carries no image_root; "
            "sleap-io resolves each file_name against a dataset root"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Beside the target rather than in a temp directory, so the move is a rename
    # within one filesystem and cannot half-succeed across a device boundary.
    staging = out_path.with_name(out_path.name + ".partial")
    coco_path = out_path.with_name(out_path.name + ".coco.json")
    _ = write_coco_keypoints(annotations, coco_path)

    invocation = tool_invocation(
        SLEAP_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin),
        executable="python",
    )
    cmd = [*invocation, "-c", _TO_SLP_SNIPPET, str(coco_path), str(root), str(staging)]
    logger.info("Running: %s", command_summary(cmd, SleapError.head))

    try:
        stdout, stderr, returncode = run_supervised(
            cmd,
            env=subprocess_env(),
            cancel_check=cancel_check,
            timeout=max_runtime,
            idle_timeout=idle_timeout,
            on_output=on_output,
        )
        if returncode != 0:
            raise SleapError(cmd, returncode, stdout, stderr)
        if not staging.exists():
            raise FileNotFoundError(
                f"sleap-io reported success but wrote no .slp at {staging}"
            )
        os.replace(staging, out_path)
    finally:
        coco_path.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)

    return out_path
