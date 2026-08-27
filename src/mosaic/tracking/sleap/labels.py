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

from mosaic.core.annotations.model import AnnotationSet
from mosaic.core.annotations.writers import write_coco_keypoints
from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.tracking.common.toolenv import subprocess_env, tool_invocation
from mosaic.tracking.sleap.run import SLEAP_PYTHON_ENV, SleapError

logger = logging.getLogger(__name__)

__all__ = ["write_slp"]

# Run by the SLEAP interpreter. Its only inputs are argv -- the COCO json, the
# directory its file_name values are relative to, and where to write -- so no
# path is ever interpolated into this source.
_TO_SLP_SNIPPET: str = """
import sys
import sleap_io

_json, _root, _out = sys.argv[1], sys.argv[2], sys.argv[3]
_labels = sleap_io.load_coco(_json, dataset_root=_root)
sleap_io.save_slp(_labels, _out)
print(f"wrote {len(_labels.labeled_frames)} labeled frames to {_out}")
"""


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
        SLEAP_PYTHON_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin),
        executable="python",
    )
    cmd = [*invocation, "-c", _TO_SLP_SNIPPET, str(coco_path), str(root), str(staging)]
    logger.info("Running: %s", " ".join(cmd[:4]) + " ...")

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
