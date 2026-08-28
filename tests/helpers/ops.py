"""The smallest params dict that validates for each registered op."""

from __future__ import annotations

import copy

from mosaic.core.pipeline.ops import OPS
from mosaic.tracking import register_ops

register_ops()

_MINIMAL: dict[str, dict[str, object]] = {
    "convert-points": {
        "cvat_xml": "labels_raw/cvat/annotations.xml",
        "images_dir": "labels_raw/cvat/images",
        "class_names": ["individual"],
        "radii": {"individual": 4.0},
    },
    "export-store": {"entry": ["A", "one"]},
    "extract-frames": {"n_frames": 1},
    "infer-localizer": {"model": "models/infer-localizer/run/best.pt"},
    "infer-points": {"model": "models/infer-points/run/best.pt"},
    "infer-pose": {"model": "models/infer-pose/run/best.pt"},
    "litpose": {"model_path": "models/litpose/run"},
    "resample-tracks": {"target_fps": 30.0},
    "sleap": {"model_paths": ["models/sleap/run"]},
    "train-litpose": {"project": "models/train-litpose/project"},
    "train-localizer": {"dataset_dir": "datasets/localizer"},
    "train-points": {"data": "datasets/points/data.yaml"},
    "train-pose": {"data": "datasets/pose/data.yaml"},
    "train-sleap": {"labels": "labels_raw/sleap/labels.slp"},
    "transcode": {"entries": [["A", "one"]]},
    "trex": {},
    "ultralytics": {"model_path": "models/ultralytics/run/model.pt"},
}


def minimal_op_params(kind: str) -> dict[str, object]:
    """The smallest params dict that validates for op *kind*.

    A plain function rather than a fixture, since it is parametrized by kind.
    Every registered op is covered. A required value naming a file is shaped
    like a path under a dataset root rather than an absolute path tied to one
    machine. Validating a ``Params`` model checks the type of a path. It does
    not check that the path exists.

    Returns a copy. The values in ``_MINIMAL`` are shared across every call in
    the process, and an op's params commonly hold a mutable ``entries``
    or ``scope`` field a caller sets after construction.
    """
    if kind not in OPS:
        message = f"{kind!r} is not a registered op"
        raise KeyError(message)
    if kind not in _MINIMAL:
        message = f"minimal_op_params does not cover op kind {kind!r}"
        raise KeyError(message)
    return copy.deepcopy(_MINIMAL[kind])
