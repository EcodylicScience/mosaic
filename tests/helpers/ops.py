"""The smallest params dict that validates for each registered op."""

from __future__ import annotations

import copy
from collections.abc import Set as AbstractSet

from pydantic import BaseModel, create_model

from mosaic.core.pipeline.ops import OPS
from mosaic.tracking import register_ops

register_ops()

_MINIMAL: dict[str, dict[str, object]] = {
    "assign-arenas": {},
    "assign-identities": {
        "source_tracks_run_id": "sleap.1.6-abcdef0123",
        "references": [
            {"identity": "a", "entry": {"sequence": "s1"}, "id": 0},
            {"identity": "b", "entry": {"sequence": "s1"}, "id": 1},
        ],
    },
    "convert-labels": {"source_format": "calms21_npy"},
    "convert-points": {
        "cvat_xml": "labels_raw/cvat/annotations.xml",
        "images_dir": "labels_raw/cvat/images",
        "class_names": ["individual"],
        "radii": {"individual": 4.0},
    },
    "convert-tracks": {"source_format": "calms21_npy"},
    "export-joined": {},
    "export-store": {},
    "extract-frames": {"n_frames": 1},
    "infer-localizer": {"model": "models/infer-localizer/run/best.pt"},
    "infer-points": {"model": "models/infer-points/run/best.pt"},
    "infer-pose": {"model": "models/infer-pose/run/best.pt"},
    "litpose": {"model_path": "models/litpose/run"},
    "predict-frames": {
        "model": ["train-pose.0.2-abcdef0123"],
        "frames_run_id": "kmeans-0123456789",
    },
    "prepare-embedding-data": {
        "sleap_run_id": "sleap.1.6-abcdef0123",
        "train_entries": [{"sequence": "s1"}],
        "val_entries": [{"sequence": "s2"}],
    },
    "prepare-training-data": {"sets": [{"set_key": "set-a"}]},
    "preprocess": {
        "steps": [{"step": "crop", "x": 0, "y": 0, "width": 320, "height": 240}]
    },
    "resample-tracks": {"target_fps": 30.0},
    "score-predictions": {
        "prediction_runs": ["predict-frames.0.1-abcdef0123"],
        "labels": [{"set_key": "set-a"}],
    },
    "select-frames": {"source_tracks_run_id": "trex.0.1-abcdef0123"},
    "sleap": {"model_paths": ["models/sleap/run"]},
    "train-litpose": {"project": "models/train-litpose/project"},
    "train-localizer": {"dataset_dir": "datasets/localizer"},
    "train-points": {"data": "datasets/points/data.yaml"},
    "train-pose": {"data": "datasets/pose/data.yaml"},
    "train-sleap": {"labels": "labels_raw/sleap/labels.slp"},
    "transcode": {},
    "trex": {},
    "ultralytics": {"model_path": "models/ultralytics/run/model.pt"},
}

CONSUMER_OWNED: dict[str, frozenset[str]] = {
    "prepare-training-data": frozenset(
        {
            "sets",
            "target",
            "pose",
            "class_by",
            "track_by",
            "sequence_groups",
            "bbox",
            "symlink_images",
        }
    ),
    "train-litpose": frozenset(
        {"project", "device", "num_workers", "base_model", "base_config"}
    ),
    "train-points": frozenset({"data", "device", "workers", "resume", "base_model"}),
    "train-pose": frozenset({"data", "device", "workers", "resume", "base_model"}),
    "train-sleap": frozenset({"labels", "device", "num_workers", "base_model"}),
}
"""The fields a consumer of the toolkit owns for an op kind.

Each value is the union of ``REQUEST_OWNED`` and ``SERVER_OWNED`` names for the
kind in mosaic-api's ``src/mosaic_api/job_params.py``. The table is copied
here, since mosaic does not depend on mosaic-api. The model that the consumer
exposes lacks these fields and the consumer completes the payload before the
worker validates it with the full model.
"""


def narrowed_params[M: BaseModel](model: type[M], without: AbstractSet[str]) -> type[M]:
    """Return a subclass of *model* without the fields named in *without*.

    Equivalent to mosaic-api's ``narrow_owned_fields``: the subclass is made by
    ``create_model`` instead of a class statement, each field is popped from its
    ``model_fields`` with a name the model lacks passed over, and it is rebuilt
    with ``model_rebuild(force=True)``. *model* is unchanged.
    """
    narrowed = create_model(f"Narrowed{model.__name__}", __base__=model)
    for field_name in without:
        _ = narrowed.model_fields.pop(field_name, None)
    _ = narrowed.model_rebuild(force=True)
    return narrowed


def minimal_op_params(kind: str) -> dict[str, object]:
    """The smallest params dict that validates for op *kind*.

    A plain function rather than a fixture, since it is parametrized by kind.
    Every registered op is covered. A required value naming a file is shaped
    like a path under a dataset root rather than an absolute path tied to one
    machine. Validating a ``Params`` model checks the type of a path. It does
    not check that the path exists.

    Returns a copy. The values in ``_MINIMAL`` are shared across every call in
    the process, and a caller commonly mutates what it gets back.
    """
    if kind not in OPS:
        message = f"{kind!r} is not a registered op"
        raise KeyError(message)
    if kind not in _MINIMAL:
        message = f"minimal_op_params does not cover op kind {kind!r}"
        raise KeyError(message)
    return copy.deepcopy(_MINIMAL[kind])
