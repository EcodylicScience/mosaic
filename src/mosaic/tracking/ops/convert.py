"""Annotation-conversion tracking ops.

``convert-points``: turn existing CVAT point annotations (CVAT "for Images 1.1" XML) +
their images into a POLO point-detection training dataset (``{train,valid,test}/{images,
labels}`` + ``data.yaml``) under ``models/convert-points/<run_id>/``. This is the
"import/convert" step that replaces in-app annotation when labels already exist, and --
because it rides the ``Op`` contract -- it is reachable identically from the CLI
(``mosaic run --kind convert-points``) and the API (``POST /runs`` ``{"kind":
"convert-points"}``) with zero extra wiring.

The heavy-ish converter imports (numpy via the schema base) load lazily inside ``run()`` so
``import mosaic.tracking`` stays light, consistent with the train/infer/trex ops.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.tracking.ops._common import ensure_models_root, fingerprint_dataset

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Converted-dataset index ---------------------------------------------


@dataclass(frozen=True, slots=True)
class ConvertedDatasetIndexRow(RunIndexRowBase):
    """Typed row for a converted-dataset index CSV (``models/convert-points/index.csv``)."""

    kind: str
    source_format: str
    data_yaml: str  # dataset-root-relative path to the written data.yaml
    class_names: str  # comma-joined, index = class id
    n_train: int
    n_valid: int
    n_test: int
    status: str


def converted_dataset_index(path: Path) -> IndexCSV[ConvertedDatasetIndexRow]:
    return IndexCSV(path, ConvertedDatasetIndexRow, dedup_keys=["run_id"])


# --- Params --------------------------------------------------------------


class ConvertPointsParams(Params):
    """Parameters for the ``convert-points`` op (CVAT points -> POLO training dataset)."""

    source_format: Literal["cvat_points"] = "cvat_points"
    # dataset-resolvable inputs (relative to the dataset root, or absolute)
    cvat_xml: str  # CVAT "for Images 1.1" XML export
    images_dir: str  # directory of images whose names match the XML <image name>
    # class + POLO radius config
    class_names: list[str]
    radii: dict[str, float]  # class name -> detection radius in pixels
    class_attribute: str = "class"  # the <attribute name=...> holding the class
    # split
    split: tuple[float, float, float] = (0.8, 0.15, 0.05)
    split_by: str = "group"  # "group" keeps frames from the same video together
    seed: int = 42
    # execution knobs (excluded from the run_id -- behavior/throughput only)
    symlink_images: Annotated[bool, HASH_EXCLUDE] = True
    overwrite: Annotated[bool, HASH_EXCLUDE] = False


# --- Op ------------------------------------------------------------------


def _count_labels(split_dir: Path) -> int:
    labels = split_dir / "labels"
    return sum(1 for _ in labels.glob("*.txt")) if labels.exists() else 0


@register_op
class ConvertPointsOp(Op[ConvertPointsParams]):
    """Convert CVAT point annotations into a POLO point-detection dataset + ``data.yaml``."""

    kind = "convert-points"
    category = "convert"
    domain = "tracking"
    version = "0.1"
    Params = ConvertPointsParams

    def target(self, params: ConvertPointsParams) -> str:
        return "cvat-points-polo"

    def run(self, ds: Dataset, params: ConvertPointsParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.converters.cvat_points import (
            convert_cvat_points_polo,
        )
        from mosaic.tracking.pose_training.prep import make_polo_data_yaml

        ensure_models_root(ds)
        xml = Path(ds.resolve_path(params.cvat_xml))
        imgs = Path(ds.resolve_path(params.images_dir))

        run_id = "{}-{}".format(
            self.kind,
            hash_params(
                {
                    "params": params.identity_dump(),
                    "xml": fingerprint_dataset(xml),
                    "images": fingerprint_dataset(imgs),
                }
            ),
        )
        ctx.set_run_id(run_id)
        out = model_run_root(ds, self.kind, run_id)
        data_yaml = out / "data.yaml"

        # Content-addressed cache hit: an identical (params, xml, images) run already
        # produced this data.yaml. Re-running is a no-op unless overwrite is set.
        if data_yaml.exists() and not params.overwrite:
            return run_id

        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        schema = convert_cvat_points_polo(
            xml,
            imgs,
            out,
            radii=params.radii,
            class_attribute=params.class_attribute,
            class_names=params.class_names or None,
            split=params.split,
            symlink_images=params.symlink_images,
            seed=params.seed,
            split_by=params.split_by,
        )

        n_train = _count_labels(out / "train")
        n_valid = _count_labels(out / "valid")
        n_test = _count_labels(out / "test")
        if n_train == 0:
            raise ValueError(
                "convert-points produced no training labels. Check that images_dir "
                f"({imgs}) contains files whose names match the <image name> entries in "
                f"{xml}, and that class_attribute='{params.class_attribute}' is correct."
            )

        make_polo_data_yaml(out, schema.names, schema.radii)

        idx = converted_dataset_index(model_index_path(ds, self.kind))
        idx.ensure()
        idx.append(
            [
                ConvertedDatasetIndexRow(
                    run_id=run_id,
                    kind=self.kind,
                    source_format=params.source_format,
                    data_yaml=ds.relative_to_root(data_yaml),
                    class_names=",".join(schema.names),
                    n_train=n_train,
                    n_valid=n_valid,
                    n_test=n_test,
                    status="finished",
                    abs_path=Path(ds.relative_to_root(out)),
                )
            ]
        )
        idx.mark_finished(run_id)
        return run_id
