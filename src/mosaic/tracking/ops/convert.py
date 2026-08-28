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

from pydantic import Field

from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)
from mosaic.tracking.ops._common import (
    claim_run_root,
    ensure_models_root,
    fingerprint_dataset,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope


def convert_points_run_id(
    kind: str,
    version: str,
    params: Params,
    xml_fingerprint: str,
    images_fingerprint: str,
) -> str:
    """Mint an annotation-conversion run identifier.

    Both sources are fingerprinted by content: re-exporting the same CVAT task
    after correcting labels must produce a different dataset, and a changed
    image set must too.
    """
    return op_run_id(
        kind,
        version,
        {
            "params": params.identity_dump(),
            "xml": xml_fingerprint,
            "images": images_fingerprint,
        },
    )


# --- Converted-dataset index ---------------------------------------------


@dataclass(frozen=True, slots=True)
class ConvertedDatasetIndexRow(RunIndexRowBase):
    """Typed row for a converted-dataset index CSV (``models/convert-points/index.csv``).

    ``data_yaml`` is stored dataset-root-relative so the index survives a move. A
    new path column here must also be added to ``_INDEX_PATH_COLUMNS["models"]``
    in ``core/dataset.py``, or the two path-repair passes will not see it and it
    silently stops being portable.
    """

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


_SOURCE_FORMAT_DESCRIPTION = "Which annotation format this run converts."

_CVAT_XML_DESCRIPTION = "The CVAT 'for Images 1.1' XML export to convert."

_IMAGES_DIR_DESCRIPTION = (
    "Directory of images whose filenames match the XML's image name attributes."
)

_CLASS_NAMES_DESCRIPTION = (
    "Ordered class names, index used as the class id. An empty list "
    "auto-detects names from the XML in order of first appearance."
)

_RADII_DESCRIPTION = (
    "Detection radius for each class name. Every class the conversion "
    "resolves needs an entry, whether class_names named it or "
    "auto-detection found it."
)

_CLASS_ATTRIBUTE_DESCRIPTION = (
    "Name of the XML attribute that names each point's class. Empty means "
    "single-class, with every point assigned class 0."
)

_SPLIT_DESCRIPTION = "Train, validation and test fractions of the annotated images."

_SPLIT_BY_DESCRIPTION = (
    "How images are grouped before the split is drawn. group keeps frames "
    "from the same video together in one split."
)

_SEED_DESCRIPTION = "Random seed for the train, validation and test split assignment."

_SYMLINK_IMAGES_DESCRIPTION = (
    "Symlink source images into the dataset instead of copying them."
)


class ConvertPointsParams(Params):
    """Parameters for the ``convert-points`` op (CVAT points -> POLO training dataset).

    This op names an XML export and an images directory, never a media entry,
    so it declares no scope and its ``scope_takes`` is ``"none"``. Whether an
    existing ``data.yaml`` is rebuilt is decided by the ``overwrite`` argument
    :meth:`ConvertPointsOp.run` receives, not by a field here.
    """

    source_format: Annotated[
        Literal["cvat_points"], Declared(_SOURCE_FORMAT_DESCRIPTION)
    ] = "cvat_points"
    # dataset-resolvable inputs (relative to the dataset root, or absolute)
    cvat_xml: Annotated[str, Declared(_CVAT_XML_DESCRIPTION)]
    images_dir: Annotated[str, Declared(_IMAGES_DIR_DESCRIPTION)]
    # class + POLO radius config
    class_names: Annotated[list[str], Declared(_CLASS_NAMES_DESCRIPTION)]
    radii: Annotated[dict[str, float], Declared(_RADII_DESCRIPTION, unit="px")]
    class_attribute: Annotated[str, Declared(_CLASS_ATTRIBUTE_DESCRIPTION)] = "class"
    # split
    split: Annotated[tuple[float, float, float], Declared(_SPLIT_DESCRIPTION)] = (
        0.8,
        0.15,
        0.05,
    )
    split_by: Annotated[
        str,
        Field(examples=["group", "image"]),
        Declared(_SPLIT_BY_DESCRIPTION),
    ] = "group"
    seed: Annotated[int, Declared(_SEED_DESCRIPTION)] = 42
    # execution knobs (excluded from the run_id -- behavior/throughput only)
    symlink_images: Annotated[
        bool, HASH_EXCLUDE, Declared(_SYMLINK_IMAGES_DESCRIPTION)
    ] = True


# --- Op ------------------------------------------------------------------


def _count_labels(split_dir: Path) -> int:
    labels = split_dir / "labels"
    return sum(1 for _ in labels.glob("*.txt")) if labels.exists() else 0


_CONVERT_IDLE_SECONDS = 600.0


@register_op
class ConvertPointsOp(Op[ConvertPointsParams]):
    """Convert CVAT point annotations into a POLO point-detection dataset + ``data.yaml``."""

    kind = "convert-points"
    category = "convert"
    domain = "tracking"
    version = "0.1"
    scope_takes = "none"
    scope_dependent = False
    Params = ConvertPointsParams

    def target(self, params: ConvertPointsParams, scope: ResolvedScope) -> str:
        return "cvat-points-polo"

    def plan_identity(
        self,
        ds: Dataset,
        params: ConvertPointsParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this conversion will be called.

        Both sources enter by content, because re-exporting the same CVAT task
        after correcting labels must produce a different dataset and a changed
        image set must too. Content has to be read, so a conversion whose inputs
        are an earlier step's output is not nameable until that step has run.
        """
        xml = Path(ds.resolve_path(params.cvat_xml))
        images = Path(ds.resolve_path(params.images_dir))
        for missing in (path for path in (xml, images) if not path.exists()):
            raise IdentityDeferred(
                self.kind,
                f"its source ({missing}) is not on disk yet, and this run's "
                f"identity covers the content of what it converts",
            )
        return OpIdentity(
            run_id=convert_points_run_id(
                self.kind,
                self.version,
                params,
                fingerprint_dataset(xml),
                fingerprint_dataset(images),
            )
        )

    def run(
        self,
        ds: Dataset,
        params: ConvertPointsParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        from mosaic.tracking.pose_training.converters.cvat_points import (
            convert_cvat_points_polo,
        )
        from mosaic.tracking.pose_training.prep import make_polo_data_yaml

        ensure_models_root(ds)
        xml = Path(ds.resolve_path(params.cvat_xml))
        imgs = Path(ds.resolve_path(params.images_dir))

        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        out = model_run_root(ds, self.kind, run_id)
        data_yaml = out / "data.yaml"

        # Content-addressed cache hit: an identical (params, xml, images) run already
        # produced this data.yaml. Re-running is a no-op unless overwrite is set.
        if data_yaml.exists() and not overwrite:
            ctx.cache_hit()
            return run_id

        # Claimed before the rmtree, not after: two executions of this identifier
        # would otherwise have one delete the other's output mid-write. The body is
        # deterministic, so the bytes agree -- the destruction is the hazard.
        out.mkdir(parents=True, exist_ok=True)
        claim_run_root(ds, ctx, out, self.kind, _CONVERT_IDLE_SECONDS)
        for child in out.iterdir():
            if child.name != ".mosaic-inflight.json":
                shutil.rmtree(child) if child.is_dir() else child.unlink()

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
