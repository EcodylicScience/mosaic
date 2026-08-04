"""Model-training tracking ops: pose, points (POLO), localizer.

Each op wraps the corresponding low-level trainer (kept in
``pose_training/``) under the Job Contract: content ``run_id``, tracked storage
under ``models/<kind>/<run_id>/``, per-epoch progress routed through
``ctx.progress``, cooperative between-epoch cancellation, retraining lineage,
and a ``TrainedModelIndexRow``. Heavy backends (ultralytics / torch / POLO) are
imported lazily inside ``run()`` so registration stays import-light.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
from typing import TYPE_CHECKING, Annotated

from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.tracking.model_refs import ModelShape, resolve_model
from mosaic.tracking.ops._common import ensure_models_root, fingerprint_dataset

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Trained-model index -------------------------------------------------


def train_run_id(
    kind: str, version: str, params: Params, data_fingerprint: str, base_run_id: str
) -> str:
    """Mint a training run identifier.

    A named function rather than a dict literal inside three near-identical
    ``run()`` bodies, so the payload shape is one thing to read, one thing to
    change, and something the golden corpus can call with fixed arguments and no
    filesystem.

    Args:
        kind: The op kind, which is also the directory the run lands in.
        version: The op's declared version -- a visible segment, not hashed.
        params: The op params; only ``identity_dump()`` enters the digest, so
            throughput knobs marked ``HASH_EXCLUDE`` do not bust the cache.
        data_fingerprint: Digest of the training data, from
            ``fingerprint_dataset``. Content, so retraining on changed
            annotations is a different model.
        base_run_id: What names the model this one fine-tunes from, or ``""``.
            The training run when there is one; the weights' content digest when
            the base was handed in as a bare path. Never the path itself -- a
            path is a location, so two fine-tunes from different weights sitting
            at one path used to mint one identifier whenever their params and
            data matched. The parameter keeps its name because the payload key
            ``"base"`` does, and renaming that would move every registered-lineage
            training identifier for no reason.
    """
    return op_run_id(
        kind,
        version,
        {
            "params": params.identity_dump(),
            "data": data_fingerprint,
            "base": base_run_id,
        },
    )


@dataclass(frozen=True, slots=True)
class TrainedModelIndexRow(RunIndexRowBase):
    """Typed row for a trained-model index CSV (``models/<kind>/index.csv``).

    ``best_model_path`` and ``metrics_path`` are stored dataset-root-relative so
    the index survives a move. A new path column here must also be added to
    ``_INDEX_PATH_COLUMNS["models"]`` in ``core/dataset.py``, or the two
    path-repair passes will not see it and it silently stops being portable.
    """

    kind: str
    base_model: str
    base_run_id: str  # lineage: the prior run_id this was retrained from ("" if none)
    best_model_path: str
    metrics_path: str
    n_epochs: int
    status: str
    # The base weights' content digest, recorded whether or not they had a run to
    # name them. Provenance, never identity: ``base_run_id`` already carries what
    # enters the hash, and this says which exact bytes were behind it. Defaulted
    # so every existing construction site keeps working, and not path-bearing, so
    # no portability rewrite list.
    base_digest: str = ""
    # What shape of artifact this run produced -- a ``ModelShape``. Empty means a
    # single weights file, which is what every row written before a model could
    # be a directory describes, so an old index keeps resolving unchanged.
    artifact_shape: str = ""
    # The artifact root, when it is not the weights file: a Lightning Pose model
    # directory, or the ordered directory a SLEAP run leaves behind. Empty falls
    # back to ``best_model_path``. Root-relative, and listed in the ``models``
    # entry of ``_INDEX_PATH_COLUMNS`` so it survives a dataset move.
    artifact_path: str = ""
    # Which head or architecture the artifact carries, read from its config.
    # Provenance, never identity -- the tracker rows already record this and the
    # trained-model row had nowhere to put it.
    model_type: str = ""


TRAINED_MODEL_INDEX_COLUMNS: list[str] = [
    field.name for field in fields(TrainedModelIndexRow)
]


def adopt_trained_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a trained-model index read off disk up to the current schema.

    ``trained_model_index`` had no ``adopt`` hook. Survivable while the schema
    was fixed; not survivable the moment it grows a column, because an absent
    one concatenated against a real row widens ``n_epochs`` and ``100`` reaches
    disk as ``100.0``. Every column built with an explicit ``object`` dtype, the
    same shape as ``tracks_index.adopt_legacy_columns``.
    """
    out = pd.DataFrame(index=df.index)
    for column in TRAINED_MODEL_INDEX_COLUMNS:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    return out


def trained_model_index(path: Path) -> IndexCSV[TrainedModelIndexRow]:
    return IndexCSV(
        path,
        TrainedModelIndexRow,
        dedup_keys=["run_id"],
        adopt=adopt_trained_model_columns,
    )


# --- Shared helpers ------------------------------------------------------


def finalize_training(
    ds: Dataset,
    kind: str,
    run_id: str,
    run_root: Path,
    p: Params,
    base_model: str,
    base_run_id: str,
    base_digest: str,
    best_model_path: Path,
    metrics_path: Path,
    n_epochs: int,
    artifact_shape: ModelShape = "file",
    artifact_path: Path | None = None,
    model_type: str = "",
) -> None:
    """Register a finished training run in ``models/<kind>/index.csv``.

    *artifact_path* names the artifact when it is not the weights file -- a model
    directory rather than one ``best.pt``. Left ``None`` for the single-file case,
    where ``best_model_path`` already says everything and storing it twice would
    be two things to keep agreeing.
    """
    idx = trained_model_index(model_index_path(ds, kind))
    idx.ensure()
    idx.append(
        [
            TrainedModelIndexRow(
                run_id=run_id,
                kind=kind,
                base_model=base_model,
                base_run_id=base_run_id,
                base_digest=base_digest,
                best_model_path=ds.relative_to_root(best_model_path),
                metrics_path=(
                    ds.relative_to_root(metrics_path) if metrics_path.exists() else ""
                ),
                n_epochs=int(n_epochs),
                status="finished",
                artifact_shape=artifact_shape,
                artifact_path=(
                    ds.relative_to_root(artifact_path)
                    if artifact_path is not None
                    else ""
                ),
                model_type=model_type,
                abs_path=Path(ds.relative_to_root(run_root)),
            )
        ]
    )
    idx.mark_finished(run_id)


# --- Params --------------------------------------------------------------


class PoseTrainParams(Params):
    data: str  # path to data.yaml
    model: str = "yolo11n-pose.pt"
    base_model: str = ""  # weights path OR a prior training run_id (retraining)
    epochs: int = 300
    imgsz: int = 640
    patience: int = 50
    resume: bool = False
    augmentation: str | None = None
    device: Annotated[str, HASH_EXCLUDE] = "0"
    batch: Annotated[int, HASH_EXCLUDE] = 16


class PointTrainParams(PoseTrainParams):
    # POLO nano localizer config. Resolves to ``cfg/models/26/polo26.yaml`` at scale ``n``
    # (Locate task) in the mooch443/POLO fork; matches the deployed ``polo26n`` detector.
    model: str = "polo26n.yaml"
    loc: float = 5.0
    loc_loss: str = "mse"
    dor: float = 0.8
    backend: Annotated[str, HASH_EXCLUDE] = "polo"


class LocalizerTrainParams(Params):
    dataset_dir: str
    base_model: str = ""  # weights path OR a prior training run_id (fine-tune)
    num_classes: int = 4
    initial_channels: int = 32
    freeze_encoder: bool = False
    epochs: int = 200
    lr: float = 1e-3
    early_stopping_patience: int = 20
    augment: bool = True
    seed: int = 42
    device: Annotated[str, HASH_EXCLUDE] = "0"
    batch_size: Annotated[int, HASH_EXCLUDE] = 128


# --- Ops -----------------------------------------------------------------


@register_op
class TrainPoseOp(Op[PoseTrainParams]):
    kind = "train-pose"
    category = "train"
    domain = "tracking"
    version = "0.1"
    Params = PoseTrainParams

    def run(self, ds: Dataset, params: PoseTrainParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.train import train_pose_model

        ensure_models_root(ds)
        data_yaml = Path(ds.resolve_path(params.data))
        model_arg = params.model
        base_run_id = ""
        base_digest = ""
        if params.base_model:
            base = resolve_model(ds, params.base_model, self.kind)
            # model_id, not run_id: a bare path has no run, and hashing "" there
            # let two fine-tunes from *different* weights collide whenever their
            # params and data matched.
            base_run_id = base.model_id
            base_digest = base.digest
            model_arg = str(base.path)

        run_id = train_run_id(
            self.kind, self.version, params, fingerprint_dataset(data_yaml), base_run_id
        )
        ctx.set_run_id(run_id)
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        train_pose_model(
            data_yaml,
            model=model_arg,
            epochs=params.epochs,
            imgsz=params.imgsz,
            batch=params.batch,
            device=params.device,
            patience=params.patience,
            resume=params.resume,
            augmentation=params.augmentation,
            project=str(run_root),
            name="train",
            callback=ctx.progress,
            cancel_check=ctx.cancel_token.is_cancelled,
        )
        ctx.check_cancel()  # raise Cancelled if a between-epoch cancel fired
        finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            base_digest,
            run_root / "train" / "weights" / "best.pt",
            run_root / "train" / "results.csv",
            params.epochs,
        )
        return run_id


@register_op
class TrainPointsOp(Op[PointTrainParams]):
    kind = "train-points"
    category = "train"
    domain = "tracking"
    version = "0.1"
    Params = PointTrainParams

    def run(self, ds: Dataset, params: PointTrainParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.train import train_point_model

        ensure_models_root(ds)
        data_yaml = Path(ds.resolve_path(params.data))
        model_arg = params.model
        base_run_id = ""
        base_digest = ""
        if params.base_model:
            base = resolve_model(ds, params.base_model, self.kind)
            # model_id, not run_id: a bare path has no run, and hashing "" there
            # let two fine-tunes from *different* weights collide whenever their
            # params and data matched.
            base_run_id = base.model_id
            base_digest = base.digest
            model_arg = str(base.path)

        run_id = train_run_id(
            self.kind, self.version, params, fingerprint_dataset(data_yaml), base_run_id
        )
        ctx.set_run_id(run_id)
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        train_point_model(
            data_yaml,
            model=model_arg,
            epochs=params.epochs,
            imgsz=params.imgsz,
            batch=params.batch,
            device=params.device,
            patience=params.patience,
            loc=params.loc,
            loc_loss=params.loc_loss,
            dor=params.dor,
            resume=params.resume,
            augmentation=params.augmentation,
            backend=params.backend,
            project=str(run_root),
            name="train",
            callback=ctx.progress,
            cancel_check=ctx.cancel_token.is_cancelled,
        )
        ctx.check_cancel()
        finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            base_digest,
            run_root / "train" / "weights" / "best.pt",
            run_root / "train" / "results.csv",
            params.epochs,
        )
        return run_id


@register_op
class TrainLocalizerOp(Op[LocalizerTrainParams]):
    kind = "train-localizer"
    category = "train"
    domain = "tracking"
    version = "0.1"
    Params = LocalizerTrainParams

    def run(self, ds: Dataset, params: LocalizerTrainParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.localizer_train import train_localizer

        ensure_models_root(ds)
        dataset_dir = Path(ds.resolve_path(params.dataset_dir))
        weights = None
        base_run_id = ""
        base_digest = ""
        if params.base_model:
            base = resolve_model(ds, params.base_model, self.kind)
            base_run_id = base.model_id
            base_digest = base.digest
            weights = str(base.path)

        run_id = train_run_id(
            self.kind,
            self.version,
            params,
            fingerprint_dataset(dataset_dir),
            base_run_id,
        )
        ctx.set_run_id(run_id)
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        result = train_localizer(
            dataset_dir,
            num_classes=params.num_classes,
            initial_channels=params.initial_channels,
            weights=weights,
            freeze_encoder=params.freeze_encoder,
            epochs=params.epochs,
            batch_size=params.batch_size,
            lr=params.lr,
            early_stopping_patience=params.early_stopping_patience,
            device=params.device,
            augment=params.augment,
            seed=params.seed,
            project=str(run_root),
            name="train",
            callback=ctx.progress,
            cancel_check=ctx.cancel_token.is_cancelled,
        )
        ctx.check_cancel()
        finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            base_digest,
            Path(result.best_model_path),
            run_root / "train" / "results.csv",
            params.epochs,
        )
        return run_id
