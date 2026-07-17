"""Model-training tracking ops: pose, points (POLO), localizer.

Each op wraps the corresponding low-level trainer (kept in
``pose_training/``) under the Job Contract: content ``run_id``, tracked storage
under ``models/<kind>/<run_id>/``, per-epoch progress routed through
``ctx.progress``, cooperative between-epoch cancellation, retraining lineage,
and a ``TrainedModelIndexRow``. Heavy backends (ultralytics / torch / POLO) are
imported lazily inside ``run()`` so registration stays import-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.tracking.registry import TrackingOp, register_tracking_op, resolve_model

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# --- Trained-model index -------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainedModelIndexRow(RunIndexRowBase):
    """Typed row for a trained-model index CSV (``models/<kind>/index.csv``)."""

    kind: str
    base_model: str
    base_run_id: str  # lineage: the prior run_id this was retrained from ("" if none)
    best_model_path: str
    metrics_path: str
    n_epochs: int
    status: str


def trained_model_index(path: Path) -> IndexCSV[TrainedModelIndexRow]:
    return IndexCSV(path, TrainedModelIndexRow, dedup_keys=["run_id"])


# --- Shared helpers ------------------------------------------------------


def _ensure_models_root(ds: Dataset) -> None:
    if not ds.has_root("models"):
        ds.set_root("models", "models")


def _fingerprint_dataset(path: Path) -> str:
    """Cheap, copy-stable digest of a training dataset (file text + size listing).

    Uses relative paths + file sizes (not mtimes) so a copied/moved dataset with
    identical contents fingerprints identically -- keeping training run_ids
    deterministic across machines.
    """
    path = Path(path)
    parts: dict[str, object] = {}
    if path.is_file():
        parts["file"] = path.name
        try:
            parts["text"] = path.read_text(errors="ignore")
        except Exception:
            parts["text"] = ""
        base = path.parent
    else:
        base = path
    listing: list[str] = []
    if base.exists():
        for f in sorted(base.rglob("*")):
            if f.is_file():
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                listing.append(f"{f.relative_to(base).as_posix()}:{size}")
    parts["listing"] = listing
    return hash_params(parts)


def _finalize_training(
    ds: Dataset,
    kind: str,
    run_id: str,
    run_root: Path,
    p: Params,
    base_model: str,
    base_run_id: str,
    best_model_path: Path,
    metrics_path: Path,
    n_epochs: int,
) -> None:
    idx = trained_model_index(model_index_path(ds, kind))
    idx.ensure()
    idx.append(
        [
            TrainedModelIndexRow(
                run_id=run_id,
                kind=kind,
                base_model=base_model,
                base_run_id=base_run_id,
                best_model_path=ds.relative_to_root(best_model_path),
                metrics_path=(
                    ds.relative_to_root(metrics_path) if metrics_path.exists() else ""
                ),
                n_epochs=int(n_epochs),
                status="finished",
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
    model: str = "polov8n.yaml"
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


@register_tracking_op
class TrainPoseOp(TrackingOp[PoseTrainParams]):
    kind = "train-pose"
    category = "train"
    version = "0.1"
    Params = PoseTrainParams

    def run(self, ds: Dataset, params: PoseTrainParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.train import train_pose_model

        _ensure_models_root(ds)
        data_yaml = Path(ds.resolve_path(params.data))
        model_arg = params.model
        base_run_id = ""
        if params.base_model:
            base_pt, base_run_id = resolve_model(ds, params.base_model, self.kind)
            model_arg = str(base_pt)

        run_id = "{}-{}".format(
            self.kind,
            hash_params(
                {
                    "params": params.identity_dump(),
                    "data": _fingerprint_dataset(data_yaml),
                    "base": base_run_id,
                }
            ),
        )
        ctx.set_run_id(run_id)
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

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
        _finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            run_root / "train" / "weights" / "best.pt",
            run_root / "train" / "results.csv",
            params.epochs,
        )
        return run_id


@register_tracking_op
class TrainPointsOp(TrackingOp[PointTrainParams]):
    kind = "train-points"
    category = "train"
    version = "0.1"
    Params = PointTrainParams

    def run(self, ds: Dataset, params: PointTrainParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.train import train_point_model

        _ensure_models_root(ds)
        data_yaml = Path(ds.resolve_path(params.data))
        model_arg = params.model
        base_run_id = ""
        if params.base_model:
            base_pt, base_run_id = resolve_model(ds, params.base_model, self.kind)
            model_arg = str(base_pt)

        run_id = "{}-{}".format(
            self.kind,
            hash_params(
                {
                    "params": params.identity_dump(),
                    "data": _fingerprint_dataset(data_yaml),
                    "base": base_run_id,
                }
            ),
        )
        ctx.set_run_id(run_id)
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

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
        _finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            run_root / "train" / "weights" / "best.pt",
            run_root / "train" / "results.csv",
            params.epochs,
        )
        return run_id


@register_tracking_op
class TrainLocalizerOp(TrackingOp[LocalizerTrainParams]):
    kind = "train-localizer"
    category = "train"
    version = "0.1"
    Params = LocalizerTrainParams

    def run(self, ds: Dataset, params: LocalizerTrainParams, ctx: JobContext) -> str:
        from mosaic.tracking.pose_training.localizer_train import train_localizer

        _ensure_models_root(ds)
        dataset_dir = Path(ds.resolve_path(params.dataset_dir))
        weights = None
        base_run_id = ""
        if params.base_model:
            base_pt, base_run_id = resolve_model(ds, params.base_model, self.kind)
            weights = str(base_pt)

        run_id = "{}-{}".format(
            self.kind,
            hash_params(
                {
                    "params": params.identity_dump(),
                    "data": _fingerprint_dataset(dataset_dir),
                    "base": base_run_id,
                }
            ),
        )
        ctx.set_run_id(run_id)
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)

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
        _finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            Path(result.best_model_path),
            run_root / "train" / "results.csv",
            params.epochs,
        )
        return run_id
