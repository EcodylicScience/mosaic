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
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import model_validator

from mosaic.core.helpers import text_cell
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.inventory._read import IndexReader
from mosaic.core.pipeline.inventory.contributors import register_inventory_contributor
from mosaic.core.pipeline.inventory.model import ArtifactRecord, InventoryScope
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.tracking.common.mint import planned_model_id
from mosaic.tracking.model_refs import ModelShape, resolve_model
from mosaic.tracking.ops._common import (
    claim_run_root,
    ensure_models_root,
    fingerprint_dataset,
    fingerprint_yolo_dataset,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


_TRAIN_IDLE_SECONDS = 1800.0

OP_SUPPLIED_TRAIN_ARGS: Final = frozenset(
    {"project", "name", "callback", "cancel_check", "task"}
)
"""Trainer arguments the op supplies that are not params fields.

``project`` / ``name`` address the claimed run root, ``callback`` and
``cancel_check`` are the Job Contract's progress and cancellation hooks, and
``task`` selects the ultralytics task. Together with the params fields
themselves these are what ``train_overrides`` may not set.
"""

# --- Trained-model index -------------------------------------------------


def train_run_id(
    kind: str,
    version: str,
    params: Params,
    data_fingerprint: str,
    base_run_id: str,
    extra: Mapping[str, object] | None = None,
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
        extra: Further payload terms for an op with an input the three above do
            not describe -- Lightning Pose's configuration file is one, since it
            carries training settings mosaic has no field for. Merged in only
            when given, so an op that passes nothing keeps the identifier it
            already mints. Pass **content**, never a path: a path is a location,
            for the reason spelled out for *base_run_id*.
    """
    payload: dict[str, object] = {
        "params": params.identity_dump(),
        "data": data_fingerprint,
        "base": base_run_id,
    }
    if extra:
        payload.update(extra)
    return op_run_id(kind, version, payload)


def planned_train_identity(
    ds: Dataset,
    *,
    kind: str,
    version: str,
    params: Params,
    data_path: Path,
    fingerprint: Callable[[Path], str],
    base_model: str,
    extra: Mapping[str, object] | None = None,
    require_data: bool = True,
) -> OpIdentity:
    """What a training run with these params will be called, without training.

    Every training op mints its identifier the same way -- params, a content
    fingerprint of the data, and what it fine-tunes from -- so this is that,
    called by both ``plan_identity`` and ``run``.

    *require_data* is what separates planning from execution. A planner asks
    whether this run is nameable and must be told when it is not; an execution
    already has its data, and the specific refusal for a missing file belongs to
    the tool that reads it -- which says which file and why, where this could only
    say that something is absent. So ``run`` passes ``False`` and computes exactly
    the identifier it always did, and the two agree whenever the answer matters:
    if the data is absent the run fails either way.

    **The fingerprint is why a training step can be unplannable**, and it is the
    right trade rather than a gap. Two runs over changed annotations must be two
    models, so the data enters the identity by content; content has to be read;
    and when the data directory is itself produced by an earlier step in the same
    graph there is nothing there yet to read. Saying so is honest, and it costs
    nothing at execution: the step resolves its own identity at its own start,
    where the directory does exist.
    """
    if require_data and not data_path.exists():
        raise IdentityDeferred(
            kind,
            f"its training data ({data_path}) is not on disk yet, and a model's "
            f"identity covers the content of what it was trained on",
        )
    base_run_id = ""
    if base_model:
        base_run_id = planned_model_id(ds, kind, [base_model], kind)
    run_id = train_run_id(
        kind, version, params, fingerprint(data_path), base_run_id, extra=extra
    )
    # A training run and the model it produces are one artifact under one name.
    return OpIdentity(run_id=run_id, model_run_id=run_id)


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


def training_is_complete(ds: Dataset, kind: str, run_id: str) -> bool:
    """Has this exact training run already finished and left its artifact?

    The completion evidence is the ``models/<kind>/index.csv`` row, because
    ``finalize_training`` writes it only after the trainer returns. Deliberately
    **not** the artifact alone: Ultralytics writes ``best.pt`` progressively, so a
    run killed mid-training leaves a plausible-looking weights file, and adopting
    that would hand a caller a half-trained model under a finished run's id. The
    row is the only thing on disk that means "the trainer returned".

    Both halves are required. The row proves the trainer finished; the artifact
    check catches the row outliving what it points at -- a swept run root, a
    hand-deleted directory -- in which case the honest answer is to train again.

    Absent index, absent row, unreadable index: not complete. A false negative
    costs a recompute, a false positive ships the wrong model.
    """
    index_path = model_index_path(ds, kind)
    if not index_path.exists():
        return False
    try:
        rows = trained_model_index(index_path).read(run_id=run_id)
    except (OSError, ValueError, KeyError):
        return False
    if rows.empty:
        return False
    if text_cell(rows.iloc[-1].get("status", "")) != "finished":
        return False
    recorded = text_cell(rows.iloc[-1].get("artifact_path", "")) or text_cell(
        rows.iloc[-1].get("best_model_path", "")
    )
    if not recorded:
        return False
    return ds.resolve_path(recorded).exists()


# --- Params --------------------------------------------------------------


class PoseTrainParams(Params):
    """Parameters for the ``train-pose`` op.

    The typed fields are the decisions mosaic has an opinion about. Everything
    else ultralytics exposes -- the learning-rate schedule, the loss weights, the
    long tail of ``yolo.train`` knobs -- is reachable through *train_overrides*,
    which reaches identity for the same reason the typed fields do: a model
    trained with a different learning rate is a different model whether or not
    there is a field for it here.
    """

    data: str  # path to data.yaml
    model: str = "yolo11n-pose.pt"
    base_model: str = ""  # weights path OR a prior training run_id (retraining)
    epochs: int = 300
    imgsz: int = 640
    patience: int = 50
    resume: bool = False
    augmentation: str | dict[str, JsonValue] | None = None
    """A preset name, or a dict -- with a ``preset`` key to start from one and
    override, or without to replace the augmentation set outright.

    ``resolve_augmentation`` has always accepted both; the op used to narrow it
    to ``str``, which put the dict forms out of reach of the CLI and the API.
    """

    train_overrides: dict[str, JsonValue] | None = None
    """Extra keyword arguments forwarded verbatim to ``yolo.train``.

    Hashed, because a different learning rate is a different model. Keys that
    would collide with a typed field or with an argument the op supplies are
    refused -- see :data:`OP_SUPPLIED_TRAIN_ARGS`.
    """

    device: Annotated[str, HASH_EXCLUDE] = "0"
    batch: Annotated[int, HASH_EXCLUDE] = 16
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    """Train again even if this exact run already finished.

    ``HASH_EXCLUDE`` because it is a throughput knob, not a property of the model:
    flipping it must not mint a second identity for the same weights.
    """

    @model_validator(mode="after")
    def _train_overrides_do_not_shadow(self) -> Self:
        """Refuse an override the op already supplies, at submit time.

        Two different failures, one guard. A key naming a *parameter* of the
        trainer (``epochs``, ``imgsz``, ...) arrives as a duplicate keyword and
        raises :exc:`TypeError` from Python itself -- on a GPU node, after the
        job was accepted and scheduled. A key naming something the trainer only
        builds internally (``data``, ``task``) does not raise at all: it lands
        via ``train_kwargs.update(extra_args)`` and silently retargets the run,
        so the identifier would describe data the run never read.

        Validating here rather than in ``run()`` is what lets mosaic-api answer
        a bad submission with a 422 the caller can act on.
        """
        overrides = self.train_overrides
        if not overrides:
            return self
        reserved = set(type(self).model_fields) | OP_SUPPLIED_TRAIN_ARGS
        shadowed = sorted(set(overrides) & reserved)
        if shadowed:
            names = ", ".join(shadowed)
            msg = (
                f"train_overrides may not set {names}: "
                "the op supplies these itself. Use the typed field instead."
            )
            raise ValueError(msg)
        return self


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
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    """Train again even if this exact run already finished.

    ``HASH_EXCLUDE`` because it is a throughput knob, not a property of the model:
    flipping it must not mint a second identity for the same weights.
    """


# --- Ops -----------------------------------------------------------------


@register_op
class TrainPoseOp(Op[PoseTrainParams]):
    """Train a YOLO pose model, registering the directory it produces."""

    kind = "train-pose"
    category = "train"
    domain = "tracking"
    # 0.2 carries ``train_overrides`` in the identity payload and fingerprints the
    # data.yaml by what it declares. Both move the digest, so the visible segment
    # moves with them: ``train-pose.0.1-*`` runs stay readable and keep their index
    # rows rather than sitting under a name that now means something else.
    version = "0.2"
    Params = PoseTrainParams

    def plan_identity(
        self, ds: Dataset, params: PoseTrainParams, *, require_data: bool = True
    ) -> OpIdentity:
        """What this run, and the model it produces, will be called.

        *require_data* separates planning from execution. A planner asks whether
        this run is nameable at all and must be told when it is not; an execution
        already has its data, and the refusal for a missing file belongs to the
        tool that reads it, which can say which file and why. The two agree
        whenever the answer matters: if the data is absent the run fails either
        way.
        """
        return planned_train_identity(
            ds,
            kind=self.kind,
            version=self.version,
            params=params,
            data_path=Path(ds.resolve_path(params.data)),
            fingerprint=fingerprint_yolo_dataset,
            base_model=params.base_model,
            require_data=require_data,
        )

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

        # Through plan_identity, so this run is named in exactly one place. A
        # second copy here would be the one that drifts when the payload changes,
        # and it is the one a planner does not read.
        run_id = self.plan_identity(ds, params, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not params.overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            return run_id
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        claim_run_root(ds, ctx, run_root, self.kind, _TRAIN_IDLE_SECONDS)
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
            **(params.train_overrides or {}),
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
    """Train a POLO point-detection model, registering the directory it produces."""

    kind = "train-points"
    category = "train"
    domain = "tracking"
    version = "0.2"  # see TrainPoseOp.version
    Params = PointTrainParams

    def plan_identity(
        self, ds: Dataset, params: PointTrainParams, *, require_data: bool = True
    ) -> OpIdentity:
        """What this run, and the model it produces, will be called.

        *require_data* separates planning from execution. A planner asks whether
        this run is nameable at all and must be told when it is not; an execution
        already has its data, and the refusal for a missing file belongs to the
        tool that reads it, which can say which file and why. The two agree
        whenever the answer matters: if the data is absent the run fails either
        way.
        """
        return planned_train_identity(
            ds,
            kind=self.kind,
            version=self.version,
            params=params,
            data_path=Path(ds.resolve_path(params.data)),
            fingerprint=fingerprint_yolo_dataset,
            base_model=params.base_model,
            require_data=require_data,
        )

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

        # Through plan_identity, so this run is named in exactly one place. A
        # second copy here would be the one that drifts when the payload changes,
        # and it is the one a planner does not read.
        run_id = self.plan_identity(ds, params, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not params.overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            return run_id
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        claim_run_root(ds, ctx, run_root, self.kind, _TRAIN_IDLE_SECONDS)
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
            **(params.train_overrides or {}),
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
    """Train the heatmap localizer, registering the directory it produces."""

    kind = "train-localizer"
    category = "train"
    domain = "tracking"
    version = "0.1"
    Params = LocalizerTrainParams

    def plan_identity(
        self, ds: Dataset, params: LocalizerTrainParams, *, require_data: bool = True
    ) -> OpIdentity:
        """What this run, and the model it produces, will be called.

        *require_data* separates planning from execution. A planner asks whether
        this run is nameable at all and must be told when it is not; an execution
        already has its data, and the refusal for a missing file belongs to the
        tool that reads it, which can say which file and why. The two agree
        whenever the answer matters: if the data is absent the run fails either
        way.
        """
        return planned_train_identity(
            ds,
            kind=self.kind,
            version=self.version,
            params=params,
            data_path=Path(ds.resolve_path(params.dataset_dir)),
            fingerprint=fingerprint_dataset,
            base_model=params.base_model,
            require_data=require_data,
        )

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

        run_id = self.plan_identity(ds, params, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not params.overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            return run_id
        ctx.set_total(params.epochs)
        run_root = model_run_root(ds, self.kind, run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        claim_run_root(ds, ctx, run_root, self.kind, _TRAIN_IDLE_SECONDS)
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


def _trained_model_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[str]]:
    """Every trained model under ``models/<kind>/<run_id>/``.

    A model is one artifact rather than a per-entry set, so its coverage is
    itself: covered or not. Covered means what :func:`training_is_complete`
    already means -- an index row that says finished *and* an artifact that
    resolves -- because Ultralytics writes ``best.pt`` progressively, so the file
    alone is not evidence the trainer returned, and a row can outlive what it
    points at.

    ``scope.entries`` does not narrow this: a model is fitted over a scope
    recorded elsewhere, and filtering it by entry would silently drop every model
    from a scoped query rather than reporting the ones that exist.
    """
    from mosaic.core.pipeline.dataset_indexes import root_subdirectories
    from mosaic.core.pipeline.inventory.model import (
        ArtifactRecord,
        Coverage,
        TrainedModelRef,
        classify,
    )
    from mosaic.core.pipeline.index_csv import index_records

    records: list[ArtifactRecord[str]] = []
    for kind in root_subdirectories(ds, "models"):
        index_path = model_index_path(ds, kind)
        reader.note(index_path)
        frame = reader.frame(
            index_path, lambda p=index_path: trained_model_index(p).read()
        )
        if frame.empty or "run_id" not in frame.columns:
            continue
        seen: dict[str, dict[str, str]] = {}
        for record in index_records(frame):
            seen.setdefault(record.get("run_id", ""), record)
        for run_id in sorted(seen):
            row = seen[run_id]
            covered = training_is_complete(ds, kind, run_id)
            coverage = Coverage(
                target=frozenset({run_id}),
                present=frozenset({run_id} if covered else ()),
            )
            records.append(
                ArtifactRecord[str](
                    ref=TrainedModelRef(op_kind=kind, run_id=run_id),
                    name=kind,
                    run_id=run_id,
                    coverage=coverage,
                    status=classify(
                        satisfied=covered,
                        any_covered=covered,
                        orphan_rows=not covered and row.get("status", "") == "finished",
                        orphan_files=False,
                        drifted=False,
                        finished=bool(row.get("finished_at", "")),
                    ),
                    run_root=model_run_root(ds, kind, run_id),
                    index_path=index_path,
                    rows=frozenset({run_id}),
                    started_at=row.get("started_at", ""),
                    finished_at=row.get("finished_at", ""),
                    upstreams=tuple(
                        value for value in (row.get("base_run_id", ""),) if value
                    ),
                )
            )
    return records


register_inventory_contributor("trained-model", _trained_model_records)
