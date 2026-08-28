"""Model-training tracking ops: pose, points (POLO), localizer.

Each op runs a trainer under the Job Contract: content ``run_id``, tracked
storage under ``models/<kind>/<run_id>/``, per-epoch progress routed through
``ctx.progress``, cooperative between-epoch cancellation, retraining lineage, and
a ``TrainedModelIndexRow``.

**Two of the three run somewhere else.** ``train-pose`` and ``train-points`` drive
Ultralytics and the POLO fork in environments the user builds, because both are
AGPL-3.0 and a mosaic that imported either would be one work with it; what crosses
is a JSON request, a JSON response, and progress lines on standard output.
``train-localizer`` is mosaic's own PyTorch and still runs in this process.

The pieces that differ between the two external ops are the request they build,
the environment they reach and what they refuse. Everything else -- the probe, the
claim, the cancel, the reporting -- is :func:`train_through_the_tool`, so the
sequence exists once. Heavy backends are imported lazily inside ``run()`` so
registration stays import-light, and so the seams stay replaceable by a test.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Final, Protocol, Self

from pydantic import Field, model_validator

from mosaic.core.helpers import text_cell
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.job import Cancelled, JobContext
from mosaic.core.pipeline.inventory._read import IndexReader
from mosaic.core.pipeline.inventory.contributors import register_inventory_contributor
from mosaic.core.pipeline.inventory.model import ArtifactRecord, InventoryScope
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.params import (
    Declared,
    HASH_EXCLUDE,
    Params,
)
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op
from mosaic.tracking.common.mint import planned_model_id
from mosaic.tracking.common.toolenv import ToolEnv, ToolExitError
from mosaic.tracking.model_refs import ModelShape, resolve_model
from mosaic.tracking.ops._common import (
    claim_run_root,
    ensure_models_root,
    fingerprint_dataset,
    fingerprint_yolo_dataset,
)
from mosaic.tracking.ops._train_descriptions import (
    BASE_MODEL_DESCRIPTION,
    EPOCHS_DESCRIPTION,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.markers import InflightMarker
    from mosaic.tracking.external.runner.ultralytics_protocol import (
        ProbeResponse,
        TrainRequestBase,
    )
    from mosaic.tracking.pose_training.ultralytics_train import TrainingOutcome


class TrainingTool[RequestT: TrainRequestBase](Protocol):
    """How a training subcommand is launched, whichever environment it runs in.

    Generic in the request because each launcher takes exactly its own shape: a
    pose tool cannot be handed a point request, and widening the parameter to the
    base would say it could.
    """

    def __call__(
        self,
        request: RequestT,
        /,
        *,
        work_dir: Path,
        idle_timeout: float,
        cancel_check: Callable[[], bool] | None,
        on_output: Callable[[str], None] | None,
    ) -> TrainingOutcome: ...


_TRAIN_IDLE_SECONDS = 1800.0
"""How long a training run may say nothing before it is presumed hung.

The tool reports an epoch when one ends and a heartbeat every thirty seconds in
between, so this bounds a genuinely silent process rather than a slow one -- and
it has to, because an epoch on a large dataset outruns any window that would
otherwise be reasonable.
"""

_TRAIN_CANCEL_GRACE_SECONDS = 900.0
"""How long a cancelled training run is given to finish its epoch and stop.

Ultralytics reads its stop flag between epochs and nowhere else, so a cancel is a
file the tool notices at the next boundary; this is how long mosaic waits for
that before falling back to the process kill every other tool gets immediately.

It must exceed one epoch, or the kill always wins and the file is decorative. It
must also stay **under** whatever grace the substrate running mosaic imposes: a
container runtime that SIGKILLs the process tree on its own timer makes a longer
value moot, and the ordering has to be one epoch, then this, then that. Nine
hundred is half the default a mosaic-queue pod is given, which leaves that
ordering intact without needing to read it from anywhere.
"""

_TRAIN_RUN_NAME = "train"
"""The subdirectory Ultralytics writes into, under the claimed run root.

Sent to the tool and composed by mosaic to read ``best.pt`` and ``results.csv``
back, which is why it is one constant rather than two literals.
"""

OP_SUPPLIED_TRAIN_ARGS: Final = frozenset(
    {"project", "name", "callback", "cancel_check", "task", "exist_ok"}
)
"""Trainer arguments the op supplies that are not params fields.

``project`` / ``name`` address the claimed run root, ``task`` selects the
ultralytics task, and ``exist_ok`` pins the directory the tool writes into --
Ultralytics renames one that is already occupied, and mosaic composes that path
itself to read the weights back, so a rename would register some other attempt's
model under this run's identifier.

``callback`` and ``cancel_check`` name nothing the trainer takes any more: they
were the Job Contract's hooks when it ran in this process, and progress and
cancellation now cross as JSON lines and a file. They stay refused because the
overrides are applied last and a caller who set either would be describing a
mechanism that no longer exists, which is worth a message rather than silence.

Together with the params fields themselves these are what ``train_overrides``
may not set.
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


def _resolved_base(ds: Dataset, kind: str, base_model: str) -> tuple[str, str, str]:
    """What this run fine-tunes from: the path to hand the tool, and its lineage.

    ``model_id`` rather than ``run_id``: a bare path has no run, and hashing an
    empty string there let two fine-tunes from *different* weights collide
    whenever their params and data matched.
    """
    if not base_model:
        return "", "", ""
    base = resolve_model(ds, base_model, kind)
    return str(base.path), base.model_id, base.digest


def build_train_request[RequestT: TrainRequestBase](
    shape: type[RequestT],
    ds: Dataset,
    params: PoseTrainParams,
    *,
    run_root: Path,
    execution_id: str,
    base_weights: str,
    data_yaml: Path,
    **extra: object,
) -> RequestT:
    """Resolve everything the tool is not allowed to decide, and say it once.

    Built **before** the run root is claimed and before an interpreter is spawned,
    which is what makes an unknown augmentation preset or a resume with no
    checkpoint a refusal at submit rather than a failure on a GPU node.

    Three things collapse into one ``model``, in the order the in-process path
    applied them: the checkpoint when the run resumes, else the resolved base
    weights when it fine-tunes, else the caller's own value verbatim -- which may
    be a bare asset name the tool resolves itself. One field because there was
    only ever one winner, and two would let the tool disagree with the identity
    mosaic minted from both.
    """
    from mosaic.tracking.pose_training.augmentation import resolve_augmentation
    from mosaic.tracking.pose_training.train import find_last_checkpoint
    from mosaic.tracking.pose_training.ultralytics_train import (
        CANCEL_SENTINEL_NAME,
        attempt_directory,
    )

    model = base_weights or params.model
    if params.resume:
        model = str(find_last_checkpoint(run_root, _TRAIN_RUN_NAME))
    # A resume restores the augmentation its checkpoint was trained under, and
    # overriding that is not what resuming means -- the in-process path skipped
    # the resolution entirely, and this records what was actually applied.
    augment = {} if params.resume else (resolve_augmentation(params.augmentation) or {})

    return shape(
        model=model,
        data_yaml=str(data_yaml),
        epochs=params.epochs,
        imgsz=params.imgsz,
        batch=params.batch,
        device=params.device,
        patience=params.patience,
        project_dir=str(run_root),
        run_name=_TRAIN_RUN_NAME,
        resume=params.resume,
        augment=augment,
        train_overrides=dict(params.train_overrides or {}),
        cancel_sentinel=str(
            attempt_directory(run_root, execution_id) / CANCEL_SENTINEL_NAME
        ),
        **extra,
    )


def train_through_the_tool[RequestT: TrainRequestBase](
    ctx: JobContext,
    request: RequestT,
    *,
    run_root: Path,
    marker: InflightMarker,
    base_weights: str,
    preflight: Callable[[ProbeResponse, str], None],
    run_tool: TrainingTool[RequestT],
    env: ToolEnv,
    failure: type[ToolExitError],
) -> TrainingOutcome:
    """Run one training job in the environment its model belongs to.

    Shared by ``train-pose`` and ``train-points``, which differ only in the
    request they build, the environment they reach and what they refuse -- all of
    which arrive as arguments, so the sequence itself exists once.

    Three things happen here that do not happen for any other external tool.

    The environment is **probed before the root is claimed**, so a missing
    Ultralytics or a checkpoint from the wrong fork is a message rather than a
    claim left behind on a run that was never going to start. It is probed after
    the reuse gate, so a cache hit does not pay for a cold torch import.

    A cancel **asks before it kills**. Handed the job's raw token the supervisor
    would kill the process group and lose the epoch in flight; handed
    :func:`~mosaic.tracking.common.cooperative_cancel.stop_then_kill` it writes
    the file the tool stats between epochs, and only escalates once the grace is
    spent.

    And the tool's own output **keeps the claim alive**. A training run outlasts
    the claim's window routinely, and until it ran out of process there was no
    line to hang a refresh on -- so the run root was read as abandoned by whatever
    came next.
    """
    from mosaic.core.pipeline.subprocess_util import ProcessCancelled
    from mosaic.tracking.common.cooperative_cancel import stop_then_kill
    from mosaic.tracking.common.entry import phase_activity
    from mosaic.tracking.common.ultralytics_env import (
        probe_environment,
        training_activity,
    )
    from mosaic.tracking.pose_training.ultralytics_train import attempt_directory

    probe = probe_environment(
        base_weights,
        env=env,
        failure=failure,
        cancel_check=ctx.cancel_token.is_cancelled,
    )
    preflight(probe, base_weights)

    work_dir = attempt_directory(run_root, ctx.execution_id)
    try:
        outcome = run_tool(
            request,
            work_dir=work_dir,
            idle_timeout=_TRAIN_IDLE_SECONDS,
            cancel_check=stop_then_kill(
                ctx.cancel_token.is_cancelled,
                Path(request.cancel_sentinel),
                _TRAIN_CANCEL_GRACE_SECONDS,
            ),
            on_output=training_activity(
                ctx,
                phase_activity(ctx, run_root, marker, _TRAIN_IDLE_SECONDS),
            ),
        )
    except ProcessCancelled as killed:
        # The tool was asked and did not stop in time, so it was killed. That is a
        # cancelled attempt, not a failed one -- the same reading the tracker
        # driver gives it.
        raise Cancelled() from killed

    ctx.check_cancel()
    if outcome.stop == "cancelled":
        # The token is normally already set, and `check_cancel` above has raised.
        # Reaching here means the tool stopped short because it was asked to while
        # this process no longer thinks it was -- registering the truncated model
        # under a finished run's identifier is the one outcome that must not
        # happen.
        raise Cancelled()
    return outcome


# --- Params --------------------------------------------------------------

_DATA_DESCRIPTION = (
    "Path to the data.yaml declaring the training dataset: its classes, "
    "keypoint shape and splits."
)

_POINT_DATA_DESCRIPTION = (
    "Path to the data.yaml declaring the training dataset: its classes, "
    "per-class radii and splits."
)

_MODEL_DESCRIPTION = (
    "What training starts from: a model config, a bare asset name Ultralytics "
    "resolves itself, or a path to weights. base_model overrides it, and a "
    "resume overrides both with this run's last checkpoint."
)

_IMGSZ_DESCRIPTION = (
    "The side a training image is resized to before the model reads it."
)

_PATIENCE_DESCRIPTION = (
    "How long training continues without improvement before stopping early."
)

_RESUME_DESCRIPTION = (
    "Continue training from this run's own last checkpoint instead of "
    "starting from the given weights."
)

_AUGMENTATION_DESCRIPTION = (
    "A preset name, or a dict with a preset key to start from one and "
    "override, or without one to replace the augmentation set outright. A "
    "resumed run applies no augmentation."
)

_TRAIN_OVERRIDES_DESCRIPTION = (
    "Extra keyword arguments forwarded verbatim to yolo.train. Keys that "
    "would collide with a typed field or with an argument the op supplies "
    "are refused."
)

_DEVICE_DESCRIPTION = "Which accelerator trains the model: a GPU index, or cpu."

_BATCH_DESCRIPTION = "How many training images the model reads in one forward pass."

_LOC_DESCRIPTION = "The localization loss weight, a POLO train keyword."

_LOC_LOSS_DESCRIPTION = "Which localization loss POLO minimizes."

_DOR_DESCRIPTION = "The Distance of Reference threshold POLO evaluates against."

_BACKEND_DESCRIPTION = (
    "The point-detection backend. polo is the only value the op accepts."
)

_DATASET_DIR_DESCRIPTION = (
    "The directory convert_coco_localizer writes its train and valid patch sets into."
)

_NUM_CLASSES_DESCRIPTION = "The number of output heatmap channels."

_INITIAL_CHANNELS_DESCRIPTION = "The base channel width of the localizer network."

_FREEZE_ENCODER_DESCRIPTION = (
    "Freeze every layer except the 1x1 output head. Useful when fine-tuning "
    "on a small dataset."
)

_LR_DESCRIPTION = "The initial Adam learning rate."

_EARLY_STOPPING_PATIENCE_DESCRIPTION = (
    "How long training continues without validation-loss improvement before "
    "stopping early."
)

_AUGMENT_DESCRIPTION = (
    "Apply the light augmentation preset -- flip and rotation -- during "
    "training. False applies none."
)

_SEED_DESCRIPTION = "The random seed for the training run."

_LOCALIZER_BATCH_SIZE_DESCRIPTION = (
    "How many training patches the model reads in one forward pass."
)


class PoseTrainParams(Params):
    """Parameters for the ``train-pose`` op.

    The typed fields are the decisions mosaic has an opinion about. Everything
    else ultralytics exposes -- the learning-rate schedule, the loss weights, the
    long tail of ``yolo.train`` knobs -- is reachable through *train_overrides*,
    which reaches identity for the same reason the typed fields do: a model
    trained with a different learning rate is a different model whether or not
    there is a field for it here.
    """

    data: Annotated[str, Declared(_DATA_DESCRIPTION)]
    model: Annotated[str, Declared(_MODEL_DESCRIPTION)] = "yolo11n-pose.pt"
    base_model: Annotated[str, Declared(BASE_MODEL_DESCRIPTION)] = ""
    epochs: Annotated[int, Declared(EPOCHS_DESCRIPTION, unit="epochs")] = 300
    imgsz: Annotated[int, Declared(_IMGSZ_DESCRIPTION, unit="px")] = 640
    patience: Annotated[int, Declared(_PATIENCE_DESCRIPTION, unit="epochs")] = 50
    resume: Annotated[bool, Declared(_RESUME_DESCRIPTION)] = False
    augmentation: Annotated[
        str | dict[str, JsonValue] | None,
        Field(examples=["none", "light", "medium", "heavy"]),
        Declared(_AUGMENTATION_DESCRIPTION),
    ] = None
    # Hashed: a different learning rate is a different model.
    train_overrides: Annotated[
        dict[str, JsonValue] | None, Declared(_TRAIN_OVERRIDES_DESCRIPTION)
    ] = None

    # Throughput / environment knobs, excluded from the run_id.
    device: Annotated[
        str,
        HASH_EXCLUDE,
        Field(examples=["cpu", "0"]),
        Declared(_DEVICE_DESCRIPTION),
    ] = "0"
    batch: Annotated[int, HASH_EXCLUDE, Declared(_BATCH_DESCRIPTION)] = 16

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
    # make_polo_data_yaml (pose_training/prep.py) writes no kpt_shape.
    data: Annotated[str, Declared(_POINT_DATA_DESCRIPTION)]
    # POLO nano localizer config. Resolves to ``cfg/models/26/polo26.yaml`` at scale ``n``
    # (Locate task) in the mooch443/POLO fork; matches the deployed ``polo26n`` detector.
    model: Annotated[str, Declared(_MODEL_DESCRIPTION)] = "polo26n.yaml"
    loc: Annotated[float, Declared(_LOC_DESCRIPTION)] = 5.0
    loc_loss: Annotated[
        str, Field(examples=["mse"]), Declared(_LOC_LOSS_DESCRIPTION)
    ] = "mse"
    dor: Annotated[float, Declared(_DOR_DESCRIPTION)] = 0.8
    backend: Annotated[
        str, HASH_EXCLUDE, Field(examples=["polo"]), Declared(_BACKEND_DESCRIPTION)
    ] = "polo"


class LocalizerTrainParams(Params):
    dataset_dir: Annotated[str, Declared(_DATASET_DIR_DESCRIPTION)]
    base_model: Annotated[str, Declared(BASE_MODEL_DESCRIPTION)] = ""
    num_classes: Annotated[int, Declared(_NUM_CLASSES_DESCRIPTION)] = 4
    initial_channels: Annotated[int, Declared(_INITIAL_CHANNELS_DESCRIPTION)] = 32
    freeze_encoder: Annotated[bool, Declared(_FREEZE_ENCODER_DESCRIPTION)] = False
    epochs: Annotated[int, Declared(EPOCHS_DESCRIPTION, unit="epochs")] = 200
    lr: Annotated[float, Declared(_LR_DESCRIPTION)] = 1e-3
    early_stopping_patience: Annotated[
        int, Declared(_EARLY_STOPPING_PATIENCE_DESCRIPTION, unit="epochs")
    ] = 20
    augment: Annotated[bool, Declared(_AUGMENT_DESCRIPTION)] = True
    seed: Annotated[int, Declared(_SEED_DESCRIPTION)] = 42
    # Throughput / environment knobs, excluded from the run_id.
    device: Annotated[
        str,
        HASH_EXCLUDE,
        Field(examples=["cpu", "0"]),
        Declared(_DEVICE_DESCRIPTION),
    ] = "0"
    batch_size: Annotated[
        int, HASH_EXCLUDE, Declared(_LOCALIZER_BATCH_SIZE_DESCRIPTION)
    ] = 128


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
    scope_takes = "none"
    scope_dependent = False
    Params = PoseTrainParams

    def plan_identity(
        self,
        ds: Dataset,
        params: PoseTrainParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
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

    def run(
        self,
        ds: Dataset,
        params: PoseTrainParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        from mosaic.tracking.common.ultralytics_env import (
            ULTRALYTICS_ENV,
            UltralyticsError,
        )
        from mosaic.tracking.external.runner.ultralytics_protocol import (
            TrainPoseRequest,
        )
        from mosaic.tracking.pose_training.ultralytics_train import (
            require_pose_training_env,
            run_pose_training_tool,
        )

        ensure_models_root(ds)
        data_yaml = Path(ds.resolve_path(params.data))
        base_weights, base_run_id, base_digest = _resolved_base(
            ds, self.kind, params.base_model
        )

        # Through plan_identity, so this run is named in exactly one place. A
        # second copy here would be the one that drifts when the payload changes,
        # and it is the one a planner does not read.
        run_id = self.plan_identity(ds, params, scope, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            ctx.cache_hit()
            return run_id

        run_root = model_run_root(ds, self.kind, run_id)
        request = build_train_request(
            TrainPoseRequest,
            ds,
            params,
            run_root=run_root,
            execution_id=ctx.execution_id,
            base_weights=base_weights,
            data_yaml=data_yaml,
        )

        ctx.set_total(params.epochs)
        run_root.mkdir(parents=True, exist_ok=True)
        marker = claim_run_root(ds, ctx, run_root, self.kind, _TRAIN_IDLE_SECONDS)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        outcome = train_through_the_tool(
            ctx,
            request,
            run_root=run_root,
            marker=marker,
            base_weights=base_weights,
            preflight=require_pose_training_env,
            run_tool=run_pose_training_tool,
            env=ULTRALYTICS_ENV,
            failure=UltralyticsError,
        )
        finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            base_digest,
            outcome.save_dir / "weights" / "best.pt",
            outcome.save_dir / "results.csv",
            outcome.epochs_completed,
        )
        return run_id


@register_op
class TrainPointsOp(Op[PointTrainParams]):
    """Train a POLO point-detection model, registering the directory it produces."""

    kind = "train-points"
    category = "train"
    domain = "tracking"
    version = "0.2"  # see TrainPoseOp.version
    scope_takes = "none"
    scope_dependent = False
    Params = PointTrainParams

    def plan_identity(
        self,
        ds: Dataset,
        params: PointTrainParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
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

    def run(
        self,
        ds: Dataset,
        params: PointTrainParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        from mosaic.tracking.common.ultralytics_env import POLO_ENV, PoloError
        from mosaic.tracking.external.runner.ultralytics_protocol import (
            TrainPointsRequest,
        )
        from mosaic.tracking.pose_training.ultralytics_train import (
            require_points_training_env,
            run_point_training_tool,
        )

        if params.backend != "polo":
            raise ValueError(
                f"Unsupported point-detection backend: {params.backend!r}. "
                "Currently only 'polo' is supported."
            )

        ensure_models_root(ds)
        data_yaml = Path(ds.resolve_path(params.data))
        base_weights, base_run_id, base_digest = _resolved_base(
            ds, self.kind, params.base_model
        )

        # Through plan_identity, so this run is named in exactly one place. A
        # second copy here would be the one that drifts when the payload changes,
        # and it is the one a planner does not read.
        run_id = self.plan_identity(ds, params, scope, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            ctx.cache_hit()
            return run_id

        run_root = model_run_root(ds, self.kind, run_id)
        request = build_train_request(
            TrainPointsRequest,
            ds,
            params,
            run_root=run_root,
            execution_id=ctx.execution_id,
            base_weights=base_weights,
            data_yaml=data_yaml,
            loc=params.loc,
            loc_loss=params.loc_loss,
            dor=params.dor,
        )

        ctx.set_total(params.epochs)
        run_root.mkdir(parents=True, exist_ok=True)
        marker = claim_run_root(ds, ctx, run_root, self.kind, _TRAIN_IDLE_SECONDS)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

        outcome = train_through_the_tool(
            ctx,
            request,
            run_root=run_root,
            marker=marker,
            base_weights=base_weights,
            preflight=require_points_training_env,
            run_tool=run_point_training_tool,
            env=POLO_ENV,
            failure=PoloError,
        )
        finalize_training(
            ds,
            self.kind,
            run_id,
            run_root,
            params,
            params.base_model,
            base_run_id,
            base_digest,
            outcome.save_dir / "weights" / "best.pt",
            outcome.save_dir / "results.csv",
            outcome.epochs_completed,
        )
        return run_id


@register_op
class TrainLocalizerOp(Op[LocalizerTrainParams]):
    """Train the heatmap localizer, registering the directory it produces."""

    kind = "train-localizer"
    category = "train"
    domain = "tracking"
    version = "0.1"
    scope_takes = "none"
    scope_dependent = False
    Params = LocalizerTrainParams

    def plan_identity(
        self,
        ds: Dataset,
        params: LocalizerTrainParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
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

    def run(
        self,
        ds: Dataset,
        params: LocalizerTrainParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
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

        run_id = self.plan_identity(ds, params, scope, require_data=False).run_id
        ctx.set_run_id(run_id)
        if not overwrite and training_is_complete(ds, self.kind, run_id):
            print(f"[{self.kind}] {run_id} already trained; reusing it.")
            ctx.cache_hit()
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
