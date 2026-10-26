"""Defines the predict-frames op, which predicts a trained model on extracted frames.

Every refusal of a misused model at plan time reads the run ids and the params
alone. A misuse is therefore refused before a model is read, and also for a SLEAP
pair whose trainings are not on disk yet. Once the run resolves the models, it
reads the heads of SLEAP models and the type of a Lightning Pose model from
their configs and checks them.

A run claims each entry that it predicts and hands the images of every held entry
to its framework's frame adapter in one call. The model then loads once per run.
The run writes one predictions file and index row per entry camera. The tool's
probe, and every check that the probe answers, run before any claim.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.annotations.pose_predictions import (
    FramePredictions,
    PredictedFrame,
    PredictedObject,
    PredictedPose,
    pose_predictions_payload,
)
from mosaic.core.entry import make_entry_key
from mosaic.core.helpers import entry_camera_path
from mosaic.core.pipeline._utils import atomic_write, hash_params
from mosaic.core.pipeline.entry_claim import (
    discard_if_empty,
    open_entry,
    phase_activity,
    release_entry,
)
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.op_identity import (
    OP_IDENTITY_SCHEME,
    op_run_id,
    parse_op_run_id,
)
from mosaic.core.pipeline.ops import ModelReference, Op, OpIdentity, register_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.wording import counted
from mosaic.tracking.common import ultralytics_env
from mosaic.tracking.common.mint import planned_model_id
from mosaic.tracking.common.ultralytics_env import progress_activity
from mosaic.tracking.frame_extraction.frame_runs import (
    FrameFile,
    frame_run_files,
    list_frame_runs,
)
from mosaic.tracking.frame_predictions.adapter import (
    AdapterResult,
    FrameAdapter,
    FrameImage,
    ToolProbe,
)
from mosaic.tracking.frame_predictions.errors import (
    PredictFramesConflict,
    PredictFramesInvalid,
    PredictFramesNotFound,
)
from mosaic.tracking.frame_predictions.index import (
    FramePredictionsRow,
    predict_frames_index,
    prediction_index_path,
    prediction_kind_root,
    write_run_params,
)
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.frame_predictions.read import list_prediction_runs
from mosaic.tracking.litpose.frames import (
    predict_litpose_frames,
    probe_litpose_version,
    require_predict_frame,
)
from mosaic.tracking.litpose.training import litpose_device_placement
from mosaic.tracking.model_refs import (
    ResolvedModel,
    model_kind_for,
    refuse_unless_run_ids,
    resolve_model_set,
    resolve_op_model,
)
from mosaic.tracking.pose_training import ultralytics_infer
from mosaic.tracking.sleap import probe as sleap_probe
from mosaic.tracking.sleap.frames import SleapFrameAdapter
from mosaic.tracking.training_provenance import training_provenance

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.entry import Entry
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "PredictFramesOp",
    "pose_after_return",
    "predict_frames_model",
    "predict_frames_payload",
    "predict_frames_run_id",
    "refuse_device_for_kind",
    "refuse_foreign_framework_fields",
    "refuse_mixed_kinds",
    "refuse_pair_unless_sleap",
]

_KIND: Final = "predict-frames"

_PREDICT_MODEL: Final = ModelReference(
    field="model",
    kinds=("train-pose", "train-points", "train-sleap", "train-litpose"),
    many=True,
)
"""The trained models that predict-frames runs: one, or a SLEAP top-down pair."""

_SLEAP_KIND: Final = "train-sleap"
_LITPOSE_KIND: Final = "train-litpose"

_TOP_DOWN_HEADS: Final = frozenset(
    {"centroid", "centered_instance", "multi_class_topdown"}
)
"""The SLEAP heads that are one half of a top-down model."""

_INSTANCE_HEADS: Final = frozenset({"centered_instance", "multi_class_topdown"})
"""The heads of the second model of a SLEAP top-down pair, which follows a centroid."""

_CONTEXT_TYPE: Final = "heatmap_mhcrnn"
"""The Lightning Pose model type that predicts a frame from the frames around it."""

_MULTIVIEW: Final = "multiview"
"""What the name of every multiview Lightning Pose model type contains."""

_FRAMEWORK_OF: Final[Mapping[str, str]] = {
    "train-pose": "yolo-pose",
    "train-points": "POLO",
    "train-sleap": "SLEAP",
    "train-litpose": "Lightning Pose",
}
"""The framework of each training kind's models, as a refusal names it."""

_READ_BY: Final[Mapping[str, tuple[str, ...]]] = {
    "conf_threshold": ("train-pose", "train-points"),
    "imgsz": ("train-pose", "train-points"),
    "iou": ("train-pose",),
    "max_det": ("train-pose", "train-points"),
    "agnostic_nms": ("train-pose", "train-points"),
    "precision": ("train-pose", "train-points"),
    "dor": ("train-points",),
    "peak_threshold": ("train-sleap",),
    "max_instances": ("train-sleap",),
}
"""The training kinds whose models read each framework field of the params."""


def _listed(items: Sequence[str], conjunction: str) -> str:
    """Join *items* as prose: ``a``, ``a and b``, or ``a, b, and c``."""
    if len(items) < 3:
        return f" {conjunction} ".join(items)
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def _run_kinds(refs: Sequence[str]) -> list[tuple[str, str]]:
    """Pair each run id in *refs* with the training kind it names, in order.

    A reference that is not a run id is skipped. :func:`refuse_unless_run_ids`
    refuses it.
    """
    runs: list[tuple[str, str]] = []
    for ref in refs:
        parsed = parse_op_run_id(ref)
        if parsed is not None:
            runs.append((ref, parsed.kind))
    return runs


def _runs_named(runs: Sequence[tuple[str, str]]) -> str:
    """Name each run id of *runs* with its training kind."""
    return _listed([f"{ref!r} is a {kind} run" for ref, kind in runs], "and")


def refuse_mixed_kinds(refs: Sequence[str]) -> None:
    """Refuse run ids of more than one training kind.

    Args:
        refs: The model references as given, in order.

    Raises:
        PredictFramesInvalid: Two run ids name different training kinds. The
            message names each reference and its kind.
    """
    runs = _run_kinds(refs)
    if len({kind for _ref, kind in runs}) < 2:
        return
    message = f"{_KIND} takes models of one training kind. {_runs_named(runs)}."
    raise PredictFramesInvalid(message)


def refuse_pair_unless_sleap(refs: Sequence[str]) -> None:
    """Refuse two or more run ids unless each is a train-sleap run.

    Only a SLEAP top-down model is a pair. The identity of any other pair would
    digest its first model alone. The head order of a train-sleap pair, a
    centroid run followed by a centered-instance run, is checked once the
    models are resolved.

    Args:
        refs: The model references as given, in order.

    Raises:
        PredictFramesInvalid: Two or more references include a run of another
            kind. The message names each reference and its kind.
    """
    runs = _run_kinds(refs)
    if len(refs) < 2 or all(kind == _SLEAP_KIND for _ref, kind in runs):
        return
    message = (
        f"{_KIND} takes two models only as a SLEAP top-down pair of "
        f"{_SLEAP_KIND} runs, centroid first. {_runs_named(runs)}."
    )
    raise PredictFramesInvalid(message)


def refuse_foreign_framework_fields(
    params: PredictFramesParams, model_kind: str
) -> None:
    """Refuse a framework field off its default for a model that does not read it.

    Args:
        params: The run's parameters.
        model_kind: The training kind of the run's models.

    Raises:
        PredictFramesInvalid: A field that *model_kind*'s framework does not read
            differs from its default. The message names every such field, the
            frameworks that read it and the model's framework.
    """
    values: dict[str, object] = params.model_dump()
    declared = PredictFramesParams.model_fields
    foreign: list[str] = []
    for name, readers in _READ_BY.items():
        default: object = declared[name].default
        if model_kind not in readers and values[name] != default:
            foreign.append(name)
    if not foreign:
        return
    framework = _FRAMEWORK_OF[model_kind]
    sentences = [f"{framework} models do not read {_listed(foreign, 'or')}."]
    for name in foreign:
        frameworks = [_FRAMEWORK_OF[reader] for reader in _READ_BY[name]]
        sentences.append(
            f"{name} is read only for {_listed(frameworks, 'and')} models."
        )
    sentences.append(
        "Leave it at its default."
        if len(foreign) == 1
        else "Leave them at their defaults."
    )
    raise PredictFramesInvalid(" ".join(sentences))


def refuse_device_for_kind(device: str, model_kind: str) -> None:
    """Refuse a device that the framework of *model_kind* cannot run on.

    Lightning Pose runs on CUDA only, so a train-litpose model takes ``auto``
    or CUDA indices.

    Args:
        device: The requested device.
        model_kind: The training kind of the run's models.

    Raises:
        PredictFramesInvalid: *device* is unusable for a train-litpose model,
            with the message of :func:`litpose_device_placement`.
    """
    if model_kind != _LITPOSE_KIND:
        return
    try:
        _ = litpose_device_placement(device)
    except ValueError as exc:
        raise PredictFramesInvalid(str(exc)) from exc


def predict_frames_model(ds: Dataset, params: PredictFramesParams) -> tuple[str, str]:
    """Refuse a misuse of the run's models, then return their kind and identity.

    The refusals read the run ids and *params* alone and come first. One run id
    is its own identity and is read from no file. A SLEAP pair is named by the
    content digest of both trainings.

    Args:
        ds: The dataset whose model indexes register the trainings of a pair.
        params: The run's parameters.

    Returns:
        The training kind of the models, and the model identity that
        :func:`predict_frames_run_id` hashes.

    Raises:
        ModelReferenceRefusedError: A reference is not a run id, or is a run of
            a training kind that predict-frames does not run.
        PredictFramesInvalid: The references are of mixed training kinds, two
            references are not a SLEAP pair, a field is set for another
            framework, or the device is unusable for the model's framework.
        IdentityDeferred: A training of a SLEAP pair is not on disk yet.
    """
    refuse_unless_run_ids(_KIND, _PREDICT_MODEL, params.model)
    refuse_mixed_kinds(params.model)
    refuse_pair_unless_sleap(params.model)
    model_kind = model_kind_for(_KIND, _PREDICT_MODEL, params.model)
    refuse_foreign_framework_fields(params, model_kind)
    refuse_device_for_kind(params.device, model_kind)
    return model_kind, planned_model_id(ds, _KIND, params.model, model_kind)


def predict_frames_payload(
    params: PredictFramesParams, *, model_id: str, model_kind: str
) -> dict[str, object]:
    """Return the identity payload of a predict-frames run.

    The run id digests it, and the run writes it as ``run_params.json``. It
    contains the references as given and the extraction run. The frame list and
    the scope stay out of it, so runs over more entries of one extraction run
    share their run id.

    Args:
        params: The run's parameters.
        model_id: The identity of the models, from :func:`predict_frames_model`.
        model_kind: The training kind of the models.
    """
    return {
        "params": params.identity_dump(),
        "model_id": model_id,
        "model_kind": model_kind,
    }


def predict_frames_run_id(
    version: str, params: PredictFramesParams, *, model_id: str, model_kind: str
) -> str:
    """Mint a predict-frames run identifier over :func:`predict_frames_payload`.

    The golden corpus calls it with fixed arguments and no filesystem.

    Args:
        version: The op's declared version.
        params: The run's parameters.
        model_id: The identity of the models, from :func:`predict_frames_model`.
        model_kind: The training kind of the models.

    Returns:
        ``predict-frames.<version>-<digest>``.
    """
    return op_run_id(
        _KIND,
        version,
        predict_frames_payload(params, model_id=model_id, model_kind=model_kind),
    )


# --- The run ---------------------------------------------------------------------


_PREDICT_IDLE_SECONDS: Final = 3600.0
"""The inactivity bound of a held claim, the bound of the inference ops.

Each line that the tool prints re-stamps every held claim. The bound covers the
longest silence of a healthy run, a cold model load.
"""

_PREDICTIONS_NAME: Final = "predictions.json"
"""What an entry camera's predictions file is called, in its directory."""

_WORK_PREFIX: Final = ".mosaic-predict-"
"""The name of an attempt's work directory in the run root, before its execution id."""


@dataclass(frozen=True, slots=True)
class _EntryCamera:
    """One camera of an entry, as the extraction run recorded it.

    Attributes:
        camera: The camera, ``""`` for single-camera media.
        method: The extraction method directory of the frames row.
        composition: The frames row's ``media_composition``.
        files: The extracted images, by frame index.
    """

    camera: str
    method: str
    composition: str
    files: tuple[FrameFile, ...]


@dataclass(frozen=True, slots=True)
class _EntryFrames:
    """The extracted images of one entry, every camera of it, in camera order."""

    group: str
    sequence: str
    cameras: tuple[_EntryCamera, ...]

    @property
    def key(self) -> str:
        """The entry key, which names the entry's claimed directory."""
        return make_entry_key(self.group, self.sequence)

    @property
    def files(self) -> tuple[FrameFile, ...]:
        """Every image of the entry, camera by camera, in the order it is sent."""
        return tuple(file for camera in self.cameras for file in camera.files)


@dataclass(frozen=True, slots=True)
class _Extraction:
    """The scoped entries of an extraction run, read before any claim.

    Attributes:
        entries: The entries whose images were read, in entry order.
        unreadable: The keys of the entries whose images could not be read.
            Each is recorded as failed.
    """

    entries: tuple[_EntryFrames, ...]
    unreadable: tuple[str, ...]

    @property
    def claimable(self) -> int:
        """How many scoped entries have frames in the extraction run."""
        return len(self.entries) + len(self.unreadable)


@dataclass(frozen=True, slots=True)
class _HeldEntry:
    """An entry whose claim the run holds while its tool runs.

    Attributes:
        frames: The entry's images.
        seq_dir: The claimed directory.
        activity: Re-stamps the claim on a line of the tool's output.
    """

    frames: _EntryFrames
    seq_dir: Path
    activity: Callable[[str], None]


@dataclass(frozen=True, slots=True)
class _FrameTool:
    """The tool that predicts a run's images, probed before any claim.

    Attributes:
        probe: What the tool reported.
        adapter: The call that predicts the images.
        names_keypoints: Whether the tool names the keypoints that it predicts.
            Its names are checked against a recorded pose after it returns. A
            tool that reports a count alone is checked before any claim.
    """

    probe: ToolProbe
    adapter: FrameAdapter
    names_keypoints: bool


@dataclass(frozen=True, slots=True)
class _Prediction:
    """The objects that the tool found in the held entries' images, and their pose."""

    pose: PredictedPose
    objects: tuple[tuple[PredictedObject, ...], ...]


@dataclass(frozen=True, slots=True)
class _RunFacts:
    """The cells that every index row of one run shares."""

    run_id: str
    frames_run_id: str
    model: str
    model_kind: str
    model_id: str
    tool_version: str
    params_hash: str


def _resolved_model(ds: Dataset, refs: Sequence[str], model_kind: str) -> ResolvedModel:
    """Resolve the references: one training run, or a SLEAP top-down pair."""
    if len(refs) == 1:
        return resolve_op_model(ds, _KIND, refs[0])
    return resolve_model_set(ds, refs, model_kind)


def _recorded_pose(
    ds: Dataset, model_kind: str, refs: Sequence[str]
) -> PredictedPose | None:
    """Return the pose that the trainings of the run's models were trained on.

    Each training of a SLEAP pair is read. When one of them records a pose,
    that pose is returned.

    Returns:
        The pose that the first training to record one records, or ``None``
        when no training records one.

    Raises:
        PredictFramesInvalid: Two trainings record different keypoint schemas.
            The message names both runs and their keypoint names, and their
            edges or mirror pairs when the names agree.
    """
    recorded: list[tuple[str, PredictedPose]] = []
    for ref in refs:
        provenance = training_provenance(ds, model_kind, ref)
        schema = provenance.keypoint_schema
        if schema is not None:
            pose = PredictedPose(
                id=provenance.pose_id, name=provenance.pose_name, schema=schema
            )
            recorded.append((ref, pose))
    if not recorded:
        return None
    first_ref, first = recorded[0]
    for ref, pose in recorded[1:]:
        if pose.schema != first.schema:
            message = (
                f"The trainings of a SLEAP pair record different poses. "
                f"{first_ref!r} was trained on "
                f"{_schema_named(first.schema, pose.schema)} and {ref!r} on "
                f"{_schema_named(pose.schema, first.schema)}."
            )
            raise PredictFramesInvalid(message)
    return first


def _schema_named(schema: KeypointSchema, other: KeypointSchema) -> str:
    """Return how a refusal names *schema*, with the part that differs from *other*.

    The keypoint names alone name a schema whose names differ. Otherwise the
    edges are added when they differ, else the mirror pairs.
    """
    named = f"the keypoints {list(schema.names)}"
    if schema.names != other.names:
        return named
    if schema.skeleton != other.skeleton:
        return f"{named} with the edges {[list(edge) for edge in schema.skeleton]}"
    pairs = [list(pair) for pair in schema.symmetries]
    return f"{named} with the mirror pairs {pairs}"


def _head_named(head: str) -> str:
    """Return how a refusal names a SLEAP model of *head*, ``""`` for no head."""
    return f"a {head} model" if head else "a model whose config names no head"


def _refuse_sleap_heads(refs: Sequence[str], heads: Sequence[str]) -> None:
    """Refuse a lone half of a top-down model, and a pair that is not one in order.

    Raises:
        PredictFramesInvalid: One model is a centroid, centered_instance or
            multi_class_topdown model, or two models are not a centroid model
            followed by a centered_instance or multi_class_topdown model. The
            message names the heads found, in order.
    """
    if len(refs) == 1:
        head = heads[0]
        if head not in _TOP_DOWN_HEADS:
            return
        message = (
            f"{refs[0]!r} is {_head_named(head)}, one half of a SLEAP top-down "
            f"model. {_KIND} takes a top-down model as a pair, its centroid run "
            "and then its centered_instance or multi_class_topdown run."
        )
        raise PredictFramesInvalid(message)
    first, second = heads
    if first == "centroid" and second in _INSTANCE_HEADS:
        return
    named = _listed(
        [
            f"{ref!r} is {_head_named(head)}"
            for ref, head in zip(refs, heads, strict=True)
        ],
        "and",
    )
    message = (
        "A SLEAP top-down pair is a centroid run followed by a centered_instance "
        f"or multi_class_topdown run. {named}."
    )
    raise PredictFramesInvalid(message)


def _refuse_single_frame_type(refs: Sequence[str], model_type: str) -> None:
    """Refuse a Lightning Pose model that does not predict one image alone.

    A heatmap_mhcrnn model reads the frames around each frame, and a multiview
    model reads every view of a frame.

    Raises:
        PredictFramesInvalid: The model is a heatmap_mhcrnn model or of a
            multiview type. The message names the type.
    """
    if model_type != _CONTEXT_TYPE and _MULTIVIEW not in model_type:
        return
    message = (
        f"{refs[0]!r} is a {model_type} model, which single-frame prediction does "
        f"not support. {_KIND} runs Lightning Pose heatmap and regression models."
    )
    raise PredictFramesInvalid(message)


def _entry_frames(
    ds: Dataset,
    frames_run_id: str,
    entry: Entry,
    rows: Sequence[Mapping[str, str]],
) -> _EntryFrames:
    """Read the images of each camera of *entry* that the extraction run recorded.

    Raises:
        ValueError: A camera's directory lies outside the dataset tree.
    """
    group, sequence = entry
    cameras = tuple(
        _EntryCamera(
            camera=row.get("camera", ""),
            method=row.get("method", ""),
            composition=row.get("media_composition", ""),
            files=frame_run_files(
                ds, frames_run_id, group, sequence, row.get("camera", "")
            ),
        )
        for row in sorted(rows, key=lambda row: row.get("camera", ""))
    )
    return _EntryFrames(group=group, sequence=sequence, cameras=cameras)


def _extraction(
    ds: Dataset, ctx: JobContext, frames_run_id: str, scope: ResolvedScope
) -> _Extraction:
    """Read the images of each scoped entry of the extraction run *frames_run_id*.

    An unset scope covers every entry of the run. A scoped entry without frames
    in the run is skipped and counted. An entry whose images cannot be read is
    recorded as failed, and the other entries are read.

    Raises:
        PredictFramesNotFound: No frames-index row names the run.
        PredictFramesConflict: No scoped entry has frames in the run.
    """
    rows = [
        row
        for row in index_records(list_frame_runs(ds))
        if row.get("run_id", "") == frames_run_id
    ]
    if not rows:
        message = (
            f"No frames index of any extraction method records the run "
            f"{frames_run_id!r}. `mosaic inventory --kind frame-run` lists the "
            "extraction runs."
        )
        raise PredictFramesNotFound(message)
    cameras: dict[Entry, list[dict[str, str]]] = {}
    for row in rows:
        entry = (row.get("group", ""), row.get("sequence", ""))
        cameras.setdefault(entry, []).append(row)
    asked = scope.op_entries
    scoped = sorted(cameras) if asked is None else sorted(set(asked))
    claimable = [entry for entry in scoped if entry in cameras]
    if not claimable:
        named = ", ".join(make_entry_key(*entry) for entry in scoped) or "no entry"
        message = (
            f"None of the scoped entries has frames in the extraction run "
            f"{frames_run_id!r}. The scope names {named}."
        )
        raise PredictFramesConflict(message)
    skipped = len(scoped) - len(claimable)
    if skipped:
        noun = "entry has" if skipped == 1 else "entries have"
        print(
            f"[{_KIND}] {skipped} scoped {noun} no frames in the extraction run "
            f"{frames_run_id}; skipping them."
        )
    entries: list[_EntryFrames] = []
    unreadable: list[str] = []
    for entry in claimable:
        try:
            entries.append(_entry_frames(ds, frames_run_id, entry, cameras[entry]))
        except ValueError as exc:
            key = make_entry_key(*entry)
            ctx.entry_failed(key, exc)
            unreadable.append(key)
    return _Extraction(entries=tuple(entries), unreadable=tuple(unreadable))


def _frame_tool(
    model_kind: str,
    model: ResolvedModel,
    recorded: PredictedPose | None,
    ctx: JobContext,
    *,
    images_root: Path,
) -> _FrameTool:
    """Probe *model_kind*'s tool, check the weights, and pick a frame adapter.

    A SLEAP model's adapter writes its input on the *recorded* pose, with each
    image named by its key under *images_root*, the dataset root.

    Raises:
        UltralyticsError: The Ultralytics environment's probe failed.
        PoloError: The POLO environment's probe failed.
        UltralyticsNotFoundError: The Ultralytics environment lacks Ultralytics.
        PoloNotFoundError: The POLO environment lacks POLO.
        UnsupportedModelError: The weights do not load, or are not a model of
            the training kind's task.
        SleapNotFoundError: No SLEAP install was found, or sleap_nn does not
            import in it.
        SleapError: The SLEAP environment's probe failed.
        LitposeNotFoundError: No Lightning Pose install was found.
        LitposeError: The Lightning Pose environment's probe failed.
        PredictFramesUnavailable: The installed Lightning Pose is older than
            the release whose single-frame prediction is verified.
        ValueError: *model_kind* is not a training kind that the op runs, or
            the Lightning Pose probe did not print a version.
    """
    model_path = str(model.path)
    cancel_check = ctx.cancel_token.is_cancelled
    adapter: FrameAdapter
    match model_kind:
        case "train-pose":
            response = ultralytics_env.probe_environment(
                model_path,
                env=ultralytics_env.ULTRALYTICS_ENV,
                failure=ultralytics_env.UltralyticsError,
                cancel_check=cancel_check,
            )
            ultralytics_infer.require_pose_model(response, model_path)
            adapter = ultralytics_infer.predict_pose_frames
        case "train-points":
            response = ultralytics_env.probe_environment(
                model_path,
                env=ultralytics_env.POLO_ENV,
                failure=ultralytics_env.PoloError,
                cancel_check=cancel_check,
            )
            ultralytics_infer.require_points_model(response, model_path)
            adapter = ultralytics_infer.predict_point_frames
        case "train-sleap":
            found = sleap_probe.probe_sleap(cancel_check=cancel_check)
            sleap_probe.require_sleap_nn(found)
            schema = None if recorded is None else recorded.schema
            return _FrameTool(
                probe=ToolProbe(tool_version=found.sleap_nn_version, n_keypoints=None),
                adapter=SleapFrameAdapter(schema=schema, images_root=images_root),
                names_keypoints=True,
            )
        case "train-litpose":
            version = probe_litpose_version(cancel_check=cancel_check)
            require_predict_frame(version)
            return _FrameTool(
                probe=ToolProbe(tool_version=version, n_keypoints=None),
                adapter=predict_litpose_frames,
                names_keypoints=True,
            )
        case _:
            message = f"{model_kind} is not a training kind that {_KIND} runs."
            raise ValueError(message)
    probe = ToolProbe(
        tool_version=response.ultralytics_version, n_keypoints=response.n_keypoints
    )
    return _FrameTool(probe=probe, adapter=adapter, names_keypoints=False)


def _refuse_another_count(
    recorded: PredictedPose | None, probe: ToolProbe, refs: Sequence[str]
) -> None:
    """Refuse a model whose keypoint count differs from its recorded pose's.

    Raises:
        PredictFramesConflict: The probe counts other keypoints than the pose
            that the model's provenance records.
    """
    count = probe.n_keypoints
    if recorded is None or count is None:
        return
    expected = recorded.schema.num_keypoints
    if count == expected:
        return
    predicted = counted(count, "keypoint")
    trained = counted(expected, "keypoint")
    message = (
        f"The model {', '.join(refs)} predicts {predicted}, and its training "
        f"recorded the pose {recorded.name!r} of {trained}: "
        f"{list(recorded.schema.names)}."
    )
    raise PredictFramesConflict(message)


def pose_after_return(
    ctx: JobContext,
    held: Sequence[str],
    recorded: PredictedPose | None,
    tool_pose: KeypointSchema,
    *,
    model: Sequence[str],
) -> PredictedPose | None:
    """Return the pose that a run writes once its tool has named its keypoints.

    Without a recorded pose, the tool's pose is written with a null pose id and
    an empty name. A recorded pose is written when the tool names its
    keypoints in the same order. Otherwise a :class:`PredictFramesConflict`
    that names both name lists and the model is recorded for every held entry.

    Args:
        ctx: The run's job context, which records the failures.
        held: The keys of the entries that the run holds.
        recorded: The pose that the model's provenance records, or ``None``.
        tool_pose: The keypoints that the tool reports, in its order.
        model: The model references as given.

    Returns:
        The pose to write, or ``None`` once every held entry is recorded as
        failed.
    """
    if recorded is None:
        return PredictedPose(id=None, name="", schema=tool_pose)
    if tool_pose.names == recorded.schema.names:
        return recorded
    message = (
        f"The model {', '.join(model)} names its keypoints "
        f"{list(tool_pose.names)}, and its training recorded the pose "
        f"{recorded.name!r} of {list(recorded.schema.names)}."
    )
    # Raised and caught: an entry's record contains a traceback only when it is
    # recorded inside an except block.
    try:
        raise PredictFramesConflict(message)
    except PredictFramesConflict as conflict:
        for key in held:
            ctx.entry_failed(key, conflict)
    return None


def _frame_images(ds: Dataset, held: Sequence[_HeldEntry]) -> list[FrameImage]:
    """Return the images of every held entry, numbered in the order they are sent."""
    images: list[FrameImage] = []
    for entry in held:
        for file in entry.frames.files:
            image = FrameImage(
                position=len(images),
                key=file.image,
                path=ds.resolve_path(file.image),
                width=file.width,
                height=file.height,
            )
            images.append(image)
    return images


def _liveness(held: Sequence[_HeldEntry]) -> Callable[[str], None]:
    """Return the callback that hands each line of the tool's output to every claim."""

    def on_line(line: str) -> None:
        for entry in held:
            entry.activity(line)

    return on_line


def _predict_held(
    ds: Dataset,
    ctx: JobContext,
    run_id: str,
    params: PredictFramesParams,
    model: ResolvedModel,
    tool: _FrameTool,
    recorded: PredictedPose | None,
    held: Sequence[_HeldEntry],
    work_dir: Path,
) -> _Prediction | None:
    """Predict the images of every held entry in one call, and decide their pose.

    Returns:
        The objects found in each image, in the order sent, and the pose to
        write. ``None`` when the tool failed or contradicted the recorded pose,
        each recorded as every held entry's failure.

    Raises:
        Cancelled: The run was cancelled.
        ProcessCancelled: The tool was stopped by a cancel.
    """
    keys = [entry.frames.key for entry in held]
    images = _frame_images(ds, held)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = tool.adapter(
            images,
            model,
            params,
            tool.probe,
            work_dir,
            on_output=progress_activity(ctx, run_id, "infer", _liveness(held)),
            cancel_check=ctx.cancel_token.is_cancelled,
        )
        _refuse_unmatched(result, images)
    except (Cancelled, ProcessCancelled):
        raise
    except Exception as exc:
        for key in keys:
            ctx.entry_failed_unless_cancelled(key, exc)
        return None
    if tool.names_keypoints:
        pose = pose_after_return(
            ctx, keys, recorded, result.tool_pose, model=params.model
        )
    else:
        pose = recorded or PredictedPose(id=None, name="", schema=result.tool_pose)
    return None if pose is None else _Prediction(pose=pose, objects=result.objects)


def _refuse_unmatched(result: AdapterResult, images: Sequence[FrameImage]) -> None:
    """Refuse an adapter result without one tuple of objects per image.

    Raises:
        ValueError: The result has objects for more or fewer images.
    """
    if len(result.objects) != len(images):
        message = (
            f"the frame adapter returned the objects of {len(result.objects)} "
            f"images for {len(images)} images"
        )
        raise ValueError(message)


def _write_entry(
    ds: Dataset,
    run_root: Path,
    facts: _RunFacts,
    entry: _EntryFrames,
    pose: PredictedPose,
    found: Sequence[tuple[PredictedObject, ...]],
) -> list[FramePredictionsRow]:
    """Write the predictions file of each camera of *entry*, and return their rows.

    *found* contains the objects of each image of the entry, in the order of
    :attr:`_EntryFrames.files`.

    Raises:
        ValueError: An object's keypoints do not match the pose.
    """
    rows: list[FramePredictionsRow] = []
    position = 0
    for camera in entry.cameras:
        frames: list[PredictedFrame] = []
        for file in camera.files:
            frames.append(
                PredictedFrame(
                    image=file.image,
                    frame_index=file.frame_index,
                    width=file.width,
                    height=file.height,
                    objects=found[position],
                )
            )
            position += 1
        predictions = FramePredictions(
            pose=pose,
            frames_run_id=facts.frames_run_id,
            group=entry.group,
            sequence=entry.sequence,
            camera=camera.camera,
            frames=tuple(frames),
        )
        payload = pose_predictions_payload(predictions)
        path = (
            entry_camera_path(run_root, entry.group, entry.sequence, camera.camera)
            / _PREDICTIONS_NAME
        )
        atomic_write(path, lambda target: target.write_bytes(payload))
        rows.append(
            FramePredictionsRow(
                run_id=facts.run_id,
                abs_path=Path(ds.relative_to_root(path)),
                group=entry.group,
                sequence=entry.sequence,
                camera=camera.camera,
                frames_method=camera.method,
                frames_run_id=facts.frames_run_id,
                model=facts.model,
                model_kind=facts.model_kind,
                model_id=facts.model_id,
                n_frames=len(frames),
                n_objects=sum(len(frame.objects) for frame in frames),
                consumed_frames_composition=camera.composition,
                tool_version=facts.tool_version,
                params_hash=facts.params_hash,
            )
        )
    return rows


def _finished_files(ds: Dataset, run_id: str) -> dict[tuple[str, str, str], str]:
    """Return the file that each finished row of *run_id* names, by entry camera."""
    return {
        (row.get("group", ""), row.get("sequence", ""), row.get("camera", "")): row.get(
            "abs_path", ""
        )
        for row in index_records(list_prediction_runs(ds))
        if row.get("run_id", "") == run_id and row.get("finished_at", "")
    }


def _is_written(
    ds: Dataset, finished: Mapping[tuple[str, str, str], str], entry: _EntryFrames
) -> bool:
    """Return whether every camera of *entry* has a finished row and its file."""
    for camera in entry.cameras:
        stored = finished.get((entry.group, entry.sequence, camera.camera), "")
        if not stored or not ds.resolve_path(stored).is_file():
            return False
    return True


def _entry_done(ctx: JobContext, done: int, total: int, key: str) -> None:
    """Report that *done* of the run's *total* entries have ended, *key* last."""
    ctx.progress.on_entry_end(done, total, key)
    ctx.heartbeat(done)


@register_op
class PredictFramesOp(Op[PredictFramesParams]):
    """Predict a trained model on the frames of one extraction run, per entry camera."""

    kind = _KIND
    category = "infer"
    domain = "tracking"
    version = "0.1"
    writes_tracks = False
    scope_takes = "any"
    scope_dependent = False
    model_reference = _PREDICT_MODEL
    Params = PredictFramesParams

    def plan_identity(
        self,
        ds: Dataset,
        params: PredictFramesParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """Name the run from its params and its models' kind and identity.

        The scope stays out of the identity. A SLEAP pair whose trainings are
        not on disk defers it, whatever *require_data* says.
        """
        model_kind, model_id = predict_frames_model(ds, params)
        return OpIdentity(
            run_id=predict_frames_run_id(
                self.version, params, model_id=model_id, model_kind=model_kind
            )
        )

    def run(
        self,
        ds: Dataset,
        params: PredictFramesParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        """Predict the scoped entries of the extraction run, and record each one.

        Without *overwrite*, an entry whose every camera has a finished row and
        its file is reused. Under *overwrite*, every entry is predicted again.
        An entry that another execution holds is skipped.
        """
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        model_kind = model_kind_for(_KIND, _PREDICT_MODEL, params.model)
        model = _resolved_model(ds, params.model, model_kind)
        if model_kind == _SLEAP_KIND:
            _refuse_sleap_heads(params.model, model.model_types)
        elif model_kind == _LITPOSE_KIND:
            _refuse_single_frame_type(params.model, model.model_type)
        recorded = _recorded_pose(ds, model_kind, params.model)
        extraction = _extraction(ds, ctx, params.frames_run_id, scope)
        if not extraction.entries:
            unreadable = extraction.unreadable
            message = (
                f"[{_KIND}] the frames of every one of {len(unreadable)} scoped "
                f"entries could not be read: {', '.join(unreadable)}. The run "
                f"run_id={run_id} predicted nothing. The per-entry errors are in "
                "this attempt's run-log."
            )
            raise AllEntriesFailed(message)
        tool = _frame_tool(model_kind, model, recorded, ctx, images_root=ds.base_dir)
        _refuse_another_count(recorded, tool.probe, params.model)

        run_root = prediction_kind_root(ds, _KIND) / run_id
        payload = predict_frames_payload(
            params, model_id=model.model_id, model_kind=model_kind
        )
        write_run_params(run_root, payload)
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)
        facts = _RunFacts(
            run_id=run_id,
            frames_run_id=params.frames_run_id,
            model=json.dumps(list(params.model)),
            model_kind=model_kind,
            model_id=model.model_id,
            tool_version=tool.probe.tool_version,
            params_hash=hash_params(payload),
        )
        index = predict_frames_index(prediction_index_path(ds, _KIND))
        finished = _finished_files(ds, run_id)
        total = len(extraction.entries)
        ctx.set_total(total)

        done = 0
        # An entry whose images could not be read was attempted and lost. An
        # entry that another execution holds is not this run's to lose. A
        # reused entry is attempted, so `entries_written` counts it, as the
        # run-log counts a cache hit's entries.
        attempted: set[str] = set(extraction.unreadable)
        lost: set[str] = set(extraction.unreadable)
        reused: set[str] = set()
        held: list[_HeldEntry] = []
        work_dir = run_root / f"{_WORK_PREFIX}{ctx.execution_id}"
        try:
            for entry in extraction.entries:
                ctx.check_cancel()
                opened = open_entry(
                    ds,
                    ctx,
                    run_root,
                    entry.key,
                    kind=_KIND,
                    overwrite=overwrite,
                    idle_seconds=_PREDICT_IDLE_SECONDS,
                )
                if opened is None:
                    done += 1
                    _entry_done(ctx, done, total, entry.key)
                    continue
                seq_dir, marker = opened
                attempted.add(entry.key)
                if not overwrite and _is_written(ds, finished, entry):
                    release_entry(seq_dir, ctx.execution_id)
                    reused.add(entry.key)
                    done += 1
                    _entry_done(ctx, done, total, entry.key)
                    continue
                activity = phase_activity(ctx, seq_dir, marker, _PREDICT_IDLE_SECONDS)
                held.append(
                    _HeldEntry(frames=entry, seq_dir=seq_dir, activity=activity)
                )

            prediction = (
                _predict_held(
                    ds, ctx, run_id, params, model, tool, recorded, held, work_dir
                )
                if held
                else None
            )
            if prediction is None:
                lost.update(entry.frames.key for entry in held)
            position = 0
            for entry in held:
                count = len(entry.frames.files)
                if prediction is not None:
                    found = prediction.objects[position : position + count]
                    try:
                        rows = _write_entry(
                            ds, run_root, facts, entry.frames, prediction.pose, found
                        )
                        index.append(rows)
                        index.mark_finished(run_id)
                    except Cancelled:
                        raise
                    except Exception as exc:
                        ctx.entry_failed_unless_cancelled(entry.frames.key, exc)
                        lost.add(entry.frames.key)
                position += count
                done += 1
                _entry_done(ctx, done, total, entry.frames.key)
        finally:
            for entry in held:
                release_entry(entry.seq_dir, ctx.execution_id)
                discard_if_empty(entry.seq_dir)
            shutil.rmtree(work_dir, ignore_errors=True)
            # Reused entries count. A rerun and a first run over one scope then
            # report the same coverage.
            ctx.entries_written(len(attempted) - len(lost))

        if extraction.claimable and len(reused) == extraction.claimable:
            ctx.cache_hit()
        if attempted and lost == attempted:
            message = (
                f"[{_KIND}] every one of {len(attempted)} attempted entries failed: "
                f"{', '.join(sorted(lost))}. The run run_id={run_id} wrote no "
                "predictions. The per-entry errors are in this attempt's run-log."
            )
            raise AllEntriesFailed(message)
        print(
            f"[{_KIND}] completed run_id={run_id}: "
            f"{len(attempted) - len(lost)} of {extraction.claimable} entries have "
            f"predictions, {len(reused)} reused -> {run_root}"
        )
        return run_id
