"""
FeralFeature -- FERAL vision-transformer behavior classifier as a Mosaic pipeline feature.

A global feature, like ``xgboost`` and ``lightning-action``: it fits a classifier
over labeled entries, then applies it per entry. Exactly one of three params
chooses the mode.

Training mode
-------------
Provide ``templates``, ``labels`` and ``training``. The ``templates`` run
(``extract-labeled-templates``) names the train entries the model fits on and
the test entries it is evaluated on. ``labels`` supplies each frame's
ground-truth label, and ``unlabeled_as`` the class of a frame with none. A
seeded ``training.val_fraction`` of the train entries validates each epoch and
chooses the best checkpoint. The run writes ``feral_labels.json``,
``config.json``, ``model_best.pt``, ``summary.csv``, ``reports.json`` and the
model file ``feral_model.joblib``.

Applying a trained run
----------------------
Provide ``model``, a trained ``feral`` run, whose ``feral_model.joblib`` the
step applies without fitting.

Importing a checkpoint
----------------------
Provide ``model_dir``, a directory holding a checkpoint trained outside mosaic:
``model_best.pt`` and, optionally, ``config.json``.

Where the videos come from
--------------------------
The step's input decides it. An ``interaction-crop-pipeline`` or
``egocentric-crop`` input supplies the crop videos under its run root. Any other
input, ``tracks`` included, is read through the entry's own media.

Output is per-frame rows with ``prob_<class>`` probability columns and a
``predicted_label`` column.

Requires the FERAL package (``pip install 'mosaic-behavior[feral]'``).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    final,
)

import numpy as np
import pandas as pd
from mosaic_media import MediaProbeError
from numpy.typing import NDArray
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from mosaic.core.helpers import UNLABELED
from mosaic.core.label_intervals import label_vocabulary
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.pipeline.column_contract import (
    AnyOf,
    Column,
    Conditional,
    Prefix,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
    DependencyLookup,
    InputRequire,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    LabeledTemplatesRef,
    Result,
    TrackInput,
)
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    DeclaredModel,
    Params,
)

from mosaic.core.pipeline.progress import CSVProgressCallback
from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.core.pipeline.storage_names import feature_slug_of
from mosaic.optional_dependency import (
    MissingOptionalDependency,
    OptionalModule,
    Prerequisite,
)

from .classifier_common import (
    FIT_SCOPE_REMEDY,
    input_run_root,
    refuse_entries_outside_scope,
)
from .helpers import entries_of_templates, write_bundle
from .named_model import named_model_file
from .registry import register_feature

if TYPE_CHECKING:
    import torch

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import TrainingProgressCallback

log = logging.getLogger(__name__)

FERAL: Final = OptionalModule("feral", "feral", "the FERAL behavior classifier")
"""The installed FERAL package, as the pipeline catalog checks for it."""


def _check_feral() -> None:
    """Raise a helpful refusal if the ``feral`` package is not importable.

    Lightweight -- importing the ``feral`` package does not pull torch (its
    public API is lazily loaded).

    Raises:
        ToolUnavailable: The package is not importable.
    """
    try:
        _ = FERAL.load()
    except MissingOptionalDependency as missing:
        raise ToolUnavailable("feral", "environment", str(missing)) from missing


def _resolve_backbone_key(value: str) -> str:
    """Translate a FERAL backbone identifier into a ``feral.backbones.BACKBONES`` key.

    Accepts either a BACKBONES key (returned as-is) or a HuggingFace slug such as
    ``"facebook/vjepa2-vitl-fpc32-256-diving48"`` (reverse-looked-up to its key,
    e.g. ``"vjepa2_vitl_diving48"``). Mosaic's saved configs store the old
    ``model_name`` HF slug, so this keeps those runs loading unchanged.
    """
    from feral.backbones import BACKBONES

    if value in BACKBONES:
        return value
    for key, entry in BACKBONES.items():
        if entry.get("slug") == value:
            return key
    raise ValueError(
        f"Unknown FERAL backbone/model_name {value!r}. Pass a BACKBONES key "
        f"({sorted(BACKBONES)}) or a known HuggingFace slug."
    )


_STATE_DICT: Final = TypeAdapter(dict[str, object])
"""A checkpoint's weights: tensors by parameter name."""


def _load_checkpoint_state_dict(
    raw: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Normalize a ``torch.load`` blob into ``(state_dict, metadata | None)``.

    Handles every checkpoint shape FeralFeature may encounter:

    * **New dict format** (``{'state_dict': ..., 'class_names': ..., ...}``,
      written by FERAL's ``save_model``) -- unwraps ``state_dict`` and returns the
      embedded ``class_names`` / ``is_multilabel`` / ``cfg`` as ``metadata``.
    * **Bare state_dict** (legacy save) -- used directly.
    * **Old ``HFModel`` layout** -- keys prefixed ``model.*`` (the encoder lived at
      ``HFModel.self.model``) are remapped to ``backbone.model.*`` so they load
      into the new ``FeralModel`` (whose encoder lives at
      ``FeralModel.self.backbone.model``). The classifier keys
      (``clip_projector.*``, ``fc_norm.*``, ``head.*``) are unchanged.
    """
    meta: dict[str, object] | None = None
    if "state_dict" in raw:
        state_dict = _STATE_DICT.validate_python(raw["state_dict"])
        meta = {k: raw[k] for k in ("class_names", "is_multilabel", "cfg") if k in raw}
    else:
        state_dict = dict(raw)
    is_old_hfmodel = any(k.startswith("model.") for k in state_dict) and not any(
        k.startswith("backbone.") for k in state_dict
    )
    if is_old_hfmodel:
        state_dict = {
            ("backbone." + k if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict, meta


def _load_into(model: torch.nn.Module, state_dict: Mapping[str, object]) -> None:
    """Load ``state_dict`` into ``model`` and fail loudly on any key mismatch.

    Uses ``strict=False`` then asserts there are no missing/unexpected keys
    (ignoring ``num_batches_tracked`` buffers). This converts a silent
    architecture-drift mismatch -- which would otherwise leave the backbone at
    random initialization -- into an actionable error.
    """
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if "num_batches_tracked" not in k]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "FERAL checkpoint does not match FeralModel "
            f"(missing={missing[:8]}, unexpected={list(result.unexpected_keys)[:8]}). "
            "The checkpoint was likely trained with a different backbone or "
            "number of classes."
        )


def _chunked[T](seq: Sequence[T], k: int) -> Iterator[Sequence[T]]:
    """Yield successive k-sized sublists of ``seq`` (last may be shorter)."""
    for i in range(0, len(seq), k):
        yield seq[i : i + k]


# --- The labels dict and the validation split -------------------------------

FeralSplit = Literal["train", "val", "test"]
"""The partitions of FERAL's labels dict that mosaic fills."""

FeralDropReason = Literal[
    "no_label_file", "too_short", "unlabeled_frames", "unseen_class"
]
"""Why :func:`build_feral_labels` leaves a video out of its split."""

_DROP_DESCRIPTIONS: Final[Mapping[FeralDropReason, str]] = {
    "no_label_file": "have no label file",
    "too_short": "are shorter than the minimum clip length",
    "unlabeled_frames": "hold unlabeled frames and unlabeled_as is unset",
    "unseen_class": "hold a label that no training video holds",
}


@dataclass(frozen=True)
class FeralVideo:
    """One video FERAL reads.

    Attributes:
        entry: The ``(group, sequence)`` the video belongs to.
        filename: The video's key in FERAL's labels dict.
        path: The absolute path of the video file.
        frames: The frame of each video frame, in video order.
        ids: The identity columns the video's predictions take: ``id1``,
            ``id2``, ``perspective`` and ``interaction_id`` for an interaction
            clip, ``id`` for an egocentric video, and none for a sequence video.
    """

    entry: tuple[str, str]
    filename: str
    path: Path
    frames: tuple[int, ...]
    ids: Mapping[str, int | str]


class FeralLabelsJson(TypedDict):
    """FERAL's labels dict for single-label classification."""

    class_names: dict[str, str]
    is_multilabel: bool
    labels: dict[str, list[int]]
    splits: dict[str, list[str]]


@dataclass(frozen=True)
class FeralLabels:
    """FERAL's labels dict, with the mapping from its classes to label values.

    Attributes:
        labels_json: FERAL's dict: ``class_names``, ``is_multilabel``,
            ``labels`` keyed by video filename, and ``splits``.
        class_values: The label value of each dense class index.
        dropped: The count of videos left out of their split, by reason.
    """

    labels_json: FeralLabelsJson
    class_values: tuple[int, ...]
    dropped: Mapping[FeralDropReason, int]


def _frame_table(video: FeralVideo) -> pd.DataFrame:
    """The ``frame`` column of *video* beside its identity columns."""
    table = pd.DataFrame({C.frame_col: np.asarray(video.frames, dtype=np.int64)})
    for column, value in video.ids.items():
        table[column] = value
    return table


def _train_vocabulary(
    entries: Sequence[tuple[str, str]], label_files: Mapping[tuple[str, str], Path]
) -> dict[int, str]:
    """The names the label files of *entries* give their label values."""
    names: dict[int, str] = {}
    for entry in entries:
        with np.load(label_files[entry], allow_pickle=True) as archive:
            for value, name in label_vocabulary(archive).items():
                _ = names.setdefault(value, name)
    return names


def build_feral_labels(
    videos: Mapping[FeralSplit, Sequence[FeralVideo]],
    label_files: Mapping[tuple[str, str], Path],
    labels: GroundTruthLabelsSource,
    *,
    unlabeled_as: int | None,
    min_frames: int,
) -> FeralLabels:
    """FERAL's labels dict for the videos of each split.

    Each video's frames are labeled from the label file of its entry
    (:func:`~mosaic.core.pipeline.loading.labels_for_rows`), so a sequence video
    aligns by frame alone and an interaction or egocentric video by its own
    individuals. A frame with no label, including a masked one, takes
    ``unlabeled_as``, or its video leaves the split when that is ``None``. A
    video leaves its split too when it has fewer than ``min_frames`` frames, its
    entry has no label file, or, in the validation and test splits, it holds a
    value that no training video holds. The sorted values left in the training
    split map to the dense class indices FERAL needs.

    Args:
        videos: The videos of each split.
        label_files: The label file of each entry.
        labels: The labels source the files were resolved for.
        unlabeled_as: The label value a frame without a label takes, or
            ``None`` to drop the video that holds one.
        min_frames: The fewest frames a video may have.

    Returns:
        The labels dict, the label value of each class index, and the counts of
        dropped videos.

    Raises:
        ValueError: No training video is left.
    """
    dropped: dict[FeralDropReason, int] = {
        "no_label_file": 0,
        "too_short": 0,
        "unlabeled_frames": 0,
        "unseen_class": 0,
    }
    labeled: dict[FeralSplit, list[tuple[FeralVideo, NDArray[np.int64]]]] = {}
    for split, split_videos in videos.items():
        labeled[split] = []
        for video in split_videos:
            label_file = label_files.get(video.entry)
            if label_file is None:
                dropped["no_label_file"] += 1
                continue
            if len(video.frames) < min_frames:
                dropped["too_short"] += 1
                continue
            values = labels_for_rows(label_file, _frame_table(video), labels)
            if unlabeled_as is not None:
                values = np.where(values == UNLABELED, unlabeled_as, values)
            elif bool((values == UNLABELED).any()):
                dropped["unlabeled_frames"] += 1
                continue
            labeled[split].append((video, values))

    train = labeled.get("train", [])
    if not train:
        raise ValueError(
            f"{FeralFeature.name}: no training video is left to train on "
            f"({_dropped_summary(dropped)})"
        )
    class_values = tuple(
        int(value) for value in np.unique(np.concatenate([v for _, v in train]))
    )
    for split in ("val", "test"):
        kept = [
            (video, values)
            for video, values in labeled.get(split, [])
            if bool(np.isin(values, class_values).all())
        ]
        dropped["unseen_class"] += len(labeled.get(split, [])) - len(kept)
        labeled[split] = kept

    for reason, count in dropped.items():
        if count:
            log.warning(
                "[feral] %d video(s) left their split: they %s",
                count,
                _DROP_DESCRIPTIONS[reason],
            )

    dense = {value: index for index, value in enumerate(class_values)}
    names = _train_vocabulary(sorted({video.entry for video, _ in train}), label_files)
    class_names = {
        str(index): names.get(value, str(value)) for value, index in dense.items()
    }
    label_lists: dict[str, list[int]] = {}
    splits: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        splits[split] = [video.filename for video, _ in labeled.get(split, [])]
        for video, values in labeled.get(split, []):
            remapped = np.vectorize(dense.__getitem__, otypes=[np.int64])(values)
            label_lists[video.filename] = remapped.tolist()
    labels_json: FeralLabelsJson = {
        "class_names": class_names,
        "is_multilabel": False,
        "labels": label_lists,
        "splits": splits,
    }
    return FeralLabels(
        labels_json=labels_json, class_values=class_values, dropped=dropped
    )


def _dropped_summary(dropped: Mapping[FeralDropReason, int]) -> str:
    """The nonzero counts of *dropped*, spelled for a refusal."""
    parts = [
        f"{count} {reason.replace('_', ' ')}"
        for reason, count in dropped.items()
        if count
    ]
    return ", ".join(parts) if parts else "none was given"


def validation_entries(
    train_entries: Sequence[tuple[str, str]], fraction: float, seed: int
) -> list[tuple[str, str]]:
    """The train entries drawn as the validation split.

    Draws ``round(fraction * n)`` of the sorted entries with
    ``np.random.default_rng(seed)``, so the same seed gives the same entries
    for any input order.

    Args:
        train_entries: The train entries.
        fraction: The fraction of the entries to draw.
        seed: The seed of the draw.

    Returns:
        The drawn entries, sorted. Empty when the count rounds to zero.

    Raises:
        ValueError: The draw leaves no train entry.
    """
    ordered = sorted(train_entries)
    count = round(fraction * len(ordered))
    if count >= len(ordered) and ordered:
        raise ValueError(
            f"{FeralFeature.name}: a validation fraction of {fraction} draws "
            f"{count} of {len(ordered)} train entries and leaves no train entry"
        )
    if count == 0:
        return []
    chosen = np.random.default_rng(seed).choice(len(ordered), size=count, replace=False)
    return [ordered[index] for index in sorted(int(i) for i in chosen)]


# --- Where the videos come from ---------------------------------------------

FeralCropMode = Literal["interaction", "egocentric"]
"""The crop inputs whose run root holds the videos FERAL reads."""

FeralInputMode = Literal["interaction", "egocentric", "sequence"]
"""Where FERAL's videos come from: a crop run's videos, or the entry's own media."""

_CROP_MODES: Final[Mapping[str, FeralCropMode]] = {
    "interaction-crop-pipeline": "interaction",
    "egocentric-crop": "egocentric",
}


def feral_input_mode(storage_name: str) -> FeralInputMode:
    """The video source an input's storage name selects.

    The input's feature slug, which a shortened storage name keeps, picks the
    source. The interaction and egocentric crop features supply their own
    videos, and any other input, a track input included, is read through the
    entry's own media.

    Args:
        storage_name: The storage name of the input, or ``"tracks"``.

    Returns:
        ``"interaction"``, ``"egocentric"`` or ``"sequence"``.
    """
    return _CROP_MODES.get(feature_slug_of(storage_name), "sequence")


def refuse_crop_without_videos(input_root: Path) -> None:
    """Refuse a crop run that wrote no video under its run root.

    A crop run that set ``output_root`` writes its videos there, and an
    egocentric run with ``output_mode="frames"`` writes no video at all. A run
    whose ``params.json`` cannot be read has an unknown output mode, so it is
    refused too.

    Args:
        input_root: The crop run's root.

    Raises:
        ValueError: The run has no videos under *input_root*, or its params
            do not say whether it has.
    """
    run = f"{input_root.parent.name}/{input_root.name}"
    read = read_run_params(input_root)
    if read.params is None:
        raise ValueError(
            f"{FeralFeature.name}: cannot tell whether crop run {run} wrote videos: "
            f"{read.finding}"
        )
    params = read.params.params
    if params.get("output_root"):
        raise ValueError(
            f"{FeralFeature.name}: crop run {run} set output_root, so no video is "
            "under its run root"
        )
    if params.get("output_mode") == "frames":
        raise ValueError(
            f"{FeralFeature.name}: crop run {run} wrote frames only "
            '(output_mode="frames"), so it holds no video'
        )


def _scalar(value: object) -> int | str:
    """An identity cell as the ``int`` or ``str`` a video's ids hold."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _entry_of(rows: pd.DataFrame) -> tuple[str, str]:
    """The one ``(group, sequence)`` that *rows* belong to."""
    entries = {
        (str(group), str(sequence))
        for group, sequence in zip(rows["group"], rows["sequence"], strict=True)
    }
    if len(entries) != 1:
        raise ValueError(
            f"{FeralFeature.name}: one entry's crop rows span {len(entries)} entries"
        )
    return next(iter(entries))


def _video_filename(path: Path, dataset_root: Path) -> str:
    """*path* relative to the dataset root, or absolute outside it.

    This is FERAL's key for a video, and FERAL joins it to the dataset root. It
    is not :meth:`Dataset.relative_to_root`, which resolves symlinks and writes
    native separators, for two reasons. A crop's videos are named from a run
    root, where no :class:`Dataset` is at hand, so one rule serves crop and
    sequence videos alike. And a key that follows the path as the dataset spells
    it stays the same for a media file that is a link to storage elsewhere,
    where a resolved path would change with the link's target and the labels
    JSON written into the run root would differ between machines. The separator
    is ``/`` on every platform for the same reason.
    """
    for root in (dataset_root, dataset_root.resolve()):
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
    return str(path)


def crop_videos(
    mode: FeralCropMode, input_root: Path, rows: pd.DataFrame, dataset_root: Path
) -> tuple[list[FeralVideo], int]:
    """The videos of one entry's crop rows.

    An interaction row is one clip, ``<input_root>/<video_path>``, over frames
    ``start_frame..end_frame``. The crop skips a frame whose focal individual is
    missing, so a clip whose ``n_frames`` is not ``end_frame - start_frame + 1``
    has no frame list FERAL can align and is dropped with a logged count. An
    egocentric id is one video,
    ``<input_root>/<group>__<sequence>/egocentric_id<id>.mp4``, that holds only
    the frames with a row, so its frames are the sorted ``frame`` values of its
    rows.

    Args:
        mode: The crop that wrote the rows.
        input_root: The crop run's root.
        rows: The crop rows of one entry.
        dataset_root: The dataset root, which FERAL joins to each filename.

    Returns:
        The videos, in row order for an interaction and id order for an
        egocentric crop, and the count of clips dropped.
    """
    if rows.empty:
        return [], 0
    entry = _entry_of(rows)
    videos: list[FeralVideo] = []
    dropped = 0
    if mode == "interaction":
        for _, row in rows.iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            if int(row["n_frames"]) != end - start + 1:
                dropped += 1
                continue
            path = input_root / str(row["video_path"])
            ids = {
                column: _scalar(row[column])
                for column in ("id1", "id2", "perspective", "interaction_id")
            }
            videos.append(
                FeralVideo(
                    entry=entry,
                    filename=_video_filename(path, dataset_root),
                    path=path,
                    frames=tuple(range(start, end + 1)),
                    ids=ids,
                )
            )
        if dropped:
            log.warning(
                "[feral] %d interaction clip(s) of %s/%s skip frames and are dropped",
                dropped,
                *entry,
            )
        return videos, dropped
    for target_id, group_rows in rows.groupby(C.id_col, sort=True):
        path = input_root / f"{entry[0]}__{entry[1]}" / f"egocentric_id{target_id}.mp4"
        frames = sorted(int(frame) for frame in group_rows[C.frame_col])
        videos.append(
            FeralVideo(
                entry=entry,
                filename=_video_filename(path, dataset_root),
                path=path,
                frames=tuple(frames),
                ids={C.id_col: _scalar(target_id)},
            )
        )
    return videos, dropped


def sequence_video(
    ds: Dataset,
    group: str,
    sequence: str,
    frame_count: Callable[[str], int | None],
) -> FeralVideo:
    """The video of one entry's own media.

    The entry must resolve to one plain file. Its frames are the sequence's
    first frame on its recording's frame axis plus ``k`` for each frame FERAL
    counts in the video, and that count must match the stored count when there
    is one.

    Args:
        ds: The dataset that resolves the entry's media.
        group: The entry's group.
        sequence: The entry's sequence.
        frame_count: FERAL's frame count of the file at a path, ``None`` when
            OpenCV cannot count it.

    Returns:
        The entry's video.

    Raises:
        ValueError: The entry spans several cameras, files or a store window,
            cannot be placed, has a missing video or one OpenCV cannot count,
            or has a count that disagrees with the stored one.
    """
    entry = (group, sequence)
    try:
        media = ds.resolve_media(group, sequence)
        first_frame = media.first_frame
    except MediaProbeError as exc:
        raise ValueError(
            f"{FeralFeature.name}: entry {entry!r} has no video FERAL can read: {exc}"
        ) from exc
    if len(media.paths) != 1:
        raise ValueError(
            f"{FeralFeature.name}: entry {entry!r} has {len(media.paths)} media files, "
            "and FERAL reads one video"
        )
    if media.windows[0] is not None:
        raise ValueError(
            f"{FeralFeature.name}: entry {entry!r} reads a window of a store, "
            "not a video file FERAL can open"
        )
    path = media.paths[0]
    try:
        counted = frame_count(str(path))
    except FileNotFoundError as exc:
        raise ValueError(
            f"{FeralFeature.name}: the video of entry {entry!r} is missing: {path}"
        ) from exc
    if counted is None:
        raise ValueError(
            f"{FeralFeature.name}: the video of entry {entry!r} is unreadable: "
            f"OpenCV cannot count the frames of {path}"
        )
    stored = media.frame_count
    if stored is not None and stored != counted:
        raise ValueError(
            f"{FeralFeature.name}: the video of entry {entry!r} has {counted} frames "
            f"by FERAL's count and {stored} in the media index"
        )
    return FeralVideo(
        entry=entry,
        filename=_video_filename(path, ds.base_dir),
        path=path,
        frames=tuple(first_frame + k for k in range(counted)),
        ids={},
    )


# --- The model file ---------------------------------------------------------


class FeralModelConfig(TypedDict):
    """The ``model`` block of FERAL's config.

    The first two keys shape the network. A run trained in mosaic holds the
    others too, and a checkpoint imported from outside mosaic need not.
    """

    fc_drop_rate: float
    freeze_encoder_layers: int
    class_weights: NotRequired[str]
    gradient_checkpointing: NotRequired[bool]
    max_class_weight: NotRequired[float | None]


class FeralNetwork(TypedDict):
    """The keys of FERAL's config that build the network.

    A trained run holds the whole config, which has these keys too. A checkpoint
    imported from outside mosaic holds them and no more.
    """

    backbone: str
    predict_per_item: int
    model: FeralModelConfig


class FeralModelBundle(TypedDict):
    """What ``feral_model.joblib`` holds."""

    state_dict: dict[str, torch.Tensor]  # CPU tensors, unwrapped from torch.compile
    class_values: list[int]  # the original label value of each dense class index
    class_names: dict[int, str]  # original label value to name
    is_multilabel: bool
    cfg: FeralNetwork  # the FERAL config the run trained with


class FeralModelArtifact(JoblibArtifact[FeralModelBundle]):
    """A trained FERAL model."""

    feature: str = "feral"
    pattern: str = "feral_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


# --- Params ---------------------------------------------------------------

_TEMPLATES_DESCRIPTION = (
    "The extract-labeled-templates run whose train entries the model fits on and "
    "whose test entries it evaluates on. Only its template_provenance.parquet is "
    "read: the model trains on the videos of those entries, not on the sampled "
    "templates. An entry the run does not list never trains or tests. The run "
    "names the training set, so the step's scope is not part of its identity: a "
    "wider scope applies the fit it has and does not fit again. Exactly one of "
    "templates, model or model_dir."
)

_LABELS_DESCRIPTION = (
    "The per-frame ground-truth labels that the frames of the videos are aligned "
    "to, resolved from labels/<kind>/index.csv. A frame no label covers takes "
    "unlabeled_as, or its video is left out when that is unset. Required with "
    "templates."
)

_UNLABELED_AS_DESCRIPTION = (
    "The label value, in the original label space, that a frame without a label "
    "trains as, a masked class included. Unset, a video that holds such a frame "
    "is left out of its split. Only with templates."
)

_MODEL_DESCRIPTION = (
    "A trained feral run to apply in place of training: its feral_model.joblib. "
    "Exactly one of templates, model or model_dir."
)

_MODEL_NAME_DESCRIPTION = (
    "The FERAL backbone: a feral.backbones.BACKBONES key or the matching "
    "HuggingFace slug. An unrecognized value raises. The choice also "
    "selects the pretrained weights FERAL downloads from the HuggingFace "
    "hub on first use, whose license differs per entry; see NOTICE. Ignored "
    "with model, whose trained run records its own. With model_dir this is a "
    "fallback: a loaded config.json or checkpoint that records its own "
    "backbone overrides it."
)

_PREDICT_PER_ITEM_DESCRIPTION = (
    "How many predictions the model emits per chunk. Ignored with model, whose "
    "trained run records its own. With model_dir this is a fallback: a loaded "
    "config.json or checkpoint that records its own value overrides it."
)

_CHUNK_LENGTH_DESCRIPTION = "The length of one video chunk."

_CHUNK_SHIFT_DESCRIPTION = "The stride between the starts of consecutive chunks."

_CHUNK_STEP_DESCRIPTION = "The stride sampled within one chunk."

_RESIZE_TO_DESCRIPTION = "The side a frame is resized to before the model reads it."

_DEVICE_DESCRIPTION = "Which PyTorch device loads and runs the model."

_INFER_BATCH_SIZE_DESCRIPTION = (
    "How many chunks apply() reads in one forward pass. Batching is "
    "numerically equivalent to reading them one at a time."
)

_INFERENCE_AUTOCAST_DESCRIPTION = (
    "Run apply()'s forward pass under bfloat16 autocast instead of full "
    "float32, roughly 1.5-2x faster at the cost of a recall shift of "
    "around 10 percentage points on threshold-sensitive classes. Read "
    "only when the model is on a CUDA device."
)

_CLASS_NAMES_DESCRIPTION = (
    "Class names, keyed by the original label value as text. Unset, the names "
    "a trained run recorded, or an imported checkpoint's metadata, are used. "
    "An imported checkpoint with neither needs this to name its classes."
)

_DECISION_THRESHOLD_DESCRIPTION = (
    "The minimum probability considered when choosing the predicted "
    "class. Unset, the predicted class is the argmax over every class. "
    "Set, the predicted class is default_class when every class falls "
    "below it."
)

_DEFAULT_CLASS_DESCRIPTION = (
    "The label value returned when decision_threshold is set and every class "
    "falls below it."
)

_MODEL_DIR_DESCRIPTION = (
    "A directory that holds a checkpoint trained outside mosaic: model_best.pt "
    "and, optionally, config.json. Exactly one of templates, model or model_dir."
)

_TRAINING_DESCRIPTION = (
    "Training hyperparameters. Required with templates, and refused without it."
)


_EPOCHS_DESCRIPTION = "The number of training epochs."

_VAL_FRACTION_DESCRIPTION = (
    "The fraction of the train entries of the templates run, drawn whole with "
    "the seed, that validate each epoch. The best epoch by validation frame-level "
    "mAP is kept. 0 trains without validation and keeps the last epoch."
)

_TRAIN_BS_DESCRIPTION = "The number of samples per training batch."

_VAL_BS_DESCRIPTION = (
    "The number of samples per validation batch, also used when evaluating "
    "a test partition after training."
)

_NUM_WORKERS_DESCRIPTION = (
    "The number of subprocess workers for the training and validation data "
    "loaders, also used when evaluating a test partition after training."
)

_LR_DESCRIPTION = "The learning rate for the AdamW optimizer."

_WEIGHT_DECAY_DESCRIPTION = "The weight decay for the AdamW optimizer."

_LABEL_SMOOTHING_DESCRIPTION = "The label smoothing factor for the cross-entropy loss."

_FC_DROP_RATE_DESCRIPTION = "The dropout rate before the classifier head."

_FREEZE_ENCODER_LAYERS_DESCRIPTION = (
    "The number of backbone encoder layers frozen during fine-tuning."
)

_CLASS_WEIGHTS_DESCRIPTION = (
    "The class-weighting scheme forwarded to FERAL's get_weights when "
    "building the loss weight tensor."
)

_EMA_DECAY_DESCRIPTION = (
    "The decay rate for an exponential moving average of the model "
    "weights. Unset, no EMA model is tracked."
)

_MIXUP_ALPHA_DESCRIPTION = (
    "The alpha parameter for torchvision's MixUp training-batch "
    "augmentation. Unset, no MixUp is applied."
)

_PART_WARMUP_DESCRIPTION = (
    "The fraction of total training steps spent in the cosine "
    "learning-rate schedule's warmup phase."
)

_PATIENCE_DESCRIPTION = (
    "The number of epochs without validation improvement before training "
    "stops early. Unset, training never stops early."
)

_COMPILE_DESCRIPTION = "Compile the model with torch.compile before training."

_DO_AA_DESCRIPTION = "Apply data augmentation when building the training dataset."

_SEED_DESCRIPTION = "The random seed applied to torch and numpy before training starts."

_PART_SAMPLE_DESCRIPTION = (
    "The fraction of the training partition included when building the "
    "training dataset."
)

_MULTILABEL_THRESHOLD_DESCRIPTION = (
    "The probability threshold for the F1 metrics of a multilabel model. A "
    "templates run trains a single-label model, so it never applies to one."
)


class FeralTrainingConfig(DeclaredModel):
    """Training hyperparameters for FERAL ViT fine-tuning.

    These mirror the FERAL default_vjepa.yaml configuration.
    """

    epochs: Annotated[int, Declared(_EPOCHS_DESCRIPTION)] = Field(default=10, ge=1)
    val_fraction: Annotated[float, Declared(_VAL_FRACTION_DESCRIPTION)] = Field(
        default=0.2, ge=0, lt=1
    )
    train_bs: Annotated[int, Declared(_TRAIN_BS_DESCRIPTION)] = Field(default=4, ge=1)
    val_bs: Annotated[int, Declared(_VAL_BS_DESCRIPTION)] = Field(default=8, ge=1)
    num_workers: Annotated[int, Declared(_NUM_WORKERS_DESCRIPTION)] = Field(
        default=4, ge=0
    )
    lr: Annotated[float, Declared(_LR_DESCRIPTION)] = Field(default=4e-5, gt=0)
    weight_decay: Annotated[float, Declared(_WEIGHT_DECAY_DESCRIPTION)] = Field(
        default=0.1, ge=0
    )
    label_smoothing: Annotated[float, Declared(_LABEL_SMOOTHING_DESCRIPTION)] = Field(
        default=0.1, ge=0, le=1
    )
    fc_drop_rate: Annotated[float, Declared(_FC_DROP_RATE_DESCRIPTION)] = Field(
        default=0.5, ge=0, le=1
    )
    freeze_encoder_layers: Annotated[
        int, Declared(_FREEZE_ENCODER_LAYERS_DESCRIPTION)
    ] = Field(default=14, ge=0)
    class_weights: Annotated[str, Declared(_CLASS_WEIGHTS_DESCRIPTION)] = (
        "inv_freq_sqrt"
    )
    ema_decay: Annotated[float | None, Declared(_EMA_DECAY_DESCRIPTION)] = 0.999
    mixup_alpha: Annotated[float | None, Declared(_MIXUP_ALPHA_DESCRIPTION)] = 0.8
    part_warmup: Annotated[float, Declared(_PART_WARMUP_DESCRIPTION)] = Field(
        default=0.2, ge=0, le=1
    )
    patience: Annotated[int | None, Declared(_PATIENCE_DESCRIPTION)] = None
    compile: Annotated[bool, Declared(_COMPILE_DESCRIPTION)] = True
    do_aa: Annotated[bool, Declared(_DO_AA_DESCRIPTION)] = True
    seed: Annotated[int, Declared(_SEED_DESCRIPTION)] = 0
    part_sample: Annotated[float, Declared(_PART_SAMPLE_DESCRIPTION)] = Field(
        default=1.0, gt=0, le=1
    )
    multilabel_threshold: Annotated[
        float, Declared(_MULTILABEL_THRESHOLD_DESCRIPTION)
    ] = Field(default=0.85, ge=0, le=1)


@final
@register_feature
class FeralFeature:
    """FERAL vision-transformer behavior classifier as a pipeline feature.

    Exactly one of three params chooses the mode.

    **Training mode** (``templates`` + ``labels`` + ``training``):
        Trains FERAL's ViT on the videos of the entries that the ``templates``
        run (``extract-labeled-templates``) assigned to ``train``, with each
        frame labeled from ``labels``. A seeded ``training.val_fraction`` of
        those entries validates each epoch and picks the best checkpoint. The
        test entries are evaluated afterwards. The ``templates`` run names the
        training set, so the run id does not cover the step's scope: a wider
        scope applies the stored model to the new entries and trains nothing
        again. The first run has to be asked for over a scope that holds the
        train entries and the labeled test entries.

    **Applying a trained run** (``model``):
        Loads the ``feral_model.joblib`` of a trained ``feral`` run and
        classifies each frame of its input.

    **Importing a checkpoint** (``model_dir``):
        Loads ``model_best.pt``, and ``config.json`` when there is one, of a
        checkpoint trained outside mosaic.

    **Inputs**:
        The step's input decides where FERAL reads its videos from:

        1. An **interaction-crop-pipeline** input (pair-level): one row per crop
           video, read from the crop run's root.
        2. An **egocentric-crop** input (individual-level): one video per
           individual under the crop run's root.
        3. Any other input, ``tracks`` included: the entry's own media, which
           must be one plain video file.

    **Precision convention**:
        Training's forward/backward pass and the per-epoch validation
        inside ``fit()`` run in FERAL's bfloat16 autocast, so the
        validation numbers that pick the best epoch follow FERAL's own.
        Everything else -- the post-training test eval
        (``_evaluate_partition``), the public ``FeralFeature.evaluate()``,
        and production inference (``apply()``) -- defaults to full
        float32. The eval and apply paths produce numbers that get stored
        and acted on downstream (reports.json metrics, per-frame
        prediction parquets, paper figures), so reproducibility wins over
        the ~1.5-2x speedup that bfloat16 would buy. bfloat16 shifts
        decision boundaries near probability 0.5 and can move recall by
        10+ points on threshold-sensitive classes -- not a trade-off worth
        taking when the cost is a one-time inference pass.

        **Opt-in override for apply()**: set
        ``params.inference_autocast = True`` to run apply()'s forward
        pass in bfloat16 autocast. Trades the ~10pp recall shift for the
        speedup; useful when wall-clock dominates (e.g., dense crop
        pipelines with chunk_shift << chunk_length) and per-frame
        argmax precision can be re-validated downstream. The flag is
        persisted in feral_config.json so consumers can tell at a glance
        which precision regime produced any given run's parquets.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.feral_feature.FeralFeature.Params`.
    """

    # Global, like `xgboost` and `lightning-action`, because it fits a classifier
    # over a collection of labeled sequences and then applies it per sequence. The
    # category groups it in the pipeline diagram and `mosaic features list`. A GPU
    # feature of category global is also offered to the queue's gpu-train lane.
    category = "global"
    costly_fit = True  # a superseded-run cleanup keeps its runs
    # A run given a `model` or a `model_dir` fits nothing, and `__init__` clears this.
    fits = True  # learns from its inputs, so a held-out group must stay out of its fit
    # V-JEPA on CUDA by default (`device="cuda"`), so a run holds a GPU.
    resource_class = "gpu"
    name = "feral"
    tool: ClassVar[str] = "feral"
    version = "0.3"
    parallelizable = False
    # The fit streams the step's inputs but trains only on the entries that
    # `templates` assigned to train, and skips the others before touching state.
    scope_dependent = False
    fit_set_pinned_by: ClassVar[str] = "templates"
    relates: RelatesLevel = "none"
    accepts_overlap = (
        False  # takes its context from the crop video, not from this frame
    )
    # A sequence input decodes the entry's own media through `resolve_media`.
    consumed_roots: tuple[str, ...] = ("media_raw",)
    prerequisites: tuple[Prerequisite, ...] = (FERAL,)
    emits: EmitsLevel = "as-input"
    follows_ids: bool = False
    # Apply predicts each frame of the videos that its input names: one per row
    # of an interaction crop table, one per individual of an egocentric crop
    # table, or one per sequence of any other input. A crop input writes a row
    # per video frame, and any other input writes a row per input row of a
    # predicted frame. Each identity column is written when the input has it.
    requires: ClassVar[tuple[Requirement, ...]] = (
        AnyOf((Column("video_path"), Column(C.frame_col))),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(Column(C.frame_col), Prefix("prob_"), Column("predicted_label")),
        conditional=(
            Conditional(Column("id1"), when_present=Column("id1")),
            Conditional(Column("id2"), when_present=Column("id1")),
            Conditional(Column("perspective"), when_present=Column("perspective")),
            Conditional(
                Column("interaction_id"), when_present=Column("interaction_id")
            ),
            Conditional(Column(C.id_col), when_present=Column(C.id_col)),
            *kept(C.group_col, C.seq_col),
        ),
    )
    ModelArtifact = FeralModelArtifact

    class Inputs(Inputs[TrackInput | Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        # Training mode: the templates run, the labels and the training config
        templates: Annotated[
            LabeledTemplatesRef | None, Declared(_TEMPLATES_DESCRIPTION)
        ] = None
        labels: Annotated[
            GroundTruthLabelsSource | None, Declared(_LABELS_DESCRIPTION)
        ] = None
        unlabeled_as: Annotated[int | None, Declared(_UNLABELED_AS_DESCRIPTION)] = None
        training: Annotated[
            FeralTrainingConfig | None, Declared(_TRAINING_DESCRIPTION)
        ] = None
        # Applying a trained run
        model: Annotated[FeralModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = None
        # Importing a checkpoint trained outside mosaic
        model_dir: Annotated[Path | None, Declared(_MODEL_DIR_DESCRIPTION)] = None
        # Shared model config. The backbone also selects the pretrained weights
        # FERAL downloads from the HuggingFace hub on first use. The default's
        # model card declares MIT; per-entry licenses are in NOTICE.
        model_name: Annotated[
            str,
            Field(
                examples=[
                    "facebook/vjepa2-vitl-fpc32-256-diving48",
                    "vjepa2_vitl_diving48",
                ]
            ),
            Declared(_MODEL_NAME_DESCRIPTION),
        ] = "facebook/vjepa2-vitl-fpc32-256-diving48"
        predict_per_item: Annotated[int, Declared(_PREDICT_PER_ITEM_DESCRIPTION)] = 64
        # Inference hyperparameters
        chunk_length: Annotated[
            int, Declared(_CHUNK_LENGTH_DESCRIPTION, unit="frames")
        ] = 64
        chunk_shift: Annotated[
            int, Declared(_CHUNK_SHIFT_DESCRIPTION, unit="frames")
        ] = 32
        chunk_step: Annotated[int, Declared(_CHUNK_STEP_DESCRIPTION, unit="frames")] = 1
        resize_to: Annotated[int, Declared(_RESIZE_TO_DESCRIPTION, unit="px")] = 256
        # Which hardware runs the model, not what the model computes. apply()
        # takes its device off the loaded model rather than off this field, and
        # nothing reads it back from disk -- fit() writes cfg["device"] and reads
        # it once from the same in-memory dict. Precision is governed by
        # `inference_autocast`, which IS hashed.
        # HASH_EXCLUDE, on the standard this repo already applied twice
        # (infer_batch_size below, arhmm's `backend`): equivalent up to
        # floating-point associativity, not bit-identity -- cpu and cuda float32
        # kernels reassociate differently. Pinning a run's identity to the card it
        # happened to land on costs a full recompute for nothing, and blocks
        # reloading finished results on a machine without a GPU.
        device: Annotated[
            str,
            HASH_EXCLUDE,
            Field(examples=["cpu", "cuda"]),
            Declared(_DEVICE_DESCRIPTION),
        ] = "cuda"
        # Chunks per forward pass during apply(). Pure throughput knob:
        # batching is numerically equivalent to the per-chunk path
        # (eval() freezes BatchNorm1d + disables dropout; all attention is
        # within-sample). Raise for GPU utilization, lower on OOM. Propagates
        # to parallel workers via params.model_dump() like any other field.
        # HASH_EXCLUDE: omitted from the run_id hash (it doesn't affect output),
        # so changing it never invalidates cached FERAL outputs.
        infer_batch_size: Annotated[
            int, HASH_EXCLUDE, Declared(_INFER_BATCH_SIZE_DESCRIPTION)
        ] = 4
        # Production inference precision override. Default False = full float32,
        # matching the class-level "Precision convention". Set True for ~1.5-2x
        # speedup at the cost of bfloat16's decision-boundary noise (~10pp
        # recall shift on threshold-sensitive classes). Stored in
        # feral_config.json so re-runs and downstream readers know which
        # precision regime produced the parquets.
        inference_autocast: Annotated[
            bool, Declared(_INFERENCE_AUTOCAST_DESCRIPTION)
        ] = False
        # Class configuration (read from the model when None)
        class_names: Annotated[
            dict[str, str] | None, Declared(_CLASS_NAMES_DESCRIPTION)
        ] = None
        # Decision threshold (matches XgboostFeature pattern)
        decision_threshold: Annotated[
            float | None, Declared(_DECISION_THRESHOLD_DESCRIPTION)
        ] = None
        default_class: Annotated[int, Declared(_DEFAULT_CLASS_DESCRIPTION)] = 0

        @model_validator(mode="after")
        def _check_mode(self) -> Self:
            """Exactly one source, and the params that go with a fit only with it."""
            given = [
                field
                for field in ("templates", "model", "model_dir")
                if getattr(self, field) is not None
            ]
            if len(given) != 1:
                raise ValueError(
                    "Exactly one of templates (train), model (apply a trained "
                    "run) or model_dir (import a checkpoint) must be set, "
                    f"and {len(given)} are: {', '.join(given) or 'none'}."
                )
            if self.templates is None:
                for field in ("labels", "training", "unlabeled_as"):
                    if getattr(self, field) is not None:
                        raise ValueError(
                            f"{field} is read when training from templates, and "
                            "a run given a model or a model_dir fits nothing."
                        )
                return self
            for field in ("labels", "training"):
                if getattr(self, field) is None:
                    raise ValueError(
                        f"{field} is required with templates: the model trains "
                        "on the videos of the entries the templates run assigned "
                        "to train, labeled from labels."
                    )
            return self

    def __init__(
        self,
        inputs: FeralFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.fits = self.params.templates is not None
        self._ds: Dataset | None = None
        self.progress_callback: TrainingProgressCallback | None = None
        self._reset()

    def _reset(self) -> None:
        self._model: torch.nn.Module | None = None
        # The four fields below describe the loaded or fitted model together.
        self._cfg: FeralNetwork | None = None
        self._class_values: tuple[int, ...] | None = None
        self._class_names: dict[int, str] | None = None
        self._is_multilabel: bool | None = None
        self._metrics: dict[str, object] | None = None
        self._training_metrics: list[dict[str, float]] = []
        self._run_root: Path | None = None
        self._input_root: Path | None = None
        self._train_entries: set[tuple[str, str]] | None = None
        self._test_entries: set[tuple[str, str]] | None = None
        self._label_files: DependencyLookup = {}

    def bind_dataset(self, ds: Dataset) -> None:
        """Store dataset reference for resolving media paths."""
        self._ds = ds

    def _input_mode(self) -> FeralInputMode:
        """Where the first feature input has FERAL read its videos from."""
        first = next(iter(self.inputs.feature_inputs), None)
        return "sequence" if first is None else feral_input_mode(first.feature)

    def _videos_of(self, ds: Dataset, df: pd.DataFrame) -> list[FeralVideo]:
        """The videos of one entry's input rows, from its crop run or its own media."""
        mode = self._input_mode()
        if mode == "sequence":
            group, sequence = _entry_of(df)
            return [sequence_video(ds, group, sequence, _feral_get_frame_count())]
        if self._input_root is None:
            raise ValueError(f"{self.name}: no crop run root was loaded for its input")
        videos, _ = crop_videos(mode, self._input_root, df, ds.base_dir)
        return videos

    # --- State management ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._reset()
        self._run_root = run_root
        self._load_input_root(dependency_lookups)
        p = self.params

        # Branch 1: cached model in run_root (from prior fit+save_state)
        cached_model = run_root / "feral_model.joblib"
        if cached_model.exists() and (run_root / "feral_config.json").exists():
            _check_feral()
            self._load_model(FeralModelArtifact().from_path(cached_model))
            return True

        # Branch 2: a trained run named by params.model
        if p.model is not None:
            path = named_model_file(self.name, p.model, artifact_paths)
            _check_feral()
            self._load_model(p.model.from_path(path))
            return True

        # Branch 3: a checkpoint trained outside mosaic, from params.model_dir
        if p.model_dir is not None:
            model_dir = (
                self._ds.resolve_path(p.model_dir)
                if self._ds is not None
                else Path(p.model_dir)
            )
            checkpoint = model_dir / "model_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"[{self.name}] model_dir {p.model_dir} resolves to {model_dir}, "
                    "which holds no model_best.pt"
                )
            config_path = model_dir / "config.json"
            self._load_model(checkpoint, config_path if config_path.exists() else None)
            return True

        # Branch 4: the entries of the templates run to fit on
        templates_path = artifact_paths.get("templates")
        if templates_path is None:
            raise FileNotFoundError(
                f"[{self.name}] the templates run holds no templates.parquet to read "
                "its train and test entries from"
            )
        train_entries = entries_of_templates(templates_path, "train")
        test_entries = entries_of_templates(templates_path, "test")
        if train_entries is None or test_entries is None:
            raise ValueError(
                f"[{self.name}] the templates run left no template_provenance.parquet "
                "beside its templates, so the entries it held out are unknown. "
                "Rerun extract-labeled-templates"
            )
        self._train_entries = set(train_entries)
        self._test_entries = set(test_entries)
        self._label_files = dependency_lookups.get("labels", {})
        return False

    def _load_input_root(self, dependency_lookups: dict[str, DependencyLookup]) -> None:
        """Record the first feature input's run root, and refuse a crop without videos."""
        mode = self._input_mode()
        self._input_root = input_run_root(dependency_lookups.get("_input_0", {}))
        if mode == "sequence":
            return
        if self._input_root is None:
            first = self.inputs.feature_inputs[0]
            raise ValueError(
                f"[{self.name}] its {mode} crop input {first.feature!r} resolved no "
                "stored output, so no crop video can be found"
            )
        refuse_crop_without_videos(self._input_root)

    def _load_model(
        self,
        source: FeralModelBundle | Path,
        config_path: Path | None = None,
    ) -> None:
        """Load a FERAL model from a trained run's bundle or an imported checkpoint.

        A bundle carries its weights, classes and config. A checkpoint path is one
        trained outside mosaic, read with its optional ``config.json``.
        """
        _check_feral()
        import torch

        p = self.params
        if isinstance(source, Path):
            state_dict, cfg, names, is_multilabel = _read_imported(
                source, config_path, p
            )
            class_values = tuple(sorted(names))
            class_names = names
        else:
            state_dict, cfg = source["state_dict"], source["cfg"]
            class_values = tuple(source["class_values"])
            class_names = dict(source["class_names"])
            is_multilabel = source["is_multilabel"]
            if p.class_names is not None:
                class_names = {
                    **class_names,
                    **_named_values(p.class_names, class_values),
                }

        model = _build_network(cfg, len(class_values))
        _load_into(model, state_dict)
        model.to(torch.device(p.device))
        model.eval()
        self._model = model
        self._cfg = cfg
        self._class_values = class_values
        self._class_names = class_names
        self._is_multilabel = is_multilabel

    # --- Fit (training) ---

    def fit(self, inputs: InputStream) -> None:
        """Train a FERAL model on the videos of the templates run's train entries.

        Streams the step's inputs and keeps the entries that the ``templates`` run
        assigned to ``train`` or ``test``. Each entry's videos come from the
        crop run or the entry's own media. A seeded ``training.val_fraction`` of
        the train entries validates, FERAL's loop pieces train, and the test
        entries are evaluated afterwards.

        Raises:
            ToolUnavailable: The ``feral`` package is not importable.
            RuntimeError: No dataset is bound, or ``load_state`` loaded no templates.
            ValueError: The scope lacks a train entry or a labeled test entry, no
                train video is left, or a test entry leaves no test video.
        """
        _check_feral()
        ds = self._ds
        if ds is None:
            raise RuntimeError(
                f"[{self.name}] fit needs the dataset, which run_feature binds with "
                "bind_dataset"
            )
        p = self.params
        training, labels_source = p.training, p.labels
        train_entries, test_entries = self._train_entries, self._test_entries
        run_root = self._run_root
        if (
            training is None
            or labels_source is None
            or train_entries is None
            or test_entries is None
            or run_root is None
        ):
            raise RuntimeError(
                f"[{self.name}] no templates were loaded to fit on: fit follows a "
                "load_state that returned False for a step given templates"
            )

        videos_of_entry: dict[tuple[str, str], list[FeralVideo]] = {}
        for _, df in inputs():
            if df.empty:
                continue
            entry = _entry_of(df)
            if entry in train_entries or entry in test_entries:
                videos_of_entry[entry] = self._videos_of(ds, df)

        seen = set(videos_of_entry)
        refuse_entries_outside_scope(
            self.name,
            "train entries of the templates run",
            train_entries,
            self._label_files,
            seen,
            consequence="The model would train on fewer entries than its templates name.",
            remedy=FIT_SCOPE_REMEDY,
        )
        refuse_entries_outside_scope(
            self.name,
            "test entries of the templates run",
            test_entries,
            self._label_files,
            seen,
            consequence="Its test metrics would cover fewer entries than its templates name.",
            remedy=FIT_SCOPE_REMEDY,
        )

        val_entries = set(
            validation_entries(
                sorted(train_entries), training.val_fraction, training.seed
            )
        )
        split_videos: dict[FeralSplit, list[FeralVideo]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for entry in sorted(videos_of_entry):
            if entry in val_entries:
                split = "val"
            elif entry in train_entries:
                split = "train"
            else:
                split = "test"
            split_videos[split].extend(videos_of_entry[entry])

        labels = build_feral_labels(
            split_videos,
            self._label_files,
            labels_source,
            unlabeled_as=p.unlabeled_as,
            min_frames=(p.chunk_length - 1) * p.chunk_step + 1,
        )
        labels_json = labels.labels_json
        renamed = (
            {}
            if p.class_names is None
            else _named_values(p.class_names, labels.class_values)
        )
        if split_videos["val"] and not labels_json["splits"]["val"]:
            log.warning(
                "[feral] none of the %d validation video(s) is left (%s), so "
                "training runs without validation and keeps the last epoch",
                len(split_videos["val"]),
                _dropped_summary(labels.dropped),
            )
        if test_entries and not labels_json["splits"]["test"]:
            raise ValueError(
                f"{self.name}: none of the {len(test_entries)} test entries of the "
                f"templates run leaves a test video to evaluate ("
                f"{len(split_videos['test'])} test videos, "
                f"{_dropped_summary(labels.dropped)})"
            )
        _validate_feral_labels(labels_json, ds.base_dir)

        cfg = feral_config(
            p, backbone=_resolve_backbone_key(p.model_name), dataset_root=ds.base_dir
        )
        run_root.mkdir(parents=True, exist_ok=True)
        _write_json(run_root / "feral_labels.json", labels_json)
        _write_json(run_root / "config.json", cfg)

        model, best_map, training_metrics = train_feral(
            cfg,
            labels_json,
            run_root=run_root,
            progress_callback=self.progress_callback,
        )

        self._model = model
        self._cfg = cfg
        self._class_values = labels.class_values
        class_names = {
            value: labels_json["class_names"][str(index)]
            for index, value in enumerate(labels.class_values)
        }
        class_names = {**class_names, **renamed}
        self._class_names = class_names
        self._is_multilabel = labels_json["is_multilabel"]

        test_metrics: dict[str, float] = {}
        if test_entries:
            test_metrics = _evaluate_partition(
                model, cfg, labels_json, "test", cfg["device"]
            )
        self._training_metrics = training_metrics
        self._metrics = {
            "class_names": {str(value): name for value, name in class_names.items()},
            "best_val_map": best_map,
            "training_metrics": training_metrics,
            "test_metrics": test_metrics,
        }

    # --- Save state ---

    def save_state(self, run_root: Path) -> None:
        model = self._model
        cfg = self._cfg
        class_values = self._class_values
        class_names = self._class_names
        is_multilabel = self._is_multilabel
        if (
            model is None
            or cfg is None
            or class_values is None
            or class_names is None
            or is_multilabel is None
        ):
            return

        run_root.mkdir(parents=True, exist_ok=True)

        # The model file goes first, then the reports, and the config last:
        # ``load_state`` tests for the model and the config, and takes the state
        # to be whole when both are there.
        unwrapped: torch.nn.Module = getattr(model, "_orig_mod", model)
        bundle: FeralModelBundle = {
            "state_dict": {
                name: weight.detach().cpu()
                for name, weight in unwrapped.state_dict().items()
            },
            "class_values": list(class_values),
            "class_names": class_names,
            "is_multilabel": is_multilabel,
            "cfg": cfg,
        }
        write_bundle(run_root / "feral_model.joblib", bundle)

        # Save training metrics if available
        if self._metrics is not None:
            reports = json.dumps(self._metrics, indent=2, default=str)
            atomic_write(
                run_root / "reports.json",
                lambda temporary: temporary.write_text(reports),
            )
            summary_df = pd.DataFrame(self._training_metrics)
            atomic_write(
                run_root / "summary.csv",
                lambda temporary: summary_df.to_csv(temporary, index=False),
            )

        _write_json(
            run_root / "feral_config.json",
            {
                "cfg": cfg,
                "class_values": list(class_values),
                "class_names": {
                    str(value): name for value, name in class_names.items()
                },
                "is_multilabel": is_multilabel,
                "inference_autocast": self.params.inference_autocast,
                "version": self.version,
            },
        )

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        _check_feral()
        if (
            self._model is None
            or self._cfg is None
            or self._class_values is None
            or self._is_multilabel is None
        ):
            raise RuntimeError("No model loaded — call load_state() first")
        if self._ds is None:
            raise RuntimeError(
                f"[{self.name}] apply needs the dataset, which run_feature binds "
                "with bind_dataset"
            )
        if df.empty:
            return pd.DataFrame()

        import torch

        p = self.params
        model = self._model
        device = next(model.parameters()).device
        resize = _video_resize(p.resize_to)
        normalize = _video_normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        scale = 1 / 255

        def prepare(video: torch.Tensor) -> torch.Tensor:
            return normalize(resize(video) * scale)

        # Precision regime, built once and reused across batches. Default full
        # float32 -- same convention as the eval paths -- because inference
        # results land in stored parquet outputs that get acted on downstream
        # (event counts, aggregate analyses, paper figures), so reproducibility
        # wins over the ~1.5-2x speedup bfloat16 autocast would buy. Opt in via
        # params.inference_autocast=True when wall-clock dominates over the
        # ~10pp recall shift on threshold-sensitive classes. See class docstring
        # "Precision convention".
        autocast: contextlib.AbstractContextManager[object] = contextlib.nullcontext()
        if p.inference_autocast and device.type == "cuda":
            import torch.amp

            autocast = torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda")

        videos = self._videos_of(self._ds, df)
        frame_tables: list[pd.DataFrame] = []
        skipped_missing = 0
        for video in videos:
            if not video.path.exists():
                skipped_missing += 1
                continue
            table = self._video_predictions(
                video, model, prepare, device, autocast, self._cfg, self._class_values
            )
            if table is not None:
                frame_tables.append(table)

        if skipped_missing:
            log.warning(
                "FeralFeature.apply: %d of %d videos not found; produced %d "
                "predictions. If all are missing, check that the upstream crop run "
                "still holds its videos.",
                skipped_missing,
                len(videos),
                sum(len(table) for table in frame_tables),
            )

        if not frame_tables:
            return pd.DataFrame()

        if self._input_mode() == "sequence":
            result = _rows_of_frames(df, pd.concat(frame_tables, ignore_index=True))
        else:
            result = pd.concat(frame_tables, ignore_index=True)

        # Carry over group/sequence from input
        for col in (C.group_col, C.seq_col):
            if col in df.columns and col not in result.columns:
                result[col] = df[col].iloc[0]

        return result

    def _video_predictions(
        self,
        video: FeralVideo,
        model: torch.nn.Module,
        prepare: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        autocast: contextlib.AbstractContextManager[object],
        cfg: FeralNetwork,
        class_values: Sequence[int],
    ) -> pd.DataFrame | None:
        """The per-frame predictions of one video, ``None`` when it has no frames.

        Overlapping chunks are averaged per frame, and a frame no chunk covers is
        left out. The table holds ``frame``, the video's identity columns,
        ``prob_<value>`` for each class and ``predicted_label``.

        Raises:
            ValueError: FERAL counts a different number of frames than the video
                has frame numbers.
        """
        p = self.params
        total_frames = _feral_get_frame_count()(str(video.path))
        if total_frames is None or total_frames == 0:
            return None
        if total_frames != len(video.frames):
            raise ValueError(
                f"{self.name}: the video {video.filename!r} has {total_frames} frames "
                f"by FERAL's count and {len(video.frames)} frame numbers"
            )

        frame_ids = _feral_get_frame_ids()(
            total_frames, p.chunk_shift, p.chunk_length, p.chunk_step
        )
        frame_preds = np.zeros((total_frames, len(class_values)), dtype=np.float32)
        frame_counts = np.zeros(total_frames, dtype=np.float32)

        # Batch chunks through the model to amortize kernel-launch overhead and
        # fill the GPU (~infer_bs x fewer forward passes).
        for batch_frames in _chunked(frame_ids, max(1, p.infer_batch_size)):
            chunk_probs_list = chunk_probabilities(
                model,
                batch_frames,
                video.path,
                prepare,
                device,
                autocast,
                is_multilabel=bool(self._is_multilabel),
                tokens_per_chunk=cfg["predict_per_item"],
            )
            for chunk_probs, frames in zip(chunk_probs_list, batch_frames, strict=True):
                for i, frame_idx in enumerate(frames):
                    if i < len(chunk_probs):
                        frame_preds[frame_idx] += chunk_probs[i]
                        frame_counts[frame_idx] += 1

        valid = frame_counts > 0
        frame_preds[valid] /= frame_counts[valid, np.newaxis]
        kept = np.flatnonzero(valid)
        if kept.size == 0:
            return None

        table = _frame_table(video).iloc[kept].reset_index(drop=True)
        for position, value in enumerate(class_values):
            table[f"prob_{value}"] = frame_preds[kept, position]
        table["predicted_label"] = [
            self._predict_label(frame_preds[index], class_values) for index in kept
        ]
        return table

    def _predict_label(
        self, frame_probs: NDArray[np.float32], class_values: Sequence[int]
    ) -> int:
        """The original label value of the class a frame's probabilities choose.

        With a ``decision_threshold``, a class below it cannot be chosen, and a
        frame where none reaches it takes ``default_class``. Both the chosen
        class and ``default_class`` are original label values.
        """
        threshold = self.params.decision_threshold
        if threshold is None:
            return class_values[int(np.argmax(frame_probs))]
        masked = np.where(frame_probs < threshold, 0.0, frame_probs)
        if masked.sum() == 0:
            return self.params.default_class
        return class_values[int(np.argmax(masked))]

    # --- Post-training utilities (operate on a saved run directory) ---

    @classmethod
    def load_trained_model(
        cls,
        run_root: Path,
        device: str = "cuda",
    ) -> torch.nn.Module:
        """Reconstruct the trained ``nn.Module`` from a saved run directory.

        Reads the weights, the classes and the nested config from
        ``feral_model.joblib``, or, without it, the weights from ``model_best.pt``
        and the nested config from the checkpoint's metadata. Moves the model to
        ``device`` and sets eval mode. Use this when you need just the model --
        e.g. to feed it custom batches outside FERAL's ClsDataset.

        Parameters
        ----------
        run_root : Path
            Directory containing the model (typically a Mosaic feature run
            directory like ``features/feral__from__.../<run_id>/``).
        device : str
            Where to load the model.

        Raises
        ------
        FileNotFoundError
            The directory holds neither ``feral_model.joblib`` nor ``model_best.pt``.
        ValueError
            ``model_best.pt`` carries no metadata to build the model from.
        """
        _check_feral()
        import torch

        bundle_path = run_root / "feral_model.joblib"
        if bundle_path.exists():
            bundle = FeralModelArtifact().from_path(bundle_path)
            model = _build_network(bundle["cfg"], len(bundle["class_values"]))
            _load_into(model, bundle["state_dict"])
        else:
            checkpoint = run_root / "model_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"No feral_model.joblib or model_best.pt under {run_root}"
                )
            state_dict, meta = _load_checkpoint_state_dict(
                torch.load(str(checkpoint), map_location="cpu")
            )
            if meta is None or "cfg" not in meta or "class_names" not in meta:
                raise ValueError(
                    f"{checkpoint} carries no class_names and cfg to build the "
                    "model from; it is not a checkpoint of a mosaic training run"
                )
            names = _CLASS_NAMES.validate_python(meta["class_names"])
            model = _build_network(_NETWORK.validate_python(meta["cfg"]), len(names))
            _load_into(model, state_dict)
        model.to(torch.device(device)).eval()
        return model

    @classmethod
    def evaluate(
        cls,
        run_root: Path,
        partition: str,
        *,
        dataset_root: Path,
        device: str = "cuda",
        batch_size: int | None = None,
        num_workers: int | None = None,
        progress_interval: int = 5,
    ) -> list[FeralAnswer]:
        """Run inference on ``partition`` of the trained model's labels JSON
        and return the FERAL `answers` list (per-chunk probabilities + truth).

        Reads ``feral_labels.json`` and the nested config of
        ``feral_config.json`` from the run, and the videos from *dataset_root*,
        which FERAL's video filenames are relative to.

        Returns an empty list if the partition is missing or empty.
        """
        _check_feral()
        import torch

        saved = _TRAINED_RUN.validate_json((run_root / "feral_config.json").read_text())
        labels_json = _LABELS_JSON.validate_json(
            (run_root / "feral_labels.json").read_text()
        )

        loader = _build_eval_loader(
            saved["cfg"],
            labels_json,
            partition,
            prefix=str(dataset_root),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if loader is None:
            return []

        model = cls.load_trained_model(run_root, device=device)
        try:
            torch_device = next(model.parameters()).device
            answers = _run_inference_loop(
                model,
                loader,
                len(labels_json["class_names"]),
                labels_json["is_multilabel"],
                torch_device,
                progress_label=partition,
                progress_interval=progress_interval,
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return answers

    @staticmethod
    def compute_metrics(
        answers: Sequence[FeralAnswer],
        labels_json: Mapping[str, object],
        partition: str,
        prefix: str | None = None,
        multilabel_threshold: float = 0.85,
    ) -> dict[str, float]:
        """Compute multiclass + F1 + frame-level mAP metrics on ``answers``.

        Thin orchestrator over FERAL's ``calculate_multiclass_metrics``,
        ``calc_frame_level_map``, and ``calculate_f1_metrics``. The chunk
        answers are ensembled into per-frame predictions first, as FERAL's own
        evaluation does, and the frame-level mAP and the F1 metrics read those.
        Returns a flat dict keyed with ``<prefix>_<metric>``. Returns ``{}`` if
        ``answers`` is empty. ``prefix`` defaults to ``partition``.
        ``multilabel_threshold`` is forwarded to ``calculate_f1_metrics``
        (only affects multilabel runs; matches FERAL's default of 0.85).
        """
        if not answers:
            return {}
        if prefix is None:
            prefix = partition

        class_names = _CLASS_NAMES.validate_python(labels_json["class_names"])
        is_multilabel = bool(labels_json.get("is_multilabel", False))
        predictions = ensembled_predictions(answers, labels_json, partition)

        metrics: dict[str, float] = {}
        metrics.update(
            _feral_calculate_multiclass_metrics()(answers, class_names, prefix)
        )
        metrics[f"{prefix}_frame_level_map"] = _feral_calc_frame_level_map()(
            predictions,
            labels_json,
            partition,
        )
        metrics.update(
            _feral_calculate_f1_metrics()(
                predictions,
                labels_json,
                partition,
                is_multilabel,
                prefix,
                multilabel_threshold,
            )
        )
        return metrics

    @staticmethod
    def read_reports(run_root: Path) -> dict:
        """Return the contents of ``reports.json`` -- the metrics persisted
        by ``save_state()`` at training time (class_names, best_val_map,
        training_metrics list, test_metrics dict). Returns ``{}`` if the
        file is missing.
        """
        reports_path = run_root / "reports.json"
        if not reports_path.exists():
            return {}
        with open(reports_path) as f:
            return json.load(f)


# --- FERAL's functions, typed -------------------------------------------------
#
# FERAL is an optional dependency that the type checker cannot resolve, and a
# module attribute read through ``importlib`` is untyped. Each function the
# training loop uses is declared as a callable protocol and fetched when it is
# called, so a test can replace one on its module with ``monkeypatch.setattr``.

_CLASS_NAMES: Final = TypeAdapter(dict[int, str])
"""FERAL's ``class_names``, whose keys are integers spelled as text."""

FeralAnswer = tuple[tuple[str, int, int], Sequence[float], Sequence[float]]
"""One chunk row of FERAL's answers: ``(filename, frame, chunk position)``, the
class probabilities and the one-hot target."""

FramePredictions = dict[str, NDArray[np.float64]]
"""Per-frame class scores of each video, by filename."""


class _FeralEma(Protocol):
    """The exponential-moving-average wrapper FERAL builds around a model."""

    ema: torch.nn.Module


class _FeralBuildLoaders(Protocol):
    def __call__(
        self, cfg: FeralConfig, labels_json: FeralLabelsJson, num_classes: int
    ) -> tuple[dict[str, object], dict[str, object]]: ...


class _FeralBuildModel(Protocol):
    def __call__(
        self,
        cfg: FeralConfig,
        num_classes: int,
        device: torch.device,
        *,
        with_ema: bool = True,
    ) -> tuple[torch.nn.Module, _FeralEma | None]: ...


class _FeralBuildTrainingObjects(Protocol):
    def __call__(
        self,
        cfg: FeralConfig,
        model: torch.nn.Module,
        train_dataset: object,
        train_loader: object,
        labels_json: FeralLabelsJson,
        device: torch.device,
    ) -> tuple[
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
        object,
    ]: ...


class _FeralTrainOneEpoch(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        loader: object,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        mixup: object,
        model_ema: _FeralEma | None,
        num_classes: int,
        is_multilabel: bool,
        predict_per_item: int,
        device: torch.device,
        log_fn: None,
        grad_clip_norm: float | None,
        log_grad_norm: bool,
        heavy_log_every: int | None,
    ) -> tuple[list[FeralAnswer], float]: ...


class _FeralEvaluate(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        loader: object,
        criterion: torch.nn.Module | None = None,
        *,
        num_classes: int,
        is_multilabel: bool,
        device: torch.device,
    ) -> tuple[list[FeralAnswer], float | None]: ...


class _FeralSaveModel(Protocol):
    def __call__(
        self, model: torch.nn.Module, path: str, metadata: Mapping[str, object]
    ) -> None: ...


class _FeralEnsemblePredictions(Protocol):
    def __call__(
        self, ans: Sequence[FeralAnswer], logits: FramePredictions
    ) -> FramePredictions: ...


class _FeralGenerateEmptyLogits(Protocol):
    def __call__(
        self, labels_json: Mapping[str, object], partition: str
    ) -> FramePredictions: ...


class _FeralPostprocessPredictions(Protocol):
    def __call__(
        self, predictions: FramePredictions, smoothing_window: int | None = None
    ) -> FramePredictions: ...


class _FeralCalcFrameLevelMap(Protocol):
    def __call__(
        self,
        predictions: FramePredictions,
        labels_json: Mapping[str, object],
        partition: str,
    ) -> float: ...


class _FeralCalculateF1Metrics(Protocol):
    def __call__(
        self,
        predictions: FramePredictions,
        labels_json: Mapping[str, object],
        partition: str,
        is_multilabel: bool,
        prefix: str,
        multilabel_threshold: float,
    ) -> dict[str, float]: ...


class _FeralCalculateMulticlassMetrics(Protocol):
    def __call__(
        self,
        ans: Sequence[FeralAnswer],
        class_names: Mapping[int, str],
        prefix: str = "",
    ) -> dict[str, float]: ...


class _FeralModelClass(Protocol):
    def __call__(
        self,
        *,
        backbone: str,
        num_classes: int,
        predict_per_item: int,
        fc_drop_rate: float,
        freeze_encoder_layers: int,
        pretrained: bool,
    ) -> torch.nn.Module: ...


class _FeralClsDataset(Protocol):
    def __call__(
        self,
        *,
        partition: str,
        label_json_dict: FeralLabelsJson,
        do_aa: bool,
        predict_per_item: int,
        num_classes: int,
        prefix: str,
        resize_to: int,
        chunk_shift: int,
        chunk_length: int,
        chunk_step: int,
    ) -> torch.utils.data.Dataset[object]: ...


class _FeralPrepForAnswers(Protocol):
    def __call__(
        self, outputs: torch.Tensor, targets: torch.Tensor, names: object
    ) -> list[FeralAnswer]: ...


class _FeralGetFrameCount(Protocol):
    def __call__(self, path: str) -> int | None: ...


class _FeralReadRangeVideo(Protocol):
    def __call__(self, path: str, frames: list[int]) -> torch.Tensor: ...


class _VideoTransform(Protocol):
    def __call__(self, video: torch.Tensor) -> torch.Tensor: ...


class _FeralGetFrameIds(Protocol):
    def __call__(
        self, total_frames: int, chunk_shift: int, chunk_length: int, chunk_step: int
    ) -> list[list[int]]: ...


class _FeralValidateLabels(Protocol):
    def __call__(
        self, labels_json: Mapping[str, object], video_folder: Path
    ) -> None: ...


class _TorchManualSeed(Protocol):
    def __call__(self, seed: int) -> object: ...


def _torch_manual_seed() -> _TorchManualSeed:
    # torch's own annotation of the generator it returns is partly unknown to the
    # type checker.
    return importlib.import_module("torch").manual_seed


def _feral_model_class() -> _FeralModelClass:
    return importlib.import_module("feral.model").FeralModel


def _feral_cls_dataset() -> _FeralClsDataset:
    return importlib.import_module("feral.dataset").ClsDataset


def _feral_collate_fn_val() -> Callable[[list[object]], object]:
    return importlib.import_module("feral.dataset").collate_fn_val


def _feral_get_frame_count() -> _FeralGetFrameCount:
    return importlib.import_module("feral.dataset").get_frame_count


def _feral_read_range_video() -> _FeralReadRangeVideo:
    return importlib.import_module("feral.dataset").read_range_video_decord


def _video_resize(size: int) -> _VideoTransform:
    # torchvision's v2 transforms are not typed for the checker.
    return importlib.import_module("torchvision.transforms.v2").Resize(
        (size, size), antialias=True
    )


def _video_normalize(
    mean: tuple[float, float, float], std: tuple[float, float, float]
) -> _VideoTransform:
    return importlib.import_module("torchvision.transforms.v2").Normalize(mean, std)


def _feral_get_frame_ids() -> _FeralGetFrameIds:
    return importlib.import_module("feral.dataset").get_frame_ids


def _feral_prep_for_answers() -> _FeralPrepForAnswers:
    return importlib.import_module("feral.utils").prep_for_answers


def _feral_validate_labels() -> _FeralValidateLabels:
    return importlib.import_module("feral").validate_labels_json


def _feral_build_loaders() -> _FeralBuildLoaders:
    return importlib.import_module("feral.data").build_datasets_and_loaders


def _feral_build_model() -> _FeralBuildModel:
    return importlib.import_module("feral.modeling").build_model


def _feral_build_training_objects() -> _FeralBuildTrainingObjects:
    return importlib.import_module("feral.modeling").build_training_objects


def _feral_train_one_epoch() -> _FeralTrainOneEpoch:
    return importlib.import_module("feral.loops").train_one_epoch


def _feral_evaluate() -> _FeralEvaluate:
    return importlib.import_module("feral.loops").evaluate


def _feral_save_model() -> _FeralSaveModel:
    return importlib.import_module("feral.utils").save_model


def _feral_ensemble_predictions() -> _FeralEnsemblePredictions:
    return importlib.import_module("feral.metrics").ensemble_predictions


def _feral_generate_empty_logits() -> _FeralGenerateEmptyLogits:
    return importlib.import_module("feral.metrics").generate_empty_logits


def _feral_postprocess_predictions() -> _FeralPostprocessPredictions:
    return importlib.import_module("feral.metrics").postprocess_predictions


def _feral_calc_frame_level_map() -> _FeralCalcFrameLevelMap:
    return importlib.import_module("feral.metrics").calc_frame_level_map


def _feral_calculate_f1_metrics() -> _FeralCalculateF1Metrics:
    return importlib.import_module("feral.metrics").calculate_f1_metrics


def _feral_calculate_multiclass_metrics() -> _FeralCalculateMulticlassMetrics:
    return importlib.import_module("feral.metrics").calculate_multiclass_metrics


def ensembled_predictions(
    answers: Sequence[FeralAnswer], labels_json: Mapping[str, object], partition: str
) -> FramePredictions:
    """Per-frame predictions of *partition* from the chunk *answers*.

    Overlapping chunk predictions are averaged and the frames no chunk covers
    are interpolated, by FERAL's ``ensemble_predictions``.
    """
    return _feral_postprocess_predictions()(
        _feral_ensemble_predictions()(
            answers, _feral_generate_empty_logits()(labels_json, partition)
        )
    )


def _validate_feral_labels(labels_json: FeralLabelsJson, dataset_root: Path) -> None:
    """Have FERAL check *labels_json* against the videos under *dataset_root*."""
    _feral_validate_labels()(labels_json, dataset_root)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write *payload* to *path* as JSON, all at once."""
    text = json.dumps(payload, indent=2, default=str)
    atomic_write(path, lambda temporary: temporary.write_text(text))


def _build_network(cfg: FeralNetwork, num_classes: int) -> torch.nn.Module:
    """FERAL's model for *cfg* and *num_classes*, with its weights untrained.

    ``pretrained=False`` skips the weight download, because the weights are
    loaded from a checkpoint or a model file.
    """
    settings = cfg["model"]
    return _feral_model_class()(
        backbone=_resolve_backbone_key(cfg["backbone"]),
        num_classes=num_classes,
        predict_per_item=cfg["predict_per_item"],
        fc_drop_rate=settings["fc_drop_rate"],
        freeze_encoder_layers=settings["freeze_encoder_layers"],
        pretrained=False,
    )


class _FlatNetwork(TypedDict, total=False):
    """The architecture keys of a ``config.json`` that mosaic wrote before 0.3."""

    model_name: str
    predict_per_item: int
    fc_drop_rate: float
    freeze_encoder_layers: int


_NETWORK: Final = TypeAdapter(FeralNetwork)
_FLAT_NETWORK: Final = TypeAdapter(_FlatNetwork)


def imported_network(
    candidates: Sequence[Mapping[str, object] | None],
    *,
    model_name: str,
    predict_per_item: int,
) -> FeralNetwork:
    """The network of an imported checkpoint, from the first config that records it.

    A nested config, as FERAL writes, wins over a flat one that mosaic wrote
    before 0.3. A checkpoint that records neither is built from the step's
    ``model_name`` and ``predict_per_item``, with the training defaults for the
    weight-agnostic dropout and freeze settings.
    """
    present = [candidate for candidate in candidates if candidate]
    for candidate in present:
        try:
            return _NETWORK.validate_python(candidate)
        except ValidationError:
            continue
    defaults = FeralTrainingConfig()
    for candidate in present:
        if "model_name" not in candidate:
            continue
        flat = _FLAT_NETWORK.validate_python(candidate)
        return {
            "backbone": flat.get("model_name", model_name),
            "predict_per_item": flat.get("predict_per_item", predict_per_item),
            "model": {
                "fc_drop_rate": flat.get("fc_drop_rate", defaults.fc_drop_rate),
                "freeze_encoder_layers": flat.get(
                    "freeze_encoder_layers", defaults.freeze_encoder_layers
                ),
            },
        }
    if present:
        log.warning(
            "[feral] no config of the imported checkpoint records its backbone and "
            "predict_per_item, so the model is built from the step's model_name "
            "and predict_per_item"
        )
    return {
        "backbone": model_name,
        "predict_per_item": predict_per_item,
        "model": {
            "fc_drop_rate": defaults.fc_drop_rate,
            "freeze_encoder_layers": defaults.freeze_encoder_layers,
        },
    }


def checkpoint_classes(
    param_names: Mapping[str, str] | None,
    ckpt_meta: Mapping[str, object] | None,
    legacy_config: Mapping[str, object] | None,
) -> tuple[dict[int, str], bool]:
    """The classes and the multilabel flag of an imported checkpoint.

    The names come from the ``class_names`` param, else from the checkpoint's
    metadata, else from a ``config.json`` that mosaic wrote before 0.3. The keys
    are the label values, which for a checkpoint of FERAL's own are its dense
    class indices. ``is_multilabel`` comes from the checkpoint's metadata, else
    from that ``config.json``.

    Args:
        param_names: The step's ``class_names`` param.
        ckpt_meta: The ``class_names``, ``is_multilabel`` and ``cfg`` a FERAL
            checkpoint embeds, when it embeds any.
        legacy_config: The ``config.json`` beside the checkpoint, when there is one.

    Returns:
        The name of each class by label value, and whether the model is multilabel.

    Raises:
        ValueError: No source names the classes.
    """
    raw: object = param_names or None
    if not raw:
        for source in (ckpt_meta, legacy_config):
            if source is not None and source.get("class_names"):
                raw = source["class_names"]
                break
    if not raw:
        raise ValueError(
            f"{FeralFeature.name}: the imported checkpoint names no classes: it "
            "embeds no class_names, and no config.json beside it records them. "
            "Give the class_names param to supply its classes"
        )
    names = _CLASS_NAMES.validate_python(raw)
    for source in (ckpt_meta, legacy_config):
        if source is not None and "is_multilabel" in source:
            return names, bool(source["is_multilabel"])
    log.warning(
        "[feral] the imported checkpoint records no is_multilabel, so it is read as "
        "single-label. FERAL's own loader would read it from a labels JSON, which "
        "an import does not have"
    )
    return names, False


def _named_values(
    param_names: Mapping[str, str], class_values: Sequence[int]
) -> dict[int, str]:
    """The ``class_names`` param as names of *class_values*.

    Raises:
        ValueError: A name is keyed by a value that is not a class of the model.
    """
    names = _CLASS_NAMES.validate_python(param_names)
    unknown = sorted(set(names) - set(class_values))
    if unknown:
        raise ValueError(
            f"{FeralFeature.name}: class_names names the values {unknown}, and the "
            f"model's classes are {sorted(class_values)}"
        )
    return names


def _read_imported(
    checkpoint: Path, config_path: Path | None, params: FeralFeature.Params
) -> tuple[dict[str, object], FeralNetwork, dict[int, str], bool]:
    """The weights, network, classes and multilabel flag of a checkpoint under ``model_dir``."""
    import torch

    state_dict, meta = _load_checkpoint_state_dict(
        torch.load(str(checkpoint), map_location="cpu")
    )
    config = (
        _OBJECT.validate_json(config_path.read_text())
        if config_path is not None
        else None
    )
    names, is_multilabel = checkpoint_classes(params.class_names, meta, config)
    embedded = None if meta is None else meta.get("cfg")
    network = imported_network(
        [config, _OBJECT.validate_python(embedded) if embedded else None],
        model_name=params.model_name,
        predict_per_item=params.predict_per_item,
    )
    return state_dict, network, names, is_multilabel


_OBJECT: Final = TypeAdapter(dict[str, object])
"""A JSON object, read without its values being checked."""

_LABELS_JSON: Final = TypeAdapter(FeralLabelsJson)


_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    C.id_col,
    "id1",
    "id2",
    "perspective",
    "interaction_id",
)


def _rows_of_frames(rows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Each row of *rows* on a predicted frame, with that frame's predictions.

    A row keeps its identity columns, whichever of ``id`` or ``id1``, ``id2``,
    ``perspective`` and ``interaction_id`` it has. A row on a frame the video
    does not cover, or that no chunk covers, is left out.
    """
    keep = [column for column in _IDENTITY_COLUMNS if column in rows.columns]
    return rows[[*keep, C.frame_col]].merge(predictions, on=C.frame_col, how="inner")


def chunk_probabilities(
    model: torch.nn.Module,
    batch_frames: Sequence[list[int]],
    video_path: Path,
    prepare: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    autocast: contextlib.AbstractContextManager[object],
    *,
    is_multilabel: bool,
    tokens_per_chunk: int,
) -> list[NDArray[np.float32]]:
    """The class probabilities of each chunk in *batch_frames*.

    Each element is a ``(tokens_per_chunk, num_classes)`` array. The model's
    output is chunk-major, so chunk ``n`` holds rows
    ``[n * tokens_per_chunk : (n + 1) * tokens_per_chunk]``. The probabilities
    are sigmoids for a multilabel model and a softmax over the classes
    otherwise.

    Batching does not change per-sample outputs: ``model.eval()`` freezes
    BatchNorm1d running stats and disables dropout, and all attention is
    within-sample. On CUDA OOM the batch is split in half and retried
    recursively, down to a single chunk.
    """
    import torch

    read = _feral_read_range_video()
    batch_tensor = torch.stack(
        [prepare(read(str(video_path), list(frames))) for frames in batch_frames],
        dim=0,
    ).to(device)
    try:
        with torch.no_grad(), autocast:
            # Autocast may leave the logits in bfloat16, which numpy cannot hold.
            output = model(batch_tensor).float()
            if is_multilabel:
                probs = torch.sigmoid(output)
            else:
                probs = torch.nn.functional.softmax(output, dim=1)
            array: NDArray[np.float32] = probs.cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        del batch_tensor
        torch.cuda.empty_cache()
        if len(batch_frames) == 1:
            raise
        mid = len(batch_frames) // 2
        return chunk_probabilities(
            model,
            batch_frames[:mid],
            video_path,
            prepare,
            device,
            autocast,
            is_multilabel=is_multilabel,
            tokens_per_chunk=tokens_per_chunk,
        ) + chunk_probabilities(
            model,
            batch_frames[mid:],
            video_path,
            prepare,
            device,
            autocast,
            is_multilabel=is_multilabel,
            tokens_per_chunk=tokens_per_chunk,
        )
    return [
        array[n * tokens_per_chunk : (n + 1) * tokens_per_chunk]
        for n in range(len(batch_frames))
    ]


def _build_eval_loader(
    cfg: FeralConfig,
    labels_json: FeralLabelsJson,
    partition: str,
    *,
    prefix: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> torch.utils.data.DataLoader[object] | None:
    """A DataLoader over *partition* of *labels_json*, through FERAL's ``ClsDataset``.

    *prefix* is the directory FERAL joins to each video filename. Returns
    ``None`` if the partition is absent or empty.
    """
    import torch

    if not labels_json["splits"].get(partition):
        return None
    data = cfg["data"]
    training = cfg["training"]
    dataset = _feral_cls_dataset()(
        partition=partition,
        label_json_dict=labels_json,
        do_aa=False,
        predict_per_item=cfg["predict_per_item"],
        num_classes=len(labels_json["class_names"]),
        prefix=prefix,
        resize_to=data["resize_to"],
        chunk_shift=data["chunk_shift"],
        chunk_length=data["chunk_length"],
        chunk_step=data["chunk_step"],
    )
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size if batch_size is not None else training["val_bs"],
        num_workers=(
            num_workers if num_workers is not None else training["num_workers"]
        ),
        pin_memory=True,
        drop_last=False,
        collate_fn=_feral_collate_fn_val(),
    )


def _run_inference_loop(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[object] | None,
    num_classes: int,
    is_multilabel: bool,
    device: torch.device | str,
    *,
    progress_label: str = "",
    progress_interval: int = 0,
) -> list[FeralAnswer]:
    """Per-batch inference loop. Returns FERAL `answers` (list of tuples).

    Runs the forward pass in full float32 -- no bfloat16 autocast.
    Evaluation is a measurement step; the speed/precision trade-off
    worth taking inside a training inner loop is not worth taking
    when computing a number we report. bfloat16 shifts decision
    boundaries near probability 0.5 and breaks reproducibility of
    the metric. See class-level docstring for the full precision
    convention across training, eval, and apply.

    If ``progress_label`` is non-empty and ``progress_interval > 0``,
    prints a progress line every N batches.
    """
    import torch

    if loader is None:
        return []

    answers: list[FeralAnswer] = []
    total = len(loader)
    with torch.no_grad():
        for i, (data, target, names) in enumerate(loader):
            data = data.to(device)
            target = target.to(device).view(-1, num_classes)
            output = model(data)
            output_prob = (
                torch.sigmoid(output)
                if is_multilabel
                else torch.nn.functional.softmax(output, 1)
            )
            answers.extend(_feral_prep_for_answers()(output_prob, target, names))
            if (
                progress_label
                and progress_interval > 0
                and ((i + 1) % progress_interval == 0 or i == total - 1)
            ):
                print(
                    f"    {progress_label}: batch {i + 1}/{total}",
                    flush=True,
                )
    return answers


def _evaluate_partition(
    model: torch.nn.Module,
    cfg: FeralConfig,
    labels_json: FeralLabelsJson,
    partition: str,
    device: torch.device | str,
) -> dict[str, float]:
    """Run evaluation on a given partition (e.g. 'test') and return metrics.

    Reads the videos under the dataset root that *cfg* names. For callers outside
    the training loop, use the public ``evaluate()`` and ``compute_metrics()``.
    The forward pass runs in full float32, so the test metrics that land in
    ``reports.json`` match what ``FeralFeature.evaluate()`` produces for this run
    later. See the class docstring for the precision convention.
    """
    loader = _build_eval_loader(
        cfg, labels_json, partition, prefix=cfg["data"]["prefix"]
    )
    answers = _run_inference_loop(
        model,
        loader,
        len(labels_json["class_names"]),
        labels_json["is_multilabel"],
        device,
    )
    return FeralFeature.compute_metrics(
        answers,
        labels_json,
        partition,
        prefix=partition,
        multilabel_threshold=cfg["multilabel_threshold"],
    )


class FeralDataConfig(TypedDict):
    """The ``data`` block of FERAL's config."""

    prefix: str
    chunk_length: int
    chunk_shift: int
    chunk_step: int
    eval_chunk_shift: int | None
    eval_smoothing_window: int | None
    resize_to: int
    resize_style: Literal["square", "rectangle"]
    do_aa: bool
    part_sample: float
    subsample_keep_rare_threshold: float | None


class FeralTrainingBlock(TypedDict):
    """The ``training`` block of FERAL's config."""

    epochs: int
    train_bs: int
    val_bs: int
    num_workers: int
    part_warmup: float
    lr: float
    weight_decay: float
    label_smoothing: float
    compile: bool
    patience: int | None
    grad_clip_norm: float | None
    log_grad_norm: bool
    heavy_log_every: int | None


class FeralConfig(TypedDict):
    """The nested config FERAL's data, model and loop functions read."""

    run_name: str
    backbone: str
    predict_per_item: int
    model: FeralModelConfig
    data: FeralDataConfig
    training: FeralTrainingBlock
    mixup_alpha: float | None
    ema_decay: float | None
    seed: int
    device: str
    starting_checkpoint: str | None
    max_batches: int | None
    multilabel_threshold: float


class FeralTrainedRun(TypedDict):
    """What ``feral_config.json`` holds."""

    cfg: FeralConfig
    class_values: list[int]
    class_names: dict[int, str]
    is_multilabel: bool
    inference_autocast: bool
    version: str


_TRAINED_RUN: Final = TypeAdapter(FeralTrainedRun)


def feral_config(
    params: FeralFeature.Params, *, backbone: str, dataset_root: Path | str
) -> FeralConfig:
    """FERAL's whole config for *params*.

    Every key FERAL reads is set here from a param or a fixed value, and none
    comes from FERAL's packaged defaults, so a change to a FERAL default never
    reaches an output under an unchanged run id.

    Args:
        params: The feature's params. ``training`` must be set.
        backbone: The ``feral.backbones.BACKBONES`` key of the backbone.
        dataset_root: The directory that FERAL's video filenames are relative
            to.

    Returns:
        The nested config.

    Raises:
        ValueError: ``params.training`` is not set.
    """
    training = params.training
    if training is None:
        raise ValueError(f"{FeralFeature.name}: the FERAL config needs training params")
    return {
        "run_name": "mosaic",
        "backbone": backbone,
        "predict_per_item": params.predict_per_item,
        "model": {
            "fc_drop_rate": training.fc_drop_rate,
            "class_weights": training.class_weights,
            "freeze_encoder_layers": training.freeze_encoder_layers,
            "gradient_checkpointing": False,
            "max_class_weight": None,
        },
        "data": {
            "prefix": str(dataset_root),
            "chunk_length": params.chunk_length,
            "chunk_shift": params.chunk_shift,
            "chunk_step": params.chunk_step,
            "eval_chunk_shift": None,
            "eval_smoothing_window": None,
            "resize_to": params.resize_to,
            "resize_style": "square",
            "do_aa": training.do_aa,
            "part_sample": training.part_sample,
            "subsample_keep_rare_threshold": None,
        },
        "training": {
            "epochs": training.epochs,
            "train_bs": training.train_bs,
            "val_bs": training.val_bs,
            "num_workers": training.num_workers,
            "part_warmup": training.part_warmup,
            "lr": training.lr,
            "weight_decay": training.weight_decay,
            "label_smoothing": training.label_smoothing,
            "compile": training.compile,
            "patience": training.patience,
            "grad_clip_norm": None,
            "log_grad_norm": False,
            "heavy_log_every": None,
        },
        "mixup_alpha": training.mixup_alpha,
        "ema_decay": training.ema_decay,
        "seed": training.seed,
        "device": params.device,
        "starting_checkpoint": None,
        "max_batches": None,
        "multilabel_threshold": training.multilabel_threshold,
    }


def train_feral(
    cfg: FeralConfig,
    labels_json: FeralLabelsJson,
    *,
    run_root: Path,
    progress_callback: TrainingProgressCallback | None,
) -> tuple[torch.nn.Module, float, list[dict[str, float]]]:
    """Train FERAL's model for the epochs of *cfg* on FERAL's loop pieces.

    The loaders, model, loss, optimizer, schedule and mixup come from FERAL, and
    so do the training epoch and the validation pass. Validation answers are
    ensembled into per-frame predictions before they are measured. The best
    model by validation mAP, the base or the EMA model, is written to
    ``model_best.pt`` in *run_root*, and the last epoch is kept when there is no
    validation split. An interrupted run resumes from ``training_checkpoint.pt``.

    Args:
        cfg: The config :func:`feral_config` builds.
        labels_json: FERAL's labels dict, with a ``train`` split and an
            optional ``val`` split.
        run_root: The directory that receives ``model_best.pt``,
            ``summary.csv`` and the resume checkpoint.
        progress_callback: Told of each epoch's metrics, or ``None``.

    Returns:
        The model, unwrapped from ``torch.compile`` and loaded with the best
        weights, the best validation mAP (``-1.0`` without validation), and the
        metrics of each epoch.
    """
    import random

    import torch

    seed = cfg["seed"]
    _ = _torch_manual_seed()(seed)
    np.random.seed(seed)
    random.seed(seed)

    training = cfg["training"]
    num_classes = len(labels_json["class_names"])
    is_multilabel = labels_json["is_multilabel"]
    device = torch.device(cfg["device"])
    run_root.mkdir(parents=True, exist_ok=True)

    datasets, loaders = _feral_build_loaders()(cfg, labels_json, num_classes)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    model, model_ema = _feral_build_model()(cfg, num_classes, device)
    criterion, optimizer, scheduler, mixup = _feral_build_training_objects()(
        cfg, model, datasets["train"], train_loader, labels_json, device
    )

    checkpoint_metadata: dict[str, object] = {
        "class_names": labels_json["class_names"],
        "is_multilabel": is_multilabel,
        "cfg": cfg,
    }
    best_path = run_root / "model_best.pt"
    checkpoint_path = run_root / "training_checkpoint.pt"
    best_map = -1.0
    epochs_without_improvement = 0
    training_metrics: list[dict[str, float]] = []
    start_epoch = 0

    if checkpoint_path.exists():
        resume = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        best_map = resume["best_map"]
        epochs_without_improvement = resume["epochs_without_improvement"]
        training_metrics = resume["training_metrics"]
        start_epoch = resume["epoch"] + 1
        if model_ema is not None and "ema_state_dict" in resume:
            model_ema.ema.load_state_dict(resume["ema_state_dict"])
        log.info(
            "Resuming training from epoch %d (best_map=%.4f)", start_epoch, best_map
        )

    csv_progress = CSVProgressCallback(run_root / "summary.csv")

    def save_model(path: Path, source: torch.nn.Module) -> None:
        # A reader that is not under the run claim can load a model file while
        # the fit is still writing it, so each write lands whole or not at all.
        atomic_write(
            path,
            lambda temporary: _feral_save_model()(
                source, str(temporary), checkpoint_metadata
            ),
        )

    try:
        for epoch in range(start_epoch, training["epochs"]):
            _, train_loss = _feral_train_one_epoch()(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                mixup=mixup,
                model_ema=model_ema,
                num_classes=num_classes,
                is_multilabel=is_multilabel,
                predict_per_item=cfg["predict_per_item"],
                device=device,
                log_fn=None,
                grad_clip_norm=training["grad_clip_norm"],
                log_grad_norm=training["log_grad_norm"],
                heavy_log_every=training["heavy_log_every"],
            )
            epoch_metrics: dict[str, float] = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            if val_loader is None:
                save_model(best_path, model)
            else:
                answers, val_loss = _feral_evaluate()(
                    model,
                    val_loader,
                    criterion,
                    num_classes=num_classes,
                    is_multilabel=is_multilabel,
                    device=device,
                )
                val_metrics = FeralFeature.compute_metrics(
                    answers,
                    dict(labels_json),
                    "val",
                    multilabel_threshold=cfg["multilabel_threshold"],
                )
                epoch_metrics.update(val_metrics)
                epoch_metrics["val_loss"] = (
                    float("nan") if val_loss is None else val_loss
                )
                val_map = epoch_metrics["val_frame_level_map"]

                ema_map = -2.0
                if model_ema is not None:
                    answers_ema, _ = _feral_evaluate()(
                        model_ema.ema,
                        val_loader,
                        None,
                        num_classes=num_classes,
                        is_multilabel=is_multilabel,
                        device=device,
                    )
                    ema_map = _feral_calc_frame_level_map()(
                        ensembled_predictions(answers_ema, labels_json, "val"),
                        labels_json,
                        "val",
                    )
                    epoch_metrics["ema_val_frame_level_map"] = ema_map

                if val_map > ema_map and val_map > best_map:
                    save_model(best_path, model)
                    best_map = val_map
                    epochs_without_improvement = 0
                elif model_ema is not None and ema_map > val_map and ema_map > best_map:
                    save_model(best_path, model_ema.ema)
                    best_map = ema_map
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            training_metrics.append(epoch_metrics)
            log.info("Epoch %d: %s", epoch, epoch_metrics)
            csv_progress.on_epoch_end(epoch, training["epochs"], epoch_metrics)
            if progress_callback is not None:
                progress_callback.on_epoch_end(epoch, training["epochs"], epoch_metrics)

            resume_state: dict[str, object] = {
                "epoch": epoch,
                "best_map": best_map,
                "epochs_without_improvement": epochs_without_improvement,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_state_dict": model.state_dict(),
                "training_metrics": training_metrics,
            }
            if model_ema is not None:
                resume_state["ema_state_dict"] = model_ema.ema.state_dict()
            atomic_write(
                checkpoint_path, lambda temporary: torch.save(resume_state, temporary)
            )

            patience = training["patience"]
            if patience is not None and epochs_without_improvement >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break
    finally:
        csv_progress.close()

    checkpoint_path.unlink(missing_ok=True)
    if model_ema is not None:
        save_model(run_root / "model_ema.pt", model_ema.ema)

    trained = getattr(model, "_orig_mod", model)
    if val_loader is not None and best_path.exists():
        trained.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
    trained.eval()
    return trained, best_map, training_metrics


def feral_setup_check(
    checkpoint_path: str | Path | None = None,
    *,
    backbone: str = "vjepa2_vitl_diving48",
    num_classes: int | None = None,
    build_model: bool = False,
    device: str = "cpu",
) -> dict:
    """Diagnose a FERAL setup (and optionally a checkpoint) for notebooks.

    Verifies the ``feral`` package imports, resolves the backbone, and -- when
    ``checkpoint_path`` is given -- reports the checkpoint format (new dict /
    bare / remapped legacy ``HFModel``) and whether it structurally matches
    ``FeralModel``. With ``build_model=True`` it constructs ``FeralModel``
    (``pretrained=False`` -- no weight download, but the backbone *config* is
    fetched from the HF hub) and does a strict load to fully validate.

    Weights are not downloaded here, but they are downloaded elsewhere. Real
    training and inference build the backbone with ``pretrained=True``, which
    fetches the checkpoint named by the resolved ``BACKBONES`` entry from the
    HuggingFace hub on first use and caches it. Mosaic vendors none of them, and
    no hub token or license acceptance is involved for any registry entry: the
    default backbone's model card declares MIT, and none of the entries is
    gated. The licenses do differ per entry, so which backbone you name decides
    what terms the weights arrive under -- see ``NOTICE``.

    Prints a human-readable report and returns a summary dict. Designed to
    replace the old cryptic failures (``HFModel`` ImportError, silent state_dict
    mismatch) with an actionable diagnostic.

    Parameters
    ----------
    checkpoint_path : str or Path or None
        A ``model_best.pt`` to inspect. ``None`` checks only
        the package + backbone.
    backbone : str
        BACKBONES key or HF slug (default the trophallaxis ViT-L diving48 model).
    num_classes : int or None
        Needed for ``build_model=True`` when the checkpoint carries no metadata.
    build_model : bool
        Build FeralModel and strict-load the checkpoint (needs torch + network).
    device : str
        Where to place the built model (only used with ``build_model=True``).
    """
    summary: dict = {"feral_importable": False}
    print("=== FERAL setup check ===")

    # 1. Package import
    try:
        _check_feral()
    except ToolUnavailable as e:
        print(f"[FAIL] {e}")
        return summary
    import feral

    version = getattr(feral, "__version__", "?")
    summary["feral_importable"] = True
    summary["feral_version"] = version
    print(f"[ OK ] feral {version} importable")

    # 2. Backbone resolution
    from feral.backbones import BACKBONES

    try:
        key = _resolve_backbone_key(backbone)
    except ValueError as e:
        print(f"[FAIL] {e}")
        print(f"       Available backbones: {sorted(BACKBONES)}")
        return summary
    entry = BACKBONES[key]
    summary["backbone_key"] = key
    print(
        f"[ OK ] backbone {backbone!r} -> {key!r} "
        f"(hidden_dim={entry.get('hidden_dim')}, img_size={entry.get('img_size')})"
    )

    # 3. Checkpoint
    if checkpoint_path is None:
        print("(no checkpoint_path -- skipping checkpoint check)")
        return summary

    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"[FAIL] checkpoint not found: {ckpt_path}")
        return summary

    raw = torch.load(str(ckpt_path), map_location="cpu")
    is_new_dict = isinstance(raw, dict) and "state_dict" in raw
    is_old = (
        not is_new_dict
        and isinstance(raw, dict)
        and any(k.startswith("model.") for k in raw)
        and not any(k.startswith("backbone.") for k in raw)
    )
    state_dict, meta = _load_checkpoint_state_dict(raw)
    if is_new_dict:
        fmt = "new dict format (embedded metadata)"
    elif is_old:
        fmt = "legacy HFModel bare state_dict (remapped model.* -> backbone.model.*)"
    else:
        fmt = "bare state_dict (new layout)"
    summary["checkpoint_format"] = fmt
    prefixes = sorted({k.split(".")[0] for k in state_dict})
    print(f"[ OK ] checkpoint loaded: {fmt}")
    print(f"       top-level key groups: {prefixes}")
    if meta:
        if meta.get("class_names"):
            print(f"       embedded class_names: {meta['class_names']}")
        if "is_multilabel" in meta:
            print(f"       is_multilabel: {meta['is_multilabel']}")

    has_head = any(k.startswith("head.") for k in state_dict)
    has_backbone = any(k.startswith("backbone.model.") for k in state_dict)
    if not (has_head and has_backbone):
        print(
            "[WARN] checkpoint is missing expected FeralModel key groups "
            "(head.* and/or backbone.model.*) -- it may not load."
        )

    # 4. Optional full build + strict load
    if build_model:
        nclasses = num_classes
        if nclasses is None and meta and meta.get("class_names"):
            nclasses = len(meta["class_names"])
        if not nclasses:
            print(
                "[WARN] build_model=True but num_classes is unknown "
                "(pass num_classes=...). Skipping strict load."
            )
            return summary
        from feral.model import FeralModel

        predict_per_item = 64
        if meta and isinstance(meta.get("cfg"), dict):
            predict_per_item = meta["cfg"].get("predict_per_item", 64)
        model = FeralModel(
            backbone=key,
            num_classes=nclasses,
            predict_per_item=predict_per_item,
            fc_drop_rate=0.5,
            freeze_encoder_layers=0,
            pretrained=False,
        )
        try:
            _load_into(model, state_dict)
            model.to(device).eval()
            summary["strict_load_ok"] = True
            print(
                f"[ OK ] strict load into FeralModel(num_classes={nclasses}) succeeded"
            )
        except RuntimeError as e:
            summary["strict_load_ok"] = False
            print(f"[FAIL] strict load failed: {e}")

    return summary
