"""``prepare-training-data``: annotation sets in, a training dataset out.

The annotator's state reaches a dataset as revisions of the ``keypoints`` label
series. A trainer reads a directory tree. This op is the step between: it takes
**named revisions** of one or more sets -- often from several projects -- and
writes one training dataset under ``models/prepare-training-data/<run_id>/``.

Three properties are what the rest of the design leans on.

- **It names the revisions it read, and its identity is their content.** A model
  trained from this run is therefore tied to exact annotation states, and a save
  that changed nothing trains nothing new. The revision *number* selects and is
  recorded; only the content digest is hashed, so asking for "the latest" and
  asking for the number it resolves to are one run.
- **It copies the images.** The tree is complete in itself: the datasets the sets
  came from can be archived, moved or unmounted and this model stays reproducible
  from here. A symlinked tree would be a list of other people's paths.
- **It gives every image a name that cannot collide.** mosaic-extracted frames
  are all called ``frame_NNNNNN.png`` under their sequence's directory, so a
  union keyed on basename would silently overwrite one sequence's frame 12 with
  another's -- image and label both.
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationSet,
    KeypointSchema,
)
from mosaic.core.annotations.narrow import AliasRole, narrow_pose_sets, narrowed_pose
from mosaic.core.annotations.pose_annotations import PoseAnnotationSet, PoseDefinition
from mosaic.core.annotations.split import (
    SplitBy,
    Subset,
    VideoFrame,
    frame_key,
    recording_unit_key,
    sequence_key,
    split_filenames,
    unit_key,
)
from mosaic.core.entry import make_entry_key, to_safe_name
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    project_to_schema,
)
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.models import (
    SPLIT_RECORD_FILENAME,
    model_index_path,
    model_run_root,
)
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.tracking.frame_extraction.frame_runs import video_frames_of
from mosaic.tracking.keypoint_sets import (
    KeypointSetRef,
    ResolvedSet,
    consumed_sets_cell,
    in_identity_order,
    keypoint_schema_cell,
    read_resolved_set,
    resolve_keypoint_sets,
    revision_image_root,
    set_label,
)

from ._common import claim_run_root, empty_claimed_run_root, ensure_models_root

if TYPE_CHECKING:
    import pandas as pd

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

__all__ = [
    "PrepareTrainingDataOp",
    "PrepareTrainingDataParams",
    "PreparedDatasetIndexRow",
    "prepare_training_data_run_id",
    "prepared_dataset_index",
]

PREPARE_KIND: Final = "prepare-training-data"
_PREPARE_IDLE_SECONDS: Final = 600.0

PrepareTarget = Literal["yolo-pose", "polo", "sleap", "litpose"]
"""Which trainer the dataset is written for, and so what the artifact is.

``yolo-pose`` and ``polo`` are a split directory tree named by its ``data.yaml``.
``sleap`` is a ``labels.slp`` of the training frames with a ``labels.val.slp`` of
the validation frames beside it, split by ``split_by`` and the first two
fractions, and its test frames are written to neither. ``litpose`` is a Lightning
Pose project directory. Only Lightning Pose draws its own train and validation
split, so ``split_by`` and the fractions beyond the first do not apply to it.
Selecting, naming and recording revisions is the same for all four.
"""

_TREE_TARGETS: Final = frozenset({"yolo-pose", "polo"})
_SPLIT_TARGETS: Final = frozenset({"yolo-pose", "polo", "sleap"})
"""The targets refused a split that cannot fill both training and validation."""

_UNIT_NOUNS: Final[dict[SplitBy, tuple[str, str]]] = {
    "recording": ("recording", "recordings"),
    "sequence": ("sequence", "sequences"),
    "group": ("group", "groups"),
    "frame": ("frame", "frames"),
}
"""Each ``split_by`` unit's noun, singular and plural."""

PlacedSplit = Subset | Literal["excluded"]
"""Where a prepared frame is placed: a split, or left out for the hold-out."""

_PLACED_SPLITS: Final[dict[str, PlacedSplit]] = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}


# --- Params --------------------------------------------------------------------


class PrepareTrainingDataParams(Params):
    """Parameters for the ``prepare-training-data`` op.

    The op reads revisions from ``labels_raw/keypoints/index.csv``, never a media
    entry, so it declares no scope and its ``scope_takes`` is ``"none"``.
    """

    sets: Annotated[
        tuple[KeypointSetRef, ...],
        Field(min_length=1),
        Declared("The annotation sets to train on, merged into one dataset."),
    ]
    target: Annotated[PrepareTarget, Declared("Which trainer's layout to write.")] = (
        "yolo-pose"
    )
    pose: Annotated[
        int | str | None,
        Declared(
            "Which pose to train, by its id or its name. Left unset, every set "
            "must hold exactly one pose. One dataset has one keypoint layout."
        ),
    ] = None
    class_by: Annotated[
        AliasRole | None,
        Declared(
            "alias makes each alias a class, and refuses an object of the pose "
            "without one. Unset, the pose is the one class. sleap and litpose "
            "train one class."
        ),
    ] = None
    track_by: Annotated[
        AliasRole | None,
        Declared(
            "sleap only: alias gives each object its alias as its track, which "
            "is what a SLEAP identity model learns."
        ),
    ] = None
    bbox: Annotated[
        BboxPolicy | None,
        Declared(
            "How a box the annotator did not draw is derived from its keypoints. "
            "Unset, each set's own padding: the box the annotator saw. A drawn "
            "box is used as drawn."
        ),
    ] = None
    point_index: Annotated[
        int,
        Field(ge=0),
        Declared("polo only: which keypoint of each instance is the point."),
    ] = 0
    radius: Annotated[
        float,
        Field(gt=0),
        Declared("polo only: the detection radius.", unit="px"),
    ] = 100.0
    split: Annotated[
        tuple[float, float, float],
        Declared("Train, validation and test fractions."),
    ] = (0.8, 0.15, 0.05)
    split_by: Annotated[
        SplitBy,
        Declared(
            "What is kept together when the split is drawn. recording keeps the "
            "frames of one recording in one split: every sequence of a divided "
            "recording, or one sequence. A validation score then measures "
            "recordings the model has not seen. sequence keeps one sequence's "
            "frames together, and the sequences of a divided recording can "
            "straddle the split. group keeps a whole sequence_groups label "
            "together. frame splits image by image, and its scores are optimistic."
        ),
    ] = "recording"
    sequence_groups: Annotated[
        dict[str, str],
        Declared(
            "group only: the group each sequence belongs to, keyed by sequence "
            "name or by '<origin_uuid>:<group>:<sequence>', each part "
            "percent-encoded, where names repeat. A sequence not listed is a group "
            "of its own."
        ),
    ] = Field(default_factory=dict)
    holdout: Annotated[
        tuple[KeypointSetRef, ...],
        Declared(
            "Annotation sets kept out of training for evaluation. Every frame of sets "
            "that shares a split_by unit with a hold-out frame is left out."
        ),
    ] = ()
    seed: Annotated[int, Declared("Random seed for the split assignment.")] = 42
    # Execution knob: how images are placed, never what the dataset holds.
    symlink_images: Annotated[
        bool,
        HASH_EXCLUDE,
        Declared(
            "Symlink images instead of copying them. Off by default: a symlinked "
            "tree stops being reproducible the moment a source dataset moves."
        ),
    ] = False

    @model_validator(mode="after")
    def _sequence_groups_only_under_a_group_split(self) -> Self:
        """Refuse ``sequence_groups`` under ``split_by="recording"``.

        A recording split draws the recordings that the datasets declare, and a
        label would regroup them.
        """
        if self.split_by == "recording" and self.sequence_groups:
            msg = (
                "sequence_groups applies only under split_by='group', and "
                "split_by='recording' draws the recordings that the datasets "
                "declare. Set split_by to 'group', or leave sequence_groups out."
            )
            raise ValueError(msg)
        return self


# --- Index ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedDatasetIndexRow(RunIndexRowBase):
    """Typed row for ``models/prepare-training-data/index.csv``.

    ``artifact_path`` is root-relative and already listed for the ``models`` root
    in ``_INDEX_PATH_COLUMNS``. ``consumed_sets`` is the provenance link from a
    prepared dataset back to the annotation revisions behind it: a JSON list of
    ``{origin_uuid, set_key, revision, digest}``, in the sorted order the run
    identifier was taken over. ``held_out_sets`` lists the hold-out's revisions
    the same way (``"[]"`` with none), ``n_held_out`` counts the frames of
    ``sets`` its units remove, and ``keypoint_schema`` is the JSON
    ``{keypoints, skeleton, symmetries}`` that a model trained on the
    preparation predicts. None of the three enters the run identifier.

    ``n_train``, ``n_valid`` and ``n_test`` count the frames written in each
    split that ``split.csv`` records, and sum to ``n_frames``. A ``sleap``
    preparation writes no ``test`` frame, so its ``n_test`` is 0. For a
    ``litpose`` preparation the three counts are mosaic's draw, and Lightning
    Pose redraws its own train and validation split from all ``n_frames``.
    """

    kind: str
    target: str
    artifact_path: str
    consumed_sets: str
    n_frames: int
    n_train: int
    n_valid: int
    n_test: int
    status: str
    # The pose the sets were narrowed to: the authoring store's id and the name
    # it had then. Provenance, never identity: ``pose`` entered the run
    # identifier as the caller spelled it, and this records which pose that
    # chose. Blank on a row written before these columns, meaning unknown.
    pose_id: str = ""
    pose_name: str = ""
    # Blank on a row written before these columns, meaning unknown.
    held_out_sets: str = ""
    n_held_out: int = 0
    keypoint_schema: str = ""


_PREPARED_DATASET_INDEX_COLUMNS: Final = tuple(
    field.name for field in fields(PreparedDatasetIndexRow)
)


def _adopt_prepared_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a prepared-data index read off disk up to the current schema.

    Runs inside the write lock, so an index is brought forward the first time a
    preparation is recorded in it, and never by a read.
    """
    return project_to_schema(df, _PREPARED_DATASET_INDEX_COLUMNS)


def prepared_dataset_index(path: Path) -> IndexCSV[PreparedDatasetIndexRow]:
    return IndexCSV(
        path,
        PreparedDatasetIndexRow,
        dedup_keys=["run_id"],
        adopt=_adopt_prepared_dataset_columns,
    )


def prepare_training_data_run_id(
    kind: str,
    version: str,
    params: PrepareTrainingDataParams,
    sets: tuple[ResolvedSet, ...],
    holdout_units: Collection[str] = (),
    divided_recordings: Collection[tuple[str, str]] = (),
) -> str:
    """Mint a preparation run identifier.

    A named function so the payload is one thing to read and something the golden
    corpus can call with fixed arguments and no filesystem.

    ``sets`` leaves the parameter dump and comes back as content. What the caller
    typed there -- a revision number, or none -- is a *selector*, and two spellings
    that select the same bytes must be one run. The triples are sorted, so the
    order sets were named in is not a difference either; the split below draws
    each unit from the unit's key for the same reason.

    ``holdout`` leaves the dump too. The split units it removes come back sorted,
    and only when there are any. A preparation without a hold-out keeps the
    identifier it had before the field existed. A label correction inside the
    hold-out names the same preparation.

    ``divided_recordings`` are the ``(origin_uuid, group)`` pairs of the divided
    recordings that hold a sequence of the sets' frames, which a recording split
    draws as one unit in place of each sequence. They enter sorted, and only when
    there are any, so declaring a continuous group names another preparation
    exactly when it changes a unit.
    """
    dumped = params.identity_dump()
    _ = dumped.pop("sets", None)
    _ = dumped.pop("holdout", None)
    payload: dict[str, object] = {
        "params": dumped,
        "sets": sorted([item.origin_uuid, item.set_key, item.digest] for item in sets),
    }
    if holdout_units:
        payload["holdout"] = sorted(holdout_units)
    if divided_recordings:
        payload["recordings"] = sorted(
            [origin, group] for origin, group in divided_recordings
        )
    return op_run_id(kind, version, payload)


def check_preparation(ds: Dataset, params: PrepareTrainingDataParams) -> None:
    """Refuse what a ``prepare-training-data`` run would refuse, without writing anything.

    Resolves the revisions and narrows them exactly as a run does, so a caller that
    queues the preparation, and a training behind it, hears now that the sets hold
    several poses and none is named, that an object lacks the alias ``class_by`` asks
    for, or that a one-class trainer is handed several.

    Raises:
        ValueError: The sets cannot be narrowed to one training dataset; a hold-out
            set is also named in ``sets``, lacks a usable frame for the pose, or has
            another keypoint layout; the hold-out removes every training frame; a
            ``sequence_groups`` key matches none of the sequences of the sets and
            the hold-out; a recording split reads a group of a dataset out of
            reach; or the split would leave a trainer with no training or
            validation images.
        KeyError: A reference matches several datasets' sets and names none.
        IdentityDeferred: A revision is not indexed here, or not on disk.
    """
    training = _read_sets(resolve_keypoint_sets(ds, params.sets, kind=PREPARE_KIND))
    narrowed, pose = _narrowed(params, training)
    holdout = _resolve_holdout(ds, params, pose, training.sets)
    _ = _split(ds, params, training, narrowed, holdout)


# --- Building the tree ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitRow:
    """One narrowed usable frame of the sets, and where the preparation places it.

    Attributes:
        origin_uuid: The origin of the revision the frame came from.
        set_key: The set the frame came from.
        revision: The revision the frame came from.
        image: The revision's image path.
        video_frame: The recorded frame behind the image.
        frame_key: ``split.frame_key`` of *video_frame*.
        unit_key: The unit that ``split_by`` keeps together.
        placed_name: The image's file name in the prepared dataset.
        split: Where the frame is placed.
    """

    origin_uuid: str
    set_key: str
    revision: int
    image: str
    video_frame: VideoFrame
    frame_key: str
    unit_key: str
    placed_name: str
    split: PlacedSplit


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Everything a preparation writes about its split.

    A placed name is not parsed back. Every fact about a frame is recorded here
    when its name is made.

    Attributes:
        name_by_frame: Each placed frame's name, keyed by the frame's ``id``.
        split_of: Each placed name's split.
        rows: Every narrowed usable frame, sorted by placed name.
        n_held_out: The rows left out for the hold-out.
    """

    name_by_frame: dict[int, str]
    split_of: dict[str, str]
    rows: tuple[SplitRow, ...]
    n_held_out: int


@dataclass(frozen=True, slots=True)
class ReadSets:
    """Revisions in identity order, and the saved state of each, read once.

    Attributes:
        sets: The revisions in identity order.
        states: The saved state of each revision, keyed by its ``set_label``.
    """

    sets: list[ResolvedSet]
    states: dict[str, PoseAnnotationSet]


def _read_sets(sets: Sequence[ResolvedSet]) -> ReadSets:
    """Order *sets* as the run identifier sorts them, and read each saved state."""
    ordered = in_identity_order(sets)
    return ReadSets(
        sets=ordered,
        states={set_label(item): read_resolved_set(item) for item in ordered},
    )


@dataclass(frozen=True, slots=True)
class HoldOut:
    """The sets kept out of training, and the recorded frames behind them.

    Attributes:
        sets: The hold-out's revisions, in identity order.
        states: The saved state of each hold-out revision, keyed by its
            ``set_label``.
        frames: The recorded frame behind every usable frame of the hold-out,
            narrowed to the training pose.
    """

    sets: tuple[ResolvedSet, ...] = ()
    states: dict[str, PoseAnnotationSet] = field(
        default_factory=dict[str, PoseAnnotationSet]
    )
    frames: tuple[VideoFrame, ...] = ()


@dataclass(frozen=True, slots=True)
class SplitUnits:
    """How a preparation keys its frames into split units.

    Attributes:
        recording_units: The ``recording_units`` of :func:`unit_key`: each
            sequence key of a member of a divided recording, mapped to its
            :func:`recording_unit_key`. Empty unless ``split_by`` is
            ``"recording"``.
        holdout: The ``split_by`` unit of every frame of the hold-out.
        divided_recordings: The ``(origin_uuid, group)`` pairs of the divided
            recordings that hold a sequence of the training sets.
    """

    recording_units: dict[str, str] = field(default_factory=dict[str, str])
    holdout: frozenset[str] = frozenset()
    divided_recordings: tuple[tuple[str, str], ...] = ()


def _recorded(
    ds: Dataset, item: ResolvedSet, annotations: AnnotationSet
) -> tuple[VideoFrame, ...]:
    """Return the recorded frame behind each frame of *annotations*, of *item*."""
    return video_frames_of(
        ds,
        revision_image_root(annotations.image_root),
        annotations.frames,
        revision_origin=item.origin_uuid,
    )


def _resolve_holdout(
    ds: Dataset,
    params: PrepareTrainingDataParams,
    pose: PoseDefinition,
    sets: Sequence[ResolvedSet],
) -> HoldOut:
    """Resolve ``holdout``, narrow it to the training pose, and read its frames.

    Reads nothing when ``holdout`` is empty. Each hold-out set is checked against
    a set that declares only the training pose. A set without that pose is refused
    first, then a set without a usable frame, then a set of another keypoint
    layout.

    Args:
        ds: The dataset that reads the revisions.
        params: The preparation's parameters.
        pose: The training pose, which the training sets were narrowed to.
        sets: The training sets in identity order.

    Raises:
        ValueError: A hold-out set is also named in ``sets``, chose another pose,
            lacks a usable frame or has another keypoint layout, or two references
            resolve to one set.
        KeyError: A reference matches several datasets' sets and names none.
        IdentityDeferred: A hold-out revision is not indexed here, or not on disk.
    """
    if not params.holdout:
        return HoldOut()
    resolved = resolve_keypoint_sets(ds, params.holdout, kind=PREPARE_KIND)
    trained = {(item.origin_uuid, item.set_key) for item in sets}
    for item in resolved:
        if (item.origin_uuid, item.set_key) in trained:
            msg = (
                f"keypoint set {item.set_key!r} is named in both sets and holdout. "
                "Train on it or hold it out."
            )
            raise ValueError(msg)
    held = _read_sets(resolved)
    ordered = held.sets
    # The pose and layout checks compare every set with the first, the training pose's.
    reference = {set_label(sets[0]): PoseAnnotationSet(poses=(pose,))}
    states = {**reference, **held.states}
    _ = narrowed_pose(states, pose=params.pose)
    for item in ordered:
        if not any(frame.usable for frame in states[set_label(item)].frames):
            msg = (
                f"hold-out set {item.set_key!r} has no usable frames for pose "
                f"{pose.name!r}"
            )
            raise ValueError(msg)
    narrowed = narrow_pose_sets(states, pose=params.pose)
    frames = tuple(
        frame
        for item in ordered
        for frame in _recorded(ds, item, narrowed[set_label(item)])
    )
    return HoldOut(sets=tuple(ordered), states=held.states, frames=frames)


def _state_frames(
    ds: Dataset,
    revisions: Sequence[ResolvedSet],
    states: Mapping[str, PoseAnnotationSet],
) -> list[VideoFrame]:
    """Return the recorded frame behind every frame of each revision's saved state.

    Every frame counts, of every pose and finished or not, so the result depends on
    the revisions' content alone. *states* holds each revision's saved state, keyed
    by its ``set_label``.
    """
    found: list[VideoFrame] = []
    for item in revisions:
        state = states[set_label(item)]
        found.extend(
            video_frames_of(
                ds,
                revision_image_root(state.image_root),
                [frame.as_annotation_frame() for frame in state.frames],
                revision_origin=item.origin_uuid,
            )
        )
    return found


def _split_units(
    ds: Dataset,
    params: PrepareTrainingDataParams,
    training: ReadSets,
    holdout: HoldOut,
) -> SplitUnits:
    """Key the hold-out's units, and under a recording split the recordings.

    The recordings are read from every frame of the saved states of the training
    sets and of the hold-out, so a plan that narrows nothing keys them as a run
    does.

    Raises:
        ValueError: Under ``split_by="recording"``, a frame of a non-empty group
            comes from a dataset that is neither *ds* nor one that it links.
    """
    recording_units: dict[str, str] = {}
    divided: tuple[tuple[str, str], ...] = ()
    if params.split_by == "recording":
        trained = _state_frames(ds, training.sets, training.states)
        recording_units = _recording_units(
            ds, [*trained, *_state_frames(ds, holdout.sets, holdout.states)]
        )
        divided = _divided_recordings(trained, recording_units)
    held = frozenset(
        unit_key(frame, params.split_by, params.sequence_groups, recording_units)
        for frame in holdout.frames
    )
    return SplitUnits(
        recording_units=recording_units, holdout=held, divided_recordings=divided
    )


def _recording_units(ds: Dataset, frames: Sequence[VideoFrame]) -> dict[str, str]:
    """Map the sequence key of each frame of a divided recording to its unit.

    The dataset whose manifest uuid is the frame's origin declares whether the
    frame's group is a divided recording. That dataset is *ds* or one that *ds*
    links. A frame of the empty group belongs to no divided recording, and its
    origin is not looked up.

    Raises:
        ValueError: A frame of a non-empty group comes from a dataset that is
            neither *ds* nor a reachable library that it links.
    """
    owners: dict[str, Dataset] | None = None
    units: dict[str, str] = {}
    for frame in frames:
        if not frame.group:
            continue
        if owners is None:
            owners = {
                library.dataset.uuid or "": library.dataset
                for library in ds.linked_libraries()
            }
            owners[ds.uuid or ""] = ds
        owner = owners.get(frame.origin_uuid)
        if owner is None:
            raise ValueError(_unreachable_origin(ds, frame))
        if owner.recording_kind(frame.group) == "divided":
            units[sequence_key(frame)] = recording_unit_key(frame)
    return units


def _unreachable_origin(ds: Dataset, frame: VideoFrame) -> str:
    """Say why *ds* cannot read the declaration of *frame*'s origin dataset."""
    asked = (
        f"split_by='recording' reads whether group {frame.group!r} of dataset "
        f"{frame.origin_uuid!r} is a divided recording"
    )
    for link in ds.manifest.libraries:
        if link.uuid == frame.origin_uuid:
            return (
                f"{asked}, and the library {link.id!r} that links it is not "
                "reachable now. Mount it, or split by sequence."
            )
    return (
        f"{asked}, and this dataset neither is that dataset nor links it. Link it "
        "under libraries, or split by sequence."
    )


def _divided_recordings(
    frames: Sequence[VideoFrame], recording_units: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    """Return the divided recordings that hold a sequence of *frames*.

    Each is its ``(origin_uuid, group)`` pair, in sorted order. A recording of
    one annotated sequence counts: its sequence is drawn as the recording.
    """
    return tuple(
        sorted(
            {
                (frame.origin_uuid, frame.group)
                for frame in frames
                if sequence_key(frame) in recording_units
            }
        )
    )


def _split(
    ds: Dataset,
    params: PrepareTrainingDataParams,
    training: ReadSets,
    narrowed: dict[str, AnnotationSet],
    holdout: HoldOut,
) -> SplitPlan:
    """Name every narrowed frame, key its unit, and draw each unit's split.

    Each frame's unit is keyed from the recorded frame behind its image, which
    :func:`video_frames_of` reads from the extraction that wrote it. A unit key
    omits the set and the camera. One recorded instant saved in two sets or seen
    by two cameras is therefore one unit under ``split_by="frame"``. Under
    ``split_by="recording"`` the sequences of one divided recording are one unit.
    A frame in a unit of the hold-out is excluded before the draw. Each remaining
    unit's split is drawn from the unit's key. The order sets were named in and
    the order a file lists its frames in do not change the assignment.

    Raises:
        ValueError: A ``sequence_groups`` key matches none of the sequences of the
            sets and the hold-out under ``split_by="group"``, a recording split
            reads a group of a dataset out of reach, the hold-out removes every
            frame, or a target that trains on this split draws fewer than two
            units or has a validation or training share of 0.
    """
    placed: list[tuple[ResolvedSet, AnnotationFrame, VideoFrame, str]] = []
    for item in training.sets:
        annotations = narrowed[set_label(item)]
        placed.extend(
            (item, frame, video_frame, _placed_name(item, frame, video_frame))
            for frame, video_frame in zip(
                annotations.frames, _recorded(ds, item, annotations), strict=True
            )
        )
    if params.split_by == "group":
        _refuse_unknown_sequence_groups(ds, params, training, holdout)
    units = _split_units(ds, params, training, holdout)

    unit_of = {
        name: unit_key(
            video_frame, params.split_by, params.sequence_groups, units.recording_units
        )
        for _item, _frame, video_frame, name in placed
    }
    kept = {name: unit for name, unit in unit_of.items() if unit not in units.holdout}
    if units.holdout and not kept:
        singular, plural = _UNIT_NOUNS[params.split_by]
        msg = (
            f"the hold-out shares a {singular} with every training frame. Choose a "
            f"hold-out from other {plural}"
        )
        msg += "." if params.split_by == "frame" else ", or split by frame."
        raise ValueError(msg)
    if params.target in _SPLIT_TARGETS:
        _refuse_small_split(params, len(set(kept.values())))
    split_of, _, _ = split_filenames(
        sorted(kept),
        params.split,
        params.seed,
        split_by="group",
        group_key=kept.__getitem__,
    )
    rows = [
        SplitRow(
            origin_uuid=item.origin_uuid,
            set_key=item.set_key,
            revision=item.revision,
            image=frame.image_path.as_posix(),
            video_frame=video_frame,
            frame_key=frame_key(video_frame),
            unit_key=unit_of[name],
            placed_name=name,
            split=_PLACED_SPLITS[split_of[name]] if name in kept else "excluded",
        )
        for item, frame, video_frame, name in placed
    ]
    return SplitPlan(
        name_by_frame={
            id(frame): name for _item, frame, _video, name in placed if name in kept
        },
        split_of=split_of,
        rows=tuple(sorted(rows, key=lambda row: row.placed_name)),
        n_held_out=sum(row.split == "excluded" for row in rows),
    )


def _refuse_unknown_sequence_groups(
    ds: Dataset,
    params: PrepareTrainingDataParams,
    training: ReadSets,
    holdout: HoldOut,
) -> None:
    """Refuse a ``sequence_groups`` key that matches no sequence of sets or hold-out.

    A key matches the sequence key or the bare sequence name of any frame of the
    saved states of ``sets`` and ``holdout``, of
    every pose and finished or not. A key for a hold-out recording puts it in its
    group, whose training recordings the hold-out then removes. A key for a
    recording that this preparation leaves out is accepted and unused. A two-part
    key whose second part names a sequence is the ``<origin_uuid>:<sequence>``
    spelling that 0.2 read, and its refusal gives the key to use instead.

    Raises:
        ValueError: A key matches none of the sequences of the sets and the
            hold-out.
    """
    if not params.sequence_groups:
        return
    keys_by_name: dict[str, set[str]] = {}
    recorded = [
        *_state_frames(ds, training.sets, training.states),
        *_state_frames(ds, holdout.sets, holdout.states),
    ]
    for video_frame in recorded:
        keys_by_name.setdefault(video_frame.sequence, set()).add(
            sequence_key(video_frame)
        )
    known = set(keys_by_name).union(*keys_by_name.values())
    unmatched = sorted(key for key in params.sequence_groups if key not in known)
    if not unmatched:
        return
    messages: list[str] = []
    generic: list[str] = []
    for key in unmatched:
        parts = key.split(":")
        if len(parts) == 2 and parts[1] in keys_by_name:
            suggested = " or ".join(
                repr(each) for each in sorted(keys_by_name[parts[1]])
            )
            messages.append(
                f"sequence_groups key {key!r} uses the <origin_uuid>:<sequence> "
                f"spelling, which {PREPARE_KIND} 0.3 no longer reads. Key it as "
                f"{suggested}."
            )
        else:
            generic.append(key)
    if generic:
        listed = ", ".join(repr(key) for key in generic)
        messages.append(
            "these sequence_groups keys match no sequence of sets or holdout: "
            f"{listed}. Key a sequence by its name, or by "
            "<origin_uuid>:<group>:<sequence> with each part percent-encoded where "
            "names repeat."
        )
    raise ValueError(" ".join(messages))


def _refuse_small_split(params: PrepareTrainingDataParams, n_units: int) -> None:
    """Refuse a split that cannot fill both a training and a validation split.

    From two units on, the splitter fills training and validation whenever both
    shares are positive.

    Raises:
        ValueError: Fewer than two units are drawn, or the validation or the
            training share is 0.
    """
    f_train, f_valid, _f_test = params.split
    if n_units < 2:
        msg = (
            "the training and validation splits need at least 2 "
            f"{_UNIT_NOUNS[params.split_by][1]} and split_by={params.split_by!r} "
            f"draws {n_units}."
        )
        if params.split_by != "frame":
            msg += (
                f" Annotate frames from another {_UNIT_NOUNS[params.split_by][0]}, "
                "or split by frame."
            )
        raise ValueError(msg)
    if f_valid == 0:
        msg = (
            f"the validation share is 0 and {params.target} needs a validation "
            "split. Give split a validation share above 0."
        )
        raise ValueError(msg)
    if f_train == 0:
        msg = "the training share is 0. Give split a training share above 0."
        raise ValueError(msg)


def _narrowed(
    params: PrepareTrainingDataParams, training: ReadSets
) -> tuple[dict[str, AnnotationSet], PoseDefinition]:
    """Each training set narrowed as this run reads it, and the pose.

    Narrowed together, in the order the run identifier sorts them, so the classes and
    the layout they are checked against do not depend on the order the sets were named
    in.

    Raises:
        ValueError: The sets cannot be narrowed to one pose's layout, or a trainer
            that learns one class is handed several.
    """
    states = training.states
    narrowed = narrow_pose_sets(
        states,
        pose=params.pose,
        class_by=params.class_by,
        track_by=params.track_by if params.target == "sleap" else None,
        bbox=params.bbox,
    )
    categories = narrowed[set_label(training.sets[0])].categories
    if params.target not in _TREE_TARGETS and len(categories) > 1:
        msg = (
            f"{params.target} trains one class, and class_by="
            f"{params.class_by!r} makes {len(categories)}: {list(categories)}"
        )
        raise ValueError(msg)
    return narrowed, narrowed_pose(states, pose=params.pose)


def _placed_name(
    item: ResolvedSet, frame: AnnotationFrame, video_frame: VideoFrame
) -> str:
    """A file name no other frame in the union can have.

    ``<origin>__<set>__<entry>[__<camera>]__frame_<index>``: the dataset, because
    sets from two projects may share a key; the set, because two sets may contain
    the same frame; the entry key, because two groups may use one sequence name
    and every sequence numbers its frames from the same place; the camera,
    because the cameras of one entry share every frame index. The index is the
    recorded frame's, or its image stem when that is unknown. Each part is
    percent-encoded so a name with a slash stays one component, and ``__frame``
    is kept as the last separator because that is what the default split
    grouping reads.
    """
    index = (
        f"{video_frame.frame_index:06d}"
        if video_frame.frame_index >= 0
        else to_safe_name(video_frame.image_stem)
    )
    origin = (item.origin_uuid or "local")[:8]
    entry = make_entry_key(video_frame.group, video_frame.sequence)
    camera = f"__{to_safe_name(video_frame.camera)}" if video_frame.camera else ""
    stem = f"{origin}__{to_safe_name(item.set_key)}__{entry}{camera}__frame_{index}"
    return f"{stem}{frame.image_path.suffix.lower() or '.png'}"


# --- Op ------------------------------------------------------------------------


@register_op
class PrepareTrainingDataOp(Op[PrepareTrainingDataParams]):
    """Merge revisions of keypoint annotation sets into one training dataset."""

    kind = PREPARE_KIND
    category = "convert"
    domain = "tracking"
    # 0.3: the split draws each recording, group or frame by a seeded hash of its
    # key. Adding frames leaves every unit already drawn in its split unless the
    # small-set guard moves it. A placed name contains the entry's group and
    # camera, read from the extraction that wrote the image. split.csv records
    # each frame's placed name and split beside the split directories. A
    # hold-out removes every training frame that shares a split unit with it,
    # and its units enter the identifier only when there are any. For sleap, the
    # training frames are written to labels.slp and the validation frames to
    # labels.val.slp beside it, and the test frames to neither. Under
    # track_by="alias", each track of those files is named by its alias.
    # split_by="recording", the default since it was added within 0.3, draws
    # the sequences of a divided recording as one unit. Without a divided
    # recording it draws as split_by="sequence".
    # 0.2: a revision is the full saved state, narrowed here, and a derived box
    # is the annotator's padded one rather than a hull read back as drawn.
    version = "0.3"
    scope_takes = "none"
    scope_dependent = False
    Params = PrepareTrainingDataParams

    def target(self, params: PrepareTrainingDataParams, scope: ResolvedScope) -> str:
        return params.target

    def plan_identity(
        self,
        ds: Dataset,
        params: PrepareTrainingDataParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this preparation will be called.

        The annotation revisions enter by content, so a changed state is a
        different dataset and an unchanged one is this one. Content has to be
        read, so a preparation whose revisions are not on disk yet is not
        nameable, and says so. A hold-out enters by the split units it removes,
        which are keyed from the hold-out narrowed to the training pose. A
        recording split enters by the divided recordings that hold a sequence of
        the sets. Only a preparation with a hold-out narrows its sets here.
        """
        _ = (scope, require_data)
        sets = resolve_keypoint_sets(ds, params.sets, kind=self.kind)
        units = SplitUnits()
        if params.holdout or params.split_by == "recording":
            training = _read_sets(sets)
            holdout = HoldOut()
            if params.holdout:
                _narrowed_sets, pose = _narrowed(params, training)
                holdout = _resolve_holdout(ds, params, pose, training.sets)
            units = _split_units(ds, params, training, holdout)
        return OpIdentity(
            run_id=prepare_training_data_run_id(
                self.kind,
                self.version,
                params,
                sets,
                units.holdout,
                units.divided_recordings,
            )
        )

    def run(
        self,
        ds: Dataset,
        params: PrepareTrainingDataParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        from mosaic.tracking.pose_training.converters.base import (
            format_polo_label_line,
            normalize_coords,
        )
        from mosaic.tracking.pose_training.converters.emit import (
            write_split_tree,
            yolo_pose_line,
        )
        from mosaic.tracking.pose_training.prep import (
            make_data_yaml,
            make_polo_data_yaml,
        )

        ensure_models_root(ds)
        # Through plan_identity, so this run is named in exactly one place. The
        # sets are resolved again for their paths; they are small files, and a
        # second copy of the minting call is the one that drifts.
        run_id = self.plan_identity(ds, params, scope).run_id
        sets = resolve_keypoint_sets(ds, params.sets, kind=self.kind)
        ctx.set_run_id(run_id)
        out = model_run_root(ds, self.kind, run_id)
        data_yaml = out / "data.yaml"

        index = prepared_dataset_index(model_index_path(ds, self.kind))
        if not overwrite and _already_prepared(ds, index, run_id, params.target):
            ctx.cache_hit()
            return run_id

        # Claimed before anything is cleared: two executions of one identifier
        # would otherwise have one delete the other's tree mid-write.
        out.mkdir(parents=True, exist_ok=True)
        _ = claim_run_root(ds, ctx.execution_id, out, self.kind, _PREPARE_IDLE_SECONDS)
        empty_claimed_run_root(out)

        training = _read_sets(sets)
        narrowed, pose = _narrowed(params, training)
        holdout = _resolve_holdout(ds, params, pose, training.sets)
        plan = _split(ds, params, training, narrowed, holdout)
        reference = narrowed[set_label(training.sets[0])]
        schema, categories = reference.schema, reference.categories
        class_ids = reference.category_ids()

        # One flat list of (frame, image on disk) over the frames the plan
        # places. The image of a frame that the hold-out removes is not read and
        # may be missing.
        gathered: list[tuple[AnnotationFrame, Path]] = []
        missing: list[str] = []
        for item in training.sets:
            annotations = narrowed[set_label(item)]
            for frame in annotations.frames:
                if id(frame) not in plan.name_by_frame:
                    continue
                source = annotations.resolve(frame)
                if not source.is_file():
                    missing.append(str(source))
                    continue
                gathered.append((frame, source))
        if missing:
            shown = ", ".join(missing[:3])
            more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
            msg = (
                f"{len(missing)} annotated image(s) are not on disk: {shown}{more}. "
                "A revision names the exact images that were annotated, so a "
                "dataset missing some of them is not the dataset that was saved."
            )
            raise FileNotFoundError(msg)

        def name_of(frame: AnnotationFrame, _source: Path) -> str:
            return plan.name_by_frame[id(frame)]

        # Every box is explicit by now: narrowing derived the ones the
        # annotator did not draw, under the policy this run was given.
        def pose_lines(frame: AnnotationFrame) -> list[str]:
            rows = (
                yolo_pose_line(
                    obj, frame.width, frame.height, class_id=class_ids[obj.category]
                )
                for obj in frame.objects
            )
            return [row for row in rows if row is not None]

        def point_lines(frame: AnnotationFrame) -> list[str]:
            rows: list[str] = []
            for obj in frame.objects:
                if params.point_index >= len(obj.keypoints):
                    continue
                point = obj.keypoints[params.point_index]
                if point.visibility == 0:
                    continue
                x, y = normalize_coords(point.x, point.y, frame.width, frame.height)
                rows.append(
                    format_polo_label_line(class_ids[obj.category], params.radius, x, y)
                )
            return rows

        ordered = sorted(gathered, key=lambda pair: name_of(pair[0], pair[1]))
        if params.target not in _TREE_TARGETS:
            artifact, n_by_split = _write_for_tool(
                params, out, ordered, name_of, plan.split_of, schema, categories
            )
            _write_split_record(out, plan.rows)
            return _register(
                ds, index, run_id, self.kind, params, sets, out, artifact,
                pose=pose, n_frames=sum(n_by_split.values()),
                n_by_split=n_by_split, holdout=holdout,
                n_held_out=plan.n_held_out, predicts=schema,
            )  # fmt: skip
        written, skipped = write_split_tree(
            ordered,
            out,
            plan.split_of,
            pose_lines if params.target == "yolo-pose" else point_lines,
            symlink_images=params.symlink_images,
            name_of=name_of,
        )
        if written == 0:
            raise ValueError(
                f"{self.kind} wrote no labels: {skipped} frame(s) held no "
                "instance this target can express"
            )

        if params.target == "yolo-pose":
            _ = make_data_yaml(
                out,
                class_ids,
                kpt_shape=schema.kpt_shape,
                # Only when the pose pairs its keypoints: without pairs,
                # Ultralytics turns flips off, which is right for them.
                flip_idx=schema.flip_idx if schema.symmetries else None,
                portable=True,
            )
        else:
            _ = make_polo_data_yaml(
                out,
                dict(class_ids),
                {class_id: params.radius for class_id in class_ids.values()},
                portable=True,
            )

        _write_split_record(out, plan.rows)
        return _register(
            ds, index, run_id, self.kind, params, sets, out, data_yaml,
            pose=pose, n_frames=written,
            n_by_split={split: _count(out / split) for split in _PLACED_SPLITS},
            holdout=holdout, n_held_out=plan.n_held_out, predicts=schema,
        )  # fmt: skip


_SPLIT_RECORD_COLUMNS: Final = (
    "origin_uuid",
    "set_key",
    "revision",
    "image",
    "group",
    "sequence",
    "camera",
    "frame_index",
    "frame_key",
    "unit_key",
    "placed_name",
    "split",
)


def _write_split_record(out: Path, rows: tuple[SplitRow, ...]) -> None:
    """Write ``split.csv``: where the preparation placed each frame.

    The file sits beside the split directories and outside the roots that
    ``data.yaml`` declares, so it does not enter the dataset's fingerprint. For
    ``litpose`` it records the draw made here, which Lightning Pose redraws from
    every frame not excluded.
    """

    def write(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(_SPLIT_RECORD_COLUMNS)
            for row in rows:
                frame = row.video_frame
                writer.writerow(
                    [
                        row.origin_uuid,
                        row.set_key,
                        row.revision,
                        row.image,
                        frame.group,
                        frame.sequence,
                        frame.camera,
                        frame.frame_index,
                        row.frame_key,
                        row.unit_key,
                        row.placed_name,
                        row.split,
                    ]
                )

    atomic_write(out / SPLIT_RECORD_FILENAME, write)


def _schema_cell(params: PrepareTrainingDataParams, schema: KeypointSchema) -> str:
    """The keypoints a model trained on this preparation predicts, as JSON.

    ``polo`` predicts the one keypoint at ``point_index`` and has no edges or
    mirror pairs. Every other target predicts the narrowed pose.
    """
    if params.target == "polo":
        schema = KeypointSchema(names=(schema.names[params.point_index],))
    return keypoint_schema_cell(schema)


def _register(
    ds: Dataset,
    index: IndexCSV[PreparedDatasetIndexRow],
    run_id: str,
    kind: str,
    params: PrepareTrainingDataParams,
    sets: tuple[ResolvedSet, ...],
    out: Path,
    artifact: Path,
    *,
    pose: PoseDefinition,
    n_frames: int,
    n_by_split: Mapping[str, int],
    holdout: HoldOut,
    n_held_out: int,
    predicts: KeypointSchema,
) -> str:
    """Record a finished preparation, naming what a trainer is to be handed.

    Args:
        ds: The dataset that holds the index.
        index: The preparation index.
        run_id: The run.
        kind: The preparation op.
        params: The run's parameters.
        sets: The training sets.
        out: The run root.
        artifact: What a trainer is handed.
        pose: The pose the sets were narrowed to.
        n_frames: The frames written.
        n_by_split: The frames written to each split, keyed ``train``,
            ``valid`` and ``test``. A split that is absent counts 0.
        holdout: The hold-out, whose revisions the row lists.
        n_held_out: The frames of ``sets`` its units remove.
        predicts: The keypoints a model trained on the preparation predicts.
    """
    index.append(
        [
            PreparedDatasetIndexRow(
                run_id=run_id,
                kind=kind,
                target=params.target,
                artifact_path=ds.relative_to_root(artifact),
                consumed_sets=consumed_sets_cell(sets),
                n_frames=n_frames,
                n_train=n_by_split.get("train", 0),
                n_valid=n_by_split.get("valid", 0),
                n_test=n_by_split.get("test", 0),
                status="finished",
                pose_id=str(pose.id),
                pose_name=pose.name,
                held_out_sets=consumed_sets_cell(holdout.sets),
                n_held_out=n_held_out,
                keypoint_schema=_schema_cell(params, predicts),
                abs_path=Path(ds.relative_to_root(out)),
            )
        ]
    )
    index.mark_finished(run_id)
    return run_id


def _write_for_tool(
    params: PrepareTrainingDataParams,
    out: Path,
    ordered: list[tuple[AnnotationFrame, Path]],
    name_of: Callable[[AnnotationFrame, Path], str],
    split_of: Mapping[str, str],
    schema: KeypointSchema,
    categories: tuple[str, ...],
) -> tuple[Path, Counter[str]]:
    """Write the union for a tool that reads one labels artifact, not a tree.

    The images are copied in first, under the same collision-free names the tree
    targets use, and the merged set is pointed at those copies. The tool's own
    writer then sees one ordinary annotation set whose images sit beside it, and
    what it produces does not reach back into the datasets the sets came from.

    ``sleap`` is written as two files: the frames that *split_of* places in
    ``train`` to ``labels.slp``, and those in ``valid`` to ``labels.val.slp``. A
    ``test`` frame is written to neither, and its image is not copied.
    ``litpose`` is handed every frame of *ordered*.

    Returns:
        What a trainer is handed (the ``.slp`` of training frames, or the
        project directory), and how many frames were written for it in each
        split that *split_of* places them in.
    """
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    placed: dict[str, Path] = {}
    frames: list[tuple[str, AnnotationFrame]] = []
    for frame, source in ordered:
        name = name_of(frame, source)
        split = split_of[name]
        if params.target == "sleap" and split == "test":
            continue
        earlier = placed.get(name)
        if earlier is not None:
            msg = (
                f"{source} and {earlier} would both be written as {name!r}, so the "
                "second would replace the first"
            )
            raise ValueError(msg)
        placed[name] = source
        if params.symlink_images:
            (images / name).symlink_to(source.resolve())
        else:
            _ = shutil.copy2(source, images / name)
        frames.append(
            (
                split,
                AnnotationFrame(
                    image_path=Path(name),
                    width=frame.width,
                    height=frame.height,
                    objects=frame.objects,
                    video=frame.video,
                    frame_index=frame.frame_index,
                    split=frame.split,
                    group=frame.group,
                ),
            )
        )

    n_by_split = Counter(split for split, _frame in frames)

    def merged(subset: Collection[str] | None = None) -> AnnotationSet:
        """Return the copied frames placed in *subset*, or every one without it."""
        return AnnotationSet(
            schema=schema,
            frames=tuple(
                frame for split, frame in frames if subset is None or split in subset
            ),
            categories=categories or ("animal",),
            image_root=images,
            source_format="coco",
        )

    if params.target == "sleap":
        from mosaic.tracking.sleap.labels import validation_labels_path, write_slp

        labels = out / "labels.slp"
        _ = write_slp(merged({"train"}), labels, images_dir=images)
        _ = write_slp(
            merged({"valid"}), validation_labels_path(labels), images_dir=images
        )
        return labels, n_by_split
    from mosaic.tracking.litpose.labels import write_litpose_dataset

    project = write_litpose_dataset(
        merged(), out / "project", train_prob=params.split[0], copy_images=True
    )
    return project, n_by_split


def _already_prepared(
    ds: Dataset,
    index: IndexCSV[PreparedDatasetIndexRow],
    run_id: str,
    target: PrepareTarget,
) -> bool:
    """Did this exact preparation finish, and is what it wrote still there?

    The row and the artifact both, for the reason a trained model needs both:
    the row is the only thing that means the writer returned, and a row can
    outlive the directory it names. For ``sleap`` the artifact is the training
    labels and the validation file beside them.
    """
    if not index.path.exists():
        return False
    # A run with no row raises rather than reading as empty, and an unreadable
    # index is not evidence of anything. Either way the answer is to prepare
    # again: a false negative costs a rebuild, a false positive hands a
    # trainer a tree nobody finished.
    try:
        rows = index.read(run_id=run_id)
    except (OSError, ValueError, KeyError):
        return False
    if rows.empty:
        return False
    stored = str(rows.iloc[-1].get("artifact_path", "") or "")
    if not stored:
        return False
    artifact = ds.resolve_path(stored)
    if not artifact.exists():
        return False
    if target == "sleap":
        from mosaic.tracking.sleap.labels import validation_labels_beside

        return validation_labels_beside(artifact) is not None
    return True


def _count(split_dir: Path) -> int:
    labels = split_dir / "labels"
    return sum(1 for _ in labels.glob("*.txt")) if labels.exists() else 0
