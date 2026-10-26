"""From a trained model back to the annotations it saw.

Each link of that chain is written where it is known, by a different writer:

```
models/<train kind>/<run_id>            index row: data_path, data_fingerprint
  -> models/prepare-training-data/<id>  index row: consumed_sets, held_out_sets,
                                        pose_id, pose_name, keypoint_schema
  -> labels_raw/keypoints/index.csv     the claimed revision, its path
  -> <revision>/manifest.json           origin: what the authoring store recorded
```

Nothing stored the whole chain, on purpose: a copy of it would be a second thing
to keep agreeing with the four rows it summarizes. This walks them instead, and
says where the walk stopped when a link is missing -- a model trained from a bare
path, a revision in a dataset that is archived or unmounted.

The same walk names the frames that a model trained or validated on
(:func:`seen_frames`), from the ``split.csv`` that the preparation wrote beside
its data, and follows a fine-tuned model to its base model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from mosaic.core.annotations.model import AnnotationFrame, KeypointSchema
from mosaic.core.annotations.narrow import narrow_pose_sets
from mosaic.core.annotations.pose_annotations import PoseAnnotationSet, canonical_pairs
from mosaic.core.annotations.projection import KEYPOINTS_SERIES
from mosaic.core.annotations.split import Subset, frame_key
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    read_revision_manifest,
)
from mosaic.core.pipeline.models import (
    PREPARED_DATA_KINDS,
    SPLIT_RECORD_FILENAME,
    model_index_path,
    model_run_root,
    prepared_artifact_cell,
)
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.tracking.frame_extraction.frame_runs import video_frames_of
from mosaic.tracking.keypoint_sets import (
    ResolvedSet,
    read_keypoint_schema_cell,
    read_resolved_set,
    revision_image_root,
    set_label,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "ConsumedRevision",
    "SeenFrames",
    "TrainingProvenance",
    "seen_frames",
    "seen_frames_of_model",
    "training_provenance",
]


def _no_origin() -> dict[str, JsonValue]:
    return {}


class _ConsumedEntry(BaseModel):
    """One entry of a prepared dataset's ``consumed_sets`` cell."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    origin_uuid: str = ""
    set_key: str = ""
    revision: int = 0
    digest: str = ""


class _PoseDataYaml(BaseModel):
    """The part of a pose preparation's ``data.yaml`` that is read here."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    kpt_shape: tuple[int, ...] = ()


_CONSUMED: Final = TypeAdapter(list[_ConsumedEntry])
"""Turns the JSON cell into typed entries, or says it is not one."""


@dataclass(frozen=True, slots=True)
class ConsumedRevision:
    """One annotation revision a model was trained on.

    Attributes:
        origin_uuid: The dataset the revision was saved in.
        set_key: The annotation set.
        revision: Which saved state of it.
        digest: Its content digest, which is what entered the identifiers.
        path: Where the revision is, or ``None`` when no indexed row names it.
        reachable: Whether the revision's file is on disk now.
        origin: What the authoring store recorded with the save -- a database
            commit, for instance. Empty when the revision is not reachable.
    """

    origin_uuid: str
    set_key: str
    revision: int
    digest: str
    path: Path | None = None
    reachable: bool = False
    origin: Mapping[str, JsonValue] = field(default_factory=_no_origin)


@dataclass(frozen=True, slots=True)
class TrainingProvenance:
    """What one trained model was made from, as far as disk can say.

    Attributes:
        kind: The training op.
        run_id: The model.
        served_by: The linked library that holds it, or ``""`` for this dataset.
        data_path: What the run was trained on, as its row recorded it.
        data_fingerprint: The content fingerprint that entered its identifier.
        base_run_id: What it was fine-tuned from, or ``""``.
        prepared_kind: The preparation op behind ``data_path``, or ``""``.
        prepared_run_id: That preparation run, or ``""``.
        pose_id: The authoring store's id of the pose the preparation narrowed
            the sets to, or ``None`` when its row does not record one.
        pose_name: That pose's name when the preparation ran, or ``""``.
        keypoint_count: How many keypoints a pose model predicts per instance,
            from the ``kpt_shape`` its prepared ``data.yaml`` declares, or
            ``None`` for a model that predicts no keypoints and when that file
            cannot be read.
        keypoint_schema: The keypoint names, skeleton edges and mirror pairs
            that the model predicts, or ``None`` when neither the preparation
            row nor its revisions record them.
        sets: The annotation revisions the preparation consumed.
        held_out_sets: The revisions the preparation kept out of training, as
            they were when it ran. A later correction of a hold-out set does
            not change them. Empty without a hold-out and for a row that
            predates the column.
        stopped_at: Why the walk went no further, or ``""`` when it reached
            every revision. Never an error: an incomplete chain is an answer.
    """

    kind: str
    run_id: str
    served_by: str = ""
    data_path: str = ""
    data_fingerprint: str = ""
    base_run_id: str = ""
    prepared_kind: str = ""
    prepared_run_id: str = ""
    pose_id: int | None = None
    pose_name: str = ""
    keypoint_count: int | None = None
    keypoint_schema: KeypointSchema | None = None
    sets: tuple[ConsumedRevision, ...] = ()
    held_out_sets: tuple[ConsumedRevision, ...] = ()
    stopped_at: str = ""

    @property
    def keypoint_names(self) -> tuple[str, ...]:
        """The names of :attr:`keypoint_schema`, or ``()`` when it is unknown."""
        return self.keypoint_schema.names if self.keypoint_schema is not None else ()

    def as_json(self) -> dict[str, JsonValue]:
        """A plain mapping, for ``--json`` and for a control plane to store."""
        schema = self.keypoint_schema
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "served_by": self.served_by,
            "data_path": self.data_path,
            "data_fingerprint": self.data_fingerprint,
            "base_run_id": self.base_run_id,
            "prepared_kind": self.prepared_kind,
            "prepared_run_id": self.prepared_run_id,
            "pose_id": self.pose_id,
            "pose_name": self.pose_name,
            "keypoint_count": self.keypoint_count,
            "keypoint_names": list(self.keypoint_names),
            "skeleton": canonical_pairs(schema.skeleton if schema is not None else ()),
            "symmetries": canonical_pairs(
                schema.symmetries if schema is not None else ()
            ),
            "stopped_at": self.stopped_at,
            "sets": [_revision_json(item) for item in self.sets],
            "held_out_sets": [_revision_json(item) for item in self.held_out_sets],
        }


def _revision_json(item: ConsumedRevision) -> dict[str, JsonValue]:
    return {
        "origin_uuid": item.origin_uuid,
        "set_key": item.set_key,
        "revision": item.revision,
        "digest": item.digest,
        "path": str(item.path) if item.path is not None else "",
        "reachable": item.reachable,
        "origin": dict(item.origin),
    }


def _row(index_path: Path, run_id: str) -> dict[str, str] | None:
    if not index_path.exists():
        return None
    frame = pd.read_csv(index_path, dtype="string", keep_default_na=False)
    if "run_id" not in frame.columns:
        return None
    matching = [r for r in index_records(frame) if r.get("run_id", "") == run_id]
    return matching[-1] if matching else None


def _holder(
    ds: Dataset, kind: str, run_id: str
) -> tuple[Dataset, str, dict[str, str]] | None:
    """The dataset whose index registers the model: this one, then each library.

    Returns ``None`` when no index registers it.
    """
    own = _row(model_index_path(ds, kind), run_id)
    if own is not None:
        return ds, "", own
    for linked in ds.linked_libraries():
        found = _row(model_index_path(linked.dataset, kind), run_id)
        if found is not None:
            return linked.dataset, linked.link.id, found
    return None


def _unregistered(kind: str, run_id: str) -> str:
    """The reason that a model no index registers has no provenance."""
    msg = (
        f"no {kind} run {run_id!r} is registered in this dataset or in a library "
        "it links"
    )
    return msg


def _prepared_run(data_path: str) -> tuple[str, str]:
    """The preparation a data path lies in, as ``(kind, run_id)``, or two blanks.

    Read from the path: a preparation writes under
    ``models/<kind>/<run_id>/``, so the run is the artifact's directory. A model
    trained from anything else -- a folder somebody exported -- has no
    preparation to name.
    """
    for part in Path(data_path).parts:
        parsed = parse_op_run_id(part)
        if parsed is not None and parsed.kind in PREPARED_DATA_KINDS:
            return parsed.kind, part
    return "", ""


def _keypoint_count(holder: Dataset, prepared: Mapping[str, str]) -> int | None:
    """The keypoints per instance that a pose preparation's ``data.yaml`` declares.

    Only a ``yolo-pose`` preparation writes one with ``kpt_shape``. ``None`` for
    every other target, and when the file is gone or does not declare the shape.
    """
    if prepared.get("target", "") != "yolo-pose":
        return None
    stored = prepared_artifact_cell(prepared)
    if not stored:
        return None
    try:
        document: object = yaml.safe_load(
            holder.resolve_path(stored).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        return None
    try:
        shape = _PoseDataYaml.model_validate(document).kpt_shape
    except ValidationError:
        return None
    return shape[0] if shape else None


def _pose_id(cell: str) -> int | None:
    """Return the pose id that a prepared-data row records, or ``None``.

    ``None`` when the cell is blank, as a row written before the column existed
    reads, or is not an integer.
    """
    try:
        return int(cell)
    except ValueError:
        return None


def _keypoint_schema(
    prepared: Mapping[str, str], sets: tuple[ConsumedRevision, ...]
) -> KeypointSchema | None:
    """Return the keypoints that a model trained on a preparation predicts, or ``None``.

    The row's ``keypoint_schema`` cell records them. For a row written before
    that column, the pose of the row's ``pose_id`` is read from the first
    reachable consumed revision to declare it. Narrowing gave that pose one
    layout in every consumed set. ``None`` for a cell that does not parse, and
    for an older row without a pose id or without a readable revision declaring
    that pose. An older ``polo`` row gives ``None`` too, because it does not
    record which keypoint the points model learned.
    """
    cell = prepared.get("keypoint_schema", "")
    if cell:
        try:
            return read_keypoint_schema_cell(cell)
        except ValueError:
            return None
    pose_id = _pose_id(prepared.get("pose_id", ""))
    if pose_id is None or prepared.get("target", "") == "polo":
        return None
    for item in sets:
        resolved = _resolved(item)
        if resolved is None:
            continue
        try:
            poses = read_resolved_set(resolved).poses
        except (OSError, ValueError):
            continue
        for definition in poses:
            if definition.id == pose_id:
                return definition.schema
    return None


def _resolved(item: ConsumedRevision) -> ResolvedSet | None:
    """*item* as :func:`read_resolved_set` takes it, or ``None`` when it is not on disk."""
    if item.path is None or not item.reachable:
        return None
    return ResolvedSet(
        origin_uuid=item.origin_uuid,
        set_key=item.set_key,
        revision=item.revision,
        digest=item.digest,
        payload=item.path,
    )


def _consumed(holder: Dataset, cell: str) -> tuple[ConsumedRevision, ...]:
    try:
        entries = _CONSUMED.validate_json(cell or "[]")
    except ValidationError:
        return ()
    indexed = index_records(read_label_series(holder, KEYPOINTS_SERIES))
    found: list[ConsumedRevision] = []
    for entry in entries:
        origin_uuid = entry.origin_uuid
        set_key = entry.set_key
        revision = entry.revision
        digest = entry.digest
        row = next(
            (
                record
                for record in indexed
                if record.get("origin_uuid", "") == origin_uuid
                and record.get("key", "") == set_key
                and record.get("revision", "") == str(revision)
            ),
            None,
        )
        path = holder.resolve_path(row["abs_path"]) if row is not None else None
        reachable = path is not None and path.is_file()
        origin: Mapping[str, JsonValue] = {}
        if path is not None and reachable:
            try:
                origin = read_revision_manifest(path).origin
            except (FileNotFoundError, ValueError):
                origin = {}
        found.append(
            ConsumedRevision(
                origin_uuid=origin_uuid,
                set_key=set_key,
                revision=revision,
                digest=digest,
                path=path,
                reachable=reachable,
                origin=origin,
            )
        )
    return tuple(found)


def training_provenance(ds: Dataset, kind: str, run_id: str) -> TrainingProvenance:
    """What the model *run_id* was trained on, walked back as far as disk allows.

    Works from the dataset that trained the model and from any dataset that links
    it, because the rows that answer live with the model.

    Args:
        ds: The dataset asking.
        kind: The training op, such as ``"train-pose"``.
        run_id: The model.

    Raises:
        KeyError: No such run is registered here or in a linked library.
    """
    walk = _walk(ds, kind, run_id)
    if walk is None:
        raise KeyError(_unregistered(kind, run_id))
    return walk.provenance


@dataclass(frozen=True, slots=True)
class _Walk:
    """The provenance of one model, and the rows behind it that the seen frames read.

    Attributes:
        holder: The dataset whose index registers the model.
        prepared: The index row of the preparation that the model was trained on,
            or ``None`` when the walk stopped before it.
        provenance: The provenance of the model.
    """

    holder: Dataset
    prepared: dict[str, str] | None
    provenance: TrainingProvenance


def _walk(ds: Dataset, kind: str, run_id: str) -> _Walk | None:
    """Walk the model *run_id* back, or return ``None`` when no index registers it."""
    registered = _holder(ds, kind, run_id)
    if registered is None:
        return None
    holder, served_by, row = registered
    data_path = row.get("data_path", "")
    base = TrainingProvenance(
        kind=kind,
        run_id=run_id,
        served_by=served_by,
        data_path=data_path,
        data_fingerprint=row.get("data_fingerprint", ""),
        base_run_id=row.get("base_run_id", ""),
    )
    if not data_path:
        reason = "the run recorded no data path; it predates that record"
        return _Walk(holder, None, _stopped(base, reason))

    prepared_kind, prepared_run_id = _prepared_run(data_path)
    if not prepared_run_id:
        reason = (
            "the data was not written by a preparation run, so nothing records "
            "which annotations it holds"
        )
        return _Walk(holder, None, _stopped(base, reason))
    prepared = _row(model_index_path(holder, prepared_kind), prepared_run_id)
    if prepared is None:
        stopped = _stopped(
            base,
            f"{prepared_kind} run {prepared_run_id} has no index row",
            prepared_kind=prepared_kind,
            prepared_run_id=prepared_run_id,
        )
        return _Walk(holder, None, stopped)
    sets = _consumed(holder, prepared.get("consumed_sets", ""))
    unreachable = [item for item in sets if not item.reachable]
    reason = ""
    if not sets:
        reason = f"{prepared_kind} run {prepared_run_id} records no annotation sets"
    elif unreachable:
        named = ", ".join(f"{i.set_key} rev{i.revision}" for i in unreachable)
        reason = (
            f"{len(unreachable)} revision(s) are not on disk ({named}); the "
            "dataset they were saved in may be archived or unmounted. The prepared "
            "data still holds a copy of their labels"
        )
    provenance = TrainingProvenance(
        kind=kind,
        run_id=run_id,
        served_by=served_by,
        data_path=data_path,
        data_fingerprint=base.data_fingerprint,
        base_run_id=base.base_run_id,
        prepared_kind=prepared_kind,
        prepared_run_id=prepared_run_id,
        pose_id=_pose_id(prepared.get("pose_id", "")),
        pose_name=prepared.get("pose_name", ""),
        keypoint_count=_keypoint_count(holder, prepared),
        keypoint_schema=_keypoint_schema(prepared, sets),
        sets=sets,
        held_out_sets=_consumed(holder, prepared.get("held_out_sets", "")),
        stopped_at=reason,
    )
    return _Walk(holder, prepared, provenance)


def _stopped(
    base: TrainingProvenance,
    reason: str,
    *,
    prepared_kind: str = "",
    prepared_run_id: str = "",
) -> TrainingProvenance:
    return TrainingProvenance(
        kind=base.kind,
        run_id=base.run_id,
        served_by=base.served_by,
        data_path=base.data_path,
        data_fingerprint=base.data_fingerprint,
        base_run_id=base.base_run_id,
        prepared_kind=prepared_kind,
        prepared_run_id=prepared_run_id,
        stopped_at=reason,
    )


# --- Seen frames ---------------------------------------------------------------


_SEEN_SPLITS: Final[Mapping[str, frozenset[Subset]]] = {
    "yolo-pose": frozenset({"train", "valid"}),
    "polo": frozenset({"train", "valid"}),
    "sleap": frozenset({"train", "valid"}),
    "litpose": frozenset({"train", "valid", "test"}),
}
"""The splits that a model trains or validates on, by its preparation's target.

Validation picks the weights of a ``yolo-pose`` or ``polo`` model. A ``sleap``
preparation writes its training and validation frames to two files and its test
frames to neither. Lightning Pose redraws its split from every frame that the
hold-out leaves.
"""


@dataclass(frozen=True, slots=True)
class SeenFrames:
    """The frames that a trained model trained or validated on.

    Attributes:
        keys: Their frame keys (:func:`~mosaic.core.annotations.split.frame_key`).
            When the walk is incomplete, the keys of the links that it read.
        complete: Whether the walk read every link, in which case *keys* names
            every frame that the model saw.
        reason: Why the walk is incomplete, or ``""`` when it is complete.
    """

    keys: frozenset[str]
    complete: bool
    reason: str


def seen_frames(ds: Dataset, kind: str, run_id: str) -> SeenFrames:
    """Return the frames that the model *run_id* trained or validated on.

    The walk reads the split record of the preparation that the model was trained
    on. A ``yolo-pose``, ``polo`` or ``sleap`` model saw the frames of the
    training and validation splits, and a ``litpose`` model every frame that the
    hold-out did not exclude. A preparation without a split record, such as one
    older than 0.3, counts every usable frame of every revision that it consumed.
    A fine-tuned model adds the frames of its base model, and of that model's
    base in turn.

    The answer is incomplete, with the reason, when a link cannot be read: the
    model is not registered, its row records no data path, the data was not
    written by a preparation, the preparation lacks an index row, its split
    record cannot be read for its target, a revision is not on disk or cannot be
    read, or a base model is incomplete. The dataset tree is left unchanged.

    Args:
        ds: The dataset asking.
        kind: The training op, such as ``"train-pose"``.
        run_id: The model.

    Raises:
        ValueError: The chain of base models returns to a model on it.
    """
    return _seen(ds, kind, run_id, ())


def seen_frames_of_model(ds: Dataset, kind: str, run_ids: Sequence[str]) -> SeenFrames:
    """Return the frames that any training of one model trained or validated on.

    A SLEAP top-down model is two trainings, a centroid and a centered-instance
    model, each on its own preparation. The frames of a model are the union of
    the frames of its trainings, and complete when each training's are. An
    incomplete answer gives the reason of the first incomplete training,
    prefixed with its run id. A model of one training has the answer of
    :func:`seen_frames`.

    Args:
        ds: The dataset asking.
        kind: The training op of every training.
        run_ids: The trainings, in the order that the model lists them.

    Raises:
        ValueError: *run_ids* is empty, or a chain of base models returns to a
            model on it.
    """
    answers = [(run_id, seen_frames(ds, kind, run_id)) for run_id in run_ids]
    if not answers:
        msg = "a model names at least one training run"
        raise ValueError(msg)
    if len(answers) == 1:
        return answers[0][1]
    keys = frozenset(key for _run_id, answer in answers for key in answer.keys)
    for run_id, answer in answers:
        if not answer.complete:
            reason = f"{run_id}: {answer.reason}"
            return SeenFrames(keys=keys, complete=False, reason=reason)
    return SeenFrames(keys=keys, complete=True, reason="")


def _unknown(reason: str) -> SeenFrames:
    return SeenFrames(keys=frozenset(), complete=False, reason=reason)


def _seen(ds: Dataset, kind: str, run_id: str, chain: tuple[str, ...]) -> SeenFrames:
    """The seen frames of *run_id*, which the models of *chain* fine-tuned in turn.

    Raises:
        ValueError: *run_id* is on *chain*, or a base model of it returns to a
            model on the chain.
    """
    if run_id in chain:
        cycle = " -> ".join((*chain, run_id))
        msg = f"the chain of base models returns to {run_id}: {cycle}"
        raise ValueError(msg)
    walk = _walk(ds, kind, run_id)
    if walk is None:
        return _unknown(_unregistered(kind, run_id))
    own = _prepared_frames(walk)
    base_run_id = walk.provenance.base_run_id
    if not base_run_id:
        return own
    # The run id of a base model names its kind. A model fine-tuned from a bare
    # path records a content digest instead, which no index registers.
    parsed = parse_op_run_id(base_run_id)
    base_kind = kind if parsed is None else parsed.kind
    base = _seen(ds, base_kind, base_run_id, (*chain, run_id))
    keys = own.keys | base.keys
    if not own.complete:
        return SeenFrames(keys=keys, complete=False, reason=own.reason)
    if not base.complete:
        reason = f"base model {base_run_id}: {base.reason}"
        return SeenFrames(keys=keys, complete=False, reason=reason)
    return SeenFrames(keys=keys, complete=True, reason="")


def _prepared_frames(walk: _Walk) -> SeenFrames:
    """The seen frames of the preparation that one model was trained on."""
    provenance = walk.provenance
    if walk.prepared is None:
        return _unknown(provenance.stopped_at)
    preparation = f"{provenance.prepared_kind} run {provenance.prepared_run_id}"
    record = (
        model_run_root(
            walk.holder, provenance.prepared_kind, provenance.prepared_run_id
        )
        / SPLIT_RECORD_FILENAME
    )
    if record.is_file():
        return _recorded_frames(record, walk.prepared.get("target", ""), preparation)
    return _consumed_frames(walk)


def _recorded_frames(record: Path, target: str, preparation: str) -> SeenFrames:
    """The frames of the split record *record* in the splits that *target* reads."""
    splits = _SEEN_SPLITS.get(target)
    if splits is None:
        return _unknown(f"{preparation} records an unknown target {target!r}")
    try:
        written = pd.read_csv(record, dtype="string", keep_default_na=False)
    except (OSError, ValueError) as exc:
        return _unknown(f"the split record of {preparation} cannot be read: {exc}")
    if "frame_key" not in written.columns or "split" not in written.columns:
        reason = f"the split record of {preparation} lacks a frame_key or split column"
        return _unknown(reason)
    keys = frozenset(
        row["frame_key"] for row in index_records(written) if row["split"] in splits
    )
    return SeenFrames(keys=keys, complete=True, reason="")


def _consumed_frames(walk: _Walk) -> SeenFrames:
    """Every usable frame of the revisions that a preparation consumed.

    Read for a preparation without a split record. Narrowing to the
    preparation's pose drops only the unusable frames, which no training read.
    The test frames of a tree target count too, because no record says which
    frames they are.
    """
    provenance = walk.provenance
    if provenance.stopped_at:
        return _unknown(provenance.stopped_at)
    revisions = [
        resolved for resolved in map(_resolved, provenance.sets) if resolved is not None
    ]
    states: dict[str, PoseAnnotationSet] = {}
    image_roots: dict[str, Path] = {}
    for revision in revisions:
        label = set_label(revision)
        try:
            state = read_resolved_set(revision)
            image_roots[label] = revision_image_root(state.image_root)
        except (OSError, ValueError) as exc:
            return _unknown(f"revision {label} cannot be read: {exc}")
        states[label] = state
    try:
        usable = _usable_frames(states, provenance.pose_id)
    except ValueError as exc:
        reason = f"the revisions do not narrow to pose {provenance.pose_id}: {exc}"
        return _unknown(reason)
    keys: set[str] = set()
    for revision in revisions:
        label = set_label(revision)
        try:
            recorded = video_frames_of(
                walk.holder,
                image_roots[label],
                usable[label],
                revision_origin=revision.origin_uuid,
            )
        except ValueError as exc:
            return _unknown(f"the frames of revision {label} cannot be located: {exc}")
        keys.update(frame_key(video_frame) for video_frame in recorded)
    return SeenFrames(keys=frozenset(keys), complete=True, reason="")


def _usable_frames(
    states: Mapping[str, PoseAnnotationSet], pose_id: int | None
) -> dict[str, tuple[AnnotationFrame, ...]]:
    """The usable frames of each revision, narrowed together to the pose *pose_id*.

    Every usable frame of every pose when *pose_id* is ``None``.

    Raises:
        ValueError: The revisions do not narrow to the pose.
    """
    if pose_id is None:
        return {
            label: tuple(
                frame.as_annotation_frame() for frame in state.frames if frame.usable
            )
            for label, state in states.items()
        }
    narrowed = narrow_pose_sets(states, pose=pose_id)
    return {label: annotations.frames for label, annotations in narrowed.items()}
