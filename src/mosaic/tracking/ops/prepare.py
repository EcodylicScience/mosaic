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

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import Field

from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationSet,
    KeypointSchema,
)
from mosaic.core.annotations.narrow import AliasRole, narrow_pose_sets, narrowed_pose
from mosaic.core.annotations.pose_annotations import (
    PoseAnnotationSet,
    PoseDefinition,
    read_pose_annotations,
)
from mosaic.core.annotations.projection import KEYPOINTS_SERIES
from mosaic.core.helpers import to_safe_name
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    read_revision_manifest,
)
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import IdentityDeferred, Op, OpIdentity, register_op

from ._common import claim_run_root, empty_claimed_run_root, ensure_models_root

if TYPE_CHECKING:
    import pandas as pd

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

__all__ = [
    "KeypointSetRef",
    "PrepareTrainingDataOp",
    "PrepareTrainingDataParams",
    "PreparedDatasetIndexRow",
    "ResolvedSet",
    "prepare_training_data_run_id",
    "prepared_dataset_index",
    "resolve_keypoint_sets",
]

PREPARE_KIND: Final = "prepare-training-data"
_PREPARE_IDLE_SECONDS: Final = 600.0

PrepareTarget = Literal["yolo-pose", "polo", "sleap", "litpose"]
"""Which trainer the dataset is written for, and so what the artifact is.

``yolo-pose`` and ``polo`` are a split directory tree named by its ``data.yaml``.
``sleap`` is one ``labels.slp`` file, and ``litpose`` a Lightning Pose project
directory; both of those tools draw their own train and validation split, so
``split_by`` and the fractions beyond the first do not apply to them. Selecting,
naming and recording revisions is the same for all four.
"""

_TREE_TARGETS: Final = frozenset({"yolo-pose", "polo"})

SplitBy = Literal["sequence", "group", "frame"]


# --- Params --------------------------------------------------------------------


class KeypointSetRef(Params):
    """One annotation set, and which saved state of it to read."""

    set_key: Annotated[str, Declared("The set's key in the keypoints label series.")]
    origin_uuid: Annotated[
        str,
        Declared(
            "The manifest uuid of the dataset the set was saved in. Needed only "
            "when two datasets' sets share a key; empty matches the one that has it."
        ),
    ] = ""
    revision: Annotated[
        int | None,
        Field(ge=1),
        Declared(
            "Which revision to read. Left unset, the latest one indexed when the "
            "run is named. The number selects and is recorded; only the content "
            "it resolves to enters the run identifier."
        ),
    ] = None


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
            "What is kept together when the split is drawn. sequence keeps one "
            "recording's frames in one split, which is what makes a validation "
            "score honest. group keeps a whole sequence_groups label together. "
            "frame splits image by image, and its scores are optimistic."
        ),
    ] = "sequence"
    sequence_groups: Annotated[
        dict[str, str],
        Declared(
            "group only: the group each sequence belongs to, keyed by sequence "
            "name or by '<origin_uuid>:<sequence>' where names repeat across "
            "datasets. A sequence not listed is a group of its own."
        ),
    ] = Field(default_factory=dict)
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


# --- Index ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedDatasetIndexRow(RunIndexRowBase):
    """Typed row for ``models/prepare-training-data/index.csv``.

    ``artifact_path`` is root-relative and already listed for the ``models`` root
    in ``_INDEX_PATH_COLUMNS``. ``consumed_sets`` is the provenance link from a
    prepared dataset back to the annotation revisions behind it: a JSON list of
    ``{origin_uuid, set_key, revision, digest}``, in the sorted order the run
    identifier was taken over.
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


# --- Resolving sets ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedSet:
    """A set reference, resolved to one revision on disk."""

    origin_uuid: str
    set_key: str
    revision: int
    digest: str
    payload: Path

    def consumed(self) -> dict[str, str | int]:
        return {
            "origin_uuid": self.origin_uuid,
            "set_key": self.set_key,
            "revision": self.revision,
            "digest": self.digest,
        }


def resolve_keypoint_sets(
    ds: Dataset, refs: tuple[KeypointSetRef, ...], *, kind: str = PREPARE_KIND
) -> tuple[ResolvedSet, ...]:
    """Each reference as one revision file, with its digest measured off disk.

    The index says which file a reference means. The file says what is in it:
    the digest is taken from the bytes now, never read from the index column, so
    a revision that was edited in place cannot mint the identifier of the state
    it used to hold.

    Raises:
        KeyError: A reference matches several datasets' sets and names none.
        ValueError: Two references resolve to one set. Two revisions of a set
            name the same images, so both would be placed under one name.
        IdentityDeferred: A reference matches no indexed revision, or the
            revision it matches is not on disk. Both are states a scan or a save
            can still change, which is what deferring means.
    """
    records = index_records(read_label_series(ds, KEYPOINTS_SERIES))
    resolved: list[ResolvedSet] = []
    for ref in refs:
        matching = [
            record
            for record in records
            if record.get("key", "") == ref.set_key
            and (
                not ref.origin_uuid or record.get("origin_uuid", "") == ref.origin_uuid
            )
        ]
        if not matching:
            known = sorted({record.get("key", "") for record in records})
            origin = f" from dataset {ref.origin_uuid}" if ref.origin_uuid else ""
            # Deferred, like data that is not on disk yet: the revision may be
            # claimed by a scan that has not run. The reason names the repair,
            # because at execution this is what the failure will say.
            raise IdentityDeferred(
                kind,
                f"no revision of keypoint set {ref.set_key!r}{origin} is indexed "
                f"here (indexed sets: {known or 'none'}). A set saved in another "
                "dataset has to be claimed first: declare a labels source with "
                "series='keypoints' naming the revision, then scan",
            )
        origins = sorted({record.get("origin_uuid", "") for record in matching})
        if len(origins) > 1:
            msg = (
                f"keypoint set {ref.set_key!r} is indexed from {len(origins)} "
                f"datasets ({', '.join(origins)}); name one with origin_uuid"
            )
            raise KeyError(msg)
        if ref.revision is None:
            chosen = max(matching, key=lambda record: int(record.get("revision", "0")))
        else:
            wanted = [
                record
                for record in matching
                if int(record.get("revision", "0")) == ref.revision
            ]
            if not wanted:
                held = sorted(int(record.get("revision", "0")) for record in matching)
                raise IdentityDeferred(
                    kind,
                    f"revision {ref.revision} of keypoint set {ref.set_key!r} is "
                    f"not indexed here (indexed: {held}); claim it and scan",
                )
            chosen = wanted[0]
        payload = ds.resolve_path(chosen["abs_path"])
        if not payload.is_file():
            raise IdentityDeferred(
                kind,
                f"revision {chosen.get('revision', '')} of keypoint set "
                f"{ref.set_key!r} ({payload}) is not on disk, and this run's "
                "identity covers the content of what it reads",
            )
        named = [
            item
            for item in resolved
            if (item.origin_uuid, item.set_key) == (origins[0], ref.set_key)
        ]
        if named:
            msg = (
                f"keypoint set {ref.set_key!r} is named twice; one dataset reads one "
                "revision of a set, because two revisions show the same images"
            )
            raise ValueError(msg)
        resolved.append(
            ResolvedSet(
                origin_uuid=origins[0],
                set_key=ref.set_key,
                revision=int(chosen.get("revision", "0")),
                digest=file_digest(payload),
                payload=payload,
            )
        )
    return tuple(resolved)


def prepare_training_data_run_id(
    kind: str,
    version: str,
    params: PrepareTrainingDataParams,
    sets: tuple[ResolvedSet, ...],
) -> str:
    """Mint a preparation run identifier.

    A named function so the payload is one thing to read and something the golden
    corpus can call with fixed arguments and no filesystem.

    ``sets`` leaves the parameter dump and comes back as content. What the caller
    typed there -- a revision number, or none -- is a *selector*, and two spellings
    that select the same bytes must be one run. The triples are sorted, so the
    order sets were named in is not a difference either; the split below is drawn
    over sorted names for the same reason.
    """
    dumped = params.identity_dump()
    _ = dumped.pop("sets", None)
    return op_run_id(
        kind,
        version,
        {
            "params": dumped,
            "sets": sorted(
                [item.origin_uuid, item.set_key, item.digest] for item in sets
            ),
        },
    )


def check_preparation(ds: Dataset, params: PrepareTrainingDataParams) -> None:
    """Refuse what a ``prepare-training-data`` run would refuse, without writing anything.

    Resolves the revisions and narrows them exactly as a run does, so a caller that
    queues the preparation, and a training behind it, hears now that the sets hold
    several poses and none is named, that an object lacks the alias ``class_by`` asks
    for, or that a one-class trainer is handed several.

    Raises:
        ValueError: The sets cannot be narrowed to one training dataset, or the
            split would leave a trainer with no training or validation images.
        KeyError: A reference matches several datasets' sets and names none.
        IdentityDeferred: A revision is not indexed here, or not on disk.
    """
    ordered_sets, narrowed, _pose = _narrowed(
        params, resolve_keypoint_sets(ds, params.sets)
    )
    _ = _split(params, ordered_sets, narrowed)


# --- Building the tree ---------------------------------------------------------


def _split(
    params: PrepareTrainingDataParams,
    ordered_sets: list[ResolvedSet],
    narrowed: dict[str, AnnotationSet],
) -> tuple[dict[int, str], dict[str, str]]:
    """Each frame's placed name, keyed by the frame's id, and each name's split.

    Named and grouped once, here, from the frame itself. The group is never read
    back out of the name: a set key may hold the separator, and a name parsed apart
    on it would file a frame under the wrong recording. The names are sorted before
    the seeded shuffle, so neither the order sets were named in nor the order a file
    lists its frames in reaches the assignment.

    Raises:
        ValueError: A split tree would hold no training or no validation images.
            Splitting by sequence draws whole recordings, so the frames of one
            recording all land in one split.
    """
    from mosaic.core.annotations.split import split_filenames

    name_by_frame: dict[int, str] = {}
    group_by_name: dict[str, str] = {}
    for item in ordered_sets:
        for frame in narrowed[_set_label(item)].frames:
            name = _placed_name(item, frame)
            sequence = _sequence_of(frame)
            group = f"{item.origin_uuid}:{sequence}"
            if params.split_by == "group":
                group = params.sequence_groups.get(
                    group, params.sequence_groups.get(sequence, group)
                )
            name_by_frame[id(frame)] = name
            group_by_name[name] = group
    split_of, n_train, n_valid = split_filenames(
        sorted(group_by_name),
        params.split,
        params.seed,
        split_by="image" if params.split_by == "frame" else "group",
        group_key=group_by_name.__getitem__,
    )
    if params.target in _TREE_TARGETS and (n_train == 0 or n_valid == 0):
        empty = "training" if n_train == 0 else "validation"
        drawn = (
            f"{len(group_by_name)} frames"
            if params.split_by == "frame"
            else f"{len(set(group_by_name.values()))} {params.split_by}(s)"
        )
        msg = (
            f"the {empty} split would be empty: split_by={params.split_by!r} "
            f"divides {drawn} by {params.split}, too few to fill every split. "
            "Annotate frames from more recordings, or split by frame."
        )
        raise ValueError(msg)
    return name_by_frame, split_of


def _narrowed(
    params: PrepareTrainingDataParams, sets: tuple[ResolvedSet, ...]
) -> tuple[list[ResolvedSet], dict[str, AnnotationSet], PoseDefinition]:
    """The sets in identity order, each narrowed as this run reads it, and the pose.

    Narrowed together, in the order the run identifier sorts them, so the classes and
    the layout they are checked against do not depend on the order the sets were named
    in.

    Raises:
        ValueError: The sets cannot be narrowed to one pose's layout, or a trainer
            that learns one class is handed several.
    """
    ordered_sets = sorted(
        sets, key=lambda item: (item.origin_uuid, item.set_key, item.digest)
    )
    states = {_set_label(item): _read_set(item) for item in ordered_sets}
    narrowed = narrow_pose_sets(
        states,
        pose=params.pose,
        class_by=params.class_by,
        track_by=params.track_by if params.target == "sleap" else None,
        bbox=params.bbox,
    )
    categories = narrowed[_set_label(ordered_sets[0])].categories
    if params.target not in _TREE_TARGETS and len(categories) > 1:
        msg = (
            f"{params.target} trains one class, and class_by="
            f"{params.class_by!r} makes {len(categories)}: {list(categories)}"
        )
        raise ValueError(msg)
    return ordered_sets, narrowed, narrowed_pose(states, pose=params.pose)


def _read_set(item: ResolvedSet) -> PoseAnnotationSet:
    """One revision's saved state, its images anchored where they are."""
    manifest = read_revision_manifest(item.payload)
    image_root = (item.payload.parent / (manifest.image_root or ".")).resolve()
    return read_pose_annotations(item.payload, image_root)


def _set_label(item: ResolvedSet) -> str:
    """How a set is named in an error: which dataset, which set, which revision."""
    return f"{item.origin_uuid or 'local'}:{item.set_key}@rev{item.revision}"


def _sequence_of(frame: AnnotationFrame) -> str:
    """Which recording a frame came from.

    The revision records it. A file that does not -- a foreign COCO export claimed
    into the series -- falls back to the image's directory, which is where
    mosaic's own frame extraction puts a sequence's frames.
    """
    return frame.video or frame.image_path.parent.name


def _placed_name(item: ResolvedSet, frame: AnnotationFrame) -> str:
    """A file name no other frame in the union can have.

    ``<origin>__<set>__<sequence>__frame_<index>``: the dataset, because sets
    from two projects may share a key; the set, because two sets may hold the same
    frame; the sequence, because every sequence numbers its frames from the same
    place. Each part is percent-encoded so a name with a slash stays one
    component, and ``__frame`` is kept as the last separator because that is what
    the default split grouping reads.
    """
    index = (
        f"{frame.frame_index:06d}"
        if frame.frame_index >= 0
        else to_safe_name(frame.image_path.stem)
    )
    origin = (item.origin_uuid or "local")[:8]
    sequence = to_safe_name(_sequence_of(frame))
    stem = f"{origin}__{to_safe_name(item.set_key)}__{sequence}__frame_{index}"
    return f"{stem}{frame.image_path.suffix.lower() or '.png'}"


# --- Op ------------------------------------------------------------------------


@register_op
class PrepareTrainingDataOp(Op[PrepareTrainingDataParams]):
    """Merge revisions of keypoint annotation sets into one training dataset."""

    kind = PREPARE_KIND
    category = "convert"
    domain = "tracking"
    # 0.2: a revision is the full saved state, narrowed here, and a derived box
    # is the annotator's padded one rather than a hull read back as drawn.
    version = "0.2"
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
        nameable, and says so.
        """
        _ = (scope, require_data)
        sets = resolve_keypoint_sets(ds, params.sets, kind=self.kind)
        return OpIdentity(
            run_id=prepare_training_data_run_id(self.kind, self.version, params, sets)
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
        if not overwrite and _already_prepared(ds, index, run_id):
            ctx.cache_hit()
            return run_id

        # Claimed before anything is cleared: two executions of one identifier
        # would otherwise have one delete the other's tree mid-write.
        out.mkdir(parents=True, exist_ok=True)
        _ = claim_run_root(ds, ctx.execution_id, out, self.kind, _PREPARE_IDLE_SECONDS)
        empty_claimed_run_root(out)

        ordered_sets, narrowed, pose = _narrowed(params, sets)
        reference = narrowed[_set_label(ordered_sets[0])]
        schema, categories = reference.schema, reference.categories
        class_ids = reference.category_ids()

        # One flat list of (frame, image on disk), each carrying the set it came
        # from so its name can say so.
        owner: dict[int, ResolvedSet] = {}
        gathered: list[tuple[AnnotationFrame, Path]] = []
        missing: list[str] = []
        for item in ordered_sets:
            annotations = narrowed[_set_label(item)]
            for frame in annotations.frames:
                source = annotations.resolve(frame)
                if not source.is_file():
                    missing.append(str(source))
                    continue
                owner[id(frame)] = item
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

        name_by_frame, split_of = _split(params, ordered_sets, narrowed)

        def name_of(frame: AnnotationFrame, _source: Path) -> str:
            return name_by_frame[id(frame)]

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
            artifact = _write_for_tool(
                params, out, ordered, name_of, schema, categories
            )
            return _register(
                ds, index, run_id, self.kind, params, sets, out, artifact,
                pose=pose, n_frames=len(ordered),
            )  # fmt: skip
        written, skipped = write_split_tree(
            ordered,
            out,
            split_of,
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

        return _register(
            ds, index, run_id, self.kind, params, sets, out, data_yaml,
            pose=pose, n_frames=written,
        )  # fmt: skip


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
) -> str:
    """Record a finished preparation, naming what a trainer is to be handed."""
    index.ensure()
    index.append(
        [
            PreparedDatasetIndexRow(
                run_id=run_id,
                kind=kind,
                target=params.target,
                artifact_path=ds.relative_to_root(artifact),
                consumed_sets=json.dumps(
                    sorted(
                        (item.consumed() for item in sets),
                        key=lambda entry: (
                            str(entry["origin_uuid"]),
                            str(entry["set_key"]),
                        ),
                    )
                ),
                n_frames=n_frames,
                n_train=_count(out / "train"),
                n_valid=_count(out / "valid"),
                n_test=_count(out / "test"),
                status="finished",
                pose_id=str(pose.id),
                pose_name=pose.name,
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
    schema: KeypointSchema,
    categories: tuple[str, ...],
) -> Path:
    """Write the union for a tool that reads one labels artifact, not a tree.

    The images are copied in first, under the same collision-free names the tree
    targets use, and the merged set is pointed at those copies. The tool's own
    writer then sees one ordinary annotation set whose images sit beside it, and
    what it produces does not reach back into the datasets the sets came from.

    Returns:
        What a trainer is handed: the ``.slp`` file, or the project directory.
    """
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    placed: dict[str, Path] = {}
    frames: list[AnnotationFrame] = []
    for frame, source in ordered:
        name = name_of(frame, source)
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
            AnnotationFrame(
                image_path=Path(name),
                width=frame.width,
                height=frame.height,
                objects=frame.objects,
                video=frame.video,
                frame_index=frame.frame_index,
                split=frame.split,
            )
        )
    merged = AnnotationSet(
        schema=schema,
        frames=tuple(frames),
        categories=categories or ("animal",),
        image_root=images,
        source_format="coco",
    )
    if params.target == "sleap":
        from mosaic.tracking.sleap.labels import write_slp

        return write_slp(merged, out / "labels.slp", images_dir=images)
    from mosaic.tracking.litpose.labels import write_litpose_dataset

    return write_litpose_dataset(
        merged, out / "project", train_prob=params.split[0], copy_images=True
    )


def _already_prepared(
    ds: Dataset, index: IndexCSV[PreparedDatasetIndexRow], run_id: str
) -> bool:
    """Did this exact preparation finish, and is what it wrote still there?

    The row and the artifact both, for the reason a trained model needs both:
    the row is the only thing that means the writer returned, and a row can
    outlive the directory it names.
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
    return bool(stored) and ds.resolve_path(stored).exists()


def _count(split_dir: Path) -> int:
    labels = split_dir / "labels"
    return sum(1 for _ in labels.glob("*.txt")) if labels.exists() else 0
