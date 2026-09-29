from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

import pandas as pd
from mosaic_media import MediaFacts
from pydantic import Field, ValidationInfo, field_validator

from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline._utils import ResolvedScope, hash_params, json_ready
from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.inventory._read import IndexReader
from mosaic.core.pipeline.inventory.contributors import register_inventory_contributor
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    CameraEntry,
    InventoryScope,
)
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase
from mosaic.core.pipeline.media_index import media_members_from_rows
from mosaic.core.pipeline.sequence_index import (
    media_compositions_for,
)
from mosaic.core.pipeline.job import CancelToken, Cancelled, JobContext
from mosaic.core.pipeline.joined_export import (
    JoinedExportMissingError,
    join_to_read,
    missing_joins,
)
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op, run_op
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    DeclaredModel,
    Params,
)
from mosaic.core.scope import Scope

from .extraction import extract_frames as _extract_frames
from .extraction import extract_frames_multi as _extract_frames_multi
from .sampling import ExtractionMethod

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.core.pipeline.progress import ProgressCallback


# --- Frame extraction index helpers ---


def frames_run_root(ds: Dataset, method: str, run_id: str) -> Path:
    return ds.get_root("frames") / method / run_id


def frames_index_path(ds: Dataset, method: str) -> Path:
    return ds.get_root("frames") / method / "index.csv"


@dataclass(frozen=True, slots=True)
class FramesIndexRow(RunIndexRowBase):
    """Typed row for the frames index CSV.

    ``video_uuids`` and ``media_composition`` are item 5.1's frames half: what
    this frame set was cut from, recorded and never hashed. The extraction
    identifier is frozen (see :func:`frames_run_id`), so this is the only place
    the answer can be written down.

    **Both, not either.** ``video_uuids`` names the individual files, which is
    what item 5.1's table asks for and what survives the sequence being
    rearranged around them. ``media_composition`` names the arrangement, and it
    is the one that matters here: ``_extract_frames_multi`` pools candidates
    across a camera's clips using *global* frame indices, so a frame set is per
    camera over an ordered clip list and a reorder genuinely changes what a
    stored index refers to. Item 5.1's "extracted frames are already per-video on
    disk" is true only of a single-clip camera.

    ``video_uuids`` is in ``video_order`` and is **never sorted** -- it is the
    arrangement, and sorting it would make two orderings record alike. Contrast
    ``consumed_source_roots`` on the tracks row, which is a set and is sorted.
    Empty means not establishable, never "no videos".
    """

    method: str
    group: str
    sequence: str
    camera: str
    video_abs_path: str
    params_hash: str
    n_frames_extracted: int = 0
    n_frames_requested: int = 0
    video_uuids: str = ""
    media_composition: str = ""


FRAMES_INDEX_COLUMNS: Final[list[str]] = [
    field.name for field in fields(FramesIndexRow)
]
"""The schema, in CSV order. Derived from the row so the two cannot drift."""


def adopt_frames_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a frames index read off disk up to the current schema, in memory.

    The same hook ``tracks_index`` and the feature index carry, added here with
    the columns above rather than after someone meets the failure. Without it the
    first append to a pre-Stage-5 index concatenates a frame missing two columns
    with one that has them, and every older row's ``n_frames_extracted`` is
    widened to float on the way through -- ``40`` reaching disk as ``40.0``,
    which is the trap ``index_csv`` documents and this repo has already paid for
    once.

    Must return a **schema-complete** frame: filling only the new columns would
    leave ``list_runs`` raising on ``finished_at``.
    """
    out = pd.DataFrame(index=df.index)
    for column in FRAMES_INDEX_COLUMNS:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    return out.reset_index(drop=True)


def frames_index(path: Path) -> IndexCSV[FramesIndexRow]:
    return IndexCSV(
        path,
        FramesIndexRow,
        # camera is part of the identity: the cameras of one recording share a
        # (run_id, group, sequence), so without it a partial re-run of one camera
        # would dedup away the other camera's row.
        dedup_keys=["run_id", "group", "sequence", "camera"],
        adopt=adopt_frames_columns,
    )


# --- Frame extraction op (registered under the Job Contract) ---


ParallelMode = Literal["thread", "process"]
"""Which executor runs the per-camera work."""

_N_FRAMES_DESCRIPTION = (
    "How many frames to write per camera. Required by 'uniform' and 'kmeans', "
    "and refused by 'list', whose listed frames are what is written."
)

_METHOD_DESCRIPTION = (
    "How the frames are chosen: 'uniform' spaces them evenly over the "
    "candidate range, 'kmeans' clusters the candidates by pixel content and "
    "keeps one frame per cluster, and 'list' writes exactly the frames "
    "listed in frames."
)

_FRAMES_DESCRIPTION = (
    "The frames to write under 'list', one item per entry, and refused by the "
    "other methods. The list chooses the entries: a run covers every listed "
    "entry, narrowed by its scope. Each item's frames apply to every camera of "
    "the entry."
)

_ENTRY_GROUP_DESCRIPTION = "The entry's group; empty for a dataset with none."

_ENTRY_SEQUENCE_DESCRIPTION = "The entry's sequence."

_ENTRY_INDICES_DESCRIPTION = (
    "Frame indices on the entry's media axis: 0 is its first frame, and a "
    "recording in several clips counts across them in order, as its tracks' "
    "frame column does. Order and repeats do not matter."
)

_START_FRAME_DESCRIPTION = (
    "First frame of the range frames are chosen from, inclusive. Unset starts "
    "at the beginning of the video. Refused by 'list'."
)

_END_FRAME_DESCRIPTION = (
    "Last frame of the range frames are chosen from, inclusive. Unset runs to "
    "the end of the video. Refused by 'list'."
)

_CANDIDATE_STEP_DESCRIPTION = (
    "Stride between candidate frames within the range. A wider stride samples "
    "a long recording without decoding every frame of it."
)

_CROP_DESCRIPTION = (
    "Crop rectangle (x, y, width, height) applied to every written frame. "
    "Unset writes the full frame."
)

_RANDOM_STATE_DESCRIPTION = "Seed for k-means and for breaking ties between candidates."

_KMEANS_RESIZE_DESCRIPTION = (
    "Width and height a candidate frame is resized to before its pixels become "
    "the clustering feature vector."
)

_KMEANS_GRAYSCALE_DESCRIPTION = (
    "Convert a candidate frame to grayscale before flattening it into a "
    "feature vector, which clusters on layout instead of color."
)

_KMEANS_MAX_CANDIDATES_DESCRIPTION = (
    "Cap on how many candidate frames are decoded for clustering; the stride "
    "widens to stay under it. Unset decodes every candidate in the range."
)

_KMEANS_BATCH_SIZE_DESCRIPTION = (
    "How many feature vectors one mini-batch k-means update reads."
)

_KMEANS_MAX_ITER_DESCRIPTION = "Ceiling on mini-batch k-means iterations."

_KMEANS_N_INIT_DESCRIPTION = (
    "How many centroid seedings k-means tries before keeping the best. 'auto' "
    "leaves the count to scikit-learn."
)

_PARALLEL_MODE_DESCRIPTION = (
    "Which executor runs the cameras: 'thread' shares one process, 'process' "
    "forks, which a decoder holding the interpreter lock needs."
)


_REVISION_DESCRIPTION = (
    "Bump to extract a second selection under the same settings. It is the one "
    "term allowed to move the extraction identifier, and only a non-zero value "
    "enters it. Revision 0 reproduces every identifier already on disk."
)

_PARALLEL_WORKERS_DESCRIPTION = (
    "How many cameras are extracted at once. 'auto' reads the machine, and an "
    "integer pins the count."
)


class EntryFrames(DeclaredModel):
    """The frames one entry contributes to a ``list`` extraction."""

    group: Annotated[str, Declared(_ENTRY_GROUP_DESCRIPTION)] = ""
    sequence: Annotated[str, Field(min_length=1), Declared(_ENTRY_SEQUENCE_DESCRIPTION)]
    indices: Annotated[
        tuple[Annotated[int, Field(ge=0)], ...],
        Field(min_length=1),
        Declared(_ENTRY_INDICES_DESCRIPTION),
    ]

    @property
    def entry(self) -> Entry:
        """The ``(group, sequence)`` pair this item names."""
        return (self.group, self.sequence)


def _validated_method(info: ValidationInfo) -> ExtractionMethod | None:
    """The method a field validator runs under, or ``None`` when it was refused.

    ``info.data`` holds the fields validated so far, which is why ``method`` is
    declared first. A refused method is absent from it, and that refusal is
    reported alone rather than beside a second one it caused.
    """
    method: object = info.data.get("method")
    if not isinstance(method, str):
        return None
    match method:
        case "uniform" | "kmeans" | "list":
            return method
        case _:
            return None


def _normalised_frames(items: tuple[EntryFrames, ...]) -> tuple[EntryFrames, ...]:
    """*items* sorted by entry, each entry's indices sorted and de-duplicated.

    The identifier is hashed from this form, so the same frames listed in
    another order, or with an index repeated, name the same run. An entry named
    twice is refused rather than merged: two items for one entry is a list built
    wrong, and merging would hide which of them was meant.
    """
    by_entry: dict[Entry, tuple[int, ...]] = {}
    for item in items:
        if item.entry in by_entry:
            message = (
                f"frames names ({item.group}, {item.sequence}) twice. List each "
                f"entry once, with all of its frames."
            )
            raise ValueError(message)
        by_entry[item.entry] = tuple(sorted(set(item.indices)))
    return tuple(
        EntryFrames(group=group, sequence=sequence, indices=indices)
        for (group, sequence), indices in sorted(by_entry.items())
    )


class ExtractFramesParams(Params):
    """Typed parameters for the ``extract-frames`` tracking op.

    The extraction settings, plus the parallelism knobs and ``revision``. The
    parallelism knobs are ``HASH_EXCLUDE``, selecting how fast the work runs
    rather than what comes out. One set of settings therefore shares a run_id
    and adds per-sequence subdirectories under it.

    ``revision`` and ``frames`` are ``HASH_EXCLUDE`` and still name the run.
    Each enters the payload only when set (:func:`frames_identity_payload`), so
    every identifier already on disk is reproduced. What a run covers and
    whether it recomputes are arguments to the run, not fields here.

    ``method`` is declared first because the validators below read it. Each
    validator is declared with ``check_fields=False``, so a consumer can remove
    the field it checks from a subclass and supply the value itself, as
    mosaic-api does for the fields it owns.
    """

    method: Annotated[ExtractionMethod, Declared(_METHOD_DESCRIPTION)] = "uniform"
    n_frames: Annotated[
        int | None, Field(validate_default=True), Declared(_N_FRAMES_DESCRIPTION)
    ] = None
    frames: Annotated[
        tuple[EntryFrames, ...] | None,
        Field(validate_default=True),
        HASH_EXCLUDE,
        Declared(_FRAMES_DESCRIPTION),
    ] = None
    start_frame: Annotated[int | None, Declared(_START_FRAME_DESCRIPTION)] = None
    end_frame: Annotated[int | None, Declared(_END_FRAME_DESCRIPTION)] = None
    candidate_step: Annotated[int, Declared(_CANDIDATE_STEP_DESCRIPTION)] = 1
    crop: Annotated[tuple[int, int, int, int] | None, Declared(_CROP_DESCRIPTION)] = (
        None
    )
    random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
    kmeans_resize: Annotated[
        tuple[int, int], Declared(_KMEANS_RESIZE_DESCRIPTION, unit="px")
    ] = (64, 64)
    kmeans_grayscale: Annotated[bool, Declared(_KMEANS_GRAYSCALE_DESCRIPTION)] = True
    kmeans_max_candidates: Annotated[
        int | None, Declared(_KMEANS_MAX_CANDIDATES_DESCRIPTION)
    ] = 5000
    kmeans_batch_size: Annotated[int, Declared(_KMEANS_BATCH_SIZE_DESCRIPTION)] = 1024
    kmeans_max_iter: Annotated[int, Declared(_KMEANS_MAX_ITER_DESCRIPTION)] = 100
    kmeans_n_init: Annotated[str | int, Declared(_KMEANS_N_INIT_DESCRIPTION)] = "auto"
    revision: Annotated[int, HASH_EXCLUDE, Declared(_REVISION_DESCRIPTION)] = 0
    parallel_workers: Annotated[
        int | str | None, HASH_EXCLUDE, Declared(_PARALLEL_WORKERS_DESCRIPTION)
    ] = "auto"
    parallel_mode: Annotated[
        ParallelMode, HASH_EXCLUDE, Declared(_PARALLEL_MODE_DESCRIPTION)
    ] = "thread"

    @field_validator("n_frames", check_fields=False)
    @classmethod
    def _n_frames_agrees_with_method(
        cls, value: int | None, info: ValidationInfo
    ) -> int | None:
        """Required by the sampling methods, refused by ``list``."""
        method = _validated_method(info)
        if method == "list" and value is not None:
            message = (
                "n_frames is not read under method 'list', whose listed frames "
                "are what is written. Leave it unset."
            )
            raise ValueError(message)
        if method is not None and method != "list" and value is None:
            message = (
                f"method {method!r} needs n_frames, how many frames to write "
                f"per camera."
            )
            raise ValueError(message)
        return value

    @field_validator("frames", check_fields=False)
    @classmethod
    def _frames_agree_with_method(
        cls, value: tuple[EntryFrames, ...] | None, info: ValidationInfo
    ) -> tuple[EntryFrames, ...] | None:
        """Required by ``list`` and refused by the others; normalised when given."""
        method = _validated_method(info)
        if method is None:
            return value
        if method != "list":
            if value is not None:
                message = (
                    f"frames is read only under method 'list', and method "
                    f"{method!r} chooses its own frames. Set method to 'list', "
                    f"or leave frames unset."
                )
                raise ValueError(message)
            return value
        if not value:
            message = (
                "method 'list' needs frames: each entry to extract, with the "
                "frame indices to write for it."
            )
            raise ValueError(message)
        return _normalised_frames(value)

    @field_validator("start_frame", "end_frame", check_fields=False)
    @classmethod
    def _no_range_under_list(
        cls, value: int | None, info: ValidationInfo
    ) -> int | None:
        """A range is a sampling window; listed frames are written where they fall."""
        if value is not None and _validated_method(info) == "list":
            message = (
                f"{info.field_name} is not read under method 'list', whose "
                f"listed frames are written wherever they fall. Leave it unset."
            )
            raise ValueError(message)
        return value


def _source_identity_maps(
    ds: Dataset, entries: Iterable[tuple[str, str]]
) -> tuple[dict[tuple[str, str, str], str], dict[tuple[str, str], str]]:
    """What each camera was cut from: its ordered uids, and its composition.

    **Read from the media index, never from ``ResolvedMedia.facts``**, though
    those are already in hand and carry a ``video_uuid`` each. ``route_media_row``
    hands back the *derivative's* facts for a sequence whose verdict marks an
    analysis transcode required, so that uid names the transcode rather than the
    source. It would look right, round-trip fine, and join to nothing in
    ``media_raw`` -- after which every staleness check on this row answers
    "unchanged" for as long as the dataset exists. Reading the index means the
    uids and the composition come from one place and cannot disagree.

    A legacy ``media``-rooted dataset records nothing, the same carve-out
    ``Dataset._write_media_compositions`` makes: that root holds derivatives, and
    a derivative has no composition of its own (rule P6).

    One member without an identity makes the whole camera unestablishable rather
    than partially named -- ``composition.py``'s completeness rule, applied to the
    same members so the two answers agree.
    """
    wanted = set(entries)
    if not wanted or ds.resolve_media_root() != "media_raw":
        return {}, {}
    members = media_members_from_rows(ds.read_media_index())
    uids: dict[tuple[str, str, str], str] = {}
    for entry, entry_members in members.items():
        if entry not in wanted:
            continue
        for camera in {member.camera for member in entry_members}:
            ordered = sorted(
                (member for member in entry_members if member.camera == camera),
                key=lambda member: member.video_order,
            )
            group, sequence = entry
            uids[(group, sequence, camera)] = (
                ""
                if any(not member.uid for member in ordered)
                else ",".join(member.uid for member in ordered)
            )
    # Through the shared helper, so this and every tracker row answer "what was
    # this entry's media" the same way rather than encoding it twice.
    return uids, media_compositions_for(ds, wanted)


@dataclass(frozen=True, slots=True)
class _ExtractSpec:
    """Picklable unit of work for one (group, sequence, camera) -- process-safe."""

    group: str
    sequence: str
    camera: str
    video_paths: tuple[Path, ...]
    facts: tuple[MediaFacts, ...]
    video_uuids: str
    media_composition: str
    seq_dir: Path
    run_id: str
    params_hash: str
    n_frames: int | None
    method: str
    frame_indices: tuple[int, ...] | None
    n_media_frames: int
    start_frame: int | None
    end_frame: int | None
    candidate_step: int
    crop: tuple[int, int, int, int] | None
    random_state: int
    kmeans_resize: tuple[int, int]
    kmeans_grayscale: bool
    kmeans_max_candidates: int | None
    kmeans_batch_size: int
    kmeans_max_iter: int
    kmeans_n_init: str | int
    overwrite: bool


def _spec_label(spec: _ExtractSpec) -> str:
    """Display label for a work spec, camera-qualified for a multi-camera entry."""
    key = make_entry_key(spec.group, spec.sequence)
    return f"{key}/{spec.camera}" if spec.camera else key


def _extract_one(spec: _ExtractSpec) -> FramesIndexRow | None:
    """Extract one (group, sequence, camera). Module-scope (picklable) so process
    mode works.

    Manifest path-rewriting (which needs the Dataset) is done by the caller.
    """
    seq_dir = spec.seq_dir
    if seq_dir.exists():
        # Reached only with overwrite=False: `_refuse_to_overwrite` has already
        # raised for the other case, before any worker started.
        print(f"[extract_frames] skip {_spec_label(spec)} (exists, overwrite=False)")
        return None

    kmeans_kw = dict(
        kmeans_resize=spec.kmeans_resize,
        kmeans_grayscale=spec.kmeans_grayscale,
        kmeans_max_candidates=spec.kmeans_max_candidates,
        kmeans_batch_size=spec.kmeans_batch_size,
        kmeans_max_iter=spec.kmeans_max_iter,
        kmeans_n_init=spec.kmeans_n_init,
    )
    try:
        if len(spec.video_paths) == 1:
            # No facts for a join, which has no index row: the reader probes it.
            result = _extract_frames(
                video_path=spec.video_paths[0],
                n_frames=spec.n_frames,
                method=spec.method,
                start_frame=spec.start_frame,
                end_frame=spec.end_frame,
                candidate_step=spec.candidate_step,
                crop=spec.crop,
                random_state=spec.random_state,
                run_id=spec.run_id,
                output_dir=seq_dir,
                facts=spec.facts[0] if spec.facts else None,
                frame_indices=spec.frame_indices,
                **kmeans_kw,
            )
        else:
            result = _extract_frames_multi(
                video_paths=list(spec.video_paths),
                n_frames=spec.n_frames,
                method=spec.method,
                start_frame=spec.start_frame,
                end_frame=spec.end_frame,
                candidate_step=spec.candidate_step,
                crop=spec.crop,
                random_state=spec.random_state,
                run_id=spec.run_id,
                output_dir=seq_dir,
                facts=list(spec.facts),
                frame_indices=spec.frame_indices,
                **kmeans_kw,
            )
    except Exception as exc:
        print(
            f"[extract_frames] ERROR processing {_spec_label(spec)}: {exc}",
            file=sys.stderr,
        )
        return None

    return FramesIndexRow(
        run_id=spec.run_id,
        method=spec.method,
        group=spec.group,
        sequence=spec.sequence,
        camera=spec.camera,
        abs_path=seq_dir,
        n_frames_extracted=result.n_extracted,
        n_frames_requested=result.n_requested,
        video_abs_path=json.dumps([str(p) for p in spec.video_paths])
        if len(spec.video_paths) > 1
        else str(spec.video_paths[0]),
        params_hash=spec.params_hash,
        video_uuids=spec.video_uuids,
        media_composition=spec.media_composition,
    )


def _relative_video_paths(ds: Dataset, stored: str) -> str:
    """Make a ``video_abs_path`` cell dataset-root-relative, keeping its shape.

    The cell is a JSON array for a multi-clip camera and a bare path otherwise.
    That encoding is **not** changed here: no production code parses it, so a
    second spelling would be pure cost, and adding one would mean an index could
    hold two forms of the same answer -- the shape this program keeps arguing
    against. Only the paths inside it move.
    """
    if not stored:
        return stored
    if not stored.startswith("["):
        return ds.relative_to_root(Path(stored))
    try:
        parsed: object = json.loads(stored)
    except ValueError:
        return stored
    if not isinstance(parsed, list):
        return stored
    listed: list[object] = parsed
    return json.dumps([ds.relative_to_root(Path(str(item))) for item in listed])


def _rewrite_manifest(ds: Dataset, seq_dir: Path) -> None:
    """Rewrite run_info.json with dataset-relative paths for portability."""
    manifest_path = seq_dir / "run_info.json"
    if not manifest_path.exists():
        return
    try:
        mdata = json.loads(manifest_path.read_text())
        mdata["output_dir"] = ds._relative_to_root(seq_dir)
        if "video_path" in mdata:
            mdata["video_path"] = ds._relative_to_root(Path(mdata["video_path"]))
        manifest_path.write_text(json.dumps(mdata, indent=2))
    except Exception as exc:
        print(
            f"[extract_frames] manifest rewrite failed for {seq_dir}: {exc}",
            file=sys.stderr,
        )


def _resolve_max_workers(parallel_workers: int | str | None) -> int:
    if parallel_workers == "auto":
        import os as _os

        return min(_os.cpu_count() or 1, 8)
    try:
        n = int(parallel_workers)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return n if n > 1 else 1


class AnnotatedFramesWouldBeDestroyed(RuntimeError):
    """``overwrite=True`` would remove a frame set an annotation may reference."""


def _refuse_to_overwrite(specs: Sequence[_ExtractSpec]) -> None:
    """Refuse to re-extract over a directory that already holds frames.

    Rule P4's first carve-out. ``overwrite`` is ``HASH_EXCLUDE``, so it does not
    reach the identifier -- meaning ``overwrite=True`` removed the *same*
    directory ``AnnotationFrame.image_path`` names, on Dolt-tracked rows carrying
    keypoint annotation labor, recoverable only by re-annotating.

    Unconditional, and that is the point. Mosaic cannot ask whether a frame set
    carries annotations: it imports no control plane and queries no Dolt. A
    marker file written by the control plane would fail *open* -- an absent
    marker is indistinguishable from a control plane that never wrote one -- and
    failing open on human labor is the one direction P4 forbids. Refusing every
    overwrite is stronger, needs no cross-repo contract, and costs only an
    operation that was never useful: re-extracting into a directory whose
    contents nothing describes.

    A deliberate new selection is a new run, through ``revision``.

    Raised before any worker starts, naming every conflict at once. Raising
    inside ``_extract_one`` would abort partway with some sequences extracted and
    the index carrying rows for them -- a half-applied run whose report is a
    traceback.
    """
    occupied = sorted(
        {
            str(spec.seq_dir)
            for spec in specs
            if spec.overwrite and spec.seq_dir.exists()
        }
    )
    if not occupied:
        return
    listed = "\n  ".join(occupied)
    message = (
        f"overwrite=True would remove {len(occupied)} existing frame "
        f"directory(ies):\n  {listed}\n"
        f"Annotations reference these images by path, and mosaic cannot tell "
        f"whether any exist. Extract a new selection instead by bumping "
        f"`revision`, which mints a new run and leaves these in place."
    )
    raise AnnotatedFramesWouldBeDestroyed(message)


class ListedFramesRefused(ValueError):
    """A ``list`` extraction names frames or entries the run cannot write."""


_SHOWN_PER_CAMERA: Final = 5
"""How many out-of-range indices a refusal names per camera."""


def _refuse_out_of_range(specs: Sequence[_ExtractSpec]) -> None:
    """Refuse listed indices past the end of a camera's media, naming them all.

    Checked against the clips' summed frame counts, which is the length the
    reader indexes and the length ``export-joined`` verifies a join against.
    Raised before any worker starts, for the reason :func:`_refuse_to_overwrite`
    gives: a worker's failure is reported per camera and the run continues, so a
    wrong index would otherwise leave that camera with no frames and no refusal.
    """
    problems: list[str] = []
    for spec in specs:
        if spec.frame_indices is None:
            continue
        past = [index for index in spec.frame_indices if index >= spec.n_media_frames]
        if not past:
            continue
        shown = ", ".join(str(index) for index in past[:_SHOWN_PER_CAMERA])
        rest = len(past) - _SHOWN_PER_CAMERA
        more = f" and {rest} more" if rest > 0 else ""
        problems.append(
            f"{_spec_label(spec)} has {spec.n_media_frames} frames (0 to "
            f"{spec.n_media_frames - 1}); listed {shown}{more} fall past its end"
        )
    if not problems:
        return
    listed = "\n  ".join(problems)
    message = (
        f"{len(problems)} camera(s) list frames their media does not hold:\n"
        f"  {listed}\n"
        f"Indices count from 0 across a recording's clips in order, as its "
        f"tracks' frame column does."
    )
    raise ListedFramesRefused(message)


def _refuse_unjoined(ds: Dataset, media_scope: Sequence[ResolvedScopeEntry]) -> None:
    """Refuse every camera whose clips need a join that cannot be read.

    ``MultiVideoReader`` refuses clips whose frame rates disagree, and stays
    strict; such a recording is read through its ``export-joined`` file.
    Building one is minutes of I/O and belongs to that op, so a missing join is
    refused here, every camera at once, before any worker starts.
    """
    missing = missing_joins(ds, media_scope, asker="extract-frames")
    if not missing:
        return
    reasons = "\n".join(join.reason for join in missing)
    message = (
        f"{len(missing)} camera(s) have clips at different frame rates and no "
        f"join to read them through:\n{reasons}"
    )
    raise JoinedExportMissingError(message)


def _listed_entries(
    listed: Mapping[Entry, tuple[int, ...]], scoped: list[Entry] | None
) -> tuple[list[Entry], int]:
    """The entries a ``list`` run extracts, and how many scoped entries it skips.

    The list chooses the entries and the scope can only narrow them. An unset
    scope covers every listed entry; a scope naming some entries covers those it
    shares with the list, and one sharing none is refused rather than run over
    nothing, because the two were built for different entries. A scope that
    names no entry at all is the empty scope every method treats the same way.
    """
    if scoped is None:
        return sorted(listed), 0
    kept = sorted(set(scoped) & set(listed))
    if scoped and not kept:
        named = ", ".join(f"({group}, {sequence})" for group, sequence in scoped)
        message = (
            f"The scope names {named}, and frames lists none of them. The list "
            f"chooses the entries and the scope can only narrow it."
        )
        raise ListedFramesRefused(message)
    return kept, len(set(scoped) - set(listed))


def _refuse_listed_without_media(
    entries: Sequence[Entry], media_scope: Sequence[ResolvedScopeEntry]
) -> None:
    """Refuse listed entries the dataset has no media for, naming each one."""
    resolved = {(entry.group, entry.sequence) for entry in media_scope}
    absent = [entry for entry in entries if entry not in resolved]
    if not absent:
        return
    named = ", ".join(f"({group}, {sequence})" for group, sequence in absent)
    message = (
        f"frames lists {named}, which this dataset holds no media for. Check "
        f"the group and sequence names against the media index."
    )
    raise ListedFramesRefused(message)


def frames_identity_payload(params: ExtractFramesParams) -> dict[str, object]:
    """What an extraction identifier is hashed from, and what ``run_params.json`` holds.

    ``identity_dump()``, plus the two ``HASH_EXCLUDE`` terms that name a run
    when they are set: ``_revision`` when non-zero, and ``_frames`` under
    ``list``. Each is added only when set, the omit-when-absent rule
    ``compute_run_id`` applies to ``_tracks`` and ``_scope_entries``, because
    ``json.dumps(sort_keys=True)`` digests an absent key differently from a key
    whose value is empty. So a run of either sampling method at revision 0
    hashes exactly what it always did.

    ``_frames`` is the list as the params validator normalised it -- entries in
    order, each entry's indices sorted and unique -- so the same frames listed
    another way name the same run.
    """
    payload = params.identity_dump()
    if params.revision:
        payload["_revision"] = int(params.revision)
    if params.frames is not None:
        payload["_frames"] = [
            [item.group, item.sequence, list(item.indices)] for item in params.frames
        ]
    return payload


def frames_run_id(method: ExtractionMethod, params: ExtractFramesParams) -> str:
    """Mint an extraction run identifier. **Frozen -- do not change this.**

    Deliberately does *not* call ``op_run_id``, and deliberately carries no
    version segment. This is the only mosaic identifier pinned outside mosaic:
    ``mosaic-api`` writes it to ``AnnotationFrame.run_id``, a Dolt-tracked
    column, *and embeds it mid-string* in ``image_path`` on rows carrying
    keypoint annotation labor, where ``image_path`` is additionally a restorable
    value column -- and it discovers runs by reading this directory name off
    disk. Moving it orphans every annotated frame path, recoverable only by
    re-annotating.

    Frozen in algorithm, digest width **and payload**: adding a field to
    ``ExtractFramesParams`` that reaches ``identity_dump()`` moves it just as
    surely as changing the format would. A deliberate new selection is expressed
    as an explicit revision parameter (item 6.4), never by re-minting this.

    ``ExtractFramesOp.version`` stays declared -- ``list_ops`` and ``describe_op``
    read it -- but it is provenance here, not identity.

    **``revision`` and ``frames`` are the two terms that may enter, and only
    when set** (:func:`frames_identity_payload`). Both are ``HASH_EXCLUDE``, so
    ``identity_dump()`` never carries them, and each is added only when set. So
    revision 0 of either sampling method reproduces every identifier on every
    dataset in existence, byte for byte, and the golden corpus proves it rather
    than this docstring asserting it. A ``list`` run is named by its frames,
    which is a new identifier and moves none.
    """
    return f"{method}-{hash_params(frames_identity_payload(params))}"


def _run_extract_frames(
    ds: Dataset,
    p: ExtractFramesParams,
    scope: ResolvedScope,
    overwrite: bool,
    ctx: JobContext,
) -> str:
    """Extraction body executed inside a job_context (the op's payload)."""
    params_hash = hash_params(p.identity_dump())
    # Through the op's plan_identity, so this run is named in one place.
    run_id = ExtractFramesOp().plan_identity(ds, p, scope).run_id
    ctx.set_run_id(run_id)
    run_root = frames_run_root(ds, p.method, run_id)

    listed = (
        None if p.frames is None else {item.entry: item.indices for item in p.frames}
    )
    skipped = 0
    entries = scope.op_entries
    if listed is not None:
        entries, skipped = _listed_entries(listed, entries)

    media_scope = ds.resolve_media_scope(entries)
    if listed is not None and entries is not None:
        _refuse_listed_without_media(entries, media_scope)
    if not media_scope:
        print(
            "[extract_frames] No media entries match the given scope.", file=sys.stderr
        )
        return run_id
    _refuse_unjoined(ds, media_scope)

    # Build picklable per-(group, sequence, camera) work specs (temporal chunks
    # of one camera merged). resolve_media_scope yields one entry per camera, so
    # the cameras of a synchronized recording run independently rather than as
    # one concatenated timeline; it also routes each entry by its transcode
    # verdict, so a required-but-unlinked entry raises (fail loud) rather than
    # opening a defective original, and a routed entry carries the analysis
    # derivative's facts.
    # Resolved once for the whole scope rather than per entry: one media-index
    # read and one pass per source root, where a per-entry lookup would re-read
    # both files for every camera.
    source_uids, source_compositions = _source_identity_maps(
        ds, [(entry.group, entry.sequence) for entry in media_scope]
    )
    specs: list[_ExtractSpec] = []
    for entry in media_scope:
        group, sequence, camera, resolved = (
            entry.group,
            entry.sequence,
            entry.camera,
            entry.resolved,
        )
        # A camera whose clips differ in frame rate is read through its join,
        # which holds the same frames on one timeline; _refuse_unjoined has
        # already refused any camera without one.
        joined = join_to_read(ds, entry, asker="extract-frames")
        paths = (joined,) if joined is not None else tuple(resolved.paths)
        facts = () if joined is not None else tuple(resolved.facts)
        # A multi-camera recording writes each camera into its own subdir so the
        # cameras never collide; single-camera media keeps the flat layout.
        key = make_entry_key(group, sequence)
        seq_dir = run_root / key / camera if camera else run_root / key
        specs.append(
            _ExtractSpec(
                group=group,
                sequence=sequence,
                camera=camera,
                video_paths=paths,
                facts=facts,
                video_uuids=source_uids.get((group, sequence, camera), ""),
                media_composition=source_compositions.get((group, sequence), ""),
                seq_dir=seq_dir,
                run_id=run_id,
                params_hash=params_hash,
                n_frames=p.n_frames,
                method=p.method,
                frame_indices=None if listed is None else listed[(group, sequence)],
                n_media_frames=sum(int(clip.frame_count) for clip in resolved.facts),
                start_frame=p.start_frame,
                end_frame=p.end_frame,
                candidate_step=int(p.candidate_step),
                crop=p.crop,
                random_state=int(p.random_state),
                kmeans_resize=p.kmeans_resize,
                kmeans_grayscale=p.kmeans_grayscale,
                kmeans_max_candidates=p.kmeans_max_candidates,
                kmeans_batch_size=int(p.kmeans_batch_size),
                kmeans_max_iter=int(p.kmeans_max_iter),
                kmeans_n_init=p.kmeans_n_init,
                overwrite=overwrite,
            )
        )

    _refuse_out_of_range(specs)
    _refuse_to_overwrite(specs)

    # Created only once every refusal has passed, so a refused run leaves no
    # run directory for a scan of the frames root to list as a run with none.
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        (run_root / "run_params.json").write_text(
            json.dumps(json_ready(frames_identity_payload(p)), indent=2)
        )
    except Exception as exc:
        print(
            f"[extract_frames:{p.method}] failed to save run_params.json: {exc}",
            file=sys.stderr,
        )

    ctx.set_total(len(specs))
    idx = frames_index(frames_index_path(ds, p.method))
    idx.ensure()
    index_rows: list[FramesIndexRow] = []

    def _collect(row: FramesIndexRow | None) -> None:
        if row is not None:
            # _rewrite_manifest needs the absolute seq_dir; store the row with a
            # dataset-root-relative abs_path so the index stays portable.
            #
            # ``video_abs_path`` is made relative the same way and for the same
            # reason. It was the one path cell in this row left absolute, and
            # "frames" is absent from ``_INDEX_PATH_COLUMNS``, so no portability
            # pass ever reached it -- a moved or synced dataset kept a frames
            # index pointing at the old machine's tree. Registering it there
            # instead would not work: those passes do prefix substring
            # replacement with no split support, and this cell is multi-valued
            # for a multi-clip camera. Storing it relative makes it portable by
            # construction. Rows written before this stay absolute and keep
            # resolving, per migration M1's add-do-not-rename rule.
            _rewrite_manifest(ds, row.abs_path)
            index_rows.append(
                dataclasses.replace(
                    row,
                    abs_path=Path(ds.relative_to_root(row.abs_path)),
                    video_abs_path=_relative_video_paths(ds, row.video_abs_path),
                )
            )

    max_workers = _resolve_max_workers(p.parallel_workers)
    p_mode = (p.parallel_mode or "thread").lower()
    if p_mode not in {"thread", "process"}:
        p_mode = "thread"

    try:
        if max_workers > 1:
            PoolCls = ProcessPoolExecutor if p_mode == "process" else ThreadPoolExecutor
            with PoolCls(max_workers=max_workers) as pool:
                futures = {pool.submit(_extract_one, spec): spec for spec in specs}
                done = 0
                for future in as_completed(futures):
                    if ctx.cancel_token.is_cancelled():
                        for f in futures:
                            f.cancel()
                        raise Cancelled()
                    row = future.result()
                    done += 1
                    spec = futures[future]
                    ctx.progress.on_entry_end(done, len(specs), _spec_label(spec))
                    ctx.heartbeat(done)
                    _collect(row)
        else:
            for i, spec in enumerate(specs):
                ctx.check_cancel()
                key = _spec_label(spec)
                ctx.progress.on_entry_start(i, len(specs), key)
                _collect(_extract_one(spec))
                ctx.progress.on_entry_end(i + 1, len(specs), key)
                ctx.heartbeat(i + 1)
    finally:
        if index_rows:
            idx.append(index_rows)
            idx.mark_finished(run_id)

    unlisted = f"; {skipped} scoped entries list no frames" if skipped else ""
    print(
        f"[extract_frames:{p.method}] completed run_id={run_id} "
        f"({len(index_rows)}/{len(specs)} sequences{unlisted}) -> {run_root}"
    )
    return run_id


@register_op
class ExtractFramesOp(Op[ExtractFramesParams]):
    """Sample representative video frames as PNGs for annotation."""

    kind = "extract-frames"
    category = "extract"
    domain = "tracking"
    version = "0.1"
    scope_takes = "any"
    scope_dependent = False
    Params = ExtractFramesParams

    def target(self, params: ExtractFramesParams, scope: ResolvedScope) -> str:
        return f"extract-{params.method}"

    def plan_identity(
        self,
        ds: Dataset,
        params: ExtractFramesParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this extraction run will be called.

        Pure in the params -- no dataset read, and nothing to defer. The
        identifier is frozen: mosaic-api embeds it mid-string in the paths of the
        images an annotator works on, so a change here moves work somebody has
        already done.
        """
        return OpIdentity(run_id=frames_run_id(params.method, params))

    def run(
        self,
        ds: Dataset,
        params: ExtractFramesParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        return _run_extract_frames(ds, params, scope, overwrite, ctx)


def extract_frames(
    ds: Dataset,
    n_frames: int | None = None,
    method: ExtractionMethod = "uniform",
    *,
    frames: Mapping[Entry, Iterable[int]] | None = None,
    revision: int = 0,
    scope: Scope | None = None,
    overwrite: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
    candidate_step: int = 1,
    crop: tuple[int, int, int, int] | None = None,
    kmeans_resize: tuple[int, int] = (64, 64),
    kmeans_grayscale: bool = True,
    kmeans_max_candidates: int | None = 5000,
    kmeans_batch_size: int = 1024,
    kmeans_max_iter: int = 100,
    kmeans_n_init: str | int = "auto",
    random_state: int = 42,
    parallel_workers: int | str | None = "auto",
    parallel_mode: ParallelMode = "thread",
    # Job Contract
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: "ProgressCallback | None" = None,
    cancel_token: CancelToken | None = None,
) -> str:
    """Extract representative frames from media as a tracked Job-Contract run.

    Ergonomic typed front door for the ``extract-frames`` tracking op: builds
    :class:`ExtractFramesParams` and dispatches via
    :func:`mosaic.core.pipeline.ops.run_op`, which records the attempt, reports
    per-sequence progress, and supports cooperative cancellation. Returns the
    content ``run_id``.

    Parameters mirror :class:`ExtractFramesParams` -- the method, the k-means
    knobs, ``revision`` and ``parallel_workers`` / ``parallel_mode`` -- plus the
    standard contract knobs
    (``execution_id``/``owner``/``track``/``progress_callback``/``cancel_token``).

    *frames* is the list for ``method="list"``: a mapping from each
    ``(group, sequence)`` entry to the frame indices to write for it, which
    becomes ``ExtractFramesParams.frames``. ``n_frames`` is then left unset. The
    list chooses the entries, and *scope* can only narrow it.

    *scope* is the selector every other entry point takes, and *overwrite* says
    whether an existing frame set is recomputed. Both reach the op through
    ``run_op`` and neither enters the params. What a run covers and whether it
    recomputes describe the attempt, where the params describe the recipe.
    """
    listed = (
        None
        if frames is None
        else tuple(
            EntryFrames(group=group, sequence=sequence, indices=tuple(indices))
            for (group, sequence), indices in frames.items()
        )
    )
    params = ExtractFramesParams(
        n_frames=n_frames,
        method=method,
        frames=listed,
        start_frame=start_frame,
        end_frame=end_frame,
        candidate_step=candidate_step,
        crop=crop,
        random_state=random_state,
        kmeans_resize=kmeans_resize,
        kmeans_grayscale=kmeans_grayscale,
        kmeans_max_candidates=kmeans_max_candidates,
        kmeans_batch_size=kmeans_batch_size,
        kmeans_max_iter=kmeans_max_iter,
        kmeans_n_init=kmeans_n_init,
        revision=revision,
        parallel_workers=parallel_workers,
        parallel_mode=parallel_mode,
    )
    return run_op(
        ds,
        "extract-frames",
        params,
        scope=scope,
        overwrite=overwrite,
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )


def list_frame_runs(ds: Dataset, method: str | None = None) -> pd.DataFrame:
    """
    List all frame extraction runs tracked in the frames index.

    Parameters
    ----------
    method : str, optional
        Filter to a specific method ("uniform", "kmeans" or "list").
        If None, returns runs across all methods.

    Returns
    -------
    pd.DataFrame
        Index rows for matching extraction runs.
    """
    frames_root = ds.get_root("frames")
    if not frames_root.exists():
        return pd.DataFrame(columns=FRAMES_INDEX_COLUMNS)

    methods = [method] if method else root_subdirectories(ds, "frames")
    dfs = []
    for m in methods:
        idx_path = frames_root / m / "index.csv"
        if idx_path.exists():
            # Through the typed reader, not a bare read_csv: inference reads an
            # all-digit cell as int64 and an empty one as NaN, so a numeric-looking
            # ``video_uuids`` or ``params_hash`` would come back as a number and a
            # blank as the float NaN that ``adopt_frames_columns`` exists to keep
            # off this frame. The same defect item 0.2 fixed for four other
            # readers, in the one that was missed.
            dfs.append(frames_index(idx_path).read())
    if not dfs:
        return pd.DataFrame(columns=FRAMES_INDEX_COLUMNS)
    return pd.concat(dfs, ignore_index=True)


def get_frame_paths(
    ds,
    method: str,
    run_id: str | None = None,
    group: str | None = None,
    sequence: str | None = None,
) -> list[Path]:
    """
    Return paths to extracted frame PNGs for a given scope.

    Parameters
    ----------
    method : str
        Extraction method ("uniform", "kmeans" or "list").
    run_id : str, optional
        Specific run_id. If None, uses the latest run.
    group, sequence : str, optional
        Filter to a specific (group, sequence).

    Returns
    -------
    list[Path]
        Sorted list of PNG file paths.
    """
    frames_root = ds.get_root("frames")
    method_root = frames_root / method
    if not method_root.exists():
        return []

    # Resolve run_id
    if run_id is None:
        idx_path = method_root / "index.csv"
        if not idx_path.exists():
            return []
        df = pd.read_csv(idx_path)
        if df.empty:
            return []
        run_id = df["run_id"].iloc[-1]

    run_root = method_root / run_id
    if not run_root.exists():
        return []

    # Collect PNG paths. rglob so a per-camera subdir (a multi-camera recording
    # writes frames/<seq>/<camera>/*.png) is descended into; single-camera frames
    # sit directly under the sequence dir and are still found.
    if group is not None or sequence is not None:
        seq_label = make_entry_key(group or "", sequence or "")
        seq_dir = run_root / seq_label
        return sorted(seq_dir.rglob("*.png")) if seq_dir.exists() else []
    return sorted(run_root.rglob("*.png"))


def get_frame_manifests(
    ds,
    method: str,
    run_id: str | None = None,
    group: str | None = None,
    sequence: str | None = None,
) -> list[dict[str, object]]:
    """
    Load run_info.json manifests from extracted frame directories.

    Parameters
    ----------
    method : str
        Extraction method ("uniform", "kmeans" or "list").
    run_id : str, optional
        Specific run_id. If None, uses the latest run.
    group, sequence : str, optional
        Filter to a specific (group, sequence).

    Returns
    -------
    list[dict]
        List of manifest dicts loaded from run_info.json files,
        one per sequence directory. Each dict contains video_path,
        files, video_meta, selected_frame_indices, etc.
    """
    frames_root = ds.get_root("frames")
    method_root = frames_root / method
    if not method_root.exists():
        return []

    # Resolve run_id
    if run_id is None:
        idx_path = method_root / "index.csv"
        if not idx_path.exists():
            return []
        df = pd.read_csv(idx_path)
        if df.empty:
            return []
        run_id = df["run_id"].iloc[-1]

    run_root = method_root / run_id
    if not run_root.exists():
        return []

    # Collect run_info.json manifests. rglob descends the optional per-camera
    # subdir, so a multi-camera recording yields one manifest per camera and a
    # single-camera sequence yields one manifest directly under its dir.
    if group is not None or sequence is not None:
        seq_label = make_entry_key(group or "", sequence or "")
        seq_root = run_root / seq_label
        manifest_paths = (
            sorted(seq_root.rglob("run_info.json")) if seq_root.exists() else []
        )
    else:
        manifest_paths = sorted(run_root.rglob("run_info.json"))

    manifests = []
    for manifest_path in manifest_paths:
        data = json.loads(manifest_path.read_text())
        # Resolve relative paths so callers always see absolute paths
        manifest_dir = manifest_path.parent
        for f in data.get("files", []):
            if "path" in f:
                fp = Path(f["path"])
                if not fp.is_absolute():
                    f["path"] = str((manifest_dir / fp).resolve())
        if "output_dir" in data:
            od = Path(data["output_dir"])
            if not od.is_absolute():
                data["output_dir"] = str(ds.resolve_path(od))
        if "video_path" in data:
            vp = Path(data["video_path"])
            if not vp.is_absolute():
                data["video_path"] = str(ds.resolve_path(vp))
        manifests.append(data)
    return manifests


def _frame_run_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[CameraEntry]]:
    """Every frame-extraction run, keyed by ``(group, sequence, camera)``.

    The camera is part of the key rather than a detail. The cameras of one
    recording share a ``(group, sequence)``, which is why this index dedups on
    the four-part key -- and without the camera here, a run that extracted one
    camera would read as covering the entry and the other would never be seen as
    missing.
    """
    from mosaic.core.pipeline.inventory.model import (
        ArtifactRecord,
        Coverage,
        FrameRunRef,
        classify,
    )
    from mosaic.core.pipeline.index_csv import index_records

    records: list[ArtifactRecord[CameraEntry]] = []
    wanted = scope.selector.entry_pairs
    for method in root_subdirectories(ds, "frames"):
        index_path = frames_index_path(ds, method)
        reader.note(index_path)
        frame = reader.frame(index_path, lambda m=method: list_frame_runs(ds, m))
        if frame.empty or "run_id" not in frame.columns:
            continue
        rows_by_run: dict[str, set[CameraEntry]] = {}
        present_by_run: dict[str, set[CameraEntry]] = {}
        started: dict[str, str] = {}
        finished: dict[str, str] = {}
        for record in index_records(frame):
            run_id = record.get("run_id", "")
            key: CameraEntry = (
                record.get("group", ""),
                record.get("sequence", ""),
                record.get("camera", ""),
            )
            if wanted is not None and (key[0], key[1]) not in wanted:
                continue
            rows_by_run.setdefault(run_id, set()).add(key)
            started.setdefault(run_id, record.get("started_at", ""))
            if record.get("finished_at", ""):
                finished.setdefault(run_id, record.get("finished_at", ""))
            stored = record.get("abs_path", "")
            if stored and ds.resolve_path(stored).exists():
                present_by_run.setdefault(run_id, set()).add(key)
        for run_id in sorted(rows_by_run):
            rows = frozenset(rows_by_run[run_id])
            present = frozenset(present_by_run.get(run_id, set()))
            coverage = Coverage(target=rows, present=present)
            records.append(
                ArtifactRecord[CameraEntry](
                    ref=FrameRunRef(method=method, run_id=run_id),
                    name=method,
                    run_id=run_id,
                    coverage=coverage,
                    status=classify(
                        satisfied=coverage.is_satisfied,
                        any_covered=bool(coverage.covered),
                        orphan_rows=bool(rows - present),
                        orphan_files=False,
                        drifted=False,
                        finished=bool(finished.get(run_id, "")),
                    ),
                    index_path=index_path,
                    rows=rows,
                    orphan_rows=rows - present,
                    started_at=started.get(run_id, ""),
                    finished_at=finished.get(run_id, ""),
                )
            )
    return records


register_inventory_contributor("frame-run", _frame_run_records)
