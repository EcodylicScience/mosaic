"""The frames index, and what the inventory says about frame-extraction runs.

Each extraction method keeps one index under frames/<method>/index.csv, one row per
(run, group, sequence, camera). This module reads and writes it and imports no tracker,
trainer or extraction code, and loads only the media modules that
``mosaic.core.pipeline.inventory`` loads, so an inventory reads frame runs without
loading what writes them. :func:`frame_rows` and :func:`frame_run_file` read a dataset's
frame runs together with those of the datasets it links. It also names the recorded
frame behind an annotated image (:func:`video_frames_of`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.annotations.split import VideoFrame
from mosaic.core.pipeline.dataset_indexes import root_subdirectories
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase, index_records
from mosaic.core.pipeline.inventory._runs import keyed_run_records
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    CameraEntry,
    FrameRunRef,
    InventoryScope,
)
from mosaic.core.pipeline.linked_rows import LINK_ID_COLUMN, merge_linked_rows
from mosaic.core.stored_paths import manifest_file_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mosaic.core.annotations.model import AnnotationFrame
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.inventory._read import IndexReader

__all__ = [
    "FRAMES_INDEX_COLUMNS",
    "FrameFile",
    "FramesIndexRow",
    "adopt_frames_columns",
    "frame_rows",
    "frame_run_file",
    "frame_run_records",
    "frame_run_roots",
    "frames_row_files",
    "frames_index",
    "frames_index_path",
    "frames_run_root",
    "latest_frame_run_id",
    "linked_frames_row_files",
    "list_frame_runs",
    "video_frames_of",
]


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
    dfs: list[pd.DataFrame] = []
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


_FRAME_ROW_KEYS: Final = ("run_id", "group", "sequence", "camera")
"""The columns that identify a frames row, the index's dedup keys."""

_FRAME_ROWS_PATH_COLUMNS: Final = ("abs_path", "video_abs_path")
"""The frames index columns that name a file each."""


def frame_rows(
    ds: Dataset,
    method: str | None = None,
    *,
    read_index: Callable[[Dataset, str | None], pd.DataFrame] = list_frame_runs,
) -> pd.DataFrame:
    """The frames index of *ds* merged with those of the datasets it links.

    Reads each dataset of ``ds.lookup_order("frames")`` with *read_index* and
    merges the frames with :func:`~mosaic.core.pipeline.linked_rows.merge_linked_rows`,
    keyed on ``(run_id, group, sequence, camera)``: an own row shadows a linked
    row of the same run and entry camera, and an earlier link shadows a later
    one. A linked row's ``abs_path`` and single-path ``video_abs_path`` are made
    absolute against its own dataset, and own cells stay as stored. Writes
    nothing.

    Args:
        ds: The reading dataset.
        method: Only the rows of this extraction method, or ``None`` for every
            method.
        read_index: Reads one dataset's rows of *method*, as
            :func:`list_frame_runs` does. A caller that reads many indexes
            passes one that remembers what it read.

    Returns:
        The merged rows with a ``link_id`` column that names the link of each
        row and is ``""`` for an own row.
    """
    return merge_linked_rows(
        [
            (
                linked.link_id,
                linked.dataset.base_dir,
                read_index(linked.dataset, method),
            )
            for linked in ds.lookup_order("frames")
        ],
        _FRAME_ROW_KEYS,
        _FRAME_ROWS_PATH_COLUMNS,
    )


def frame_run_roots(
    ds: Dataset, method: str, run_id: str
) -> tuple[tuple[Dataset, Path], ...]:
    """Each existing root of a frames run, in lookup order, with its dataset.

    A run directory that *ds* created and a linked one that holds the rest of
    the same run are both listed, *ds* first once it exists.

    Returns:
        The dataset and run root of each dataset of ``ds.lookup_order("frames")``
        whose run directory exists. Empty when none does.
    """
    roots = (
        (linked.dataset, frames_run_root(linked.dataset, method, run_id))
        for linked in ds.lookup_order("frames")
    )
    return tuple((owner, root) for owner, root in roots if root.is_dir())


def frame_run_file(ds: Dataset, method: str, run_id: str, name: str) -> Path | None:
    """The file *name* of a frames run in the first run root that holds it.

    Returns:
        The file's path, or ``None`` when no root of :func:`frame_run_roots`
        holds it.
    """
    for _owner, root in frame_run_roots(ds, method, run_id):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def latest_frame_run_id(ds: Dataset, method: str) -> str | None:
    """The latest frames run of *method*: *ds*'s own, else the first link's.

    A dataset's latest run is the run of the last row of its index. A later run
    in a link does not displace *ds*'s own.

    Returns:
        The run's identifier, or ``None`` when no dataset's index holds a row.
    """
    for linked in ds.lookup_order("frames"):
        path = frames_index_path(linked.dataset, method)
        if not path.exists():
            continue
        rows = frames_index(path).read()
        if not rows.empty:
            return str(rows["run_id"].iloc[-1])
    return None


def frame_run_records(
    ds: Dataset, scope: InventoryScope, reader: IndexReader
) -> list[ArtifactRecord[CameraEntry]]:
    """Every frame-extraction run, keyed by ``(group, sequence, camera)``.

    The camera is part of the key rather than a detail. The cameras of one
    recording share a ``(group, sequence)``, which is why this index dedups on
    the four-part key -- and without the camera here, a run that extracted one
    camera would read as covering the entry and the other would never be seen as
    missing.
    """
    return keyed_run_records(
        ds,
        scope,
        reader,
        root_key="frames",
        kinds=None,
        read_index=list_frame_runs,
        key_of=lambda record: (
            record.get("group", ""),
            record.get("sequence", ""),
            record.get("camera", ""),
        ),
        make_ref=lambda method, run_id, _rows: FrameRunRef(
            method=method, run_id=run_id
        ),
        drifted=lambda _run_id, _rows: (),
    )


@dataclass(frozen=True, slots=True)
class FrameFile:
    """One extracted PNG of an entry camera.

    Attributes:
        frame_index: The frame's index on the entry camera's frame axis.
        image: The PNG's dataset-relative POSIX path, the key under which an
            annotation revision names the same image.
        width: The PNG's width in pixels.
        height: The PNG's height in pixels.
    """

    frame_index: int
    image: str
    width: int
    height: int


def _relative_directory(ds: Dataset, stored: str) -> str:
    """The POSIX form of a frames row's entry directory, root-relative where it can be.

    A row written as root-relative text is used as stored. An older row stores an
    absolute directory. It is mapped through the dataset's ``path_map`` and then
    made root-relative, and both forms give one key. A directory outside the
    dataset tree stays absolute.
    """
    cell = Path(stored)
    relative = (
        ds.relative_to_root(ds.resolve_path(cell)) if cell.is_absolute() else stored
    )
    return Path(relative).as_posix()


def _entry_directory_key(
    ds: Dataset, stored: str, *, run_id: str, entry: tuple[str, str, str]
) -> str:
    """The dataset-relative POSIX form of a frames row's entry directory.

    Raises:
        ValueError: If the directory is not under the dataset tree, which has
            no dataset-relative key.
    """
    relative = _relative_directory(ds, stored)
    if Path(relative).is_absolute():
        group, sequence, camera = entry
        raise ValueError(
            f"Frames run {run_id!r}, entry (group={group!r}, sequence={sequence!r}, "
            f"camera={camera!r}): the directory {relative!r} is not under the "
            "dataset tree (it lies outside it, or it is another machine's path that "
            "path_map does not map), so its images have no dataset-relative path."
        )
    return relative


def _entry_files(
    ds: Dataset, stored: str, *, run_id: str, entry: tuple[str, str, str]
) -> tuple[FrameFile, ...]:
    """The files that a frames row's ``run_info.json`` records, by frame index.

    A record's path is read through :func:`manifest_file_path`, which takes an
    absolute legacy record by its file name.

    Raises:
        ValueError: If the row's directory lies outside the dataset tree, or its
            ``run_info.json`` is missing, unreadable or lacks a record field.
    """
    directory = PurePosixPath(
        _entry_directory_key(ds, stored, run_id=run_id, entry=entry)
    )
    manifest = ds.resolve_path(stored) / "run_info.json"
    try:
        info = json.loads(manifest.read_text())
        files = [
            FrameFile(
                frame_index=int(record["frame_index"]),
                image=str(directory / manifest_file_path(str(record["path"]))),
                width=int(record["width"]),
                height=int(record["height"]),
            )
            for record in info.get("files", [])
        ]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        group, sequence, camera = entry
        msg = (
            f"Frames run {run_id!r}, entry (group={group!r}, sequence={sequence!r}, "
            f"camera={camera!r}): its run_info.json at {manifest} cannot be read "
            f"({type(exc).__name__}: {exc}), so the frame index of its images is "
            "unknown."
        )
        raise ValueError(msg) from exc
    return tuple(sorted(files, key=lambda file: file.frame_index))


def frames_row_files(ds: Dataset, row: Mapping[str, str]) -> tuple[FrameFile, ...]:
    """The extracted PNGs that one frames-index row records.

    Reads the ``files`` records of the ``run_info.json`` in the row's
    ``abs_path`` directory. Nothing is created or written.

    Args:
        ds: The dataset holding the frames run.
        row: One record of :func:`list_frame_runs`, read for its ``run_id``,
            ``group``, ``sequence``, ``camera`` and ``abs_path``.

    Returns:
        The files sorted by frame index.

    Raises:
        ValueError: If the row's directory lies outside the dataset tree, or its
            ``run_info.json`` is missing, unreadable or lacks a record field.
    """
    entry = (row.get("group", ""), row.get("sequence", ""), row.get("camera", ""))
    return _entry_files(
        ds, row.get("abs_path", ""), run_id=row.get("run_id", ""), entry=entry
    )


def linked_frames_row_files(
    ds: Dataset, row: Mapping[str, str]
) -> tuple[FrameFile, ...]:
    """The extracted PNGs of one :func:`frame_rows` record, each named from *ds*.

    An own row reads as :func:`frames_row_files` reads it. A linked row is read
    in the dataset of its ``link_id``, and each image is named by
    ``ds.relative_to_root``, which keeps a file outside *ds* absolute.

    Raises:
        ValueError: If no reachable link has the row's ``link_id``, the row's
            directory lies outside its dataset's tree, or its ``run_info.json``
            is missing, unreadable or lacks a record field.
    """
    link_id = row.get(LINK_ID_COLUMN, "")
    if not link_id:
        return frames_row_files(ds, row)
    owner = next(
        (
            linked.dataset
            for linked in ds.lookup_order("frames")
            if linked.link_id == link_id
        ),
        None,
    )
    if owner is None:
        msg = f"No reachable link of {ds.name!r} has the id {link_id!r}."
        raise ValueError(msg)
    return tuple(
        replace(
            file,
            image=Path(ds.relative_to_root(owner.resolve_path(file.image))).as_posix(),
        )
        for file in frames_row_files(owner, row)
    )


@dataclass(frozen=True, slots=True)
class _ExtractedEntry:
    """One frames row's entry camera, and the frame index of each file it wrote."""

    group: str
    sequence: str
    camera: str
    index_by_image: dict[str, int]


@dataclass(frozen=True, slots=True)
class _OwnedImage:
    """An annotated image's owner, by resolved manifest path, and its key there."""

    manifest: Path
    key: str


def video_frames_of(
    ds: Dataset,
    image_root: Path,
    frames: Sequence[AnnotationFrame],
    *,
    revision_origin: str,
) -> tuple[VideoFrame, ...]:
    """The recorded frame behind each annotation frame, one per frame in input order.

    Each image is looked up in the frames indexes of its owner. A relative image path
    that resolves inside *image_root*'s directory tree is owned by the dataset whose
    base directory is *image_root*, which is *ds* or one of its linked datasets. An
    absolute image path, and a relative one that resolves outside that tree, such as a
    ``../`` path, is owned by the dataset whose manifest is nearest around it
    (:meth:`Dataset.open_enclosing`), whether or not a dataset is at *image_root*. A
    frames row of the owner matches an image when the row's entry directory is the
    image's directory. The image then takes the row's group, sequence and camera, and
    the owner's manifest uuid as its origin. Its frame index is the one that the
    ``run_info.json`` records for the image's dataset-relative path, or -1 when the run
    did not write it. The annotation frame's sequence and index are not read for a
    matched image, because an import of extracted PNGs records index -1.

    An image that no row matches, or that has no owner, takes *revision_origin*, camera
    ``""``, the group its annotation frame records, the sequence it records, else the
    image's directory's name, and the frame index it records. Index -1 keeps the image's
    stem. No dataset tree is changed.

    Args:
        ds: The dataset that reads the revision.
        image_root: The revision's image root, which its relative image paths
            are relative to.
        frames: The annotation frames.
        revision_origin: The manifest uuid of the dataset that saved the
            revision.

    Returns:
        One recorded frame per annotation frame, in input order.

    Raises:
        ValueError: If a matched row's ``run_info.json`` cannot be read.
    """
    by_path = [_located_by_path(frame.image_path, image_root) for frame in frames]
    enclosing = ds.open_enclosing({path.parent for path in by_path if path is not None})
    root_owner = _dataset_at(ds, image_root)
    owners: dict[Path, Dataset] = {}
    owned: list[_OwnedImage | None] = []
    for frame, path in zip(frames, by_path, strict=True):
        owner = root_owner if path is None else enclosing[path.parent]
        key = None if owner is None else _image_key(owner, frame.image_path, path)
        if owner is None or key is None:
            owned.append(None)
            continue
        manifest = owner.manifest_path.resolve()
        _ = owners.setdefault(manifest, owner)
        owned.append(_OwnedImage(manifest=manifest, key=key))
    entries = {
        manifest: _entries_holding(
            owner,
            {
                _directory_of(image.key)
                for image in owned
                if image is not None and image.manifest == manifest
            },
        )
        for manifest, owner in owners.items()
    }
    located: list[VideoFrame] = []
    for frame, image in zip(frames, owned, strict=True):
        entry = (
            None
            if image is None
            else entries[image.manifest].get(_directory_of(image.key))
        )
        if image is None or entry is None:
            located.append(_unextracted(frame, revision_origin))
            continue
        index = entry.index_by_image.get(image.key, -1)
        located.append(
            VideoFrame(
                origin_uuid=owners[image.manifest].uuid or "",
                group=entry.group,
                sequence=entry.sequence,
                camera=entry.camera,
                frame_index=index,
                image_stem=frame.image_path.stem if index == -1 else "",
            )
        )
    return tuple(located)


def _dataset_at(ds: Dataset, image_root: Path) -> Dataset | None:
    """*ds*, or the linked dataset, whose base directory is *image_root*.

    Returns ``None`` when neither is.
    """
    target = image_root.resolve()
    for linked in ds.lookup_order():
        if linked.dataset.base_dir.resolve() == target:
            return linked.dataset
    return None


def _located_by_path(image: Path, image_root: Path) -> Path | None:
    """The path by which an image's owner is found, or ``None`` for the image root's.

    An absolute image path is returned as it is, and a relative one that
    resolves outside *image_root*'s directory tree is returned joined to
    *image_root*.
    """
    if image.is_absolute():
        return image
    joined = image_root / image
    inside = joined.resolve().is_relative_to(image_root.resolve())
    return None if inside else joined


def _image_key(owner: Dataset, image: Path, by_path: Path | None) -> str | None:
    """The image's dataset-relative POSIX path in *owner*, or ``None`` outside it.

    An image keyed at the image root keeps its relative path, which is relative
    to *owner*'s base directory. An image found by its path, *by_path*, is made
    relative to *owner*.
    """
    if by_path is None:
        return image.as_posix()
    relative = Path(owner.relative_to_root(by_path))
    return None if relative.is_absolute() else relative.as_posix()


def _directory_of(image: str) -> str:
    return PurePosixPath(image).parent.as_posix()


def _entries_holding(
    owner: Dataset, directories: set[str]
) -> dict[str, _ExtractedEntry]:
    """The frames rows of *owner* whose entry directory is one of *directories*.

    Keyed by that directory's dataset-relative POSIX path. A row outside the
    dataset tree is skipped, because its images lack a dataset-relative path.
    """
    found: dict[str, _ExtractedEntry] = {}
    if not directories:
        return found
    for record in index_records(list_frame_runs(owner)):
        stored = record.get("abs_path", "")
        directory = _relative_directory(owner, stored) if stored else ""
        if directory not in directories or directory in found:
            continue
        files = frames_row_files(owner, record)
        found[directory] = _ExtractedEntry(
            group=record.get("group", ""),
            sequence=record.get("sequence", ""),
            camera=record.get("camera", ""),
            index_by_image={file.image: file.frame_index for file in files},
        )
    return found


def _unextracted(frame: AnnotationFrame, revision_origin: str) -> VideoFrame:
    """The recorded frame of an image that no frames row matches, from its revision.

    A revision without a recorded sequence, such as a foreign COCO export claimed
    into the series, falls back to the image's directory, which is where
    mosaic's frame extraction puts a sequence's frames.
    """
    index = frame.frame_index if frame.frame_index >= 0 else -1
    return VideoFrame(
        origin_uuid=revision_origin,
        group=frame.group,
        sequence=frame.video or frame.image_path.parent.name,
        camera="",
        frame_index=index,
        image_stem=frame.image_path.stem if index == -1 else "",
    )
