"""Building media files, media-index rows, and transcode derivatives.

The row builders exist because a media row has far more columns than any one
test cares about, and a row missing the probed facts is not a row the toolkit
produces -- so a test built on one measures a shape that cannot occur.
"""

from __future__ import annotations

import dataclasses
import subprocess
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive
from mosaic_media.transcode import Target

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline.composition import MediaMember, media_composition
from mosaic.core.pipeline.joined_export import (
    JoinedExportParams,
    joined_export_path,
    joined_recipe_hash,
)
from tests.helpers.datasets import make_dataset
from tests.helpers.environment import require_ffmpeg


def _shade_for_name(name: str) -> int:
    """The grey level *name* stands for, so two clips named apart look apart."""
    return sum(name.encode()) % 200 + 20


_CAMERA_ARGS: tuple[str, ...] = ("-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p")
_LOSSLESS_RGB_ARGS: tuple[str, ...] = (
    "-c:v",
    "libx264rgb",
    "-qp",
    "0",
    "-pix_fmt",
    "bgr24",
)


def write_h264_mp4(
    path: Path,
    *,
    frames: int = 6,
    size: tuple[int, int] = (64, 48),
    shade: int | Literal["from-name"] = 0,
    fps: float = 30.0,
    paint: Callable[[int], npt.NDArray[np.uint8]] | None = None,
    lossless: bool = False,
) -> None:
    """A small constant-frame-rate H.264 mp4, written by a subprocess ffmpeg.

    Every frame is the flat *shade*, unless *paint* is given. Frame ``i`` is then
    ``paint(i)``, a BGR image of *size*, so a test can make every frame tell which
    frame it is.

    *lossless* writes RGB H.264 with libx264rgb at quantizer 0 and does not convert
    the pixel format. Each frame then decodes to the BGR values that were written. The
    default is yuv420p at CRF 18, the format of a camera's file.

    H.264 because that is what source media *is* -- a camera writes it, and a
    tool mosaic hands a file to can always decode it. A fixture in a codec a
    reader might not hold would exercise a path that cannot occur for an
    original, and would trip the gate in
    :func:`~mosaic.tracking.common.tool_input.refuse_undecodable_codec` for a
    reason nothing about the test is asking about.

    A subprocess rather than ``FFmpegVideoWriter``, which encodes AV1 and only
    AV1: PyAV links FFmpeg into this process, so naming a GPL encoder in-process
    would link libx264 into everything that imports the toolkit. An argv does
    not link, which is the same reason ``joined_export`` shells out to normalise
    a clip.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    level = _shade_for_name(path.name) if shade == "from-name" else int(shade)
    width, height = size
    if paint is None:
        payload = np.full((height, width, 3), level, np.uint8).tobytes() * frames
    else:
        painted = [paint(i) for i in range(frames)]
        wrong = [
            i for i, image in enumerate(painted) if image.shape != (height, width, 3)
        ]
        if wrong:
            message = f"paint({wrong[0]}) is not a {width}x{height} BGR image"
            raise ValueError(message)
        payload = b"".join(image.tobytes() for image in painted)
    _ = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            *(_LOSSLESS_RGB_ARGS if lossless else _CAMERA_ARGS),
            str(path),
        ],
        input=payload,
        check=True,
        capture_output=True,
    )


_CODE_GRID: Final = (4, 3)
"""Columns and rows of the cells a frame code is painted in, one bit per cell."""
_CODE_BITS: Final = _CODE_GRID[0] * _CODE_GRID[1]
_CODE_ZERO: Final = 32
_CODE_ONE: Final = 224
_CODE_THRESHOLD: Final = (_CODE_ZERO + _CODE_ONE) // 2
_CODE_MARGIN: Final = 48
"""The least distance from the threshold at which a decoded cell is read as a bit."""


def _code_cells(width: int, height: int) -> list[tuple[slice, slice, slice, slice]]:
    """Return each code cell of a *width* by *height* frame, in reading order.

    Each cell is ``(rows, columns, inner_rows, inner_columns)``: its whole extent,
    and the central half of it that :func:`read_frame_code` measures.
    """
    columns, rows = _CODE_GRID
    cells: list[tuple[slice, slice, slice, slice]] = []
    for row in range(rows):
        top, bottom = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            left, right = column * width // columns, (column + 1) * width // columns
            inset_y, inset_x = (bottom - top) // 4, (right - left) // 4
            cells.append(
                (
                    slice(top, bottom),
                    slice(left, right),
                    slice(top + inset_y, bottom - inset_y),
                    slice(left + inset_x, right - inset_x),
                )
            )
    return cells


def paint_frame_code(
    frame: int, size: tuple[int, int] = (64, 48)
) -> npt.NDArray[np.uint8]:
    """Return a BGR frame of *size* that spells *frame* in binary, one bit per cell.

    The frame is a 4 by 3 grid of cells, most significant bit first in reading
    order, each flat at 32 for a 0 and 224 for a 1. A lossy encode moves a cell by a
    few gray levels and a bit is 96 levels from the threshold, so
    :func:`read_frame_code` recovers *frame* exactly after several encodes. A flat
    gray level per frame does not: four color conversions darken one by as much as
    6 levels on x86_64 Linux, more than half the distance to a neighboring frame's
    level.

    Raises:
        ValueError: If *frame* does not fit in 12 bits.
    """
    if not 0 <= frame < 1 << _CODE_BITS:
        message = f"frame {frame} does not fit in a {_CODE_BITS}-bit frame code"
        raise ValueError(message)
    width, height = size
    image = np.empty((height, width, 3), np.uint8)
    for bit, (rows, columns, _, _) in enumerate(_code_cells(width, height)):
        one = frame >> (_CODE_BITS - 1 - bit) & 1
        image[rows, columns] = _CODE_ONE if one else _CODE_ZERO
    return image


def read_frame_code(image: npt.NDArray[np.uint8]) -> int:
    """Return the frame that :func:`paint_frame_code` spelled in decoded *image*.

    Each cell is read from the mean of its central half, away from the ringing a
    lossy encode leaves along the cell's edges. The grid is placed in proportion to
    *image*'s size, so a resized frame reads as well.

    Raises:
        ValueError: If a cell's mean is within 48 levels of the threshold, so a
            frame too damaged to read fails as unreadable rather than reading as
            another frame.
    """
    height, width = image.shape[:2]
    code = 0
    for bit, (_, _, rows, columns) in enumerate(_code_cells(width, height)):
        level = float(np.mean(image[rows, columns]))
        if abs(level - _CODE_THRESHOLD) < _CODE_MARGIN:
            message = (
                f"bit {bit} of the frame code reads {level:.1f}, within"
                f" {_CODE_MARGIN} of the threshold {_CODE_THRESHOLD}"
            )
            raise ValueError(message)
        code = code << 1 | int(level > _CODE_THRESHOLD)
    return code


def write_painted_entry(
    dataset: Dataset,
    sequence: str,
    clips: Sequence[tuple[int, float]],
    paint: Callable[[int], npt.NDArray[np.uint8]],
    *,
    size: tuple[int, int] = (64, 48),
) -> list[Path]:
    """Write and index *sequence*'s clips, each ``(frames, fps)``, in order.

    Frame ``i`` of the entry, counted across all its clips, is ``paint(i)``. The clips
    continue one another, and a decoded frame identifies its entry frame. The clips are
    ``clip0.mp4``, ``clip1.mp4``, ... under ``media_raw/<sequence>/``, returned in
    order.
    """
    directory = dataset.get_root("media_raw") / sequence
    paths: list[Path] = []
    offset = 0
    for position, (frames, fps) in enumerate(clips):

        def paint_clip(frame: int, first: int = offset) -> npt.NDArray[np.uint8]:
            return paint(first + frame)

        path = directory / f"clip{position}.mp4"
        write_h264_mp4(path, frames=frames, fps=fps, size=size, paint=paint_clip)
        paths.append(path)
        offset += frames
    index_media_sequence(dataset, sequence, [path.name for path in paths])
    return paths


def dot_image(size: tuple[int, int], center: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """Return a dark BGR image of *size* with one bright 3x3 dot centered on *center*.

    The dot peaks at its center: 255 there, 192 on its four sides and 128 at its
    corners, on a background of 16. The nine pixels of a flat dot tie, and a
    lossy encode then puts the brightest one anywhere in the dot. A peaked dot
    keeps its brightest pixel at *center*, measured at least 50 gray levels clear
    of every other pixel after an H.264 encode and an AV1 encode of it.

    Raises:
        ValueError: If the dot does not fit inside the image.
    """
    width, height = size
    x, y = center
    if not (1 <= x < width - 1 and 1 <= y < height - 1):
        message = f"a 3x3 dot centered on {center} does not fit in {width}x{height}"
        raise ValueError(message)
    image = np.full((height, width, 3), 16, np.uint8)
    image[y - 1 : y + 2, x - 1 : x + 2] = 128
    image[y - 1 : y + 2, x] = 192
    image[y, x - 1 : x + 2] = 192
    image[y, x] = 255
    return image


def write_mpeg4_mp4(
    path: Path,
    *,
    frames: int = 6,
    size: tuple[int, int] = (64, 48),
    shade: int | Literal["from-name"] = 0,
) -> None:
    """Write a small MPEG-4 clip through OpenCV, parent directories created.

    MPEG-4 rather than the AV1 the ``write_cfr_mp4`` fixture encodes, and the
    codec is load-bearing rather than incidental: the read-target gate refuses an
    ``"analysis"`` read whose verdict carries
    ``unverified_frame_correspondence``, which every codec outside the measured
    frame-exact set does. AV1 is inside that set and MPEG-4 is outside it, so a
    suite measuring what mosaic does with a clip it cannot read frame-exactly
    needs this one. A suite wanting a clip that passes the gate asks for the
    fixture.

    *shade* is the value every pixel of every frame carries. ``"from-name"``
    derives it from the file's name, which is what a caller needs when two clips
    must differ: two all-black clips are byte-identical and therefore share one
    ``video_uuid`` by design, so an ordering or composition assertion over them
    passes without measuring anything.

    Guards the ffmpeg toolchain even though the write itself is OpenCV's, because
    what these suites do with the clip -- probing it, indexing it -- shells out.
    Without the guard a missing binary surfaced as a bare ``FileNotFoundError``
    rather than a skip.
    """
    require_ffmpeg()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = _shade_for_name(path.name) if isinstance(shade, str) else shade
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, size)
    for _ in range(frames):
        writer.write(np.full((height, width, 3), value, np.uint8))
    writer.release()


def add_media_sequence(
    dataset: Dataset,
    sequence: str,
    *,
    videos: tuple[str, ...] = ("a.mp4", "b.mp4"),
    frames: int = 6,
) -> None:
    """Give *sequence* real videos under ``media_raw`` and index them.

    Driven through ``Dataset.write_media_index``, the assignment path the control
    plane uses, so the media index and the composition it projects are the ones
    production produces rather than a hand-built stand-in.

    Each video's content varies with its filename. Two all-black videos are
    byte-identical and therefore share one ``video_uuid`` by design, so a
    composition over them is genuinely unchanged by a reorder -- which would make
    an ordering assertion pass while testing nothing.

    Guards the toolchain itself, rather than leaving that to whichever fixture a
    caller happened to request: the write is in-process PyAV, but the indexing
    that follows shells out, so without ffmpeg this produced a bare
    ``FileNotFoundError`` in the three suites that call it directly.
    """
    require_ffmpeg()

    directory = dataset.get_root("media_raw") / sequence
    directory.mkdir(parents=True, exist_ok=True)
    for name in videos:
        write_h264_mp4(directory / name, frames=frames, shade=_shade_for_name(name))
    index_media_sequence(dataset, sequence, videos)


def index_media_sequence(
    dataset: Dataset, sequence: str, videos: Sequence[str]
) -> None:
    """Index the *videos* already written under ``media_raw/<sequence>/``, in order.

    It is the indexing half of :func:`add_media_sequence`, for a test that writes
    its own clips: at another rate, with painted frames, or replaced after a first
    index. A clip replaced since the last index is measured again.
    """
    from mosaic.core.pipeline.media_index import MediaIndexScope

    directory = dataset.get_root("media_raw") / sequence
    _ = dataset.write_media_index(
        [
            MediaIndexScope(
                directory=directory,
                group="",
                sequence=sequence,
                order_by_name={name: i for i, name in enumerate(videos)},
            )
        ],
        extensions=(".mp4",),
    )


def clean_facts_cells(
    video_uuid: str = "",
    *,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    frame_count: int = 100,
    rotation: int = 0,
) -> dict[str, object]:
    """A complete, verdict-clean set of media-facts cells for one index row.

    The tracker marker suites all need a media row a tracker will actually run
    against: probed dimensions, a container and pixel format that derive to a
    clean verdict, and -- when *video_uuid* is given -- the content identity that
    lets a marker tell a video replaced in place from one merely renamed.
    *width*, *height*, *fps*, *frame_count* and *rotation* describe the clip
    itself, defaulting to a fixed 640x480, 30 fps, 100-frame, upright shape.
    """
    facts: MediaFacts = store_facts(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        codec="h264",
        duration=frame_count / fps if fps else 0.0,
        video_uuid=video_uuid,
        identity_scheme="video/1" if video_uuid else "",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
        rotation_degrees=rotation,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


@dataclass
class MediaClip:
    """One media-index row to write.

    *sequence* and *filename* are the two values a row cannot do without.
    Every other field defaults to one uncalibrated clip: no group, no camera,
    first in its sequence's order, no recorded content identity, and the fixed
    dimensions ``clean_facts_cells`` assumes. A caller building several clips
    of one sequence, a multi-camera sequence, or facts that vary from row to
    row supplies the differing fields. A plain filename-keyed lookup cannot
    express two rows sharing one sequence name.
    """

    sequence: str = "sess"
    filename: str = ""
    group: str = ""
    camera: str = ""
    video_order: int = 0
    video_uuid: str = ""
    fps: float = 30.0
    width: int = 640
    height: int = 480
    rotation: int = 0
    frame_count: int = 100


def write_media_index(
    dataset: Dataset,
    rows: Sequence[str | MediaClip],
    *,
    filenames: dict[str, str] | None = None,
    uids: dict[str, str] | None = None,
) -> None:
    """Index one stub video per row, with full facts cells.

    A plain string in *rows* is shorthand for one stub video named after the
    sequence. *filenames* and *uids* override its name and content identity by
    sequence -- the same file under a new name keeps its uid, a replacement
    changes it. A :class:`MediaClip` names its filename and identity directly
    and consults neither dict.

    The bytes are a placeholder: every tracker marker suite fakes the tool, so
    nothing decodes them.
    """
    media_root = dataset.get_root(dataset.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    for entry in rows:
        clip = entry if isinstance(entry, MediaClip) else MediaClip(sequence=entry)
        filename = clip.filename or (filenames or {}).get(
            clip.sequence, f"{clip.sequence}.mp4"
        )
        video_uuid = clip.video_uuid or (uids or {}).get(clip.sequence, "")
        video = media_root / filename
        if not video.exists():
            _ = video.write_bytes(b"fake")
        written.append(
            {
                "name": filename,
                "group": clip.group,
                "sequence": clip.sequence,
                "group_safe": clip.group,
                "sequence_safe": clip.sequence,
                "camera": clip.camera,
                "abs_path": dataset.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": clip.width,
                "height": clip.height,
                "fps": clip.fps,
                "codec": "h264",
                "media_type": "video",
                "video_order": clip.video_order,
                **clean_facts_cells(
                    video_uuid,
                    width=clip.width,
                    height=clip.height,
                    fps=clip.fps,
                    frame_count=clip.frame_count,
                    rotation=clip.rotation,
                ),
            }
        )
    pd.DataFrame(written).to_csv(media_root / "index.csv", index=False)


def stub_media_dataset(base: Path, sequences: Sequence[str]) -> Dataset:
    """Return a saved dataset at *base* indexing one stub video per sequence.

    The videos come from `write_media_index`, so nothing can decode them. The
    dataset suits a test that fakes the tool reading its media.
    """
    dataset = make_dataset(base)
    write_media_index(dataset, list(sequences))
    return dataset


def index_session(
    dataset: Dataset,
    *filenames: str,
    widths: Mapping[str, int] | None = None,
    frame_count: int = 100,
    joined: bool = True,
) -> None:
    """Index *filenames* as the clips of one sequence, ``sess``, in the order given.

    Each clip's identity is ``uid-<filename>``. *widths* overrides a clip's frame
    width, for a case that needs clips which cannot be read as one video.
    *frame_count* is how many frames each clip holds, so a caller can state the
    media axis a table is compared against. *joined* also writes the joined export
    a multi-clip entry resolves to. ``False`` leaves it absent, which a run must
    refuse.
    """
    sizes = widths or {}
    write_media_index(
        dataset,
        [
            MediaClip(
                sequence="sess",
                filename=name,
                video_order=order,
                video_uuid=f"uid-{name}",
                width=sizes.get(name, 640),
                frame_count=frame_count,
            )
            for order, name in enumerate(filenames)
        ],
    )
    if len(filenames) > 1 and joined:
        _ = stub_join(dataset, [f"uid-{name}" for name in filenames])


def stub_join(dataset: Dataset, uids: Sequence[str]) -> Path:
    """Write a placeholder join of the clips whose identities are *uids*, in order.

    It is written at the address a tracker resolves for these clips in this order,
    under the default recipe, which is a current one. Every tracker suite fakes its
    tool, so nothing decodes the bytes. A test that reorders, adds or replaces a clip
    writes the new address, or sees the refusal of a missing join.
    """
    members = [
        MediaMember(camera="", video_order=order, uid=uid)
        for order, uid in enumerate(uids)
    ]
    path = joined_export_path(
        dataset,
        media_composition(members).digest,
        joined_recipe_hash(JoinedExportParams()),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"joined")
    return path


MakeStore = Callable[..., tuple[Path, list[np.ndarray]]]
"""The ``make_imgstore`` fixture's factory: writes a store, returns it and its frames."""

MOTIF_SYNC_UUID: Final = "f064059f9ea046429f227bc7addab1eb"
"""The synchronization uuid that :func:`store_dataset` gives a recording's cameras."""


def video_store_maker(
    chunk_frames: Sequence[int],
    *,
    fps: float = 30.0,
    size: tuple[int, int] = (64, 48),
    paint: Callable[[int], npt.NDArray[np.uint8]] | None = None,
) -> MakeStore:
    """Return a ``make_imgstore``-shaped factory that writes H.264 video stores.

    Each store has one chunk per count in *chunk_frames*, and store frame ``i`` is
    ``paint(i)``, by default the frame code of ``i`` at *size*. The layout is the
    one the imgstore package writes for ``avc1/mp4``: ``metadata.yaml``,
    ``NNNNNN.mp4`` chunks and ``NNNNNN.npz`` indexes. The chunks are encoded by
    ffmpeg (:func:`write_h264_mp4`), because the imgstore package encodes through
    OpenCV, whose Linux wheel cannot write H.264.

    The factory takes ``name``, ``parent`` and ``extra_metadata`` as
    ``make_imgstore`` does, and ignores the rest of that fixture's arguments.
    """
    width, height = size

    def paint_code(frame: int) -> npt.NDArray[np.uint8]:
        return paint_frame_code(frame, size)

    paint_frame = paint if paint is not None else paint_code

    def make(
        *,
        name: str = "store",
        parent: Path | None = None,
        extra_metadata: Mapping[str, object] | None = None,
        **_ignored: object,
    ) -> tuple[Path, list[np.ndarray]]:
        if parent is None:
            message = "a video store is written under a parent directory"
            raise ValueError(message)
        store = parent / name
        store.mkdir(parents=True)
        descriptor: dict[str, object] = {
            "chunksize": max(chunk_frames),
            "class": "VideoImgStore",
            "encoding": None,
            "extension": ".mp4",
            "format": "avc1/mp4",
            "imgdtype": "uint8",
            "imgshape": [height, width, 3],
            "uuid": hashlib.sha256(name.encode()).hexdigest()[:32],
            "version": 2,
        }
        metadata: dict[str, object] = {"__store": descriptor, **(extra_metadata or {})}
        _ = (store / "metadata.yaml").write_text(yaml.safe_dump(metadata))
        first = 0
        for chunk, count in enumerate(chunk_frames):

            def paint_chunk(frame: int, offset: int = first) -> npt.NDArray[np.uint8]:
                return paint_frame(offset + frame)

            write_h264_mp4(
                store / f"{chunk:06d}.mp4",
                frames=count,
                fps=fps,
                size=size,
                paint=paint_chunk,
            )
            numbers = np.arange(first, first + count, dtype=np.int64)
            np.savez(
                store / f"{chunk:06d}.npz",
                frame_number=numbers,
                frame_time=numbers / fps,
            )
            first += count
        return store, [paint_frame(frame) for frame in range(first)]

    return make


def store_dataset(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
    *,
    cameras: Sequence[str] = ("",),
    name: str = "rec",
    **store: object,
) -> Dataset:
    """Return a dataset holding one indexed imgstore per camera of one recording.

    The stores are written into ``media_raw/recordings`` and indexed from there,
    which gives the dataset an originals index separate from the derivative index
    under ``media``. A named camera's store is ``<name>.<camera>`` and carries the
    Motif metadata that groups a synced recording's cameras. ``""`` is one store
    with no camera.

    Args:
        tmp_path: Where the dataset is made, under ``dataset``.
        make_media_dataset: The ``make_media_dataset`` fixture.
        make_imgstore: The ``make_imgstore`` fixture.
        cameras: The camera of each store.
        name: The recording's name, which each store is named for.
        **store: Passed to *make_imgstore* for every store.
    """
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    search = ds.get_root("media_raw") / "recordings"
    search.mkdir(parents=True, exist_ok=True)
    for serial in cameras:
        extra = (
            {
                "camera_serial": serial,
                "synchronizationuuid": MOTIF_SYNC_UUID,
                "synchronization": "framenumber",
            }
            if serial
            else None
        )
        _ = make_imgstore(
            name=f"{name}.{serial}" if serial else name,
            parent=search,
            extra_metadata=extra,
            **store,
        )
    ds.index_media([search])
    return ds


def point_at_a_store(
    dataset: Dataset, sequence: str, store: Path, *, video_order: int | None = None
) -> Path:
    """Re-address *sequence*'s indexed media at an imgstore recording, *store*.

    A store is a directory holding a ``metadata.yaml`` naming ``__store``. This
    one describes an image-directory store with no chunks, which needs no chunk
    files and no imgstore package. Its chunks are not video, so a tool that reads
    files outside mosaic's process reads it only through its export.
    *video_order* names the one clip to re-address, and ``None`` every clip of the
    sequence. Returns *store*.
    """
    store.mkdir(parents=True, exist_ok=True)
    _ = (store / "metadata.yaml").write_text(
        "__store: {class: DirectoryImgStore, format: npy, imgshape: [48, 64, 3]}\n"
    )
    index_path = dataset.get_root(dataset.resolve_media_root()) / "index.csv"
    table = pd.read_csv(index_path)
    is_entry = table["sequence"] == sequence
    if video_order is not None:
        is_entry &= table["video_order"] == video_order
    table.loc[is_entry, "abs_path"] = dataset.relative_to_root(store)
    table.loc[is_entry, "media_type"] = "imgstore"
    table.to_csv(index_path, index=False)
    return store


def add_transcode_derivative(
    dataset: Dataset,
    sequence: str,
    *,
    target: Target = "playback",
    clip: str | None = None,
    encode: Callable[[Path, Path], MediaFacts] | None = None,
) -> Path:
    """Register a derivative for one of *sequence*'s videos, as the transcode op does.

    The derivative is named under the op's scheme, with the recipe computed
    through the op's own function rather than hard-coded: the recipe folds
    environment-driven thresholds, so a literal would pin the suite to one
    machine. Both links are written in the order the op writes them, the
    back-link row into the ``media`` index and then the forward-link cell onto
    the original. The forward link always goes through the op's writer, and the
    back-link row does when the derivative has facts.

    Args:
        dataset: The dataset holding the original.
        sequence: The sequence whose video gets a derivative.
        target: Which derivative. ``playback`` by default, matching the scenario
            this exists for: a proxy made so a browser can play the video, which
            the tracker, frame extraction, crops and every feature ignore.
        clip: Which of the sequence's originals, by name. ``None`` is the first.
        encode: Writes a source's derivative to a path and returns the facts
            measured off it. ``None`` writes stub bytes instead, for a test that
            reads only the derivative's name. A stub has no facts, so its
            back-link row names it and nothing more, and a read routed to it
            raises.

    Returns:
        The derivative's path.
    """
    from mosaic_media.transcode import ANALYSIS_ENCODING, PLAYBACK_ENCODING

    from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
    from mosaic.core.pipeline.media_index import (
        frame_from_rows,
        read_media_index,
        write_media_index_rows,
    )
    from mosaic.core.pipeline.transcode import (
        TRANSCODE_KIND_DIRECTORY,
        TranscodeParams,
        relative_to_anchor,
        set_back_link,
        set_forward_link,
        transcode_recipe_hash,
    )
    from mosaic.media_probe_config import media_thresholds

    raw_index = dataset.get_root("media_raw") / "index.csv"
    matches = [
        dict(row)
        for row in read_media_index(raw_index)
        if row.get("sequence") == sequence and clip in (None, row.get("name"))
    ]
    if not matches:
        raise AssertionError(f"no media_raw row for sequence {sequence!r} ({clip})")
    original = matches[0]
    video_uuid = str(original["video_uuid"])
    source = dataset.resolve_path(str(original["abs_path"]))

    recipe = transcode_recipe_hash(
        TranscodeParams(target=target),
        ANALYSIS_ENCODING if target == "analysis" else PLAYBACK_ENCODING,
        CHROME_149,
        media_thresholds(),
    )
    media_root = dataset.get_root("media")
    transcode_root = media_root / TRANSCODE_KIND_DIRECTORY
    transcode_root.mkdir(parents=True, exist_ok=True)
    derivative = transcode_root / f"{video_uuid}.{recipe}.{target}.mp4"

    if encode is not None:
        facts = encode(source, derivative)
        set_back_link(
            dataset,
            str(original.get("group", "")),
            sequence,
            source,
            derivative,
            facts,
            derive(facts, CHROME_149, media_thresholds()),
            int(str(original.get("video_order", "") or 0)),
            source_video_uuid=video_uuid,
            recipe_hash=recipe,
            encoder=facts.codec_name,
        )
    else:
        _ = derivative.write_bytes(b"stub")
        media_index = media_root / "index.csv"
        rows: list[dict[str, object]] = [
            dict(row) for row in read_media_index(media_index)
        ]
        row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
        row.update(
            {
                "name": derivative.name,
                "group": original.get("group", ""),
                "sequence": sequence,
                "abs_path": dataset.relative_to_root(str(derivative)),
                "source_video_uuid": video_uuid,
                "recipe_hash": recipe,
            }
        )
        rows.append(row)
        write_media_index_rows(media_index, frame_from_rows(rows))

    set_forward_link(
        dataset, source, video_uuid, relative_to_anchor(derivative, media_root), target
    )
    return derivative


def clip_facts(
    *,
    fps: float = 30.0,
    frame_count: int = 300,
    width: int = 64,
    height: int = 48,
    rotation: int = 0,
    duration: float | None = None,
    start_time: float = 0.0,
    video_uuid: str = "uuid",
) -> MediaFacts:
    """One clip's probed facts, carrying what a timeline or a rate check reads.

    Built rather than probed, so a test can state a 31 fps clip of three hundred
    frames without encoding one.
    """
    return MediaFacts(
        container="mp4",
        codec_name="h264",
        pixel_format="yuv420p",
        color_range="",
        color_primaries="",
        color_transfer="",
        width=width,
        height=height,
        rotation_degrees=rotation,
        square_pixels=True,
        progressive=True,
        has_audio=False,
        video_stream_count=1,
        duration=(frame_count / fps if fps else 0.0) if duration is None else duration,
        fps=fps,
        frame_count=frame_count,
        start_time=start_time,
        constant_frame_rate=True,
        max_instantaneous_fps=None,
        declared_duration=frame_count / fps if fps else 0.0,
        declared_fps=fps,
        declared_frame_count=frame_count,
        moov_at_start=True,
        max_keyframe_interval_frames=1,
        max_gop_bytes=1,
        discard_flagged_packets=0,
        leading_non_keyframe_frames=0,
        coded_reordering_depth=0,
        max_timestamp_gap_frame_periods=1.0,
        timing_source="presentation",
        video_uuid=video_uuid,
        content_digest="digest",
        identity_scheme="1",
        prober_version="test",
    )
