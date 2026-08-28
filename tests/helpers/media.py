"""Building media files, media-index rows, and transcode derivatives.

The row builders exist because a media row has far more columns than any one
test cares about, and a row missing the probed facts is not a row the toolkit
produces -- so a test built on one measures a shape that cannot occur.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive
from mosaic_media.io.writer import FFmpegVideoWriter
from mosaic_media.transcode import Target

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from tests.helpers.environment import require_ffmpeg


def _shade_for_name(name: str) -> int:
    """The grey level *name* stands for, so two clips named apart look apart."""
    return sum(name.encode()) % 200 + 20


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
    from mosaic.core.pipeline.media_index import MediaIndexScope

    require_ffmpeg()

    directory = dataset.get_root("media_raw") / sequence
    directory.mkdir(parents=True, exist_ok=True)
    for name in videos:
        shade = _shade_for_name(name)
        with FFmpegVideoWriter(
            directory / name, width=64, height=48, fps=30.0
        ) as writer:
            for _ in range(frames):
                writer.write(np.full((48, 64, 3), shade, np.uint8))

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


def add_transcode_derivative(
    dataset: Dataset, sequence: str, *, target: Target = "playback"
) -> Path:
    """Register a derivative for *sequence*'s first video, without encoding one.

    A stub, because nothing being tested reads a derivative's bytes -- what is
    read is its *name*, so it is written under the scheme the transcode op uses
    and the recipe is computed through the op's own function rather than
    hard-coded (the recipe folds environment-driven thresholds, so a literal
    would pin the suite to one machine).

    Both links are written, in the order the op writes them: the back-link row
    into the ``media`` index, then the forward-link cell onto the original.

    ``playback`` by default, matching the scenario this exists for -- a proxy
    made so a browser can play the video, which the tracker, frame extraction,
    crops and every feature ignore.
    """
    from mosaic_media import CHROME_149
    from mosaic_media.transcode import ANALYSIS_ENCODING, PLAYBACK_ENCODING

    from mosaic.core.media.facts_columns import (
        MEDIA_INDEX_COLUMNS,
        derivative_column_for_target,
    )
    from mosaic.core.pipeline.media_index import (
        frame_from_rows,
        read_media_index,
        write_media_index_rows,
    )
    from mosaic.core.pipeline.transcode import (
        TRANSCODE_KIND_DIRECTORY,
        TranscodeParams,
        transcode_recipe_hash,
    )
    from mosaic.media_probe_config import media_thresholds

    raw_index = dataset.get_root("media_raw") / "index.csv"
    originals = [dict(row) for row in read_media_index(raw_index)]
    matches = [row for row in originals if row.get("sequence") == sequence]
    if not matches:
        raise AssertionError(f"no media_raw row for sequence {sequence!r}")
    original = matches[0]
    video_uuid = original["video_uuid"]

    recipe = transcode_recipe_hash(
        TranscodeParams(target=target),
        ANALYSIS_ENCODING if target == "analysis" else PLAYBACK_ENCODING,
        CHROME_149,
        media_thresholds(),
    )
    transcode_root = dataset.get_root("media") / TRANSCODE_KIND_DIRECTORY
    transcode_root.mkdir(parents=True, exist_ok=True)
    derivative = transcode_root / f"{video_uuid}.{recipe}.{target}.mp4"
    _ = derivative.write_bytes(b"stub")

    media_index = dataset.get_root("media") / "index.csv"
    rows = [dict(row) for row in read_media_index(media_index)]
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

    column = derivative_column_for_target(target)
    for candidate in originals:
        if candidate.get("video_uuid") == video_uuid:
            candidate[column] = f"{TRANSCODE_KIND_DIRECTORY}/{derivative.name}"
    write_media_index_rows(raw_index, frame_from_rows(list(originals)))
    return derivative
