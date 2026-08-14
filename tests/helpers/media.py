"""Building media files, media-index rows, and transcode derivatives.

The row builders exist because a media row has far more columns than any one
test cares about, and a row missing the probed facts is not a row the toolkit
produces -- so a test built on one measures a shape that cannot occur.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive
from mosaic_media.io.writer import FFmpegVideoWriter
from mosaic_media.transcode import Target

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from tests.helpers.environment import require_ffmpeg


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
        shade = sum(name.encode()) % 200 + 20
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


def clean_facts_cells(video_uuid: str = "") -> dict[str, object]:
    """A complete, verdict-clean set of media-facts cells for one index row.

    The tracker marker suites all need a media row a tracker will actually run
    against: probed dimensions, a container and pixel format that derive to a
    clean verdict, and -- when *video_uuid* is given -- the content identity that
    lets a marker tell a video replaced in place from one merely renamed.
    """
    facts: MediaFacts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid=video_uuid,
        identity_scheme="video/1" if video_uuid else "",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def write_media_index(
    dataset: Dataset,
    sequences: list[str],
    *,
    filenames: dict[str, str] | None = None,
    uids: dict[str, str] | None = None,
) -> None:
    """Index one stub video per sequence, with full facts cells.

    The bytes are a placeholder: every tracker marker suite fakes the tool, so
    nothing decodes them. *filenames* and *uids* are what the rename-versus-replace
    scenarios vary -- the same file under a new name keeps its uid, a replacement
    changes it.
    """
    media_root = dataset.get_root(dataset.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seq in sequences:
        filename = (filenames or {}).get(seq, f"{seq}.mp4")
        video = media_root / filename
        if not video.exists():
            _ = video.write_bytes(b"fake")
        rows.append(
            {
                "name": filename,
                "group": "",
                "sequence": seq,
                "group_safe": "",
                "sequence_safe": seq,
                "abs_path": dataset.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": 0,
                **clean_facts_cells((uids or {}).get(seq, "")),
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)


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
        TranscodeParams(entry=("", sequence), target=target),
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
