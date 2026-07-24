import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import (
    derivative_cell,
    derivative_path_for_target,
    store_facts,
)


def test_derivative_cell_reads_the_target_column_or_empty() -> None:
    row = {
        "analysis_derivative_path": "d/a.analysis.mp4",
        "playback_derivative_path": "",
    }
    assert derivative_cell(row, "analysis") == "d/a.analysis.mp4"
    assert derivative_cell(row, "playback") == ""
    assert derivative_cell({"analysis_derivative_path": "nan"}, "analysis") == ""
    assert derivative_cell({"analysis_derivative_path": float("nan")}, "analysis") == ""


def test_derivative_path_anchors_under_the_media_root() -> None:
    media_root = Path("/data/media")
    row = {"analysis_derivative_path": "d/a.analysis.mp4"}
    assert derivative_path_for_target(row, "analysis", media_root) == (
        media_root / "d/a.analysis.mp4"
    )


def test_derivative_path_is_none_when_unregistered() -> None:
    media_root = Path("/data/media")
    assert derivative_path_for_target({}, "analysis", media_root) is None
    assert (
        derivative_path_for_target(
            {"analysis_derivative_path": "nan"}, "analysis", media_root
        )
        is None
    )


def _facts_cell(frame_count: int) -> str:
    facts = store_facts(
        width=64,
        height=48,
        fps=30.0,
        frame_count=frame_count,
        codec="h264",
        duration=frame_count / 30.0,
    )
    return json.dumps(dataclasses.asdict(facts))


def test_pass_two_matches_source_uuid_never_the_sibling(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """Routing resolves the derivative by the source's ``video_uuid``.

    Both siblings share ``source_video_uuid = "U"``; only the basename guard
    keeps the playback row (ordered first) from crossing into the analysis
    lookup. Their abs_path cells resolve away from the routed file, so the exact
    pass misses and the uuid form is what returns the row; empty ``source_path``
    cells make the path form unable to match, so the result can only come from
    the uuid form.

    Driven through :meth:`Dataset.route_media_row`, the public entry point that
    performs the lookup: a row marked ``analysis_transcode="required"`` resolves
    to its registered analysis derivative and carries that derivative's stored
    facts.
    """
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)

    original = base / "media_raw" / "entry.mp4"
    original.touch()
    routed = base / "media" / "entry.analysis.mp4"
    routed.touch()

    synced = base / "synced"
    synced.mkdir()
    analysis_abs = synced / "entry.analysis.mp4"
    analysis_abs.touch()
    playback_abs = synced / "entry.playback.mp4"
    playback_abs.touch()

    # Playback row first, so a missing basename guard would return it (999).
    rows: list[dict[str, object]] = [
        {
            "abs_path": str(playback_abs),
            "source_path": "",
            "source_video_uuid": "U",
            "media_facts": _facts_cell(999),
        },
        {
            "abs_path": str(analysis_abs),
            "source_path": "",
            "source_video_uuid": "U",
            "media_facts": _facts_cell(123),
        },
    ]
    derivative_df = pd.DataFrame(rows)
    entry = pd.Series(
        {
            "abs_path": str(original),
            "video_uuid": "U",
            "analysis_transcode": "required",
            "analysis_derivative_path": "entry.analysis.mp4",
        }
    )

    resolved, facts = ds.route_media_row("g1", "entry", entry, True, derivative_df)

    assert resolved == routed
    assert facts is not None
    assert facts.frame_count == 123
    assert facts.frame_count != 999
