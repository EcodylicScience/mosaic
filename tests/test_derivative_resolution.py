import dataclasses
import json
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


def _make_dataset(tmp_path: Path) -> Dataset:
    for sub in ("media_raw", "media", "tracks"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
        },
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


def test_pass_two_matches_source_uuid_never_the_sibling(tmp_path: Path) -> None:
    """Pass 2 resolves the derivative by the source's ``video_uuid``.

    Both siblings share ``source_video_uuid = "U"``; only the basename guard
    keeps the playback row (ordered first) from crossing into the analysis
    lookup. Their abs_path cells resolve away from the routed file, so pass 1
    misses and the uuid form of pass 2 is what returns the row; empty
    ``source_path`` cells make the path form of pass 2 unable to match, so the
    result can only come from the uuid form.
    """
    ds = _make_dataset(tmp_path)

    original = tmp_path / "media_raw" / "entry.mp4"
    original.touch()
    routed = tmp_path / "media" / "entry.analysis.mp4"
    routed.touch()

    synced = tmp_path / "synced"
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

    facts = ds._derivative_facts(
        "g1", "entry", routed, str(original), "U", derivative_df
    )

    assert facts.frame_count == 123
    assert facts.frame_count != 999
