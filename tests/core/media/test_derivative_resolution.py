import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import MediaProbeError

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
        video_uuid="",
        identity_scheme="",
    )
    return json.dumps(dataclasses.asdict(facts))


def test_the_uuid_fallback_never_crosses_into_the_sibling_target(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The uuid fallback's basename guard, pinned on a hand-edited index.

    **A synthetic guard, not a production path.** No writer produces a
    derivative row whose ``abs_path`` fails to resolve to the routed file:
    ``_set_back_link`` builds every row through ``_derivative_row``, which
    writes ``abs_path`` from ``output_path``, and the forward link the router
    resolves through is written from that same path -- so the exact pass always
    matches what mosaic wrote, and nothing of its making reaches the fallback.
    The rows below are seeded into the shape only an index edited outside mosaic
    could hold: ``abs_path`` cells resolving away from the routed file.

    What the guard is for: both siblings share ``source_video_uuid = "U"``, so
    the uuid match alone cannot tell the analysis derivative from the playback
    one, and only the basename comparison keeps the playback row -- ordered
    first, so an absent guard would return it -- out of the analysis lookup.

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


def test_a_derivative_reachable_only_by_source_path_raises(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """An original with no ``video_uuid`` no longer resolves by ``source_path``.

    The routed derivative's ``abs_path`` does not resolve to the file the caller
    opens, and the original carries no uuid, so the only prior match was the
    source-path pass -- now deleted. Its absence is a raise, not a silent
    resolution onto a path-matched row.

    Driven through :meth:`Dataset.route_media_row`, the public entry point: a row
    marked ``analysis_transcode="required"`` whose derivative facts cannot be
    found raises :class:`~mosaic_media.MediaProbeError`.
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

    # The derivative row is reachable only through source_path: its abs_path
    # resolves away from the routed file and it carries no source_video_uuid, so
    # the exact pass and the uuid pass both miss. Under the old code source_path
    # -- root-relative to media_raw -- resolved to the original and returned the
    # facts; that pass is gone.
    rows: list[dict[str, object]] = [
        {
            "abs_path": str(analysis_abs),
            "source_path": "entry.mp4",
            "source_video_uuid": "",
            "media_facts": _facts_cell(123),
        },
    ]
    derivative_df = pd.DataFrame(rows)
    entry = pd.Series(
        {
            "abs_path": str(original),
            "video_uuid": "",
            "analysis_transcode": "required",
            "analysis_derivative_path": "entry.analysis.mp4",
        }
    )

    with pytest.raises(MediaProbeError, match="has no matching row with stored facts"):
        _ = ds.route_media_row("g1", "entry", entry, True, derivative_df)
