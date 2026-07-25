"""Verdict-based media routing in :meth:`Dataset.resolve_media`."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from mosaic_media import MediaProbeError, probe_media

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import to_safe_name
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS


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


def _write_mp4(path: Path, nframes: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    for _ in range(nframes):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def _facts_json(path: Path) -> str:
    return json.dumps(dataclasses.asdict(probe_media(path)))


def _row(
    *,
    group: str,
    sequence: str,
    abs_path: Path,
    analysis_transcode: str = "",
    analysis_derivative_path: str = "",
    playback_derivative_path: str = "",
    source_path: str = "",
) -> dict[str, object]:
    """Build a full media-index row for *abs_path* with a real facts JSON cell."""
    return {
        "name": abs_path.name,
        "group": group,
        "sequence": sequence,
        "group_safe": to_safe_name(group) if group else "",
        "sequence_safe": to_safe_name(sequence),
        "abs_path": str(abs_path),
        "size_bytes": abs_path.stat().st_size,
        "mtime_iso": "",
        "width": 64,
        "height": 48,
        "fps": 30.0,
        "codec": "h264",
        "media_type": "video",
        "frame_count": probe_media(abs_path).frame_count,
        "analysis_transcode": analysis_transcode,
        "stream_transcode": "",
        "analysis_derivative_path": analysis_derivative_path,
        "playback_derivative_path": playback_derivative_path,
        "source_path": source_path,
        "media_facts": _facts_json(abs_path),
        "video_order": 0,
    }


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows, columns=MEDIA_INDEX_COLUMNS).to_csv(path, index=False)


def test_clean_row_returns_original_with_stored_facts(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clean.mp4"
    _write_mp4(original, nframes=6)
    # The stored cell deliberately disagrees with the file (a fresh probe of
    # this clip returns 6), so a returned count of 999 can only have come from
    # the stored cell, never from a re-probe.
    stored_facts = dataclasses.replace(probe_media(original), frame_count=999)
    row = _row(group="g1", sequence="clean", abs_path=original)
    row["media_facts"] = json.dumps(dataclasses.asdict(stored_facts))
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    resolved = ds.resolve_media("g1", "clean")
    assert [p.resolve() for p in resolved.paths] == [original.resolve()]
    assert resolved.facts[0].frame_count == 999


def test_row_with_an_unreconstructable_facts_cell_raises_the_reprobe_remedy(
    tmp_path: Path,
):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clean.mp4"
    _write_mp4(original, nframes=6)
    row = _row(group="g1", sequence="clean", abs_path=original)
    # A media_facts cell written before the identity fields existed: it lacks
    # the keys MediaFacts now requires, so it cannot be reconstructed.
    row["media_facts"] = json.dumps({"width": 64, "height": 48})
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    with pytest.raises(
        MediaProbeError, match="mosaic reprobe-media --apply"
    ) as excinfo:
        ds.resolve_media("g1", "clean")
    assert str(original) in str(excinfo.value)


def test_row_with_no_facts_cell_raises_the_reprobe_remedy(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "clean.mp4"
    _write_mp4(original, nframes=6)
    row = _row(group="g1", sequence="clean", abs_path=original)
    row["media_facts"] = ""
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    with pytest.raises(MediaProbeError, match="mosaic reprobe-media --apply"):
        ds.resolve_media("g1", "clean")


def test_a_required_row_with_no_facts_still_reports_the_transcode_remedy(
    tmp_path: Path,
):
    """A required-and-unlinked row that also carries no facts reports the
    transcode remedy, not the reprobe remedy: the file the caller would open
    is the derivative, so the derivative's absence is the fault to report."""
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "orphan.mp4"
    _write_mp4(original, nframes=6)
    row = _row(
        group="g1",
        sequence="orphan",
        abs_path=original,
        analysis_transcode="required",
        analysis_derivative_path="",
    )
    row["media_facts"] = ""
    _write_index(tmp_path / "media_raw" / "index.csv", [row])

    with pytest.raises(
        MediaProbeError, match="requires an analysis transcode"
    ) as excinfo:
        ds.resolve_media("g1", "orphan")
    assert "reprobe-media" not in str(excinfo.value)


def test_required_row_routes_to_derivative_with_derivative_facts(tmp_path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "needs.mp4"
    _write_mp4(original, nframes=6)
    derivative = tmp_path / "media" / "g1__needs.analysis.mp4"
    _write_mp4(derivative, nframes=10)

    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(
                group="g1",
                sequence="needs",
                abs_path=original,
                analysis_transcode="required",
                analysis_derivative_path="g1__needs.analysis.mp4",
            )
        ],
    )
    _write_index(
        tmp_path / "media" / "index.csv",
        [
            _row(
                group="g1",
                sequence="needs",
                abs_path=derivative,
                source_path="needs.mp4",
            )
        ],
    )

    resolved = ds.resolve_media("g1", "needs")
    assert [p.resolve() for p in resolved.paths] == [derivative.resolve()]
    # Derivative facts, not the original's (distinct frame counts).
    assert resolved.facts[0].frame_count == probe_media(derivative).frame_count
    assert resolved.facts[0].frame_count != probe_media(original).frame_count


def test_required_row_without_derivative_raises(tmp_path):
    ds = _make_dataset(tmp_path)
    original = tmp_path / "media_raw" / "orphan.mp4"
    _write_mp4(original, nframes=6)
    _write_index(
        tmp_path / "media_raw" / "index.csv",
        [
            _row(
                group="g1",
                sequence="orphan",
                abs_path=original,
                analysis_transcode="required",
                analysis_derivative_path="",
            )
        ],
    )

    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        ds.resolve_media("g1", "orphan")


def test_legacy_media_only_required_row_raises(tmp_path):
    """A legacy dataset with only a ``media`` root (no ``media_raw`` split).

    A required row has no silent-degrade arm: routing must fail loud, telling
    the user to adopt the media_raw/media split and transcode, rather than
    opening the defective original.
    """
    (tmp_path / "media").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tracks").mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
        },
    )
    assert ds.resolve_media_root() == "media"
    original = tmp_path / "media" / "legacy.mp4"
    _write_mp4(original, nframes=6)
    _write_index(
        tmp_path / "media" / "index.csv",
        [
            _row(
                group="",
                sequence="legacy",
                abs_path=original,
                analysis_transcode="required",
            )
        ],
    )

    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        ds.resolve_media("", "legacy")
