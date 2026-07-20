import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, derive, probe_media
from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import FACTS_COLUMNS, facts_to_row, row_to_facts


def _cfr_mp4(path: Path, n: int = 10) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(n):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()


def _make_dataset(tmp_path: Path) -> Dataset:
    for sub in ("media", "tracks", "frames"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
            "frames": str(tmp_path / "frames"),
        },
    )


def test_facts_row_has_all_columns(tmp_path: Path) -> None:
    mp4 = tmp_path / "v.mp4"
    _cfr_mp4(mp4)
    facts = probe_media(mp4)
    verdict = derive(facts, CHROME_149, DEFAULT_THRESHOLDS)
    row = facts_to_row(facts, verdict)
    assert set(FACTS_COLUMNS) <= set(row)
    assert row["frame_count"] == 10
    assert row_to_facts(row) == facts


def test_row_to_facts_preserves_rotation(tmp_path: Path) -> None:
    mp4 = tmp_path / "v.mp4"
    _cfr_mp4(mp4, 4)
    rotated = replace(probe_media(mp4), rotation_degrees=90)  # coded 64x48
    row = facts_to_row(rotated, derive(rotated, CHROME_149, DEFAULT_THRESHOLDS))
    got = row_to_facts(row)
    assert (got.width, got.height, got.rotation_degrees) == (64, 48, 90)


def test_index_media_facts_round_trip_through_csv(tmp_path: Path) -> None:
    """The facts columns survive the full persistence path, not just the in-memory dict.

    Runs Dataset.index_media over a scratch dir holding one real mp4, reads the
    resulting index.csv back with pandas.read_csv (the same untyped path the
    rest of mosaic uses), and reconstructs MediaFacts from the persisted row.
    """
    ds = _make_dataset(tmp_path)
    search = tmp_path / "raw"
    search.mkdir()
    mp4 = search / "clip.mp4"
    _cfr_mp4(mp4, 10)
    expected_facts = probe_media(mp4)

    out_csv = ds.index_media([search], extensions=(".mp4",))
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert set(FACTS_COLUMNS) <= set(df.columns)
    assert len(df) == 1

    row = df.iloc[0]
    assert int(row["frame_count"]) == expected_facts.frame_count

    got_facts = row_to_facts({str(key): value for key, value in row.items()})
    assert got_facts.frame_count == expected_facts.frame_count
    assert got_facts.width == expected_facts.width
    assert got_facts.height == expected_facts.height


def test_reindex_preserves_derivative_links(tmp_path: Path) -> None:
    """A media reindex re-measures every column but keeps the transcode links.

    ``index_media`` rebuilds the originals index wholesale; the per-target
    derivative links (a transcode decision, not a measurement) must survive it,
    while a link whose derivative file was deleted is dropped.
    """
    for sub in ("media_raw", "media", "tracks"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
        },
    )
    search = tmp_path / "raw"
    search.mkdir()
    _cfr_mp4(search / "clip.mp4", 10)

    ds.index_media([search], extensions=(".mp4",))
    raw_index = ds.get_root("media_raw") / "index.csv"

    # Stand in for the transcode job: real per-target derivatives + forward links.
    analysis_deriv = ds.get_root("media") / "clip.analysis.mp4"
    playback_deriv = ds.get_root("media") / "clip.playback.mp4"
    _cfr_mp4(analysis_deriv, 10)
    _cfr_mp4(playback_deriv, 10)
    df = pd.read_csv(raw_index)
    for column in ("analysis_derivative_path", "playback_derivative_path"):
        df[column] = df[column].astype("object")
    df.loc[0, "analysis_derivative_path"] = "clip.analysis.mp4"
    df.loc[0, "playback_derivative_path"] = "clip.playback.mp4"
    df.to_csv(raw_index, index=False)

    # Reindexing the same dir carries both links forward.
    ds.index_media([search], extensions=(".mp4",))
    df2 = pd.read_csv(raw_index)
    assert str(df2.loc[0, "analysis_derivative_path"]) == "clip.analysis.mp4"
    assert str(df2.loc[0, "playback_derivative_path"]) == "clip.playback.mp4"

    # Deleting the analysis derivative drops its dangling link on reindex; the
    # still-present playback link survives.
    analysis_deriv.unlink()
    ds.index_media([search], extensions=(".mp4",))
    df3 = pd.read_csv(raw_index)
    assert str(df3.loc[0, "analysis_derivative_path"]) in ("", "nan")
    assert str(df3.loc[0, "playback_derivative_path"]) == "clip.playback.mp4"


def test_row_to_facts_treats_missing_or_empty_facts_as_no_stored_facts() -> None:
    """A row with no media_facts key (or an empty/NaN cell) has no stored facts
    -- row_to_facts must raise so the caller's probe-fallback contract holds,
    instead of silently returning wrong facts.
    """
    with pytest.raises(KeyError):
        row_to_facts({"frame_count": 10})  # column missing entirely

    with pytest.raises(KeyError):
        row_to_facts({"media_facts": ""})  # empty string cell


def test_row_to_facts_rejects_nan_media_facts_from_csv_round_trip(
    tmp_path: Path,
) -> None:
    """pandas turns an empty CSV cell into a NaN float, not an empty string.

    row_to_facts's `not isinstance(payload, str)` guard must catch this case
    too, so a row with an empty cell read back through pandas still raises rather
    than being mistaken for a string payload.
    """
    csv_path = tmp_path / "no_facts.csv"
    csv_path.write_text("frame_count,media_facts\n10,\n")
    df = pd.read_csv(csv_path)
    payload = df.iloc[0]["media_facts"]
    assert isinstance(payload, float) and math.isnan(payload)

    with pytest.raises(KeyError):
        row_to_facts({str(key): value for key, value in df.iloc[0].items()})


def test_media_index_columns_public_and_reexported():
    from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS as from_media
    from mosaic.core.dataset import MEDIA_INDEX_COLUMNS as from_dataset

    assert from_media is from_dataset
    assert "analysis_derivative_path" in from_media
