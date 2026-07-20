"""End-to-end tests for the multi-video transcode job and derivative routing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import (
    CHROME_149,
    DEFAULT_THRESHOLDS,
    MediaProbeError,
    derive,
    probe_media,
)

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import row_to_facts
from mosaic.core.pipeline.transcode import TranscodeParams, run_transcode_op


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


def _write_analysis_required_mp4(path: Path) -> None:
    """Write a variable-frame-rate mp4 that ``derive`` marks analysis-required.

    Height/width are >= 64 so the analysis (SVT-AV1) encoder accepts the source.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=128x128:rate=30",
        "-vf",
        "setpts=N/(30+8*sin(N))/TB",
        "-fps_mode",
        "vfr",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not path.exists():
        pytest.skip("this environment could not produce an mp4 fixture with ffmpeg")


def _require_analysis_required(path: Path) -> None:
    facts = probe_media(path)
    verdict = derive(facts, CHROME_149, DEFAULT_THRESHOLDS)
    if verdict.analysis_transcode != "required":
        pytest.skip(
            "this environment's ffmpeg did not produce an analysis-required fixture"
        )


def _indexed_entry(ds: Dataset, source_dir: Path) -> tuple[str, str]:
    ds.index_media([source_dir])
    df = pd.read_csv(ds.get_root("media_raw") / "index.csv")
    row = df.iloc[0]
    group = "" if pd.isna(row["group"]) else str(row["group"])
    return group, str(row["sequence"])


def test_transcode_op_writes_derivative_and_links(tmp_path):
    ds = _make_dataset(tmp_path)
    source_dir = tmp_path / "raw_src"
    source_dir.mkdir()
    original = source_dir / "vfr.mp4"
    _write_analysis_required_mp4(original)
    _require_analysis_required(original)

    group, sequence = _indexed_entry(ds, source_dir)

    # Before transcoding, the required row has no derivative -> routing errors.
    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        ds.resolve_media(group, sequence)

    summary = run_transcode_op(
        ds, TranscodeParams(entry=(group, sequence), target="analysis")
    )
    assert summary  # a derivative path was written

    # A derivative exists under media/ and re-probes clean.
    derivative_files = list(ds.get_root("media").glob("*.mp4"))
    assert len(derivative_files) == 1
    derivative = derivative_files[0]
    derivative_verdict = derive(
        probe_media(derivative), CHROME_149, DEFAULT_THRESHOLDS
    )
    assert derivative_verdict.analysis_transcode is None

    # Forward link: the original's media_raw row now names the analysis
    # derivative, and only that per-target column is set.
    raw_df = pd.read_csv(ds.get_root("media_raw") / "index.csv")
    assert str(raw_df.iloc[0]["analysis_derivative_path"]).strip() not in ("", "nan")
    assert str(raw_df.iloc[0]["playback_derivative_path"]).strip() in ("", "nan")

    # Back link: the media index row carries source_path and reconstructable facts.
    media_df = pd.read_csv(ds.get_root("media") / "index.csv")
    assert len(media_df) == 1
    deriv_row = {str(k): v for k, v in media_df.iloc[0].items()}
    assert str(deriv_row["source_path"]).strip() not in ("", "nan")
    reconstructed = row_to_facts(deriv_row)
    assert reconstructed.frame_count == probe_media(derivative).frame_count

    # Routing end-to-end: resolve_media now returns the derivative + its facts.
    resolved = ds.resolve_media(group, sequence)
    assert resolved.paths[0].resolve() == derivative.resolve()
    assert resolved.facts is not None
    routed_verdict = derive(resolved.facts[0], CHROME_149, DEFAULT_THRESHOLDS)
    assert routed_verdict.analysis_transcode is None
    assert resolved.facts[0].frame_count == probe_media(derivative).frame_count


def test_analysis_facts_not_crossed_when_playback_transcoded_first(tmp_path):
    ds = _make_dataset(tmp_path)
    source_dir = tmp_path / "raw_src"
    source_dir.mkdir()
    original = source_dir / "vfr.mp4"
    _write_analysis_required_mp4(original)
    _require_analysis_required(original)

    group, sequence = _indexed_entry(ds, source_dir)

    # Transcode PLAYBACK first, then ANALYSIS. The playback derivative row (which
    # shares source_path with the analysis row) then precedes the analysis row in
    # the media index, so a source_path-first lookup would return playback facts
    # for the analysis route.
    run_transcode_op(ds, TranscodeParams(entry=(group, sequence), target="playback"))
    run_transcode_op(ds, TranscodeParams(entry=(group, sequence), target="analysis"))

    analysis_files = list(ds.get_root("media").glob("*.analysis.mp4"))
    playback_files = list(ds.get_root("media").glob("*.playback.mp4"))
    assert len(analysis_files) == 1
    assert len(playback_files) == 1
    analysis_derivative = analysis_files[0]

    # Reconstruct each per-target derivative's stored facts from its media-index row.
    media_df = pd.read_csv(ds.get_root("media") / "index.csv")

    def _stored_facts(suffix: str):
        rows = [
            {str(k): v for k, v in row.items()}
            for _, row in media_df.iterrows()
            if str(row["abs_path"]).endswith(suffix)
        ]
        assert len(rows) == 1
        return row_to_facts(rows[0])

    analysis_stored = _stored_facts(".analysis.mp4")
    playback_stored = _stored_facts(".playback.mp4")
    if analysis_stored == playback_stored:
        pytest.skip(
            "this environment produced indistinguishable analysis/playback derivatives"
        )

    # Routing opens the analysis derivative and must carry ITS facts, not the
    # playback derivative's, even though playback was transcoded (and indexed) first.
    resolved = ds.resolve_media(group, sequence)
    assert resolved.paths[0].resolve() == analysis_derivative.resolve()
    assert resolved.facts is not None
    assert resolved.facts[0] == analysis_stored
    assert resolved.facts[0] != playback_stored
    assert (
        resolved.facts[0].frame_count == probe_media(analysis_derivative).frame_count
    )


def test_transcode_op_is_idempotent(tmp_path):
    ds = _make_dataset(tmp_path)
    source_dir = tmp_path / "raw_src"
    source_dir.mkdir()
    original = source_dir / "vfr.mp4"
    _write_analysis_required_mp4(original)
    _require_analysis_required(original)

    group, sequence = _indexed_entry(ds, source_dir)

    first = run_transcode_op(
        ds, TranscodeParams(entry=(group, sequence), target="analysis")
    )
    second = run_transcode_op(
        ds, TranscodeParams(entry=(group, sequence), target="analysis")
    )
    assert first == second

    # Re-running replaces, not duplicates, the derivative and its index row.
    assert len(list(ds.get_root("media").glob("*.mp4"))) == 1
    media_df = pd.read_csv(ds.get_root("media") / "index.csv")
    assert len(media_df) == 1
    raw_df = pd.read_csv(ds.get_root("media_raw") / "index.csv")
    assert len(raw_df) == 1
