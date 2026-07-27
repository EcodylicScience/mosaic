"""Tests for video_metadata_or_probe: use supplied facts, or fall back to probing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from mosaic_media import MediaFacts, probe_media

from mosaic.core.media.video_io import video_metadata_or_probe


def test_supplied_facts_are_used_without_probing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_cfr_mp4: Callable[..., None],
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = probe_media(clip)

    def refuse(_path: Path) -> MediaFacts:
        raise AssertionError("a caller holding facts must not probe")

    monkeypatch.setattr("mosaic.core.media.video_io.probe_media", refuse)
    meta = video_metadata_or_probe(tmp_path / "absent.mp4", facts)
    assert meta.frame_count == facts.frame_count
    assert meta.fps == facts.fps


def test_no_facts_falls_back_to_the_probe(
    tmp_path: Path, write_cfr_mp4: Callable[..., None]
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    meta = video_metadata_or_probe(clip, None)
    assert meta.frame_count > 0
