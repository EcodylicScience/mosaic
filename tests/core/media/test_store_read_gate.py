"""The read gate passes an imgstore at the store level.

A store's facts describe a directory of chunks, and every video chunk is gated at
``"raw"`` when :class:`~mosaic.core.media.imgstore_native.NativeStore` opens it. So
:func:`~mosaic.core.media.read_target.verified_read_facts` returns store facts for
every target, including facts whose verdict would require a transcode of a file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from mosaic_media import CHROME_149, MediaFacts, MediaProbeError, derive

import mosaic.core.media.read_target as read_target
from mosaic.core.media.facts_columns import store_facts
from mosaic.core.media.imgstore_native import NativeStore
from mosaic.core.media.read_target import ReadTarget, verified_read_facts
from mosaic.media_probe_config import media_thresholds

from tests.helpers import video_store_maker


def _store_facts(*, codec: str, constant_frame_rate: bool) -> MediaFacts:
    """The facts of a 20 fps store of 100 frames, with *codec* and timing."""
    return store_facts(
        640,
        480,
        20.0,
        100,
        codec,
        4.95,
        video_uuid="",
        identity_scheme="",
        constant_frame_rate=constant_frame_rate,
        max_timestamp_gap_frame_periods=1.0 if constant_frame_rate else 6.0,
        max_instantaneous_fps=None if constant_frame_rate else 21.0,
    )


_FACTS = {
    "variable-rate": _store_facts(codec="h264", constant_frame_rate=False),
    "npy": _store_facts(codec="npy", constant_frame_rate=True),
    "format-string": _store_facts(
        codec="nvenc-slow-yuv420-lq30-gray", constant_frame_rate=True
    ),
}


@pytest.mark.parametrize("name", sorted(_FACTS))
def test_the_facts_require_an_analysis_transcode_of_a_file(name: str) -> None:
    verdict = derive(_FACTS[name], CHROME_149, media_thresholds())

    assert verdict.analysis_transcode == "required"


@pytest.mark.parametrize("target", ["analysis", "playback", "raw"])
@pytest.mark.parametrize("name", sorted(_FACTS))
def test_store_facts_pass_the_gate(
    tmp_path: Path, name: str, target: ReadTarget
) -> None:
    """The suite raises on a warning, so a raw read's warning fails this too."""
    facts = _FACTS[name]

    assert verified_read_facts(tmp_path / "rec", facts, target) == [facts]


def test_the_same_facts_of_a_file_are_refused(tmp_path: Path) -> None:
    facts = dataclasses.replace(_FACTS["variable-rate"], container="mp4")

    with pytest.raises(MediaProbeError, match="variable_frame_rate"):
        _ = verified_read_facts(tmp_path / "rec.mp4", facts, "analysis")


@pytest.mark.media
def test_a_store_read_warns_on_a_chunk_that_needs_a_transcode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunk whose probe claims 999 frames has unreliable timing metadata."""
    store_dir, _ = video_store_maker([5])(parent=tmp_path)
    real_probe = read_target.probe_media

    def _miscounted(path: Path) -> MediaFacts:
        return dataclasses.replace(real_probe(path), declared_frame_count=999)

    monkeypatch.setattr(read_target, "probe_media", _miscounted)

    with (
        NativeStore(store_dir) as store,
        pytest.warns(UserWarning, match="raw read of") as record,
    ):
        _ = store.frame(0)

    assert "unreliable_timing_metadata" in str(record[0].message)
