"""The writer of a color interaction crop on an ffmpeg build without NVENC.

The writer's NVENC probe raises ``UnknownCodecError``, a ``ValueError``, on a
build without ``av1_nvenc``. The color path then asks for SVT-AV1.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from av.codec.codec import UnknownCodecError

from mosaic.behavior.visualization_library import interaction_crop


class _Opened:
    """A writer stand-in that records the encoder it was opened for."""

    def __init__(self, hwaccel: bool) -> None:
        self.hwaccel = hwaccel


def test_a_build_without_nvenc_writes_the_crop_with_svt_av1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[bool] = []

    def without_nvenc(
        path: Path,
        width: int,
        height: int,
        *,
        fps: float,
        hwaccel: bool,
        av1_preset: int,
    ) -> _Opened:
        asked.append(hwaccel)
        if hwaccel:
            raise UnknownCodecError("av1_nvenc")
        return _Opened(hwaccel)

    monkeypatch.setattr(interaction_crop, "FFmpegVideoWriter", without_nvenc)

    writer = interaction_crop.color_crop_writer(tmp_path / "crop.mp4", 64, 64, 30.0)

    assert asked == [True, False]
    assert isinstance(writer, _Opened)
    assert not writer.hwaccel
