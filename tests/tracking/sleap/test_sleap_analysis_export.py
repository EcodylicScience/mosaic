"""The analysis export of a ``.slp``, written by sleap-io in the SLEAP environment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mosaic.tracking.sleap.run as sleap_run
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapError, run_sleap_analysis_export
from tests.helpers import (
    StubInstance,
    install_fake_tool_python,
    write_stub_sleap_io,
    write_stub_slp,
)


def _answering(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    writes: bool = True,
    published: Path | None = None,
) -> list[list[str]]:
    """Stand in for the SLEAP interpreter: write an export to the last argument."""
    commands: list[list[str]] = []

    def invocation(*_args: object, **_kwargs: object) -> list[str]:
        return ["/fake/bin/python"]

    def run(cmd: list[str], **_kwargs: object) -> tuple[str, str, int]:
        commands.append(cmd)
        if writes:
            _ = Path(cmd[-1]).write_bytes(b"h5")
        if published is not None:
            assert not published.exists()
        return ("", "the snippet failed", returncode)

    monkeypatch.setattr(sleap_run, "tool_invocation", invocation)
    monkeypatch.setattr(sleap_run, "run_supervised", run)
    return commands


def test_the_export_takes_its_paths_in_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slp = tmp_path / "e1.predictions.slp"
    h5 = tmp_path / "e1.analysis.h5"
    commands = _answering(monkeypatch)

    result = run_sleap_analysis_export(slp, h5)

    (cmd,) = commands
    assert cmd[:2] == ["/fake/bin/python", "-c"]
    assert cmd[3:] == [str(slp), str(tmp_path / "e1.analysis.partial.h5")]
    assert str(tmp_path) not in cmd[2]
    assert "sleap-convert" not in cmd
    assert result.analysis_h5_path == h5


def test_the_export_is_published_only_once_the_snippet_has_written_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5 = tmp_path / "e1.analysis.h5"
    _ = h5.write_bytes(b"stale")
    _ = _answering(monkeypatch, published=h5)

    _ = run_sleap_analysis_export(tmp_path / "e1.predictions.slp", h5)

    assert h5.read_bytes() == b"h5"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["e1.analysis.h5"]


def test_a_failed_export_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5 = tmp_path / "e1.analysis.h5"
    _ = h5.write_bytes(b"stale")
    _ = _answering(monkeypatch, returncode=1)

    with pytest.raises(SleapError, match="the snippet failed"):
        _ = run_sleap_analysis_export(tmp_path / "e1.predictions.slp", h5)

    assert list(tmp_path.iterdir()) == []


def _stub_sleap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imports = write_stub_sleap_io(tmp_path / "stub")
    _ = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", imports=imports
    )


def test_the_snippet_exports_every_frame_of_the_video_in_the_matlab_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    slp = write_stub_slp(
        tmp_path / "work" / "e1.predictions.slp",
        [(3, [StubInstance("a")]), (1, [StubInstance("b")])],
        video="clip.mp4",
    )
    h5 = tmp_path / "work" / "e1.analysis.h5"

    _ = run_sleap_analysis_export(slp, h5)

    assert json.loads(h5.read_text(encoding="utf-8")) == {
        "video": "clip.mp4",
        "labels_path": str(slp),
        "all_frames": True,
        "preset": "matlab",
        "frames": [3, 1],
    }
    assert not (tmp_path / "work" / "e1.analysis.partial.h5").exists()


def test_a_video_without_labeled_frames_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    work = tmp_path / "work"
    slp = write_stub_slp(work / "e1.predictions.slp", [], video="clip.mp4")
    h5 = work / "e1.analysis.h5"

    with pytest.raises(FileNotFoundError, match="No labeled frames in clip.mp4"):
        _ = run_sleap_analysis_export(slp, h5)

    assert sorted(path.name for path in work.iterdir()) == ["e1.predictions.slp"]
