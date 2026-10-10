"""The ``Running: <summary>`` log line, proved against real trackers.

``tests/core/pipeline/test_tool_env.py`` covers the shared tool-placement
primitives on their own terms. This test needs TRex and SLEAP to launch
through, so it lives here, in ``tests/tracking/common/``, beside the
trackers rather than beside the generic ``ToolEnv`` tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.tracking.sleap import run as sleap_run
from mosaic.tracking.sleap import training as sleap_training
from mosaic.tracking.trex import run as trex_run
from mosaic.tracking.trex.params import TrexParams


class _Launched(Exception):
    """Raised by a stand-in for ``run_supervised`` once it has the argv."""


@pytest.mark.parametrize("site", ["trex", "sleap", "sleap-train"])
def test_every_running_line_is_the_summary_of_the_argv_run(
    site: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One function prints an argv, so a program after ``-c`` is never quoted."""
    module = {"trex": trex_run, "sleap": sleap_run, "sleap-train": sleap_training}[site]
    launched: list[list[str]] = []
    summarized: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> tuple[str, str, int]:
        launched.append(list(argv))
        raise _Launched

    def summary(cmd: list[str], head: int | None = None) -> str:
        summarized.append(list(cmd))
        return "<summary>"

    monkeypatch.setattr(module, "run_supervised", run)
    monkeypatch.setattr(module, "command_summary", summary)
    caplog.set_level("INFO")

    with pytest.raises(_Launched):
        if site == "trex":
            _ = trex_run.run_trex_convert(
                tmp_path / "clip.mp4",
                tmp_path / "out",
                params=TrexParams(),
                env=trex_run.TREX_ENV.placed(bin_path=tmp_path / "trex"),
            )
        elif site == "sleap":
            _ = sleap_run.run_sleap_convert(
                tmp_path / "in.slp",
                tmp_path / "out.h5",
                sleap_bin=tmp_path / "sleap-convert",
            )
        else:
            labels = tmp_path / "in.slp"
            _ = labels.write_bytes(b"slp")
            _ = sleap_training.train_sleap(
                labels,
                tmp_path / "run",
                sleap_bin=tmp_path / "sleap-nn-train",
            )

    running = [r.getMessage() for r in caplog.records if "Running:" in r.getMessage()]
    assert running == ["Running: <summary>"]
    assert summarized == launched
