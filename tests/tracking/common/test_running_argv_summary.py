"""The ``Running: <summary>`` log line of the TRex and SLEAP launches.

``tests/core/pipeline/test_tool_env.py`` tests the shared tool-placement
primitives. This test launches through the TRex and SLEAP modules, and it
belongs in ``tests/tracking/common/`` beside the tests of those trackers.
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
    """Checks the log line of each launch against ``command_summary`` of its argv.

    ``command_summary`` renders a program after ``-c`` as ``<program>``.
    """
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
            _ = sleap_run.run_sleap_analysis_export(
                tmp_path / "in.slp",
                tmp_path / "out.h5",
                sleap_bin=tmp_path / "sleap-convert",
            )
        else:
            labels = tmp_path / "in.slp"
            _ = labels.write_bytes(b"slp")
            config = sleap_training.sleap_train_config(
                labels,
                tmp_path / "run",
                head="centered_instance",
                backbone="unet",
                max_epochs=1,
                seed=42,
                validation_fraction=0.1,
                validation_labels=None,
                base_weights=None,
                run_name="model",
            )
            _ = sleap_training.train_sleap(
                labels,
                tmp_path / "run",
                config=config,
                sleap_bin=tmp_path / "sleap-nn",
            )

    running = [r.getMessage() for r in caplog.records if "Running:" in r.getMessage()]
    assert running == ["Running: <summary>"]
    assert summarized == launched
