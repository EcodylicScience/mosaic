"""The probe against the built lmt-analysis environment and a real checkout.

Skipped where the environment is not built or the checkout is not cloned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from mosaic.core.pipeline.toolenv import missing_interpreter
from mosaic.tracking.common.lmt_env import (
    LMT_ENV,
    lmt_analysis_checkout,
    probe_lmt_analysis,
)
from tests.helpers import git_in_checkout

_MISSING: Final = missing_interpreter(LMT_ENV)
"""How to provide the environment, or ``None`` when its ladder resolves."""

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _MISSING is not None,
        reason=f"The lmt-analysis environment does not resolve: {_MISSING}",
    ),
    pytest.mark.skipif(
        not lmt_analysis_checkout().is_dir(),
        reason="The lmt-analysis checkout does not exist.",
    ),
]

_BUILDER_COUNT: Final = 30


def _never() -> bool:
    return False


def test_the_probe_reports_the_thirty_builders_of_the_script(tmp_path: Path) -> None:
    commit = git_in_checkout(lmt_analysis_checkout(), "rev-parse", "HEAD")

    report = probe_lmt_analysis(
        commit, None, tmp_path / "probe-report.json", cancel_check=_never
    )

    assert report.commit == commit
    assert report.builders is not None
    assert len(report.builders) == _BUILDER_COUNT
    assert report.builders[-1] == "BuildEventFlickering"
