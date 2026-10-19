"""`check_memory_headroom` refuses an estimate the host's memory cannot hold."""

from __future__ import annotations

import pytest

from mosaic.behavior.feature_library import helpers
from mosaic.behavior.feature_library.helpers import check_memory_headroom

GIB = 1024**3


def _host(monkeypatch: pytest.MonkeyPatch, available: int) -> None:
    monkeypatch.setattr(helpers, "host_memory_bytes", lambda: (16 * GIB, available))


def test_an_estimate_inside_the_available_memory_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host(monkeypatch, 10 * GIB)

    check_memory_headroom(7 * GIB, "xgboost", "fitting 5 models", "Lower n_estimators.")


def test_an_estimate_that_would_leave_no_headroom_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host(monkeypatch, 10 * GIB)

    with pytest.raises(MemoryError) as raised:
        check_memory_headroom(
            9 * GIB, "xgboost", "fitting 5 models", "Lower n_per_class."
        )

    message = str(raised.value)
    assert message.startswith("[xgboost] fitting 5 models")
    assert "9.0 GB" in message
    assert "10.0 GB" in message
    assert message.endswith("Lower n_per_class.")


def test_a_host_that_reports_no_memory_is_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(helpers, "host_memory_bytes", lambda: (0, 0))

    check_memory_headroom(1000 * GIB, "xgboost", "fitting", "Lower n_per_class.")
