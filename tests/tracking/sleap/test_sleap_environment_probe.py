"""What the probe reports of the SLEAP environment's analysis export."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.tracking.sleap.probe import (
    SleapProbeResponse,
    probe_sleap,
    require_analysis_export,
)
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapNotFoundError
from tests.helpers import install_fake_tool_python, write_stub_sleap_io


def _sleap_io_without_export(directory: Path) -> Path:
    package = directory / "sleap_io"
    package.mkdir(parents=True)
    _ = (package / "__init__.py").write_text("__version__ = '0.1'\n", encoding="utf-8")
    return directory


def test_a_sleap_io_with_the_export_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imports = write_stub_sleap_io(tmp_path / "stub")
    _ = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", imports=imports
    )

    probe = probe_sleap()

    assert probe.has_analysis_export
    assert probe.analysis_export_error == ""


def test_a_sleap_io_without_the_export_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imports = _sleap_io_without_export(tmp_path / "stub")
    _ = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", imports=imports
    )

    probe = probe_sleap()

    assert not probe.has_analysis_export
    assert "save_analysis_h5" in probe.analysis_export_error


def test_an_environment_without_sleap_io_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imports = tmp_path / "stub"
    imports.mkdir()
    _ = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", imports=imports
    )

    probe = probe_sleap()

    assert not probe.has_analysis_export
    assert "sleap_io" in probe.analysis_export_error


def test_a_missing_export_is_refused_with_the_install_that_provides_it() -> None:
    probe = SleapProbeResponse(
        analysis_export_error="ModuleNotFoundError: No module named 'sleap_io'"
    )

    with pytest.raises(SleapNotFoundError) as caught:
        require_analysis_export(probe)

    message = str(caught.value)
    assert "save_analysis_h5" in message
    assert "No module named 'sleap_io'" in message
    assert "sleap-io" in message


def test_a_present_export_is_accepted() -> None:
    require_analysis_export(SleapProbeResponse(has_analysis_export=True))
