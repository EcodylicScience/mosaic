"""Unit tests for the env-aware SLEAP CLI invocation resolution.

Covers :func:`mosaic.tracking.sleap.run._sleap_invocation` (how a SLEAP console
script is launched: in a conda env, via a sibling of an explicit binary, or from
``$PATH``) and the ``_run_sleap`` wiring, without invoking the real SLEAP.
"""

from __future__ import annotations

import pytest

from mosaic.tracking.common import toolenv

from mosaic.tracking.sleap import run as sleap_run
from mosaic.tracking.sleap.run import (
    SleapNotFoundError,
    _run_sleap,
    _sleap_invocation,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove SLEAP env vars and make ``which`` resolve fake script/conda paths."""
    for var in (
        "MOSAIC_SLEAP_CONDA_ENV",
        "MOSAIC_SLEAP_BIN",
        "CONDA_EXE",
        "CONDA_ENVS_DIRS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        toolenv.shutil,
        "which",
        lambda name: {
            "sleap-track": "/p/sleap-track",
            "sleap-convert": "/p/sleap-convert",
            "conda": "/p/bin/conda",
        }.get(name),
    )


# --- precedence: param conda env > param bin > env conda > env bin > which ---


def test_param_conda_env_wins():
    assert _sleap_invocation("sleap-track", sleap_conda_env="sleap") == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "sleap",
        "sleap-track",
    ]


def test_param_bin_resolves_the_named_script_as_a_sibling():
    # MOSAIC_SLEAP_BIN may point at any one console script; the requested script
    # is resolved in the same directory.
    assert _sleap_invocation("sleap-convert", sleap_bin="/x/bin/sleap-track") == [
        "/x/bin/sleap-convert"
    ]


def test_param_conda_beats_param_bin():
    got = _sleap_invocation(
        "sleap-track", sleap_conda_env="sleap", sleap_bin="/x/sleap-track"
    )
    assert got[0] == "/p/bin/conda"


def test_env_conda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_SLEAP_CONDA_ENV", "envc")
    assert _sleap_invocation("sleap-track") == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "envc",
        "sleap-track",
    ]


def test_param_beats_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_SLEAP_CONDA_ENV", "envc")
    assert _sleap_invocation("sleap-track", sleap_bin="/x/bin/sleap-track") == [
        "/x/bin/sleap-track"
    ]


def test_env_bin_resolves_sibling(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", "/y/bin/sleap-track")
    assert _sleap_invocation("sleap-convert") == ["/y/bin/sleap-convert"]


def test_default_path_lookup():
    assert _sleap_invocation("sleap-track") == ["/p/sleap-track"]
    assert _sleap_invocation("sleap-convert") == ["/p/sleap-convert"]


# --- error paths ---


def test_default_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    with pytest.raises(SleapNotFoundError):
        _sleap_invocation("sleap-track")


def test_conda_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    with pytest.raises(SleapNotFoundError):
        _sleap_invocation("sleap-track", sleap_conda_env="sleap")


def test_conda_uses_conda_exe_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    assert _sleap_invocation("sleap-track", sleap_conda_env="sleap")[0] == (
        "/opt/conda/bin/conda"
    )


# --- _run_sleap wires invocation + args into the supervised subprocess call ---


def test_run_sleap_threads_invocation_and_neutralizes_mpl(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_supervised(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env")
        return ("ok", "", 0)  # (stdout, stderr, returncode)

    monkeypatch.setattr(sleap_run, "run_supervised", fake_supervised)
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    out, err = _run_sleap(
        ["/p/bin/conda", "run", "-n", "sleap", "sleap-track"],
        ["video.mp4", "-o", "out.slp"],
        idle_timeout=5,
    )
    assert captured["cmd"][:5] == ["/p/bin/conda", "run", "-n", "sleap", "sleap-track"]
    assert captured["cmd"][-3:] == ["video.mp4", "-o", "out.slp"]
    # an inherited Jupyter module:// backend is neutralised for the subprocess
    assert captured["env"]["MPLBACKEND"] == "Agg"
    assert out == "ok"
