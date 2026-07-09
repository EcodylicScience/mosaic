"""Unit tests for the env-aware TRex CLI invocation resolution.

Covers :func:`mosaic.tracking.trex.run._trex_invocation` (how ``trex`` is
launched: in a conda env, via an explicit binary, or from ``$PATH``) and the
``DISPLAY`` overlay, without invoking the real ``trex`` binary.
"""

from __future__ import annotations

import pytest

from mosaic.tracking.trex import run as trex_run
from mosaic.tracking.trex.run import (
    TRexNotFoundError,
    _resolve_display,
    _trex_invocation,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove TREX env vars and make ``which`` resolve fake trex/conda paths."""
    for var in (
        "MOSAIC_TREX_CONDA_ENV",
        "MOSAIC_TREX_BIN",
        "MOSAIC_TREX_DISPLAY",
        "CONDA_EXE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        trex_run.shutil,
        "which",
        lambda name: {"trex": "/p/trex", "conda": "/p/conda"}.get(name),
    )


# --- precedence: param conda env > param bin > env conda > env bin > which ---


def test_param_conda_env_wins():
    assert _trex_invocation(trex_conda_env="track") == [
        "/p/conda",
        "run",
        "--no-capture-output",
        "-n",
        "track",
        "trex",
    ]


def test_param_bin():
    assert _trex_invocation(trex_bin="/x/trex") == ["/x/trex"]


def test_param_conda_beats_param_bin():
    assert _trex_invocation(trex_conda_env="track", trex_bin="/x/trex")[0] == "/p/conda"


def test_env_conda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_CONDA_ENV", "envc")
    assert _trex_invocation() == [
        "/p/conda",
        "run",
        "--no-capture-output",
        "-n",
        "envc",
        "trex",
    ]


def test_param_beats_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_CONDA_ENV", "envc")
    assert _trex_invocation(trex_bin="/x/trex") == ["/x/trex"]


def test_env_bin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_BIN", "/y/trex")
    assert _trex_invocation() == ["/y/trex"]


def test_default_path_lookup():
    assert _trex_invocation() == ["/p/trex"]


# --- error paths ---


def test_default_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trex_run.shutil, "which", lambda name: None)
    with pytest.raises(TRexNotFoundError):
        _trex_invocation()


def test_conda_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trex_run.shutil, "which", lambda name: None)
    with pytest.raises(TRexNotFoundError):
        _trex_invocation(trex_conda_env="track")


def test_conda_uses_conda_exe_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        trex_run.shutil,
        "which",
        lambda name: "/p/trex" if name == "trex" else None,
    )
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    assert _trex_invocation(trex_conda_env="track")[0] == "/opt/conda/bin/conda"


# --- DISPLAY overlay ---


def test_resolve_display_explicit():
    assert _resolve_display(":99") == {"DISPLAY": ":99"}


def test_resolve_display_none():
    assert _resolve_display(None) is None


def test_resolve_display_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_DISPLAY", ":7")
    assert _resolve_display(None) == {"DISPLAY": ":7"}


# --- _run_trex wires invocation + env into the supervised subprocess call ---


def test_run_trex_passes_invocation_and_env(monkeypatch: pytest.MonkeyPatch):
    """_run_trex now runs via the killable ``run_supervised`` helper; verify it
    still threads the resolved invocation prefix, args, and env overlay through."""
    captured: dict = {}

    def fake_supervised(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env")
        captured["cancel_check"] = kwargs.get("cancel_check")
        return ("ok", "", 0)  # (stdout, stderr, returncode)

    monkeypatch.setattr(trex_run, "run_supervised", fake_supervised)
    out, err = trex_run._run_trex(
        ["-task", "convert"],
        timeout=5,
        invocation=["/p/conda", "run", "-n", "track", "trex"],
        env={"DISPLAY": ":99"},
    )
    assert captured["cmd"][:5] == ["/p/conda", "run", "-n", "track", "trex"]
    assert captured["cmd"][-2:] == ["-task", "convert"]
    assert captured["env"]["DISPLAY"] == ":99"
    assert out == "ok"
