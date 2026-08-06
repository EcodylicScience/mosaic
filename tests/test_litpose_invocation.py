"""Unit tests for the env-aware Lightning Pose invocation resolution.

Covers :func:`mosaic.tracking.litpose.run._litpose_invocation` (how the Lightning
Pose environment's ``python`` is launched: in a conda env, as a sibling of an
explicit binary, or from ``$PATH`` via the ``litpose`` script) and the
``_run_litpose`` wiring, without invoking the real Lightning Pose.
"""

from __future__ import annotations

import pytest

from mosaic.tracking.common import toolenv

from mosaic.tracking.litpose import run as litpose_run
from mosaic.tracking.litpose.run import (
    LitposeNotFoundError,
    _litpose_invocation,
    _run_litpose,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove Lightning Pose env vars and make ``which`` resolve fake script paths.

    The fake conda sits two levels deep for a reason. ``conda_invocation``
    resolves the environment's own executable by climbing ``parent.parent`` from
    the conda binary and probing ``<base>/bin/<executable>`` on the real
    filesystem. A one-level fake such as ``/p/conda`` makes that base ``/``, so
    the probe finds ``/bin/python`` wherever the host happens to have one and
    the test's result depends on the machine rather than on the code. ``/p``
    exists nowhere, so every candidate misses and the bare name is used.
    """
    for var in (
        "MOSAIC_LITPOSE_CONDA_ENV",
        "MOSAIC_LITPOSE_BIN",
        "CONDA_EXE",
        "CONDA_ENVS_DIRS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        toolenv.shutil,
        "which",
        lambda name: {
            "litpose": "/p/bin/litpose",
            "conda": "/p/bin/conda",
        }.get(name),
    )


# --- precedence: param conda env > param bin > env conda > env bin > which ---


def test_param_conda_env_wins():
    assert _litpose_invocation(litpose_conda_env="lp") == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "lp",
        "python",
    ]


def test_param_bin_resolves_python_as_a_sibling():
    # MOSAIC_LITPOSE_BIN may point at the litpose script (or any bin entry); the
    # env's python is resolved in the same directory.
    assert _litpose_invocation(litpose_bin="/x/bin/litpose") == ["/x/bin/python"]


def test_param_conda_beats_param_bin():
    got = _litpose_invocation(litpose_conda_env="lp", litpose_bin="/x/bin/litpose")
    assert got[0] == "/p/bin/conda"


def test_env_conda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_LITPOSE_CONDA_ENV", "envc")
    assert _litpose_invocation() == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "envc",
        "python",
    ]


def test_param_beats_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_LITPOSE_CONDA_ENV", "envc")
    assert _litpose_invocation(litpose_bin="/x/bin/litpose") == ["/x/bin/python"]


def test_env_bin_resolves_sibling(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_LITPOSE_BIN", "/y/bin/litpose")
    assert _litpose_invocation() == ["/y/bin/python"]


def test_default_path_lookup():
    # The python beside the litpose script on $PATH.
    assert _litpose_invocation() == ["/p/bin/python"]


# --- error paths ---


def test_default_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    with pytest.raises(LitposeNotFoundError):
        _litpose_invocation()


def test_conda_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    with pytest.raises(LitposeNotFoundError):
        _litpose_invocation(litpose_conda_env="lp")


def test_conda_uses_conda_exe_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    assert _litpose_invocation(litpose_conda_env="lp")[0] == "/opt/conda/bin/conda"


# --- _run_litpose wires invocation + args into the supervised subprocess call --


def test_run_litpose_threads_invocation_and_neutralizes_mpl(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_supervised(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env")
        return ("ok", "", 0)  # (stdout, stderr, returncode)

    monkeypatch.setattr(litpose_run, "run_supervised", fake_supervised)
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    out, err = _run_litpose(
        ["/p/bin/conda", "run", "-n", "lp", "python"],
        ["-c", "code", "model", "video.mp4", "out.csv", "fp32"],
        idle_timeout=5,
    )
    assert captured["cmd"][:5] == ["/p/bin/conda", "run", "-n", "lp", "python"]
    assert captured["cmd"][5:] == [
        "-c",
        "code",
        "model",
        "video.mp4",
        "out.csv",
        "fp32",
    ]
    # an inherited Jupyter module:// backend is neutralised for the subprocess
    assert captured["env"]["MPLBACKEND"] == "Agg"
    assert out == "ok"
