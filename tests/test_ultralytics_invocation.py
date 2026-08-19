"""Unit tests for the env-aware Ultralytics invocation resolution.

Covers :func:`mosaic.tracking.ultralytics_track.run._ultralytics_invocation` (how
the Ultralytics environment's ``python`` is launched: in a conda env, as a
sibling of an explicit binary, or from ``$PATH`` via the ``yolo`` console script)
and the ``_run_runner`` wiring, with no Ultralytics environment anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from mosaic.tracking.common import toolenv
from mosaic.tracking.ultralytics_track import run as ultralytics_run
from mosaic.tracking.external.runner.ultralytics_protocol import ProbeResponse
from mosaic.tracking.ultralytics_track.run import (
    PROBE_DEADLINE_FLOOR_SECONDS,
    UltralyticsNotFoundError,
    _run_runner,
    _ultralytics_invocation,
    probe_ultralytics,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove Ultralytics env vars and make ``which`` resolve fake script paths.

    The fake conda sits two levels deep for a reason. ``conda_invocation``
    resolves the environment's own executable by climbing ``parent.parent`` from
    the conda binary and probing ``<base>/bin/<executable>`` on the real
    filesystem. A one-level fake such as ``/p/conda`` makes that base ``/``, so
    the probe finds ``/bin/python`` wherever the host happens to have one and
    the test's result depends on the machine rather than on the code. ``/p``
    exists nowhere, so every candidate misses and the bare name is used.
    """
    for var in (
        "MOSAIC_ULTRALYTICS_CONDA_ENV",
        "MOSAIC_ULTRALYTICS_BIN",
        "CONDA_EXE",
        "CONDA_ENVS_DIRS",
    ):
        monkeypatch.delenv(var, raising=False)
    fakes = {"yolo": "/p/bin/yolo", "conda": "/p/bin/conda"}

    def fake_which(name: str) -> str | None:
        return fakes.get(name)

    monkeypatch.setattr(toolenv.shutil, "which", fake_which)


def _nothing_on_path(_name: str) -> str | None:
    """A ``$PATH`` holding neither the tool nor conda."""
    return None


# --- precedence: param conda env > param bin > env conda > env bin > which ---


def test_param_conda_env_wins() -> None:
    assert _ultralytics_invocation(conda_env="ul") == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "ul",
        "python",
    ]


def test_param_bin_resolves_python_as_a_sibling() -> None:
    # MOSAIC_ULTRALYTICS_BIN may point at the yolo script (or any bin entry);
    # what runs is the interpreter in the same directory, because a bare
    # `python` on $PATH would be the caller's own.
    assert _ultralytics_invocation(bin_path="/x/bin/yolo") == ["/x/bin/python"]


def test_param_conda_beats_param_bin() -> None:
    got = _ultralytics_invocation(conda_env="ul", bin_path="/x/bin/yolo")
    assert got[0] == "/p/bin/conda"


def test_env_conda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOSAIC_ULTRALYTICS_CONDA_ENV", "envc")
    assert _ultralytics_invocation() == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "envc",
        "python",
    ]


def test_param_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOSAIC_ULTRALYTICS_CONDA_ENV", "envc")
    assert _ultralytics_invocation(bin_path="/x/bin/yolo") == ["/x/bin/python"]


def test_env_bin_resolves_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOSAIC_ULTRALYTICS_BIN", "/y/bin/yolo")
    assert _ultralytics_invocation() == ["/y/bin/python"]


def test_default_path_lookup() -> None:
    # The python beside the yolo script on $PATH.
    assert _ultralytics_invocation() == ["/p/bin/python"]


# --- error paths ---


def test_default_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(toolenv.shutil, "which", _nothing_on_path)
    with pytest.raises(UltralyticsNotFoundError):
        _ = _ultralytics_invocation()


def test_conda_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(toolenv.shutil, "which", _nothing_on_path)
    with pytest.raises(UltralyticsNotFoundError):
        _ = _ultralytics_invocation(conda_env="ul")


def test_conda_uses_conda_exe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(toolenv.shutil, "which", _nothing_on_path)
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    assert _ultralytics_invocation(conda_env="ul")[0] == "/opt/conda/bin/conda"


# --- _run_runner wires invocation + request into the supervised subprocess ---


def test_the_runner_command_carries_the_request_and_neutralizes_mpl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded_command: list[str] = []
    recorded_env: dict[str, str] = {}

    def fake_supervised(
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
        poll_interval: float = 0.5,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[str, str, int]:
        recorded_command.extend(str(token) for token in argv)
        recorded_env.update(env or {})
        return ("ok", "", 0)

    monkeypatch.setattr(ultralytics_run, "run_supervised", fake_supervised)
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    request = tmp_path / "probe-request.json"
    response = tmp_path / "probe-response.json"
    stdout, stderr = _run_runner(
        "probe",
        request,
        response,
        idle_timeout=5,
        max_runtime=None,
        conda_env=None,
        bin_path="/x/bin/yolo",
        cancel_check=None,
        on_output=None,
    )

    assert recorded_command[0] == "/x/bin/python"
    # Resolved from the package rather than joined out of strings, so the file a
    # checkout, an editable install and a wheel each name is the one that ships.
    assert Path(recorded_command[1]).name == "ultralytics_runner.py"
    assert Path(recorded_command[1]).is_file()
    assert recorded_command[2:] == [
        "probe",
        "--request",
        str(request),
        "--out",
        str(response),
    ]
    # an inherited Jupyter module:// backend is neutralized for the subprocess
    assert recorded_env["MPLBACKEND"] == "Agg"
    assert (stdout, stderr) == ("ok", "")


# --- the probe is bounded by a deadline, not by an idleness window ---


def test_a_short_tracking_window_does_not_put_a_stopwatch_on_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe prints nothing until it answers, so silence is not evidence.

    A caller shortens ``idle_timeout`` so a hung tracker dies quickly -- a batch
    takes seconds -- and the same number would otherwise be the whole time a cold
    torch import and a checkpoint load off a network mount are allowed.
    """
    recorded_idle: list[float | None] = []

    def fake_supervised(
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
        poll_interval: float = 0.5,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[str, str, int]:
        recorded_idle.append(idle_timeout)
        tokens = [str(token) for token in argv]
        answer = ProbeResponse(
            has_ultralytics=True,
            has_lap=True,
            ultralytics_version="8.4.63",
            tracker_names=["bytetrack"],
            model_task="pose",
            n_keypoints=2,
            installed_tracker_table={},
        )
        _ = Path(tokens[tokens.index("--out") + 1]).write_text(answer.model_dump_json())
        return ("", "", 0)

    monkeypatch.setattr(ultralytics_run, "run_supervised", fake_supervised)

    probe = probe_ultralytics(
        "best.pt", tracker="bytetrack", idle_timeout=5, bin_path="/x/bin/yolo"
    )
    assert probe.n_keypoints == 2
    assert recorded_idle == [PROBE_DEADLINE_FLOOR_SECONDS]

    # A caller who asked for longer keeps it: the floor raises, never lowers.
    _ = probe_ultralytics(
        "best.pt", tracker="bytetrack", idle_timeout=4000, bin_path="/x/bin/yolo"
    )
    assert recorded_idle[1] == 4000
