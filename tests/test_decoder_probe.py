"""Test the decode probe that runs in a tool's environment before the tool reads a file.

SLEAP and Lightning Pose are handed a file in a codec outside their declarations
only after a program run by the interpreter of the tool's environment decodes a
frame of it. The environments here are fakes. Each is a ``python`` shell script
that the tool's location ladder finds. It records its arguments and exits with a
chosen code. The probe programs themselves run against fake ``sleap_io`` and
``nvidia.dali`` modules, and against a real SLEAP environment where one resolves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.subprocess_util import ProcessCancelled
from mosaic.core.pipeline.tracking_roots import (
    DECODE_PROBE_IMPORT_FAILED,
    TRACKING_ROOTS,
    ToolCodecError,
)
from mosaic.tracking.common.tool_input import (
    DecodeProbe,
    refuse_undecodable_codec,
)
from mosaic.tracking.common.toolenv import ToolEnv, ToolNotFoundError, tool_invocation
from mosaic.tracking.litpose import dataset_runs as litpose_runs
from mosaic.tracking.litpose.params import LitposeParams
from mosaic.tracking.litpose.run import LITPOSE_ENV
from mosaic.tracking.sleap import dataset_runs as sleap_runs
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.sleap.run import SLEAP_ENV

from tests.helpers import (
    FakeToolPython,
    index_media_sequence,
    install_fake_litpose,
    install_fake_sleap,
    install_fake_tool_python,
    make_dataset,
    write_h264_mp4,
    write_litpose_model,
    write_sleap_model,
)

WriteVideo = Callable[..., None]

_SLEAP_FAILURE = "sleap_io could not read frame 0: IndexError: Failed to read frame 0"
"""The output of the SLEAP probe when OpenCV cannot decode the file."""


def _av1_dataset(
    tmp_path: Path, write_cfr_mp4: WriteVideo, sequences: Sequence[str] = ("s",)
) -> Dataset:
    """Return a dataset whose entries are each one AV1 clip, indexed."""
    ds = make_dataset(tmp_path / "ds")
    for sequence in sequences:
        write_cfr_mp4(ds.get_root("media_raw") / sequence / "clip0.mp4")
        index_media_sequence(ds, sequence, ["clip0.mp4"])
    return ds


def _clip(ds: Dataset, sequence: str = "s") -> Path:
    return ds.get_root("media_raw") / sequence / "clip0.mp4"


# --- the check ----------------------------------------------------------------


def test_a_file_that_the_environment_decodes_is_handed_to_the_tool(
    tmp_path: Path,
    write_cfr_mp4: WriteVideo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The probe is SLEAP's program, it is given the file, and one line says so."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "bin")

    refuse_undecodable_codec(
        ds,
        _clip(ds),
        kind="sleap",
        group="",
        sequence="s",
        decode_probe=DecodeProbe(SLEAP_ENV),
    )

    assert python.calls() == [("-c", str(_clip(ds)))]
    assert python.last_program() == TRACKING_ROOTS["sleap"].decoder.probe
    (line,) = capsys.readouterr().err.splitlines()
    assert line.startswith("[sleap]")
    assert "av1" in line
    assert str(python.interpreter) in line


def test_a_file_that_the_environment_cannot_decode_is_refused_with_its_output(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal quotes the probe and names the remedy and the setting."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", exit_code=1, output=_SLEAP_FAILURE
    )

    with pytest.raises(ToolCodecError) as refused:
        refuse_undecodable_codec(
            ds,
            _clip(ds),
            kind="sleap",
            group="",
            sequence="s",
            decode_probe=DecodeProbe(SLEAP_ENV),
        )

    message = str(refused.value)
    assert "(, s) resolves to clip0.mp4, which is av1" in message
    assert "did not decode av1 in clip0.mp4:" in message
    assert _SLEAP_FAILURE in message
    assert str(python.interpreter) in message
    assert "pip uninstall -y opencv-python opencv-python-headless" in message
    assert "conda install -c conda-forge py-opencv" in message
    assert "--update-all" in message
    assert "MOSAIC_ALLOW_TOOL_CODECS=av1" in message


def test_a_refusal_is_not_remembered_for_the_codec(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each file that the environment does not decode is tested itself.

    The SLEAP probe reads the file, so one unreadable file says nothing about the
    next file of its codec.
    """
    ds = make_dataset(tmp_path / "ds")
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    write_cfr_mp4(first)
    write_cfr_mp4(second)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", exit_code=1, output=_SLEAP_FAILURE
    )
    probe = DecodeProbe(SLEAP_ENV)
    messages: list[str] = []

    for path, sequence in ((first, "a"), (second, "b")):
        with pytest.raises(ToolCodecError) as refused:
            refuse_undecodable_codec(
                ds, path, kind="sleap", group="", sequence=sequence, decode_probe=probe
            )
        messages.append(str(refused.value))

    assert python.calls() == [("-c", str(first)), ("-c", str(second))]
    assert "did not decode av1 in first.mp4:" in messages[0]
    assert "did not decode av1 in second.mp4:" in messages[1]
    assert all("earlier in this run" not in message for message in messages)


_FAKE_SLEAP_IO_BY_NAME = """
import numpy as np


class _Video:
    def __init__(self, filename):
        self.filename = filename

    def __getitem__(self, index):
        if "bad" in self.filename:
            raise IndexError(f"Failed to read frame {index}")
        return np.zeros((4, 4, 3), np.uint8)


def load_video(filename):
    return _Video(filename)
"""
"""A ``sleap_io`` that reads every file but one whose name holds ``bad``."""


def test_a_file_after_a_refused_one_is_handed_over_when_it_decodes(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    bad, good = tmp_path / "bad.mp4", tmp_path / "good.mp4"
    write_cfr_mp4(bad)
    write_cfr_mp4(good)
    fakes = tmp_path / "fakes"
    (fakes / "sleap_io").mkdir(parents=True)
    _ = (fakes / "sleap_io" / "__init__.py").write_text(_FAKE_SLEAP_IO_BY_NAME)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", imports=fakes
    )
    probe = DecodeProbe(SLEAP_ENV)

    with pytest.raises(ToolCodecError, match="did not decode av1 in bad.mp4:"):
        refuse_undecodable_codec(
            ds, bad, kind="sleap", group="", sequence="a", decode_probe=probe
        )
    refuse_undecodable_codec(
        ds, good, kind="sleap", group="", sequence="b", decode_probe=probe
    )

    assert python.calls() == [("-c", str(bad)), ("-c", str(good))]


def test_the_result_is_kept_per_interpreter_and_not_per_codec(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codec that one interpreter decodes is tested again under another.

    ``MOSAIC_SLEAP_BIN`` is read on each check, so the interpreter can change
    within one probe's life. Both decode, so a result kept per codec alone would
    answer the second from the first.
    """
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    first = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "first")
    second = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "second")
    probe = DecodeProbe(SLEAP_ENV)

    for python in (first, second, first):
        monkeypatch.setenv(SLEAP_ENV.bin_var, str(python.directory / "sleap-convert"))
        refuse_undecodable_codec(
            ds, _clip(ds), kind="sleap", group="", sequence="s", decode_probe=probe
        )

    assert first.calls() == [("-c", str(_clip(ds)))]
    assert second.calls() == [("-c", str(_clip(ds)))]


def _conda_with_env(root: Path, env_name: str, output: str) -> Path:
    """Write a fake ``conda`` whose *env_name* holds a ``python`` that exits 1.

    Like ``conda run``, it appends a report of the failure after the child's
    output, and the report echoes the whole command, program included.
    """
    python = root / "envs" / env_name / "bin" / "python"
    python.parent.mkdir(parents=True)
    _ = python.write_text(f"#!/bin/sh\necho {output!r} >&2\nexit 1\n")
    python.chmod(0o755)
    conda = root / "bin" / "conda"
    conda.parent.mkdir(parents=True)
    _ = conda.write_text(
        "#!/bin/sh\n"
        "shift 4\n"
        'target="$1"\n'
        "shift\n"
        '"$target" "$@"\n'
        "code=$?\n"
        'printf "ERROR conda.cli.main_run:execute(125): \\`conda run %s %s\\` '
        'failed. (See above for error)\\n" "$target" "$*" >&2\n'
        "exit $code\n"
    )
    conda.chmod(0o755)
    return conda.parent


@pytest.mark.parametrize("placement", ["bin", "conda"])
def test_a_refusal_does_not_quote_the_probe_program(
    placement: str,
    tmp_path: Path,
    write_cfr_mp4: WriteVideo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The program is one argv token, and a refusal quotes the reader's error alone.

    Under conda the launcher's report echoes the command after the child's
    output, so the refusal quotes what came before it.
    """
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", exit_code=1, output=_SLEAP_FAILURE
    )
    if placement == "conda":
        bin_dir = _conda_with_env(tmp_path / "conda", "sleap", _SLEAP_FAILURE)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.delenv("CONDA_ENVS_DIRS", raising=False)
        monkeypatch.delenv(SLEAP_ENV.bin_var)
        monkeypatch.setenv(SLEAP_ENV.conda_env_var, "sleap")

    with pytest.raises(ToolCodecError) as refused:
        refuse_undecodable_codec(
            ds,
            _clip(ds),
            kind="sleap",
            group="",
            sequence="s",
            decode_probe=DecodeProbe(SLEAP_ENV),
        )

    message = str(refused.value)
    assert _SLEAP_FAILURE in message
    assert "sio.load_video" not in message
    assert "import sleap_io" not in message
    assert "conda.cli.main_run" not in message
    assert python.calls() == ([("-c", str(_clip(ds)))] if placement == "bin" else [])


def test_an_interpreter_that_cannot_start_refuses_the_file(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", startable=False
    )

    with pytest.raises(ToolCodecError) as refused:
        refuse_undecodable_codec(
            ds,
            _clip(ds),
            kind="sleap",
            group="",
            sequence="s",
            decode_probe=DecodeProbe(SLEAP_ENV),
        )

    message = str(refused.value)
    missing = f"No such file or directory: '{python.interpreter}'"
    assert f"The interpreter did not start: [Errno 2] {missing}" in message
    assert "MOSAIC_ALLOW_TOOL_CODECS=av1" in message


def test_a_probe_that_does_not_finish_refuses_the_file(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    _ = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "bin", seconds=30)

    with pytest.raises(ToolCodecError, match="did not finish within 0.5 seconds"):
        refuse_undecodable_codec(
            ds,
            _clip(ds),
            kind="sleap",
            group="",
            sequence="s",
            decode_probe=DecodeProbe(SLEAP_ENV, timeout=0.5),
        )


def test_a_cancel_stops_the_probe_and_is_not_remembered(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe cancelled once it started raises, and the next check tests again."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", seconds=30
    )
    probe = DecodeProbe(SLEAP_ENV)
    started = time.monotonic()

    with pytest.raises(ProcessCancelled):
        refuse_undecodable_codec(
            ds,
            _clip(ds),
            kind="sleap",
            group="",
            sequence="s",
            decode_probe=probe,
            cancel_check=lambda: bool(python.calls()),
        )

    assert time.monotonic() - started < 20
    _ = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "bin")
    refuse_undecodable_codec(
        ds, _clip(ds), kind="sleap", group="", sequence="s", decode_probe=probe
    )
    assert len(python.calls()) == 2


def test_a_file_whose_header_does_not_read_is_handed_to_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file a tool cannot open at all is the tool's to report, in its words."""
    ds = make_dataset(tmp_path / "ds")
    clip = _clip(ds)
    clip.parent.mkdir(parents=True)
    _ = clip.write_bytes(b"not a video")
    python = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "bin")

    refuse_undecodable_codec(
        ds,
        clip,
        kind="sleap",
        group="",
        sequence="s",
        decode_probe=DecodeProbe(SLEAP_ENV),
    )

    assert python.calls() == []


def test_a_listed_codec_skips_the_probe(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", exit_code=1, output=_SLEAP_FAILURE
    )
    monkeypatch.setenv("MOSAIC_ALLOW_TOOL_CODECS", "av1")

    refuse_undecodable_codec(
        ds,
        _clip(ds),
        kind="sleap",
        group="",
        sequence="s",
        decode_probe=DecodeProbe(SLEAP_ENV),
    )

    assert python.calls() == []


def test_a_caller_without_the_environment_gets_the_declared_answer(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SLEAP's declaration does not list AV1, and the file is refused untested."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "bin")

    with pytest.raises(ToolCodecError, match="MOSAIC_ALLOW_TOOL_CODECS=av1"):
        refuse_undecodable_codec(ds, _clip(ds), kind="sleap", group="", sequence="s")

    assert python.calls() == []


_UNPROBED_KINDS = (
    "trex",
    "trex-convert",
    "ultralytics",
    "infer-pose",
    "infer-points",
    "infer-localizer",
)


def test_only_sleap_and_lightning_pose_declare_a_probe() -> None:
    probed = {kind for kind, root in TRACKING_ROOTS.items() if root.decoder.probe}
    assert probed == {"sleap", "litpose"}
    assert set(TRACKING_ROOTS) == probed | set(_UNPROBED_KINDS)


@pytest.mark.parametrize("kind", _UNPROBED_KINDS)
def test_a_tool_without_a_probe_is_never_probed(
    kind: str,
    tmp_path: Path,
    write_cfr_mp4: WriteVideo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declaration alone decides a listed codec and an unlisted one."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "bin")
    probe = DecodeProbe(SLEAP_ENV)
    raw = tmp_path / "raw.avi"
    _ = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=16x16:r=5:d=0.4",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            str(raw),
        ],
        check=True,
        capture_output=True,
    )

    refuse_undecodable_codec(
        ds, _clip(ds), kind=kind, group="", sequence="s", decode_probe=probe
    )
    with pytest.raises(ToolCodecError, match="which is rawvideo"):
        refuse_undecodable_codec(
            ds, raw, kind=kind, group="", sequence="s", decode_probe=probe
        )

    assert python.calls() == []


# --- the runs -----------------------------------------------------------------


def test_two_entries_of_one_sleap_run_probe_once(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run keeps the result, and the next run tests again."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4, ("a", "b"))
    python = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "bin")
    sleap = install_fake_sleap(monkeypatch)
    params = SleapParams(model_paths=[str(write_sleap_model(tmp_path / "model"))])

    _ = sleap_runs.run_sleap(ds, params)

    assert sorted(sleap.tracked) == [_clip(ds, "a"), _clip(ds, "b")]
    assert python.calls() == [("-c", str(sleap.tracked[0]))]

    _ = sleap_runs.run_sleap(ds, params, overwrite=True)

    assert len(sleap.tracked) == 4
    assert len(python.calls()) == 2


def test_sleap_probes_the_environment_that_its_run_placed(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sleap_bin`` names the environment, above ``MOSAIC_SLEAP_BIN``."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    unplaced = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "unplaced", exit_code=1
    )
    placed = install_fake_tool_python(monkeypatch, SLEAP_ENV, tmp_path / "placed")
    monkeypatch.setenv(SLEAP_ENV.bin_var, str(unplaced.directory / "sleap-convert"))
    sleap = install_fake_sleap(monkeypatch)
    params = SleapParams(model_paths=[str(write_sleap_model(tmp_path / "model"))])

    _ = sleap_runs.run_sleap(ds, params, sleap_bin=placed.directory / "sleap-convert")

    assert sleap.tracked == [_clip(ds)]
    assert placed.calls() == [("-c", str(_clip(ds)))]
    assert unplaced.calls() == []


def test_lightning_pose_probes_the_environment_that_its_run_placed(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``litpose_bin`` names the environment, and the program is Lightning Pose's."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    unplaced = install_fake_tool_python(
        monkeypatch, LITPOSE_ENV, tmp_path / "unplaced", exit_code=1
    )
    placed = install_fake_tool_python(monkeypatch, LITPOSE_ENV, tmp_path / "placed")
    monkeypatch.setenv(LITPOSE_ENV.bin_var, str(unplaced.directory / "litpose"))
    litpose = install_fake_litpose(monkeypatch)
    params = LitposeParams(model_path=str(write_litpose_model(tmp_path / "model")))

    _ = litpose_runs.run_litpose(ds, params, litpose_bin=placed.directory / "litpose")

    assert litpose.predicted == [_clip(ds)]
    assert placed.calls() == [("-c", str(_clip(ds)))]
    assert placed.last_program() == TRACKING_ROOTS["litpose"].decoder.probe
    assert unplaced.calls() == []


def _cancel_once_probed(token: CancelToken, python: FakeToolPython) -> None:
    """Cancel *token* once *python* has started a probe, from another thread."""

    def watch() -> None:
        deadline = time.monotonic() + 60
        while not python.calls() and time.monotonic() < deadline:
            time.sleep(0.05)
        token.cancel()

    threading.Thread(target=watch, daemon=True).start()


def test_a_cancelled_sleap_run_stops_its_probe(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", seconds=30
    )
    sleap = install_fake_sleap(monkeypatch)
    params = SleapParams(model_paths=[str(write_sleap_model(tmp_path / "model"))])
    token = CancelToken()
    _cancel_once_probed(token, python)
    started = time.monotonic()

    with pytest.raises(Cancelled):
        _ = sleap_runs.run_sleap(ds, params, cancel_token=token)

    assert time.monotonic() - started < 20
    assert len(python.calls()) == 1
    assert sleap.tracked == []


def test_a_cancelled_lightning_pose_run_stops_its_probe(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, LITPOSE_ENV, tmp_path / "bin", seconds=30
    )
    litpose = install_fake_litpose(monkeypatch)
    params = LitposeParams(model_path=str(write_litpose_model(tmp_path / "model")))
    token = CancelToken()
    _cancel_once_probed(token, python)
    started = time.monotonic()

    with pytest.raises(Cancelled):
        _ = litpose_runs.run_litpose(ds, params, cancel_token=token)

    assert time.monotonic() - started < 20
    assert len(python.calls()) == 1
    assert litpose.predicted == []


# --- the programs -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sleap", "litpose"])
def test_each_probe_program_compiles(kind: str) -> None:
    _ = compile(TRACKING_ROOTS[kind].decoder.probe, f"<{kind} decode probe>", "exec")


def _run_program(
    kind: str, path: Path, fakes: Path, overlay: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run *kind*'s probe program in this interpreter, with *fakes* importable first."""
    search = os.pathsep.join(filter(None, [str(fakes), os.environ.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-c", TRACKING_ROOTS[kind].decoder.probe, str(path)],
        env={**os.environ, "PYTHONPATH": search, **overlay},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


_FAKE_SLEAP_IO = """
import os

import numpy as np


class _Video:
    def __getitem__(self, index):
        frame = os.environ["FAKE_FRAME"]
        if frame == "array":
            return np.zeros((4, 4, 3), np.uint8)
        if frame == "none":
            return None
        if frame == "empty":
            return np.zeros((0, 4, 3), np.uint8)
        raise IndexError(f"Failed to read frame {index}")


def load_video(filename):
    with open(os.environ["FAKE_LOG"], "w") as handle:
        handle.write(filename)
    if os.environ["FAKE_FRAME"] == "unopenable":
        raise FileNotFoundError(f"Video does not exist or cannot be opened: {filename}")
    return _Video()
"""


@pytest.mark.parametrize(
    ("frame", "exit_code", "said"),
    [
        ("array", 0, "sleap_io read frame 0 with shape (4, 4, 3)"),
        ("none", 1, "sleap_io read frame 0 as None"),
        ("empty", 1, "sleap_io read frame 0 as array([], shape=(0, 4, 3)"),
        ("raises", 1, _SLEAP_FAILURE),
        ("unopenable", 1, "FileNotFoundError: Video does not exist"),
    ],
)
def test_the_sleap_program_exits_zero_only_for_a_decoded_frame(
    frame: str, exit_code: int, said: str, tmp_path: Path
) -> None:
    """Each way that reading a frame can fail exits 1 with the reader's message."""
    fakes = tmp_path / "fakes"
    (fakes / "sleap_io").mkdir(parents=True)
    _ = (fakes / "sleap_io" / "__init__.py").write_text(_FAKE_SLEAP_IO)
    log = tmp_path / "loaded.txt"
    video = tmp_path / "clip.mp4"

    ran = _run_program(
        "sleap", video, fakes, {"FAKE_FRAME": frame, "FAKE_LOG": str(log)}
    )

    assert ran.returncode == exit_code, ran.stderr
    assert said in (ran.stdout if exit_code == 0 else ran.stderr)
    assert log.read_text() == str(video)


_FAKE_DALI = """
import json
import os

_RECORD = {}


def _note(key, value):
    _RECORD[key] = value
    with open(os.environ["FAKE_LOG"], "w") as handle:
        json.dump(_RECORD, handle)


class _Readers:
    def video(self, **arguments):
        _note("reader", {**arguments, "dtype": str(arguments["dtype"])})
        return "frames"


class _Functions:
    readers = _Readers()


fn = _Functions()


class _DataTypes:
    FLOAT = "DALIDataType.FLOAT"


class types:
    DALIDataType = _DataTypes


class _Pipeline:
    def build(self):
        _note("built", True)

    def run(self):
        failure = os.environ.get("FAKE_FAILURE", "")
        if failure:
            raise RuntimeError(failure)
        _note("ran", True)
        return ("frames",)


def pipeline_def(**settings):
    _note("pipeline", settings)

    def decorate(define):
        def make():
            define()
            return _Pipeline()

        return make

    return decorate
"""


def _fake_dali(tmp_path: Path) -> Path:
    fakes = tmp_path / "fakes"
    (fakes / "nvidia" / "dali").mkdir(parents=True)
    _ = (fakes / "nvidia" / "__init__.py").write_text("")
    _ = (fakes / "nvidia" / "dali" / "__init__.py").write_text(_FAKE_DALI)
    return fakes


def test_the_lightning_pose_program_reads_one_frame_as_its_reader_does(
    tmp_path: Path,
) -> None:
    """The reader gets the arguments of Lightning Pose's prediction reader."""
    log = tmp_path / "dali.json"
    video = tmp_path / "clip.mp4"

    ran = _run_program("litpose", video, _fake_dali(tmp_path), {"FAKE_LOG": str(log)})

    assert ran.returncode == 0, ran.stderr
    recorded = json.loads(log.read_text())
    assert recorded == {
        "pipeline": {"batch_size": 1, "num_threads": 1, "device_id": 0},
        "reader": {
            "device": "gpu",
            "filenames": [str(video)],
            "sequence_length": 1,
            "normalized": False,
            "dtype": "DALIDataType.FLOAT",
            "file_list_include_preceding_frame": True,
            "skip_vfr_check": True,
        },
        "built": True,
        "ran": True,
    }


def test_the_lightning_pose_program_exits_one_with_the_dali_error(
    tmp_path: Path,
) -> None:
    log = tmp_path / "dali.json"
    failure = "[/opt/dali/video_loader.cc:365] Unhandled codec 225 in clip.mp4"

    ran = _run_program(
        "litpose",
        tmp_path / "clip.mp4",
        _fake_dali(tmp_path),
        {"FAKE_LOG": str(log), "FAKE_FAILURE": failure},
    )

    assert ran.returncode == 1
    assert f"RuntimeError: {failure}" in ran.stderr


_DALI_REASON = (
    "Error in thread 0: [/opt/dali/dali/operators/video/legacy/reader/"
    "video_loader.cc:365] Unhandled codec 225 in /data/clip0.mp4"
)
"""The reason that DALI 2.3 gives for an AV1 file, as measured on an RTX 4000 Ada."""


def test_a_dali_refusal_ends_with_the_reason_and_not_the_stacktrace(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DALI appends a native stacktrace, and the refusal quotes the reason alone."""
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    frames = "\n".join(
        f"[frame {index}]: /opt/dali/lib/libdali_operators.so(+0x{index:06x})"
        for index in range(60)
    )
    failure = (
        "Critical error in pipeline:\n"
        "Error in GPU operator `nvidia.dali.fn.readers.video`,\n"
        "encountered:\n\n"
        f"{_DALI_REASON}\n"
        f"Stacktrace (60 entries):\n{frames}\n"
    )
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "dali.json"))
    monkeypatch.setenv("FAKE_FAILURE", failure)
    _ = install_fake_tool_python(
        monkeypatch, LITPOSE_ENV, tmp_path / "bin", imports=_fake_dali(tmp_path)
    )

    with pytest.raises(ToolCodecError) as refused:
        refuse_undecodable_codec(
            ds,
            _clip(ds),
            kind="litpose",
            group="",
            sequence="s",
            decode_probe=DecodeProbe(LITPOSE_ENV),
        )

    message = str(refused.value)
    assert _DALI_REASON in message
    assert "Stacktrace (" not in message
    assert "[frame " not in message


_READERS = {"sleap": "sleap_io", "litpose": "nvidia.dali"}
"""The package whose reader each probe program imports."""


def _unimportable_reader(tmp_path: Path, kind: str) -> Path:
    """Return a module directory in which *kind*'s reader raises on import."""
    fakes = tmp_path / "fakes"
    package = fakes.joinpath(*_READERS[kind].split("."))
    package.mkdir(parents=True)
    for parent in package.relative_to(fakes).parents:
        if parent != Path("."):
            _ = (fakes / parent / "__init__.py").write_text("")
    _ = (package / "__init__.py").write_text(
        "raise ImportError('not installed in this environment')\n"
    )
    return fakes


@pytest.mark.parametrize("kind", ["sleap", "litpose"])
def test_a_program_that_cannot_import_its_reader_exits_with_its_own_code(
    kind: str, tmp_path: Path
) -> None:
    """An interpreter without the reader is told apart from a reader that fails."""
    ran = _run_program(
        kind, tmp_path / "clip.mp4", _unimportable_reader(tmp_path, kind), {}
    )

    assert ran.returncode == DECODE_PROBE_IMPORT_FAILED, ran.stderr
    reason = "ImportError: not installed in this environment"
    assert f"{_READERS[kind]} did not import: {reason}" in ran.stderr


@pytest.mark.parametrize(
    ("kind", "env"), [("sleap", SLEAP_ENV), ("litpose", LITPOSE_ENV)]
)
def test_an_interpreter_that_does_not_import_the_reader_is_refused_as_such(
    kind: str,
    env: ToolEnv,
    tmp_path: Path,
    write_cfr_mp4: WriteVideo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal names the placement or the installation, not a decoder.

    It names neither variable of the placement ladder, because an argument such
    as ``sleap_bin=`` overrides both.
    """
    ds = _av1_dataset(tmp_path, write_cfr_mp4)
    python = install_fake_tool_python(
        monkeypatch, env, tmp_path / "bin", imports=_unimportable_reader(tmp_path, kind)
    )

    with pytest.raises(ToolCodecError) as refused:
        refuse_undecodable_codec(
            ds,
            _clip(ds),
            kind=kind,
            group="",
            sequence="s",
            decode_probe=DecodeProbe(env),
        )

    message = str(refused.value)
    assert (
        f"did not import that reader, because it is not an environment of {kind} "
        "or the reader's installation in it is broken"
    ) in message
    assert f"{_READERS[kind]} did not import" in message
    assert str(python.interpreter) in message
    assert "the environment this run was placed in" in message
    assert env.conda_env_var not in message
    assert env.bin_var not in message
    assert TRACKING_ROOTS[kind].decoder.remedy not in message
    assert "did not decode" not in message


def _probe_in_environment(
    interpreter: Sequence[str], kind: str, path: Path
) -> subprocess.CompletedProcess[str]:
    """Run *kind*'s probe program on *path* with a real tool environment's interpreter."""
    return subprocess.run(
        [*interpreter, "-c", TRACKING_ROOTS[kind].decoder.probe, str(path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_the_sleap_program_reads_real_files_in_a_sleap_environment(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    """A SLEAP environment reads H.264, and reports AV1 with one of two messages.

    The OpenCV installed in the environment decides whether it decodes AV1, and the
    probe tests that OpenCV. The program prints a message for either result. The
    test is skipped where a SLEAP environment does not resolve.
    """
    try:
        interpreter = tool_invocation(SLEAP_ENV, executable="python")
    except ToolNotFoundError as absent:
        pytest.skip(f"no SLEAP environment resolves: {absent}")
    h264 = tmp_path / "h264.mp4"
    write_h264_mp4(h264)
    av1 = tmp_path / "av1.mp4"
    write_cfr_mp4(av1)

    read = _probe_in_environment(interpreter, "sleap", h264)
    assert read.returncode == 0, read.stderr
    assert "sleap_io read frame 0 with shape (48, 64, " in read.stdout
    answered = _probe_in_environment(interpreter, "sleap", av1)
    if answered.returncode == 0:
        assert "sleap_io read frame 0 with shape (48, 64, " in answered.stdout
    else:
        assert "sleap_io could not read frame 0" in answered.stderr


def test_the_lightning_pose_program_reads_real_files_in_a_lightning_pose_environment(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    """A Lightning Pose environment reads H.264, and reports AV1 with one of two messages.

    DALI's ``fn.readers.video`` accepts a fixed list of codecs. The list in every
    DALI release measured omits AV1, and the reader then fails with "Unhandled
    codec" on any GPU. A decode is accepted too, for a release whose list includes
    AV1. The frames are 320 by 240, larger than the smallest frame that NVDEC
    decodes. The test is skipped where a Lightning Pose environment does not
    resolve.
    """
    try:
        interpreter = tool_invocation(LITPOSE_ENV, executable="python")
    except ToolNotFoundError as absent:
        pytest.skip(f"no Lightning Pose environment resolves: {absent}")
    h264 = tmp_path / "h264.mp4"
    write_h264_mp4(h264, frames=30, size=(320, 240))
    av1 = tmp_path / "av1.mp4"
    write_cfr_mp4(av1, frames=30, size=(320, 240))

    read = _probe_in_environment(interpreter, "litpose", h264)
    assert read.returncode == 0, read.stderr
    assert "DALI read one frame" in read.stdout
    answered = _probe_in_environment(interpreter, "litpose", av1)
    if answered.returncode == 0:
        assert "DALI read one frame" in answered.stdout
    else:
        assert "Unhandled codec" in answered.stderr
