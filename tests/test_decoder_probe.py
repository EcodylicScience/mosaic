"""Test the decode probe that a tool's environment answers before it is handed a file.

SLEAP and Lightning Pose are handed a file in a codec outside their declarations
only after a program run by the interpreter of the tool's environment decodes a
frame of it. The environments here are fakes. Each is a ``python`` shell script
that the tool's location ladder finds. It records its arguments and answers with a
chosen exit code. The probe programs themselves run against fake ``sleap_io`` and
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
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.tracking.common.tool_input import (
    DecodeProbe,
    ToolCodecError,
    refuse_undecodable_codec,
)
from mosaic.tracking.common.toolenv import ToolNotFoundError, tool_invocation
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
"""What the SLEAP probe prints when OpenCV cannot decode the file."""


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
    """The refusal quotes the probe and names the remedy and the setting.

    A second file of the codec is refused from the answer that the first probe
    gave, without a second run.
    """
    ds = _av1_dataset(tmp_path, write_cfr_mp4, ("s", "t"))
    python = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", exit_code=1, output=_SLEAP_FAILURE
    )
    probe = DecodeProbe(SLEAP_ENV)

    for sequence in ("s", "t"):
        with pytest.raises(ToolCodecError) as refused:
            refuse_undecodable_codec(
                ds,
                _clip(ds, sequence),
                kind="sleap",
                group="",
                sequence=sequence,
                decode_probe=probe,
            )
        message = str(refused.value)
        assert f"(, {sequence}) resolves to clip0.mp4, which is av1" in message
        assert _SLEAP_FAILURE in message
        assert str(python.interpreter) in message
        assert "pip uninstall -y opencv-python opencv-python-headless" in message
        assert "conda install -c conda-forge py-opencv" in message
        assert "--update-all" in message
        assert "MOSAIC_ALLOW_TOOL_CODECS=av1" in message

    assert python.calls() == [("-c", str(_clip(ds, "s")))]


def test_a_remembered_refusal_names_the_file_that_was_tested(
    tmp_path: Path, write_cfr_mp4: WriteVideo, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    assert "did not decode av1 in first.mp4:" in messages[0]
    assert "resolves to second.mp4, which is av1" in messages[1]
    assert f"did not decode av1 in {first} earlier in this run:" in messages[1]
    assert _SLEAP_FAILURE in messages[1]
    assert python.calls() == [("-c", str(first))]


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
    """Its declaration answers, for a codec it lists and for one it does not."""
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
    """A run remembers the answer, and the next run asks again."""
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
    """Each way a frame can fail to arrive exits 1 with the reader's message."""
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
    """DALI appends a native stacktrace, which would fill the tail of the output."""
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


def test_the_sleap_program_reads_real_files_in_a_sleap_environment(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    """A SLEAP environment reads H.264, and answers AV1 with one of two messages.

    Whether the environment decodes AV1 depends on the OpenCV installed there,
    which is why the probe exists. The program reports either answer with its own
    message. Skipped where no SLEAP environment resolves.
    """
    try:
        interpreter = tool_invocation(SLEAP_ENV, executable="python")
    except ToolNotFoundError as absent:
        pytest.skip(f"no SLEAP environment resolves: {absent}")
    h264 = tmp_path / "h264.mp4"
    write_h264_mp4(h264)
    av1 = tmp_path / "av1.mp4"
    write_cfr_mp4(av1)
    program = TRACKING_ROOTS["sleap"].decoder.probe

    def probe(path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*interpreter, "-c", program, str(path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    read = probe(h264)
    assert read.returncode == 0, read.stderr
    assert "sleap_io read frame 0 with shape (48, 64, " in read.stdout
    answered = probe(av1)
    if answered.returncode == 0:
        assert "sleap_io read frame 0 with shape (48, 64, " in answered.stdout
    else:
        assert "sleap_io could not read frame 0" in answered.stderr
