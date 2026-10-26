"""The ``extract-frames`` op under the job contract, with the frame decode faked.

Its run-log lifecycle, its run id and the knobs kept out of it, reuse of a
finished run, cancel, the report of a camera whose extraction failed, and the
excluded frames that each camera receives.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mosaic.tracking.frame_extraction.dataset_runs as dataset_runs
from mosaic.cli import app
from mosaic.core.helpers import entry_camera_path
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.run_log import read_run_progress, read_runs, run_log_dir
from mosaic.core.dataset import Dataset
from mosaic.runlog import read_entry_errors, read_run
from mosaic.tracking.frame_extraction.dataset_runs import (
    ExtractFramesParams,
    ListedFramesRefused,
    extract_frames,
    frames_run_id,
)
from mosaic.tracking.frame_extraction.frame_runs import (
    frames_index,
    frames_index_path,
    frames_run_root,
)
from mosaic.tracking.frame_extraction.sampling import Exclusion

from tests.helpers import (
    MediaClip,
    make_dataset,
    stub_media_dataset,
    write_h264_mp4,
    write_media_index,
)


# --- run_id determinism ----------------------------------------------------


def test_hash_exclude_does_not_change_run_id():
    from mosaic.core.pipeline._utils import hash_params

    a = ExtractFramesParams(
        n_frames=10,
        method="uniform",
        parallel_workers=1,
        parallel_mode="thread",
    )
    b = ExtractFramesParams(
        n_frames=10,
        method="uniform",
        parallel_workers=8,
        parallel_mode="process",
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    # a real param DOES change it
    c = ExtractFramesParams(n_frames=11, method="uniform")
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


# --- extract-frames op (mocked decode) -------------------------------------


def _install_fake_extract(monkeypatch):
    """Fake the low-level frame extractor: write a PNG + run_info.json."""
    import mosaic.tracking.frame_extraction.dataset_runs as dr

    class _Res:
        def __init__(self, n):
            self.n_extracted = n
            self.n_requested = n

    def fake(video_path, n_frames, method, output_dir, run_id, **kw):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_0.png").write_bytes(b"x")
        (out / "run_info.json").write_text(
            json.dumps({"output_dir": str(out), "video_path": str(video_path)})
        )
        listed = kw.get("frame_indices")
        return _Res(len(listed) if listed is not None else n_frames)

    monkeypatch.setattr(dr, "_extract_frames", fake)
    return dr


def test_extract_frames_lifecycle(tmp_path, monkeypatch):
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _install_fake_extract(monkeypatch)

    from mosaic.tracking import extract_frames

    run_id = extract_frames(ds, n_frames=3, method="uniform")
    assert run_id.startswith("uniform-")

    runs = read_runs(run_log_dir(ds.base_dir), kind="extract-frames")
    assert len(runs) == 1 and runs[0]["status"] == "finished"
    assert runs[0]["run_id"] == run_id
    assert int(runs[0]["progress_total"]) == 2

    # FramesIndexRow per sequence
    from mosaic.tracking.frame_extraction.frame_runs import (
        frames_index,
        frames_index_path,
    )

    idx = frames_index(frames_index_path(ds, "uniform"))
    df = idx.read(run_id=run_id)
    assert set(df["sequence"]) == {"vid1", "vid2"}

    # per-entry progress recorded
    prog = read_run_progress(run_log_dir(ds.base_dir), runs[0]["execution_id"])
    assert len([p for p in prog if p["step_type"] == "entry"]) == 2

    # cache hit: same params -> same run_id, new attempt
    run_id2 = extract_frames(ds, n_frames=3, method="uniform")
    assert run_id2 == run_id
    assert len(read_runs(run_log_dir(ds.base_dir), kind="extract-frames")) == 2


def test_extract_frames_list_lifecycle(tmp_path, monkeypatch):
    """A list run is named by its frames, covers the listed entries, and reuses."""
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _install_fake_extract(monkeypatch)

    from mosaic.tracking import extract_frames
    from mosaic.tracking.frame_extraction.frame_runs import (
        frames_index,
        frames_index_path,
    )

    run_id = extract_frames(ds, method="list", frames={("", "vid1"): [3, 1, 3]})
    assert run_id.startswith("list-")

    runs = read_runs(run_log_dir(ds.base_dir), kind="extract-frames")
    assert len(runs) == 1 and runs[0]["status"] == "finished"
    assert int(runs[0]["progress_total"]) == 1

    df = frames_index(frames_index_path(ds, "list")).read(run_id=run_id)
    assert list(df["sequence"]) == ["vid1"]
    assert list(df["n_frames_requested"]) == [2]

    # The same frames, listed another way: the same run, a new attempt.
    assert extract_frames(ds, method="list", frames={("", "vid1"): [1, 3]}) == run_id
    assert len(read_runs(run_log_dir(ds.base_dir), kind="extract-frames")) == 2


def test_extract_frames_cancel(tmp_path, monkeypatch):
    ds = stub_media_dataset(tmp_path, ["a", "b", "c"])
    from mosaic.tracking import extract_frames

    token = CancelToken()
    writer = _FailingExtractor(failing=())

    calls = {"n": 0}

    def fake_cancelling(
        *, video_path: Path, n_frames: int | None, output_dir: Path, **_: object
    ) -> _Written:
        calls["n"] += 1
        if calls["n"] == 1:
            token.cancel()  # request cancel after the first sequence
        return writer(video_path=video_path, n_frames=n_frames, output_dir=output_dir)

    monkeypatch.setattr(dataset_runs, "_extract_frames", fake_cancelling)

    with pytest.raises(Cancelled):
        extract_frames(
            ds, n_frames=2, method="uniform", parallel_workers=1, cancel_token=token
        )

    runs = read_runs(run_log_dir(ds.base_dir), kind="extract-frames")
    assert len(runs) == 1 and runs[0]["status"] == "cancelled"


# --- a camera whose extraction failed ----------------------------------------


@dataclass(frozen=True, slots=True)
class _Written:
    """The counts the extractor returns for one camera."""

    n_extracted: int
    n_requested: int


@dataclass(frozen=True, slots=True)
class _FailingExtractor:
    """Writes one placeholder frame per camera, and raises for the *failing* files.

    A file is named by its stem. Failing by name keeps the outcome independent of
    the order the workers finish in. With *partway*, a failing file writes its
    frame before it raises.
    """

    failing: Collection[str]
    partway: bool = False

    def __call__(
        self,
        *,
        video_path: Path,
        n_frames: int | None,
        output_dir: Path,
        **_settings: object,
    ) -> _Written:
        fails = video_path.stem in self.failing
        if self.partway or not fails:
            output_dir.mkdir(parents=True, exist_ok=True)
            _ = (output_dir / "frame_0.png").write_bytes(b"x")
        if fails:
            message = f"cannot decode {video_path.name}"
            raise OSError(message)
        count = n_frames or 0
        return _Written(n_extracted=count, n_requested=count)


def _fail_extracting(
    monkeypatch: pytest.MonkeyPatch, *failing: str, partway: bool = False
) -> None:
    """Fake the frame decode so that the files with the stems *failing* raise.

    With *partway*, each failing file writes a frame before it raises.
    """
    extractor = _FailingExtractor(failing, partway=partway)
    monkeypatch.setattr(dataset_runs, "_extract_frames", extractor)


def _rig_dataset(base: Path) -> Dataset:
    """Return a dataset of a two-camera entry ``rig`` and a one-camera entry ``solo``.

    Each camera's file is ``<sequence>_<camera>.mp4``, and ``solo.mp4`` for the
    camera of ``solo``.
    """
    dataset = make_dataset(base)
    write_media_index(
        dataset,
        [
            MediaClip(sequence="rig", camera="left", filename="rig_left.mp4"),
            MediaClip(sequence="rig", camera="right", filename="rig_right.mp4"),
            MediaClip(sequence="solo", filename="solo.mp4"),
        ],
    )
    return dataset


def _indexed_cameras(dataset: Dataset, run_id: str) -> list[tuple[str, str]]:
    """Return the (sequence, camera) of every frames-index row of *run_id*, sorted."""
    table = frames_index(frames_index_path(dataset, "uniform")).read(run_id=run_id)
    return sorted(zip(table["sequence"].astype(str), table["camera"].astype(str)))


@pytest.mark.parametrize("parallel_workers", [1, 2], ids=["inline", "parallel"])
def test_a_failed_camera_loses_its_entry_and_keeps_the_other_cameras_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parallel_workers: int
) -> None:
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch, "rig_right")

    run_id = extract_frames(
        dataset,
        n_frames=2,
        method="uniform",
        parallel_workers=parallel_workers,
        execution_id="RIGRIGHT",
    )

    snapshot = read_run(run_log_dir(dataset.base_dir), "RIGRIGHT")
    assert snapshot is not None
    assert snapshot["status"] == "finished"
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == 1
    errors = read_entry_errors(run_log_dir(dataset.base_dir), "RIGRIGHT")["items"]
    assert [error["key"] for error in errors] == ["rig"]
    assert errors[0]["error_type"] == "FrameExtractionFailed"
    assert "rig/right" in errors[0]["message"]
    assert "OSError: cannot decode rig_right.mp4" in errors[0]["message"]
    assert "rig/left" not in errors[0]["message"]
    # The run-log record is captured inside an except block, with a traceback.
    assert "Traceback" in json.loads(errors[0]["error_json"])["traceback"]
    # Rig's other camera is indexed, and so is solo.
    assert _indexed_cameras(dataset, run_id) == [("rig", "left"), ("solo", "")]


def test_an_entry_is_recorded_once_naming_every_failed_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch, "rig_left", "rig_right")

    run_id = extract_frames(
        dataset, n_frames=2, method="uniform", parallel_workers=2, execution_id="RIG"
    )

    errors = read_entry_errors(run_log_dir(dataset.base_dir), "RIG")["items"]
    assert [error["key"] for error in errors] == ["rig"]
    message = errors[0]["message"]
    assert "rig/left" in message and "rig_left.mp4" in message
    assert "rig/right" in message and "rig_right.mp4" in message
    snapshot = read_run(run_log_dir(dataset.base_dir), "RIG")
    assert snapshot is not None
    assert (snapshot["entries_failed"], snapshot["entries_written"]) == (1, 1)
    assert _indexed_cameras(dataset, run_id) == [("solo", "")]


def test_a_clean_run_records_every_entry_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch)

    _ = extract_frames(
        dataset, n_frames=2, method="uniform", parallel_workers=1, execution_id="CLEAN"
    )

    snapshot = read_run(run_log_dir(dataset.base_dir), "CLEAN")
    assert snapshot is not None
    assert (snapshot["entries_failed"], snapshot["entries_written"]) == (0, 2)


def test_an_existing_frame_directory_is_skipped_and_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A camera whose directory exists is neither attempted nor lost."""
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch, "solo")
    run_id = extract_frames(
        dataset, n_frames=2, method="uniform", parallel_workers=1, execution_id="FIRST"
    )

    # Again with the decode repaired: rig's cameras exist and are skipped.
    _fail_extracting(monkeypatch)
    again = extract_frames(
        dataset, n_frames=2, method="uniform", parallel_workers=1, execution_id="SECOND"
    )

    assert again == run_id
    snapshot = read_run(run_log_dir(dataset.base_dir), "SECOND")
    assert snapshot is not None
    assert snapshot["status"] == "finished"
    assert (snapshot["entries_failed"], snapshot["entries_written"]) == (0, 1)
    assert read_entry_errors(run_log_dir(dataset.base_dir), "SECOND")["total"] == 0
    # Every camera is skipped now. The run finishes with zero entries written.
    _ = extract_frames(
        dataset, n_frames=2, method="uniform", parallel_workers=1, execution_id="THIRD"
    )
    snapshot = read_run(run_log_dir(dataset.base_dir), "THIRD")
    assert snapshot is not None
    assert snapshot["status"] == "finished"
    assert (snapshot["entries_failed"], snapshot["entries_written"]) == (0, 0)


def test_a_camera_that_failed_partway_is_extracted_on_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed camera leaves no directory, so the next run attempts it again."""
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch, "rig_right", partway=True)
    run_id = extract_frames(dataset, n_frames=2, method="uniform", parallel_workers=1)

    run_root = frames_run_root(dataset, "uniform", run_id)
    assert not entry_camera_path(run_root, "", "rig", "right").exists()
    assert entry_camera_path(run_root, "", "rig", "left").is_dir()

    _fail_extracting(monkeypatch)
    _ = extract_frames(
        dataset, n_frames=2, method="uniform", parallel_workers=1, execution_id="RETRY"
    )

    snapshot = read_run(run_log_dir(dataset.base_dir), "RETRY")
    assert snapshot is not None
    assert (snapshot["entries_failed"], snapshot["entries_written"]) == (0, 1)
    assert _indexed_cameras(dataset, run_id) == [
        ("rig", "left"),
        ("rig", "right"),
        ("solo", ""),
    ]


@pytest.mark.parametrize(
    "failing",
    [("rig_right", "solo"), ("rig_left", "rig_right", "solo")],
    ids=["a-camera-of-each", "every-camera"],
)
def test_losing_every_attempted_entry_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: Sequence[str]
) -> None:
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch, *failing)

    with pytest.raises(AllEntriesFailed, match="2 attempted entries"):
        _ = extract_frames(
            dataset,
            n_frames=2,
            method="uniform",
            parallel_workers=1,
            execution_id="ALL",
        )

    snapshot = read_run(run_log_dir(dataset.base_dir), "ALL")
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] == 2
    assert snapshot["entries_written"] == 0
    errors = read_entry_errors(run_log_dir(dataset.base_dir), "ALL")["items"]
    assert sorted(error["key"] for error in errors) == ["rig", "solo"]


def test_an_entry_lost_after_a_skip_still_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skipped entry is not attempted. A run that loses every other entry fails."""
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch, "solo")
    _ = extract_frames(dataset, n_frames=2, method="uniform", parallel_workers=1)

    with pytest.raises(AllEntriesFailed, match="1 attempted entries"):
        _ = extract_frames(dataset, n_frames=2, method="uniform", parallel_workers=1)


def test_the_cli_reports_a_run_that_lost_an_entry_as_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _rig_dataset(tmp_path)
    _fail_extracting(monkeypatch, "solo")
    params = {"n_frames": 2, "method": "uniform", "parallel_workers": 1}

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--kind",
            "extract-frames",
            "-m",
            str(dataset.manifest_path),
            "--params",
            json.dumps(params),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "partial"
    assert payload["entries_failed"] == 1
    assert payload["entries_written"] == 1


# --- the frames each camera excludes ------------------------------------------


@dataclass(frozen=True, slots=True)
class _RecordingExtractor:
    """Writes one placeholder frame per camera, and records its exclusion by stem."""

    received: dict[str, Exclusion | None] = field(default_factory=dict)

    def __call__(
        self,
        *,
        video_path: Path,
        n_frames: int | None,
        output_dir: Path,
        excluded: Exclusion | None,
        **_settings: object,
    ) -> _Written:
        self.received[video_path.stem] = excluded
        output_dir.mkdir(parents=True, exist_ok=True)
        _ = (output_dir / "frame_0.png").write_bytes(b"x")
        count = n_frames or 0
        return _Written(n_extracted=count, n_requested=count)


def _record_exclusions(monkeypatch: pytest.MonkeyPatch) -> _RecordingExtractor:
    """Fake the frame decode, and return the record of each camera's exclusion."""
    extractor = _RecordingExtractor()
    monkeypatch.setattr(dataset_runs, "_extract_frames", extractor)
    return extractor


@pytest.mark.parametrize("parallel_workers", [1, 2], ids=["inline", "parallel"])
def test_every_camera_receives_the_exclusion_of_its_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    parallel_workers: int,
) -> None:
    """``ghost`` names no entry of the run. Every camera records it as ignored."""
    dataset = _rig_dataset(tmp_path)
    extractor = _record_exclusions(monkeypatch)

    _ = extract_frames(
        dataset,
        n_frames=2,
        method="uniform",
        parallel_workers=parallel_workers,
        exclude={("", "rig"): [5, 2, 5], ("", "ghost"): [1]},
        exclude_spacing=3,
    )

    assert extractor.received == {
        "rig_left": Exclusion((2, 5), 3, 1),
        "rig_right": Exclusion((2, 5), 3, 1),
        "solo": Exclusion((), 3, 1),
    }
    assert "1 of 2 exclude entries" in capsys.readouterr().out


def test_a_run_without_exclude_excludes_nothing_from_any_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _rig_dataset(tmp_path)
    extractor = _record_exclusions(monkeypatch)

    _ = extract_frames(dataset, n_frames=2, method="uniform", parallel_workers=1)

    assert extractor.received == {"rig_left": None, "rig_right": None, "solo": None}


def test_an_excluded_frame_past_the_end_is_refused_before_any_camera_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each camera holds 100 frames. An index past the end names other media."""
    dataset = _rig_dataset(tmp_path)
    extractor = _record_exclusions(monkeypatch)

    with pytest.raises(ListedFramesRefused) as excinfo:
        _ = extract_frames(
            dataset,
            n_frames=2,
            method="uniform",
            parallel_workers=1,
            exclude={("", "solo"): [0, 100, 140]},
        )

    message = str(excinfo.value)
    assert "1 camera(s) have exclude indices" in message
    assert "solo has 100 frames" in message
    assert "100, 140" in message
    assert extractor.received == {}
    assert not (dataset.get_root("frames") / "uniform").exists()


@pytest.mark.media
def test_every_outcome_returns_from_a_process_worker(tmp_path: Path) -> None:
    """A process worker returns each camera's outcome to the run by pickle.

    One entry decodes, one has bytes that do not, and one already has its
    directory. Each outcome is recorded as the run reports it.
    """
    dataset = make_dataset(tmp_path)
    media_root = dataset.get_root(dataset.resolve_media_root())
    write_h264_mp4(media_root / "decodable.mp4", frames=6, size=(64, 48))
    write_media_index(
        dataset,
        [
            MediaClip(sequence="decodable", width=64, height=48, frame_count=6),
            MediaClip(sequence="undecodable"),
            MediaClip(sequence="existing"),
        ],
    )
    params = ExtractFramesParams(
        n_frames=2, method="uniform", parallel_workers=2, parallel_mode="process"
    )
    run_root = frames_run_root(dataset, "uniform", frames_run_id("uniform", params))
    entry_camera_path(run_root, "", "existing", "").mkdir(parents=True)

    run_id = extract_frames(
        dataset,
        n_frames=2,
        method="uniform",
        parallel_workers=2,
        parallel_mode="process",
        execution_id="PROCESS",
    )

    snapshot = read_run(run_log_dir(dataset.base_dir), "PROCESS")
    assert snapshot is not None
    assert (snapshot["entries_failed"], snapshot["entries_written"]) == (1, 1)
    errors = read_entry_errors(run_log_dir(dataset.base_dir), "PROCESS")["items"]
    assert [error["key"] for error in errors] == ["undecodable"]
    assert _indexed_cameras(dataset, run_id) == [("decodable", "")]
