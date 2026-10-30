"""The ``prepare-embedding-data`` op: training windows cut from a sleap tracker run.

The tracker runs against :func:`~tests.helpers.install_fake_sleap`. The readback of
each entry's tracked instances and the writer of the two labels files are replaced
by fakes, the writer's recording the windows it is handed. The writer's snippet is
then run against the stand-in ``sleap_io``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.tracking.ops.prepare_embedding as prepare_embedding
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.markers import INFLIGHT_MARKER_NAME
from mosaic.core.pipeline.models import (
    holds_trained_models,
    model_index_path,
    model_run_root,
)
from mosaic.core.pipeline.ops import OPS, run_op
from mosaic.core.pipeline.run_log import run_log_path
from mosaic.core.track_library.sleap import AppearanceReadback
from mosaic.tracking.ops._common import resolve_training_data
from mosaic.tracking.ops.prepare_embedding import (
    EmbeddingLabelsFile,
    EmbeddingSource,
    PrepareEmbeddingDataParams,
    prepare_embedding_data_run_id,
    write_embedding_labels,
)
from mosaic.tracking.sleap.dataset_runs import sleap_index_path, sleap_run_root
from mosaic.tracking.sleap.embedding_windows import EntryWindows, plan_windows
from mosaic.tracking.sleap.labels import validation_labels_beside
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapError
from tests.helpers import (
    MediaClip,
    ReadbackRecord,
    StubInstance,
    install_fake_sleap,
    install_fake_tool_python,
    make_dataset,
    read_stub_labels,
    stub_media_dataset,
    write_appearance_readback,
    write_media_index,
    write_sleap_model,
    write_stub_sleap_io,
    write_stub_slp,
)

KIND = "prepare-embedding-data"

SEQUENCES = ("s1", "s2", "s3")


# --- fakes -------------------------------------------------------------------------


type TracksOf = Mapping[str, Sequence[str]]
"""The tracks present in every frame of an entry, by sequence."""


def _records(tracks: Sequence[str], n_frames: int) -> list[ReadbackRecord]:
    """One predicted instance of each of *tracks* in every one of *n_frames*."""
    return [
        ReadbackRecord(frame, track) for frame in range(n_frames) for track in tracks
    ]


@dataclass
class FakeReadback:
    """Stands in for the readback of tracked instances.

    Attributes:
        tracks: The tracks of each sequence's frames. A sequence not named has
            tracks ``a`` and ``b``.
        n_frames: How many frames each entry holds.
        read: The predictions file of each call, in order.
    """

    tracks: TracksOf = field(default_factory=dict[str, Sequence[str]])
    n_frames: int = 10
    read: list[Path] = field(default_factory=list[Path])

    def __call__(self, predicted: Path, out: Path, **_kwargs: object) -> Path:
        self.read.append(predicted)
        sequence = predicted.name.removesuffix(".predictions.slp")
        tracks = self.tracks.get(sequence, ("a", "b"))
        return write_appearance_readback(out, _records(tracks, self.n_frames), dim=0)


@dataclass
class FakeWriter:
    """Stands in for the labels writer, recording the files it is handed.

    Each file is written as a placeholder.

    Attributes:
        calls: The files of each call, in order.
    """

    calls: list[tuple[EmbeddingLabelsFile, ...]] = field(
        default_factory=list[tuple[EmbeddingLabelsFile, ...]]
    )

    def __call__(
        self, files: Sequence[EmbeddingLabelsFile], request: Path, **_kwargs: object
    ) -> None:
        _ = request
        self.calls.append(tuple(files))
        for each in files:
            _ = each.out.write_bytes(b"slp")

    @property
    def files(self) -> tuple[EmbeddingLabelsFile, ...]:
        (only,) = self.calls
        return only


@dataclass(frozen=True)
class World:
    """A dataset with a finished sleap run and the fakes of a preparation."""

    ds: Dataset
    sleap_run_id: str
    readback: FakeReadback
    writer: FakeWriter


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    _ = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, list(SEQUENCES))
    model = write_sleap_model(tmp_path / "model")
    sleap_run_id = run_op(ds, "sleap", {"model_paths": [str(model)]})
    readback = FakeReadback()
    writer = FakeWriter()
    monkeypatch.setattr(prepare_embedding, "read_back_embeddings", readback)
    monkeypatch.setattr(prepare_embedding, "write_embedding_labels", writer)
    return World(ds=ds, sleap_run_id=sleap_run_id, readback=readback, writer=writer)


def _params(
    sleap_run_id: str,
    *,
    train: Sequence[str] = ("s1", "s2"),
    val: Sequence[str] = ("s3",),
) -> dict[str, object]:
    return {
        "sleap_run_id": sleap_run_id,
        "train_entries": [{"sequence": name} for name in train],
        "val_entries": [{"sequence": name} for name in val],
        "window_frames": 5,
        "frames_per_window": 3,
    }


def _prepare(
    world: World,
    *,
    execution_id: str | None = None,
    train: Sequence[str] = ("s1", "s2"),
    val: Sequence[str] = ("s3",),
) -> str:
    params = _params(world.sleap_run_id, train=train, val=val)
    return run_op(world.ds, KIND, params, execution_id=execution_id)


def _rows(ds: Dataset) -> list[dict[str, str]]:
    path = model_index_path(ds, KIND)
    if not path.exists():
        return []
    return index_records(pd.read_csv(path, dtype="string", keep_default_na=False))


def _predictions(ds: Dataset, sleap_run_id: str, sequence: str) -> Path:
    return sleap_run_root(ds, sleap_run_id) / sequence / f"{sequence}.predictions.slp"


def _refusal(caught: pytest.ExceptionInfo[Exception]) -> ClassifiedError:
    assert isinstance(caught.value, ClassifiedError)
    return caught.value


# --- registration and identity -----------------------------------------------------


def test_the_op_declares_a_scope_free_cpu_conversion() -> None:
    op = OPS[KIND]

    assert (op.version, op.category, op.domain) == ("0.1", "convert", "tracking")
    assert op.scope_takes == "none"
    assert op.scope_dependent is False
    assert op.resource_class in {"", "cpu"}


def test_the_run_id_is_minted_over_the_params_alone(world: World) -> None:
    run_id = _prepare(world)

    params = PrepareEmbeddingDataParams.model_validate(_params(world.sleap_run_id))
    assert run_id == prepare_embedding_data_run_id(params)
    assert run_id.startswith("prepare-embedding-data.0.1-")


def test_prepared_embedding_data_is_not_a_trained_model() -> None:
    assert holds_trained_models(KIND) is False


# --- refusals ----------------------------------------------------------------------


def test_a_run_id_of_another_kind_is_refused(world: World) -> None:
    with pytest.raises(ClassifiedError, match="sleap") as caught:
        _ = run_op(world.ds, KIND, _params("trex.0.1-abcdef0123"))

    assert _refusal(caught).error_kind == "invalid"
    assert world.readback.read == []


def test_an_unknown_run_id_is_not_found(world: World) -> None:
    with pytest.raises(ClassifiedError, match="sleap.1.6-0000000000") as caught:
        _ = run_op(world.ds, KIND, _params("sleap.1.6-0000000000"))

    assert _refusal(caught).error_kind == "not_found"


def test_an_entry_the_run_did_not_track_is_not_found(world: World) -> None:
    with pytest.raises(ClassifiedError, match="s9") as caught:
        _ = _prepare(world, val=("s9",))

    assert _refusal(caught).error_kind == "not_found"
    assert world.readback.read == []


def test_an_entry_whose_predictions_are_gone_is_not_found(world: World) -> None:
    _predictions(world.ds, world.sleap_run_id, "s2").unlink()

    with pytest.raises(ClassifiedError, match="s2") as caught:
        _ = _prepare(world)

    assert _refusal(caught).error_kind == "not_found"
    assert world.readback.read == []


def test_a_file_of_fewer_than_two_tracks_is_refused_after_the_writer(
    world: World,
) -> None:
    """No frame of a validation entry that holds one track qualifies."""
    world.readback.tracks = {"s3": ("a",)}

    with pytest.raises(ClassifiedError, match="labels.val.slp") as caught:
        _ = _prepare(world)

    assert _refusal(caught).error_kind == "conflict"
    assert len(world.writer.calls) == 1
    params = PrepareEmbeddingDataParams.model_validate(_params(world.sleap_run_id))
    root = model_run_root(world.ds, KIND, prepare_embedding_data_run_id(params))
    assert list(root.iterdir()) == []
    assert not (root / INFLIGHT_MARKER_NAME).exists()
    assert _rows(world.ds) == []


def test_two_divisions_of_one_recording_on_opposite_sides_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation measures another recording, so one recording is on one side."""
    _ = install_fake_sleap(monkeypatch)
    ds = make_dataset(tmp_path)
    write_media_index(
        ds, [MediaClip(sequence=name, group="rec") for name in ("d1", "d2", "d3")]
    )
    ds.set_continuous_groups(["rec"])
    model = write_sleap_model(tmp_path / "model")
    sleap_run_id = run_op(ds, "sleap", {"model_paths": [str(model)]})
    readback = FakeReadback()
    monkeypatch.setattr(prepare_embedding, "read_back_embeddings", readback)
    monkeypatch.setattr(prepare_embedding, "write_embedding_labels", FakeWriter())
    params = {
        **_params(sleap_run_id),
        "train_entries": [{"group": "rec", "sequence": "d1"}],
        "val_entries": [{"group": "rec", "sequence": "d3"}],
    }

    with pytest.raises(ClassifiedError, match="rec") as caught:
        _ = run_op(ds, KIND, params)

    message = str(caught.value)
    assert "'rec'" in message
    assert "rec__d1" in message
    assert "rec__d3" in message
    assert _refusal(caught).error_kind == "invalid"
    assert readback.read == []


# --- reuse -------------------------------------------------------------------------


def _run_log_events(ds: Dataset, execution_id: str) -> list[str]:
    lines = run_log_path(ds.base_dir, execution_id).read_text().splitlines()
    return [json.loads(line)["ev"] for line in lines]


def _rerun(world: World, *, overwrite: bool = False, execution_id: str) -> str:
    params = _params(world.sleap_run_id)
    return run_op(
        world.ds, KIND, params, overwrite=overwrite, execution_id=execution_id
    )


def test_a_finished_preparation_is_reused_and_reports_a_cache_hit(
    world: World,
) -> None:
    run_id = _prepare(world, execution_id="exec-first")
    assert "cache_hit" not in _run_log_events(world.ds, "exec-first")

    again = _rerun(world, execution_id="exec-again")

    assert again == run_id
    assert len(world.writer.calls) == 1
    assert len(world.readback.read) == len(SEQUENCES)
    assert "cache_hit" in _run_log_events(world.ds, "exec-again")
    assert len(_rows(world.ds)) == 1


def test_a_preparation_missing_a_labels_file_is_prepared_again(world: World) -> None:
    run_id = _prepare(world)
    (model_run_root(world.ds, KIND, run_id) / "labels.val.slp").unlink()

    _ = _rerun(world, execution_id="exec-again")

    assert len(world.writer.calls) == 2
    assert "cache_hit" not in _run_log_events(world.ds, "exec-again")


def test_overwrite_prepares_again(world: World) -> None:
    _ = _prepare(world)

    _ = _rerun(world, overwrite=True, execution_id="exec-again")

    assert len(world.writer.calls) == 2
    assert "cache_hit" not in _run_log_events(world.ds, "exec-again")
    assert len(_rows(world.ds)) == 1


def test_a_changed_predictions_file_is_named_in_the_reuse_message(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    _ = _prepare(world)
    changed = _predictions(world.ds, world.sleap_run_id, "s2")
    _ = changed.write_bytes(changed.read_bytes() + b"more tracks")

    with caplog.at_level("WARNING", logger=prepare_embedding.__name__):
        _ = _rerun(world, execution_id="exec-again")

    assert len(world.writer.calls) == 1
    (message,) = [record.getMessage() for record in caplog.records]
    assert "'s2'" in message
    assert "'s1'" not in message
    assert "overwrite" in message
    assert "cache_hit" in _run_log_events(world.ds, "exec-again")


def test_a_touched_predictions_file_with_the_same_bytes_is_reused_silently(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    _ = _prepare(world)
    touched = _predictions(world.ds, world.sleap_run_id, "s1")
    stat = touched.stat()
    os.utime(touched, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    with caplog.at_level("WARNING", logger=prepare_embedding.__name__):
        _ = _rerun(world, execution_id="exec-again")

    assert caplog.records == []
    assert len(world.writer.calls) == 1


def test_unchanged_size_and_mtime_hash_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _prepare(world)
    hashed: list[Path] = []

    def counting(path: Path) -> str:
        hashed.append(path)
        return file_digest(path)

    monkeypatch.setattr(prepare_embedding, "file_digest", counting)

    _ = _rerun(world, execution_id="exec-again")

    assert hashed == []


def test_only_a_file_whose_size_or_mtime_changed_is_hashed(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _prepare(world)
    changed = _predictions(world.ds, world.sleap_run_id, "s3")
    _ = changed.write_bytes(changed.read_bytes() + b"x")
    hashed: list[Path] = []

    def counting(path: Path) -> str:
        hashed.append(path)
        return file_digest(path)

    monkeypatch.setattr(prepare_embedding, "file_digest", counting)

    _ = _rerun(world, execution_id="exec-again")

    assert hashed == [changed]


def test_a_swept_predictions_file_is_reused_without_comparison(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    run_id = _prepare(world)
    _predictions(world.ds, world.sleap_run_id, "s2").unlink()

    with caplog.at_level("WARNING", logger=prepare_embedding.__name__):
        again = _rerun(world, execution_id="exec-again")

    assert again == run_id
    assert caplog.records == []
    assert len(world.writer.calls) == 1
    assert "cache_hit" in _run_log_events(world.ds, "exec-again")


def _blank_slp_path(world: World, sequence: str) -> None:
    path = sleap_index_path(world.ds)
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    mask = (frame["run_id"] == world.sleap_run_id) & (frame["sequence"] == sequence)
    assert mask.any()
    frame.loc[mask, "slp_path"] = ""
    frame.to_csv(path, index=False)


@pytest.mark.parametrize("listed", ["train", "val"])
def test_an_entry_with_no_detections_is_refused_by_name(
    world: World, listed: str
) -> None:
    name = "s2" if listed == "train" else "s3"
    _blank_slp_path(world, name)

    with pytest.raises(ClassifiedError, match="no detections") as caught:
        _ = _prepare(world)

    assert name in str(caught.value)
    assert _refusal(caught).error_kind == "not_found"
    assert world.readback.read == []
    assert world.writer.calls == []


# --- what is written ---------------------------------------------------------------


def test_both_labels_files_are_written_in_the_run_root(world: World) -> None:
    run_id = _prepare(world)

    root = model_run_root(world.ds, KIND, run_id)
    labels = root / "labels.slp"
    assert labels.is_file()
    assert validation_labels_beside(labels) == root / "labels.val.slp"
    train, val = world.writer.files
    assert (train.out, val.out) == (labels, root / "labels.val.slp")


def test_each_file_reads_its_entries_predictions(world: World) -> None:
    _ = _prepare(world)

    train, val = world.writer.files
    expected = {
        name: _predictions(world.ds, world.sleap_run_id, name) for name in SEQUENCES
    }
    assert [source.predictions for source in train.sources] == [
        expected["s1"],
        expected["s2"],
    ]
    assert [source.predictions for source in val.sources] == [expected["s3"]]
    assert sorted(world.readback.read) == sorted(expected.values())


def test_the_writer_is_handed_the_planned_windows(world: World) -> None:
    _ = _prepare(world)

    train, _val = world.writer.files
    assert train.sources[0].windows == _windows_of("s1")


def test_validation_keeps_the_requested_windows_per_entry(world: World) -> None:
    params = {**_params(world.sleap_run_id), "val_windows_per_entry": 1}

    _ = run_op(world.ds, KIND, params)

    train, val = world.writer.files
    assert [len(source.windows.windows) for source in train.sources] == [2, 2]
    (only,) = val.sources
    assert [window.index for window in only.windows.windows] == [0]
    (row,) = _rows(world.ds)
    assert int(row["n_windows"]) == 5


def test_validation_keeps_every_window_when_none_is_set(world: World) -> None:
    params = {**_params(world.sleap_run_id), "val_windows_per_entry": None}

    _ = run_op(world.ds, KIND, params)

    _train, val = world.writer.files
    assert [len(source.windows.windows) for source in val.sources] == [2]


def test_the_videos_of_both_files_number_the_windows(world: World) -> None:
    run_id = _prepare(world)

    (row,) = _rows(world.ds)
    windows = [
        window
        for each in world.writer.files
        for source in each.sources
        for window in source.windows.windows
    ]
    assert int(row["n_windows"]) == len(windows) == 6
    assert row["run_id"] == run_id


def test_track_names_are_unique_across_windows_entries_and_files(
    world: World,
) -> None:
    _ = _prepare(world)

    names_by_window = [
        {instance.window_track for instance in window.instances}
        for each in world.writer.files
        for source in each.sources
        for window in source.windows.windows
    ]
    every = [name for names in names_by_window for name in names]
    assert len(every) == len(set(every)) == 12
    assert "s3/w00001/b" in every


def test_the_row_records_the_preparation(world: World) -> None:
    run_id = _prepare(world)

    (row,) = _rows(world.ds)
    assert row["kind"] == KIND
    assert row["status"] == "finished"
    assert row["finished_at"]
    assert row["sleap_run_id"] == world.sleap_run_id
    labels = model_run_root(world.ds, KIND, run_id) / "labels.slp"
    assert world.ds.resolve_path(row["artifact_path"]) == labels
    assert not Path(row["artifact_path"]).is_absolute()
    counts = {
        name: int(row[name])
        for name in ("n_windows", "n_frames", "n_train", "n_valid", "n_tracks")
    }
    assert counts == {
        "n_windows": 6,
        "n_frames": 18,
        "n_train": 12,
        "n_valid": 6,
        "n_tracks": 12,
    }


def test_the_row_records_each_predictions_file_read(world: World) -> None:
    _ = _prepare(world, train=("s3", "s1"), val=("s2",))

    (row,) = _rows(world.ds)
    consumed = json.loads(row["consumed_predictions"])
    expected: list[dict[str, object]] = []
    for name in SEQUENCES:
        path = _predictions(world.ds, world.sleap_run_id, name)
        stat = path.stat()
        expected.append(
            {
                "group": "",
                "sequence": name,
                "digest": file_digest(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    assert consumed == expected


def test_a_finished_run_reports_its_entries_before_its_end(world: World) -> None:
    _ = _prepare(world, execution_id="exec-prepare")

    lines = run_log_path(world.ds.base_dir, "exec-prepare").read_text().splitlines()
    events = [json.loads(line)["ev"] for line in lines]
    assert events[-1] == "finished"
    written = [
        json.loads(line)["entries_written"]
        for line in lines
        if json.loads(line)["ev"] == "entries_written"
    ]
    assert written == [3]


# --- what reads it -----------------------------------------------------------------


def test_a_training_reference_resolves_to_the_labels(world: World) -> None:
    run_id = _prepare(world)

    labels = model_run_root(world.ds, KIND, run_id) / "labels.slp"
    assert resolve_training_data(world.ds, run_id) == labels


def test_the_inventory_lists_prepared_data_and_no_model(world: World) -> None:
    run_id = _prepare(world)

    found = inventory(world.ds, kinds=["trained-model", "prepared-dataset"])
    by_kind = {(record.ref.kind, record.run_id): record for record in found.records}

    assert ("prepared-dataset", run_id) in by_kind
    assert by_kind["prepared-dataset", run_id].status == "complete"
    assert ("trained-model", run_id) not in by_kind


# --- the writer's snippet ----------------------------------------------------------


def _stub_sleap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, save_seconds: float = 0.0
) -> None:
    imports = write_stub_sleap_io(tmp_path / "stub", save_seconds=save_seconds)
    _ = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", imports=imports
    )


def _stub_predictions(
    tmp_path: Path,
    sequence: str,
    *,
    video: Path,
    instances: Sequence[StubInstance] = (StubInstance("a"), StubInstance("b")),
) -> Path:
    """A predictions file of 10 frames, each holding *instances*."""
    frames = [(frame, instances) for frame in range(10)]
    return write_stub_slp(
        tmp_path / "work" / f"{sequence}.predictions.slp", frames, video=str(video)
    )


def _windows_of(sequence: str, tracks: Sequence[str] = ("a", "b")) -> EntryWindows:
    """The windows that the world's params plan over an entry of 10 frames."""
    records = _records(tracks, 10)
    n = len(records)
    presence = AppearanceReadback(
        frame_idx=np.asarray([r.frame_idx for r in records], dtype=np.int64),
        track_name=np.asarray([r.track_name for r in records], dtype=np.str_),
        is_user=np.zeros(n, dtype=np.bool_),
        has_vector=np.zeros(n, dtype=np.bool_),
        vectors=np.zeros((n, 0), dtype=np.float32),
    )
    return plan_windows(
        presence,
        entry_key=sequence,
        window_frames=5,
        frames_per_window=3,
        min_tracks_present=2,
        split_tracks_at_gap=None,
    )


def _labels_files(
    tmp_path: Path,
    videos: Mapping[str, Path],
    instances: Sequence[StubInstance] = (StubInstance("a"), StubInstance("b")),
) -> list[EmbeddingLabelsFile]:
    sources = {
        name: EmbeddingSource(
            predictions=_stub_predictions(
                tmp_path, name, video=video, instances=instances
            ),
            windows=_windows_of(name),
        )
        for name, video in videos.items()
    }
    return [
        EmbeddingLabelsFile(
            out=tmp_path / "out" / "labels.slp", sources=(sources["s1"], sources["s2"])
        ),
        EmbeddingLabelsFile(
            out=tmp_path / "out" / "labels.val.slp", sources=(sources["s3"],)
        ),
    ]


def _videos(tmp_path: Path) -> dict[str, Path]:
    videos = {name: tmp_path / "media" / f"{name}.mp4" for name in SEQUENCES}
    for video in videos.values():
        video.parent.mkdir(parents=True, exist_ok=True)
        _ = video.write_bytes(b"video")
    return videos


def test_the_snippet_writes_one_video_per_window_of_user_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    files = _labels_files(tmp_path, _videos(tmp_path))
    (tmp_path / "out").mkdir()
    lines: list[str] = []

    write_embedding_labels(
        files, tmp_path / "out" / "request.json", on_activity=lines.append
    )

    train = read_stub_labels(tmp_path / "out" / "labels.slp")
    val = read_stub_labels(tmp_path / "out" / "labels.val.slp")
    assert (train["embed"], val["embed"]) == (True, True)
    assert (len(train["videos"]), len(val["videos"])) == (4, 2)
    assert train["videos"][0] == str(tmp_path / "media" / "s1.mp4")
    assert train["tracks"] == [
        "s1/w00000/a",
        "s1/w00000/b",
        "s1/w00001/a",
        "s1/w00001/b",
        "s2/w00000/a",
        "s2/w00000/b",
        "s2/w00001/a",
        "s2/w00001/b",
    ]
    first = train["frames"][0]
    assert (first["video"], first["frame_idx"]) == (0, 0)
    assert first["instances"] == [
        {"track": "s1/w00000/a", "predicted": False, "x": 0.0},
        {"track": "s1/w00000/b", "predicted": False, "x": 0.0},
    ]
    assert [frame["frame_idx"] for frame in val["frames"]] == [0, 2, 4, 5, 7, 9]
    assert any("wrote 4 videos and 8 tracks" in line for line in lines)
    assert any("wrote 2 videos and 4 tracks" in line for line in lines)
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "labels.slp",
        "labels.val.slp",
    ]


def test_the_snippet_keeps_a_user_instance_and_then_the_last_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    instances = (
        StubInstance("a", predicted=False, x=2.0),
        StubInstance("a", x=1.0),
        StubInstance("b", x=3.0),
        StubInstance("b", x=4.0),
    )
    files = _labels_files(tmp_path, _videos(tmp_path), instances)
    (tmp_path / "out").mkdir()

    write_embedding_labels(files, tmp_path / "out" / "request.json")

    written = [
        (instance["track"], instance["x"])
        for name in ("labels.slp", "labels.val.slp")
        for frame in read_stub_labels(tmp_path / "out" / name)["frames"]
        for instance in frame["instances"]
    ]
    assert written
    assert {(str(track).rsplit("/", 1)[-1], x) for track, x in written} == {
        ("a", 2.0),
        ("b", 4.0),
    }


def test_the_snippet_prints_while_it_embeds_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path, save_seconds=0.5)
    files = _labels_files(tmp_path, _videos(tmp_path))
    (tmp_path / "out").mkdir()
    lines: list[str] = []

    write_embedding_labels(
        files,
        tmp_path / "out" / "request.json",
        on_activity=lines.append,
        heartbeat_seconds=0.05,
    )

    beats = [line for line in lines if "embedding frames into" in line]
    assert any("labels.partial.slp" in line for line in beats)
    assert any("labels.val.partial.slp" in line for line in beats)


def test_the_snippet_fails_on_a_missing_video_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    videos = _videos(tmp_path)
    videos["s3"].unlink()
    files = _labels_files(tmp_path, videos)
    (tmp_path / "out").mkdir()

    with pytest.raises(SleapError, match=str(videos["s3"])):
        write_embedding_labels(files, tmp_path / "out" / "request.json")

    assert list((tmp_path / "out").iterdir()) == []
