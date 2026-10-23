"""VameFeature through ``run_feature``, against a fake VAME runner.

The fake serves the commands of ``vame_protocol`` under this interpreter. A fit
writes a state dict of the keypoint names and reports two epochs. ``load_model``
reads a model bundle as the real runner does. An embedding writes one latent per
window. A latent's first dimension is the frame at the window's center, and its
others are the model's ``seq_mean``. Every runner records its start and each
command in a journal, which the tests read.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final, NoReturn

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from mosaic.behavior.feature_library import vame_feature as vame_module
from mosaic.behavior.feature_library.external.vame_protocol import (
    VAME_VERSION,
    Training,
)
from mosaic.behavior.feature_library.runner_client import RunnerVersionError
from mosaic.behavior.feature_library.vame_feature import (
    MODEL_SETTINGS,
    VAME_ENV,
    VAME_MODEL_FILE,
    VAME_RUNNER_LOG,
    VameFeature,
    VameModelBundle,
    VameNotFoundError,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from tests.helpers import (
    VAME_PROTOCOL,
    SOURCE_ROOT,
    RunnerJournal,
    add_tracks_table,
    add_tracks_variant,
    assert_latents_at_centers,
    assert_process_exited,
    cancel_when,
    entry_error_lines,
    isolate_tool_location,
    journal_source,
    latest_events,
    latest_snapshot,
    make_dataset,
    model_reference,
    result_outputs,
    result_run_root,
    window_centers,
    write_fake_runner,
)

_VARIANT: Final = "convert-test.0.1-aaaaaaaaaa"
"""The tracks variant of each test's dataset."""

_SEQUENCES: Final = ("s1", "s2")
"""The sequences of each test's dataset, each with two individuals."""

_N_ROWS: Final = 40
"""The frames of each sequence."""

_TIME_WINDOW: Final = 5
"""The frames of one window. A window's latent is placed two frames after its start."""

_ZDIMS: Final = 3

_KEYPOINTS: Final = ("nose", "left", "right", "tail")

_SEQ_MEAN: Final = 0.25
"""The training mean that the fake fit reports, and its latents repeat."""

_VOLATILE_KEYS: Final = frozenset(
    {"project_path", "creation_datetime", "session_names", "creation_timestamp"}
)
"""The keys of VAME's project config that change from one fit to the next."""

_HANDLERS: Final = """
import joblib
import numpy as np

from vame_protocol import (
    Embed,
    EmbeddedSession,
    EmbedResult,
    EpochEvent,
    Fit,
    FitResult,
    FittedModel,
    LoadModel,
    StageEvent,
    TrainingMetadata,
)

MODEL = []
FITTED = False


def fit(payload, emit):
    global FITTED
    request = Fit.model_validate(payload)
    record(
        "fit",
        sessions=[session.name for session in request.sessions],
        exchange_dir=request.exchange_dir,
    )
    print(f"fit: {len(request.sessions)} sessions")
    emit(StageEvent.name, StageEvent(stage="train", message="training").model_dump())
    time.sleep(FIT_SECONDS)
    total = request.training.max_epochs - 1
    for epoch in (1, 2):
        losses = {"train_loss": 1.0 / epoch, "kmeans_loss": None}
        event = EpochEvent(epoch=epoch, total=total, losses=losses)
        emit(EpochEvent.name, event.model_dump())
    Path(request.state_dict).write_bytes(
        ("weights:" + ",".join(request.keypoints)).encode()
    )
    frames = 0
    for session in request.sessions:
        with np.load(session.track) as track:
            frames += len(track["frames"])
    FITTED = True
    return FitResult(
        project_name="mosaic",
        metadata=TrainingMetadata(keypoints_used=request.keypoints, extra_features=[]),
        num_features=2 * len(request.keypoints) - 3,
        seq_mean=SEQ_MEAN,
        seq_std=2.0,
        device="cpu",
        epochs=2,
        sessions=[session.name for session in request.sessions],
        train_frames=frames,
        test_frames=0,
        straddling_fraction=0.25,
    )


def load_model(payload, emit):
    request = LoadModel.model_validate(payload)
    MODEL[:] = [FittedModel.model_validate(joblib.load(request.path))]
    record("load_model", path=request.path, device=request.device)


def embed(payload, emit):
    request = Embed.model_validate(payload)
    if not MODEL:
        raise RuntimeError("embed came before load_model")
    if CRASH_AT_AN_EMBED_AFTER_THE_FIT and FITTED:
        record("crashing")
        os._exit(3)
    record(
        "embed",
        sessions=[session.name for session in request.sessions],
        exchanged=sorted(os.listdir(request.exchange_dir)),
    )
    print(f"embed: {len(request.sessions)} sessions")
    time.sleep(EMBED_SECONDS)
    model = MODEL[0]
    window = model.training.time_window
    embedded = []
    for session in request.sessions:
        with np.load(session.track) as track:
            frames = track["frames"]
        n_windows = len(frames) - window + 1
        latents = np.full((n_windows, model.training.zdims), model.seq_mean)
        latents[:, 0] = frames[window // 2 : window // 2 + n_windows]
        np.savez(session.latents, latents=latents.astype(np.float32))
        embedded.append(EmbeddedSession(name=session.name, windows=n_windows))
    return EmbedResult(device="cpu", sessions=embedded)


DISPATCH = {
    Fit.command: fit,
    LoadModel.command: load_model,
    Embed.command: embed,
}
"""


class _FakeVame:
    """The fake runner of one test, the features that it serves, and its journal."""

    def __init__(self, directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._directory = directory
        self._monkeypatch = monkeypatch
        self.journal = RunnerJournal(directory / "journal.jsonl")
        self._made: list[VameFeature] = []
        self.program()

    def program(
        self,
        *,
        fit_seconds: float = 0.0,
        embed_seconds: float = 0.0,
        crash_at_an_embed_after_the_fit: bool = False,
        version: str = VAME_VERSION,
        heartbeat_seconds: float = 30.0,
    ) -> None:
        """Write the fake runner, which each feature made afterwards starts.

        Args:
            fit_seconds: How long a fit takes.
            embed_seconds: How long an embedding takes.
            crash_at_an_embed_after_the_fit: Whether a runner that fitted a
                model exits during its first embedding.
            version: The VAME version that the runner reports.
            heartbeat_seconds: The interval of the runner's heartbeats.
        """
        settings = (
            f"FIT_SECONDS = {fit_seconds!r}\n"
            f"EMBED_SECONDS = {embed_seconds!r}\n"
            f"CRASH_AT_AN_EMBED_AFTER_THE_FIT = {crash_at_an_embed_after_the_fit!r}\n"
            f"SEQ_MEAN = {_SEQ_MEAN!r}\n"
        )
        program = write_fake_runner(
            self._directory,
            journal_source(self.journal.path) + settings + _HANDLERS,
            tool="vame",
            version=version,
            heartbeat_seconds=heartbeat_seconds,
            siblings=(SOURCE_ROOT / VAME_PROTOCOL,),
        )
        self._monkeypatch.setattr(vame_module, "VAME_RUNNER_SCRIPT", program)

    def feature(self, **overrides: object) -> VameFeature:
        params: dict[str, object] = {
            "pose": {"pose_n": len(_KEYPOINTS), "keypoint_names": list(_KEYPOINTS)},
            "centered_reference_keypoint": "nose",
            "orientation_reference_keypoint": "tail",
            "time_window": _TIME_WINDOW,
            "prediction_steps": _TIME_WINDOW,
            "zdims": _ZDIMS,
            "vame_python": sys.executable,
            **overrides,
        }
        feature = VameFeature(inputs=VameFeature.Inputs(("tracks",)), params=params)
        self._made.append(feature)
        return feature

    def close_all(self) -> None:
        for feature in self._made:
            feature.close()


@pytest.fixture
def fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> Iterator[_FakeVame]:
    """The fake runner, with the machine's rungs unset."""
    isolate_tool_location(monkeypatch, VAME_ENV, found={})
    runner = tmp_path / "runner"
    runner.mkdir()
    made = _FakeVame(runner, monkeypatch)
    yield made
    made.close_all()


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "dataset")
    add_tracks_variant(
        ds, _VARIANT, *_SEQUENCES, n_rows=_N_ROWS, keypoints=len(_KEYPOINTS)
    )
    return ds


def _check_latents(table: pd.DataFrame) -> None:
    """Assert that *table* contains the fake's latents, each on its window's center."""
    centers = window_centers([range(_N_ROWS)], _TIME_WINDOW)
    assert_latents_at_centers(table, _ZDIMS, {0: centers, 1: centers})
    np.testing.assert_array_equal(table["vame_z0"], table["frame"])
    assert (table["vame_z1"] == np.float32(_SEQ_MEAN)).all()


type _Builtins = (
    dict[str, _Builtins] | list[_Builtins] | str | bytes | int | float | bool | None
)
"""A value built of the types that a bundle may contain."""

_BUILTINS: Final[TypeAdapter[_Builtins]] = TypeAdapter(_Builtins)
"""Checks, without converting, that a value is a :data:`_Builtins`."""


class _BuiltinsOnly(pickle.Unpickler):
    """Unpickles a stream, and raises at a reference to a class or a function."""

    def find_class(self, module_name: str, global_name: str, /) -> NoReturn:
        msg = f"the bundle references {module_name}.{global_name}"
        raise pickle.UnpicklingError(msg)


def _read_bundle(ds: Dataset, result: Result) -> VameModelBundle:
    """Read the model bundle that *result*'s run wrote."""
    return VameModelBundle.model_validate(
        joblib.load(result_run_root(ds, result) / VAME_MODEL_FILE)
    )


def _keys(value: _Builtins) -> Iterator[str]:
    """Yield every key of every dictionary inside *value*."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


# --- fit, then apply -------------------------------------------------------


def test_a_fit_is_applied_in_the_run_that_fits_it(
    fake: _FakeVame, dataset: Dataset, scratch: Path
) -> None:
    """``save_state`` keeps the runner, which loads the bundle and embeds each entry."""
    result = run_feature(dataset, fake.feature())

    assert result.failed_entries == ()
    outputs = result_outputs(dataset, result)
    assert sorted(outputs) == sorted(_SEQUENCES)
    for table in outputs.values():
        _check_latents(table)
    (pid,) = fake.journal.starts()
    assert fake.journal.commands(pid) == ["fit", "load_model", "embed", "embed"]
    run_root = result_run_root(dataset, result)
    assert fake.journal.details("load_model", "path") == [
        str(run_root / VAME_MODEL_FILE)
    ]
    log = (run_root / VAME_RUNNER_LOG).read_text()
    assert "fit: 4 sessions" in log
    assert "embed: 2 sessions" in log
    assert_process_exited(pid, "run_feature did not close the feature")
    assert list(scratch.iterdir()) == [], "close left the exchange directory"


def test_an_embedding_finds_none_of_the_fits_files(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """The exchange directory then contains the embedding's directory alone."""
    _ = run_feature(dataset, fake.feature())

    for exchanged in fake.journal.details("embed", "exchanged"):
        assert isinstance(exchanged, list)
        (directory,) = exchanged
        assert str(directory).startswith("embed-")


def test_the_exchange_directory_is_made_in_scratch_dir_and_removed(
    fake: _FakeVame, dataset: Dataset, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    _ = run_feature(dataset, fake.feature(scratch_dir=str(elsewhere)))

    (exchange,) = fake.journal.details("fit", "exchange_dir")
    assert isinstance(exchange, str)
    assert Path(exchange).parent == elsewhere
    assert list(elsewhere.iterdir()) == [], "close left the exchange directory"


def test_the_fit_reports_its_stages_epochs_and_device_to_the_run_log(
    fake: _FakeVame, dataset: Dataset
) -> None:
    _ = run_feature(dataset, fake.feature())

    phases = [
        (str(event["phase"]), str(event["message"]))
        for event in latest_events(dataset, "phase")
    ]
    assert ("train", "training") in phases
    assert ("device", "fit on cpu") in phases
    assert ("device", "embed on cpu") in phases
    epochs = latest_events(dataset, "epoch")
    assert [event["epoch"] for event in epochs] == [0, 1]
    assert {event["total_epochs"] for event in epochs} == {499}
    assert epochs[0]["metrics"] == {"train_loss": 1.0}


def test_a_heartbeat_reaches_the_run_log_at_most_once_an_interval(
    fake: _FakeVame, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner beats every 0.05 s. The run log takes one beat each 0.5 s."""
    fake.program(fit_seconds=2.0, heartbeat_seconds=0.05)
    monkeypatch.setattr(vame_module, "HEARTBEAT_REPORT_SECONDS", 0.5)

    _ = run_feature(dataset, fake.feature())

    running = [
        event
        for event in latest_events(dataset, "phase")
        if str(event["message"]).startswith("running")
    ]
    assert 1 <= len(running) <= 5
    assert {event["phase"] for event in running} == {"train"}


def test_a_rerun_is_a_cache_hit_that_starts_no_runner(
    fake: _FakeVame, dataset: Dataset
) -> None:
    first = run_feature(dataset, fake.feature())

    again = run_feature(dataset, fake.feature())

    assert again.run_id == first.run_id
    assert again.cache_hit is True
    assert len(fake.journal.starts()) == 1


def test_sessions_are_fitted_in_the_order_of_entry_individual_and_frame(
    fake: _FakeVame, dataset: Dataset
) -> None:
    result = run_feature(dataset, fake.feature())

    sessions = _read_bundle(dataset, result).fit_report.sessions
    assert [
        (session.name, session.sequence, session.individual) for session in sessions
    ] == [
        ("s000000", "s1", "0"),
        ("s000001", "s1", "1"),
        ("s000002", "s2", "0"),
        ("s000003", "s2", "1"),
    ]
    assert fake.journal.details("fit", "sessions") == [
        ["s000000", "s000001", "s000002", "s000003"]
    ]


# --- the model bundle --------------------------------------------------------


def test_the_bundle_unpickles_to_builtins_without_a_volatile_key(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """The bundle loads in either environment, without a key that changes per fit."""
    result = run_feature(dataset, fake.feature())
    path = result_run_root(dataset, result) / VAME_MODEL_FILE

    with path.open("rb") as stream:
        loaded = _BUILTINS.validate_python(_BuiltinsOnly(stream).load(), strict=True)

    assert not set(_keys(loaded)) & _VOLATILE_KEYS
    bundle = VameModelBundle.model_validate(loaded)
    assert bundle.vame_version == VAME_VERSION
    assert bundle.device == "cpu"
    assert bundle.state_dict == b"weights:nose,left,right,tail"
    assert bundle.keypoints == list(_KEYPOINTS)
    assert bundle.seq_mean == _SEQ_MEAN
    assert bundle.fit_report.epochs == 2
    assert bundle.fit_report.straddling_fraction == 0.25


def test_a_torn_write_leaves_no_bundle(
    fake: _FakeVame, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_half(value: object, filename: Path) -> NoReturn:
        _ = Path(filename).write_bytes(pickle.dumps(value)[:20])
        msg = "No space left on device"
        raise OSError(msg)

    monkeypatch.setattr(joblib, "dump", write_half)

    with pytest.raises(OSError, match="No space left"):
        _ = run_feature(dataset, fake.feature())

    (params,) = dataset.get_root("features").glob("vame*/*/params.json")
    run_root = params.parent
    assert not (run_root / VAME_MODEL_FILE).exists()
    assert not list(run_root.glob(".vame_model*"))


def test_every_model_setting_is_a_param() -> None:
    """A bundle records each model setting from the param of the same name."""
    assert set(MODEL_SETTINGS) <= set(VameFeature.Params.model_fields)


def test_every_training_setting_is_a_param() -> None:
    renamed = {"project_random_state"}

    assert set(Training.model_fields) - renamed <= set(VameFeature.Params.model_fields)


# --- an apply-only run --------------------------------------------------------


def test_a_model_is_applied_by_a_run_that_does_not_fit(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """A training-only setting that differs from the model's is not read."""
    fitted = run_feature(dataset, fake.feature())

    applied = run_feature(
        dataset, fake.feature(model=model_reference(fitted), learning_rate=1e-3)
    )

    assert applied.run_id != fitted.run_id
    assert applied.failed_entries == ()
    outputs = result_outputs(dataset, applied)
    for name, table in result_outputs(dataset, fitted).items():
        pd.testing.assert_frame_equal(outputs[name], table)
    _, pid = fake.journal.starts()
    assert fake.journal.commands(pid) == ["load_model", "embed", "embed"]
    bundle = result_run_root(dataset, fitted) / VAME_MODEL_FILE
    assert fake.journal.details("load_model", "path")[-1] == str(bundle)
    assert not (result_run_root(dataset, applied) / VAME_MODEL_FILE).exists()


def test_an_apply_over_a_wider_scope_is_the_same_run_of_the_same_model(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """The entries in scope are where a model is applied, not what the run is."""
    fitted = run_feature(dataset, fake.feature(), scope=Scope(entries=[("", "s1")]))
    bundle = result_run_root(dataset, fitted) / VAME_MODEL_FILE
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    reference = model_reference(fitted)

    narrow = run_feature(
        dataset, fake.feature(model=reference), scope=Scope(entries=[("", "s1")])
    )
    wide = run_feature(dataset, fake.feature(model=reference))

    assert wide.run_id == narrow.run_id
    assert wide.run_id != fitted.run_id
    assert sorted(result_outputs(dataset, wide)) == sorted(_SEQUENCES)
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == digest
    assert not (result_run_root(dataset, wide) / VAME_MODEL_FILE).exists()


def test_a_fit_over_a_wider_scope_is_another_run(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """Without a model the entries in scope are the training set."""
    narrow = run_feature(dataset, fake.feature(), scope=Scope(entries=[("", "s1")]))
    wide = run_feature(dataset, fake.feature())

    assert wide.run_id != narrow.run_id


def test_a_model_is_refused_with_session_settings_other_than_its_own(
    fake: _FakeVame, dataset: Dataset
) -> None:
    fitted = run_feature(dataset, fake.feature())
    feature = fake.feature(
        model=model_reference(fitted), pose_confidence=0.5, max_gap_frames=10
    )

    with pytest.raises(ValueError, match="max_gap_frames.*pose_confidence"):
        _ = run_feature(dataset, feature)

    assert len(fake.journal.starts()) == 1


def test_a_model_that_does_not_resolve_is_refused_and_not_refitted(
    fake: _FakeVame, dataset: Dataset
) -> None:
    feature = fake.feature(model={"feature": "vame", "run_id": "0.1-0000000000"})

    with pytest.raises(FileNotFoundError, match="0.1-0000000000"):
        _ = run_feature(dataset, feature)

    assert fake.journal.starts() == []


def test_an_apply_without_an_interpreter_is_refused_before_any_entry(
    fake: _FakeVame, dataset: Dataset, tmp_path: Path
) -> None:
    """``load_state`` refuses the run once, without starting a runner."""
    fitted = run_feature(dataset, fake.feature())
    missing = tmp_path / "no-environment" / "bin" / "python"
    feature = fake.feature(model=model_reference(fitted), vame_python=str(missing))

    with pytest.raises(VameNotFoundError, match="the binary this run was placed with"):
        _ = run_feature(dataset, feature)

    assert len(fake.journal.starts()) == 1


def test_a_model_of_another_vame_version_is_refused(
    fake: _FakeVame, dataset: Dataset
) -> None:
    fitted = run_feature(dataset, fake.feature())
    path = result_run_root(dataset, fitted) / VAME_MODEL_FILE
    bundle = _read_bundle(dataset, fitted).model_copy(update={"vame_version": "0.13.0"})
    _ = joblib.dump(bundle.model_dump(), path)

    with pytest.raises(ValueError, match="0.13.0"):
        _ = run_feature(dataset, fake.feature(model=model_reference(fitted)))

    assert len(fake.journal.starts()) == 1


def test_a_runner_of_another_vame_version_is_refused(
    fake: _FakeVame, dataset: Dataset
) -> None:
    fake.program(version="0.0.0")

    with pytest.raises(RunnerVersionError, match="0.0.0"):
        _ = run_feature(dataset, fake.feature())

    (pid,) = fake.journal.starts()
    assert fake.journal.commands(pid) == []
    assert_process_exited(pid)


# --- exclusions and failed entries -------------------------------------------


def _excluded_messages(ds: Dataset) -> list[str]:
    """Return the message of each ``exclude`` phase of the run in *ds* that logged last."""
    return [
        str(event["message"])
        for event in latest_events(ds, "phase")
        if event["phase"] == "exclude"
    ]


def _drop_nose(ds: Dataset, individual: int, frames: range) -> None:
    """Make the nose of *individual* in ``s1`` missing at *frames*."""
    path = tracks_variant_root(ds.get_root("tracks"), _VARIANT) / "s1.parquet"
    table = pd.read_parquet(path)
    rows = (table["id"] == individual) & table["frame"].isin(frames)
    table.loc[rows, "poseX0"] = np.nan
    add_tracks_table(ds, _VARIANT, table)


def test_exclusions_are_reported_and_an_entry_without_a_latent_fails(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """``s3`` contains 3 frames, fewer than a window and than the fit's 11 frames."""
    add_tracks_variant(dataset, _VARIANT, "s3", n_rows=3, keypoints=len(_KEYPOINTS))

    result = run_feature(dataset, fake.feature())

    assert result.failed_entries == ("s3",)
    assert sorted(result_outputs(dataset, result)) == sorted(_SEQUENCES)
    excluded = _excluded_messages(dataset)
    assert excluded == [
        "fit, s3: frames 0 to 2 of id 0: 3 frames, fewer than 2 * time_window + 1 "
        "= 11, which one training window needs",
        "fit, s3: frames 0 to 2 of id 1: 3 frames, fewer than 2 * time_window + 1 "
        "= 11, which one training window needs",
        "embed, s3: frames 0 to 2 of id 0: 3 frames, fewer than time_window=5",
        "embed, s3: frames 0 to 2 of id 1: 3 frames, fewer than time_window=5",
    ]
    assert result.execution_id is not None
    (error,) = entry_error_lines(dataset, result.execution_id)
    assert "s3" in error
    assert "no row receives a latent" in error
    report = _read_bundle(dataset, result).fit_report
    assert report.excluded == [
        message.removeprefix("fit, ") for message in excluded[:2]
    ]
    assert report.gap_rows == {}


def test_rows_in_a_reference_gap_longer_than_max_gap_frames_are_reported(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """The nose of id 0 in ``s1`` is missing at frames 15 to 19, 5 frames."""
    _drop_nose(dataset, 0, range(15, 20))

    result = run_feature(dataset, fake.feature(max_gap_frames=3))

    assert result.failed_entries == ()
    gap = (
        "5 rows at frames where reference keypoint 'nose' or 'tail' lacks a finite "
        "point with confidence at or above pose_confidence=0.99, in a run longer "
        "than max_gap_frames=3 or at either end of an id's frames"
    )
    assert _excluded_messages(dataset) == [f"fit, s1: {gap}", f"embed, s1: {gap}"]
    report = _read_bundle(dataset, result).fit_report
    assert report.gap_rows == {"s1": 5}
    assert report.excluded == []
    split = window_centers([range(15), range(20, _N_ROWS)], _TIME_WINDOW)
    whole = window_centers([range(_N_ROWS)], _TIME_WINDOW)
    output = result_outputs(dataset, result)["s1"]
    assert_latents_at_centers(output, _ZDIMS, {0: split, 1: whole})


def test_a_fit_without_a_usable_session_is_refused(
    fake: _FakeVame, tmp_path: Path
) -> None:
    ds = make_dataset(tmp_path / "short")
    add_tracks_variant(ds, _VARIANT, "s1", n_rows=8, keypoints=len(_KEYPOINTS))

    with pytest.raises(ValueError, match=r"2 \* time_window \+ 1 = 11"):
        _ = run_feature(ds, fake.feature())

    assert fake.journal.starts() == []


def test_a_refused_fit_names_the_rows_outside_every_session(
    fake: _FakeVame, tmp_path: Path
) -> None:
    """The nose of id 0 is missing at frames 0 and 1, and each id is too short."""
    ds = make_dataset(tmp_path / "short")
    add_tracks_variant(ds, _VARIANT, "s1", n_rows=8, keypoints=len(_KEYPOINTS))
    _drop_nose(ds, 0, range(2))

    with pytest.raises(ValueError, match=r"of id 1: 8 frames, .*; s1: 2 rows at "):
        _ = run_feature(ds, fake.feature())


def test_a_refused_entry_is_named_by_the_fit_refusal(
    fake: _FakeVame, dataset: Dataset
) -> None:
    """Id 1 of ``s2`` keeps 3 rows, 19 frames apart, and the fit refuses ``s2``."""
    path = tracks_variant_root(dataset.get_root("tracks"), _VARIANT) / "s2.parquet"
    table = pd.read_parquet(path)
    kept = (table["id"] == 0) | table["frame"].isin((0, 19, 38))
    add_tracks_table(dataset, _VARIANT, table[kept])

    with pytest.raises(ValueError, match=r"^s2: vame: id 1 steps a median of 19 "):
        _ = run_feature(dataset, fake.feature())

    assert fake.journal.starts() == []


def test_a_crash_fails_its_entry_and_a_new_runner_embeds_the_next(
    fake: _FakeVame, dataset: Dataset
) -> None:
    fake.program(crash_at_an_embed_after_the_fit=True)

    result = run_feature(dataset, fake.feature())

    assert result.failed_entries == ("s1",)
    assert sorted(result_outputs(dataset, result)) == ["s2"]
    fitted, restarted = fake.journal.starts()
    assert fake.journal.commands(fitted) == ["fit", "load_model", "crashing"]
    assert fake.journal.commands(restarted) == ["load_model", "embed"]
    bundle = str(result_run_root(dataset, result) / VAME_MODEL_FILE)
    assert fake.journal.details("load_model", "path") == [bundle, bundle]


# --- cancel ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "fit_seconds", "embed_seconds"),
    [("fit", 60.0, 0.0), ("embed", 0.0, 60.0)],
    ids=["fit", "embed"],
)
def test_a_cancel_stops_the_runner_and_ends_the_run_cancelled(
    fake: _FakeVame,
    dataset: Dataset,
    command: str,
    fit_seconds: float,
    embed_seconds: float,
) -> None:
    fake.program(fit_seconds=fit_seconds, embed_seconds=embed_seconds)
    token = CancelToken()
    fired: list[float] = []
    canceller = cancel_when(lambda: fake.journal.recorded(command), token, fired)

    with pytest.raises(Cancelled):
        _ = run_feature(dataset, fake.feature(), cancel_token=token)

    canceller.join()
    assert time.monotonic() - fired[0] < 2.5
    assert latest_snapshot(dataset)["status"] == "cancelled"
    (pid,) = fake.journal.starts()
    assert_process_exited(pid)
    bundles = list(dataset.get_root("features").glob(f"vame*/*/{VAME_MODEL_FILE}"))
    assert bool(bundles) is (command == "embed")


# --- one instance, several runs, and close -----------------------------------


def test_one_instance_runs_twice(fake: _FakeVame, dataset: Dataset) -> None:
    """The second run fits its scope in a new runner."""
    feature = fake.feature()

    first = run_feature(dataset, feature, scope=Scope(entries=[("", "s1")]))
    second = run_feature(dataset, feature)

    assert first.run_id != second.run_id
    assert sorted(result_outputs(dataset, first)) == ["s1"]
    assert sorted(result_outputs(dataset, second)) == sorted(_SEQUENCES)
    one, two = fake.journal.starts()
    assert_process_exited(one)
    assert fake.journal.commands(one) == ["fit", "load_model", "embed"]
    assert fake.journal.commands(two) == ["fit", "load_model", "embed", "embed"]


def test_close_stops_the_runner_and_a_later_apply_starts_another(
    fake: _FakeVame, dataset: Dataset, scratch: Path
) -> None:
    run_root = result_run_root(dataset, run_feature(dataset, fake.feature()))
    tracks = tracks_variant_root(dataset.get_root("tracks"), _VARIANT)
    frame = pd.read_parquet(tracks / "s1.parquet")
    feature = fake.feature()

    assert feature.load_state(run_root, {}, {}) is True
    assert len(fake.journal.starts()) == 1, "load_state started a runner"
    first = feature.apply(frame)
    _, pid = fake.journal.starts()
    feature.close()
    feature.close()

    assert_process_exited(pid)
    assert list(scratch.iterdir()) == []
    log = (run_root / VAME_RUNNER_LOG).read_text()
    assert "fit: 4 sessions" in log
    assert log.count("embed: 2 sessions") == 3, "close did not append to the log"
    second = feature.apply(frame)
    _, _, restarted = fake.journal.starts()
    assert fake.journal.commands(restarted) == ["load_model", "embed"]
    pd.testing.assert_frame_equal(first, second)
