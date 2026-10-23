"""KpmsFeature through ``run_feature``, against a fake keypoint-MoSeq runner.

The fake serves the commands of ``kpms_protocol`` under this interpreter. It
counts the frames of each track that it is sent, saves a model as JSON, and
labels each frame of an applied track. Every runner records its start and each
command in a journal, which the tests read. The license is accepted in-process.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.behavior.feature_library import kpms as kpms_module
from mosaic.behavior.feature_library.kpms import (
    KPMS_ENV,
    KPMS_LICENSE_ENV,
    KpmsFeature,
    KpmsNotFoundError,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.types import InputStream
from mosaic.core.scope import Scope
from tests.helpers import (
    KPMS_PROTOCOL,
    SOURCE_ROOT,
    RunnerJournal,
    add_tracks_variant,
    assert_process_exited,
    cancel_when,
    isolate_tool_location,
    journal_source,
    keypoint_rows,
    kpms_params,
    make_dataset,
    model_reference,
    result_outputs,
    result_run_root,
    wait_for,
    write_fake_runner,
)

_VARIANT: Final = "convert-test.0.1-aaaaaaaaaa"
"""The tracks variant of each test's dataset."""

_SEQUENCES: Final = ("s1", "s2")
"""The sequences of each test's dataset, each with two individuals."""

_N_ROWS: Final = 40
"""The frames of each sequence."""

_HANDLERS: Final = """
import json

import numpy as np

from kpms_protocol import AddTrack, Apply, Fit, LoadModel, SaveModel


TRACKS = {}
MODEL = {}
FITTED = False


def exit_soon():
    time.sleep(0.2)
    record("exiting")
    os._exit(0)


def add_track(payload, emit):
    request = AddTrack.model_validate(payload)
    with np.load(request.track) as track:
        TRACKS[request.key] = len(track["coords"])
    record("add_track", key=request.key)
    if EXIT_AFTER_A_TRACK:
        threading.Thread(target=exit_soon).start()


def fit(payload, emit):
    global FITTED
    request = Fit.model_validate(payload)
    record("fit", keys=sorted(TRACKS), bodyparts=request.config.bodyparts)
    print(f"fit: {len(TRACKS)} recordings")
    time.sleep(FIT_SECONDS)
    MODEL["keys"] = sorted(TRACKS)
    TRACKS.clear()
    FITTED = True


def load_model(payload, emit):
    request = LoadModel.model_validate(payload)
    MODEL.update(json.loads(Path(request.path).read_text()))
    record("load_model", path=request.path)


def apply(payload, emit):
    request = Apply.model_validate(payload)
    if not MODEL:
        raise RuntimeError("apply came before fit and load_model")
    if CRASH_AT_AN_APPLY_AFTER_THE_FIT and FITTED:
        record("crashing")
        os._exit(3)
    with np.load(request.track) as track:
        frames = len(track["coords"])
    labels = np.arange(frames, dtype=np.int32) % len(MODEL["keys"])
    np.savez(request.syllables, syllables=labels)
    record("apply", key=request.key)


def save_model(payload, emit):
    request = SaveModel.model_validate(payload)
    text = json.dumps(MODEL)
    Path(request.path).write_text(text[: len(text) // 2])
    record("saving")
    time.sleep(SAVE_SECONDS)
    Path(request.path).write_text(text)
    record("save_model", path=request.path)


DISPATCH = {
    AddTrack.command: add_track,
    Fit.command: fit,
    LoadModel.command: load_model,
    Apply.command: apply,
    SaveModel.command: save_model,
}
"""


class _FakeKpms:
    """The fake runner of one test, the features that it serves, and its journal."""

    def __init__(self, directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._directory = directory
        self._monkeypatch = monkeypatch
        self.journal = RunnerJournal(directory / "journal.jsonl")
        self._made: list[KpmsFeature] = []
        self.program(fit_seconds=0.0)

    def program(
        self,
        *,
        fit_seconds: float,
        save_seconds: float = 0.0,
        exit_after_a_track: bool = False,
        crash_at_an_apply_after_the_fit: bool = False,
    ) -> None:
        """Write the fake runner, which each feature made afterwards starts.

        Args:
            fit_seconds: How long a fit takes.
            save_seconds: How long a save leaves half of the model written.
            exit_after_a_track: Whether a runner exits 0.2 s after it is sent a
                track.
            crash_at_an_apply_after_the_fit: Whether a runner that fitted a
                model exits during its first apply.
        """
        settings = (
            f"FIT_SECONDS = {fit_seconds!r}\n"
            f"SAVE_SECONDS = {save_seconds!r}\n"
            f"EXIT_AFTER_A_TRACK = {exit_after_a_track!r}\n"
            f"CRASH_AT_AN_APPLY_AFTER_THE_FIT = {crash_at_an_apply_after_the_fit!r}\n"
        )
        program = write_fake_runner(
            self._directory,
            journal_source(self.journal.path) + settings + _HANDLERS,
            tool="keypoint-moseq",
            siblings=(SOURCE_ROOT / KPMS_PROTOCOL,),
        )
        self._monkeypatch.setattr(kpms_module, "KPMS_SERVER_SCRIPT", program)

    def feature(self, **overrides: object) -> KpmsFeature:
        params = kpms_params(**{"kpms_python": sys.executable, **overrides})
        feature = KpmsFeature(inputs=KpmsFeature.Inputs(("tracks",)), params=params)
        self._made.append(feature)
        return feature

    def close_all(self) -> None:
        for feature in self._made:
            feature.close()


@pytest.fixture
def fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> Iterator[_FakeKpms]:
    """The fake runner, with the license accepted and the machine's rungs unset."""
    monkeypatch.setenv(KPMS_LICENSE_ENV, "1")
    isolate_tool_location(monkeypatch, KPMS_ENV, found={})
    runner = tmp_path / "runner"
    runner.mkdir()
    made = _FakeKpms(runner, monkeypatch)
    yield made
    made.close_all()


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "dataset")
    add_tracks_variant(ds, _VARIANT, *_SEQUENCES, n_rows=_N_ROWS, keypoints=3)
    return ds


# --- fit, then apply ---------------------------------------------------------


def test_a_fit_is_applied_in_the_run_that_fits_it(
    fake: _FakeKpms, dataset: Dataset, scratch: Path
) -> None:
    """``save_state`` keeps the runner, with its model, for ``apply``."""
    result = run_feature(dataset, fake.feature())

    assert result.failed_entries == ()
    outputs = result_outputs(dataset, result)
    assert sorted(outputs) == sorted(_SEQUENCES)
    for table in outputs.values():
        assert len(table) == 2 * _N_ROWS
        assert set(table["syllable"]) == {0, 1, 2, 3}
    (pid,) = fake.journal.starts()
    assert fake.journal.commands(pid) == [
        *["add_track"] * 4,
        "fit",
        "saving",
        "save_model",
        *["apply"] * 4,
    ]
    run_root = result_run_root(dataset, result)
    assert (run_root / "kpms_model.joblib").is_file()
    assert "fit: 4 recordings" in (run_root / "kpms_server.log").read_text()
    assert_process_exited(pid, "run_feature did not close the feature")
    assert list(scratch.iterdir()) == [], "close left the exchange directory"


def test_a_rerun_is_a_cache_hit_that_starts_no_runner(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    first = run_feature(dataset, fake.feature())

    again = run_feature(dataset, fake.feature())

    assert again.run_id == first.run_id
    assert again.cache_hit is True
    assert len(fake.journal.starts()) == 1


def test_a_model_is_applied_by_a_run_that_does_not_fit(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    fitted = run_feature(dataset, fake.feature())
    model = result_run_root(dataset, fitted) / "kpms_model.joblib"

    applied = run_feature(dataset, fake.feature(model=model_reference(fitted)))

    assert applied.run_id != fitted.run_id
    assert applied.failed_entries == ()
    assert sorted(result_outputs(dataset, applied)) == sorted(_SEQUENCES)
    _, pid = fake.journal.starts()
    assert fake.journal.commands(pid) == ["load_model", *["apply"] * 4]
    assert fake.journal.details("load_model", "path") == [str(model)]


def test_an_apply_over_a_wider_scope_is_the_same_run_of_the_same_model(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    """The entries in scope are where a model is applied, not what the run is."""
    fitted = run_feature(dataset, fake.feature(), scope=Scope(entries=[("", "s1")]))
    model = result_run_root(dataset, fitted) / "kpms_model.joblib"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    reference = model_reference(fitted)

    narrow = run_feature(
        dataset, fake.feature(model=reference), scope=Scope(entries=[("", "s1")])
    )
    wide = run_feature(dataset, fake.feature(model=reference))

    assert wide.run_id == narrow.run_id
    assert wide.run_id != fitted.run_id
    assert sorted(result_outputs(dataset, wide)) == sorted(_SEQUENCES)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == digest
    assert set(fake.journal.details("load_model", "path")) == {str(model)}
    fits = [fake.journal.commands(pid).count("fit") for pid in fake.journal.starts()]
    assert sum(fits) == 1, "only the first run fits"


def test_a_fit_over_a_wider_scope_is_another_run(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    """Without a model the entries in scope are the training set."""
    narrow = run_feature(dataset, fake.feature(), scope=Scope(entries=[("", "s1")]))
    wide = run_feature(dataset, fake.feature())

    assert wide.run_id != narrow.run_id


def test_a_model_that_does_not_resolve_is_refused_and_not_refitted(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    feature = fake.feature(model={"feature": "kpms", "run_id": "0.1-0000000000"})

    with pytest.raises(FileNotFoundError, match="0.1-0000000000"):
        _ = run_feature(dataset, feature)

    assert fake.journal.starts() == []


def test_a_runner_that_crashes_after_the_fit_is_replaced_with_the_saved_model(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    """The runner started in place of the one that fitted loads ``save_state``'s model.

    The crash fails the entry being applied, and the later entry is applied.
    """
    fake.program(fit_seconds=0.0, crash_at_an_apply_after_the_fit=True)

    result = run_feature(dataset, fake.feature())

    assert result.failed_entries == ("s1",)
    assert sorted(result_outputs(dataset, result)) == ["s2"]
    fitted, restarted = fake.journal.starts()
    assert fake.journal.commands(fitted)[-4:] == [
        "fit",
        "saving",
        "save_model",
        "crashing",
    ]
    assert fake.journal.commands(restarted) == ["load_model", "apply", "apply"]
    assert fake.journal.details("load_model", "path") == [
        str(result_run_root(dataset, result) / "kpms_model.joblib")
    ]


# --- refusals before any entry -------------------------------------------------


def test_an_apply_without_the_license_is_refused_before_any_entry(
    fake: _FakeKpms, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_state`` refuses the run once, without starting a runner."""
    fitted = run_feature(dataset, fake.feature())
    monkeypatch.delenv(KPMS_LICENSE_ENV)

    with pytest.raises(RuntimeError, match="non-commercial"):
        _ = run_feature(dataset, fake.feature(model=model_reference(fitted)))

    assert len(fake.journal.starts()) == 1


def test_an_apply_without_an_interpreter_is_refused_before_any_entry(
    fake: _FakeKpms, dataset: Dataset, tmp_path: Path
) -> None:
    """``load_state`` refuses the run once, without starting a runner."""
    fitted = run_feature(dataset, fake.feature())
    missing = tmp_path / "no-environment" / "bin" / "python"
    feature = fake.feature(model=model_reference(fitted), kpms_python=str(missing))

    with pytest.raises(KpmsNotFoundError, match="the binary this run was placed with"):
        _ = run_feature(dataset, feature)

    assert len(fake.journal.starts()) == 1


# --- cancel and close ---------------------------------------------------------


def test_a_cancel_stops_a_fit_within_about_a_second(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    fake.program(fit_seconds=60.0)
    token = CancelToken()
    fired: list[float] = []
    canceller = cancel_when(lambda: fake.journal.recorded("fit"), token, fired)

    with pytest.raises(Cancelled):
        _ = run_feature(dataset, fake.feature(), cancel_token=token)

    canceller.join()
    assert time.monotonic() - fired[0] < 2.5
    (pid,) = fake.journal.starts()
    assert_process_exited(pid)


def test_a_cancel_during_the_save_leaves_no_model_and_the_next_run_fits(
    fake: _FakeKpms, dataset: Dataset
) -> None:
    """A model file in the run root is the cached model, and only a whole one is there."""
    fake.program(fit_seconds=0.0, save_seconds=60.0)
    token = CancelToken()
    canceller = cancel_when(lambda: fake.journal.recorded("saving"), token)

    with pytest.raises(Cancelled):
        _ = run_feature(dataset, fake.feature(), cancel_token=token)

    canceller.join()
    (params,) = dataset.get_root("features").glob("kpms*/*/params.json")
    run_root = params.parent
    assert not (run_root / "kpms_model.joblib").exists()
    assert not list(run_root.glob(".kpms_model*"))
    fake.program(fit_seconds=0.0)

    result = run_feature(dataset, fake.feature())

    assert result.failed_entries == ()
    _, refitted = fake.journal.starts()
    assert fake.journal.commands(refitted)[:6] == [
        *["add_track"] * 4,
        "fit",
        "saving",
    ]
    assert (run_root / "kpms_model.joblib").is_file()


def test_close_stops_the_runner_and_a_later_request_starts_another(
    fake: _FakeKpms, dataset: Dataset, scratch: Path
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
    second = feature.apply(frame)
    _, _, restarted = fake.journal.starts()
    assert fake.journal.commands(restarted) == ["load_model", "apply", "apply"]
    pd.testing.assert_frame_equal(first, second)


def test_a_runner_that_exits_holding_the_tracks_fails_the_fit(
    fake: _FakeKpms,
) -> None:
    """The runner started in place of one with the sent tracks lacks them."""
    fake.program(fit_seconds=0.0, exit_after_a_track=True)
    feature = fake.feature()
    rows = keypoint_rows(range(_N_ROWS), n_keypoints=3)

    def entries() -> Iterator[tuple[str, pd.DataFrame]]:
        yield "s1", rows
        wait_for(
            lambda: fake.journal.recorded("exiting"),
            60.0,
            "the runner did not exit",
        )
        time.sleep(0.5)
        yield "s2", rows

    with pytest.raises(RuntimeError, match="the tracks sent for the fit"):
        feature.fit(InputStream(entries, n_entries=2))

    assert len(fake.journal.starts()) == 2
