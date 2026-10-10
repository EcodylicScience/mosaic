"""VameFeature through ``run_feature``, against VAME in a separate environment.

The tests skip unless :data:`VAME_ENV` locates an interpreter: the environment
built under ``external/vame-env/``, or the one that ``MOSAIC_VAME_PYTHON`` or
``MOSAIC_VAME_CONDA_ENV`` names. They train and embed on the CPU, where VAME's
results are bit-reproducible.

The feature starts an observed runner, a program that runs the VAME runner after
recording its process id in a journal. It also saves each array that VAME's
embedding passes to the encoder.

The synthetic tracks contain five entries in two groups, each with two individuals
whose six keypoints move as one undulating body. One individual is lost for
longer than ``max_gap_frames``. Another returns for fewer frames than one window.
Every tracklet of the entry ``g2/brief`` is shorter than one window.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library import vame_feature as vame_module
from mosaic.behavior.feature_library.external.vame_protocol import first_saving_epoch
from mosaic.behavior.feature_library.extract_templates import ExtractTemplates
from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering
from mosaic.behavior.feature_library.vame_feature import (
    VAME_ENV,
    VAME_MODEL_FILE,
    VAME_RUNNER_LOG,
    VAME_RUNNER_SCRIPT,
    VameFeature,
    VameModelBundle,
)
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.graph import (
    TRACKS_INPUT,
    ColumnSet,
    Recipe,
    resolved_columns,
)
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.progress import NullProgressCallback
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.toolenv import missing_interpreter
from mosaic.core.scope import Scope
from tests.helpers import (
    RunnerJournal,
    add_tracks_table,
    assert_latents_at_centers,
    assert_process_exited,
    assert_writes_declared_columns,
    entry_error_lines,
    journal_source,
    latest_events,
    make_dataset,
    model_reference,
    result_outputs,
    result_run_root,
    wait_for,
    window_centers,
)

_MISSING: Final = missing_interpreter(VAME_ENV)
"""How to provide VAME's environment, or ``None`` when it resolves."""

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _MISSING is not None, reason=f"VAME's environment does not resolve: {_MISSING}"
    ),
]

_VARIANT: Final = "convert-synthetic.0.1-aaaaaaaaaa"
"""The tracks variant of the synthetic tracks."""

_KEYPOINTS: Final = ("snout", "head", "neck", "center", "hip", "tailbase")

_BODY: Final = (30.0, 20.0, 10.0, 0.0, -10.0, -25.0)
"""Each keypoint's distance ahead of the body's center along its axis, in pixels."""

_CENTER: Final = _KEYPOINTS.index("center")

_TIME_WINDOW: Final = 10

_ZDIMS: Final = 4

_ENCODED_FEATURES: Final = 2 * len(_KEYPOINTS) - 3
"""The values of a frame that the encoder reads.

VAME drops both coordinates of the centered reference keypoint and one of the
orientation reference keypoint.
"""

_TRACKLETS: Final[dict[Entry, dict[int, tuple[range, ...]]]] = {
    # Individual 0 is lost for 40 frames, more than max_gap_frames, and its
    # frames form two sessions.
    ("g1", "a"): {0: (range(120), range(160, 300)), 1: (range(300),)},
    # Individual 1 returns for 7 frames, fewer than one window.
    ("g1", "b"): {0: (range(300),), 1: (range(200), range(260, 267))},
    ("g2", "c"): {0: (range(300),), 1: (range(300),)},
    ("g2", "brief"): {0: (range(8),), 1: (range(8),)},
    ("g2", "held_out"): {0: (range(200),), 1: (range(200),)},
}
"""Each individual's tracklets in each entry. Each tracklet is one session."""

# The fit's scope leaves out g2/brief, whose apply fails. A failed entry has no
# output, and every later run of the scope applies it again. A rerun of the fit
# is then not a cache hit.
_FITTED: Final = (("g1", "a"), ("g1", "b"), ("g2", "c"))
"""The entries that the fit trains on."""

_APPLIED: Final = (("g1", "a"), ("g2", "held_out"), ("g2", "brief"))
"""The entries of the run that applies the fitted model.

One entry is in the fit's scope, one is outside it, and every session of the
third is excluded.
"""

_CHUNK_FRAMES: Final = 25
"""Chunks of this many frames split each session of the applied entries."""

_PARAMS: Final[dict[str, object]] = {
    "pose": {"pose_n": len(_KEYPOINTS), "keypoint_names": list(_KEYPOINTS)},
    "centered_reference_keypoint": "center",
    "orientation_reference_keypoint": "snout",
    "device": "cpu",
    "zdims": _ZDIMS,
    "time_window": _TIME_WINDOW,
    "prediction_steps": _TIME_WINDOW // 2,
    "hidden_size_layer_1": 32,
    "hidden_size_rec": 32,
    "hidden_size_pred": 32,
    "batch_size": 16,
    "kl_start": 0,
    "annealtime": 1,
    # VAME runs epochs 1 to max_epochs - 1, and saves a model from the first
    # epoch whose KL weight exceeds 0.99.
    "max_epochs": first_saving_epoch(0, 1, "linear") + 1,
    "steps_per_epoch": 4,
}

_OBSERVED_RUNNER: Final = '''\
"""The VAME runner, recording its process id and each array that VAME encodes."""

import functools
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, {runners!r})

{journal}

import vame_runner

ENCODED = Path({encoded!r})
COUNT = itertools.count()
load_libraries = vame_runner.load_libraries


@functools.cache
def load_and_observe():
    """Load VAME, and save each array that its embedding passes to the encoder."""
    libraries = load_libraries()
    import numpy as np
    from vame.analysis import pose_segmentation

    format_for_rnn = pose_segmentation.format_xarray_for_rnn

    def observed(*arguments, **keywords):
        data, information = format_for_rnn(*arguments, **keywords)
        np.save(ENCODED / f"{{os.getpid()}}-{{next(COUNT):06d}}.npy", data)
        return data, information

    pose_segmentation.format_xarray_for_rnn = observed
    return libraries


vame_runner.load_libraries = load_and_observe

if __name__ == "__main__":
    vame_runner.main()
'''
"""The program of the observed runner, which imports the runner from its directory."""


@dataclass(frozen=True, slots=True)
class _ObservedRunner:
    """The journal of a test's runners, and the arrays that they encoded.

    Attributes:
        journal: The start of each runner.
        encoded: The directory of the arrays.
    """

    journal: RunnerJournal
    encoded: Path

    def arrays(self, pid: int) -> list[npt.NDArray[np.float64]]:
        """Return the arrays that the runner *pid* passed to the encoder, in order."""
        return [np.load(path) for path in sorted(self.encoded.glob(f"{pid}-*.npy"))]


@pytest.fixture
def observed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _ObservedRunner:
    """Return the observed runner, which each feature of the test starts."""
    directory = tmp_path / "runner"
    encoded = directory / "encoded"
    encoded.mkdir(parents=True)
    journal = RunnerJournal(directory / "journal.jsonl")
    program = directory / "observed_vame_runner.py"
    _ = program.write_text(
        _OBSERVED_RUNNER.format(
            runners=str(VAME_RUNNER_SCRIPT.parent),
            journal=journal_source(journal.path),
            encoded=str(encoded),
        )
    )
    monkeypatch.setattr(vame_module, "VAME_RUNNER_SCRIPT", program)
    return _ObservedRunner(journal, encoded)


def _individual(seed: int, frames: int) -> npt.NDArray[np.float64]:
    """Return one individual's keypoints, ``(frames, keypoints, 2)``, in pixels.

    The body walks at 2 pixels a frame along a randomly turning heading. Each
    keypoint is displaced sideways by a wave whose phase lags with the keypoint's
    distance behind the snout, and is measured with Gaussian noise of 0.5 pixels.
    """
    rng = np.random.default_rng(seed)
    heading = np.cumsum(rng.normal(0.0, 0.05, frames))
    axis = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    normal = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
    center = 400.0 + np.cumsum(2.0 * axis, axis=0)
    along = np.asarray(_BODY)
    phase = 2.0 * np.pi * np.arange(frames) / 25.0
    lateral = 4.0 * np.sin(phase[:, np.newaxis] - along[np.newaxis, :] / 15.0)
    position = (
        center[:, np.newaxis, :]
        + along[np.newaxis, :, np.newaxis] * axis[:, np.newaxis, :]
        + lateral[:, :, np.newaxis] * normal[:, np.newaxis, :]
    )
    return position + rng.normal(0.0, 0.5, position.shape)


def _entry_table(entry: Entry, seed: int) -> pd.DataFrame:
    """Return the track table of *entry*, whose individuals are seeded from *seed*."""
    group, sequence = entry
    tables: list[pd.DataFrame] = []
    for individual, tracklets in _TRACKLETS[entry].items():
        frames = np.concatenate([np.asarray(tracklet) for tracklet in tracklets])
        position = _individual(seed + individual, int(frames.max()) + 1)[frames]
        columns: dict[str, object] = {
            "frame": frames,
            "time": frames / 30.0,
            "id": np.full(len(frames), individual),
            "group": group,
            "sequence": sequence,
            "X": position[:, _CENTER, 0],
            "Y": position[:, _CENTER, 1],
        }
        for keypoint in range(len(_KEYPOINTS)):
            columns[f"poseX{keypoint}"] = position[:, keypoint, 0]
            columns[f"poseY{keypoint}"] = position[:, keypoint, 1]
            columns[f"poseP{keypoint}"] = np.ones(len(frames))
        tables.append(pd.DataFrame(columns))
    return pd.concat(tables, ignore_index=True)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "dataset")
    for index, entry in enumerate(_TRACKLETS):
        add_tracks_table(
            ds, _VARIANT, _entry_table(entry, seed=10 * index), std_format="mosaic_v1"
        )
    return ds


def _feature(**overrides: object) -> VameFeature:
    return VameFeature(
        inputs=VameFeature.Inputs(("tracks",)), params={**_PARAMS, **overrides}
    )


def _scope(entries: tuple[Entry, ...]) -> Scope:
    return Scope(entries=list(entries))


def _key(entry: Entry) -> str:
    return make_entry_key(*entry)


def _check_latents(table: pd.DataFrame, entry: Entry) -> None:
    """Assert that *table*'s rows are the center frames of *entry*'s sessions."""
    assert_latents_at_centers(
        table,
        _ZDIMS,
        {
            individual: centers
            for individual, tracklets in _TRACKLETS[entry].items()
            if (centers := window_centers(tracklets, _TIME_WINDOW))
        },
    )


def _declared_columns() -> ColumnSet:
    """Return the columns that validation predicts a vame step on the tracks writes."""
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "vame",
                    "type": "feature",
                    "feature": "vame",
                    "inputs": [TRACKS_INPUT],
                    "params": _PARAMS,
                }
            ]
        }
    )
    return resolved_columns(recipe)["vame"]


def _fit_sessions() -> list[tuple[str, str, str, int, int]]:
    """Return each session of the fit as entry, individual, first and last frame."""
    least = 2 * _TIME_WINDOW + 1
    return [
        (group, sequence, str(individual), tracklet.start, tracklet.stop - 1)
        for group, sequence in _FITTED
        for individual, tracklets in _TRACKLETS[group, sequence].items()
        for tracklet in tracklets
        if len(tracklet) >= least
    ]


def _embed_stages(ds: Dataset) -> list[tuple[int, int]]:
    """Return the chunks and the sessions of each embedding in the run that logged last.

    The runner reports both when an embedding starts. The stage's heartbeats are
    skipped.
    """
    stages: list[tuple[int, int]] = []
    for event in latest_events(ds, "phase"):
        message = str(event["message"])
        found = re.fullmatch(r"(\d+) chunks of (\d+) sessions", message)
        if event["phase"] == "embed" and found is not None:
            stages.append((int(found[1]), int(found[2])))
    return stages


def _group_is_empty(group: int) -> bool:
    """Return True when the process group *group* is empty."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return True
    return False


def _assert_runners_exited(journal: RunnerJournal) -> list[int]:
    """Assert that each runner, and each process of its group, has exited.

    Returns:
        The process id of each runner, in the order that they started.
    """
    pids = journal.starts()
    assert pids, "no runner started"
    for pid in pids:
        assert_process_exited(pid, f"the runner {pid} outlived its run")
        # A runner leads its process group, whose id is the runner's.
        wait_for(
            functools.partial(_group_is_empty, pid),
            10.0,
            f"a process of the runner {pid}'s group outlived the run",
        )
    return pids


class _ExchangedAtEmbedding(NullProgressCallback):
    """Lists the exchange directory when the first embedding reports its device.

    The exchange directory is the one directory in *scratch*, beside the runner's
    log file.

    Attributes:
        listed: The names in the exchange directory, or ``None`` before then.
    """

    def __init__(self, scratch: Path) -> None:
        self._scratch = scratch
        self.listed: list[str] | None = None

    @override
    def on_phase(self, phase: str, message: str) -> None:
        if self.listed is None and phase == "device" and message.startswith("embed"):
            self.listed = sorted(
                path.relative_to(self._scratch).as_posix()
                for path in self._scratch.glob("*/*")
            )


def test_a_fit_is_applied_rerun_and_applied_by_a_run_that_names_it(
    observed: _ObservedRunner, dataset: Dataset, scratch: Path
) -> None:
    exchanged = _ExchangedAtEmbedding(scratch)

    fitted = run_feature(
        dataset, _feature(), scope=_scope(_FITTED), progress_callback=exchanged
    )

    assert fitted.failed_entries == ()
    # The exchange directory then contains the embedding's directory alone.
    assert exchanged.listed is not None
    (embedding,) = exchanged.listed
    assert embedding.split("/")[1].startswith("embed-"), embedding
    fitted_outputs = result_outputs(dataset, fitted)
    assert sorted(fitted_outputs) == sorted(_key(entry) for entry in _FITTED)
    declared = _declared_columns()
    for entry in _FITTED:
        _check_latents(fitted_outputs[_key(entry)], entry)
        assert_writes_declared_columns("vame", declared, fitted_outputs[_key(entry)])
    run_root = result_run_root(dataset, fitted)
    bundle = VameModelBundle.model_validate(joblib.load(run_root / VAME_MODEL_FILE))
    assert bundle.device == "cpu"
    report = bundle.fit_report
    assert [
        (
            session.group,
            session.sequence,
            session.individual,
            session.first_frame,
            session.last_frame,
        )
        for session in report.sessions
    ] == _fit_sessions()
    (excluded,) = report.excluded
    assert excluded.startswith("g1__b: frames 260 to 266 of id 1: 7 frames")
    # A lost individual lacks rows at the frames that split its sessions.
    assert report.gap_rows == {}
    log = (run_root / VAME_RUNNER_LOG).read_text()
    assert "device cpu" in log, "save_state did not write the fit's log"
    assert VAME_MODEL_FILE in log, "close did not append the embedding's log"
    (runner,) = _assert_runners_exited(observed.journal)
    assert list(scratch.iterdir()) == [], "close left the exchange directory"

    # The encoder reads each training session normalized by the training set's
    # mean and standard deviation.
    encoded = observed.arrays(runner)
    assert len(encoded) == len(report.sessions)
    assert {array.shape[0] for array in encoded} == {_ENCODED_FEATURES}
    values = np.concatenate(encoded, axis=1)
    assert abs(float(values.mean())) < 0.1
    assert abs(float(values.std()) - 1.0) < 0.1

    again = run_feature(dataset, _feature(), scope=_scope(_FITTED))

    assert again.cache_hit is True
    assert again.run_id == fitted.run_id
    assert observed.journal.starts() == [runner]

    applied = run_feature(
        dataset, _feature(model=model_reference(fitted)), scope=_scope(_APPLIED)
    )

    assert applied.run_id != fitted.run_id
    assert applied.failed_entries == (_key(("g2", "brief")),)
    assert applied.execution_id is not None
    (error,) = entry_error_lines(dataset, applied.execution_id)
    assert _key(("g2", "brief")) in error
    assert "no row receives a latent" in error
    unchunked = result_outputs(dataset, applied)
    assert sorted(unchunked) == sorted(
        _key(entry) for entry in _APPLIED if entry != ("g2", "brief")
    )
    pd.testing.assert_frame_equal(
        unchunked[_key(("g1", "a"))],
        fitted_outputs[_key(("g1", "a"))],
        check_exact=True,
    )
    _check_latents(unchunked[_key(("g2", "held_out"))], ("g2", "held_out"))
    assert sorted(_embed_stages(dataset)) == [(2, 2), (3, 3)]
    assert not (result_run_root(dataset, applied) / VAME_MODEL_FILE).exists()

    # embed_chunk_frames leaves the run identifier unchanged, and overwrite
    # computes the run again.
    chunked = run_feature(
        dataset,
        _feature(model=model_reference(fitted), embed_chunk_frames=_CHUNK_FRAMES),
        overwrite=True,
        scope=_scope(_APPLIED),
    )

    assert chunked.run_id == applied.run_id
    stages = _embed_stages(dataset)
    assert len(stages) == 2
    assert all(chunks > sessions for chunks, sessions in stages)
    outputs = result_outputs(dataset, chunked)
    assert sorted(outputs) == sorted(unchunked)
    for name, table in outputs.items():
        pd.testing.assert_frame_equal(table, unchunked[name], rtol=0.0, atol=1e-5)
    _ = _assert_runners_exited(observed.journal)

    templates = run_feature(
        dataset,
        ExtractTemplates(
            inputs=ExtractTemplates.Inputs((fitted,)),
            params={"n_templates": 200},
        ),
    )
    clusters = run_feature(
        dataset,
        GlobalKMeansClustering(
            inputs=GlobalKMeansClustering.Inputs((fitted,)),
            params={
                "templates": {"feature": templates.feature, "run_id": templates.run_id},
                "k": 4,
            },
        ),
    )

    assert clusters.failed_entries == ()
    labeled = result_outputs(dataset, clusters)
    for name, table in fitted_outputs.items():
        assert len(labeled[name]) == len(table)
        assert labeled[name]["cluster"].between(0, 3).all()

    # The applied latents are placed in the fit's clusters by its model.
    other_clusters = run_feature(
        dataset,
        GlobalKMeansClustering(
            inputs=GlobalKMeansClustering.Inputs((applied,)),
            params={
                "model": GlobalKMeansClustering.ModelArtifact().from_result(clusters)
            },
        ),
    )

    assert other_clusters.failed_entries == ()
    other_labeled = result_outputs(dataset, other_clusters)
    pd.testing.assert_frame_equal(
        other_labeled[_key(("g1", "a"))], labeled[_key(("g1", "a"))]
    )
    assert other_labeled[_key(("g2", "held_out"))]["cluster"].between(0, 3).all()


class _CancelAtFirstEpoch(NullProgressCallback):
    """Fires a cancel token at the end of the first epoch.

    Attributes:
        epochs: Each epoch reported, from 0.
    """

    def __init__(self, token: CancelToken) -> None:
        self._token = token
        self.epochs: list[int] = []

    @override
    def on_epoch_end(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        self.epochs.append(epoch)
        self._token.cancel()


@pytest.mark.usefixtures("scratch")
def test_a_cancel_at_the_first_epoch_stops_the_fit_and_writes_no_bundle(
    observed: _ObservedRunner, dataset: Dataset
) -> None:
    """The fit may train for 9999 epochs, and the cancel stops it after the first."""
    token = CancelToken()
    progress = _CancelAtFirstEpoch(token)
    feature = _feature(max_epochs=10_000, model_convergence=10_000)

    with pytest.raises(Cancelled):
        _ = run_feature(
            dataset,
            feature,
            scope=_scope(_FITTED),
            progress_callback=progress,
            cancel_token=token,
        )

    assert progress.epochs[:1] == [0]
    bundles = dataset.get_root("features").glob(f"vame*/*/{VAME_MODEL_FILE}")
    assert list(bundles) == []
    _ = _assert_runners_exited(observed.journal)
