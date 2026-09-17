"""A one-shot training op holds its run root for as long as it trains.

:func:`~mosaic.tracking.ops._common.claim_run_root` returns its marker so the
caller can re-stamp it while the tool runs, and three ops called it as a bare
statement. A claim stamped once expires ``idle_timeout`` after it is taken; the
next execution along then reads the root as abandoned, clears it, and trains
into the directory the first run is still writing. For a nondeterministic
trainer that is a corrupt model rather than a slow one, and every real training
run outlives the window.

The three ops differ in where the activity signal comes from -- two read their
tool's output, the third trains in this process and has only its own progress
callbacks -- so each is driven here through its own seam. What every case
asserts is the same thing in the same way: let the claim go stale, fire one
activity signal, and require that a peer's reading of the root changes from
*expired* back to *live*.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
import yaml

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.job import CancelToken, JobContext
from mosaic.core.pipeline.markers import (
    InflightState,
    inflight_state,
    new_inflight,
    read_inflight,
    write_inflight,
)
from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.progress import (
    NullProgressCallback,
    TrainingProgressCallback,
)
from mosaic.tracking import register_ops
from mosaic.tracking.ops._common import RunRootHeld, claim_run_root

from tests.helpers import make_dataset

register_ops()

_PEER = "01JPEER000000000000000000"
"""A second execution, asking about a root it does not hold."""

_LONG_AGO = "2020-01-01T00:00:00+00:00"
"""An expiry far enough in the past that no clock skew reaches it."""


def _peer_reads(ds: Dataset, run_root: Path) -> InflightState:
    """How another execution would classify the claim on *run_root* right now."""
    return inflight_state(
        read_inflight(run_root), run_log_base=ds.base_dir, execution_id=_PEER
    )


def _peer_context(kind: str) -> JobContext:
    """A second execution's context, with no run-log of its own.

    Untracked on purpose: an absent run-log is not evidence of anything, so the
    claim it meets falls through to its expiry -- which is the case the expiry
    exists for, and the one being tested.
    """
    return JobContext(
        execution_id=_PEER,
        kind=kind,
        target=kind,
        run_log=None,
        progress=NullProgressCallback(),
        cancel_token=CancelToken(),
    )


def _go_stale(ds: Dataset, run_root: Path) -> None:
    """Age the claim on *run_root* to the point a peer would take it.

    Only ``expires_at`` moves, so the marker still names its holder. This is the
    state a real run reaches by outliving ``idle_timeout``, reached here without
    a clock.
    """
    held = read_inflight(run_root)
    assert held is not None, "the op must have claimed its run root"
    write_inflight(run_root, held.model_copy(update={"expires_at": _LONG_AGO}))
    assert _peer_reads(ds, run_root) == "expired"


def _the_signal_restores_the_claim(
    ds: Dataset, run_root: Path, fire: Callable[[], None]
) -> None:
    """Let the claim lapse, fire one activity signal, and require it back."""
    _go_stale(ds, run_root)
    fire()
    assert _peer_reads(ds, run_root) == "live", (
        "one activity signal must re-stamp the claim the op is holding"
    )


# --- the hazard, with nothing defending against it -------------------------


def test_a_lapsed_claim_is_taken_by_the_next_execution(tmp_path: Path) -> None:
    """The control the rest of this file is measured against.

    Without it every assertion below could hold for the wrong reason -- a peer
    that can never take a root proves nothing about a refresh that stops it.
    """
    ds = make_dataset(tmp_path, save=False)
    run_root = tmp_path / "models" / "train-sleap" / "run"
    run_root.mkdir(parents=True)
    write_inflight(
        run_root,
        new_inflight(
            execution_id="01JHOLDER0000000000000000",
            host="otherhost",
            pid=4242,
            phase=None,
            idle_seconds=60.0,
        ),
    )
    _go_stale(ds, run_root)

    taken = claim_run_root(
        ds, _peer_context("train-sleap"), run_root, "train-sleap", 60
    )

    assert taken.execution_id == _PEER, "a lapsed claim is cleared and taken"
    held = read_inflight(run_root)
    assert held is not None and held.execution_id == _PEER


# --- the two ops whose trainer is a subprocess -----------------------------


def _point_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *tools: str) -> None:
    """Put a stub console script where the tool's location ladder will find it."""
    (tmp_path / "bin").mkdir(exist_ok=True)
    for name in (*tools, "python"):
        _ = (tmp_path / "bin" / name).write_text("")


def _fake_sleap_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the preflight probe without launching anything.

    ``train-sleap`` probes its environment before it claims the root, so a suite
    that leaves this out runs the empty stub :func:`_point_at` wrote.
    """
    import json

    from mosaic.tracking.sleap import probe as probe_module

    def run(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        answer: dict[str, object] = {
            "has_sleap_nn": True,
            "sleap_nn_version": "0.3.1",
            "sleap_io_version": "0.9.2",
            "n_labeled_frames": 12,
            "n_tracks": 2,
        }
        _ = Path(argv[argv.index("-c") + 3]).write_text(json.dumps(answer))
        return ("probed", "", 0)

    monkeypatch.setattr(probe_module, "run_supervised", run)


def _run_sleap(
    ds: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mid_run: Callable[[Path, Callable[[], None]], None],
) -> None:
    """Drive ``train-sleap`` to completion, calling *mid_run* inside the trainer.

    *mid_run* is handed the run root and a callable firing one line at whatever
    the op passed as the activity callback -- which is the thing under test, so
    its absence is the failure rather than a skipped assertion.
    """
    from mosaic.tracking.sleap import training as training_module

    _point_at(tmp_path, monkeypatch, "sleap-nn-train")
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-nn-train"))
    _fake_sleap_probe(monkeypatch)

    def run(
        argv: Sequence[str],
        *,
        on_activity: Callable[[str], None] | None = None,
        **kw: object,
    ) -> tuple[str, str, int]:
        run_root = Path(argv[argv.index("--config-dir") + 1])
        config = yaml.safe_load((run_root / "config.yaml").read_text())
        produced = run_root / config["trainer_config"]["run_name"]
        produced.mkdir(parents=True, exist_ok=True)
        _ = (produced / "best.ckpt").write_bytes(b"weights")
        _ = (produced / "training_config.yaml").write_text(
            "head_configs:\n  centroid: {}\n"
        )

        assert on_activity is not None, (
            "the op must hand the trainer its liveness callback"
        )
        fire = on_activity
        mid_run(run_root, lambda: fire("Epoch 1: 40%"))
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", run)

    labels = tmp_path / "session.slp"
    _ = labels.write_bytes(b"slp")
    _ = run_op(
        ds,
        "train-sleap",
        {"labels": str(labels), "max_epochs": 2, "head": "centroid"},
    )


def _run_litpose(
    ds: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mid_run: Callable[[Path, Callable[[], None]], None],
) -> None:
    """Drive ``train-litpose`` to completion, calling *mid_run* inside the trainer."""
    from mosaic.tracking.litpose import training as training_module

    _point_at(tmp_path, monkeypatch, "litpose")
    monkeypatch.setenv("MOSAIC_LITPOSE_BIN", str(tmp_path / "bin" / "litpose"))

    def run(
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        on_activity: Callable[[str], None] | None = None,
        **kw: object,
    ) -> tuple[str, str, int]:
        run_root = Path(argv[argv.index("-c") + 4])
        checkpoints = run_root / "tb_logs" / "run" / "version_0" / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        _ = (run_root / "config.yaml").write_text("model:\n  model_type: heatmap\n")
        _ = (checkpoints / "best.ckpt").write_bytes(b"weights")

        assert on_activity is not None, (
            "the op must hand the trainer its liveness callback"
        )
        fire = on_activity
        mid_run(run_root, lambda: fire("Epoch 1: 40%"))
        return ("done", "", 0)

    monkeypatch.setattr(training_module, "run_supervised", run)

    project = tmp_path / "project"
    project.mkdir()
    _ = (project / "config.yaml").write_text("data:\n  num_keypoints: 3\n")
    base_config = tmp_path / "litpose_default.yaml"
    _ = base_config.write_text("training:\n  num_gpus: 0\n")
    _ = run_op(
        ds,
        "train-litpose",
        {
            "project": str(project),
            "base_config": str(base_config),
            "max_epochs": 2,
        },
    )


def test_train_sleap_refreshes_the_root_it_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured case: a ten-hour run whose claim lapsed after thirty minutes."""
    ds = make_dataset(tmp_path, save=False)
    _run_sleap(
        ds,
        tmp_path,
        monkeypatch,
        lambda run_root, fire: _the_signal_restores_the_claim(ds, run_root, fire),
    )


def test_train_litpose_refreshes_the_root_it_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path, save=False)
    _run_litpose(
        ds,
        tmp_path,
        monkeypatch,
        lambda run_root, fire: _the_signal_restores_the_claim(ds, run_root, fire),
    )


def test_a_refreshing_run_refuses_the_peer_that_would_have_stolen_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point, stated as the outcome rather than as the marker's contents.

    Two trainers writing one run root interleave its artifacts, so the second
    execution has to be refused while the first is still going -- which it can
    only be if the first has kept its claim alive.
    """
    ds = make_dataset(tmp_path, save=False)

    def peer_tries_to_take_it(run_root: Path, fire: Callable[[], None]) -> None:
        _the_signal_restores_the_claim(ds, run_root, fire)
        with pytest.raises(RunRootHeld):
            _ = claim_run_root(
                ds, _peer_context("train-sleap"), run_root, "train-sleap", 60
            )

    _run_sleap(ds, tmp_path, monkeypatch, peer_tries_to_take_it)


# --- the op whose trainer is in this process -------------------------------


def test_train_localizer_refreshes_the_root_it_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No output line to hang a refresh on, so its progress callbacks are the signal.

    The op composes a claim refresher beside ``ctx.progress``, so the trainer
    calling either object is proof it is alive.
    """
    import mosaic.tracking.pose_training.localizer_train as localizer

    ds = make_dataset(tmp_path, save=False)
    dataset_dir = tmp_path / "patches"
    (dataset_dir / "train").mkdir(parents=True)
    _ = (dataset_dir / "train" / "patches.npy").write_bytes(b"patches")

    def fake_train_localizer(
        dataset_dir: str | Path,
        *,
        project: str | Path,
        name: str,
        callback: TrainingProgressCallback | None = None,
        **kw: object,
    ) -> localizer.TrainingResult:
        run_root = Path(project)
        assert callback is not None, "the op must hand the trainer a progress callback"
        _go_stale(ds, run_root)
        callback.on_epoch_end(0, 2, {})
        assert _peer_reads(ds, run_root) == "live", (
            "an in-process trainer's own progress is what re-stamps its claim"
        )

        run_dir = run_root / name
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        weights = run_dir / "weights" / "best.pt"
        _ = weights.write_bytes(b"localizer weights")
        _ = (run_dir / "results.csv").write_text("epoch,loss\n0,0.1\n")
        return localizer.TrainingResult(
            best_model_path=weights,
            last_model_path=weights,
            run_dir=run_dir,
            best_epoch=0,
            best_val_loss=0.1,
        )

    monkeypatch.setattr(localizer, "train_localizer", fake_train_localizer)

    run_id = run_op(
        ds,
        "train-localizer",
        {"dataset_dir": str(dataset_dir), "epochs": 2, "device": "cpu"},
    )
    assert model_run_root(ds, "train-localizer", run_id).is_dir()


def test_the_refresh_does_not_outlive_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished run leaves a claim that expires on its own, not one held forever.

    ``claim_run_root`` has no release -- a holder whose run-log went terminal
    reads as orphaned -- so what a mid-run refresh must not do is push the
    expiry so far out that the reading never gets the chance.
    """
    ds = make_dataset(tmp_path, save=False)
    seen: list[Path] = []
    _run_sleap(
        ds,
        tmp_path,
        monkeypatch,
        lambda run_root, fire: (fire(), seen.append(run_root))[1],
    )

    run_root = seen[0]
    held = read_inflight(run_root)
    assert held is not None
    expiry = datetime.datetime.fromisoformat(held.expires_at)
    horizon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    assert expiry < horizon, "a refresh re-stamps the claim, it does not pin it open"
