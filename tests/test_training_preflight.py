"""What `train-pose` and `train-points` refuse, and how a cancel is asked for.

Every refusal here is decided from a :class:`ProbeResponse` alone, with no
environment anywhere -- the same arrangement the inference refusals use, and for
the same reason: the runner reports and mosaic decides, so the messages can name
mosaic commands and mosaic's own documentation, and they are testable with no
Ultralytics installed at all.

The cancel predicate is here too. It is the one piece of this stage that has no
precedent in the repository: every other external tool answers a cancel with a
process-group kill, and training cannot, because Ultralytics loses the epoch it
is inside.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.tracking.common.cooperative_cancel import stop_then_kill
from mosaic.tracking.common.ultralytics_env import (
    ModelLoadError,
    PoloNotFoundError,
    UltralyticsNotFoundError,
    UnsupportedModelError,
)
from mosaic.tracking.external.runner.ultralytics_protocol import ProbeResponse
from mosaic.tracking.pose_training.ultralytics_train import (
    CANCEL_SENTINEL_NAME,
    attempt_directory,
    require_points_training_env,
    require_pose_training_env,
)

pytestmark = pytest.mark.tracker


def _probe(**overrides: object) -> ProbeResponse:
    """A healthy upstream probe, with the fields a case is about replaced."""
    fields: dict[str, object] = {
        "has_ultralytics": True,
        "has_lap": True,
        "has_locate": False,
        "ultralytics_version": "8.4.63",
        "tracker_names": ["botsort"],
        "model_task": "pose",
        "n_keypoints": 4,
        "model_load_error": "",
        "installed_tracker_table": {},
    }
    fields.update(overrides)
    return ProbeResponse.model_validate(fields)


# --- what training refuses -------------------------------------------------


def test_pose_training_needs_only_an_environment_when_it_starts_from_scratch() -> None:
    """The ordinary case: no local weights to check, so nothing about them is checked.

    A fresh run's model is a bare asset name Ultralytics resolves itself, so the
    probe was asked about the environment alone and reports an empty task. Reading
    that as "not a pose model" would refuse every first training run there is.
    """
    require_pose_training_env(_probe(model_task="", n_keypoints=0), "")


def test_pose_training_refuses_an_environment_with_no_ultralytics() -> None:
    with pytest.raises(UltralyticsNotFoundError, match="uv sync"):
        require_pose_training_env(_probe(has_ultralytics=False), "")


def test_pose_training_accepts_the_fork() -> None:
    """POLO keeps every upstream task, so refusing it would refuse what works."""
    require_pose_training_env(_probe(has_locate=True), "")


def test_pose_training_refuses_a_point_checkpoint_and_says_where_it_goes() -> None:
    with pytest.raises(UnsupportedModelError, match="train-points") as raised:
        require_pose_training_env(_probe(model_task="locate"), "weights.pt")
    assert "'locate' model" in str(raised.value)


def test_point_training_refuses_an_upstream_environment() -> None:
    """The check the in-process trainer could only make by importing the fork."""
    with pytest.raises(PoloNotFoundError, match="not the POLO fork"):
        require_points_training_env(_probe(has_locate=False), "")


def test_point_training_checks_the_fork_before_the_weights() -> None:
    """Order is the whole value: upstream cannot load a fork checkpoint at all.

    A user who never set MOSAIC_POLO_BIN reaches upstream through the ``$PATH``
    rung, and would otherwise be told only that their weights would not load --
    which is true, and says nothing about why.
    """
    with pytest.raises(PoloNotFoundError, match="not the POLO fork"):
        require_points_training_env(
            _probe(has_locate=False, model_load_error="AttributeError: nope"),
            "weights.pt",
        )


def test_point_training_refuses_a_pose_checkpoint() -> None:
    with pytest.raises(UnsupportedModelError, match="train-pose"):
        require_points_training_env(
            _probe(has_locate=True, model_task="pose"), "weights.pt"
        )


def test_base_weights_that_will_not_load_are_refused_by_name() -> None:
    with pytest.raises(ModelLoadError, match="could not be loaded"):
        require_pose_training_env(
            _probe(model_task="", model_load_error="RuntimeError: truncated"),
            "weights.pt",
        )


def test_a_healthy_fine_tune_is_accepted() -> None:
    require_pose_training_env(_probe(model_task="pose"), "weights.pt")
    require_points_training_env(
        _probe(has_locate=True, model_task="locate"), "weights.pt"
    )


# --- asking before killing -------------------------------------------------


def test_nothing_is_written_while_the_job_is_not_cancelled(tmp_path: Path) -> None:
    sentinel = tmp_path / CANCEL_SENTINEL_NAME
    check = stop_then_kill(lambda: False, sentinel, grace=60.0)

    assert check() is False
    assert not sentinel.exists()


def test_the_first_poll_after_a_cancel_asks_rather_than_kills(tmp_path: Path) -> None:
    """The whole point: the supervisor keeps waiting while the tool finishes."""
    sentinel = tmp_path / CANCEL_SENTINEL_NAME
    check = stop_then_kill(lambda: True, sentinel, grace=60.0)

    assert check() is False, "a fired token must not immediately mean 'kill'"
    assert sentinel.is_file(), "the tool is asked by the file appearing"


def test_the_kill_still_happens_once_the_grace_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool that ignores the sentinel is not immortal."""
    import mosaic.tracking.common.cooperative_cancel as cancel_module

    clock = [1000.0]
    monkeypatch.setattr(cancel_module.time, "monotonic", lambda: clock[0])
    sentinel = tmp_path / CANCEL_SENTINEL_NAME
    check = stop_then_kill(lambda: True, sentinel, grace=30.0)

    assert check() is False
    clock[0] += 29.0
    assert check() is False, "still inside the grace"
    clock[0] += 2.0
    assert check() is True, "past the grace, the kill that was always there"


def test_the_sentinel_is_written_once(tmp_path: Path) -> None:
    """Repeated polls must not rewrite it -- the tool may be reading it."""
    sentinel = tmp_path / CANCEL_SENTINEL_NAME
    check = stop_then_kill(lambda: True, sentinel, grace=60.0)

    _ = check()
    written = sentinel.stat().st_mtime_ns
    _ = check()
    assert sentinel.stat().st_mtime_ns == written


def test_a_sentinel_that_cannot_be_written_falls_back_to_the_kill(
    tmp_path: Path,
) -> None:
    """An unwritable run root is a reason to kill, never a reason to crash."""
    blocked = tmp_path / "not-a-directory"
    _ = blocked.write_text("this is a file")
    check = stop_then_kill(lambda: True, blocked / CANCEL_SENTINEL_NAME, grace=0.0)

    assert check() is False, "the first poll starts the clock either way"
    assert check() is True, "with a zero grace the next poll kills"


def test_each_attempt_gets_its_own_exchange_directory(tmp_path: Path) -> None:
    """Training reuses one run root, so a fixed name would leak between attempts.

    A sentinel left by a cancelled attempt would stop the next one at its first
    epoch boundary, and a stale response would be read as this attempt's answer by
    a tool that exited without writing one.
    """
    first = attempt_directory(tmp_path, "01JQ0000000000000000000000")
    second = attempt_directory(tmp_path, "01JQ1111111111111111111111")

    assert first != second
    assert first.parent == second.parent == tmp_path / ".mosaic-train"


# --- reporting an epoch back into the ledger -------------------------------


def test_every_epoch_is_reported_and_none_is_throttled_away() -> None:
    """An epoch is minutes, not milliseconds, so the per-batch throttle is wrong here.

    A two-epoch run finishing inside the throttle window would report its first
    epoch and swallow its second, and the ledger would say 1/2 for a finished
    model.
    """
    from mosaic.core.pipeline.job import CancelToken, JobContext
    from mosaic.core.pipeline.progress import NullProgressCallback
    from mosaic.tracking.common.ultralytics_env import training_activity

    reported: list[tuple[int, int, dict[str, float]]] = []

    class Recorder(NullProgressCallback):
        def on_epoch_end(
            self, epoch: int, total_epochs: int, metrics: dict[str, float]
        ) -> None:
            reported.append((epoch, total_epochs, dict(metrics)))

    ctx = JobContext(
        execution_id="01JQ0000000000000000000000",
        kind="train-pose",
        target="train-pose",
        run_log=None,
        progress=Recorder(),
        cancel_token=CancelToken(),
    )
    seen: list[str] = []
    on_line = training_activity(ctx, seen.append)

    for epoch in range(3):
        on_line(
            '{"event":"epoch","epoch":%d,"total_epochs":3,"metrics":{"loss":0.5}}'
            % epoch
        )
    on_line('{"event":"heartbeat"}')
    on_line("Ultralytics chatter that is not an event at all")

    assert [epoch for epoch, _total, _metrics in reported] == [0, 1, 2]
    assert len(seen) == 5, "liveness sees every line, event or not"


def test_the_epoch_count_survives_the_next_heartbeat() -> None:
    """``phase_activity`` heartbeats with no count, and the reduction takes the last.

    Without the epoch setting the count as it goes, the next bare heartbeat after
    an epoch resets the reduced progress to zero.
    """
    from mosaic.core.pipeline.job import CancelToken, JobContext
    from mosaic.core.pipeline.progress import NullProgressCallback
    from mosaic.tracking.common.ultralytics_env import training_activity

    ctx = JobContext(
        execution_id="01JQ0000000000000000000000",
        kind="train-pose",
        target="train-pose",
        run_log=None,
        progress=NullProgressCallback(),
        cancel_token=CancelToken(),
    )
    on_line = training_activity(ctx, lambda _line: None)
    on_line('{"event":"epoch","epoch":4,"total_epochs":9,"metrics":{}}')

    assert ctx.done == 5, "five epochs have finished, counting from zero"
