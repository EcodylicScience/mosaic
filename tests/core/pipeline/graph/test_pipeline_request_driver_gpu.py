"""The GPU card a request's driver leases to each GPU step.

Each step is a stand-in that writes what it sees of its card to a ready file and
runs until the test releases it, so a test can hold a step on its card, kill it,
or start a second driver beside it. ``step_lane`` is patched to put named steps on
a GPU lane, and a fake ``nvidia-smi`` lists the cards.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    DEFAULT_LANE,
    GPU_TRAIN_LANE,
    Recipe,
    Request,
    RequestRun,
    computes_path,
    driver_lease_path,
    load_computes,
    plan_pipeline,
    read_driver_lease,
    request_status_wire,
    run_pipeline,
    run_request,
    save_computes,
    submit_request,
)
from mosaic.core.pipeline import gpu_lease
from mosaic.core.pipeline.graph import run as graph_run
from mosaic.core.pipeline.gpu_lease import (
    GPU_CARD_VAR,
    LEASE_FDS_VAR,
    GpuCard,
    GpuCardsRefused,
    leasable_cards,
    try_lease,
)
from mosaic.core.pipeline.job import CancelToken
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    SPEED_OF_CONVERTED_STEP,
    TWO_CARDS,
    add_tracks_variant,
    calms21_dataset,
    classifier_recipe,
    converting_recipe,
    install_fake_nvidia_smi,
    make_dataset,
    wait_for,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows takes no GPU card lease"
)

VARIANT = "convert-trex.0.2-1111111111"

STEP_SECONDS = 60.0
"""The longest a test waits for a stand-in step to start or a card to change hands."""

FIRST_UUID = "GPU-11111111-1111-1111-1111-111111111111"

# A stand-in for a step. Its first argument is a directory; the step's argv
# follows. It writes what it sees of its card to <request>.<step>.ready in that
# directory, then runs until <request>.<step>.release or all.release exists.
STAND_IN = (
    "import json, os, sys, time\n"
    "directory, args = sys.argv[1], sys.argv[2:]\n"
    "request = args[args.index('--graph-request') + 1]\n"
    "name = request + '.' + args[args.index('--step') + 1]\n"
    "def lease_held():\n"
    "    try:\n"
    "        item = os.environ['MOSAIC_GPU_LEASE_FDS']\n"
    "        fd, device, inode = (int(part) for part in item.split(':'))\n"
    "        status = os.fstat(fd)\n"
    "    except (KeyError, ValueError, OSError):\n"
    "        return False\n"
    "    return (status.st_dev, status.st_ino) == (device, inode)\n"
    "keys = ('CUDA_VISIBLE_DEVICES', 'MOSAIC_GPU_CARD', 'MOSAIC_GPU_LEASE_FDS')\n"
    "seen = {key: os.environ.get(key) for key in keys}\n"
    "seen['lease_held'] = lease_held()\n"
    "seen['pid'] = os.getpid()\n"
    "path = os.path.join(directory, name)\n"
    "with open(path + '.part', 'w') as part:\n"
    "    json.dump(seen, part)\n"
    "os.replace(path + '.part', path + '.ready')\n"
    "release = (path + '.release', os.path.join(directory, 'all.release'))\n"
    "deadline = time.monotonic() + 120\n"
    "while not any(map(os.path.exists, release)) and time.monotonic() < deadline:\n"
    "    time.sleep(0.05)\n"
)

# Runs the steps named in its first argument as real mosaic steps, and every
# other step as the stand-in, whose arguments follow.
REAL_OR_STAND_IN = (
    "import os, sys\n"
    "real = sys.argv.pop(1).split(',')\n"
    "step_args = sys.argv[2:]\n"
    "if step_args[step_args.index('--step') + 1] in real:\n"
    "    command = [sys.executable, '-m', 'mosaic', *step_args]\n"
    "    os.execv(sys.executable, command)\n"
) + STAND_IN

TEMPLATES_OF_CONVERTED: dict[str, JsonValue] = {
    "id": "templates",
    "type": "feature",
    "feature": "extract-templates",
    "inputs": [{"step": "speed"}],
    "params": {"n_templates": 10},
}
"""A fit over the ``speed`` step of the tracks that the ``tracks`` step converts."""


def _speed_step(step_id: str, step_size: int) -> dict[str, JsonValue]:
    return {
        "id": step_id,
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": ["tracks"],
        "params": {"step_size": step_size},
    }


def _recipe(*step_ids: str) -> Recipe:
    """Independent ``speed-angvel`` steps, in the order given."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "independent steps",
            "steps": [
                _speed_step(step_id, size) for size, step_id in enumerate(step_ids, 1)
            ],
        }
    )


@dataclass
class StandIns:
    """The stand-in steps of one test and the files they write.

    Attributes:
        directory: Where the stand-ins write their ready files.
    """

    directory: Path

    def _path(self, request: Request, step_id: str, suffix: str) -> Path:
        return self.directory / f"{request.request_id}.{step_id}.{suffix}"

    def started(self, request: Request, step_id: str) -> bool:
        """Whether the step's stand-in has written its ready file."""
        return self._path(request, step_id, "ready").exists()

    def seen(self, request: Request, step_id: str) -> dict[str, JsonValue]:
        """What the step's stand-in saw, once it is ready."""
        wait_for(
            lambda: self.started(request, step_id),
            STEP_SECONDS,
            f"step {step_id} of {request.request_id} never started",
        )
        loaded: dict[str, JsonValue] = json.loads(
            self._path(request, step_id, "ready").read_text()
        )
        return loaded

    def release(self, request: Request, step_id: str) -> None:
        """Let the step's stand-in exit."""
        _ = self._path(request, step_id, "release").write_text("")

    def kill(self, request: Request, step_id: str) -> None:
        """Kill the step's stand-in with SIGKILL."""
        pid = self.seen(request, step_id)["pid"]
        assert isinstance(pid, int)
        os.kill(pid, signal.SIGKILL)

    def drive(
        self,
        ds: Dataset,
        request: Request,
        *,
        cancel_token: CancelToken | None = None,
        gpu_cards: tuple[str, ...] | None = None,
    ) -> Callable[[], RequestRun]:
        """Start ``run_request`` on a thread, and return what joins it."""
        outcome: list[RequestRun | BaseException] = []

        def target() -> None:
            try:
                outcome.append(
                    run_request(
                        ds, request, cancel_token=cancel_token, gpu_cards=gpu_cards
                    )
                )
            except BaseException as error:
                outcome.append(error)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        def joined() -> RequestRun:
            thread.join(STEP_SECONDS)
            assert not thread.is_alive(), "the driver did not end"
            [result] = outcome
            if isinstance(result, BaseException):
                raise result
            return result

        return joined


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


@pytest.fixture
def stand_ins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[StandIns]:
    """Every step a stand-in, released when the test ends, and short lease retries."""
    directory = tmp_path / "stand-ins"
    directory.mkdir()
    monkeypatch.setattr(
        graph_run, "MOSAIC_COMMAND", (sys.executable, "-c", STAND_IN, str(directory))
    )
    monkeypatch.setattr(gpu_lease, "LEASE_RETRY_SECONDS", 0.05)
    yield StandIns(directory)
    _ = (directory / "all.release").write_text("")


def _on_gpu(monkeypatch: pytest.MonkeyPatch, *gpu_steps: str) -> None:
    """Put *gpu_steps* on the GPU training lane and every other step on the CPU's."""

    def lane(recipe: Recipe, step_id: str) -> str:
        return GPU_TRAIN_LANE if step_id in gpu_steps else DEFAULT_LANE

    monkeypatch.setattr(graph_run, "step_lane", lane)


def _cards() -> tuple[GpuCard, ...]:
    return leasable_cards().cards


def _card_is_free(cards: tuple[GpuCard, ...]) -> bool:
    lease = try_lease(cards)
    if lease is None:
        return False
    lease.close()
    return True


def _status(ds: Dataset, request: Request, step_id: str) -> str:
    snapshot = read_run(run_log_dir(ds.base_dir), request.execution_of(step_id))
    return "" if snapshot is None else snapshot["status"]


def _gpu_wait_step(ds: Dataset, request: Request) -> str:
    lease = read_driver_lease(ds.base_dir, request.request_id)
    return "" if lease is None else lease.gpu_wait_step


def test_a_gpu_step_is_narrowed_to_its_card_and_a_cpu_step_keeps_the_drivers_cards(
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)
    _on_gpu(monkeypatch, "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv(GPU_CARD_VAR, "1:GPU-inherited")
    monkeypatch.setenv(LEASE_FDS_VAR, "99:1:2")
    request = submit_request(tracked, _recipe("gpu", "cpu")).request

    joined = stand_ins.drive(tracked, request)
    gpu = stand_ins.seen(request, "gpu")
    stand_ins.release(request, "gpu")
    cpu = stand_ins.seen(request, "cpu")
    stand_ins.release(request, "cpu")
    done = joined()

    assert (gpu["CUDA_VISIBLE_DEVICES"], gpu["MOSAIC_GPU_CARD"]) == (
        FIRST_UUID,
        f"0:{FIRST_UUID}",
    )
    assert gpu["lease_held"] is True, "the step holds the descriptor it was named"
    assert (cpu["CUDA_VISIBLE_DEVICES"], cpu["MOSAIC_GPU_CARD"]) == ("0,1", None)
    assert cpu["MOSAIC_GPU_LEASE_FDS"] is None
    assert done.ran == ("gpu", "cpu")


def test_the_driver_keeps_no_lock_once_its_step_has_started(
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killing the step frees its card while the driver goes on to the next step."""
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS[:1])
    _on_gpu(monkeypatch, "gpu")
    request = submit_request(tracked, _recipe("gpu", "cpu")).request
    cards = _cards()

    joined = stand_ins.drive(tracked, request)
    _ = stand_ins.seen(request, "gpu")
    assert not _card_is_free(cards), "the step holds its card"
    stand_ins.kill(request, "gpu")
    _ = stand_ins.seen(request, "cpu")

    wait_for(
        lambda: _card_is_free(cards),
        STEP_SECONDS,
        "the card stayed leased after its step was killed",
    )
    stand_ins.release(request, "cpu")
    _ = joined()


@pytest.mark.parametrize("first_ends", ["exits", "is killed"])
def test_a_second_driver_waits_for_the_card_until_the_first_step_ends(
    first_ends: str,
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS[:1])
    _on_gpu(monkeypatch, "gpu")
    first = submit_request(tracked, _recipe("gpu")).request
    second = submit_request(tracked, _recipe("gpu")).request

    first_joined = stand_ins.drive(tracked, first)
    _ = stand_ins.seen(first, "gpu")
    second_joined = stand_ins.drive(tracked, second)
    wait_for(
        lambda: _gpu_wait_step(tracked, second) == "gpu",
        STEP_SECONDS,
        "the second driver never waited for the card",
    )
    assert not stand_ins.started(second, "gpu")

    if first_ends == "exits":
        stand_ins.release(first, "gpu")
    else:
        stand_ins.kill(first, "gpu")
    seen = stand_ins.seen(second, "gpu")
    stand_ins.release(second, "gpu")
    _ = first_joined()
    done = second_joined()

    assert seen["MOSAIC_GPU_CARD"] == f"0:{FIRST_UUID}"
    assert _gpu_wait_step(tracked, second) == ""
    log = done.log_path.read_text()
    stamp = r"\[mosaic\] \S+ step gpu"
    assert re.search(rf"{stamp} waits for a GPU\n", log), log
    assert re.search(rf"{stamp} took GPU 0:{FIRST_UUID}\n", log), log
    assert log.index("waits for a GPU") < log.index("took GPU")


def test_two_drivers_run_at_once_on_two_cards(
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)
    _on_gpu(monkeypatch, "gpu")
    first = submit_request(tracked, _recipe("gpu")).request
    second = submit_request(tracked, _recipe("gpu")).request

    joins = [stand_ins.drive(tracked, first), stand_ins.drive(tracked, second)]
    cards = sorted(
        str(stand_ins.seen(request, "gpu")["MOSAIC_GPU_CARD"])
        for request in (first, second)
    )
    stand_ins.release(first, "gpu")
    stand_ins.release(second, "gpu")
    for joined in joins:
        _ = joined()

    assert cards == [f"0:{FIRST_UUID}", f"1:{TWO_CARDS[1].split(', ')[1]}"]


def test_a_cancel_while_waiting_for_a_card_records_the_step_cancelled(
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS[:1])
    _on_gpu(monkeypatch, "gpu")
    request = submit_request(tracked, _recipe("gpu")).request
    held = try_lease(_cards())
    assert held is not None
    token = CancelToken()
    try:
        joined = stand_ins.drive(tracked, request, cancel_token=token)
        wait_for(
            lambda: _gpu_wait_step(tracked, request) == "gpu",
            STEP_SECONDS,
            "the driver never waited for the card",
        )
        token.cancel()
        done = joined()
    finally:
        held.close()

    assert (done.status, done.ran, done.cancelled) == ("cancelled", (), ("gpu",))
    assert _status(tracked, request, "gpu") == "cancelled"
    assert not stand_ins.started(request, "gpu")


def _computes_only(ds: Dataset, request: Request, *step_ids: str) -> None:
    """Record that *request* computes the runs of *step_ids* and of no other step.

    A submission leaves out a step whose run is already complete, which a step
    of the stand-in recipe never is.
    """
    computes = load_computes(ds.base_dir, request.request_id)
    assert computes is not None
    _ = save_computes(
        ds.base_dir,
        request.request_id,
        [run for run in computes if run.step in step_ids],
    )


def test_a_gpu_step_whose_run_is_not_computed_starts_while_every_card_is_held(
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step only records a cache hit, so it takes no card and does not wait."""
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS[:1])
    _on_gpu(monkeypatch, "cached", "computed")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(GPU_CARD_VAR, "0:GPU-inherited")
    request = submit_request(tracked, _recipe("cached", "computed")).request
    _computes_only(tracked, request, "computed")
    held = try_lease(_cards())
    assert held is not None
    try:
        joined = stand_ins.drive(tracked, request)
        cached = stand_ins.seen(request, "cached")
        stand_ins.release(request, "cached")
        wait_for(
            lambda: _gpu_wait_step(tracked, request) == "computed",
            STEP_SECONDS,
            "the computed step never waited for the card",
        )
        assert not stand_ins.started(request, "computed")
    finally:
        held.close()
    computed = stand_ins.seen(request, "computed")
    stand_ins.release(request, "computed")
    done = joined()

    assert (cached["CUDA_VISIBLE_DEVICES"], cached["MOSAIC_GPU_CARD"]) == ("0", None)
    assert cached["MOSAIC_GPU_LEASE_FDS"] is None
    assert computed["MOSAIC_GPU_CARD"] == f"0:{FIRST_UUID}"
    log = done.log_path.read_text()
    assert "step cached waits for a GPU" not in log, log
    assert done.ran == ("cached", "computed")


def test_a_gpu_step_below_a_conversion_takes_a_card(
    tmp_path: Path, stand_ins: StandIns, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fit planned after a conversion has no run id at submit, and computes."""
    ds = calms21_dataset(tmp_path / "study", {"train": {"a": {"s1": 90}}}, seed=11)
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS[:1])
    _on_gpu(monkeypatch, "templates")
    monkeypatch.setattr(
        graph_run,
        "MOSAIC_COMMAND",
        (
            sys.executable,
            "-c",
            REAL_OR_STAND_IN,
            "tracks,speed",
            str(stand_ins.directory),
        ),
    )
    request = submit_request(
        ds, converting_recipe(SPEED_OF_CONVERTED_STEP, TEMPLATES_OF_CONVERTED)
    ).request
    computes = load_computes(ds.base_dir, request.request_id)
    assert computes is not None
    assert "templates" not in [run.step for run in computes]
    held = try_lease(_cards())
    assert held is not None
    try:
        joined = stand_ins.drive(ds, request)
        wait_for(
            lambda: _gpu_wait_step(ds, request) == "templates",
            STEP_SECONDS,
            "the fit below the conversion never waited for the card",
        )
        assert not stand_ins.started(request, "templates")
    finally:
        held.close()
    seen = stand_ins.seen(request, "templates")
    stand_ins.release(request, "templates")
    done = joined()

    assert seen["MOSAIC_GPU_CARD"] == f"0:{FIRST_UUID}"
    assert done.ran == ("tracks", "speed", "templates")


def test_a_request_without_its_computed_runs_leases_every_gpu_step(
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request an older mosaic submitted records no runs, so each GPU step waits."""
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS[:1])
    _on_gpu(monkeypatch, "gpu")
    request = submit_request(tracked, _recipe("gpu")).request
    computes_path(tracked.base_dir, request.request_id).unlink()
    held = try_lease(_cards())
    assert held is not None
    try:
        joined = stand_ins.drive(tracked, request)
        wait_for(
            lambda: _gpu_wait_step(tracked, request) == "gpu",
            STEP_SECONDS,
            "the step never waited for the card",
        )
        assert not stand_ins.started(request, "gpu")
    finally:
        held.close()
    seen = stand_ins.seen(request, "gpu")
    stand_ins.release(request, "gpu")
    _ = joined()

    assert seen["MOSAIC_GPU_CARD"] == f"0:{FIRST_UUID}"


def test_a_hidden_card_is_refused_before_the_driver_takes_its_lease(
    tracked: Dataset,
    stand_ins: StandIns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit is checked even when no step is on a GPU lane."""
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)
    _on_gpu(monkeypatch)
    request = submit_request(tracked, _recipe("cpu")).request

    with pytest.raises(GpuCardsRefused, match="'7'"):
        _ = run_request(tracked, request, gpu_cards=("7",))

    assert not driver_lease_path(tracked.base_dir, request.request_id).exists()
    assert request_status_wire(tracked.base_dir, request).verdict != "orphaned"
    assert _status(tracked, request, "cpu") == ""
    assert not stand_ins.started(request, "cpu")


def test_without_nvidia_smi_a_gpu_step_starts_unleased_after_one_warning(
    tracked: Dataset, stand_ins: StandIns, monkeypatch: pytest.MonkeyPatch
) -> None:
    _on_gpu(monkeypatch, "gpu", "also-gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    request = submit_request(tracked, _recipe("gpu", "also-gpu")).request

    joined = stand_ins.drive(tracked, request)
    seen = [stand_ins.seen(request, "gpu")]
    stand_ins.release(request, "gpu")
    seen.append(stand_ins.seen(request, "also-gpu"))
    stand_ins.release(request, "also-gpu")
    done = joined()

    assert [
        (step["CUDA_VISIBLE_DEVICES"], step["MOSAIC_GPU_CARD"]) for step in seen
    ] == [
        ("0", None),
        ("0", None),
    ]
    warnings = [
        line
        for line in done.log_path.read_text().splitlines()
        if line.startswith("[mosaic] warning: GPU steps take no card lease: ")
    ]
    assert warnings == [
        "[mosaic] warning: GPU steps take no card lease: nvidia-smi is not on PATH"
    ]


def test_a_step_type_this_installation_does_not_know_is_on_the_default_lane() -> None:
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [{"id": "gone", "type": "feature", "feature": "no-such-feature"}],
        }
    )

    assert graph_run.step_lane(recipe, "gone") == DEFAULT_LANE


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_the_driver_puts_each_step_on_the_lane_its_plan_names(
    tmp_path: Path, device: str
) -> None:
    ds = calms21_dataset(
        tmp_path / "study", {"train": {"a": {"s1": 90, "s2": 90}}}, seed=11, n_classes=2
    )
    _ = run_pipeline(
        ds,
        Recipe.model_validate(
            {"schema_version": 1, "steps": [CONVERT_TRACKS_STEP, CONVERT_LABELS_STEP]}
        ),
    )
    document = classifier_recipe("clf").model_dump(mode="json")
    for step in document["steps"]:
        if step["id"] == "clf":
            step["params"]["device"] = device
    recipe = Recipe.model_validate(document)

    plan = plan_pipeline(ds, recipe, refuse=False)

    assert [
        (step.step_id, graph_run.step_lane(recipe, step.step_id)) for step in plan.steps
    ] == [(step.step_id, step.lane) for step in plan.steps]
    assert (GPU_TRAIN_LANE in [step.lane for step in plan.steps]) == (device == "cuda")
