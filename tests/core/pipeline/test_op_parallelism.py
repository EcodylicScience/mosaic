"""Op steps that compute their entries at once.

``convert-tracks`` and ``resample-tracks`` convert one raw file or one table at a
time, so they take the workers a run is given. What they produce does not depend on
how many: the index, the parquet bytes and the run id are those of one worker. A
failing entry is reported as it is with one worker, and a cancel stops the pool.
"""

from __future__ import annotations

import inspect
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import mosaic.core.dataset as dataset_module
import mosaic.core.pipeline.graph.step as step_module
import mosaic.core.pipeline.parallelism as parallelism
from mosaic.core.dataset import ConversionFacts, Dataset, TrackFileOutcome
from mosaic.core.params import Params
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.job import CancelToken, Cancelled, JobContext
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import (
    OPS,
    Op,
    OpIdentity,
    ensure_core_ops_registered,
    run_op,
)
from mosaic.core.pipeline.parallelism import ParallelSettings
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import read_tracks_index, write_tracks_row
from mosaic.core.pipeline.writers import write_parquet_atomic
import mosaic.tracking.ops.resample as resample_module
from mosaic.tracking import register_ops
from tests.helpers import (
    calms21_dataset,
    entry_error_lines,
    latest_snapshot,
    make_dataset,
    write_trex_npz,
)

SOURCE_VARIANT = "convert-trex_npz.0.2-aaaaaaaaaa"

CONFIGS = [
    pytest.param(ParallelSettings(workers=1), id="one-worker"),
    pytest.param(ParallelSettings(workers=4, mode="thread"), id="4-threads"),
    pytest.param(ParallelSettings(workers=4, mode="process"), id="4-processes"),
]


@dataclass(frozen=True)
class Scenario:
    """One parallelizable op over a dataset of several entries."""

    kind: str
    params: dict[str, object]
    build: Callable[[Path], Dataset]
    breaks: Callable[[Dataset], None]
    """Spoils the input of one entry."""
    broken: str
    """What the run-log of that failure names."""
    entries: int
    """How many entries a clean run writes."""
    survivors: int
    """How many entries a run writes when one input is spoiled."""
    stem: str
    """What the table names of the spoiled entries start with."""


def _calms(base: Path) -> Dataset:
    files = {f"g{i}": {"a": {"s1": 4, "s2": 3}} for i in range(6)}
    return calms21_dataset(base, files)


def _break_calms(ds: Dataset) -> None:
    _ = (ds.base_dir / "raw" / "g2.npy").write_bytes(b"not an npy file")


def _trex(base: Path) -> Dataset:
    ds = make_dataset(base)
    raw = ds.base_dir / "raw"
    for sequence in range(6):
        for individual in range(2):
            write_trex_npz(raw / f"seq{sequence}_fish{individual}.npz")
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")
    return ds


def _break_trex(ds: Dataset) -> None:
    _ = (ds.base_dir / "raw" / "seq3_fish1.npz").write_bytes(b"not an npz file")


def _table(sequence: str, fps: float = 34.679, n: int = 300) -> pd.DataFrame:
    frames = np.arange(n)
    return pd.DataFrame(
        {
            "frame": frames.astype(np.int64),
            "time": (frames / fps).astype(np.float32),
            "id": np.zeros(n, dtype=np.int64),
            "group": [""] * n,
            "sequence": [sequence] * n,
            "X": frames * 0.2,
            "Y": np.sin(frames / 25.0) * 10.0,
            "ANGLE": ((frames * 0.3 + np.pi) % (2 * np.pi)) - np.pi,
            "cm_per_pixel": np.where(frames == 0, 0.05, np.nan),
            "tracklet_id": (frames // 100).astype(np.int64),
            "frame_rate": np.full(n, fps),
        }
    )


def _resample(base: Path) -> Dataset:
    ds = make_dataset(base)
    root = tracks_variant_root(ds.get_root("tracks"), SOURCE_VARIANT)
    for index in range(6):
        sequence = f"s{index}"
        frame = _table(sequence)
        out_path = root / f"{sequence}.parquet"
        _ = write_parquet_atomic(frame, out_path)
        write_tracks_row(
            ds,
            run_id=SOURCE_VARIANT,
            group="",
            sequence=sequence,
            out_path=out_path,
            producer="convert-trex_npz",
            std_format="trex_v2",
            n_rows=len(frame),
            consumed_source_roots=("tracks_raw",),
            carries_tracklets=None,
        )
    return ds


def _break_resample(ds: Dataset) -> None:
    root = tracks_variant_root(ds.get_root("tracks"), SOURCE_VARIANT)
    _ = (root / "s3.parquet").write_bytes(b"not a parquet file")


SCENARIOS = [
    pytest.param(
        Scenario(
            kind="convert-tracks",
            params={"source_format": "calms21_npy"},
            build=_calms,
            breaks=_break_calms,
            broken="g2.npy",
            stem="g2",
            entries=12,
            survivors=10,
        ),
        id="convert-individual-files",
    ),
    pytest.param(
        Scenario(
            kind="convert-tracks",
            params={"source_format": "trex_npz"},
            build=_trex,
            breaks=_break_trex,
            broken="seq3_fish1.npz",
            stem="seq3",
            entries=6,
            survivors=5,
        ),
        id="convert-merged-files",
    ),
    pytest.param(
        Scenario(
            kind="resample-tracks",
            params={"target_fps": 30.0},
            build=_resample,
            breaks=_break_resample,
            broken="s3",
            stem="s3",
            entries=6,
            survivors=5,
        ),
        id="resample",
    ),
]


@pytest.fixture(autouse=True)
def _tracking_ops() -> None:
    register_ops()


def _copy(ds: Dataset, destination: Path) -> Dataset:
    """A byte-for-byte copy of *ds*, so two runs start from one state."""
    _ = shutil.copytree(ds.base_dir, destination)
    return Dataset(manifest_path=destination / ds.manifest_path.name).load()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """The bytes of every file under *root* but the index, which holds timestamps."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".lock" and path.name != "index.csv"
    }


def _index_rows(ds: Dataset) -> list[tuple[str, ...]]:
    """The tracks index in file order, without the columns that record when."""
    index = read_tracks_index(ds).drop(columns=["started_at", "finished_at"])
    return [tuple(str(cell) for cell in row) for row in index.to_numpy().tolist()]


@pytest.mark.parametrize("settings", CONFIGS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_workers_change_neither_the_outputs_nor_the_run_id(
    tmp_path: Path, scenario: Scenario, settings: ParallelSettings
) -> None:
    base = scenario.build(tmp_path / "base")
    alone = _copy(base, tmp_path / "alone")
    together = _copy(base, tmp_path / "together")

    first = run_op(alone, scenario.kind, scenario.params, track=False)
    second = run_op(
        together, scenario.kind, scenario.params, track=False, parallel=settings
    )

    assert second == first
    assert _tree_bytes(together.get_root("tracks")) == _tree_bytes(
        alone.get_root("tracks")
    )
    assert _index_rows(together) == _index_rows(alone)


@pytest.mark.parametrize("settings", CONFIGS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_failing_entry_is_reported_and_the_others_are_written(
    tmp_path: Path, scenario: Scenario, settings: ParallelSettings
) -> None:
    ds = scenario.build(tmp_path / "ds")
    scenario.breaks(ds)
    sources = len(read_tracks_index(ds))

    with pytest.raises(RuntimeError):
        _ = run_op(ds, scenario.kind, scenario.params, parallel=settings)

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] == 1
    [line] = entry_error_lines(ds, snapshot["execution_id"])
    assert scenario.broken in line
    assert len(read_tracks_index(ds)) - sources == scenario.survivors


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(ParallelSettings(workers=1), id="one-worker"),
        pytest.param(ParallelSettings(workers=2, mode="thread"), id="2-threads"),
    ],
)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_cancel_stops_the_pool_before_every_entry_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Scenario,
    settings: ParallelSettings,
) -> None:
    ds = scenario.build(tmp_path / "ds")
    sources = len(read_tracks_index(ds))
    token = CancelToken()
    beat = JobContext.heartbeat

    def heartbeat(self: JobContext, done: int | None = None) -> None:
        beat(self, done)
        token.cancel()

    monkeypatch.setattr(JobContext, "heartbeat", heartbeat)

    with pytest.raises(Cancelled):
        _ = run_op(
            ds,
            scenario.kind,
            scenario.params,
            track=False,
            parallel=settings,
            cancel_token=token,
        )

    index = read_tracks_index(ds)
    assert 0 < len(index) - sources < scenario.entries
    # Every table that a worker finished is indexed, so a rerun does not skip it as
    # already current while the index lacks it.
    tables = {
        path.relative_to(ds.base_dir).as_posix()
        for path in ds.get_root("tracks").rglob("*.parquet")
    }
    assert tables == {Path(str(cell)).as_posix() for cell in index["abs_path"]}


def test_index_rows_follow_the_raw_index_whichever_file_finishes_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _calms(tmp_path / "base")
    alone = _copy(base, tmp_path / "alone")
    together = _copy(base, tmp_path / "together")
    convert = Dataset.convert_track_file

    def slower_when_earlier(
        self: Dataset,
        raw_row: pd.Series,
        params: dict[str, object],
        overwrite: bool,
        facts: ConversionFacts | None = None,
    ) -> TrackFileOutcome:
        # g0 finishes last and g5 first.
        position = int(str(raw_row["group"]).removeprefix("g") or 0)
        time.sleep(0.05 * (5 - position))
        return convert(self, raw_row, params, overwrite, facts)

    monkeypatch.setattr(Dataset, "convert_track_file", slower_when_earlier)

    _ = run_op(alone, "convert-tracks", {"source_format": "calms21_npy"}, track=False)
    _ = run_op(
        together,
        "convert-tracks",
        {"source_format": "calms21_npy"},
        track=False,
        parallel=ParallelSettings(workers=4, mode="thread"),
    )

    assert _index_rows(together) == _index_rows(alone)
    assert [row[4] for row in _index_rows(together)] == sorted(
        row[4] for row in _index_rows(together)
    )


def test_the_op_run_id_has_no_worker_input() -> None:
    assert not [
        name
        for name in inspect.signature(op_run_id).parameters
        if "worker" in name or "parallel" in name
    ]


class _ProbeParams(Params):
    pass


class _Probe(Op[_ProbeParams]):
    kind = "probe-parallel"
    category = "convert"
    domain = "tracking"
    scope_takes = "none"
    scope_dependent = False
    Params = _ProbeParams
    seen: ClassVar[list[ParallelSettings]] = []

    def plan_identity(
        self,
        ds: Dataset,
        params: _ProbeParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        return OpIdentity(run_id="probe-run")

    def run(
        self,
        ds: Dataset,
        params: _ProbeParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        self.seen.append(ctx.parallel)
        return "probe-run"


class _ParallelProbe(_Probe):
    kind = "probe-parallel-yes"
    parallelizable = True


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[ParallelSettings]:
    ensure_core_ops_registered()
    _Probe.seen.clear()
    monkeypatch.setitem(OPS, _Probe.kind, _Probe)
    monkeypatch.setitem(OPS, _ParallelProbe.kind, _ParallelProbe)
    monkeypatch.setattr(parallelism, "host_cpus", lambda: 8)
    return _Probe.seen


def test_an_op_is_not_parallelizable_unless_it_says_so() -> None:
    ensure_core_ops_registered()
    register_ops()

    assert Op.parallelizable is False
    assert sorted(kind for kind, op in OPS.items() if op.parallelizable) == [
        "convert-tracks",
        "resample-tracks",
    ]


def test_an_op_that_is_not_parallelizable_is_given_one_worker(
    tmp_path: Path, probes: list[ParallelSettings]
) -> None:
    ds = make_dataset(tmp_path / "ds")

    _ = run_op(ds, _Probe.kind, {}, track=False, parallel=ParallelSettings(workers=4))

    assert probes == [ParallelSettings()]


def test_a_parallelizable_op_reads_its_settings_from_the_run_context(
    tmp_path: Path, probes: list[ParallelSettings]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    settings = ParallelSettings(workers=4, mode="process", min_free_memory_fraction=0.2)

    _ = run_op(ds, _ParallelProbe.kind, {}, track=False, parallel=settings)

    assert probes == [settings]


def test_a_parallelizable_op_is_given_no_more_workers_than_processors(
    tmp_path: Path, probes: list[ParallelSettings]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    parallelism_cpus = 8

    _ = run_op(
        ds,
        _ParallelProbe.kind,
        {},
        track=False,
        parallel=ParallelSettings(workers=parallelism_cpus * 2),
    )

    assert [settings.workers for settings in probes] == [parallelism_cpus]


def test_an_op_run_without_settings_runs_in_turn(
    tmp_path: Path, probes: list[ParallelSettings]
) -> None:
    ds = make_dataset(tmp_path / "ds")

    _ = run_op(ds, _ParallelProbe.kind, {}, track=False)

    assert probes == [ParallelSettings()]


def test_an_op_step_is_run_with_the_settings_the_step_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.core.pipeline.graph import Recipe, execute_step, submit_request

    ds = _calms(tmp_path / "ds")
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "convert",
            "steps": [
                {
                    "id": "convert",
                    "type": "op",
                    "kind": "convert-tracks",
                    "params": {"source_format": "calms21_npy"},
                }
            ],
        }
    )
    request = submit_request(ds, recipe).request
    seen: dict[str, object] = {}

    def run_op_stub(*_args: object, **kwargs: object) -> str:
        seen.update(kwargs)
        raise _Stopped

    monkeypatch.setattr(step_module, "run_op", run_op_stub)
    settings = ParallelSettings(workers=3, mode="process", min_free_memory_fraction=0.2)

    with pytest.raises(_Stopped):
        _ = execute_step(ds, request, "convert", parallel=settings)

    assert seen["parallel"] == settings


class _Stopped(Exception):
    """Ends a step at the call of ``run_op``, after its arguments are seen."""


# --- two files naming one entry, and tables that no row describes --------------

COLLIDING = {"source_format": "calms21_npy", "group_from": "infile"}


def _colliding(base: Path) -> Dataset:
    """Six files that each hold the entries ``("a", "s1")`` and ``("a", "s2")``."""
    files = {f"g{i}": {"a": {"s1": 4, "s2": 3}} for i in range(6)}
    return calms21_dataset(base, files, seed=1)


@pytest.mark.parametrize("overwrite", [False, True], ids=["keep", "overwrite"])
@pytest.mark.parametrize("settings", CONFIGS)
def test_files_naming_one_entry_are_settled_by_the_order_of_the_raw_index(
    tmp_path: Path, settings: ParallelSettings, overwrite: bool
) -> None:
    base = _colliding(tmp_path / "base")
    alone = _copy(base, tmp_path / "alone")
    together = _copy(base, tmp_path / "together")

    _ = run_op(alone, "convert-tracks", COLLIDING, overwrite=overwrite, track=False)
    _ = run_op(
        together,
        "convert-tracks",
        COLLIDING,
        overwrite=overwrite,
        track=False,
        parallel=settings,
    )

    assert _index_rows(together) == _index_rows(alone)
    assert _tree_bytes(together.get_root("tracks")) == _tree_bytes(
        alone.get_root("tracks")
    )
    # Without overwrite the first file keeps its entries, as it does when files are
    # converted in turn, and with it the last file's tables replace the earlier ones.
    sources = set(read_tracks_index(together)["source_abs_path"])
    assert sources == {"raw/g5.npy" if overwrite else "raw/g0.npy"}


def _produced_rows(ds: Dataset) -> pd.DataFrame:
    index = read_tracks_index(ds)
    return index[index["producer"].map(lambda name: str(name) != "convert-trex_npz")]


def _forget_one_row(ds: Dataset, produced_by: str) -> tuple[str, str]:
    """Remove one row of what the op wrote from the index, keeping its table."""
    path = ds.get_root("tracks") / "index.csv"
    index = pd.read_csv(path, dtype=str, keep_default_na=False)
    produced = index[index["producer"].str.startswith(produced_by)]
    forgotten = produced.index[0]
    entry = (str(index.loc[forgotten, "group"]), str(index.loc[forgotten, "sequence"]))
    index.drop(index=forgotten).to_csv(path, index=False)
    return entry


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(ParallelSettings(workers=1), id="one-worker"),
        pytest.param(ParallelSettings(workers=4, mode="thread"), id="4-threads"),
    ],
)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_table_that_no_row_describes_is_made_again_and_indexed(
    tmp_path: Path, scenario: Scenario, settings: ParallelSettings
) -> None:
    ds = scenario.build(tmp_path / "ds")
    _ = run_op(ds, scenario.kind, scenario.params, track=False)
    complete = sorted(_index_rows(ds))
    tables = _tree_bytes(ds.get_root("tracks"))
    producer = "resample-tracks" if scenario.kind == "resample-tracks" else "convert-"
    group, sequence = _forget_one_row(ds, producer)
    assert len(read_tracks_index(ds)) == len(complete) - 1

    _ = run_op(ds, scenario.kind, scenario.params, track=False, parallel=settings)

    assert sorted(_index_rows(ds)) == complete
    assert _tree_bytes(ds.get_root("tracks")) == tables
    assert (group, sequence) in set(
        zip(read_tracks_index(ds)["group"], read_tracks_index(ds)["sequence"])
    )


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(ParallelSettings(workers=1), id="one-worker"),
        pytest.param(ParallelSettings(workers=4, mode="thread"), id="4-threads"),
    ],
)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_staged_file_left_by_a_killed_run_is_removed_and_never_indexed(
    tmp_path: Path, scenario: Scenario, settings: ParallelSettings
) -> None:
    ds = scenario.build(tmp_path / "ds")
    _ = run_op(ds, scenario.kind, scenario.params, track=False)
    complete = sorted(_index_rows(ds))
    tables = _tree_bytes(ds.get_root("tracks"))
    produced = _produced_rows(ds) if scenario.kind == "resample-tracks" else None
    variant_dir = ds.resolve_path(
        str(
            (read_tracks_index(ds) if produced is None else produced)["abs_path"].iloc[
                -1
            ]
        )
    ).parent
    left = variant_dir / ".stage-0123456789abcdef.staged"
    _ = left.write_bytes(b"a table that was being written when the run was killed")

    _ = run_op(ds, scenario.kind, scenario.params, track=False, parallel=settings)

    assert not left.exists()
    assert sorted(_index_rows(ds)) == complete
    assert _tree_bytes(ds.get_root("tracks")) == tables
    assert not [cell for cell in read_tracks_index(ds)["abs_path"] if "stage" in cell]


def test_the_file_that_finishes_last_does_not_replace_the_entry_of_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _colliding(tmp_path / "ds")
    convert = Dataset.convert_track_file

    def slow_fourth_file(
        self: Dataset,
        raw_row: pd.Series,
        params: dict[str, object],
        overwrite: bool,
        facts: ConversionFacts | None = None,
    ) -> TrackFileOutcome:
        if str(raw_row["abs_path"]).endswith("g3.npy"):
            time.sleep(0.5)
        return convert(self, raw_row, params, overwrite, facts)

    monkeypatch.setattr(Dataset, "convert_track_file", slow_fourth_file)

    _ = run_op(
        ds,
        "convert-tracks",
        COLLIDING,
        track=False,
        parallel=ParallelSettings(workers=4, mode="thread"),
    )

    assert set(read_tracks_index(ds)["source_abs_path"]) == {"raw/g0.npy"}


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(ParallelSettings(workers=1), id="one-worker"),
        pytest.param(ParallelSettings(workers=4, mode="thread"), id="4-threads"),
    ],
)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_staged_table_removed_before_it_is_published_is_one_failed_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Scenario,
    settings: ParallelSettings,
) -> None:
    ds = scenario.build(tmp_path / "ds")
    sources = len(read_tracks_index(ds))

    def removed_by_a_concurrent_run(staged: Callable[..., Path]) -> Callable[..., Path]:
        def stage(df: pd.DataFrame, final: Path) -> Path:
            path = staged(df, final)
            if final.name.startswith(scenario.stem):
                path.unlink()
            return path

        return stage

    monkeypatch.setattr(
        dataset_module,
        "stage_table",
        removed_by_a_concurrent_run(dataset_module.stage_table),
    )
    monkeypatch.setattr(
        resample_module,
        "stage_table",
        removed_by_a_concurrent_run(resample_module.stage_table),
    )

    with pytest.raises(RuntimeError):
        _ = run_op(ds, scenario.kind, scenario.params, parallel=settings)

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] >= 1
    lines = entry_error_lines(ds, snapshot["execution_id"])
    assert lines
    assert all(scenario.stem in line and "concurrent run" in line for line in lines)
    assert len(read_tracks_index(ds)) - sources == scenario.survivors
