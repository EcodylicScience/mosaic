"""A feature run writes the same bytes in serial, in threads and in processes.

The worker count and mode belong to the machine and not to the run, so a run
reports one run id and writes one set of files whichever way it computes its
entries. Each case here runs one feature over one fixture three times, each in
its own copy of the dataset, and compares the three:

- ``per-frame``: ``SpeedAngvel``.
- ``aggregate``: ``IndividualAggregate``, which reduces each entry to one row
  per individual.
- ``fitted``: ``GlobalScaler``, which fits on a templates matrix. The copy runs
  ``SpeedAngvel`` and ``ExtractTemplates`` in turn first, and the scaler fits
  there and applies to each entry in the mode under test. The templates are the
  fit's input, because the scaler fits on nothing else.
- ``pair``: ``PairPositionFeatures``, which writes a row per ordered pair of the
  individuals of a frame.
- ``definition``: the feature definition ``OriginDistance`` of
  ``tests/helpers/feature_definitions.py``, imported from the dataset in each
  worker.
- ``overlap``: ``SpeedAngvel`` with ``overlap_frames=2`` over continuous groups,
  so that each entry is read with its neighbors' frames, computed, and trimmed
  to its own.

The datasets stay under the test's ``tmp_path`` as ``<mode>-<case>``, so a run
with a fixed ``--basetemp`` leaves them for
``benchmarks/step-cost/fingerprint.py``.

``params.json`` records the attempt that wrote it and ``fit_scope.json`` the
time of the fit, so their bytes are the two differences between attempts that
the comparison leaves out. Both files are still in the set of files compared.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.extract_templates import ExtractTemplates
from mosaic.behavior.feature_library.global_scaler import GlobalScaler
from mosaic.behavior.feature_library.individual_aggregate import IndividualAggregate
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.define import (
    define_feature,
    load_feature_definition,
)
from mosaic.core.pipeline.parallelism import ParallelMode, process_capable
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import backfill_frame_extents
from mosaic.core.pipeline.types import Feature, Result
from mosaic.runlog import reduce_run_log, run_log_dir
from tests.helpers import (
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    add_track_sequences,
    definitions_forgotten,
    latest_events,
    make_dataset,
    result_run_root,
)

_GROUPS: Final = {"ga": ("a1", "a2"), "gb": ("b1", "b2")}
"""The groups of the fixture and the sequences of each, in order."""

_ENTRIES: Final = sum(len(names) for names in _GROUPS.values())
"""The entries of the fixture."""

_ROWS: Final = 20
"""The frames of each sequence."""

_PER_ATTEMPT: Final = frozenset({"params.json", "fit_scope.json"})
"""The files of a run root whose bytes carry the attempt or the time that wrote them."""


class _Mode(NamedTuple):
    """How a run computes its entries."""

    name: str
    workers: int
    kind: ParallelMode


_MODES: Final = (
    _Mode("serial", 1, "thread"),
    _Mode("thread", 3, "thread"),
    _Mode("process", 3, "process"),
)


@dataclass(frozen=True)
class _Case:
    """One feature run over the fixture.

    Attributes:
        name: The case's name, which the datasets' directories carry.
        build: Returns the feature to run on a copy of the fixture, after
            running whatever that feature reads from an upstream run.
        overlap_frames: The ``overlap_frames`` the run asks for.
    """

    name: str
    build: Callable[[Dataset], Feature]
    overlap_frames: int = 0


def _fixture(base: Path) -> Dataset:
    """A dataset at *base* of four sequences in two continuous groups.

    Each sequence holds two individuals per frame and a heading, and each group's
    sequences follow one another on a frame axis.
    """
    ds = make_dataset(base, name="byte-identity", uuid="byte-identity")
    for group, names in _GROUPS.items():
        for position, name in enumerate(names):
            add_track_sequences(
                ds, (group, name), n_rows=_ROWS, n_ids=2, start_frame=position * _ROWS
            )
    for path in sorted(ds.get_root("tracks").glob("*.parquet")):
        table = pd.read_parquet(path)
        table["ANGLE"] = np.sin(table["frame"].to_numpy() / 7.0)
        table.to_parquet(path)
    _ = backfill_frame_extents(ds)
    ds.set_continuous_groups(list(_GROUPS))
    return ds


def _per_frame(_ds: Dataset) -> Feature:
    return SpeedAngvel()


def _aggregate(_ds: Dataset) -> Feature:
    return IndividualAggregate(
        params={"columns": ["X", "Y", "feat_a"], "aggs": ["mean", "std"]}
    )


def _fitted(ds: Dataset) -> Feature:
    speed = run_feature(ds, SpeedAngvel())
    templates = run_feature(
        ds,
        ExtractTemplates(
            inputs=ExtractTemplates.Inputs((speed,)),
            params={"n_templates": 8, "columns": ["speed", "vx", "vy"]},
        ),
    )
    return GlobalScaler(
        inputs=GlobalScaler.Inputs((speed,)),
        params={
            "templates": {
                "feature": templates.feature,
                "run_id": templates.run_id,
                "pattern": "templates.parquet",
            }
        },
    )


def _pair(_ds: Dataset) -> Feature:
    return PairPositionFeatures()


def _definition(ds: Dataset) -> Feature:
    _ = define_feature(ds, MINIMAL_NAME, MINIMAL_SOURCE)
    return load_feature_definition(ds, f"def:{MINIMAL_NAME}")()


_CASES: Final = (
    _Case("per-frame", _per_frame),
    _Case("aggregate", _aggregate),
    _Case("fitted", _fitted),
    _Case("pair", _pair),
    _Case("definition", _definition),
    _Case("overlap", _per_frame, overlap_frames=2),
)


@dataclass(frozen=True)
class _Written:
    """What one run left on disk.

    Attributes:
        run: The storage and run id of the run.
        digests: The SHA-256 of each file under the run root, by its path there,
            and ``None`` for a file of ``_PER_ATTEMPT``.
        unfinished: Each run-log of the dataset that does not end ``finished``.
    """

    run: tuple[str, str | None]
    digests: dict[str, str | None]
    unfinished: list[str]


def _written(ds: Dataset, result: Result) -> _Written:
    root = result_run_root(ds, result)
    digests: dict[str, str | None] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = (
                None
                if path.name in _PER_ATTEMPT
                else hashlib.sha256(path.read_bytes()).hexdigest()
            )
    unfinished: list[str] = []
    for log in sorted(run_log_dir(ds.base_dir).glob("*.jsonl")):
        snapshot = reduce_run_log(log)
        status = None if snapshot is None else snapshot["status"]
        if status != "finished":
            unfinished.append(f"{log.name}: {status}")
    return _Written((result.feature, result.run_id), digests, unfinished)


def _own_frames(ds: Dataset, result: Result) -> None:
    """Assert that each output holds the frames of its own entry and no neighbor's."""
    root = result_run_root(ds, result)
    outputs = sorted(root.glob("*.parquet"))
    assert len(outputs) == _ENTRIES
    for output in outputs:
        own = pd.read_parquet(ds.get_root("tracks") / output.name, columns=["frame"])
        written = pd.read_parquet(output, columns=["frame"])
        assert sorted(set(written["frame"].tolist())) == sorted(
            set(own["frame"].tolist())
        ), output.name


def _ran_every_entry_as_asked(ds: Dataset, mode: _Mode) -> None:
    """Assert that the last attempt computed each entry, in the mode asked for.

    A cache hit writes an ``entry_end`` without a ``mode``, and a run that fell
    back to threads writes ``thread`` where processes were asked for.
    """
    ends = latest_events(ds, "entry_end")
    assert [end.get("mode") for end in ends] == [mode.kind] * _ENTRIES, mode.name


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_every_mode_writes_the_same_files_under_one_run_id(
    case: _Case, tmp_path: Path
) -> None:
    """Serial, thread 3 and process 3 agree on the run id, the files and every byte.

    Each mode runs in its own copy of the fixture, so no run is a cache hit of
    another and the fitted case fits in each copy.
    """
    written: dict[str, _Written] = {}
    with definitions_forgotten():
        for mode in _MODES:
            ds = _fixture(tmp_path / f"{mode.name}-{case.name}")
            feature = case.build(ds)
            assert process_capable(feature)
            result = run_feature(
                ds,
                feature,
                parallel_workers=mode.workers,
                parallel_mode=mode.kind,
                overlap_frames=case.overlap_frames,
            )
            assert result.failed_entries == ()
            _ran_every_entry_as_asked(ds, mode)
            if case.overlap_frames:
                _own_frames(ds, result)
            written[mode.name] = _written(ds, result)

    serial = written["serial"]
    assert serial.run[1] is not None
    outputs = {
        f"{group}__{name}.parquet" for group, names in _GROUPS.items() for name in names
    }
    assert outputs <= {name for name, digest in serial.digests.items() if digest}
    for mode in _MODES:
        other = written[mode.name]
        assert other.unfinished == [], mode.name
        assert other.run == serial.run, mode.name
        assert sorted(other.digests) == sorted(serial.digests), mode.name
        differing = sorted(
            name
            for name, digest in serial.digests.items()
            if other.digests[name] != digest
        )
        assert differing == [], mode.name
