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
- ``widening``: ``TemporalStackingFeature``, an ``as-input`` consumer over the
  pair rows of ``PairPositionFeatures`` and the individual rows of
  ``SpeedAngvel``, so that each entry's individual rows are widened onto its
  pair rows wherever the entry loads.
- ``pair-filter``: ``TemporalStackingFeature`` over the pair rows of
  ``PairPositionFeatures`` with ``pair_filter`` the run of ``NearestNeighbor``,
  over three individuals per frame, so that the filter drops the pairs between
  the two individuals that are not each other's neighbor, and a start frame cuts
  the first sequence of each group, so that the neighbor filter and the bounds
  both run wherever the entry loads.
- ``bounds``: ``SpeedAngvel`` with a start frame that cuts the first sequence of
  each group and an end time that cuts the last, so that each filter runs
  wherever the entry loads.

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
from typing import Final

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.extract_templates import ExtractTemplates
from mosaic.behavior.feature_library.global_scaler import GlobalScaler
from mosaic.behavior.feature_library.individual_aggregate import IndividualAggregate
from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.behavior.feature_library.temporal_stacking import TemporalStackingFeature
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.definitions.define import (
    define_feature,
    load_feature_definition,
)
from mosaic.core.pipeline.parallelism import process_capable
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import backfill_frame_extents
from mosaic.core.pipeline.types import Feature, NNResult, Result
from mosaic.runlog import reduce_run_log, run_log_dir
from tests.helpers import (
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    add_track_sequences,
    assert_ran_in_mode,
    definitions_forgotten,
    make_dataset,
    neighbor_pairs,
    ordered_pairs,
    result_outputs,
    result_run_root,
    run_modes,
)

_GROUPS: Final = {"ga": ("a1", "a2"), "gb": ("b1", "b2")}
"""The groups of the fixture and the sequences of each, in order."""

_ENTRIES: Final = sum(len(names) for names in _GROUPS.values())
"""The entries of the fixture."""

_ROWS: Final = 20
"""The frames of each sequence."""

_CUT_START_FRAME: Final = 5
"""The first frame the ``bounds`` case keeps."""

_CUT_END_TIME: Final = 34.5 / 30.0
"""The time the ``bounds`` case keeps rows before: frames up to 34 at 30 fps."""

_PER_ATTEMPT: Final = frozenset({"params.json", "fit_scope.json"})
"""The files of a run root whose bytes carry the attempt or the time that wrote them."""


_MODES: Final = run_modes(3)


@dataclass(frozen=True)
class _Case:
    """One feature run over the fixture.

    Attributes:
        name: The case's name, which the datasets' directories carry.
        build: Returns the feature to run on a copy of the fixture, after
            running whatever that feature reads from an upstream run.
        overlap_frames: The ``overlap_frames`` the run asks for.
        frame_bounds: The start and end of ``filter_start_frame`` and
            ``filter_end_frame``, each ``None`` where the run sets none.
        time_window: The start and end of ``filter_start_time`` and
            ``filter_end_time``, each ``None`` where the run sets none.
        widens: Whether each output must carry the columns of both
            individuals of its pair rows.
        individuals: The individuals of each frame of the fixture.
        pair_filtered: Whether each output must hold the neighbor pairs of its
            pair rows and fewer than those.
    """

    name: str
    build: Callable[[Dataset], Feature]
    overlap_frames: int = 0
    frame_bounds: tuple[int | None, int | None] | None = None
    time_window: tuple[float | None, float | None] | None = None
    widens: bool = False
    individuals: int = 2
    pair_filtered: bool = False


def _fixture(base: Path, individuals: int) -> Dataset:
    """A dataset at *base* of four sequences in two continuous groups.

    Each sequence holds *individuals* individuals per frame and a heading, and
    each group's sequences follow one another on a frame axis.
    """
    ds = make_dataset(base, name="byte-identity", uuid="byte-identity")
    for group, names in _GROUPS.items():
        for position, name in enumerate(names):
            add_track_sequences(
                ds,
                (group, name),
                n_rows=_ROWS,
                n_ids=individuals,
                start_frame=position * _ROWS,
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


def _widening(ds: Dataset) -> Feature:
    pairs = run_feature(ds, PairPositionFeatures())
    speed = run_feature(ds, SpeedAngvel())
    return TemporalStackingFeature(
        inputs=TemporalStackingFeature.Inputs((pairs, speed)),
        params={"half": 2, "skip": 1, "add_pool": False},
    )


def _pair_filter(ds: Dataset) -> Feature:
    neighbors = run_feature(ds, NearestNeighbor())
    pairs = run_feature(ds, PairPositionFeatures())
    return TemporalStackingFeature(
        inputs=TemporalStackingFeature.Inputs((pairs,)),
        params={
            "half": 2,
            "skip": 1,
            "add_pool": False,
            "pair_filter": NNResult(feature=neighbors.feature, run_id=neighbors.run_id),
        },
    )


def _definition(ds: Dataset) -> Feature:
    _ = define_feature(ds, MINIMAL_NAME, MINIMAL_SOURCE)
    return load_feature_definition(ds, f"def:{MINIMAL_NAME}")()


_CASES: Final = (
    _Case("per-frame", _per_frame),
    _Case("aggregate", _aggregate),
    _Case("fitted", _fitted),
    _Case("pair", _pair),
    _Case("definition", _definition),
    _Case("widening", _widening, widens=True),
    _Case(
        "pair-filter",
        _pair_filter,
        frame_bounds=(_CUT_START_FRAME, None),
        individuals=3,
        pair_filtered=True,
    ),
    _Case("overlap", _per_frame, overlap_frames=2),
    # The frames of a sequence are 20 * position .. 20 * position + 19 and the
    # time of a frame is frame / 30. A start frame of 5 cuts frames 0 to 4 from
    # a1 and b1 and leaves a2 and b2 whole. An end time of 34.5 / 30 s cuts
    # frames 35 to 39 from a2 and b2 and leaves a1 and b1 whole. A run sets
    # one filter per boundary, so the start is a frame and the end a time.
    _Case(
        "bounds",
        _per_frame,
        frame_bounds=(_CUT_START_FRAME, None),
        time_window=(None, _CUT_END_TIME),
    ),
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


def _kept_only_the_bounds(ds: Dataset, result: Result, case: _Case) -> None:
    """Assert that each output holds the frames its entry has inside the bounds.

    The bounds must cut rows, or the case would compare runs that filter nothing.
    """
    root = result_run_root(ds, result)
    cut = 0
    for output in sorted(root.glob("*.parquet")):
        own = pd.read_parquet(
            ds.get_root("tracks") / output.name, columns=["frame", "time"]
        )
        keep = pd.Series(True, index=own.index)
        if case.frame_bounds is not None:
            start, end = case.frame_bounds
            if start is not None:
                keep &= own["frame"] >= start
            if end is not None:
                keep &= own["frame"] < end
        if case.time_window is not None:
            start_time, end_time = case.time_window
            if start_time is not None:
                keep &= own["time"] >= start_time
            if end_time is not None:
                keep &= own["time"] < end_time
        written = pd.read_parquet(output, columns=["frame"])
        assert sorted(set(written["frame"].tolist())) == sorted(
            set(own.loc[keep, "frame"].tolist())
        ), output.name
        cut += int((~keep).sum())
    assert cut > 0
    assert len(outputs := sorted(root.glob("*.parquet"))) == _ENTRIES, outputs


def _widened_onto_the_pair_rows(ds: Dataset, result: Result) -> None:
    """Assert that each output carries a column of each individual of its pairs.

    Without them the case would compare runs that widen nothing.
    """
    root = result_run_root(ds, result)
    outputs = sorted(root.glob("*.parquet"))
    assert len(outputs) == _ENTRIES
    for output in outputs:
        columns = pd.read_parquet(output).columns
        for slot in ("id1", "id2"):
            prefix = f"speed__{slot}"
            assert any(name.startswith(prefix) for name in columns), (
                output.name,
                prefix,
            )


def _kept_only_neighbor_pairs(ds: Dataset, result: Result, feature: Feature) -> None:
    """Assert that each output holds the neighbor pairs of its entry's pair rows.

    The pair rows of an entry are the ones the start frame leaves. The filter
    must drop some of them, or the case would compare runs that filter nothing.
    """
    assert isinstance(feature, TemporalStackingFeature)
    neighbors = feature.params.pair_filter
    assert neighbors is not None
    (pairs,) = feature.inputs.feature_inputs
    held = result_outputs(ds, pairs)
    outputs = result_outputs(ds, result)
    dropped = 0
    for group, names in _GROUPS.items():
        for name in names:
            key = make_entry_key(group, name)
            assert {"frame", "id1", "id2"} <= set(outputs[key].columns)
            rows = [
                pair for pair in ordered_pairs(held[key]) if pair[0] >= _CUT_START_FRAME
            ]
            neighbor = set(neighbor_pairs(ds, neighbors, group, name))
            written = ordered_pairs(outputs[key])
            assert written == [pair for pair in rows if pair in neighbor], key
            dropped += len(rows) - len(written)
    assert dropped > 0


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_every_mode_writes_the_same_files_under_one_run_id(
    case: _Case, tmp_path: Path
) -> None:
    """Serial, thread 3 and process 3 agree on the run id, the files and every byte.

    Each mode runs in its own copy of the fixture, so no run is a cache hit of
    another and the fitted case fits in each copy.
    """
    written: dict[str, _Written] = {}
    bounds, window = case.frame_bounds, case.time_window
    with definitions_forgotten():
        for mode in _MODES:
            ds = _fixture(tmp_path / f"{mode.name}-{case.name}", case.individuals)
            feature = case.build(ds)
            assert process_capable(feature)
            result = run_feature(
                ds,
                feature,
                **mode.run_kwargs(),
                overlap_frames=case.overlap_frames,
                filter_start_frame=None if bounds is None else bounds[0],
                filter_end_frame=None if bounds is None else bounds[1],
                filter_start_time=None if window is None else window[0],
                filter_end_time=None if window is None else window[1],
            )
            assert result.failed_entries == ()
            assert_ran_in_mode(ds, mode, entries=_ENTRIES)
            if case.overlap_frames:
                _own_frames(ds, result)
            if case.widens:
                _widened_onto_the_pair_rows(ds, result)
            if case.pair_filtered:
                _kept_only_neighbor_pairs(ds, result, feature)
            if bounds is not None or window is not None:
                _kept_only_the_bounds(ds, result, case)
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
