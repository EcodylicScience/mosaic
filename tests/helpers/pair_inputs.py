"""Stub features that write the pair rows and the individual rows a consumer merges.

``PairRowsFeature`` and ``IndividualRowsFeature`` read the tracks and write the
two inputs of a widening ``as-input`` consumer, and ``EchoFeature`` is the
consumer: it writes the table its inputs merge to. Each is built as
``cls(inputs, params)``, or without arguments for a feature over the tracks, so
that a process worker can build it. This module also holds the checks that the
runs of these stubs share: the entry that holds no pair rows, an input that is
not parquet, and a run that ends without an ``entry_error``. It imports no
feature library.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.params import Params
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_index import backfill_frame_extents
from mosaic.core.pipeline.types import (
    Feature,
    Inputs,
    InputStream,
    NNResult,
    Result,
    TrackInput,
)
from mosaic.core.scope import Scope
from tests.helpers.datasets import make_dataset
from tests.helpers.modes import RunMode
from tests.helpers.results import result_outputs, result_run_root
from tests.helpers.runlog import latest_events, latest_snapshot
from tests.helpers.tracks import add_track_sequences

Pair = tuple[int, int, int]
"""A pair row's ``(frame, id1, id2)``."""

EMPTY_SEQUENCE: Final = "empty"
"""The sequence for which ``PairRowsFeature`` writes no pair rows."""

NO_INDIVIDUAL_ROWS: Final = "no-individuals"
"""The sequence for which ``IndividualRowsFeature`` writes no individual rows."""

ROWS: Final = 6
"""The frames of each sequence that ``pair_run_results`` writes."""

IDENTITIES: Final = 3
"""The individuals of each frame that ``pair_run_results`` writes."""

NO_PAIR_ROWS_MESSAGE: Final = (
    "input 0 holds no pair rows for this entry, so the individual rows of "
    "input 1 have none to widen onto"
)
"""What the entry without pair rows reports."""

NO_INDIVIDUAL_ROWS_MESSAGE: Final = (
    "input 1 holds no individual rows for this entry, so the pair rows of the "
    "merged inputs would lack the columns it widens them with"
)
"""What the entry without individual rows reports."""


class StubFeature:
    """The boilerplate of a test feature; subclasses set the inputs and ``apply``."""

    name = "test-stub"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "individual"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()

    class Params(Params):
        pass

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params: StubFeature.Params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class TracksFeature(StubFeature):
    """A stub reading the tracks."""

    class Inputs(Inputs[TrackInput]):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.inputs: TracksFeature.Inputs = self.Inputs(("tracks",))


class PairRowsFeature(TracksFeature):
    """Both orderings of every pair of individuals per frame, none for ``EMPTY_SEQUENCE``."""

    name = "test-pair-rows"
    emits: EmitsLevel = "pair"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = sorted(int(i) for i in df["id"].unique())
        rows = [
            {"frame": int(frame), "id1": a, "id2": b, "perspective": int(a > b)}
            for frame in sorted(df["frame"].unique())
            for a in ids
            for b in ids
            if a != b
        ]
        table = pd.DataFrame(rows, columns=["frame", "id1", "id2", "perspective"])
        table["dist"] = 1.0
        if (df["sequence"] == EMPTY_SEQUENCE).all():
            return table.iloc[0:0]
        return table


class IndividualRowsFeature(TracksFeature):
    """``speed`` names its cell, ten times the frame plus the id, none for ``NO_INDIVIDUAL_ROWS``."""

    name = "test-individual-rows"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        table = pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "speed": 10.0 * df["frame"] + df["id"],
            }
        )
        if (df["sequence"] == NO_INDIVIDUAL_ROWS).all():
            return table.iloc[0:0]
        return table


class EchoFeature(StubFeature):
    """An ``as-input`` consumer that writes its merged input."""

    name = "test-echo"
    emits: EmitsLevel = "as-input"

    class Inputs(Inputs[Result]):
        pass

    def __init__(
        self, inputs: EchoFeature.Inputs, params: dict[str, object] | None = None
    ) -> None:
        super().__init__(params)
        self.inputs: EchoFeature.Inputs = inputs


def entry_errors(ds: Dataset) -> list[tuple[object, dict[str, str]]]:
    """The key and the parsed error of each ``entry_error`` of the last attempt."""
    errors: list[tuple[object, dict[str, str]]] = []
    for event in latest_events(ds, "entry_error"):
        error = event["error"]
        assert isinstance(error, str)
        errors.append((event["key"], json.loads(error)))
    return errors


def assert_lost_the_entry_without_pair_rows(ds: Dataset, result: Result) -> None:
    """Assert that the run lost the entry of ``EMPTY_SEQUENCE``, recorded it and went on."""
    assert result.failed_entries == (make_entry_key("g", EMPTY_SEQUENCE),)
    errors = entry_errors(ds)
    assert len(errors) == 1
    key, error = errors[0]
    assert key == make_entry_key("g", EMPTY_SEQUENCE)
    assert error["type"] == "PairRowsMissing"
    assert error["message"] == NO_PAIR_ROWS_MESSAGE
    assert latest_snapshot(ds)["status"] == "finished"


def assert_lost_the_entry_without_individual_rows(ds: Dataset, result: Result) -> None:
    """Assert that the run lost the entry of ``NO_INDIVIDUAL_ROWS``, recorded it and went on."""
    assert result.failed_entries == (make_entry_key("g", NO_INDIVIDUAL_ROWS),)
    errors = entry_errors(ds)
    assert len(errors) == 1
    key, error = errors[0]
    assert key == make_entry_key("g", NO_INDIVIDUAL_ROWS)
    assert error["type"] == "IndividualRowsMissing"
    assert error["message"] == NO_INDIVIDUAL_ROWS_MESSAGE
    assert latest_snapshot(ds)["status"] == "finished"


def run_in_mode(
    ds: Dataset,
    feature: Feature,
    mode: RunMode,
    *,
    overlap_frames: int = 0,
    filter_start_time: float | None = None,
) -> Result:
    """Run *feature* over *ds* in *mode*, with the overlap and start time given."""
    return run_feature(
        ds,
        feature,
        **mode.run_kwargs(),
        overlap_frames=overlap_frames,
        filter_start_time=filter_start_time,
    )


def corrupt_output(ds: Dataset, result: Result, sequence: str) -> None:
    """Overwrite the output of *result* for *sequence* with bytes that are not parquet."""
    path = result_run_root(ds, result) / f"{make_entry_key('g', sequence)}.parquet"
    assert path.exists()
    _ = path.write_bytes(b"not a parquet file")


def assert_ended_without_an_entry_error(ds: Dataset, raised: BaseException) -> None:
    """Assert that the run failed with *raised* and wrote no ``entry_error``."""
    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert json.loads(snapshot["error_json"])["type"] == type(raised).__name__
    assert latest_events(ds, "entry_error") == []


def ordered_pairs(table: pd.DataFrame) -> list[Pair]:
    """The ``(frame, id1, id2)`` of each row of *table*, sorted."""
    return sorted(
        (int(frame), int(first), int(second))
        for frame, first, second in zip(
            table["frame"], table["id1"], table["id2"], strict=True
        )
    )


def neighbor_pairs(
    ds: Dataset, neighbors: Result, group: str, sequence: str
) -> list[Pair]:
    """The ordered pairs of an entry where one individual's neighbor is the other.

    Computed from the ``frame``, ``id`` and ``nn_id`` columns of the neighbor
    run's table for the entry, over every pair of the individuals it holds in a
    frame.

    Args:
        ds: The dataset that holds the neighbor run.
        neighbors: The nearest-neighbor run.
        group: The entry's group.
        sequence: The entry's sequence.

    Returns:
        The ``(frame, id1, id2)`` of each such pair, sorted.
    """
    table = result_outputs(ds, neighbors)[make_entry_key(group, sequence)]
    neighbor_of: dict[tuple[int, int], int] = {}
    ids_of: dict[int, set[int]] = {}
    for frame, identity, neighbor in zip(
        table["frame"], table["id"], table["nn_id"], strict=True
    ):
        ids_of.setdefault(int(frame), set()).add(int(identity))
        if not pd.isna(neighbor):
            neighbor_of[(int(frame), int(identity))] = int(neighbor)
    return sorted(
        (frame, first, second)
        for frame, ids in ids_of.items()
        for first in ids
        for second in ids
        if first != second
        and (
            neighbor_of.get((frame, first)) == second
            or neighbor_of.get((frame, second)) == first
        )
    )


def add_pair_tracks(
    ds: Dataset, sequences: Sequence[str], *, continuous: bool = False
) -> None:
    """Write ``IDENTITIES`` individuals over ``ROWS`` frames for each of *sequences*.

    The sequences are in group ``g``. The tracks offset individual ``k`` by ``k``
    on both axes, so the neighbor of individuals 0 and 2 is 1, the neighbor of 1
    is 0, and the pairs ``(0, 2)`` and ``(2, 0)`` are nobody's neighbor pairs.
    Under *continuous* the sequences follow one another on one frame axis and the
    group is continuous.
    """
    for position, sequence in enumerate(sequences):
        add_track_sequences(
            ds,
            ("g", sequence),
            n_rows=ROWS,
            n_ids=IDENTITIES,
            start_frame=position * ROWS if continuous else 0,
        )
    if continuous:
        _ = backfill_frame_extents(ds)
        ds.set_continuous_groups(["g"])


def pair_run_results(
    base: Path,
    *sequences: str,
    neighbors_of: Sequence[str] | None = None,
    continuous: bool = False,
) -> tuple[Dataset, Result, Result, NNResult]:
    """A dataset over *sequences* with the pair rows, the individual rows and the neighbors.

    Args:
        base: Where the dataset goes, as ``base / "ds"``.
        sequences: The sequences of group ``g``.
        neighbors_of: The sequences the neighbor run covers, or ``None`` for
            every one.
        continuous: Whether the sequences follow one another on one frame axis.

    Returns:
        The dataset and the runs of the pair rows, of the individual rows and
        of the nearest neighbors.
    """
    from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor

    ds = make_dataset(base / "ds")
    add_pair_tracks(ds, sequences, continuous=continuous)
    pairs = run_feature(ds, PairRowsFeature())
    individuals = run_feature(ds, IndividualRowsFeature())
    scope = (
        None
        if neighbors_of is None
        else Scope(entries=[("g", sequence) for sequence in neighbors_of])
    )
    neighbors = run_feature(ds, NearestNeighbor(), scope=scope)
    return (
        ds,
        Result(feature=pairs.feature, run_id=pairs.run_id),
        Result(feature=individuals.feature, run_id=individuals.run_id),
        NNResult(feature=neighbors.feature, run_id=neighbors.run_id),
    )
