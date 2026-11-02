"""A run widens an individual-level input onto the pair rows of its input 0.

The loader's rule is pinned in ``test_multi_input_alignment.py``. These run
``run_feature`` over a pair-level and an individual-level result, so the flag
reaches the loader from the feature's ``emits``, and an entry whose pair input
holds no rows is reported as lost.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.params import Params
from mosaic.core.pipeline.loading import MultiInputAlignmentError
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.runlog import run_log_path
from tests.helpers import add_track_sequences, make_dataset, result_outputs

_EMPTY = "empty"


class _Stub:
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

    def __init__(self) -> None:
        self.params: _Stub.Params = self.Params.from_overrides(None)

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


class _OverTracks(_Stub):
    """A stub reading the tracks."""

    class Inputs(Inputs[TrackInput]):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.inputs: _OverTracks.Inputs = self.Inputs(("tracks",))


class _PairRows(_OverTracks):
    """Both orderings of every pair of individuals per frame, none for ``_EMPTY``."""

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
        if (df["sequence"] == _EMPTY).all():
            return table.iloc[0:0]
        return table


class _IndividualRows(_OverTracks):
    """``speed`` names its cell: ten times the frame plus the id."""

    name = "test-individual-rows"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "speed": 10.0 * df["frame"] + df["id"],
            }
        )


class _Echo(_Stub):
    """An ``as-input`` consumer that writes its merged input."""

    name = "test-echo"
    emits: EmitsLevel = "as-input"

    class Inputs(Inputs[Result]):
        pass

    def __init__(self, inputs: _Echo.Inputs) -> None:
        super().__init__()
        self.inputs: _Echo.Inputs = inputs


class _EchoPerIndividual(_Echo):
    name = "test-echo-individual"
    emits: EmitsLevel = "individual"


def _results(tmp_path: Path, *sequences: str) -> tuple[Dataset, Result, Result]:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *(("g", s) for s in sequences), n_rows=6, n_ids=2)
    pairs = run_feature(ds, _PairRows())
    individuals = run_feature(ds, _IndividualRows())
    return (
        ds,
        Result(feature=pairs.feature, run_id=pairs.run_id),
        Result(feature=individuals.feature, run_id=individuals.run_id),
    )


def test_an_as_input_run_reads_each_pair_row_with_both_individuals(
    tmp_path: Path,
) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1")

    echoed = run_feature(ds, _Echo(_Echo.Inputs((pairs, individuals))))

    (table,) = result_outputs(ds, echoed).values()
    assert {"id1", "id2", "dist", "speed__id1", "speed__id2"} <= set(table.columns)
    assert "id" not in table.columns
    assert len(table) == 12
    for slot in ("id1", "id2"):
        expected = 10.0 * table["frame"] + table[slot]
        assert list(table[f"speed__{slot}"]) == list(expected)


def test_an_individual_level_consumer_keeps_the_refusal(tmp_path: Path) -> None:
    ds, pairs, individuals = _results(tmp_path, "s1")

    with pytest.raises(MultiInputAlignmentError, match="pair every row"):
        _ = run_feature(
            ds, _EchoPerIndividual(_EchoPerIndividual.Inputs((pairs, individuals)))
        )


def test_an_entry_without_pair_rows_is_an_entry_error(tmp_path: Path) -> None:
    """The entry is lost and reported, and the run keeps the others."""
    ds, pairs, individuals = _results(tmp_path, "s1", _EMPTY)
    execution_id = "TESTWIDEN1"

    result = run_feature(
        ds, _Echo(_Echo.Inputs((pairs, individuals))), execution_id=execution_id
    )

    assert result.failed_entries == (make_entry_key("g", _EMPTY),)
    errors = [
        line
        for line in run_log_path(ds.base_dir, execution_id).read_text().splitlines()
        if '"entry_error"' in line
    ]
    assert len(errors) == 1
    assert "PairRowsMissing" in errors[0]
