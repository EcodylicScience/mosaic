"""A step over two runs that share ``arena``, and its rows as the earlier merge left them.

Two features write ``alpha`` and ``beta`` beside an ``arena`` column, and
:class:`JoinConsumer` merges the runs of both. Its rows record the summary key
``arena`` that step 1 of the merge joins, which a run before that rule left
blank. :func:`old_merged_run` writes such a run and blanks the cells of its
rows, for the tests of ``mosaic reconcile`` and of the inventory's pending note.
:class:`JoinDownstream` reads the consumer's run, for the tests of what a run that
reconcile cannot mark passes on to the runs that read it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.params import Params
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    Inputs,
    Result,
    TrackInput,
)
from tests.helpers.results import result_run_root


class _TracksStep:
    """The protocol of a step that carries no state, over the tracks."""

    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Params(Params):
        pass

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        _ = run_root, artifact_paths, dependency_lookups
        return True

    def fit(self, inputs: InputStream) -> None:
        _ = inputs

    def save_state(self, run_root: Path) -> None:
        _ = run_root


class JoinSource(_TracksStep):
    """Writes one payload column and an ``arena`` for each ``(frame, id)``."""

    name = "test-join-source"
    column = "value"

    class Inputs(Inputs[TrackInput]):
        pass

    def __init__(
        self,
        inputs: JoinSource.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: JoinSource.Inputs = inputs or self.Inputs(("tracks",))
        self.params: _TracksStep.Params = self.Params.from_overrides(params)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                self.column: df["feat_a"],
                "arena": "a",
            }
        )


class JoinAlpha(JoinSource):
    name = "test-join-alpha"
    column = "alpha"


class JoinBeta(JoinSource):
    name = "test-join-beta"
    column = "beta"


class JoinConsumer(_TracksStep):
    """Reads the runs of :class:`JoinAlpha` and :class:`JoinBeta` and sums their columns.

    :attr:`apply_calls` counts the entries it computed.
    """

    name = "test-join-consumer"

    class Inputs(Inputs[TrackInput | Result]):
        pass

    def __init__(
        self,
        inputs: JoinConsumer.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: JoinConsumer.Inputs = inputs
        self.params: _TracksStep.Params = self.Params.from_overrides(params)
        self.apply_calls = 0

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.apply_calls += 1
        return pd.DataFrame(
            {"frame": df["frame"], "id": df["id"], "total": df["alpha"] + df["beta"]}
        )


class JoinDownstream(_TracksStep):
    """Reads the run of :class:`JoinConsumer` alone and doubles its total."""

    name = "test-join-downstream"

    class Inputs(Inputs[Result]):
        pass

    def __init__(
        self,
        inputs: JoinDownstream.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: JoinDownstream.Inputs = inputs
        self.params: _TracksStep.Params = self.Params.from_overrides(params)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"frame": df["frame"], "id": df["id"], "double": df["total"] * 2}
        )


JOIN_STEPS = (JoinAlpha, JoinBeta, JoinConsumer, JoinDownstream)
"""The steps to register for ``mosaic reconcile`` to rebuild their runs."""


def summary_cells(ds: Dataset, result: Result) -> dict[str, str]:
    """Each entry's recorded ``consumed_summary_join`` in the run *result* names."""
    assert result.run_id is not None
    index = feature_index(feature_index_path(ds, result.feature)).read(
        run_id=result.run_id
    )
    return dict(zip(index["sequence"], index["consumed_summary_join"], strict=True))


def old_merged_run(ds: Dataset, *, unjoined: str) -> tuple[Result, JoinConsumer.Inputs]:
    """Run :class:`JoinConsumer` over two runs sharing ``arena``, with its cells blank.

    The first run of the step records ``arena`` for each entry whose two inputs
    both hold it. The ``arena`` column of the first input is dropped from the
    entry *unjoined* before the step runs, so that entry joins no summary key.
    Both cells are then blanked, as a run before the rule left them: the other
    entry is one the current rule merges differently, and *unjoined* one it
    merges as the old rule did.

    Returns:
        The consumer's result and its inputs, which a fresh :class:`JoinConsumer`
        takes to run the same step again.
    """
    alpha = run_feature(ds, JoinAlpha())
    beta = run_feature(ds, JoinBeta())
    table = result_run_root(ds, alpha) / f"{unjoined}.parquet"
    pd.read_parquet(table).drop(columns="arena").to_parquet(table)
    inputs = JoinConsumer.Inputs((alpha, beta))
    result = run_feature(ds, JoinConsumer(inputs))
    assert result.run_id is not None
    _ = feature_index(feature_index_path(ds, result.feature)).update_cells(
        result.run_id, {"consumed_summary_join": ""}
    )
    return result, inputs
