"""Whether two inputs of a step may both have a summary key, read from declarations.

A run asks it before it reads any input footer for the cache check. The answer
comes from each input's producer: the dataset's tracks as the registered schemas
describe them, a feature as its declaration, and a producer that declares nothing
as one that may provide any column.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from mosaic.core.params import Params
from mosaic.core.pipeline.column_contract import (
    Column,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.graph.columns import may_share_summary_key
from mosaic.core.pipeline.types import Inputs, InputStream, Result, TrackInput
from mosaic.core.schema import ARENA_COLUMN
from tests.helpers import registered_features


class _Plain:
    name = "test-sharing-plain"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "individual"
    parallelizable = True
    accepts_overlap = False
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ()
    requires: ClassVar[tuple[Requirement, ...]] = ()
    provides: ClassVar[Provides | None] = None

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: _Plain.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs or self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)

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


class _Quiet(_Plain):
    """Declares its columns, and none of them is a summary key."""

    name = "test-sharing-quiet"
    requires: ClassVar[tuple[Requirement, ...]] = ()
    provides: ClassVar[Provides | None] = Provides(
        adds=(Column("frame"), Column("id"), Column("value"))
    )


class _Arenic(_Plain):
    """Declares an ``arena`` column that it keeps where its input has it."""

    name = "test-sharing-arenic"
    requires: ClassVar[tuple[Requirement, ...]] = ()
    provides: ClassVar[Provides | None] = Provides(
        adds=(Column("frame"), Column("id"), Column("count")),
        conditional=kept(ARENA_COLUMN),
    )


class _Silent(_Plain):
    """Declares no columns, so its output may hold any."""

    name = "test-sharing-silent"


_PRODUCERS = (_Plain, _Quiet, _Arenic, _Silent)


@pytest.fixture(autouse=True)
def registered() -> Iterator[None]:
    """Put the stand-in producers in the registry for one test."""
    with registered_features(*_PRODUCERS):
        yield


class _Consumer(_Plain):
    """A step that reads *items*, which is all the answer looks at."""

    def __init__(self, *items: str, name: str = "test-sharing-consumer") -> None:
        super().__init__(
            _Plain.Inputs(
                tuple(
                    item
                    if item == "tracks"
                    else Result(
                        feature=f"{item}__from__tracks", run_id=f"0.1-{position}"
                    )
                    for position, item in enumerate(items)
                )
            )
        )
        self.name = name


def test_a_step_of_one_input_answers_no() -> None:
    assert not may_share_summary_key(_Consumer(_Arenic.name))


def test_two_producers_that_may_hold_a_summary_key_answer_yes() -> None:
    assert may_share_summary_key(_Consumer(_Arenic.name, _Silent.name))


def test_a_producer_that_declares_no_column_may_provide_any() -> None:
    assert may_share_summary_key(_Consumer(_Silent.name, _Silent.name))


def test_producers_that_declare_no_summary_key_answer_no() -> None:
    assert not may_share_summary_key(_Consumer(_Quiet.name, _Quiet.name))


def test_one_producer_that_may_hold_the_key_is_not_two() -> None:
    assert not may_share_summary_key(_Consumer(_Quiet.name, _Arenic.name))


def test_the_tracks_may_hold_the_summary_key_beside_a_producer_that_may() -> None:
    assert may_share_summary_key(_Consumer("tracks", _Arenic.name))
    assert not may_share_summary_key(_Consumer("tracks", _Quiet.name))


def test_a_cross_join_consumer_answers_no() -> None:
    consumer = _Consumer(_Arenic.name, _Arenic.name, name="interaction-crop-pipeline")

    assert not may_share_summary_key(consumer)
