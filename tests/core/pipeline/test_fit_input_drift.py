"""A feature input carries its upstream's fit state down a chain.

The chain is the k-means fit, its apply through ``model``, a per-entry feature
over the apply's output and a second one over that. A fit written again moves
the apply's ``consumed_model_state``, which the feature reading the apply
records in its inputs cell, and the one below it in its own, so each recomputes
in turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.composition import input_part
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.index import feature_index_path
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import Inputs, Result
from tests.helpers import (
    KMeansChain,
    PlainFeature,
    add_tracks_variant,
    drifted,
    kmeans_apply,
    kmeans_fit_and_apply,
    make_dataset,
    model_cells,
    move_tracks,
    refit_kmeans,
    refusals,
    run_step,
)

VARIANT: Final = "convert-trex.0.2-1111111111"
SEQUENCES: Final = ("seq_a", "seq_b")
BOTH: Final = (("", "seq_a"), ("", "seq_b"))
WORKERS: Final = 2
MODE: Final = "thread"


class _BelowApply(PlainFeature):
    """A per-entry feature over the output of the apply."""

    name = "fit-below-apply"
    accepts_overlap = False
    emits: EmitsLevel = "as-input"


class _BelowThat(_BelowApply):
    """A per-entry feature over the output of :class:`_BelowApply`."""

    name = "fit-below-that"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two track tables of one variant, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, *SEQUENCES, std_format="mosaic_v1")
    return dataset


def _read(upstream: Result[str]) -> Inputs:
    """The inputs of a feature reading the run of *upstream*."""
    return Inputs((Result(feature=upstream.feature, run_id=upstream.run_id),))


def _inputs_cells(ds: Dataset, result: Result[str]) -> dict[str, str]:
    """The ``consumed_input_composition`` cell of each own row of *result*'s run."""
    rows = pd.read_csv(
        feature_index_path(ds, result.feature), dtype=str, keep_default_na=False
    )
    rows = rows[rows["run_id"] == result.run_id]
    return dict(
        zip(
            rows["sequence"].astype(str),
            rows["consumed_input_composition"].astype(str),
            strict=True,
        )
    )


def _below(
    ds: Dataset, chain: KMeansChain, capsys: pytest.CaptureFixture[str]
) -> tuple[Result[str], Result[str], str]:
    """Run both features below the apply, and what they printed to stderr."""
    _ = capsys.readouterr()
    first = run_feature(ds, _BelowApply(_read(chain.apply)), track=False)
    second = run_feature(ds, _BelowThat(_read(first)), track=False)
    return first, second, capsys.readouterr().err


def _settled(
    ds: Dataset, capsys: pytest.CaptureFixture[str]
) -> tuple[KMeansChain, Result[str], Result[str]]:
    """The chain over tracks moved once, every run recomputed and served again."""
    chain = kmeans_fit_and_apply(ds, workers=WORKERS, mode=MODE)
    move_tracks(ds, VARIANT, SEQUENCES)
    speed = run_step(ds, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    _ = kmeans_apply(ds, chain, workers=WORKERS, mode=MODE)
    first, second, _ = _below(ds, chain, capsys)
    assert drifted(ds, chain.apply) == ()
    assert drifted(ds, first) == ()
    assert drifted(ds, second) == ()
    return chain, first, second


def test_a_feature_input_records_the_state_of_the_fit_its_upstream_read(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = model_cells(tracked, chain.apply)["seq_a"]
    assert digest

    first, second, _ = _below(tracked, chain, capsys)

    part = input_part("", "", digest)
    assert part == f"{digest}|/"
    assert _inputs_cells(tracked, first) == {"seq_a": part, "seq_b": part}
    deeper = input_part("", part, "")
    assert _inputs_cells(tracked, second) == {"seq_a": deeper, "seq_b": deeper}
    assert deeper == f"/{digest}|/"


def test_a_fit_written_again_recomputes_every_row_below_it_in_turn(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain, first, second = _settled(tracked, capsys)

    digest = refit_kmeans(tracked, chain, workers=WORKERS, mode=MODE)

    assert drifted(tracked, chain.apply) == BOTH
    assert drifted(tracked, first) == ()
    assert drifted(tracked, second) == ()

    _ = kmeans_apply(tracked, chain, workers=WORKERS, mode=MODE)
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert drifted(tracked, chain.apply) == ()
    assert drifted(tracked, first) == BOTH
    assert drifted(tracked, second) == ()

    _ = capsys.readouterr()
    first_again = run_feature(tracked, _BelowApply(_read(chain.apply)), track=False)
    err = capsys.readouterr().err
    assert refusals(err, _BelowApply.name).keys() == set(SEQUENCES)
    assert first_again.run_id == first.run_id
    assert drifted(tracked, first) == ()
    assert drifted(tracked, second) == BOTH

    _ = capsys.readouterr()
    _ = run_feature(tracked, _BelowThat(_read(first)), track=False)
    err = capsys.readouterr().err
    assert refusals(err, _BelowThat.name).keys() == set(SEQUENCES)
    assert drifted(tracked, second) == ()
    part = input_part("", "", digest)
    assert _inputs_cells(tracked, first) == {"seq_a": part, "seq_b": part}


def test_a_chain_whose_upstream_rows_record_no_state_keeps_every_cells_form(
    tracked: Dataset,
) -> None:
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    first = run_feature(tracked, _BelowApply(_read(speed)), track=False)
    second = run_feature(tracked, _BelowThat(_read(first)), track=False)

    assert set(model_cells(tracked, speed).values()) == {""}
    for result in (first, second):
        cells = _inputs_cells(tracked, result)
        assert set(cells) == set(SEQUENCES)
        assert all("|" not in cell for cell in cells.values())
        assert set(cells.values()) == {""}
        assert drifted(tracked, result) == ()
