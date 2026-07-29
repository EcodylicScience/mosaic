"""Item 5.3: what a run's state was fitted over, as distinct from applied to.

The distinction only bites for a **params-level fitter** -- scope-free, so every
apply scope resolves to one ``run_id`` and one run root. A ``scope_dependent``
feature gets a different identifier for a different scope and therefore cannot
reuse another scope's state at all, which is why item 5.3 calls itself narrower
than it looks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.fit_scope import (
    FIT_SCOPE_NAME,
    fit_and_apply_scopes,
    read_fit_scope,
)
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import Inputs, Params


class _P(Params):
    pass


class _ScopeFreeFit:
    """Fits from its stream but is scope-free -- the shape item 5.3 is about.

    Not a violation of the fit-source rule (item 1.4): the rule's static check
    walks the registered library, and this is a test stub standing in for a
    params-level fitter whose training set arrives through a pinned reference.
    What matters here is the *runtime* shape -- one run root, many apply scopes.
    """

    name = "fit-scope-probe"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.inputs = Inputs(("tracks",))
        self.params = _P()
        self.fits = 0

    def load_state(
        self, run_root: Path, artifact_paths: object, dependency_lookups: object
    ) -> bool:
        return (run_root / "state.txt").exists()

    def fit(self, inputs: object) -> None:
        self.fits += 1

    def save_state(self, run_root: Path) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        _ = (run_root / "state.txt").write_text("fitted")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def _run(ds: Dataset, feature: _ScopeFreeFit, sequences: list[str] | None):
    return run_feature(ds, feature, sequences=sequences)


def test_a_fit_records_the_scope_it_was_fitted_over(
    scenario_dataset: Dataset,
) -> None:
    feature = _ScopeFreeFit()
    result = _run(scenario_dataset, feature, ["seq_a"])

    run_root = feature_run_root(
        scenario_dataset, "fit-scope-probe__from__tracks", result.run_id
    )
    fitted = read_fit_scope(run_root)
    assert fitted is not None
    assert fitted.entries == (("", "seq_a"),)
    assert fitted.scope_dependent is False
    assert fitted.identity_scheme


def test_a_wider_apply_leaves_the_fit_scope_narrow(
    scenario_dataset: Dataset,
) -> None:
    """The finding this item exists for, pinned in both directions.

    ``params.json`` is written unconditionally at the top of every run, so its
    ``_scope.entries`` widens to whatever ran last. The fit record is written
    only when a fit actually ran, so it stays what it says it is. Asserting only
    the second half would pass against a record that simply never updated.
    """
    feature = _ScopeFreeFit()
    first = _run(scenario_dataset, feature, ["seq_a"])
    assert feature.fits == 1

    second = _run(scenario_dataset, feature, ["seq_a", "seq_b"])
    assert second.run_id == first.run_id, "a scope-free feature must not re-identify"
    assert feature.fits == 1, "the state was reloaded, so no second fit ran"

    run_root = feature_run_root(
        scenario_dataset, "fit-scope-probe__from__tracks", second.run_id
    )
    fitted = read_fit_scope(run_root)
    assert fitted is not None
    assert fitted.entries == (("", "seq_a"),), "the fit scope followed the apply scope"

    saved = json.loads((run_root / "params.json").read_text())
    widened = {tuple(entry) for entry in saved["_scope"]["entries"]}
    assert widened == {("", "seq_a"), ("", "seq_b")}, (
        "params.json is expected to widen; that is why it cannot be the fit record"
    )


def test_the_pairing_is_derived_in_one_place(scenario_dataset: Dataset) -> None:
    """Fit and apply come back together, and a row is in both or only in apply."""
    feature = _ScopeFreeFit()
    first = _run(scenario_dataset, feature, ["seq_a"])
    _ = _run(scenario_dataset, feature, ["seq_a", "seq_b"])

    fit, apply = fit_and_apply_scopes(
        scenario_dataset, "fit-scope-probe__from__tracks", first.run_id
    )
    assert fit == frozenset({("", "seq_a")})
    assert apply == frozenset({("", "seq_a"), ("", "seq_b")})
    assert fit is not None and fit < apply, "seq_b was applied but never fitted"


def test_a_run_that_predates_the_record_reads_as_unknown(tmp_path: Path) -> None:
    """``None`` is unknown, never "fitted over nothing".

    Reading an absent record as an empty fit scope would report every run written
    before item 5.3 as having trained on nothing -- a confident wrong answer where
    an honest unknown was available.
    """
    assert read_fit_scope(tmp_path) is None


def test_a_cache_hit_does_not_rewrite_the_record(scenario_dataset: Dataset) -> None:
    """It is written in the fit branch, so a reload must leave it alone."""
    feature = _ScopeFreeFit()
    result = _run(scenario_dataset, feature, ["seq_a"])
    run_root = feature_run_root(
        scenario_dataset, "fit-scope-probe__from__tracks", result.run_id
    )
    before = (run_root / FIT_SCOPE_NAME).stat().st_mtime_ns

    _ = _run(scenario_dataset, feature, ["seq_a"])

    assert (run_root / FIT_SCOPE_NAME).stat().st_mtime_ns == before
