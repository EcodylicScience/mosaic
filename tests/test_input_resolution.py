"""Item 1.1: an unpinned upstream is pinned before identity is computed.

``Result.run_id=None`` means "the latest run". Until this landed, that ``None``
is what ``compute_run_id`` hashed, and the resolution to a concrete run happened
afterwards, in four places, into a local that was discarded. Two runs consuming
two *different* upstream runs therefore shared one identifier and one directory,
and the second was reported as a cache hit.

The sharpest case is not the feature graph but the model graph: ``ArtifactSpec``
extends ``Result``, so a ``GlobalModelParams`` feature's ``templates`` reference
-- which *is* its training set -- inherited the same unpinned default. That is
what ``test_a_downstream_identifier_moves_when_its_unpinned_templates_move``
covers, and it is the single assertion that proves M1's invariant.

The mock features are deliberately trivial. These are tests about identity, not
about what a feature computes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.pipeline import FeatureStep, build_step_feature
from mosaic.core.pipeline.resolve import Resolution, resolve_references
from mosaic.core.pipeline.run import compute_run_id, feature_run_root, run_feature
from mosaic.core.pipeline.types import (
    Feature,
    InputRequire,
    Inputs,
    InputStream,
    NNResult,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
    TrackInput,
)
from mosaic.core.params import Params

UPSTREAM_DIR = "resolution-upstream__from__tracks"
UPSTREAM_ENTRY = "seq_a"
"""One of the two sequences ``scenario_dataset`` holds, so a reference to the
upstream run names a file rather than globbing both of its outputs."""
DOWNSTREAM_DIR = "resolution-downstream__from__tracks"


# --- Mock features -----------------------------------------------------------


class _FeatureBase:
    """The protocol members these tests do not care about."""

    version = "0.1"
    parallelizable = True
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ()

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})


class _Upstream(_FeatureBase):
    """Produces runs to point at. ``knob`` exists only to mint a second one."""

    name = "resolution-upstream"

    class Inputs(Inputs[TrackInput]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        knob: int = 0

    def __init__(self, knob: int = 0) -> None:
        self.inputs = self.Inputs(("tracks",))
        self.params = self.Params(knob=knob)


class _Downstream(_FeatureBase):
    """Reads tracks, but carries a params-level reference to an upstream run.

    The same shape as every ``GlobalModelParams`` feature: the artifact
    reference in params is the training set, and it defaults to unpinned.
    """

    name = "resolution-downstream"

    class Inputs(Inputs[TrackInput]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        templates: ParquetArtifact | None = None

    def __init__(self, templates: ParquetArtifact | None = None) -> None:
        self.inputs = self.Inputs(("tracks",))
        self.params = self.Params(templates=templates)


def _templates_ref(run_id: str | None = None) -> ParquetArtifact:
    """An artifact reference to the upstream feature, unpinned by default.

    "Unpinned" here is about ``run_id`` and nothing else. ``pattern`` is spelled
    because the upstream writes one output per sequence and the fixture has two,
    so a derived ``*.parquet`` names no single file and resolution refuses it --
    a different question, answered in ``test_artifact_resolution``.
    """
    return ParquetArtifact(
        feature=UPSTREAM_DIR,
        run_id=run_id,
        pattern=f"{UPSTREAM_ENTRY}.parquet",
        load=ParquetLoadSpec(),
    )


class _WithInputReference(_FeatureBase):
    """Consumes an upstream run through ``inputs`` rather than through params."""

    name = "resolution-consumer"

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        pass

    def __init__(self, run_id: str | None = None) -> None:
        self.inputs = self.Inputs((Result(feature=UPSTREAM_DIR, run_id=run_id),))
        self.params = self.Params()


def _upstream_run_id(dataset: Dataset, knob: int = 0) -> str:
    result = run_feature(dataset, _Upstream(knob=knob))
    assert result.run_id is not None
    return result.run_id


# --- Pinning ------------------------------------------------------------------


def test_an_unpinned_input_is_pinned_to_the_latest_run(
    scenario_dataset: Dataset,
) -> None:
    upstream = _upstream_run_id(scenario_dataset)
    feature = _WithInputReference()

    resolutions = resolve_references(scenario_dataset, feature)

    assert feature.inputs.root[0].run_id == upstream
    assert resolutions == (
        Resolution(where="inputs[0]", feature=UPSTREAM_DIR, run_id=upstream),
    )


def test_an_unpinned_params_reference_is_pinned_to_the_latest_run(
    scenario_dataset: Dataset,
) -> None:
    upstream = _upstream_run_id(scenario_dataset)
    feature = _Downstream(templates=_templates_ref())

    resolutions = resolve_references(scenario_dataset, feature)

    assert feature.params.templates is not None
    assert feature.params.templates.run_id == upstream
    assert resolutions == (
        Resolution(where="params.templates", feature=UPSTREAM_DIR, run_id=upstream),
    )


def test_the_latest_run_is_the_one_the_consumer_would_have_read(
    scenario_dataset: Dataset,
) -> None:
    """Pinning must not change *which* run is used, only when it is decided."""
    _ = _upstream_run_id(scenario_dataset, knob=0)
    newest = _upstream_run_id(scenario_dataset, knob=1)

    feature = _WithInputReference()
    _ = resolve_references(scenario_dataset, feature)

    assert feature.inputs.root[0].run_id == newest


def test_a_user_pin_is_never_overridden(scenario_dataset: Dataset) -> None:
    """An explicit choice outranks "latest", even when a newer run exists."""
    chosen = _upstream_run_id(scenario_dataset, knob=0)
    _ = _upstream_run_id(scenario_dataset, knob=1)

    feature = _WithInputReference(run_id=chosen)
    _ = resolve_references(scenario_dataset, feature)

    assert feature.inputs.root[0].run_id == chosen


def test_a_reference_to_a_feature_that_has_not_run_stays_unpinned(
    scenario_dataset: Dataset,
) -> None:
    """An honest None. The chain runner previews cold datasets step by step."""
    feature = _WithInputReference()

    resolutions = resolve_references(scenario_dataset, feature)

    assert feature.inputs.root[0].run_id is None
    assert resolutions == (
        Resolution(where="inputs[0]", feature=UPSTREAM_DIR, run_id=None),
    )


def test_resolution_is_idempotent(scenario_dataset: Dataset) -> None:
    upstream = _upstream_run_id(scenario_dataset)
    feature = _WithInputReference()

    first = resolve_references(scenario_dataset, feature)
    # A newer run lands between the two passes; the pin must not follow it.
    _ = _upstream_run_id(scenario_dataset, knob=1)
    second = resolve_references(scenario_dataset, feature)

    assert first == second
    assert feature.inputs.root[0].run_id == upstream


def test_an_unset_artifact_reference_is_not_a_dependency(
    scenario_dataset: Dataset,
) -> None:
    """``GlobalModelParams.templates`` defaults to an empty feature name.

    That is "no reference", not "a reference to nothing", and it must not
    appear in the report or reach the index.
    """
    feature = _Downstream(templates=ParquetArtifact(feature="", load=ParquetLoadSpec()))

    assert resolve_references(scenario_dataset, feature) == ()


def test_every_result_subclass_is_reached(scenario_dataset: Dataset) -> None:
    """``NNResult`` is a ``Result``, so the generic scan pins it too.

    A pair filter used to resolve to "latest" at load time while being absent
    from identity -- the fourth site, and the same defect as ``templates``.
    """
    upstream = _upstream_run_id(scenario_dataset)

    class _WithPairFilter(_FeatureBase):
        name = "resolution-pair-filter"

        class Inputs(Inputs[TrackInput]):
            _require: ClassVar[InputRequire] = "any"

        class Params(Params):
            pair_filter: NNResult = NNResult(feature=UPSTREAM_DIR)

        def __init__(self) -> None:
            self.inputs = self.Inputs(("tracks",))
            self.params = self.Params()

    feature = _WithPairFilter()
    _ = resolve_references(scenario_dataset, feature)

    assert feature.params.pair_filter.run_id == upstream


# --- The invariant the milestone exists for -----------------------------------


def test_a_downstream_identifier_moves_when_its_unpinned_templates_move(
    scenario_dataset: Dataset,
) -> None:
    """Re-fit the training set, re-run unpinned, get a different classifier.

    Before item 1.1 both runs minted the same identifier and the same run root,
    so the second was served from cache -- a model reported as trained on the
    new templates while holding the state fitted from the old ones. Six global
    features sit on this path (global-scaler, global-tsne, global-kmeans,
    global-ward, xgboost, lightning-action).
    """
    _ = _upstream_run_id(scenario_dataset, knob=0)
    first = run_feature(scenario_dataset, _Downstream(templates=_templates_ref()))

    _ = _upstream_run_id(scenario_dataset, knob=1)
    second = run_feature(scenario_dataset, _Downstream(templates=_templates_ref()))

    assert first.run_id != second.run_id, (
        "an unpinned templates reference is the training set; two training sets "
        "cannot share one classifier identifier"
    )
    assert second.cache_hit is False


def test_two_runs_of_one_upstream_state_share_one_identifier(
    scenario_dataset: Dataset,
) -> None:
    """The converse: nothing moved, so nothing recomputes."""
    _ = _upstream_run_id(scenario_dataset)

    first = run_feature(scenario_dataset, _Downstream(templates=_templates_ref()))
    second = run_feature(scenario_dataset, _Downstream(templates=_templates_ref()))

    assert first.run_id == second.run_id


# --- What resolution must not disturb -----------------------------------------


def test_resolution_does_not_move_the_storage_directory(
    scenario_dataset: Dataset,
) -> None:
    """Identity goes in the hash; the upstream *name* goes in the directory.

    The H2 invariant. ``storage_suffix`` reads ``item.feature`` and resolution
    writes ``item.run_id``, so they cannot collide -- asserted rather than
    argued, because the two runs below must land in one directory.
    """
    feature = _WithInputReference()
    before = feature.inputs.storage_suffix()

    _ = _upstream_run_id(scenario_dataset)
    _ = resolve_references(scenario_dataset, feature)

    assert feature.inputs.storage_suffix() == before


def test_the_tracks_literal_is_untouched(scenario_dataset: Dataset) -> None:
    """There is no tracks identity to pin yet; item 3.3 resolves it.

    Pinning it to anything that varies with which sequences are on disk would
    move a scope-free feature's identifier whenever a sequence is added, which
    workflow H5 forbids.
    """
    feature = _Upstream()

    resolutions = resolve_references(scenario_dataset, feature)

    assert feature.inputs.root == ("tracks",)
    assert resolutions == ()


def test_a_pinned_reference_survives_the_process_worker_round_trip(
    scenario_dataset: Dataset,
) -> None:
    """``run_id`` is a plain field of the dump, so the worker rebuilds it."""
    upstream = _upstream_run_id(scenario_dataset)
    feature = _WithInputReference()
    _ = resolve_references(scenario_dataset, feature)

    rebuilt = Inputs.model_validate(feature.inputs.model_dump())

    # The worker rebuilds the *base* Inputs, so the item type widens back to
    # include the tracks literal; narrowing here is what the worker's own
    # consumers do.
    item = rebuilt.root[0]
    assert isinstance(item, Result)
    assert item.run_id == upstream


# --- Provenance ---------------------------------------------------------------


def test_the_resolved_edges_are_recorded_in_params_json(
    scenario_dataset: Dataset,
) -> None:
    """Readable provenance, so an edge walk need not re-derive it (item 6.1)."""
    upstream = _upstream_run_id(scenario_dataset)
    result = run_feature(scenario_dataset, _Downstream(templates=_templates_ref()))
    assert result.run_id is not None

    run_root = feature_run_root(scenario_dataset, DOWNSTREAM_DIR, result.run_id)
    saved = json.loads((run_root / "params.json").read_text())

    assert saved["_resolved"] == [
        {"where": "params.templates", "feature": UPSTREAM_DIR, "run_id": upstream}
    ]


# --- The rule, as a check over the whole feature -------------------------------


@pytest.mark.parametrize(
    ("factory", "cold"),
    [
        (lambda: _WithInputReference(), False),
        (lambda: _Downstream(templates=_templates_ref()), False),
        (lambda: _WithInputReference(), True),
    ],
    ids=["input-reference", "params-reference", "cold-upstream"],
)
def test_no_unpinned_reference_survives_resolution(
    scenario_dataset: Dataset, factory: Callable[[], Feature], cold: bool
) -> None:
    """After the pass, every reference is pinned or its upstream does not exist.

    Stronger than deleting the four now-redundant fallbacks would be: a
    reference type the scan forgets fails here, rather than surfacing later as
    a ``None`` somewhere downstream.
    """
    if not cold:
        _ = _upstream_run_id(scenario_dataset)
    feature = factory()

    resolutions = resolve_references(scenario_dataset, feature)

    for record in resolutions:
        index = scenario_dataset.get_root("features") / record.feature / "index.csv"
        assert (record.run_id is not None) == index.exists()


class _StepFeature(_FeatureBase):
    """Built the way ``Pipeline`` builds one: ``(inputs=..., params=...)``."""

    name = "resolution-step"

    class Inputs(Inputs[TrackInput]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        templates: ParquetArtifact | None = None

    def __init__(self, inputs: object, params: dict[str, object] | None) -> None:
        self.inputs = self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)


def test_a_step_does_not_accumulate_pins_from_being_previewed(
    scenario_dataset: Dataset,
) -> None:
    """``build_step_feature`` gives each build its own params.

    Pydantic accepts an already-built model instance by reference, so a
    ``FeatureStep`` whose params dict holds a ``ParquetArtifact`` would hand the
    same object to every feature built from it. Since resolution pins in place,
    that would let a ``status()`` call freeze an upstream choice into a ``run()``
    that happens much later, and let a second ``run()`` reuse the first's pin.
    """
    _ = _upstream_run_id(scenario_dataset)
    shared = _templates_ref()
    step = FeatureStep("d", _StepFeature, {"templates": shared})

    feature = build_step_feature(step, ("tracks",))
    _ = resolve_references(scenario_dataset, feature)

    assert shared.run_id is None, "the step's own reference must stay unpinned"


def test_compute_run_id_needs_no_dataset(scenario_dataset: Dataset) -> None:
    """Resolution reads the filesystem; hashing does not.

    That split is what lets the golden corpus pin literal identifiers and lets
    the control plane predict one before spawning work.
    """
    from mosaic.core.pipeline._utils import Scope

    upstream = _upstream_run_id(scenario_dataset)
    pinned, _ = compute_run_id(
        _WithInputReference(run_id=upstream), None, None, Scope()
    )
    unpinned, _ = compute_run_id(_WithInputReference(), None, None, Scope())

    assert pinned != unpinned
