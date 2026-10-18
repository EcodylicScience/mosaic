"""``params.json`` records which attempt and which toolkit produced a run.

Without these two keys the document is strictly present-tense -- it says what a
run *is*, never when or under what -- and the run-log holding the answer is
reachable from the artifact by no path at all. Both are provenance and neither
is hashed, so they must move no identifier; and a re-address must carry them
rather than restamp them, or every artifact ends up claiming it was produced by
whichever mosaic last reconciled.
"""

from __future__ import annotations

import json
from pathlib import Path

from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory.params import RunParams, ScopeBlock
from mosaic.core.pipeline.run import build_run_params_payload
from mosaic.core.scope import Scope
from mosaic.version import DISTRIBUTION_NAME, installed_version
from tests.helpers import ScopedFeature


def _saved(ds: Dataset, storage: str, run_id: str) -> dict[str, object]:
    root = feature_run_root(ds, storage, run_id)
    return json.loads((root / "params.json").read_text())


def test_a_run_records_its_execution_and_its_toolkit(
    scenario_dataset: Dataset,
) -> None:
    feature = build_feature("speed-angvel", None, None)

    result = scenario_dataset.run_feature(feature)

    saved = _saved(scenario_dataset, "speed-angvel__from__tracks", str(result.run_id))
    assert saved["_execution_id"], "no attempt recorded on the artifact"
    assert saved["_mosaic_version"] == installed_version()


def test_the_recorded_execution_id_names_a_real_run_log(
    scenario_dataset: Dataset,
) -> None:
    """The point of the key: artifact -> run-log is a link, not a guess."""
    feature = build_feature("speed-angvel", None, None)

    result = scenario_dataset.run_feature(feature)

    saved = _saved(scenario_dataset, "speed-angvel__from__tracks", str(result.run_id))
    log = (
        Path(scenario_dataset.base_dir)
        / ".mosaic"
        / "runs"
        / f"{saved['_execution_id']}.jsonl"
    )
    assert log.exists(), f"recorded execution_id resolves to nothing: {log}"


def test_neither_key_moves_an_identifier(scenario_dataset: Dataset) -> None:
    """Provenance, never hashed. A run computed before these existed keeps its
    directory name, which is what makes this additive rather than a migration."""
    feature = build_feature("speed-angvel", None, None)

    first = scenario_dataset.run_feature(feature)
    second = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))

    assert first.run_id == second.run_id


def test_a_reconcile_carries_the_stamp_rather_than_restamping_it() -> None:
    """A re-address moves an artifact somebody else produced.

    Asserted on the payload builder directly: the reconcile path passes what it
    read off the old document, so the builder must not consult the environment
    for either value. Reading the installed version inside would make every
    re-addressed run claim the reconciling toolkit produced it.
    """
    feature = build_feature("speed-angvel", None, None)

    payload = build_run_params_payload(
        feature,
        None,
        None,
        ResolvedScope(),
        [],
        execution_id="01OLDATTEMPT",
        mosaic_version="0.1.0-ancient",
    )

    assert payload["_execution_id"] == "01OLDATTEMPT"
    assert payload["_mosaic_version"] == "0.1.0-ancient"


def test_an_unstamped_payload_reads_unknown_rather_than_guessing() -> None:
    """Empty means unknown, as it does for every other unestablishable cell."""
    feature = build_feature("speed-angvel", None, None)

    payload = build_run_params_payload(feature, None, None, ResolvedScope(), [])

    assert payload["_execution_id"] == ""
    assert payload["_mosaic_version"] == ""


def test_the_version_is_looked_up_under_the_distribution_name() -> None:
    """mosaic is imported as ``mosaic`` and distributed as ``mosaic-behavior``.

    Asking for the import name finds nothing, or finds an unrelated project of
    that name -- and either way the toolkit stamp goes quietly empty or wrong.
    """
    assert DISTRIBUTION_NAME == "mosaic-behavior"
    assert installed_version() == "" or installed_version()[0].isdigit()


def _scope_block(payload: dict[str, object]) -> ScopeBlock:
    return RunParams.model_validate(payload).scope


def test_the_payload_records_the_asked_scope_it_is_given() -> None:
    feature = build_feature("speed-angvel", None, None)
    scope = ResolvedScope(entries={("", "seq_b")})

    payload = build_run_params_payload(
        feature, None, None, scope, [], asked={("", "seq_b"), ("", "seq_a")}
    )

    block = _scope_block(payload)
    assert block.entries == [["", "seq_b"]]
    assert block.asked == [["", "seq_a"], ["", "seq_b"]]
    assert block.asked_all is False


def test_an_unspecified_asked_scope_is_this_invocations_entries() -> None:
    feature = build_feature("speed-angvel", None, None)
    scope = ResolvedScope(entries={("", "seq_a")})

    block = _scope_block(build_run_params_payload(feature, None, None, scope, []))

    assert block.asked == [["", "seq_a"]]
    assert block.asked_all is False


def test_each_invocation_adds_its_entries_to_the_asked_scope(
    scenario_dataset: Dataset,
) -> None:
    """``_scope.entries`` names the latest invocation, and ``asked`` all of them."""
    storage = "speed-angvel__from__tracks"
    first = scenario_dataset.run_feature(
        build_feature("speed-angvel", None, None), scope=Scope(sequences=["seq_a"])
    )
    second = scenario_dataset.run_feature(
        build_feature("speed-angvel", None, None), scope=Scope(sequences=["seq_b"])
    )

    block = _scope_block(_saved(scenario_dataset, storage, str(second.run_id)))
    assert second.run_id == first.run_id
    assert block.entries == [["", "seq_b"]]
    assert block.asked == [["", "seq_a"], ["", "seq_b"]]
    assert block.asked_all is False


def test_an_unscoped_invocation_asks_for_every_entry(
    scenario_dataset: Dataset,
) -> None:
    """Later invocations over a subset keep it."""
    storage = "speed-angvel__from__tracks"
    _ = scenario_dataset.run_feature(build_feature("speed-angvel", None, None))
    result = scenario_dataset.run_feature(
        build_feature("speed-angvel", None, None), scope=Scope(sequences=["seq_a"])
    )

    block = _scope_block(_saved(scenario_dataset, storage, str(result.run_id)))
    assert block.asked_all is True


def test_an_unscoped_scope_dependent_run_asks_for_its_own_entries(
    scenario_dataset: Dataset,
) -> None:
    """Its scope is hashed into its id. More entries make another run."""
    result = scenario_dataset.run_feature(ScopedFeature())

    block = _scope_block(
        _saved(scenario_dataset, "prov-scoped__from__tracks", str(result.run_id))
    )
    assert block.asked == [["", "seq_a"], ["", "seq_b"]]
    assert block.asked_all is False
