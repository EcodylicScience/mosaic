"""One entity fails, and the graph above and below it behaves differently.

The case the whole partial-failure model exists for, driven end to end through a
request. A feature that raises on one sequence and succeeds on the others gives
the two shapes at once:

* a **scope-free** consumer proceeds over what there is, because its outputs are
  per entry and the missing one arrives later under the same identifier;
* a **scope-dependent** consumer refuses, because over less it produces *one*
  artifact that is not the artifact anyone asked for, under a name saying it is.

``allow_partial`` is the only way past the second, and it is a scientific
decision rather than a maintenance one -- which is why it is recorded on the
request and scoped to it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    QUARANTINE_AFTER,
    CoverageShortfall,
    FileFailureStore,
    Recipe,
    execute_step,
    submit_request,
)
from mosaic.core.pipeline.types import (
    Inputs,
    InputStream,
    TrackInput,
)
from mosaic.core.params import Params
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import add_tracks_variant, make_dataset

DOOMED = "seq_b"
VARIANT = "convert-trex.0.2-1111111111"
STORAGE = "test-partial-one__from__tracks"

type Document = dict[str, object]


class FailsOnOneSequence:
    """Raises for one sequence and succeeds for every other.

    Registered by name so a recipe can wire it, because the point is to drive the
    real graph rather than to call ``run_feature`` directly: what is under test
    is what the step below does about the shortfall.
    """

    name = "test-partial-one"
    version = "0.1"
    category = "per-frame"
    emits = "as-input"
    parallelizable = True
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        doomed: str = DOOMED

    def __init__(
        self,
        inputs: FailsOnOneSequence.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: FailsOnOneSequence.Inputs = inputs or self.Inputs(("tracks",))
        self.params: FailsOnOneSequence.Params = self.Params.from_overrides(params)

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
        if str(df["sequence"].iloc[0]) == self.params.doomed:
            msg = "apply failed on purpose"
            raise RuntimeError(msg)
        # ``sequence`` is carried through because a consumer reads it: a template
        # extractor has to know which entry a row came from, and a fixture that
        # dropped it would be testing a table no feature produces.
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "sequence": df["sequence"],
                "value": df["X"] * 2,
            }
        )


@pytest.fixture(autouse=True)
def registered() -> Iterator[None]:
    """Put the fixture feature in the registry for the length of one test.

    The declaration catalog is memoized, so registering a feature after anything
    else has asked for one leaves it invisible to validation. Cleared on the way
    in and on the way out, so this suite neither reads a catalog built without it
    nor leaves one built with it.
    """
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.graph import declaration_catalog

    FEATURES["FailsOnOneSequence"] = FailsOnOneSequence
    declaration_catalog.cache_clear()
    try:
        yield
    finally:
        del FEATURES["FailsOnOneSequence"]
        declaration_catalog.cache_clear()


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Three schema-valid track tables, one of which the feature refuses."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(
        dataset, VARIANT, "seq_a", DOOMED, "seq_c", std_format="mosaic_v1"
    )
    return dataset


CONSUMER_PARAMS: dict[str, Document] = {
    "frame-aggregate": {"column": "value", "agg": "mean"},
    "extract-templates": {"n_templates": 4},
}
"""The two consumers, and the one thing that separates them.

``frame-aggregate`` is scope-free and ``extract-templates`` is scope-dependent,
which is the whole reason both appear here: they meet the same shortfall and are
right to answer it differently.
"""


def _chain(consumer: str) -> Document:
    return {
        "schema_version": 1,
        "name": "one entity fails",
        "steps": [
            {
                "id": "flaky",
                "type": "feature",
                "feature": "test-partial-one",
                "inputs": ["tracks"],
            },
            {
                "id": "consumer",
                "type": "feature",
                "feature": consumer,
                "inputs": [{"step": "flaky"}],
                "params": CONSUMER_PARAMS[consumer],
            },
        ],
    }


def _run_flaky(dataset: Dataset, consumer: str):
    """Submit the chain and run its first step, which loses one entity."""
    request = submit_request(dataset, Recipe.model_validate(_chain(consumer))).request
    return request, execute_step(dataset, request, "flaky")


def test_the_lost_entity_is_named_in_the_run_log_and_the_outcome(
    tracked: Dataset,
) -> None:
    """Under a queue the child's stderr is discarded, so the log is the record."""
    request, flaky = _run_flaky(tracked, "frame-aggregate")

    assert flaky.failed_entries == (DOOMED,)
    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("flaky"))
    assert logged is not None
    assert logged["entries_failed"] == 1
    assert logged["status"] == "finished", "the attempt did what it could"


def test_coverage_is_short_by_exactly_the_lost_entity(tracked: Dataset) -> None:
    """A partial run keeps its work: the entities that succeeded are on disk."""
    _, flaky = _run_flaky(tracked, "frame-aggregate")

    assert flaky.covered == 2
    assert flaky.target == 3
    run_root = next(Path(tracked.get_root("features")).glob(f"*/{flaky.run_id}"))
    assert not (run_root / f"{DOOMED}.parquet").exists()
    assert (run_root / "seq_a.parquet").exists()


def test_the_failure_record_counts_the_lost_entity(tracked: Dataset) -> None:
    """The one thing derived status cannot answer, and the only bound on retrying."""
    _, flaky = _run_flaky(tracked, "frame-aggregate")

    store = FileFailureStore(tracked.base_dir)
    assert store.entry_record((STORAGE, flaky.run_id, "", DOOMED)).attempts == 1
    assert store.entry_record((STORAGE, flaky.run_id, "", "seq_a")).attempts == 0


def test_a_rapid_retry_waits_rather_than_attempting_again(
    tracked: Dataset,
) -> None:
    """A backoff is a wait, and the step has nothing else left to do.

    Reported as a stall rather than as a failure: nothing has gone wrong, and an
    attempt counter that grew here would render a correct pipeline red.
    """
    _, first = _run_flaky(tracked, "frame-aggregate")
    request, again = _run_flaky(tracked, "frame-aggregate")
    _ = request

    assert again.state == "stalled"
    store = FileFailureStore(tracked.base_dir)
    assert store.entry_record((STORAGE, first.run_id, "", DOOMED)).attempts == 1


def test_a_wait_is_never_turned_into_an_exclusion(tracked: Dataset) -> None:
    """The expensive mistake: an exclusion changes what a fit is, permanently.

    An entry that needed a few more seconds must not be dropped from the request
    because a gesture meant for a hopeless one was in force.
    """
    _, _ = _run_flaky(tracked, "frame-aggregate")
    request = submit_request(
        tracked, Recipe.model_validate(_chain("frame-aggregate")), allow_partial=True
    ).request

    outcome = execute_step(tracked, request, "flaky")

    assert outcome.state == "stalled"
    store = FileFailureStore(tracked.base_dir)
    assert store.exclusions(request.request_id).entries == frozenset()


def test_a_scope_free_consumer_proceeds_over_what_there_is(
    tracked: Dataset,
) -> None:
    """Its outputs are per entry, and the missing one arrives under the same name."""
    request, _ = _run_flaky(tracked, "frame-aggregate")

    consumer = execute_step(tracked, request, "consumer")

    assert consumer.state == "ran"
    assert consumer.covered == 2


def test_a_scope_dependent_consumer_refuses(tracked: Dataset) -> None:
    """Over less it is a different artifact, under a name saying it is the same."""
    request, _ = _run_flaky(tracked, "extract-templates")

    with pytest.raises(CoverageShortfall) as raised:
        _ = execute_step(tracked, request, "consumer")

    assert raised.value.reason == "coverage_shortfall"
    assert raised.value.covered == 2
    assert raised.value.target == 3
    assert "allow_partial" in str(raised.value)


def test_the_refusal_is_a_failed_attempt_with_a_reason_and_no_new_status(
    tracked: Dataset,
) -> None:
    """``partial`` is deliberately not terminal, and neither is a refusal."""
    from mosaic.runlog import TERMINAL_STATUSES

    request, _ = _run_flaky(tracked, "extract-templates")
    with pytest.raises(CoverageShortfall):
        _ = execute_step(tracked, request, "consumer")

    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("consumer"))
    assert logged is not None
    assert logged["status"] == "failed"
    assert json.loads(logged["error_json"])["reason"] == "coverage_shortfall"
    assert TERMINAL_STATUSES == frozenset({"finished", "failed", "cancelled"})


def test_allow_partial_is_the_only_way_forward(tracked: Dataset) -> None:
    """The gesture that records the decision, and it is recorded on the request."""
    request, _ = _run_flaky(tracked, "extract-templates")
    proceeding = request.model_copy(update={"allow_partial": True})

    consumer = execute_step(tracked, proceeding, "consumer")

    assert consumer.state == "ran"
    assert consumer.run_id


def _quarantine_the_doomed_entity(dataset: Dataset, run_id: str) -> None:
    """Drive the entity past the attempt bound, without waiting out a backoff.

    Written through the store rather than by re-running, because what is under
    test is what a step does about a quarantine -- and reaching one by execution
    would mean waiting out the backoff between attempts, which is a property of
    the clock rather than of the step.
    """
    store = FileFailureStore(dataset.base_dir)
    for _ in range(QUARANTINE_AFTER):
        _ = store.note_entry_failure(
            (STORAGE, run_id, "", DOOMED), error="apply failed on purpose"
        )


def test_a_quarantined_entity_holds_a_step_back_until_it_is_answered(
    tracked: Dataset,
) -> None:
    """Past the attempt bound the entity stops being attempted at all.

    The step is then short by it, and the same explicit gesture is what proceeds
    -- so a bad sequence bounds itself rather than blocking a branch forever.
    """
    request, flaky = _run_flaky(tracked, "extract-templates")
    _quarantine_the_doomed_entity(tracked, flaky.run_id)

    with pytest.raises(CoverageShortfall) as raised:
        _ = execute_step(tracked, request, "flaky")

    assert "held back after repeated failures" in str(raised.value)
    assert raised.value.reason == "coverage_shortfall"


def test_allow_partial_records_the_excluded_entity_against_the_request(
    tracked: Dataset,
) -> None:
    """The decision is durable, per request, and narrows the whole graph.

    Narrowing the graph is the honest reading: a request run without a sequence
    is run without it everywhere, and a step below the one that excluded it must
    not silently see it again.
    """
    request, flaky = _run_flaky(tracked, "extract-templates")
    _quarantine_the_doomed_entity(tracked, flaky.run_id)
    proceeding = request.model_copy(update={"allow_partial": True})

    _ = execute_step(tracked, proceeding, "flaky")

    store = FileFailureStore(tracked.base_dir)
    assert store.exclusions(request.request_id).entries == frozenset({("", DOOMED)})
    assert store.exclusions("some-other-request").entries == frozenset()
