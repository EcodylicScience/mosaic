"""One rule decides which run an unpinned reference reads.

There were three, and one of them claimed in its docstring to be the others:
``resolve._latest_run_id`` sorted on the recorded ``finished_at`` / ``started_at``
strings and asserted it "uses the same rule as every consumer of an unpinned
reference ... so pinning cannot change *which* run a run would have read", while the
query path walked the chain -- and that walk fell back to the clock twice, whenever a
feature's runs were not track-shaped, which is every ordinary derived feature.

So pinning could change which run a run read, on exactly the dataset the chain walk
exists for. That is the defect pinning was introduced to prevent.

The rule is chain-aware rather than leaf-always. Sibling runs -- the ordinary "re-ran
it with one parameter changed" state -- are not an ambiguous chain, and leaf-always
would make every such dataset start raising instead of answering. Two leaves of a
*real* chain still refuse.
"""

from __future__ import annotations

import json

import pandas as pd

from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.track_universe import current_run_id


def _run(ds: object, storage: str, run_id: str, *, consumes: str = "") -> None:
    """Register one run of *storage*, optionally recording an upstream edge."""
    root = ds.get_root("features") / storage / run_id  # pyright: ignore[reportAttributeAccessIssue]
    root.mkdir(parents=True, exist_ok=True)
    resolved = (
        [{"where": "inputs[0]", "feature": storage, "run_id": consumes}]
        if consumes
        else []
    )
    _ = (root / "params.json").write_text(json.dumps({"_resolved": resolved}))
    index = feature_index(feature_index_path(ds, storage))  # pyright: ignore[reportArgumentType]
    index.ensure()
    path = root / "out.parquet"
    pd.DataFrame({"frame": [0], "id": [0]}).to_parquet(path)
    rows = pd.read_csv(index.path) if index.path.exists() else pd.DataFrame()
    new = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "feature": storage,
                "version": "0.1",
                "group": "",
                "sequence": "s1",
                "params_hash": run_id.split("-")[-1],
                "abs_path": str(path),
                "started_at": f"2026-01-0{len(rows) + 1}T00:00:00+00:00",
                "finished_at": f"2026-01-0{len(rows) + 1}T01:00:00+00:00",
            }
        ]
    )
    pd.concat([rows, new], ignore_index=True).to_csv(index.path, index=False)


def test_two_sibling_runs_resolve_by_recorded_time(scenario_dataset: object) -> None:
    """No edges among them, so they are siblings and the clock answers.

    Leaf-always would see two unconsumed runs and refuse, which would break the
    commonest state a dataset is in.
    """
    _run(scenario_dataset, "sib", "0.1-aaaaaaaaaa")
    _run(scenario_dataset, "sib", "0.1-bbbbbbbbbb")

    assert current_run_id(scenario_dataset, "sib") == "0.1-bbbbbbbbbb"  # pyright: ignore[reportArgumentType]


def test_a_chain_resolves_to_its_leaf_even_when_not_track_shaped(
    scenario_dataset: object,
) -> None:
    """The old walk required a track-shaped, materialised run; this does not.

    The leaf here is written *first*, so the clock would pick the other one -- which
    is what makes this the assertion that the rule is not the clock.
    """
    _run(scenario_dataset, "chain", "0.1-leafleafle", consumes="0.1-upstreamup")
    _run(scenario_dataset, "chain", "0.1-upstreamup")

    assert current_run_id(scenario_dataset, "chain") == "0.1-leafleafle"  # pyright: ignore[reportArgumentType]


def test_pinning_reads_the_run_the_query_path_would_have_read(
    scenario_dataset: object,
) -> None:
    """The parity the old docstring asserted and did not have."""
    from mosaic.core.pipeline.manifest import _leaf_run_of
    from mosaic.core.pipeline.resolve import _latest_run_id

    _run(scenario_dataset, "both", "0.1-leafleafle", consumes="0.1-upstreamup")
    _run(scenario_dataset, "both", "0.1-upstreamup")

    assert _latest_run_id(scenario_dataset, "both") == _leaf_run_of(  # pyright: ignore[reportArgumentType]
        scenario_dataset,  # pyright: ignore[reportArgumentType]
        "both",
    )


def test_an_unrun_feature_still_resolves_to_nothing(scenario_dataset: object) -> None:
    """A missing index is "has not run here", not an error to the pinning pass."""
    from mosaic.core.pipeline.resolve import _latest_run_id

    assert _latest_run_id(scenario_dataset, "never-run") is None  # pyright: ignore[reportArgumentType]
