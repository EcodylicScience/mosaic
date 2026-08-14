"""The widened "which tracks" selector -- item 9.4.

M2 built the narrow half (one variant within ``tracks/``). These pin the
widening: a track-shaped table in ``features/`` is enumerated too, the default is
the leaf of the chain rather than the newest run, and two leaves refuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.track_universe import (
    AmbiguousTrackLeaf,
    is_track_shaped,
    track_leaf,
    track_universe,
)

from tests.helpers import add_track_sequences


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="universe", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    add_track_sequences(dataset, "seq_a")
    return dataset


def _feature_run(
    ds: Dataset,
    storage: str,
    run_id: str,
    *,
    track_shaped: bool,
    consumed: list[dict[str, str]] | None = None,
) -> Path:
    """A materialized feature run, written the way ``run_feature`` writes one."""
    from mosaic.core.pipeline.index import (
        FeatureIndexRow,
        feature_index,
        feature_index_path,
    )

    root = ds.get_root("features") / storage / run_id
    root.mkdir(parents=True, exist_ok=True)
    columns: dict[str, object] = {"frame": [0, 1], "id": [0, 0]}
    if track_shaped:
        columns["X"] = [1.0, 2.0]
        columns["Y"] = [3.0, 4.0]
    else:
        columns["speed"] = [0.1, 0.2]
    output = root / "seq_a.parquet"
    pd.DataFrame(columns).to_parquet(output)
    _ = (root / "params.json").write_text(json.dumps({"_resolved": consumed or []}))

    index = feature_index(feature_index_path(ds, storage))
    index.ensure()
    index.append(
        [
            FeatureIndexRow(
                run_id=run_id,
                feature=storage,
                version="0.1",
                group="",
                sequence="seq_a",
                abs_path=Path(ds.relative_to_root(output)),
                params_hash="",
                n_rows=2,
            )
        ]
    )
    index.mark_finished(run_id)
    return output


# --- Membership ---------------------------------------------------------------


def test_a_derived_feature_is_not_track_shaped(tmp_path: Path) -> None:
    """Dropping X/Y is what makes a derived table not a track."""
    ds = _dataset(tmp_path)
    derived = _feature_run(
        ds, "speed-angvel__from__tracks", "0.1-aaa", track_shaped=False
    )

    assert not is_track_shaped(derived)
    assert [s.storage for s in track_universe(ds)] == ["tracks"]


def test_a_track_producing_feature_joins_the_universe(tmp_path: Path) -> None:
    """The whole widening: a smoothed track is still a track."""
    ds = _dataset(tmp_path)
    smoothed = _feature_run(
        ds, "trajectory-smooth__from__tracks", "0.1-bbb", track_shaped=True
    )

    assert is_track_shaped(smoothed)
    assert {s.storage for s in track_universe(ds)} == {
        "tracks",
        "trajectory-smooth__from__tracks",
    }


def test_a_run_that_wrote_nothing_cannot_be_classified(tmp_path: Path) -> None:
    """Truth-based membership needs an output; a cold run has none.

    Recorded rather than worked around: it is how ``build_manifest`` has always
    decided what a track is, and inventing an answer here would be the selector
    offering a table the manifest then rejects.
    """
    ds = _dataset(tmp_path)
    (ds.get_root("features") / "trajectory-smooth__from__tracks" / "0.1-ccc").mkdir(
        parents=True
    )

    assert [s.storage for s in track_universe(ds)] == ["tracks"]


def test_the_universe_is_stable(tmp_path: Path) -> None:
    """Filesystem order is not, and a refusal message names these."""
    ds = _dataset(tmp_path)
    for run_id in ("0.1-zzz", "0.1-aaa", "0.1-mmm"):
        _ = _feature_run(
            ds, "trajectory-smooth__from__tracks", run_id, track_shaped=True
        )

    assert track_universe(ds) == track_universe(ds)


# --- The leaf, and the refusal -------------------------------------------------


def test_the_only_variant_is_the_leaf(tmp_path: Path) -> None:
    """The ordinary dataset: the widening costs nothing."""
    ds = _dataset(tmp_path)

    leaf = track_leaf(ds)

    assert leaf.is_tracks


def test_a_chain_leaf_beats_the_variant_it_was_built_from(tmp_path: Path) -> None:
    """Smoothed-from-tracks is the leaf; the tracks variant is consumed."""
    ds = _dataset(tmp_path)
    variant = track_universe(ds)[0].run_id
    _ = _feature_run(
        ds,
        "trajectory-smooth__from__tracks",
        "0.1-bbb",
        track_shaped=True,
        consumed=[{"where": "inputs[tracks]", "feature": "tracks", "run_id": variant}],
    )

    leaf = track_leaf(ds)

    assert leaf.storage == "trajectory-smooth__from__tracks"
    assert leaf.run_id == "0.1-bbb"


def test_a_two_step_chain_resolves_to_its_end(tmp_path: Path) -> None:
    """Feature-on-feature: reading only the tracks edge would see two leaves.

    Both edges live in the same ``_resolved`` record -- a tracks variant and an
    upstream ``Result`` -- and reading only the first makes every chain of two
    look ambiguous.
    """
    ds = _dataset(tmp_path)
    variant = track_universe(ds)[0].run_id
    _ = _feature_run(
        ds,
        "trajectory-smooth__from__tracks",
        "0.1-bbb",
        track_shaped=True,
        consumed=[{"where": "inputs[tracks]", "feature": "tracks", "run_id": variant}],
    )
    _ = _feature_run(
        ds,
        "track-subsample__from__trajectory-smooth",
        "0.1-ccc",
        track_shaped=True,
        consumed=[
            {"where": "inputs[0]", "feature": "trajectory-smooth", "run_id": "0.1-bbb"}
        ],
    )

    leaf = track_leaf(ds)

    assert leaf.run_id == "0.1-ccc"


def test_two_leaves_refuse_rather_than_pick(tmp_path: Path) -> None:
    """Two chains off one variant is legitimate, and there is no tiebreak.

    The same position ``select_variant_rows`` takes for two recipes on one entry:
    guessing serves a silent wrong answer, and the refusal names both.
    """
    ds = _dataset(tmp_path)
    variant = track_universe(ds)[0].run_id
    for storage in ("trajectory-smooth__from__tracks", "track-subsample__from__tracks"):
        _ = _feature_run(
            ds,
            storage,
            "0.1-bbb",
            track_shaped=True,
            consumed=[
                {"where": "inputs[tracks]", "feature": "tracks", "run_id": variant}
            ],
        )

    with pytest.raises(AmbiguousTrackLeaf, match="2 track-shaped artifacts"):
        _ = track_leaf(ds)


def test_a_dataset_with_no_tracks_says_so(tmp_path: Path) -> None:
    """ "No leaf" and "two leaves" are different failures with different repairs."""
    manifest = new_dataset_manifest(name="bare", base_dir=tmp_path / "bare")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)

    with pytest.raises(LookupError, match="no track-shaped artifact"):
        _ = track_leaf(ds)


def test_the_default_is_not_the_newest_run(tmp_path: Path) -> None:
    """What "never the newest by modification time" forbids, as an assertion.

    ``latest_run_id`` sorts on the recorded finish timestamps, so the run written
    *last* wins there. The leaf ignores that entirely: here the newest run is the
    one that was consumed, so a clock-based default would return the wrong table
    on a dataset whose runs happened in chain order.
    """
    from mosaic.core.pipeline.index import feature_index, feature_index_path

    ds = _dataset(tmp_path)
    variant = track_universe(ds)[0].run_id
    storage = "trajectory-smooth__from__tracks"
    # The leaf, written first.
    _ = _feature_run(ds, storage, "0.1-leaf", track_shaped=True)
    # Its upstream, written second, so it is the "latest" by the clock.
    _ = _feature_run(
        ds,
        storage,
        "0.1-upstream",
        track_shaped=True,
        consumed=[{"where": "inputs[tracks]", "feature": "tracks", "run_id": variant}],
    )
    # And the leaf consumes it, which is what makes it the leaf.
    root = ds.get_root("features") / storage / "0.1-leaf"
    _ = (root / "params.json").write_text(
        json.dumps({"_resolved": [{"where": "inputs[0]", "run_id": "0.1-upstream"}]})
    )

    newest = feature_index(feature_index_path(ds, storage)).latest_run_id()
    leaf = [s for s in track_universe(ds) if s.storage == storage]
    consumed = {run for s in track_universe(ds) for run in s.consumed}

    assert newest == "0.1-upstream", "the clock favours the consumed run"
    assert [s.run_id for s in leaf if s.run_id not in consumed] == ["0.1-leaf"]
