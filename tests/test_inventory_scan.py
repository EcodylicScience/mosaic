"""What a dataset holds, read from its indexes and its files.

Nothing in the stack answered this before: ``mosaic sequences`` is a tracks
listing, ``mosaic runs`` reports attempts, and ``mosaic features list`` is the
registry. These pin the answer for the kinds ``core`` can report on by itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory import (
    FeatureRunRef,
    TracksVariantRef,
    inventory,
)
from mosaic.core.pipeline.inventory.scan import (
    entry_universe,
    narrow_target,
    run_covers,
)

STORAGE = "speed-angvel__from__tracks"


def _run(ds: Dataset) -> str:
    return str(ds.run_feature(build_feature("speed-angvel", None, None)).run_id)


# --- the universe and its narrowing ------------------------------------------


def test_the_universe_is_what_can_actually_be_processed(
    scenario_dataset: Dataset,
) -> None:
    """Rows whose table is gone are not in it, matching what a run would resolve."""
    assert entry_universe(scenario_dataset) == frozenset({("", "seq_a"), ("", "seq_b")})


def test_narrowing_intersects_every_axis() -> None:
    universe = frozenset({("g", "a"), ("g", "b"), ("h", "a")})

    assert narrow_target(universe, groups=["g"]) == frozenset({("g", "a"), ("g", "b")})
    assert narrow_target(universe, sequences=["a"]) == frozenset(
        {("g", "a"), ("h", "a")}
    )
    assert narrow_target(universe, groups=["g"], sequences=["a"]) == frozenset(
        {("g", "a")}
    )
    assert narrow_target(universe) == universe


# --- coverage over a run root -------------------------------------------------


def test_coverage_names_the_entries_rather_than_answering_yes_or_no(
    scenario_dataset: Dataset,
) -> None:
    """The change in substance: a bool cannot say 89 of 90."""
    run_id = _run(scenario_dataset)
    run_root = feature_run_root(scenario_dataset, STORAGE, run_id)
    target = frozenset({("", "seq_a"), ("", "seq_b")})

    assert run_covers(run_root, target).covered == target

    (run_root / "seq_b.parquet").unlink()
    short = run_covers(run_root, target)

    assert short.covered == frozenset({("", "seq_a")})
    assert short.missing == frozenset({("", "seq_b")})
    assert not short.is_satisfied


def test_a_global_marker_answers_for_every_entry(tmp_path: Path) -> None:
    """A global fit writes one artifact, so counting entries reports zero of ninety."""
    import pandas as pd

    run_root = tmp_path / "run"
    run_root.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(run_root / "__global__.parquet", index=False)

    coverage = run_covers(run_root, frozenset({("", "seq_a"), ("", "seq_b")}))

    assert coverage.covers_all
    assert coverage.is_satisfied


def test_an_absent_run_root_covers_nothing_rather_than_raising(tmp_path: Path) -> None:
    assert run_covers(tmp_path / "never", frozenset({("", "s")})).missing


# --- the scan -----------------------------------------------------------------


def test_a_computed_feature_run_is_reported_complete(
    scenario_dataset: Dataset,
) -> None:
    run_id = _run(scenario_dataset)

    found = inventory(scenario_dataset, kinds=["feature"])
    record = found.record(FeatureRunRef(name=STORAGE, run_id=run_id))

    assert record is not None
    assert record.status == "complete"
    assert record.coverage.covered == frozenset({("", "seq_a"), ("", "seq_b")})
    assert record.params_state == "present"
    assert record.identity_scheme


def test_a_row_naming_a_deleted_output_is_inconsistent(
    scenario_dataset: Dataset,
) -> None:
    """The index says the entry is there and disk says it is not. Damage, and
    named as damage rather than folded into "not complete"."""
    run_id = _run(scenario_dataset)
    run_root = feature_run_root(scenario_dataset, STORAGE, run_id)
    (run_root / "seq_b.parquet").unlink()

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert record is not None
    assert record.status == "inconsistent"
    assert record.orphan_rows == frozenset({("", "seq_b")})


def test_a_run_covering_some_of_the_universe_is_partial(
    scenario_dataset: Dataset,
) -> None:
    """The distinction the four-value vocabulary could not make.

    Both the index and disk agree this run holds one entry -- nothing is
    damaged. It is simply not the whole dataset, and reporting that as
    ``absent`` would say nothing ran when half of it did.
    """
    feature = build_feature("speed-angvel", None, None)
    run_id = str(scenario_dataset.run_feature(feature, entries=[("", "seq_a")]).run_id)

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert record is not None
    assert record.status == "partial"
    assert record.coverage.covered == frozenset({("", "seq_a")})
    assert record.coverage.missing == frozenset({("", "seq_b")})
    assert record.orphan_rows == frozenset()
    assert record.orphan_files == frozenset()


def test_the_converted_tracks_are_reported(scenario_dataset: Dataset) -> None:
    """The ops-free half of the answer: what this dataset was converted from."""
    found = inventory(scenario_dataset, kinds=["tracks-variant"])

    assert found.records
    for record in found.records:
        assert isinstance(record.ref, TracksVariantRef)
        assert record.status in {"complete", "partial", "inconsistent"}


def test_a_kind_with_no_contributor_is_reported_not_silently_empty(
    scenario_dataset: Dataset,
) -> None:
    """Answering "no tracker runs" to a process that never imported the producers
    would be a wrong answer where "nobody can tell you" is a true one.

    Run in a subprocess, because registration is a process-global import side
    effect: any other test that imports ``mosaic.tracking`` fills the registry
    for the whole session, and in-process this would pass or fail on collection
    order rather than on the behaviour.
    """
    probe = f"""
import json
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.inventory import inventory

ds = Dataset(manifest_path={str(scenario_dataset.manifest_path)!r}).load()
found = inventory(ds, kinds=["tracker-run"])
print(json.dumps({{
    "records": len(found.records),
    "unavailable": sorted(found.unavailable_kinds),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    reported = json.loads(completed.stdout.strip().splitlines()[-1])

    assert reported["records"] == 0
    assert reported["unavailable"] == ["tracker-run"]


def test_an_empty_dataset_reports_nothing_rather_than_raising(
    make_media_dataset, tmp_path: Path
) -> None:
    """A dataset with no artifacts is an ordinary answer, not an error.

    The transcode kinds are still reported, because "no media rows need a
    derivative" is a true statement about a dataset with no media -- and it is
    the statement that keeps an empty corpus from reading as work to do.
    """
    ds = make_media_dataset(tmp_path / "empty")

    found = inventory(ds)

    assert found.errors == ()
    assert all(record.status == "absent" for record in found.records)
    assert all(record.coverage.target == frozenset() for record in found.records)
    assert {record.ref.kind for record in found.records} <= {"media-derivative"}


def test_a_run_is_recognised_when_the_tracks_it_came_from_are_gone(
    scenario_dataset: Dataset,
) -> None:
    """Found on a real dataset whose tracks index named files on another volume.

    An output file is a ``<group>__<sequence>`` stem and that encoding does not
    invert, so an entry is only recognisable if something named it first. When
    the tracks are unresolvable the entry universe is empty, and measuring a run
    against that alone made every finished run read as holding nothing -- and
    then as ``inconsistent``, because its index rows named entries the files
    "did not" have. The run's own rows are what keep it recognisable.
    """
    run_id = _run(scenario_dataset)
    for table in (scenario_dataset.get_root("tracks")).glob("*.parquet"):
        table.unlink()

    assert entry_universe(scenario_dataset) == frozenset()

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert record is not None
    assert record.status == "complete", (
        "a run whose outputs are all present should not read as damaged because "
        "its upstream tracks moved"
    )
    assert record.orphan_rows == frozenset()
