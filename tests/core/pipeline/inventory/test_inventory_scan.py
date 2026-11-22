"""What a dataset holds, read from its indexes and its files.

Nothing in the stack answered this before: ``mosaic sequences`` is a tracks
listing, ``mosaic runs`` reports attempts, and ``mosaic features list`` is the
registry. These pin the answer for the kinds ``core`` can report on by itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.cli._features import build_feature
from mosaic.core import dataset as dataset_module
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.manifest import DatasetLink
from mosaic.core.pipeline.graph import Recipe, run_pipeline
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory import (
    Coverage,
    FeatureRunRef,
    LabelsVariantRef,
    TracksVariantRef,
    inventory,
    inventory_wire,
    reclassify,
)
from mosaic.core.pipeline.inventory._read import IndexReader
from mosaic.core.pipeline.inventory.label_series import label_series_records
from mosaic.core.pipeline.inventory.model import InventoryScope
from mosaic.core.pipeline.inventory.params import read_linked_run_params
from mosaic.core.pipeline.inventory.scan import (
    entries_held,
    entry_universe,
    narrow_target,
    reportable_universe,
    run_coverage,
    run_covers,
)
from mosaic.core.pipeline import parquet_reader
from mosaic.core.pipeline.parquet_reader import ParquetFooterMemo
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERTED_ENTRIES,
    LABELS_VARIANT,
    SOURCE_ENTRIES,
    add_labels,
    converted_source,
    copy_of,
    labelled_source,
    LossyFeature,
    ScopedFeature,
    add_feature_run,
    add_track_sequences,
    add_tracks_table,
    add_tracks_variant,
    make_dataset,
    raw_source,
    record_path_calls,
    run_fresh,
    track_table,
)

STORAGE = "speed-angvel__from__tracks"
SCOPED = "prov-scoped__from__tracks"
LOSSY = "prov-lossy__from__tracks"


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

    assert narrow_target(universe, Scope(groups=["g"])) == frozenset(
        {("g", "a"), ("g", "b")}
    )
    assert narrow_target(universe, Scope(sequences=["a"])) == frozenset(
        {("g", "a"), ("h", "a")}
    )
    assert narrow_target(universe, Scope(groups=["g"], sequences=["a"])) == frozenset(
        {("g", "a")}
    )
    assert narrow_target(universe) == universe
    assert narrow_target(universe, Scope()) == universe


def test_a_named_selector_that_lists_nothing_narrows_to_nothing() -> None:
    """The empty selection covers none, and an unset one covers everything.

    An empty list used to read here as no restriction at all. A misspelled
    scope then measured coverage against the whole dataset.
    """
    universe = frozenset({("g", "a"), ("g", "b")})

    assert narrow_target(universe, Scope(entries=[])) == frozenset()
    assert narrow_target(universe, Scope(groups=[])) == frozenset()


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


def test_the_stems_a_run_holds_account_for_entries_by_key_or_key_and_double_underscore() -> (
    None
):
    """The rule each entry is tested against, whatever the number of stems."""
    from mosaic.core.helpers import make_entry_key

    entries = frozenset(
        {
            ("", "a"),
            ("", "ab"),
            ("", "a_"),
            ("g", "s"),
            ("g", "t"),
            ("", "g"),
            ("", "x__y"),
            ("", "z"),
        }
    )
    stems = {"a__id0", "ab", "g__s", "x__y__id1", "a___q", "zz", "g__"}

    held = entries_held(stems, entries)

    expected = {
        entry
        for entry in entries
        for key in (make_entry_key(*entry),)
        if any(stem == key or stem.startswith(f"{key}__") for stem in stems)
    }
    assert held == expected
    assert ("", "z") not in held
    assert ("", "a_") in held


def test_a_footer_is_read_once_until_its_file_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    target = frozenset({("", "seq_a"), ("", "seq_b")})
    for name in ("seq_a", "seq_b"):
        pd.DataFrame({"a": [1]}).to_parquet(run_root / f"{name}.parquet", index=False)
    read = record_path_calls(monkeypatch, parquet_reader, "parquet_is_readable")
    footers = ParquetFooterMemo()

    first = run_covers(run_root, target, footers=footers)
    again = run_covers(run_root, target, footers=footers)

    assert first.covered == again.covered == target
    assert len(read) == 2

    _ = (run_root / "seq_b.parquet").write_bytes(b"torn")
    torn = run_covers(run_root, target, footers=footers)

    assert torn.covered == frozenset({("", "seq_a")})
    assert read[2:] == [run_root / "seq_b.parquet"]


def test_a_deleted_output_is_forgotten_by_the_next_scan_of_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    for name in ("seq_a", "seq_b"):
        pd.DataFrame({"a": [1]}).to_parquet(run_root / f"{name}.parquet", index=False)
    footers = ParquetFooterMemo()
    assert footers.whole_stems(run_root) == {"seq_a", "seq_b"}
    output = run_root / "seq_b.parquet"
    aside = tmp_path / "seq_b.aside"
    _ = output.replace(aside)

    assert footers.whole_stems(run_root) == {"seq_a"}

    parsed = record_path_calls(monkeypatch, parquet_reader, "parquet_is_readable")
    _ = aside.replace(output)

    assert footers.whole_stems(run_root) == {"seq_a", "seq_b"}
    assert parsed == [output]


def test_a_scan_keeps_the_stored_footer_of_an_unmoved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path, index=False)
    footers = ParquetFooterMemo()
    stored = footers.footer(path)
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    assert footers.whole_stems(tmp_path) == {"a"}

    assert footers.footer(path) is stored
    assert parsed == []


def test_a_scan_records_a_footer_for_a_later_footer_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(path, index=False)
    footers = ParquetFooterMemo()
    assert footers.whole_stems(tmp_path) == {"a"}
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    assert footers.footer(path).n_rows == 2
    assert footers.footer(path).n_rows == 2

    assert parsed == [path]


def test_a_scan_forgets_the_footer_of_a_deleted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    path = run_root / "a.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path, index=False)
    footers = ParquetFooterMemo()
    _ = footers.footer(path)
    aside = tmp_path / "a.aside"
    _ = path.replace(aside)
    assert footers.whole_stems(run_root) == set()
    _ = aside.replace(path)
    parsed = record_path_calls(monkeypatch, parquet_reader, "parse_footer")

    _ = footers.footer(path)

    assert parsed == [path]


def test_a_rebuild_after_an_index_moved_reads_no_footer_of_a_file_that_did_not(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.core.pipeline.inventory import InventoryCache

    _ = _run(scenario_dataset)
    holder = InventoryCache(scenario_dataset)
    _ = holder.get(kinds=["feature"])
    index_path = scenario_dataset.get_root("features") / STORAGE / "index.csv"
    _ = index_path.write_text(index_path.read_text() + "\n")
    read = record_path_calls(monkeypatch, parquet_reader, "parquet_is_readable")

    assert holder.revalidate().stale

    assert read == []


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


def test_a_scope_free_run_is_complete_over_the_entries_it_was_asked_for(
    scenario_dataset: Dataset,
) -> None:
    """A step scope applies a scope-free classifier to the test groups alone.

    The run has every entry it was asked for, and the rest of the dataset is
    an entry nobody asked it to cover.
    """
    feature = build_feature("speed-angvel", None, None)
    run_id = str(
        scenario_dataset.run_feature(
            feature, scope=Scope(entries=[("", "seq_a")])
        ).run_id
    )

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert record is not None
    assert record.status == "complete"
    assert record.coverage.target == frozenset({("", "seq_a")})
    assert record.orphan_rows == frozenset()
    assert record.orphan_files == frozenset()


def test_an_unscoped_run_reads_partial_once_the_dataset_gains_a_sequence(
    scenario_dataset: Dataset,
) -> None:
    """An invocation without a selector asked for every entry, later ones too.

    Running it again extends the same run id, so ``partial`` names the action.
    """
    run_id = _run(scenario_dataset)
    add_track_sequences(scenario_dataset, "seq_c")

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert record is not None
    assert record.status == "partial"
    assert record.coverage.missing == frozenset({("", "seq_c")})


def test_an_unscoped_recipe_step_reads_partial_once_the_dataset_gains_a_sequence(
    scenario_dataset: Dataset,
) -> None:
    """The step runs over the entries its plan names and still asks for all."""
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                }
            ],
        }
    )
    run_id = run_pipeline(scenario_dataset, recipe).outcomes[0].run_id
    add_track_sequences(scenario_dataset, "seq_c")

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert record is not None
    assert record.params is not None
    assert record.params.scope.asked_all
    assert record.status == "partial"
    assert record.coverage.missing == frozenset({("", "seq_c")})


def _lossy_run(ds: Dataset, *entries: str, fail_on: tuple[str, ...] = ()) -> str:
    """Run the scope-free failing stand-in over *entries*; its run id."""
    scope = Scope(entries=[("", sequence) for sequence in entries])
    return str(ds.run_feature(LossyFeature(fail_on=fail_on), scope=scope).run_id)


def test_a_scope_free_run_that_lost_an_entry_is_partial(
    scenario_dataset: Dataset,
) -> None:
    """A failed entry was asked for and its output is missing."""
    run_id = _lossy_run(scenario_dataset, "seq_a", "seq_b", fail_on=("seq_b",))

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=LOSSY, run_id=run_id)
    )

    assert record is not None
    assert record.status == "partial"
    assert record.coverage.missing == frozenset({("", "seq_b")})


def test_a_lost_entry_stays_partial_after_an_invocation_over_other_entries(
    scenario_dataset: Dataset,
) -> None:
    """The second invocation rewrites ``_scope.entries`` and keeps the first's."""
    add_track_sequences(scenario_dataset, "seq_c")
    first = _lossy_run(scenario_dataset, "seq_a", "seq_b", fail_on=("seq_b",))
    second = _lossy_run(scenario_dataset, "seq_c")

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=LOSSY, run_id=second)
    )

    assert second == first
    assert record is not None
    assert record.status == "partial"
    assert record.coverage.target == frozenset(
        {("", "seq_a"), ("", "seq_b"), ("", "seq_c")}
    )
    assert record.coverage.missing == frozenset({("", "seq_b")})


def test_a_run_recorded_before_the_asked_key_is_measured_by_its_rows_and_entries(
    scenario_dataset: Dataset,
) -> None:
    """Its ``entries`` name the latest invocation, and its rows the earlier ones."""
    _ = _lossy_run(scenario_dataset, "seq_a")
    run_id = _lossy_run(scenario_dataset, "seq_b")
    params_path = feature_run_root(scenario_dataset, LOSSY, run_id) / "params.json"
    document = json.loads(params_path.read_text())
    del document["_scope"]["asked"], document["_scope"]["asked_all"]
    _ = params_path.write_text(json.dumps(document))
    add_track_sequences(scenario_dataset, "seq_c")

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=LOSSY, run_id=run_id)
    )

    assert record is not None
    assert record.params is not None
    assert record.params.entry_scope() == {("", "seq_b")}
    assert record.status == "complete"
    assert record.coverage.target == frozenset({("", "seq_a"), ("", "seq_b")})


def _scoped_run(ds: Dataset, *entries: str, fail_on: tuple[str, ...] = ()) -> str:
    """Run the scope-dependent stand-in over *entries*; its run id."""
    scope = Scope(entries=[("", sequence) for sequence in entries])
    return str(ds.run_feature(ScopedFeature(fail_on=fail_on), scope=scope).run_id)


def test_a_scope_dependent_run_is_complete_over_its_declared_scope(
    scenario_dataset: Dataset,
) -> None:
    """Its scope is in its run id, so the rest of the dataset is another run.

    Measured against every entry it read ``partial`` for ever, and no action
    could finish it: running it wider makes a different run.
    """
    run_id = _scoped_run(scenario_dataset, "seq_a")

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=SCOPED, run_id=run_id)
    )

    assert record is not None
    assert record.status == "complete"
    assert record.coverage.target == frozenset({("", "seq_a")})


def test_a_scope_dependent_run_that_lost_an_entry_is_still_partial(
    scenario_dataset: Dataset,
) -> None:
    """A failed entry is inside the declared scope and has no output."""
    run_id = _scoped_run(scenario_dataset, "seq_a", "seq_b", fail_on=("seq_b",))

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=SCOPED, run_id=run_id)
    )

    assert record is not None
    assert record.status == "partial"
    assert record.coverage.missing == frozenset({("", "seq_b")})


def test_a_selector_outside_a_scope_dependent_runs_scope_reads_absent(
    scenario_dataset: Dataset,
) -> None:
    """Asked about an entry it never covered, the run holds none of it.

    Cut to its scope, the asked target would be empty, and an empty target reads
    as "anything at all" -- complete over an entry the run never held.
    """
    run_id = _scoped_run(scenario_dataset, "seq_a")

    record = inventory(
        scenario_dataset, kinds=["feature"], scope=Scope(sequences=["seq_b"])
    ).record(FeatureRunRef(name=SCOPED, run_id=run_id))

    assert record is not None
    assert record.status == "absent"
    assert record.coverage.target == frozenset({("", "seq_b")})


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

    Run in a subprocess, because registration is process-global: any other test
    that calls ``register_ops`` fills the registry for the whole session, and
    in-process this would pass or fail on collection order instead of on the
    behaviour.
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
    reported = json.loads(run_fresh(probe).splitlines()[-1])

    assert reported["records"] == 0
    assert reported["unavailable"] == ["tracker-run"]


def test_an_empty_dataset_reports_nothing_rather_than_raising(
    make_media_dataset: Callable[[Path], Dataset], tmp_path: Path
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


def test_a_global_fit_is_not_reported_as_damaged(scenario_dataset: Dataset) -> None:
    """Found on real datasets: every t-SNE, k-means, Ward and keypoint-MoSeq run
    read as damaged.

    A global fit writes one ``__global__.parquet`` and records a matching
    ``('', '__global__')`` index row. Treating that row as a real entry made it
    a row the per-entry file set could never contain, which reads as a row with
    no output -- the two in fact agree exactly.
    """
    import pandas as pd

    from mosaic.core.pipeline.inventory.scan import GLOBAL_ENTRY

    run_root = scenario_dataset.get_root("features") / "fit" / "0.1-aaaaaaaaaa"
    run_root.mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_parquet(run_root / "__global__.parquet", index=False)

    coverage = run_covers(run_root, frozenset(), known=frozenset({GLOBAL_ENTRY}))

    assert coverage.covers_all
    assert GLOBAL_ENTRY in coverage.present


def test_per_individual_outputs_count_for_their_entry(
    scenario_dataset: Dataset,
) -> None:
    """Found on a real keypoint-MoSeq apply run.

    A feature that splits an entry by individual writes ``<entry key>__id0``,
    ``__id1`` and so on. Matching the entry key exactly read those runs as
    holding nothing, and their index rows then looked like rows with no output.
    """
    import pandas as pd

    run_root = scenario_dataset.get_root("features") / "split" / "0.1-bbbbbbbbbb"
    run_root.mkdir(parents=True)
    for individual in range(3):
        pd.DataFrame({"a": [1]}).to_parquet(
            run_root / f"seq_a__id{individual}.parquet", index=False
        )

    coverage = run_covers(run_root, frozenset({("", "seq_a")}))

    assert coverage.covered == frozenset({("", "seq_a")})


def test_a_run_whose_outputs_are_not_parquet_is_not_called_damaged(
    scenario_dataset: Dataset,
) -> None:
    """Found on a real global t-SNE run, which stores ``.npz`` and ``.joblib``.

    Nothing here can attribute those to an entry, and that is missing evidence
    rather than contradictory evidence. Reporting a row with no output would
    claim damage on a run that is perfectly good.
    """
    import numpy as np

    run_id = _run(scenario_dataset)
    run_root = feature_run_root(scenario_dataset, STORAGE, run_id)
    for parquet in run_root.glob("*.parquet"):
        parquet.unlink()
    np.savez(run_root / "global_coords_seq=seq_a.npz", a=np.zeros(3))

    record = inventory(scenario_dataset, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert record is not None
    assert record.status != "inconsistent", (
        "attributing no files is missing evidence, not evidence of damage"
    )
    assert record.orphan_rows == frozenset()
    assert reclassify(record, record.coverage) == record.status
    held = inventory(scenario_dataset)
    every = Scope(entries=[("", "seq_a"), ("", "seq_b")])
    ref = f"feature/{STORAGE}/{run_id}"
    [narrowed] = inventory_wire(scenario_dataset, held, scope=every, ref=ref).artifacts
    assert narrowed.status == record.status


# --- a tool that read another number of frames than its media holds -----------


def _measured_variant(
    ds: Dataset, *, read: int, media: int | None, run_id: str = "measured"
) -> str:
    """One tracks variant over one entry, with both measurements recorded."""
    from mosaic.core.pipeline.tracks_index import write_tracks_row
    import pandas as pd

    out = ds.get_root("tracks") / run_id / "seq_m.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"frame": range(read), "id": [0] * read, "X": 0.0, "Y": 0.0}
    ).to_parquet(out)
    write_tracks_row(
        ds,
        run_id=run_id,
        group="",
        sequence="seq_m",
        out_path=out,
        producer="trex",
        std_format="trex_v2",
        n_rows=read,
        media_frames=media,
        frames_read=read,
        carries_tracklets=None,
    )
    return run_id


def test_a_short_frame_axis_is_named_on_the_variant(scenario_dataset: Dataset) -> None:
    """Reported in ``extra``, never as a status.

    The status set is five closed members crossing the CLI and mosaic-api's
    wire, and this is not a state of the artifact: the table is there, it is
    complete, and everything computed inside it is right. What it does not do is
    address the video its frames name.
    """
    run_id = _measured_variant(scenario_dataset, read=1782, media=1800)

    found = inventory(scenario_dataset, kinds=["tracks-variant"])
    record = next(r for r in found.records if r.run_id == run_id)

    assert record.extra["frame_axis_mismatch"] == frozenset({"seq_m"})
    assert record.extra["frame_tail_short"] == frozenset()
    assert record.status == "complete"


def test_a_known_tail_loss_is_named_apart_from_a_mismatch(
    scenario_dataset: Dataset,
) -> None:
    run_id = _measured_variant(scenario_dataset, read=1798, media=1800)

    found = inventory(scenario_dataset, kinds=["tracks-variant"])
    record = next(r for r in found.records if r.run_id == run_id)

    assert record.extra["frame_tail_short"] == frozenset({"seq_m"})
    assert record.extra["frame_axis_mismatch"] == frozenset()
    assert record.status == "complete"


def test_an_agreeing_variant_names_nothing(scenario_dataset: Dataset) -> None:
    run_id = _measured_variant(scenario_dataset, read=1800, media=1800)

    found = inventory(scenario_dataset, kinds=["tracks-variant"])
    record = next(r for r in found.records if r.run_id == run_id)

    assert record.extra["frame_axis_mismatch"] == frozenset()
    assert record.extra["frame_tail_short"] == frozenset()


@pytest.mark.parametrize(
    ("read", "finding"),
    [(1782, "frame_axis_mismatch"), (1798, "frame_tail_short")],
)
def test_a_finding_is_named_only_on_the_variant_it_belongs_to(
    scenario_dataset: Dataset, read: int, finding: str
) -> None:
    """Two variants of one entry are judged apart, from one read of the index."""
    short = _measured_variant(scenario_dataset, read=read, media=1800, run_id="short")
    whole = _measured_variant(scenario_dataset, read=1800, media=1800, run_id="whole")

    found = inventory(scenario_dataset, kinds=["tracks-variant"])
    records = {record.run_id: record for record in found.records}

    assert records[short].extra[finding] == frozenset({"seq_m"})
    assert records[whole].extra[finding] == frozenset()


def test_media_drift_is_named_only_on_the_variant_it_belongs_to(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A variant made from earlier media drifts; one made from today's does not."""
    import mosaic.core.pipeline.sequence_index as sequence_index
    from mosaic.core.pipeline.tracks_index import tracks_index_path

    for run_id in ("then", "now"):
        _ = _measured_variant(scenario_dataset, read=1800, media=1800, run_id=run_id)
    path = tracks_index_path(scenario_dataset)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    mine = rows["run_id"].isin(["then", "now"])
    rows.loc[mine, "consumed_media_composition"] = rows.loc[mine, "run_id"]
    rows.to_csv(path, index=False)

    def today(_dataset: Dataset, _entries: object) -> dict[tuple[str, str], str]:
        return {("", "seq_m"): "now"}

    monkeypatch.setattr(sequence_index, "media_compositions_for", today)

    found = inventory(scenario_dataset, kinds=["tracks-variant"])
    records = {record.run_id: record for record in found.records}

    assert records["then"].status == "complete-but-drifted"
    assert records["now"].status != "complete-but-drifted"


def test_an_unmeasured_variant_names_nothing(scenario_dataset: Dataset) -> None:
    """Absence of an answer is not evidence that the two agree."""
    run_id = _measured_variant(scenario_dataset, read=1782, media=None)

    found = inventory(scenario_dataset, kinds=["tracks-variant"])
    record = next(r for r in found.records if r.run_id == run_id)

    assert record.extra["frame_axis_mismatch"] == frozenset()


def test_the_tracks_index_is_read_once_however_many_variants_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each variant's frame axis and drift are taken from the one read."""
    import mosaic.core.pipeline.inventory.scan as scan
    import mosaic.core.pipeline.tracks_index as tracks_index

    ds = make_dataset(tmp_path / "ds")
    for variant in ("v1", "v2", "v3"):
        add_tracks_variant(ds, variant, "s1", "s2")
    read = tracks_index.read_tracks_index
    reads: list[Dataset] = []

    def counted(dataset: Dataset) -> pd.DataFrame:
        reads.append(dataset)
        return read(dataset)

    monkeypatch.setattr(tracks_index, "read_tracks_index", counted)
    monkeypatch.setattr(scan, "read_tracks_index", counted)

    found = inventory(ds, kinds=["tracks-variant"])

    assert sorted(record.run_id for record in found.records) == ["v1", "v2", "v3"]
    assert len(reads) == 1


def _index_record_reads_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: int
) -> int:
    """How many times a scan of one variant of *entries* entries builds index records."""
    import mosaic.core.pipeline.inventory.scan as scan

    ds = make_dataset(tmp_path / f"ds{entries}")
    add_tracks_variant(ds, "v1", *(f"s{index}" for index in range(entries)), n_rows=2)
    reads: list[int] = []
    real = scan.index_records

    def counted(frame: pd.DataFrame) -> list[dict[str, str]]:
        reads.append(len(frame))
        return real(frame)

    monkeypatch.setattr(scan, "index_records", counted)
    _ = inventory(ds, kinds=["tracks-variant"])
    return len(reads)


def test_a_variants_tables_are_checked_by_one_pass_over_its_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost is rows, not rows times entries, which an 8,400-entry plan cannot pay."""
    few = _index_record_reads_for(tmp_path, monkeypatch, 3)
    many = _index_record_reads_for(tmp_path, monkeypatch, 12)

    assert many == few


def test_a_copy_has_the_entry_universe_of_its_source(tmp_path: Path) -> None:
    source = converted_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")

    assert entry_universe(copy) == frozenset(CONVERTED_ENTRIES)
    assert reportable_universe(copy) == frozenset(CONVERTED_ENTRIES)


def test_a_report_over_a_copy_with_two_recipes_for_an_entry_names_the_linked_entries(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    add_tracks_table(copy, "other.0.1-bbbbbbbbbb", track_table(*CONVERTED_ENTRIES[0]))

    with pytest.raises(ValueError):
        _ = entry_universe(copy)
    assert reportable_universe(copy) == frozenset(CONVERTED_ENTRIES)


LINKED_RUN = "0.1-aaaaaaaaaa"


def _write_outputs(ds: Dataset, entries: tuple[tuple[str, str], ...]) -> list[Path]:
    """Write a readable output of each of *entries* into *ds*'s run of the storage."""
    run_root = feature_run_root(ds, STORAGE, LINKED_RUN)
    run_root.mkdir(parents=True, exist_ok=True)
    outputs = [run_root / f"{make_entry_key(*entry)}.parquet" for entry in entries]
    for output in outputs:
        pd.DataFrame({"a": [1]}).to_parquet(output, index=False)
    return outputs


def test_a_copy_covers_its_target_from_the_sources_run_root(tmp_path: Path) -> None:
    source = converted_source(tmp_path / "source")
    _ = _write_outputs(source, CONVERTED_ENTRIES)
    copy = copy_of(source, tmp_path / "copy")
    target = entry_universe(copy)

    coverage = run_coverage(copy, STORAGE, LINKED_RUN, target)

    assert coverage.target == target
    assert coverage.covered == frozenset(CONVERTED_ENTRIES)


def test_a_copy_covers_its_target_from_its_own_root_and_the_sources(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    _ = _write_outputs(source, CONVERTED_ENTRIES[1:])
    copy = copy_of(source, tmp_path / "copy")
    _ = _write_outputs(copy, CONVERTED_ENTRIES[:1])
    target = entry_universe(copy)

    assert not run_covers(
        feature_run_root(copy, STORAGE, LINKED_RUN), target
    ).is_satisfied
    coverage = run_coverage(copy, STORAGE, LINKED_RUN, target)

    assert coverage.is_satisfied
    assert coverage.covered == frozenset(CONVERTED_ENTRIES)


def test_a_footer_memo_reads_each_linked_footer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = converted_source(tmp_path / "source")
    linked = _write_outputs(source, CONVERTED_ENTRIES[1:])
    copy = copy_of(source, tmp_path / "copy")
    own = _write_outputs(copy, CONVERTED_ENTRIES[:1])
    target = entry_universe(copy)
    read = record_path_calls(monkeypatch, parquet_reader, "parquet_is_readable")
    footers = ParquetFooterMemo()

    first = run_coverage(copy, STORAGE, LINKED_RUN, target, footers=footers)
    again = run_coverage(copy, STORAGE, LINKED_RUN, target, footers=footers)

    assert first == again
    assert sorted(read) == sorted([*own, *linked])


# --- a dataset's links in its inventory -----------------------------------------

LINKED_VARIANT = "convert-trex.0.2-1111111111"
LINKED_ENTRIES = frozenset({("", "seq_a"), ("", "seq_b")})


def _computed_source(base: Path) -> tuple[Dataset, str]:
    """A dataset with two tracks tables of one variant and a feature run over both."""
    source = make_dataset(base, name="source")
    add_tracks_variant(source, LINKED_VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return source, _run(source)


def test_a_copy_lists_the_sources_runs_as_linked_records(tmp_path: Path) -> None:
    source, run_id = _computed_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    link_id = f"source-{source.uuid}"

    found = inventory(copy, kinds=["feature", "tracks-variant"])

    feature = found.record(FeatureRunRef(name=STORAGE, run_id=run_id))
    assert feature is not None
    assert feature.link_id == link_id
    assert feature.run_root == feature_run_root(source, STORAGE, run_id)
    assert not feature_run_root(copy, STORAGE, run_id).exists()
    target = entry_universe(copy)
    assert feature.coverage.target == target
    assert feature.coverage.covered == LINKED_ENTRIES
    over_copy = Coverage[Entry](target=target, present=LINKED_ENTRIES)
    assert feature.status == reclassify(feature, over_copy) == "complete"
    assert feature.linked_only == LINKED_ENTRIES
    assert feature.params == read_linked_run_params(copy, STORAGE, run_id).params
    assert feature.params is not None
    tracks = found.record(TracksVariantRef(run_id=LINKED_VARIANT))
    assert tracks is not None
    assert (tracks.link_id, tracks.run_root) == (link_id, None)
    assert tracks.status == reclassify(tracks, over_copy) == "complete"
    assert tracks.linked_only == LINKED_ENTRIES


def test_a_run_with_an_own_entry_keeps_its_own_root(tmp_path: Path) -> None:
    source, run_id = _computed_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    add_tracks_variant(copy, LINKED_VARIANT, "seq_c", std_format="mosaic_v1")

    assert _run(copy) == run_id
    found = inventory(copy, kinds=["feature", "tracks-variant"])

    feature = found.record(FeatureRunRef(name=STORAGE, run_id=run_id))
    assert feature is not None
    assert feature.run_root == feature_run_root(copy, STORAGE, run_id)
    assert feature.link_id == ""
    assert feature.linked_only == LINKED_ENTRIES
    assert feature.coverage.covered == LINKED_ENTRIES | {("", "seq_c")}
    assert feature.status == "complete"
    tracks = found.record(TracksVariantRef(run_id=LINKED_VARIANT))
    assert tracks is not None
    assert (tracks.link_id, tracks.linked_only) == ("", LINKED_ENTRIES)


def test_a_linked_entry_the_copy_converted_again_is_drift(tmp_path: Path) -> None:
    source, run_id = _computed_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    add_tracks_table(
        copy,
        LINKED_VARIANT,
        track_table("", "seq_a"),
        std_format="mosaic_v1",
        consumed_composition="converted-again",
    )

    feature = inventory(copy, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert feature is not None
    assert feature.drift == (("", "seq_a"),)
    assert feature.status == "complete-but-drifted"
    assert (
        inventory(source, kinds=["feature"]).status(
            FeatureRunRef(name=STORAGE, run_id=run_id)
        )
        == "complete"
    )


def test_a_linked_series_is_the_record_its_dataset_lists(tmp_path: Path) -> None:
    source = raw_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    scope = InventoryScope(kinds=frozenset({"label-series"}))
    [held] = label_series_records(source, scope, IndexReader())

    [record] = inventory(copy, kinds=["label-series"]).of_kind("label-series")

    assert record.ref == held.ref
    assert record.link_id == f"source-{source.uuid}"
    assert (record.coverage, record.status) == (held.coverage, held.status)


def test_an_inventory_reads_one_level_of_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    far = make_dataset(tmp_path / "far", name="far")
    _ = add_feature_run(far, "far-store", "0.1-3333333333")
    near = make_dataset(tmp_path / "near", name="near")
    _ = add_feature_run(near, "near-store", "0.1-2222222222")
    _ = near.add_link(DatasetLink(id="far", path=str(far.base_dir)))
    reading = make_dataset(tmp_path / "reading", name="reading")
    _ = add_feature_run(reading, "own-store", "0.1-1111111111")
    _ = reading.add_link(DatasetLink(id="near", path=str(near.base_dir)))
    fresh = Dataset(manifest_path=reading.manifest_path).load()
    loaded: list[Path] = []
    resolved: list[Path] = []
    load = Dataset.load
    resolve = dataset_module.resolve_manifest_path

    def spy_load(self: Dataset, ensure_roots: bool = True) -> Dataset:
        loaded.append(Path(self.manifest_path).resolve())
        return load(self, ensure_roots)

    def spy_resolve(path: Path) -> Path:
        resolved.append(Path(path).resolve())
        return resolve(path)

    monkeypatch.setattr(Dataset, "load", spy_load)
    monkeypatch.setattr(dataset_module, "resolve_manifest_path", spy_resolve)

    found = inventory(fresh)

    assert {record.name for record in found.of_kind("feature")} == {
        "own-store",
        "near-store",
    }
    assert found.errors == ()
    far_base = far.base_dir.resolve()
    assert not any(path.is_relative_to(far_base) for path in [*loaded, *resolved])


def test_a_mismatched_link_costs_its_kinds_and_keeps_the_own_records(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds", name="ds")
    _ = add_feature_run(ds, "own-store", "0.1-1111111111")
    other = make_dataset(tmp_path / "other", name="other")
    _ = ds.add_link(
        DatasetLink(id="other", path=str(other.base_dir), provides=("label-series",))
    )
    for child in other.base_dir.iterdir():
        if child.is_file():
            child.unlink()
    _ = make_dataset(other.base_dir, name="impostor")
    fresh = Dataset(manifest_path=ds.manifest_path).load()

    found = inventory(fresh, kinds=["feature", "label-series"])

    [error] = found.errors
    assert error.startswith("label-series: ")
    assert "was linked to dataset" in error
    assert [record.name for record in found.of_kind("feature")] == ["own-store"]


def test_a_mismatched_link_of_every_kind_keeps_the_own_records(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds", name="ds")
    add_tracks_variant(ds, LINKED_VARIANT, "seq_a", std_format="mosaic_v1")
    _ = add_feature_run(ds, "own-store", "0.1-1111111111")
    other = make_dataset(tmp_path / "other", name="other")
    _ = ds.add_link(DatasetLink(id="other", path=str(other.base_dir)))
    for child in other.base_dir.iterdir():
        if child.is_file():
            child.unlink()
    _ = make_dataset(other.base_dir, name="impostor")
    fresh = Dataset(manifest_path=ds.manifest_path).load()

    found = inventory(fresh, kinds=["feature", "tracks-variant", "label-series"])

    assert [error.split(": ", 1)[0] for error in found.errors] == [
        "feature",
        "label-series",
        "tracks-variant",
    ]
    assert all("was linked to dataset" in error for error in found.errors)
    assert [record.name for record in found.of_kind("feature")] == ["own-store"]
    assert [record.run_id for record in found.of_kind("tracks-variant")] == [
        LINKED_VARIANT
    ]


def test_a_linked_run_root_comes_with_the_linked_index_path(tmp_path: Path) -> None:
    """An own index row whose run directory is gone yields to the linked record."""
    import shutil

    source, run_id = _computed_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    add_tracks_variant(copy, LINKED_VARIANT, "seq_c", std_format="mosaic_v1")
    assert _run(copy) == run_id
    shutil.rmtree(feature_run_root(copy, STORAGE, run_id))
    held = inventory(source, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )
    assert held is not None

    feature = inventory(copy, kinds=["feature"]).record(
        FeatureRunRef(name=STORAGE, run_id=run_id)
    )

    assert feature is not None
    assert feature.link_id == f"source-{source.uuid}"
    assert feature.run_root == held.run_root
    assert feature.index_path == held.index_path


LINKED_LABELS = LabelsVariantRef(label_kind="behavior", run_id=LABELS_VARIANT)
"""The converted labels variant of :func:`labelled_source`."""


def test_a_copy_lists_its_sources_labels_variant_over_the_variants_rows(
    tmp_path: Path,
) -> None:
    source = labelled_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    first = SOURCE_ENTRIES[0]

    whole = inventory(copy, kinds=["labels-variant"]).record(LINKED_LABELS)
    scoped = inventory(
        copy, kinds=["labels-variant"], scope=Scope(entries=[first])
    ).record(LINKED_LABELS)

    assert whole is not None and scoped is not None
    assert whole.link_id == f"source-{source.uuid}"
    assert whole.status == "complete"
    assert whole.coverage.target == frozenset(SOURCE_ENTRIES)
    assert (scoped.coverage.target, scoped.status) == (frozenset({first}), "complete")


def test_an_own_labels_row_of_one_entry_keeps_the_own_record(tmp_path: Path) -> None:
    source = labelled_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    first = SOURCE_ENTRIES[0]
    _ = add_labels(copy, *first, [(0, 0, 1, 1)])
    [own] = inventory(copy.without_links({f"source-{source.uuid}"})).of_kind(
        "labels-variant"
    )

    record = inventory(copy, kinds=["labels-variant"]).record(LINKED_LABELS)

    assert record is not None
    assert (record.link_id, record.run_root) == ("", own.run_root)
    assert record.coverage.covered == frozenset(SOURCE_ENTRIES)
    assert record.status == "complete"


def test_an_own_labels_row_whose_file_is_missing_counts_its_entry_absent(
    tmp_path: Path,
) -> None:
    source = labelled_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    first, second = SOURCE_ENTRIES
    add_labels(copy, *first, [(0, 0, 1, 1)]).unlink()

    record = inventory(copy, kinds=["labels-variant"]).record(LINKED_LABELS)

    assert record is not None
    assert record.coverage.target == frozenset(SOURCE_ENTRIES)
    assert record.coverage.covered == frozenset({second})
    assert record.status != "complete"
