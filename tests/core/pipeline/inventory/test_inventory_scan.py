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

import pandas as pd
import pytest

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
from mosaic.core.scope import Scope
from tests.helpers import add_tracks_variant, make_dataset

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
    run_id = str(
        scenario_dataset.run_feature(
            feature, scope=Scope(entries=[("", "seq_a")])
        ).run_id
    )

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
