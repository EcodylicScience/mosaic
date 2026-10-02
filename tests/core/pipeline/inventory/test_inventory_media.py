"""Test transcode coverage, the kind without a run directory, and variant coverage.

The case a single coverage signature gets wrong, and gets wrong in the worst
direction: asked for a run directory that was never supposed to exist, a
directory-shaped check reports zero of N, so a corpus with nothing to transcode
reads as permanently incomplete and anything acting on it resubmits forever.

A media variant is the opposite case, a run directory per variant with one file
per entry and camera, and it is reported the way a frame run is.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.composition import MediaMember, media_composition
from mosaic.core.pipeline.inventory import (
    MediaDerivativeRef,
    MediaVariantRef,
    inventory,
)
from mosaic.core.pipeline.inventory import media as inventory_media
from mosaic.core.pipeline.media_index import read_media_index
from mosaic.core.pipeline.preprocess_layout import (
    media_variant_path,
    media_variant_run_root,
    media_variant_work_root,
)
from mosaic.core.pipeline.sequence_index import (
    media_compositions_for,
    write_sequence_compositions,
)
from mosaic.core.scope import Scope

from tests.helpers import (
    add_media_variant,
    add_transcode_derivative,
    finish_media_variant,
    make_dataset,
    write_media_index,
)


def _record(ds: Dataset, target: str = "analysis"):
    found = inventory(ds, kinds=["media-derivative"])
    return found.record(MediaDerivativeRef(target=target))


def test_an_already_clean_corpus_reads_complete(scenario_dataset: Dataset) -> None:
    """The load-bearing case. Nothing is missing because nothing was ever
    supposed to be produced, and no run directory is consulted to say so."""
    write_media_index(scenario_dataset, ["seq_a", "seq_b"])

    record = _record(scenario_dataset)

    assert record is not None
    assert record.status == "complete"
    assert record.coverage.missing == frozenset()
    assert record.extra["needs_transcode"] == frozenset()
    assert record.extra["needs_probe"] == frozenset()


def test_a_row_needing_a_transcode_is_named_as_needing_one(
    scenario_dataset: Dataset,
) -> None:
    """And named separately from a row needing a re-probe, because the two
    remedies differ and "incomplete" tells a user neither."""
    write_media_index(scenario_dataset, ["seq_a"])
    _mark_required(scenario_dataset.get_root("media_raw") / "index.csv")

    record = _record(scenario_dataset)

    assert record is not None
    assert record.status in {"absent", "partial"}
    assert record.extra["needs_transcode"]
    assert record.extra["needs_probe"] == frozenset()
    assert record.coverage.missing == record.extra["needs_transcode"]


def test_a_row_with_no_measurement_is_named_as_needing_a_reprobe(
    scenario_dataset: Dataset,
) -> None:
    write_media_index(scenario_dataset, ["seq_a"])
    _blank_facts(scenario_dataset.get_root("media_raw") / "index.csv")

    record = _record(scenario_dataset)

    assert record is not None
    assert record.extra["needs_probe"]
    assert record.extra["needs_transcode"] == frozenset()


def test_missing_is_exactly_the_two_remedies(scenario_dataset: Dataset) -> None:
    """No third way to be short: every missing row wants one of the two."""
    write_media_index(scenario_dataset, ["seq_a", "seq_b"])
    _mark_required(scenario_dataset.get_root("media_raw") / "index.csv")

    record = _record(scenario_dataset)

    assert record is not None
    assert record.coverage.missing == (
        record.extra["needs_transcode"] | record.extra["needs_probe"]
    )


def test_a_registered_derivative_covers_its_source(
    scenario_dataset_with_media: Dataset,
) -> None:
    """Both halves, matching the reuse gate transcode itself applies: the link
    records the registration and the file is the output."""
    ds = scenario_dataset_with_media
    _ = add_transcode_derivative(ds, "seq_a", target="playback")
    index_path = ds.get_root("media_raw") / "index.csv"
    # Only the row that actually got a derivative: the fixture holds two videos
    # per sequence, and marking both required would leave the other genuinely
    # short -- a true answer, but not the one under test.
    linked = _mark_required_where_linked(index_path, "playback_derivative_path")

    record = _record(ds, target="playback")

    assert record is not None
    assert linked in record.coverage.covered, (
        "a registered, present derivative should cover its source row"
    )
    assert linked not in record.extra["needs_transcode"]


def test_a_linked_derivative_whose_file_is_gone_needs_it_again(
    scenario_dataset_with_media: Dataset,
) -> None:
    """The link alone is not the artifact. An unlinked or absent file is the
    recoverable interrupted state, and it reads as work still to do."""
    ds = scenario_dataset_with_media
    written = add_transcode_derivative(ds, "seq_a", target="playback")
    index_path = ds.get_root("media_raw") / "index.csv"
    linked = _mark_required_where_linked(index_path, "playback_derivative_path")
    Path(written).unlink()

    record = _record(ds, target="playback")

    assert record is not None
    assert linked in record.extra["needs_transcode"]


def test_the_two_targets_are_reported_independently(
    scenario_dataset: Dataset,
) -> None:
    """A playback transcode never satisfies an analysis read; reporting one
    would hide the other."""
    write_media_index(scenario_dataset, ["seq_a"])

    found = inventory(scenario_dataset, kinds=["media-derivative"])

    assert {r.ref for r in found.records} == {
        MediaDerivativeRef(target="analysis"),
        MediaDerivativeRef(target="playback"),
    }


def _mark_required(index_path: Path, column: str = "analysis_transcode") -> None:
    import pandas as pd

    frame = pd.read_csv(index_path, keep_default_na=False, dtype=str)
    frame[column] = "required"
    frame.to_csv(index_path, index=False)


def _mark_required_where_linked(index_path: Path, link_column: str) -> str:
    """Mark only the row carrying a link as needing one, and name its key."""
    import pandas as pd

    frame = pd.read_csv(index_path, keep_default_na=False, dtype=str)
    linked = frame[frame[link_column].astype(str).str.len() > 0]
    assert not linked.empty, "the fixture registered no derivative"
    uuid = str(linked.iloc[0]["video_uuid"])
    verdict = (
        "stream_transcode"
        if link_column.startswith("playback")
        else ("analysis_transcode")
    )
    frame.loc[frame["video_uuid"] == uuid, verdict] = "required"
    frame.to_csv(index_path, index=False)
    return uuid


def _blank_facts(index_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(index_path, keep_default_na=False, dtype=str)
    frame["media_facts"] = ""
    frame.to_csv(index_path, index=False)


def test_a_dataset_with_no_media_root_reads_empty_rather_than_raising(
    tmp_path: Path,
) -> None:
    """Found on a real tracks-only dataset.

    ``resolve_media_root`` falls back to ``"media"`` when ``media_raw`` is unset
    and returns that name whether or not ``media`` is set either. A dataset that
    declares both roots and fills neither -- which every tracks-only dataset
    does -- therefore names a root ``get_root`` refuses, and the whole inventory
    died on a KeyError rather than reporting the rest of the dataset.
    """
    from mosaic.core.dataset import new_dataset_manifest

    manifest = new_dataset_manifest(name="tracks-only", base_dir=tmp_path / "ds")
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    ds.roots["media_raw"] = ""
    ds.roots["media"] = ""

    record = _record(ds)

    assert record is not None
    assert record.status == "absent"
    assert record.coverage.target == frozenset()
    assert record.extra["needs_transcode"] == frozenset()


# --- media variants -----------------------------------------------------------

_VARIANT = "preprocess.0.1-aaaaaaaaaa"
_CHAINED = "preprocess.0.1-bbbbbbbbbb"


def _variant(ds: Dataset, run_id: str = _VARIANT, scope: Scope | None = None):
    found = inventory(ds, kinds=["media-variant"], scope=scope)
    return found.record(MediaVariantRef(run_id=run_id))


def _set_media_composition(ds: Dataset, *sequences: str) -> None:
    """Record a current media composition for each of *sequences*."""
    _ = write_sequence_compositions(
        ds,
        "media_raw",
        compositions={
            ("", sequence): media_composition(
                [MediaMember(camera="", video_order=0, uid=f"uid-{sequence}")]
            )
            for sequence in sequences
        },
    )


def test_a_finished_variant_holding_every_entry_it_names_reads_complete(
    tmp_path: Path,
) -> None:
    """A variant is keyed by entry and camera, as a frame run is."""
    ds = make_dataset(tmp_path / "ds")
    _ = add_media_variant(ds, _VARIANT, "s1")
    _ = add_media_variant(ds, _VARIANT, "s2", camera="cam0")
    finish_media_variant(ds, _VARIANT)

    record = _variant(ds)

    assert record is not None
    assert record.status == "complete"
    assert record.coverage.present == frozenset({("", "s1", ""), ("", "s2", "cam0")})
    assert record.name == "preprocess"
    assert record.run_root == media_variant_run_root(ds, _VARIANT)
    assert record.finished_at != ""
    assert record.drift == ()


def test_a_variant_still_writing_reads_partial_until_it_finishes(
    tmp_path: Path,
) -> None:
    """A file ahead of its row is a run in progress, and damage once it finished.

    The op renames a file into place and then writes its row. Between the two, the
    file exists and a consumer, which reads the row, cannot use it yet.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, ["s1", "s2"])
    _ = add_media_variant(ds, _VARIANT, "s1")
    ahead = media_variant_path(ds, _VARIANT, "", "s2", "")
    _ = ahead.write_bytes(b"variant")

    writing = _variant(ds)
    finish_media_variant(ds, _VARIANT)
    finished = _variant(ds)

    assert writing is not None
    assert writing.status == "partial"
    assert writing.coverage.missing == frozenset({("", "s2", "")})
    assert writing.orphan_files == frozenset({("", "s2", "")})
    assert finished is not None
    assert finished.status == "inconsistent"


def test_a_row_whose_file_is_gone_reads_inconsistent(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = add_media_variant(ds, _VARIANT, "s1")
    _ = add_media_variant(ds, _VARIANT, "s2")
    media_variant_path(ds, _VARIANT, "", "s2", "").unlink()
    finish_media_variant(ds, _VARIANT)

    record = _variant(ds)

    assert record is not None
    assert record.status == "inconsistent"
    assert record.orphan_rows == frozenset({("", "s2", "")})


def test_a_variant_of_media_that_moved_reads_complete_but_drifted(
    tmp_path: Path,
) -> None:
    """Drift is a known recorded composition that differs from a known current one.

    A blank recorded cell is unknown, and so is an entry whose current
    composition is not recorded. Unknown is not drift.
    """
    ds = make_dataset(tmp_path / "ds")
    _set_media_composition(ds, "moved", "current", "unrecorded")
    current = media_compositions_for(ds, [("", "current")])[("", "current")]
    assert current != ""
    _ = add_media_variant(ds, _VARIANT, "moved", composition="an-earlier-composition")
    _ = add_media_variant(ds, _VARIANT, "current", composition=current)
    _ = add_media_variant(ds, _VARIANT, "unrecorded", composition="")
    _ = add_media_variant(ds, _VARIANT, "unprojected", composition="a-composition")
    finish_media_variant(ds, _VARIANT)

    record = _variant(ds)

    assert record is not None
    assert record.status == "complete-but-drifted"
    assert record.drift == (("", "moved"),)


def test_the_work_directory_and_a_partial_file_are_not_variant_files(
    tmp_path: Path,
) -> None:
    """An entry's claim and its encode in progress are in the run directory too.

    Both are under the entry's work directory and away from the entry's variant
    path. A finished run that contains them reads complete rather than damaged.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, ["s1", "s2", "s3"])
    _ = add_media_variant(ds, _VARIANT, "s1")
    finish_media_variant(ds, _VARIANT)
    claim = media_variant_work_root(ds, _VARIANT) / "s2"
    claim.mkdir(parents=True)
    _ = (claim / "s2.mp4").write_bytes(b"claimed")
    encoding = media_variant_work_root(ds, _VARIANT) / "s3"
    encoding.mkdir(parents=True)
    _ = (encoding / "s3.partial.mp4").write_bytes(b"encoding")

    record = _variant(ds)

    assert record is not None
    assert record.status == "complete"
    assert record.coverage.target == frozenset({("", "s1", "")})
    assert record.orphan_files == frozenset()


def test_every_variant_run_in_the_index_is_one_record(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = add_media_variant(ds, _VARIANT, "s1")
    _ = add_media_variant(ds, _CHAINED, "s1", upstream=_VARIANT)

    found = inventory(ds, kinds=["media-variant"])

    assert {record.ref for record in found.records} == {
        MediaVariantRef(run_id=_VARIANT),
        MediaVariantRef(run_id=_CHAINED),
    }
    chained = found.record(MediaVariantRef(run_id=_CHAINED))
    assert chained is not None
    assert chained.upstreams == (_VARIANT,)


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record each read of the media index and of the compositions that a scan makes."""
    calls: list[str] = []

    def media_index(path: Path) -> list[dict[str, str]]:
        calls.append("media index")
        return read_media_index(path)

    def compositions(
        dataset: Dataset, entries: Iterable[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        calls.append("compositions")
        return media_compositions_for(dataset, entries)

    monkeypatch.setattr(inventory_media, "read_media_index", media_index)
    monkeypatch.setattr(inventory_media, "media_compositions_for", compositions)
    return calls


def test_a_scan_reads_the_media_index_and_the_compositions_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every variant is looked up against the same two shared reads."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, ["s1"])
    _ = add_media_variant(ds, _VARIANT, "s1")
    _ = add_media_variant(ds, _CHAINED, "s1", upstream=_VARIANT)
    calls = _record_reads(monkeypatch)

    found = inventory(ds, kinds=["media-variant"])

    assert len(found.records) == 2
    assert sorted(calls) == ["compositions", "media index"]


def test_a_scan_with_no_variant_reads_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, ["s1"])
    calls = _record_reads(monkeypatch)

    found = inventory(ds, kinds=["media-variant"])

    assert len(found.records) == 0
    assert calls == []


def test_a_selector_narrows_a_variant_to_the_entries_it_names(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, ["s1", "s2", "s3"])
    _ = add_media_variant(ds, _VARIANT, "s1")
    _ = add_media_variant(ds, _VARIANT, "s2")
    ahead = media_variant_path(ds, _VARIANT, "", "s3", "")
    _ = ahead.write_bytes(b"variant")
    finish_media_variant(ds, _VARIANT)

    record = _variant(ds, scope=Scope(entries=[("", "s1")]))

    assert record is not None
    assert record.coverage.target == frozenset({("", "s1", "")})
    assert record.status == "complete"


def test_a_dataset_with_no_media_root_reports_no_variant(tmp_path: Path) -> None:
    """Variants are stored under the media root. A dataset without one reports none."""
    ds = make_dataset(tmp_path / "ds")
    ds.roots["media"] = ""

    found = inventory(ds, kinds=["media-variant"])

    assert found.records == ()
    assert found.errors == ()
