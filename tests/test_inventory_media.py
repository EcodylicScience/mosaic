"""Transcode coverage, the kind with no run directory to look in.

The case a single coverage signature gets wrong, and gets wrong in the worst
direction: asked for a run directory that was never supposed to exist, a
directory-shaped check reports zero of N, so a corpus with nothing to transcode
reads as permanently incomplete and anything acting on it resubmits forever.
"""

from __future__ import annotations

from pathlib import Path

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.inventory import MediaDerivativeRef, inventory

from tests.helpers import add_transcode_derivative, write_media_index


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
