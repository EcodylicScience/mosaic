import csv
import dataclasses
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

import mosaic.core.media.probe_row as probe_row_module
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.media_index import MediaIndexScope
from mosaic_media import MediaFacts, probe_media

MakeDataset = Callable[[Path], Dataset]
WriteVideo = Callable[..., None]


def _record_probe_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Count probe_media calls made through the row builder, still probing.

    The injection contract is negative -- a mapped file is *not* probed -- so the
    test needs the call list rather than the result. Patched on probe_row, which
    is where the row builder resolves the name.
    """
    calls: list[Path] = []
    real_probe = probe_row_module.probe_media

    def counting_probe(path: Path) -> MediaFacts:
        calls.append(path)
        return real_probe(path)

    monkeypatch.setattr(probe_row_module, "probe_media", counting_probe)
    return calls


def _seeded_clip(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> tuple[Dataset, Path, Path]:
    """A saved dataset, one sequence directory under media_raw, and one clip in it.

    The manifest is written (via the shared factory) rather than merely named, so
    the dataset base directory is the manifest's parent and a stored ``abs_path``
    comes back root-relative.
    """
    base = (tmp_path / "dataset").resolve()
    dataset = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "seqA"
    clip = sequence_dir / "clip.mp4"
    write_cfr_mp4(clip)
    return dataset, sequence_dir, clip


def _indexed_scope(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> tuple[Dataset, MediaIndexScope, Path]:
    """A dataset with one minted media row, plus the scope and clip that made it.

    Writes one video under a media_raw sequence directory and runs
    ``write_media_index`` once with no injected facts, so the row is probed and
    minted with full stored facts. Re-passing the returned scope drives the
    reuse path under test.
    """
    dataset, seq_dir, clip = _seeded_clip(tmp_path, make_media_dataset, write_cfr_mp4)
    scope = MediaIndexScope(directory=seq_dir, group="g", sequence="seqA")
    _ = dataset.write_media_index([scope])
    return dataset, scope, clip


def _row_named(dataset: Dataset, name: str) -> dict[str, str]:
    """The single media-index row whose ``name`` cell equals *name*."""
    rows = dataset.read_media_index()
    return next(row for row in rows if row["name"] == name)


def _rewrite_facts_cell(index_csv: Path, name: str, cell: str) -> None:
    """Replace the ``media_facts`` cell of the row named *name*, in place."""
    with index_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    for row in rows:
        if row["name"] == name:
            row["media_facts"] = cell
    with index_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_an_unchanged_minted_file_is_not_re_probed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    dataset, scope, _clip = _indexed_scope(tmp_path, make_media_dataset, write_cfr_mp4)
    before = _row_named(dataset, "clip.mp4")

    calls = _record_probe_calls(monkeypatch)
    _ = dataset.write_media_index([scope])

    assert calls == [], "an unchanged, already-minted file must not re-probe"
    after = _row_named(dataset, "clip.mp4")
    assert after["video_uuid"] == before["video_uuid"]
    assert after["media_facts"] == before["media_facts"]


def test_a_changed_file_is_re_probed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    dataset, scope, clip = _indexed_scope(tmp_path, make_media_dataset, write_cfr_mp4)
    before = _row_named(dataset, "clip.mp4")

    # Rewrite the file with different content, then bump its mtime explicitly so
    # the test does not depend on filesystem timestamp granularity.
    write_cfr_mp4(clip, frames=12)
    future = clip.stat().st_mtime + 10
    os.utime(clip, (future, future))

    calls = _record_probe_calls(monkeypatch)
    _ = dataset.write_media_index([scope])

    assert calls == [clip], "a changed file must be re-probed"
    after = _row_named(dataset, "clip.mp4")
    assert after["video_uuid"] != before["video_uuid"]


def test_injected_facts_win_over_the_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    dataset, scope, clip = _indexed_scope(tmp_path, make_media_dataset, write_cfr_mp4)
    cached_uuid = _row_named(dataset, "clip.mp4")["video_uuid"]
    injected = dataclasses.replace(probe_media(clip), video_uuid="injected-override")
    scope_with_injection = dataclasses.replace(
        scope, facts_by_name={clip.name: injected}
    )

    calls = _record_probe_calls(monkeypatch)
    _ = dataset.write_media_index([scope_with_injection])

    assert calls == [], "an injected file is neither probed nor served from cache"
    written = _row_named(dataset, "clip.mp4")
    assert written["video_uuid"] == "injected-override"
    assert written["video_uuid"] != cached_uuid


def test_a_row_whose_stored_facts_predate_the_identity_fields_is_re_probed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    dataset, scope, clip = _indexed_scope(tmp_path, make_media_dataset, write_cfr_mp4)
    # A media_facts cell written before the identity fields existed: it lacks the
    # keys MediaFacts now requires, so row_facts_or_none cannot reconstruct it and
    # there is nothing to reuse.
    index_csv = dataset.get_root("media_raw") / "index.csv"
    _rewrite_facts_cell(index_csv, "clip.mp4", json.dumps({"width": 64, "height": 48}))

    calls = _record_probe_calls(monkeypatch)
    _ = dataset.write_media_index([scope])

    assert calls == [clip], "a row with unreconstructable facts must be re-probed"


def test_the_cache_does_not_cross_two_sequences_sharing_a_basename(
    tmp_path: Path,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    base = (tmp_path / "dataset").resolve()
    dataset = make_media_dataset(base)
    clip_a = base / "media_raw" / "seqA" / "video.mp4"
    clip_b = base / "media_raw" / "seqB" / "video.mp4"
    write_cfr_mp4(clip_a, frames=6)
    write_cfr_mp4(clip_b, frames=12)
    scope_a = MediaIndexScope(directory=clip_a.parent, group="g", sequence="seqA")
    scope_b = MediaIndexScope(directory=clip_b.parent, group="g", sequence="seqB")
    _ = dataset.write_media_index([scope_a, scope_b])

    def uuid_for(sequence: str) -> str:
        rows = dataset.read_media_index()
        return next(row for row in rows if row["sequence"] == sequence)["video_uuid"]

    uuid_a, uuid_b = uuid_for("seqA"), uuid_for("seqB")
    assert uuid_a != uuid_b, "distinct content must mint distinct identities"

    # Second write reuses the cache. A basename-keyed cache would serve one
    # sequence's "video.mp4" measurement for the other's file; a resolved-path
    # cache keeps them apart.
    _ = dataset.write_media_index([scope_a, scope_b])
    assert uuid_for("seqA") == uuid_a
    assert uuid_for("seqB") == uuid_b


def test_index_media_still_probes_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    base = (tmp_path / "dataset").resolve()
    dataset = make_media_dataset(base)
    media_raw = base / "media_raw"
    write_cfr_mp4(media_raw / "one.mp4")
    write_cfr_mp4(media_raw / "two.mp4")

    calls = _record_probe_calls(monkeypatch)
    _ = dataset.index_media([media_raw])

    assert {c.name for c in calls} == {"one.mp4", "two.mp4"}
    assert len(calls) == 2, "index_media takes no cache and probes every file"


def test_an_injected_scope_writes_facts_without_probing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    dataset, seq_dir, clip = _seeded_clip(tmp_path, make_media_dataset, write_cfr_mp4)
    facts = probe_media(clip)

    calls = _record_probe_calls(monkeypatch)

    scope = MediaIndexScope(
        directory=seq_dir,
        group="g",
        sequence="seqA",
        facts_by_name={clip.name: facts},
    )
    _ = dataset.write_media_index([scope])
    assert calls == [], "probe_media was called for an injected file"

    rows = dataset.read_media_index()
    written = next(r for r in rows if r["name"] == clip.name)
    assert written["video_uuid"] == facts.video_uuid
    assert written["content_digest"] == facts.content_digest


def test_a_file_absent_from_the_map_is_probed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_media_dataset: MakeDataset,
    write_cfr_mp4: WriteVideo,
) -> None:
    dataset, seq_dir, clip = _seeded_clip(tmp_path, make_media_dataset, write_cfr_mp4)

    calls = _record_probe_calls(monkeypatch)

    scope = MediaIndexScope(directory=seq_dir, group="g", sequence="seqA")
    _ = dataset.write_media_index([scope])
    assert any(c.name == clip.name for c in calls), "an unmapped file was not probed"


def test_a_changed_uuid_is_reported_and_the_write_still_completes(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    dataset, seq_dir, clip = _seeded_clip(tmp_path, make_media_dataset, write_cfr_mp4)
    facts = probe_media(clip)

    # First write: no prior index, so no disagreement, and it establishes the
    # prior row carrying the file's real uuid.
    first = MediaIndexScope(
        directory=seq_dir,
        group="g",
        sequence="seqA",
        facts_by_name={clip.name: facts},
    )
    result = dataset.write_media_index([first])
    assert result.disagreements == []

    # Second write: inject a DIFFERENT uuid for the same path -- as if the file
    # were replaced on disk between probe and finalize.
    other = dataclasses.replace(facts, video_uuid="a-different-uuid")
    second = MediaIndexScope(
        directory=seq_dir,
        group="g",
        sequence="seqA",
        facts_by_name={clip.name: other},
    )
    result = dataset.write_media_index([second])
    assert len(result.disagreements) == 1
    record = result.disagreements[0]
    assert record.basename == clip.name
    assert record.injected_uuid == "a-different-uuid"
    assert record.prior_uuid == facts.video_uuid
    # The write still landed the injected value.
    rows = dataset.read_media_index()
    written = next(r for r in rows if r["name"] == clip.name)
    assert written["video_uuid"] == "a-different-uuid"


def test_an_agreeing_or_unminted_prior_reports_no_disagreement(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    dataset, seq_dir, clip = _seeded_clip(tmp_path, make_media_dataset, write_cfr_mp4)
    facts = probe_media(clip)
    scope = MediaIndexScope(
        directory=seq_dir,
        group="g",
        sequence="seqA",
        facts_by_name={clip.name: facts},
    )
    # No prior row on the first write.
    assert dataset.write_media_index([scope]).disagreements == []
    # Same uuid again -> agreement, no disagreement.
    assert dataset.write_media_index([scope]).disagreements == []
