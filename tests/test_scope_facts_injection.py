import dataclasses
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
