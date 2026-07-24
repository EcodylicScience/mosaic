import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.media_index import MediaIndexScope
from mosaic_media import probe_media


def _write_mp4(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(frame_count):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def _make_dataset(tmp_path: Path) -> Dataset:
    for sub in ("media_raw", "media"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
        },
    )


def test_an_injected_scope_writes_facts_without_probing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _make_dataset(tmp_path)
    seq_dir = tmp_path / "media_raw" / "seqA"
    seq_dir.mkdir()
    clip = seq_dir / "clip.mp4"
    _write_mp4(clip)
    facts = probe_media(clip)

    import mosaic.core.dataset as dataset_module

    calls: list[Path] = []
    real_probe = dataset_module.probe_media

    def counting_probe(path: Path):
        calls.append(path)
        return real_probe(path)

    monkeypatch.setattr(dataset_module, "probe_media", counting_probe)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _make_dataset(tmp_path)
    seq_dir = tmp_path / "media_raw" / "seqA"
    seq_dir.mkdir()
    clip = seq_dir / "clip.mp4"
    _write_mp4(clip)

    import mosaic.core.dataset as dataset_module

    calls: list[Path] = []
    real_probe = dataset_module.probe_media

    def counting_probe(path: Path):
        calls.append(path)
        return real_probe(path)

    monkeypatch.setattr(dataset_module, "probe_media", counting_probe)

    scope = MediaIndexScope(directory=seq_dir, group="g", sequence="seqA")
    _ = dataset.write_media_index([scope])
    assert any(c.name == clip.name for c in calls), "an unmapped file was not probed"


def test_a_changed_uuid_is_reported_and_the_write_still_completes(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    seq_dir = tmp_path / "media_raw" / "seqA"
    seq_dir.mkdir()
    clip = seq_dir / "clip.mp4"
    _write_mp4(clip)
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
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    seq_dir = tmp_path / "media_raw" / "seqA"
    seq_dir.mkdir()
    clip = seq_dir / "clip.mp4"
    _write_mp4(clip)
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
