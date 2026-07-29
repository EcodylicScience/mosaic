"""End-to-end ``run_sleap`` reuse and provenance, without a real SLEAP binary.

``sleap-track`` / ``sleap-convert`` are replaced with recording fakes (the shape
established in ``test_trex_run_markers.py``): the fake inference writes a ``.slp``
and the fake export writes a small, converter-readable analysis ``.h5``. This
exercises the Job-Contract machinery -- content ``run_id``, phase-marker reuse,
the analysis-h5 -> tracks bridge, and the two index writers -- with no models.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive

import mosaic.tracking.sleap.dataset_runs as dr
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.tracking.sleap.dataset_runs import sleap_index_path, sleap_run_root
from mosaic.tracking.sleap.run import SleapConvertResult, SleapTrackResult

# The bridge reads the analysis HDF5 with h5py (a [recommended] extra); skip the
# whole module when it is absent rather than fail a minimal install.
pytest.importorskip("h5py")


# --- fixtures --------------------------------------------------------------


def _clean_facts_cells() -> dict[str, object]:
    facts: MediaFacts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid="",
        identity_scheme="",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def _write_media_index(ds: Dataset, sequences: list[str]) -> None:
    media_root = ds.get_root(ds.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seq in sequences:
        video = media_root / f"{seq}.mp4"
        if not video.exists():
            video.write_bytes(b"fake")
        rows.append(
            {
                "name": f"{seq}.mp4",
                "group": "",
                "sequence": seq,
                "group_safe": "",
                "sequence_safe": seq,
                "abs_path": ds.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": 0,
                **_clean_facts_cells(),
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    _write_media_index(dataset, ["vid1"])
    return dataset


@pytest.fixture
def model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.ckpt").write_bytes(b"weights")
    return model_dir


def _write_analysis_h5(path: Path, *, n: int = 6) -> None:
    """A matlab-layout analysis HDF5: 1 track, 1 node, *n* frames."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = np.random.default_rng(0).random((n, 1, 1, 2))
    arr = np.transpose(tracks, (1, 3, 2, 0))  # -> (track, xy, node, frame)
    with h5py.File(str(path), "w") as f:
        d = f.create_dataset("tracks", data=arr)
        d.attrs["dims"] = json.dumps(["track", "xy", "node", "frame"])


@dataclass
class FakeSleap:
    """Recording stand-ins for the two SLEAP phases."""

    tracked: list[Path] = field(default_factory=list)
    converted: list[Path] = field(default_factory=list)
    frames: int = 6

    def track(
        self, video_path: Path, output_slp: Path, **_kwargs: object
    ) -> SleapTrackResult:
        self.tracked.append(Path(video_path))
        out = Path(output_slp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"slp")
        return SleapTrackResult(slp_path=out, stdout="", stderr="")

    def convert(
        self, slp_path: Path, output_h5: Path, **_kwargs: object
    ) -> SleapConvertResult:
        self.converted.append(Path(slp_path))
        out = Path(output_h5)
        _write_analysis_h5(out, n=self.frames)
        return SleapConvertResult(analysis_h5_path=out, stdout="", stderr="")


@pytest.fixture
def sleap(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSleap]:
    fake = FakeSleap()
    monkeypatch.setattr(dr, "run_sleap_track", fake.track)
    monkeypatch.setattr(dr, "run_sleap_convert", fake.convert)
    yield fake


def _tracks_rows(ds: Dataset) -> pd.DataFrame:
    return read_tracks_index(ds)


# --- a fresh run produces tracks + both index rows -------------------------


def test_a_fresh_run_infers_converts_and_bridges(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    run_id = dr.run_sleap(ds, model_paths=[str(model)])

    assert run_id.startswith("sleap.1.6-")
    assert sleap.tracked and sleap.converted  # both phases ran once
    # the run wrote a .slp and an .analysis.h5 into the run root
    seq_dir = sleap_run_root(ds, run_id) / "vid1"
    assert (seq_dir / "vid1.predictions.slp").exists()
    assert (seq_dir / "vid1.analysis.h5").exists()

    # the tracks index carries the SLEAP producer path
    tracks = _tracks_rows(ds)
    assert len(tracks) == 1
    row = tracks.iloc[0]
    assert str(row["producer"]) == "sleap"
    assert str(row["run_id"]).startswith("sleap.1.6-")
    assert str(row["producer_run_id"]) == run_id
    assert int(row["n_rows"]) == 6

    # the sleap run index records the run
    sidx = pd.read_csv(sleap_index_path(ds))
    assert set(sidx["sequence"]) == {"vid1"}
    assert str(sidx.iloc[0]["model_id"]) != ""


# --- a second identical run reuses the inference ---------------------------


def test_a_completed_run_reuses_the_inference(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    first = dr.run_sleap(ds, model_paths=[str(model)])
    assert len(sleap.tracked) == 1

    second = dr.run_sleap(ds, model_paths=[str(model)])
    assert second == first
    # inference is not re-run: the phase marker proves it is done, and the .h5
    # already exists so the analysis export is not re-run either.
    assert len(sleap.tracked) == 1
    assert len(sleap.converted) == 1

    # The reuse run's index row keeps the true track count -- it re-derives it
    # from the existing parquet rather than replacing the row with a zero.
    sidx = pd.read_csv(sleap_index_path(ds))
    assert len(sidx) == 1
    assert int(sidx.iloc[0]["n_tracks"]) == 1


def test_an_interrupted_analysis_export_is_not_trusted(
    ds: Dataset, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed sleap-convert must not leave a partial .h5 that reuse trusts."""
    fake = FakeSleap()
    monkeypatch.setattr(dr, "run_sleap_track", fake.track)

    def dying_convert(slp_path: Path, output_h5: Path, **_kw: object):
        # h5py opens "w" and truncates immediately; a kill leaves a partial file.
        Path(output_h5).parent.mkdir(parents=True, exist_ok=True)
        Path(output_h5).write_bytes(b"partial-not-a-real-h5")
        raise RuntimeError("killed mid-convert")

    monkeypatch.setattr(dr, "run_sleap_convert", dying_convert)
    with pytest.raises(RuntimeError):
        dr.run_sleap(ds, model_paths=[str(model)])

    sleap_root = ds.get_root("sleap")
    # Inference finished (its .slp and marker are valid) but no canonical .h5 was
    # published -- only a *.partial temp, which the ensure step ignores.
    assert list(sleap_root.rglob("*.predictions.slp"))
    assert not list(sleap_root.rglob("*.analysis.h5"))
    assert len(fake.tracked) == 1

    # A working export now publishes the canonical .h5 and bridges to tracks,
    # WITHOUT re-running the already-complete inference.
    monkeypatch.setattr(dr, "run_sleap_convert", fake.convert)
    dr.run_sleap(ds, model_paths=[str(model)])
    assert len(fake.tracked) == 1  # inference reused
    assert list(sleap_root.rglob("*.analysis.h5"))
    assert len(read_tracks_index(ds)) == 1


# --- overwrite forces a recompute ------------------------------------------


def test_overwrite_forces_a_recompute(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    dr.run_sleap(ds, model_paths=[str(model)])
    assert len(sleap.tracked) == 1

    dr.run_sleap(ds, model_paths=[str(model)], overwrite=True)
    assert len(sleap.tracked) == 2  # inference ran again


# --- different weights are a different run ---------------------------------


def test_different_weights_are_a_different_run(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    (m1 / "best.ckpt").write_bytes(b"weights-A")
    (m2 / "best.ckpt").write_bytes(b"weights-B")

    a = dr.run_sleap(ds, model_paths=[str(m1)])
    b = dr.run_sleap(ds, model_paths=[str(m2)])
    assert a != b
    assert sleap_run_root(ds, a) != sleap_run_root(ds, b)
