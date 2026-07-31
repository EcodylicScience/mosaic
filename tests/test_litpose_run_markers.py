"""End-to-end ``run_litpose`` reuse and provenance, without a real Lightning Pose.

``run_litpose_predict`` is replaced with a recording fake that writes a small,
converter-readable DeepLabCut-style CSV (the shape Lightning Pose exports). This
exercises the Job-Contract machinery -- content ``run_id``, phase-marker reuse,
the CSV -> tracks bridge via the reused ``deeplabcut`` converter, and the two
index writers -- with no models and no GPU.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive

import mosaic.tracking.litpose.dataset_runs as dr
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline.markers import read_phase_marker
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.tracking.litpose.dataset_runs import litpose_index_path, litpose_run_root
from mosaic.tracking.litpose.run import LitposePredictResult

_BODYPARTS = ("nose", "tail")


# --- fixtures --------------------------------------------------------------


def _clean_facts_cells(video_uuid: str = "") -> dict[str, object]:
    facts: MediaFacts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid=video_uuid,
        identity_scheme="video/1" if video_uuid else "",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def _write_media_index(
    ds: Dataset,
    sequences: list[str],
    *,
    filenames: dict[str, str] | None = None,
    uids: dict[str, str] | None = None,
) -> None:
    media_root = ds.get_root(ds.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seq in sequences:
        filename = (filenames or {}).get(seq, f"{seq}.mp4")
        video = media_root / filename
        if not video.exists():
            video.write_bytes(b"fake")
        rows.append(
            {
                "name": filename,
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
                **_clean_facts_cells((uids or {}).get(seq, "")),
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)


def _write_dlc_csv(path: Path, *, n: int = 6) -> None:
    """Write a single-animal DeepLabCut / Lightning Pose CSV (scorer/bodyparts/coords)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    header_scorer = ["scorer"]
    header_bp = ["bodyparts"]
    header_coord = ["coords"]
    for bp in _BODYPARTS:
        header_scorer += ["heatmap_tracker"] * 3
        header_bp += [bp] * 3
        header_coord += ["x", "y", "likelihood"]
    lines = [",".join(header_scorer), ",".join(header_bp), ",".join(header_coord)]
    for i in range(n):
        row = [str(i)]
        for _bp in _BODYPARTS:
            x, y = rng.uniform(0, 100, 2)
            lk = rng.uniform(0.5, 1.0)
            row += [f"{x:.6f}", f"{y:.6f}", f"{lk:.6f}"]
        lines.append(",".join(row))
    path.write_text("\n".join(lines))


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    _write_media_index(dataset, ["vid1"])
    return dataset


def _make_model(model_dir: Path, *, weights: bytes = b"weights") -> Path:
    """A minimal Lightning Pose model directory: config.yaml + one checkpoint."""
    ckpt = model_dir / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(weights)
    (model_dir / "config.yaml").write_text(
        "model:\n  model_type: heatmap\ndata:\n  keypoint_names: [nose, tail]\n"
    )
    return model_dir


@pytest.fixture
def model(tmp_path: Path) -> Path:
    return _make_model(tmp_path / "litpose_model")


@dataclass
class FakeLitpose:
    """Recording stand-in for the single Lightning Pose inference phase."""

    predicted: list[Path] = field(default_factory=list)
    frames: int = 6

    def predict(
        self, video_path: Path, out_csv: Path, **_kwargs: object
    ) -> LitposePredictResult:
        self.predicted.append(Path(video_path))
        out = Path(out_csv)
        _write_dlc_csv(out, n=self.frames)
        return LitposePredictResult(csv_path=out, stdout="", stderr="")


@pytest.fixture
def litpose(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeLitpose]:
    fake = FakeLitpose()
    monkeypatch.setattr(dr, "run_litpose_predict", fake.predict)
    yield fake


# --- a fresh run produces tracks + both index rows -------------------------


def test_a_fresh_run_predicts_and_bridges(
    ds: Dataset, model: Path, litpose: FakeLitpose
) -> None:
    run_id = dr.run_litpose(ds, model_path=str(model))

    assert run_id.startswith("litpose.2.3-")
    assert len(litpose.predicted) == 1  # the one gated phase ran once
    # the run wrote the predictions CSV into the run root
    seq_dir = litpose_run_root(ds, run_id) / "vid1"
    assert (seq_dir / "vid1.predictions.csv").exists()

    # the tracks index carries the Lightning Pose producer path
    tracks = read_tracks_index(ds)
    assert len(tracks) == 1
    row = tracks.iloc[0]
    assert str(row["producer"]) == "litpose"
    assert str(row["run_id"]).startswith("litpose.2.3-")
    assert str(row["producer_run_id"]) == run_id
    assert int(row["n_rows"]) == 6

    # raw output lives under _tracking/, not tracks_raw/
    assert "_tracking/litpose" in str(ds.relative_to_root(seq_dir))

    # the litpose run index records the run
    lidx = pd.read_csv(litpose_index_path(ds))
    assert set(lidx["sequence"]) == {"vid1"}
    assert str(lidx.iloc[0]["model_id"]) != ""


# --- a second identical run reuses the inference ---------------------------


def test_a_completed_run_reuses_the_inference(
    ds: Dataset, model: Path, litpose: FakeLitpose
) -> None:
    first = dr.run_litpose(ds, model_path=str(model))
    assert len(litpose.predicted) == 1

    second = dr.run_litpose(ds, model_path=str(model))
    assert second == first
    # inference is not re-run: the phase marker proves it is done.
    assert len(litpose.predicted) == 1

    # The reuse run's index row keeps the true count -- re-derived from the
    # existing parquet rather than replaced with a zero.
    lidx = pd.read_csv(litpose_index_path(ds))
    assert len(lidx) == 1
    assert int(lidx.iloc[0]["n_ids"]) == 1


def test_an_interrupted_predict_is_not_trusted(
    ds: Dataset, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed predict must leave no completion marker and no tracks row."""

    def dying_predict(video_path: Path, out_csv: Path, **_kw: object):
        # A killed run may leave a partial CSV at the canonical path.
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(out_csv).write_text("partial-not-a-real-csv")
        raise RuntimeError("killed mid-predict")

    monkeypatch.setattr(dr, "run_litpose_predict", dying_predict)
    with pytest.raises(RuntimeError):
        dr.run_litpose(ds, model_path=str(model))

    # No completion marker was written, so nothing bridged to tracks.
    litpose_root = ds.get_root("litpose")
    seq_dirs = [p.parent for p in litpose_root.rglob("*.predictions.csv")]
    for seq_dir in seq_dirs:
        assert read_phase_marker(seq_dir, "track") is None
    assert len(read_tracks_index(ds)) == 0

    # A working predict now re-runs (clearing the partial) and bridges to tracks.
    fake = FakeLitpose()
    monkeypatch.setattr(dr, "run_litpose_predict", fake.predict)
    dr.run_litpose(ds, model_path=str(model))
    assert len(fake.predicted) == 1  # the partial was not reused
    assert len(read_tracks_index(ds)) == 1


# --- overwrite forces a recompute ------------------------------------------


def test_overwrite_forces_a_recompute(
    ds: Dataset, model: Path, litpose: FakeLitpose
) -> None:
    dr.run_litpose(ds, model_path=str(model))
    assert len(litpose.predicted) == 1

    dr.run_litpose(ds, model_path=str(model), overwrite=True)
    assert len(litpose.predicted) == 2  # inference ran again


# --- different weights are a different run ---------------------------------


def test_different_weights_are_a_different_run(
    ds: Dataset, tmp_path: Path, litpose: FakeLitpose
) -> None:
    m1 = _make_model(tmp_path / "m1", weights=b"weights-A")
    m2 = _make_model(tmp_path / "m2", weights=b"weights-B")

    a = dr.run_litpose(ds, model_path=str(m1))
    b = dr.run_litpose(ds, model_path=str(m2))
    assert a != b
    assert litpose_run_root(ds, a) != litpose_run_root(ds, b)


def test_a_different_config_is_a_different_run(
    ds: Dataset, tmp_path: Path, litpose: FakeLitpose
) -> None:
    """The config.yaml is part of model identity (it shapes the output)."""
    m1 = _make_model(tmp_path / "c1")
    m2 = _make_model(tmp_path / "c2")
    # Same weights, different config -> different run.
    (m2 / "config.yaml").write_text(
        "model:\n  model_type: heatmap\ndata:\n  keypoint_names: [nose, tail, mid]\n"
    )
    a = dr.run_litpose(ds, model_path=str(m1))
    b = dr.run_litpose(ds, model_path=str(m2))
    assert a != b


# --- the reuse comparison is uid-first, with the path as fallback ------------


def test_a_video_replaced_in_place_forces_a_recompute(
    ds: Dataset, model: Path, litpose: FakeLitpose
) -> None:
    """The case a path comparison cannot see at all.

    Same sequence, same filename, different bytes. Lightning Pose recorded
    ``source_uid`` on its marker and never read it back, so the second run
    reused a prediction over content that no longer exists.
    """
    _write_media_index(ds, ["vid1"], uids={"vid1": "uid-aaa"})
    run_id = dr.run_litpose(ds, model_path=str(model))

    _write_media_index(ds, ["vid1"], uids={"vid1": "uid-bbb"})
    second = dr.run_litpose(ds, model_path=str(model))

    assert second == run_id, "settings did not change, so neither does the identity"
    assert len(litpose.predicted) == 2, "the replaced video was not re-predicted"


def test_the_same_video_under_a_new_name_is_not_a_recompute(
    ds: Dataset, model: Path, litpose: FakeLitpose
) -> None:
    """The other direction, and the saving the uid comparison buys."""
    _write_media_index(ds, ["vid1"], uids={"vid1": "uid-aaa"})
    run_id = dr.run_litpose(ds, model_path=str(model))

    _write_media_index(
        ds, ["vid1"], filenames={"vid1": "renamed.mp4"}, uids={"vid1": "uid-aaa"}
    )
    second = dr.run_litpose(ds, model_path=str(model))

    assert second == run_id
    assert len(litpose.predicted) == 1, "the same bytes were predicted twice"


def test_an_absent_uid_still_falls_back_to_the_path(
    ds: Dataset, model: Path, litpose: FakeLitpose
) -> None:
    """Media indexed before the identity columns carries no uid."""
    run_id = dr.run_litpose(ds, model_path=str(model))
    _write_media_index(ds, ["vid1"], filenames={"vid1": "vid2.mp4"})

    second = dr.run_litpose(ds, model_path=str(model))

    assert second == run_id
    assert len(litpose.predicted) == 2, "a changed source with no uid must re-predict"
