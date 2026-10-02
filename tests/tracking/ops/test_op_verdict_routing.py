"""An analysis-required original reaches the per-frame ops as its derivative.

``extract-frames``, ``infer-pose`` and ``trex`` each read the clean analysis
transcode of an original whose verdict requires one, and refuse an original whose
row links none before any tool opens it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaProbeError, derive

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline.ops import run_op

from tests.helpers import install_fake_trex, scope_over


def _derivative_facts_cells() -> dict:
    """Flat + JSON facts cells describing one clean analysis derivative."""
    facts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid="",
        identity_scheme="",
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def _routing_dataset(
    tmp_path: Path, *, analysis_derivative_path: str
) -> tuple[Dataset, Path, Path | None]:
    """A media_raw dataset with one analysis-required original.

    When *analysis_derivative_path* is non-empty, also write the matching
    media-index derivative row and the derivative file, so routing resolves to
    it; otherwise the required row stays unlinked and routing must fail loud.
    """
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()
    raw_root = ds.get_root("media_raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    original = raw_root / "vid1.mp4"
    original.write_bytes(b"fake")
    raw_row = {
        "name": "vid1",
        "group": "",
        "sequence": "vid1",
        "group_safe": "",
        "sequence_safe": "vid1",
        "abs_path": str(original),
        "size_bytes": 4,
        "mtime_iso": "",
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "codec": "h264",
        "media_type": "video",
        "analysis_transcode": "required",
        "analysis_derivative_path": analysis_derivative_path,
        "video_order": 0,
    }
    pd.DataFrame([raw_row]).to_csv(raw_root / "index.csv", index=False)

    derivative: Path | None = None
    if analysis_derivative_path:
        media_root = ds.get_root("media")
        media_root.mkdir(parents=True, exist_ok=True)
        derivative = media_root / analysis_derivative_path
        derivative.write_bytes(b"fake-derivative")
        media_row = {
            "name": derivative.name,
            "group": "",
            "sequence": "vid1",
            "group_safe": "",
            "sequence_safe": "vid1",
            "abs_path": str(derivative),
            "size_bytes": derivative.stat().st_size,
            "mtime_iso": "",
            "width": 640,
            "height": 480,
            "fps": 30.0,
            "codec": "h264",
            "media_type": "video",
            "video_order": 0,
            **_derivative_facts_cells(),
            "source_path": "vid1.mp4",
        }
        pd.DataFrame([media_row]).to_csv(media_root / "index.csv", index=False)
    return ds, original, derivative


def test_extract_frames_routes_required_row_to_derivative(tmp_path, monkeypatch):
    ds, _original, derivative = _routing_dataset(
        tmp_path, analysis_derivative_path="vid1.analysis.mp4"
    )
    assert derivative is not None
    import mosaic.tracking.frame_extraction.dataset_runs as dr

    seen: list[Path] = []

    class _Res:
        n_extracted = 1
        n_requested = 1

    def fake(video_path, n_frames, method, output_dir, run_id, **kw):
        seen.append(Path(video_path))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_0.png").write_bytes(b"x")
        (out / "run_info.json").write_text(
            json.dumps({"output_dir": str(out), "video_path": str(video_path)})
        )
        return _Res()

    monkeypatch.setattr(dr, "_extract_frames", fake)

    from mosaic.tracking import extract_frames

    extract_frames(ds, n_frames=1, method="uniform", parallel_workers=1)
    # The op read the clean analysis derivative, never the defective original.
    assert [p.resolve() for p in seen] == [derivative.resolve()]


def test_extract_frames_required_unlinked_raises(tmp_path):
    ds, _original, _ = _routing_dataset(tmp_path, analysis_derivative_path="")
    from mosaic.tracking import extract_frames

    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        extract_frames(ds, n_frames=1, method="uniform", parallel_workers=1)


def test_infer_required_unlinked_raises(tmp_path):
    ds, _original, _ = _routing_dataset(tmp_path, analysis_derivative_path="")
    model = tmp_path / "m.pt"
    model.write_bytes(b"w")

    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        run_op(ds, "infer-pose", {"model": str(model)})


def test_run_trex_routes_required_row_to_derivative(tmp_path, monkeypatch):
    ds, _original, derivative = _routing_dataset(
        tmp_path, analysis_derivative_path="vid1.analysis.mp4"
    )
    assert derivative is not None
    import mosaic.tracking.trex.dataset_runs as dr
    from mosaic.tracking.trex.params import TrexParams

    fake = install_fake_trex(monkeypatch)

    dr.run_trex(ds, TrexParams(), scope_over(("", "vid1")))
    # TREx tracked the clean analysis derivative, never the defective original.
    assert [p.resolve() for p in fake.converted] == [derivative.resolve()]


def test_run_trex_required_unlinked_raises(tmp_path, monkeypatch):
    ds, _original, _ = _routing_dataset(tmp_path, analysis_derivative_path="")
    import mosaic.tracking.trex.dataset_runs as dr
    from mosaic.tracking.trex.params import TrexParams

    def _fail(*args, **kw):
        raise AssertionError("TREx must not run for a required-unlinked entry")

    monkeypatch.setattr(dr, "run_trex_convert", _fail)
    monkeypatch.setattr(dr, "run_trex_track", _fail)

    # The required-but-unlinked entry raises during media resolution, before any
    # TREx subprocess opens the defective original.
    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        dr.run_trex(ds, TrexParams(), scope_over(("", "vid1")))
