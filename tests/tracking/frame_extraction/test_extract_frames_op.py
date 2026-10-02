"""The ``extract-frames`` op under the job contract, with the frame decode faked.

Its run-log lifecycle, its run id and the knobs kept out of it, reuse of a
finished run, and cancel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.run_log import read_run_progress, read_runs, run_log_dir
from mosaic.tracking.frame_extraction.dataset_runs import ExtractFramesParams

from tests.helpers import stub_media_dataset


# --- run_id determinism ----------------------------------------------------


def test_hash_exclude_does_not_change_run_id():
    from mosaic.core.pipeline._utils import hash_params

    a = ExtractFramesParams(
        n_frames=10,
        method="uniform",
        parallel_workers=1,
        parallel_mode="thread",
    )
    b = ExtractFramesParams(
        n_frames=10,
        method="uniform",
        parallel_workers=8,
        parallel_mode="process",
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    # a real param DOES change it
    c = ExtractFramesParams(n_frames=11, method="uniform")
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


# --- extract-frames op (mocked decode) -------------------------------------


def _install_fake_extract(monkeypatch):
    """Fake the low-level frame extractor: write a PNG + run_info.json."""
    import mosaic.tracking.frame_extraction.dataset_runs as dr

    class _Res:
        def __init__(self, n):
            self.n_extracted = n
            self.n_requested = n

    def fake(video_path, n_frames, method, output_dir, run_id, **kw):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_0.png").write_bytes(b"x")
        (out / "run_info.json").write_text(
            json.dumps({"output_dir": str(out), "video_path": str(video_path)})
        )
        listed = kw.get("frame_indices")
        return _Res(len(listed) if listed is not None else n_frames)

    monkeypatch.setattr(dr, "_extract_frames", fake)
    return dr


def test_extract_frames_lifecycle(tmp_path, monkeypatch):
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _install_fake_extract(monkeypatch)

    from mosaic.tracking import extract_frames

    run_id = extract_frames(ds, n_frames=3, method="uniform")
    assert run_id.startswith("uniform-")

    runs = read_runs(run_log_dir(ds.base_dir), kind="extract-frames")
    assert len(runs) == 1 and runs[0]["status"] == "finished"
    assert runs[0]["run_id"] == run_id
    assert int(runs[0]["progress_total"]) == 2

    # FramesIndexRow per sequence
    from mosaic.tracking.frame_extraction.dataset_runs import (
        frames_index,
        frames_index_path,
    )

    idx = frames_index(frames_index_path(ds, "uniform"))
    df = idx.read(run_id=run_id)
    assert set(df["sequence"]) == {"vid1", "vid2"}

    # per-entry progress recorded
    prog = read_run_progress(run_log_dir(ds.base_dir), runs[0]["execution_id"])
    assert len([p for p in prog if p["step_type"] == "entry"]) == 2

    # cache hit: same params -> same run_id, new attempt
    run_id2 = extract_frames(ds, n_frames=3, method="uniform")
    assert run_id2 == run_id
    assert len(read_runs(run_log_dir(ds.base_dir), kind="extract-frames")) == 2


def test_extract_frames_list_lifecycle(tmp_path, monkeypatch):
    """A list run is named by its frames, covers the listed entries, and reuses."""
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _install_fake_extract(monkeypatch)

    from mosaic.tracking import extract_frames
    from mosaic.tracking.frame_extraction.dataset_runs import (
        frames_index,
        frames_index_path,
    )

    run_id = extract_frames(ds, method="list", frames={("", "vid1"): [3, 1, 3]})
    assert run_id.startswith("list-")

    runs = read_runs(run_log_dir(ds.base_dir), kind="extract-frames")
    assert len(runs) == 1 and runs[0]["status"] == "finished"
    assert int(runs[0]["progress_total"]) == 1

    df = frames_index(frames_index_path(ds, "list")).read(run_id=run_id)
    assert list(df["sequence"]) == ["vid1"]
    assert list(df["n_frames_requested"]) == [2]

    # The same frames, listed another way: the same run, a new attempt.
    assert extract_frames(ds, method="list", frames={("", "vid1"): [1, 3]}) == run_id
    assert len(read_runs(run_log_dir(ds.base_dir), kind="extract-frames")) == 2


def test_extract_frames_cancel(tmp_path, monkeypatch):
    ds = stub_media_dataset(tmp_path, ["a", "b", "c"])
    dr = _install_fake_extract(monkeypatch)
    from mosaic.tracking import extract_frames

    token = CancelToken()
    orig = dr._extract_frames

    calls = {"n": 0}

    def fake_cancelling(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            token.cancel()  # request cancel after the first sequence
        return orig(*args, **kw)

    monkeypatch.setattr(dr, "_extract_frames", fake_cancelling)

    with pytest.raises(Cancelled):
        extract_frames(
            ds, n_frames=2, method="uniform", parallel_workers=1, cancel_token=token
        )

    runs = read_runs(run_log_dir(ds.base_dir), kind="extract-frames")
    assert len(runs) == 1 and runs[0]["status"] == "cancelled"
