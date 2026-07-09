"""Tests for the tracking-op registry and the ops under the Job Contract.

Uses a real ``Dataset`` + a synthetic ``media/index.csv`` and monkeypatches the
heavy low-level backends (video decode, ultralytics, torch) so the contract
machinery -- runs-row lifecycle, content ``run_id``, progress, cancel, lineage,
and the inference->tracks bridge -- is exercised without any real models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.progress import read_progress
from mosaic.core.pipeline.registry import read_run, read_runs
from mosaic.tracking import (
    TRACKING_OPS,
    describe_tracking_op,
    list_tracking_ops,
    resolve_model,
    run_tracking_op,
)
from mosaic.tracking.frame_extraction.dataset_runs import ExtractFramesParams


# --- fixtures --------------------------------------------------------------


def _make_dataset(tmp_path: Path, seqs=("vid1", "vid2")) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()
    media_root = ds.get_root(ds.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in seqs:
        vp = media_root / f"{s}.mp4"
        vp.write_bytes(b"fake")
        rows.append(
            {
                "name": s,
                "group": "",
                "sequence": s,
                "group_safe": "",
                "sequence_safe": s,
                "abs_path": str(vp),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": 0,
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)
    return ds


def _db(ds: Dataset) -> Path:
    return ds.get_root("features") / ".mosaic.db"


# --- registry & discovery --------------------------------------------------


def test_registry_has_builtin_ops():
    kinds = set(TRACKING_OPS)
    assert kinds >= {
        "extract-frames",
        "train-pose",
        "train-points",
        "train-localizer",
        "infer-pose",
        "infer-points",
        "infer-localizer",
    }
    for op in list_tracking_ops():
        assert op["kind"] in kinds and op["category"] in {"extract", "train", "infer"}


def test_describe_returns_params_schema():
    d = describe_tracking_op("train-pose")
    schema = d["params_schema"]
    assert "properties" in schema
    assert {"data", "epochs", "model"} <= set(schema["properties"])


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        run_tracking_op(object(), "nope", {})


# --- run_id determinism ----------------------------------------------------


def test_hash_exclude_does_not_change_run_id():
    from mosaic.core.pipeline._utils import hash_params

    a = ExtractFramesParams(
        n_frames=10, method="uniform", parallel_workers=1, overwrite=True, groups=["g"]
    )
    b = ExtractFramesParams(
        n_frames=10, method="uniform", parallel_workers=8, overwrite=False, groups=None
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
        return _Res(n_frames)

    monkeypatch.setattr(dr, "_extract_frames", fake)
    return dr


def test_extract_frames_lifecycle(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path)
    _install_fake_extract(monkeypatch)

    from mosaic.tracking import extract_frames

    run_id = extract_frames(ds, n_frames=3, method="uniform")
    assert run_id.startswith("uniform-")

    runs = read_runs(_db(ds), kind="extract-frames")
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
    prog = read_progress(_db(ds), runs[0]["execution_id"])
    assert len([p for p in prog if p["step_type"] == "entry"]) == 2

    # cache hit: same params -> same run_id, new attempt
    run_id2 = extract_frames(ds, n_frames=3, method="uniform")
    assert run_id2 == run_id
    assert len(read_runs(_db(ds), kind="extract-frames")) == 2


def test_extract_frames_cancel(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path, seqs=("a", "b", "c"))
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

    runs = read_runs(_db(ds), kind="extract-frames")
    assert len(runs) == 1 and runs[0]["status"] == "cancelled"


# --- train-pose op (mocked trainer) + lineage ------------------------------


def _install_fake_pose_trainer(monkeypatch):
    import mosaic.tracking.pose_training.train as tr

    def fake_train(
        data_yaml, *, project, name, callback=None, cancel_check=None, epochs=1, **kw
    ):
        run_dir = Path(project) / name
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"weights")
        (run_dir / "results.csv").write_text("epoch,loss\n0,0.1\n")
        for e in range(2):
            if callback is not None:
                callback.on_epoch_end(e, epochs, {"loss": 0.1})
            if cancel_check is not None and cancel_check():
                break
        return None

    monkeypatch.setattr(tr, "train_pose_model", fake_train)


def test_train_pose_lifecycle_and_lineage(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path)
    _install_fake_pose_trainer(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("kpt_shape: [4, 3]\n")

    r1 = run_tracking_op(
        ds, "train-pose", {"data": str(data_yaml), "epochs": 2, "device": "cpu"}
    )
    assert r1.startswith("train-pose-")
    row = read_run(_db(ds), read_runs(_db(ds), kind="train-pose")[0]["execution_id"])
    assert row["status"] == "finished" and row["run_id"] == r1

    # model index row written with the best.pt path
    from mosaic.tracking.ops.train import trained_model_index
    from mosaic.core.pipeline.models import model_index_path

    midx = trained_model_index(model_index_path(ds, "train-pose"))
    mdf = midx.read(run_id=r1)
    assert len(mdf) == 1
    assert mdf.iloc[0]["best_model_path"].endswith("best.pt")
    assert mdf.iloc[0]["base_run_id"] == ""

    # resolve_model turns the run_id into its best.pt (train->track handoff)
    best, base = resolve_model(ds, r1, "train-pose")
    assert best.name == "best.pt" and base == r1

    # retrain from r1 -> lineage recorded
    r2 = run_tracking_op(
        ds, "train-pose", {"data": str(data_yaml), "epochs": 2, "base_model": r1}
    )
    mdf2 = trained_model_index(model_index_path(ds, "train-pose")).read(run_id=r2)
    assert mdf2.iloc[0]["base_run_id"] == r1


def test_train_pose_cancel(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path)
    import mosaic.tracking.pose_training.train as tr

    def fake_train(data_yaml, *, project, name, cancel_check=None, **kw):
        # simulate a between-epoch cancel firing during training
        return None

    monkeypatch.setattr(tr, "train_pose_model", fake_train)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("kpt_shape: [4, 3]\n")

    token = CancelToken()
    token.cancel()  # already cancelled -> ctx.check_cancel() after train raises
    with pytest.raises(Cancelled):
        run_tracking_op(
            ds, "train-pose", {"data": str(data_yaml), "epochs": 1}, cancel_token=token
        )
    assert read_runs(_db(ds), kind="train-pose")[0]["status"] == "cancelled"


# --- infer-pose op -> tracks bridge (mocked model) -------------------------


def test_infer_pose_bridges_to_tracks(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path)
    import mosaic.tracking.pose_training.inference as inf

    def fake_run_inference(model, video, output_dir=None, **kw):
        return ["r"]  # opaque; consumed by fake_to_df

    def fake_to_df(results):
        n = 4
        return pd.DataFrame(
            {
                "frame": range(n),
                "id": [0] * n,
                "poseX0": [1.0] * n,
                "poseY0": [2.0] * n,
                "poseP0": [0.9] * n,
            }
        )

    monkeypatch.setattr(inf, "run_inference", fake_run_inference)
    monkeypatch.setattr(inf, "inference_to_dataframe", fake_to_df)

    # a raw model path (no training run needed)
    model = tmp_path / "m.pt"
    model.write_bytes(b"w")
    run_id = run_tracking_op(
        ds, "infer-pose", {"model": str(model), "convert_to_tracks": True}
    )
    assert run_id.startswith("infer-pose-")

    runs = read_runs(_db(ds), kind="infer-pose")
    assert len(runs) == 1 and runs[0]["status"] == "finished"

    # standardized tracks written for each sequence
    for s in ("vid1", "vid2"):
        tp = ds.get_root("tracks") / f"{s}.parquet"
        assert tp.exists()
        tdf = pd.read_parquet(tp)
        assert {"frame", "time", "id", "group", "sequence", "poseX0", "poseY0"} <= set(
            tdf.columns
        )
    tracks_idx = pd.read_csv(ds.get_root("tracks") / "index.csv")
    assert set(tracks_idx["sequence"]) == {"vid1", "vid2"}

    # inference index row per sequence
    from mosaic.tracking.ops.infer import inference_index, prediction_index_path

    iidx = inference_index(prediction_index_path(ds, "infer-pose"))
    assert set(iidx.read(run_id=run_id)["sequence"]) == {"vid1", "vid2"}
