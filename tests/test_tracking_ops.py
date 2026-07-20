"""Tests for the tracking-op registry and the ops under the Job Contract.

Uses a real ``Dataset`` + a synthetic ``media/index.csv`` and monkeypatches the
heavy low-level backends (video decode, ultralytics, torch) so the contract
machinery -- run-log lifecycle, content ``run_id``, progress, cancel, lineage,
and the inference->tracks bridge -- is exercised without any real models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaProbeError, derive

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.ops import OPS, describe_op, list_ops, run_op
from mosaic.core.pipeline.run_log import (
    read_run,
    read_run_progress,
    read_runs,
    run_log_dir,
)
from mosaic.tracking import resolve_model
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


def _run_dir(ds: Dataset) -> Path:
    return run_log_dir(ds.base_dir)


# --- registry & discovery --------------------------------------------------


def test_registry_has_builtin_ops():
    kinds = set(OPS)
    assert kinds >= {
        "extract-frames",
        "train-pose",
        "train-points",
        "train-localizer",
        "infer-pose",
        "infer-points",
        "infer-localizer",
    }
    assert "trex" in kinds
    for op in list_ops():
        assert op["kind"] in kinds and op["category"] in {
            "extract",
            "train",
            "infer",
            "convert",
        }


def test_describe_returns_params_schema():
    d = describe_op("train-pose")
    schema = d["params_schema"]
    assert "properties" in schema
    assert {"data", "epochs", "model"} <= set(schema["properties"])


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        run_op(object(), "nope", {})


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

    runs = read_runs(_run_dir(ds), kind="extract-frames")
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
    prog = read_run_progress(_run_dir(ds), runs[0]["execution_id"])
    assert len([p for p in prog if p["step_type"] == "entry"]) == 2

    # cache hit: same params -> same run_id, new attempt
    run_id2 = extract_frames(ds, n_frames=3, method="uniform")
    assert run_id2 == run_id
    assert len(read_runs(_run_dir(ds), kind="extract-frames")) == 2


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

    runs = read_runs(_run_dir(ds), kind="extract-frames")
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

    r1 = run_op(
        ds, "train-pose", {"data": str(data_yaml), "epochs": 2, "device": "cpu"}
    )
    assert r1.startswith("train-pose-")
    row = read_run(
        _run_dir(ds), read_runs(_run_dir(ds), kind="train-pose")[0]["execution_id"]
    )
    assert row["status"] == "finished" and row["run_id"] == r1
    # per-epoch on_epoch_end advances the coarse runs-row counter (2 epochs -> 2/2),
    # so `status --json` progress_done tracks training epochs, not just the stream.
    assert row["progress_done"] == 2 and row["progress_total"] == 2

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
    r2 = run_op(
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
        run_op(
            ds, "train-pose", {"data": str(data_yaml), "epochs": 1}, cancel_token=token
        )
    assert read_runs(_run_dir(ds), kind="train-pose")[0]["status"] == "cancelled"


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
    run_id = run_op(
        ds, "infer-pose", {"model": str(model), "convert_to_tracks": True}
    )
    assert run_id.startswith("infer-pose-")

    runs = read_runs(_run_dir(ds), kind="infer-pose")
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


# --- trex op (registered; run_id parity with the standalone run_trex) -------


def test_trex_registered_as_gpu_convert_op():
    assert "trex" in OPS
    d = describe_op("trex")
    assert d["category"] == "convert"
    assert {"detect_model", "track_max_individuals", "entries"} <= set(
        d["params_schema"]["properties"]
    )
    from mosaic.core.pipeline.ops import op_resource_class

    # declared "gpu" despite category "convert" (TREx needs the GPU for YOLO detect)
    assert op_resource_class("trex") == "gpu"


def test_trex_op_run_id_matches_standalone_run_trex(tmp_path):
    # TrexOp must produce the same content run_id as calling run_trex directly for the same
    # settings, so existing TREx tracks stay cache-valid after the op refactor. Scope to a
    # missing sequence so the run short-circuits (empty media) before any trex binary is used.
    from mosaic.tracking import run_trex

    ds = _make_dataset(tmp_path)
    direct = run_trex(ds, sequences=["nonexistent"])
    via_op = run_op(ds, "trex", {"sequences": ["nonexistent"]})
    assert direct == via_op
    assert direct.startswith("trex-")


def test_trex_params_exclude_throughput_from_run_id():
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.ops.trex import TrexParams

    a = TrexParams(
        detect_model="m.pt", timeout=600, overwrite=False, convert_to_tracks=True
    )
    b = TrexParams(
        detect_model="m.pt", timeout=30, overwrite=True, convert_to_tracks=False
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    c = TrexParams(detect_model="other.pt")
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


# --- convert-points op (real converter, no heavy backend) ------------------


def _write_cvat_points_fixture(root: Path, n_groups: int = 5, per_group: int = 2):
    """Write a tiny CVAT 'for Images 1.1' XML + matching (empty) image files.

    Returns (xml_path, images_dir). Filenames use the ``<stem>__frame_XXXXXX.png``
    convention so ``split_by='group'`` groups by video stem.
    """
    images_dir = root / "cvat" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>"]
    for g in range(n_groups):
        for f in range(per_group):
            name = f"v{g}__frame_{f:06d}.png"
            (images_dir / name).write_bytes(b"")  # existence only; dims come from XML
            lines.append(f'  <image name="{name}" width="640" height="480">')
            lines.append('    <points points="100.0,120.0">')
            lines.append('      <attribute name="class">UnmarkedBee</attribute>')
            lines.append("    </points>")
            lines.append("  </image>")
    lines.append("</annotations>")
    xml_path = root / "cvat" / "annotations.xml"
    xml_path.write_text("\n".join(lines))
    return xml_path, images_dir


def test_convert_points_registered():
    assert "convert-points" in OPS
    d = describe_op("convert-points")
    assert d["category"] == "convert"
    assert {"cvat_xml", "images_dir", "class_names", "radii"} <= set(
        d["params_schema"]["properties"]
    )


def test_point_train_default_model_is_polo26n():
    from mosaic.tracking.ops.train import PointTrainParams

    assert PointTrainParams(data="d.yaml").model == "polo26n.yaml"


def test_convert_points_lifecycle(tmp_path):
    ds = _make_dataset(tmp_path)
    xml, images_dir = _write_cvat_points_fixture(ds.base_dir)

    params = {
        "cvat_xml": ds.relative_to_root(xml),
        "images_dir": ds.relative_to_root(images_dir),
        "class_names": ["UnmarkedBee"],
        "radii": {"UnmarkedBee": 100.0},
        "split_by": "group",
        "symlink_images": False,
    }
    run_id = run_op(ds, "convert-points", dict(params))
    assert run_id.startswith("convert-points-")

    # runs-row lifecycle
    runs = read_runs(_run_dir(ds), kind="convert-points")
    assert len(runs) == 1 and runs[0]["status"] == "finished"
    assert runs[0]["run_id"] == run_id

    # data.yaml + splits written under models/convert-points/<run_id>/
    from mosaic.core.pipeline.models import model_run_root

    out = model_run_root(ds, "convert-points", run_id)
    data_yaml = out / "data.yaml"
    assert data_yaml.exists()
    n_labels = sum(
        len(list((out / split / "labels").glob("*.txt")))
        for split in ("train", "valid", "test")
        if (out / split / "labels").exists()
    )
    assert n_labels == 10  # 5 groups x 2 frames

    # index row recorded + finished
    from mosaic.tracking.ops.convert import (
        converted_dataset_index,
    )
    from mosaic.core.pipeline.models import model_index_path

    idx = converted_dataset_index(model_index_path(ds, "convert-points"))
    df = idx.read(run_id=run_id)
    assert len(df) == 1
    assert df.iloc[0]["class_names"] == "UnmarkedBee"
    assert int(df.iloc[0]["n_train"]) >= 1

    # deterministic + cache hit: identical inputs -> same run_id, no error
    run_id2 = run_op(ds, "convert-points", dict(params))
    assert run_id2 == run_id


def test_convert_points_no_matching_images_raises(tmp_path):
    ds = _make_dataset(tmp_path)
    xml, images_dir = _write_cvat_points_fixture(ds.base_dir)
    empty_dir = ds.base_dir / "cvat" / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="no training labels"):
        run_op(
            ds,
            "convert-points",
            {
                "cvat_xml": ds.relative_to_root(xml),
                "images_dir": ds.relative_to_root(empty_dir),
                "class_names": ["UnmarkedBee"],
                "radii": {"UnmarkedBee": 100.0},
            },
        )


def test_run_trex_resolves_detect_model_run_id_to_weights(tmp_path, monkeypatch):
    """run_trex must resolve a training run_id (detect_model) to its best.pt for TREx.

    Regression: previously the raw run_id string was passed to the trex ``-m`` flag,
    so the train->track handoff (``detect_model=<train run_id>``) gave TREx a
    non-existent model path.
    """
    from pathlib import Path

    from mosaic.core.pipeline.models import model_index_path, model_run_root
    from mosaic.tracking import run_trex
    from mosaic.tracking.ops.train import TrainedModelIndexRow, trained_model_index

    ds = _make_dataset(tmp_path)

    # Seed a trained-model index row + a fake best.pt (as train-points would).
    rid = "train-points-deadbeef01"
    run_root = model_run_root(ds, "train-points", rid)
    weights = run_root / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"pt")
    idx = trained_model_index(model_index_path(ds, "train-points"))
    idx.ensure()
    idx.append(
        [
            TrainedModelIndexRow(
                run_id=rid,
                kind="train-points",
                base_model="",
                base_run_id="",
                best_model_path=ds.relative_to_root(weights),
                metrics_path="",
                n_epochs=1,
                status="finished",
                abs_path=Path(ds.relative_to_root(run_root)),
            )
        ]
    )
    idx.mark_finished(rid)

    # Capture what detect_model run_trex_convert receives, then abort before the binary.
    class _Stop(Exception):
        pass

    captured: dict[str, object] = {}
    import mosaic.tracking.trex.dataset_runs as dr

    def fake_convert(video_path, seq_dir, *, detect_model=None, **kw):
        captured["detect_model"] = detect_model
        raise _Stop()

    monkeypatch.setattr(dr, "run_trex_convert", fake_convert)

    try:
        run_trex(ds, sequences=["vid1"], detect_model=rid, detect_type="yolo")
    except _Stop:
        pass

    assert captured["detect_model"] == weights  # resolved run_id -> absolute best.pt


# --- verdict routing in the per-frame ops (analysis-required originals) -----


def _derivative_facts_cells() -> dict:
    """Flat + JSON facts cells describing one clean analysis derivative."""
    facts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
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
    from mosaic.tracking.trex.run import TRexConvertResult, TRexTrackResult

    seen: list[Path] = []

    def fake_convert(video_path, seq_dir, **kw):
        seen.append(Path(video_path))
        pv_path = Path(seq_dir) / "vid1.pv"
        pv_path.write_bytes(b"")
        return TRexConvertResult(
            pv_path=pv_path,
            settings_path=Path(seq_dir) / "vid1.settings",
            background_path=None,
            stdout="",
            stderr="",
        )

    def fake_track(pv_path, seq_dir, **kw):
        return TRexTrackResult()

    monkeypatch.setattr(dr, "run_trex_convert", fake_convert)
    monkeypatch.setattr(dr, "run_trex_track", fake_track)

    dr.run_trex(ds, entries=[("", "vid1")])
    # TREx tracked the clean analysis derivative, never the defective original.
    assert [p.resolve() for p in seen] == [derivative.resolve()]


def test_run_trex_required_unlinked_raises(tmp_path, monkeypatch):
    ds, _original, _ = _routing_dataset(tmp_path, analysis_derivative_path="")
    import mosaic.tracking.trex.dataset_runs as dr

    def _fail(*args, **kw):
        raise AssertionError("TREx must not run for a required-unlinked entry")

    monkeypatch.setattr(dr, "run_trex_convert", _fail)
    monkeypatch.setattr(dr, "run_trex_track", _fail)

    # The required-but-unlinked entry raises during media resolution, before any
    # TREx subprocess opens the defective original.
    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        dr.run_trex(ds, entries=[("", "vid1")])


# --- generic registry moved into core/pipeline ------------------------------


def test_registry_lives_in_core_pipeline():
    from mosaic.core.pipeline.ops import (
        OPS,
        Op,
        describe_op,
        list_ops,
        op_resource_class,
        register_op,
        run_op,
    )

    assert callable(run_op) and callable(register_op)
    assert isinstance(OPS, dict)
    assert callable(describe_op) and callable(list_ops) and callable(op_resource_class)
    assert isinstance(Op, type)


# The op domains this codebase recognizes. Extend it only when a genuinely new op
# domain is introduced (a deliberate act) -- new ops within an existing domain need
# no edit. Deliberately NOT imported from the source, so a stray new value fails here.
KNOWN_OP_DOMAINS = {"tracking", "media"}


def test_every_op_declares_a_known_domain():
    from mosaic.core.pipeline.ops import OPS

    for kind, op_cls in OPS.items():
        assert op_cls.domain in KNOWN_OP_DOMAINS, (kind, op_cls.domain)


def test_tracking_package_ops_declare_tracking_domain():
    from mosaic.core.pipeline.ops import OPS

    for kind, op_cls in OPS.items():
        if op_cls.__module__.startswith("mosaic.tracking"):
            assert op_cls.domain == "tracking", (kind, op_cls.__module__)


def test_list_ops_filters_by_domain_and_carries_domain():
    from mosaic.core.pipeline.ops import list_ops

    tracking = list_ops(domain="tracking")
    assert tracking, "expected registered tracking ops"
    assert all(entry["domain"] == "tracking" for entry in tracking)
    assert list_ops(domain="nonexistent") == []


def test_describe_op_includes_domain():
    from mosaic.core.pipeline.ops import describe_op

    info = describe_op("infer-pose")
    assert info["domain"] == "tracking"
    assert "params_schema" in info


def test_resolve_model_moved_to_model_refs():
    from mosaic.tracking.model_refs import resolve_model

    assert callable(resolve_model)


def test_register_ops_populates_registry_in_a_fresh_interpreter():
    import subprocess
    import sys

    script = (
        "from mosaic.core.pipeline.ops import list_ops\n"
        "assert not list_ops(domain='tracking'), 'tracking ops registered too early'\n"
        "from mosaic.tracking import register_ops\n"
        "register_ops()\n"
        "kinds = {e['kind'] for e in list_ops(domain='tracking')}\n"
        "assert 'infer-pose' in kinds and 'trex' in kinds, sorted(kinds)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
