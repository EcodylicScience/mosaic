"""Tests for the tracking-op registry and the ops under the Job Contract.

Uses a real ``Dataset`` + a synthetic ``media/index.csv`` and monkeypatches the
heavy low-level backends (video decode, ultralytics, torch) so the contract
machinery -- run-log lifecycle, content ``run_id``, progress, cancel, lineage,
and the inference->tracks bridge -- is exercised without any real models.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import (
    CHROME_149,
    DEFAULT_THRESHOLDS,
    MediaFacts,
    MediaProbeError,
    derive,
)

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.ops import OPS, describe_op, list_ops, run_op
from mosaic.tracking.external.runner.ultralytics_protocol import ProbeResponse
from mosaic.tracking.pose_training.ultralytics_infer import InferenceOutcome
from mosaic.core.pipeline.run_log import (
    read_run,
    read_run_progress,
    read_runs,
    run_log_dir,
)
from mosaic.tracking import resolve_model
from mosaic.tracking.frame_extraction.dataset_runs import ExtractFramesParams

from tests.helpers import make_dataset


# --- fixtures --------------------------------------------------------------


def _clean_media_facts(
    *, width: int, height: int, fps: float, frame_count: int, codec: str
) -> MediaFacts:
    """An analysis-clean :class:`MediaFacts` for a synthetic media-index row.

    No probe runs over the fixture's placeholder video bytes, so this reuses
    :func:`store_facts`'s hand-built shape (declared values matching the
    measured ones, empty identity fields) and overrides only the three fields
    where a plain mp4 differs from an imgstore: a real container and pixel
    format, and a moov atom at the start rather than store_facts's "no such
    concept" ``None``.
    """
    facts = store_facts(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        codec=codec,
        duration=frame_count / fps,
        video_uuid="",
        identity_scheme="",
    )
    return dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )


def _clean_facts_cells(
    *, width: int, height: int, fps: float, frame_count: int, codec: str
) -> dict[str, object]:
    """Flat + JSON facts cells describing one clean, analysis-fit media row."""
    facts = _clean_media_facts(
        width=width, height=height, fps=fps, frame_count=frame_count, codec=codec
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def _make_dataset(tmp_path: Path, seqs=("vid1", "vid2")) -> Dataset:
    ds = make_dataset(tmp_path)
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
                **_clean_facts_cells(
                    width=640, height=480, fps=30.0, frame_count=100, codec="h264"
                ),
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
    assert "sleap" in kinds
    for op in list_ops(domain="tracking"):
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
    assert {"data", "epochs", "model", "train_overrides"} <= set(schema["properties"])


def test_train_points_describes_the_polo_knobs():
    """What a front-end form renders for POLO training.

    ``train_overrides`` is what carries the hyperparameters mosaic has no field
    for -- the learning-rate schedule the deployed detector is trained on -- so a
    caller can reproduce it through the op rather than around it.
    """
    d = describe_op("train-points")
    assert d["category"] == "train"
    assert {"data", "epochs", "loc", "loc_loss", "dor", "train_overrides"} <= set(
        d["params_schema"]["properties"]
    )


def test_train_overrides_may_not_shadow_what_the_op_supplies():
    """Refuse at submit time what would otherwise fail on the GPU node.

    A key naming a trainer parameter arrives as a duplicate keyword and raises
    ``TypeError`` from Python itself, after the job has been accepted, queued and
    scheduled. ``data`` and ``task`` are worse: the trainer only builds those
    internally, so an override replaces them silently and the run trains on data
    its identifier does not describe.
    """
    from pydantic import ValidationError

    from mosaic.tracking.ops.train import PointTrainParams

    assert PointTrainParams.model_validate(
        {"data": "d.yaml", "train_overrides": {"lr0": 0.0044, "weight_decay": 0.000139}}
    ).train_overrides == {"lr0": 0.0044, "weight_decay": 0.000139}

    for shadowed in ("epochs", "imgsz", "patience", "loc", "backend"):
        with pytest.raises(ValidationError, match=shadowed):
            _ = PointTrainParams.model_validate(
                {"data": "d.yaml", "train_overrides": {shadowed: 1}}
            )
    for supplied in ("data", "task", "project", "name", "callback", "cancel_check"):
        with pytest.raises(ValidationError, match=supplied):
            _ = PointTrainParams.model_validate(
                {"data": "d.yaml", "train_overrides": {supplied: "x"}}
            )


def test_augmentation_accepts_the_dict_forms():
    """``resolve_augmentation`` has always taken a dict; the op used to narrow it.

    Both forms it accepts -- a preset plus overrides, and a bare override set --
    were unreachable through the op and so through the CLI and the API.
    """
    from mosaic.tracking.ops.train import PoseTrainParams

    preset_plus = PoseTrainParams.model_validate(
        {"data": "d.yaml", "augmentation": {"preset": "medium", "flipud": 0.5}}
    )
    assert preset_plus.augmentation == {"preset": "medium", "flipud": 0.5}
    assert (
        PoseTrainParams.model_validate(
            {"data": "d.yaml", "augmentation": "medium"}
        ).augmentation
        == "medium"
    )


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
    assert r1.startswith("train-pose.")
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
    resolved = resolve_model(ds, r1, "train-pose")
    assert resolved.path.name == "best.pt"
    assert resolved.run_id == r1
    # A registered model is named by its run, so the digest never reaches
    # identity -- but it is measured and recorded either way.
    assert resolved.model_id == r1
    assert len(resolved.digest) == 16

    # retrain from r1 -> lineage recorded
    r2 = run_op(
        ds, "train-pose", {"data": str(data_yaml), "epochs": 2, "base_model": r1}
    )
    mdf2 = trained_model_index(model_index_path(ds, "train-pose")).read(run_id=r2)
    assert mdf2.iloc[0]["base_run_id"] == r1
    assert mdf2.iloc[0]["base_digest"] == resolved.digest


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


def _install_fake_point_trainer(monkeypatch) -> dict[str, object]:
    """Stand in for the POLO trainer, recording what the op handed it."""
    import mosaic.tracking.pose_training.train as tr

    seen: dict[str, object] = {}

    def fake_train(
        data_yaml, *, project, name, callback=None, cancel_check=None, epochs=1, **kw
    ):
        seen.update(kw)
        seen["data_yaml"] = str(data_yaml)
        run_dir = Path(project) / name
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"weights")
        (run_dir / "results.csv").write_text("epoch,loss\n0,0.1\n")
        for e in range(epochs):
            if callback is not None:
                callback.on_epoch_end(e, epochs, {"loss": 0.1})
        return None

    monkeypatch.setattr(tr, "train_point_model", fake_train)
    return seen


def test_train_points_lifecycle_and_polo_knobs(tmp_path, monkeypatch):
    """``train-points`` executes, registers, and delivers its fork-only arguments.

    ``train-pose`` has had an execution test since it landed and ``train-points``
    has had none, so every POLO-specific term -- ``loc``, ``loc_loss``, ``dor``,
    ``backend`` -- was declared, hashed into the identifier, and never once
    observed arriving at a trainer. They are asserted here because they are
    exactly what a move to another process has to carry across.
    """
    ds = _make_dataset(tmp_path)
    seen = _install_fake_point_trainer(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [bee]\nradii: {0: 5.0}\n")

    run_id = run_op(
        ds,
        "train-points",
        {
            "data": str(data_yaml),
            "epochs": 2,
            "device": "cpu",
            "loc": 7.5,
            "loc_loss": "hausdorff",
            "dor": 0.6,
        },
    )

    assert run_id.startswith("train-points.")
    row = read_run(
        _run_dir(ds), read_runs(_run_dir(ds), kind="train-points")[0]["execution_id"]
    )
    assert row["status"] == "finished" and row["run_id"] == run_id
    assert row["progress_done"] == 2 and row["progress_total"] == 2

    assert seen["loc"] == 7.5
    assert seen["loc_loss"] == "hausdorff"
    assert seen["dor"] == 0.6
    assert seen["backend"] == "polo"
    assert seen["model"] == "polo26n.yaml"

    from mosaic.core.pipeline.models import model_index_path
    from mosaic.tracking.ops.train import trained_model_index

    rows = trained_model_index(model_index_path(ds, "train-points")).read(run_id=run_id)
    assert len(rows) == 1
    assert rows.iloc[0]["best_model_path"].endswith("best.pt")

    resolved = resolve_model(ds, run_id, "train-points")
    assert resolved.path.name == "best.pt"
    assert resolved.run_id == run_id


def test_a_point_knob_moves_the_identity(tmp_path, monkeypatch):
    """``dor`` names a different model here, unlike on the inference path.

    On ``infer-points`` the same field reaches no argument at all, which is its
    own recorded defect. On this op it is a real ``train`` keyword, so two values
    are two models and the identifier has to say so.
    """
    ds = _make_dataset(tmp_path)
    _ = _install_fake_point_trainer(monkeypatch)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [bee]\nradii: {0: 5.0}\n")
    base = {"data": str(data_yaml), "epochs": 1, "device": "cpu"}

    first = run_op(ds, "train-points", dict(base))
    second = run_op(ds, "train-points", {**base, "dor": 0.55})

    assert first != second


def test_train_localizer_mints_and_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The localizer op reaches its minter, and records what it produced.

    Nothing used to run this op. The registry tests assert the kind is
    registered and the golden corpus calls ``train_run_id`` directly with fixed
    arguments, so both stayed green while ``TrainLocalizerOp.run`` passed it one
    argument too many -- a ``TypeError`` on every real localizer training run.
    Only the type checker saw it. This is the test that would have.
    """
    import mosaic.tracking.pose_training.localizer_train as lt
    from mosaic.core.pipeline.models import model_index_path
    from mosaic.tracking.ops.train import trained_model_index

    ds = _make_dataset(tmp_path)
    dataset_dir = tmp_path / "patches"
    (dataset_dir / "train").mkdir(parents=True)
    _ = (dataset_dir / "train" / "patches.npy").write_bytes(b"patches")

    def fake_train_localizer(
        dataset_dir: str | Path,
        *,
        project: str | Path,
        name: str,
        epochs: int = 1,
        **kw: object,
    ) -> lt.TrainingResult:
        run_dir = Path(project) / name
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        weights = run_dir / "weights" / "best.pt"
        _ = weights.write_bytes(b"localizer weights")
        _ = (run_dir / "results.csv").write_text("epoch,loss\n0,0.1\n")
        return lt.TrainingResult(
            best_model_path=weights,
            last_model_path=weights,
            run_dir=run_dir,
            best_epoch=0,
            best_val_loss=0.1,
        )

    monkeypatch.setattr(lt, "train_localizer", fake_train_localizer)

    run_id = run_op(
        ds,
        "train-localizer",
        {"dataset_dir": str(dataset_dir), "epochs": 1, "device": "cpu"},
    )
    assert run_id.startswith("train-localizer.")

    index = trained_model_index(model_index_path(ds, "train-localizer"))
    row = index.read().iloc[0]
    assert row["run_id"] == run_id
    assert row["status"] == "finished"
    assert str(row["best_model_path"]).endswith("best.pt")


# --- infer-pose op -> tracks bridge (mocked model) -------------------------


def _fake_pose_backend(monkeypatch) -> None:
    """Replace the two module-scope seams `infer-pose` reaches its model through.

    The op now spawns a runner in the Ultralytics environment, so what a test
    stands in for is the probe and the tool call -- not an in-process function
    returning results objects. The stand-in writes a real parquet at the path the
    request names, because the op reads it back to bridge it, exactly as the
    runner would have written it.
    """
    import mosaic.tracking.common.ultralytics_env as tool_env
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    def fake_probe(model_path, **_kwargs):
        return ProbeResponse(
            has_ultralytics=True,
            has_lap=True,
            has_locate=False,
            ultralytics_version="8.4.63",
            tracker_names=[],
            model_task="pose",
            n_keypoints=1,
            model_load_error="",
            installed_tracker_table={},
        )

    def fake_run(request, *, work_dir, **_kwargs):
        table = pd.DataFrame(
            {
                "frame": range(4),
                "id": [0] * 4,
                "poseX0": [1.0] * 4,
                "poseY0": [2.0] * 4,
                "poseP0": [0.9] * 4,
            }
        )
        published = Path(request.output_parquet)
        published.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(published, index=False)
        return InferenceOutcome(
            predictions_path=published, n_frames=4, n_rows=len(table)
        )

    monkeypatch.setattr(tool_env, "probe_environment", fake_probe)
    monkeypatch.setattr(infer_run, "run_pose_inference_tool", fake_run)


def test_infer_pose_bridges_to_tracks(tmp_path, monkeypatch):
    ds = _make_dataset(tmp_path)
    _fake_pose_backend(monkeypatch)

    # a raw model path (no training run needed)
    model = tmp_path / "m.pt"
    model.write_bytes(b"w")
    run_id = run_op(ds, "infer-pose", {"model": str(model), "convert_to_tracks": True})
    assert run_id.startswith("infer-pose.")

    runs = read_runs(_run_dir(ds), kind="infer-pose")
    assert len(runs) == 1 and runs[0]["status"] == "finished"

    # standardized tracks written for each sequence, under the variant directory
    # the bridge minted -- located through the index, as every reader does.
    tracks_idx = pd.read_csv(ds.get_root("tracks") / "index.csv")
    assert set(tracks_idx["sequence"]) == {"vid1", "vid2"}
    for _, row in tracks_idx.iterrows():
        tp = ds.resolve_path(str(row["abs_path"]))
        assert tp.exists()
        assert tp.parent.name.startswith("infer-pose.")
        tdf = pd.read_parquet(tp)
        assert {"frame", "time", "id", "group", "sequence", "poseX0", "poseY0"} <= set(
            tdf.columns
        )

    # The audit parquet per sequence, under _tracking rather than a root of its
    # own (item 8.7). There is no inference index to assert against any more:
    # the edge from a tracks table back to the run that produced it is
    # ``producer_run_id``, asserted above through the variant directory, and the
    # index this replaced was written, never read, and non-portable.
    from mosaic.tracking.ops.infer import infer_run_root

    run_root = infer_run_root(ds, "infer-pose", run_id)
    assert {p.name for p in run_root.iterdir() if p.is_dir()} == {"vid1", "vid2"}
    for sequence in ("vid1", "vid2"):
        assert (run_root / sequence / "predictions.parquet").exists()


# --- infer under the marker protocol (items 8.2 / 8.3, via 8.7) ------------


def _fake_points_backend(monkeypatch) -> None:
    """The same two seams, for the POLO fork's environment.

    `infer-points` had no execution test at all before it ran out of process:
    nothing in the suite called the op, faked its backend, or installed the
    `polo` extra, and no CI job did either. This is the first.
    """
    import mosaic.tracking.common.ultralytics_env as tool_env
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    def fake_probe(model_path, **_kwargs):
        return ProbeResponse(
            has_ultralytics=True,
            has_lap=True,
            has_locate=True,
            ultralytics_version="8.4.84",
            tracker_names=[],
            model_task="locate",
            n_keypoints=1,
            model_load_error="",
            installed_tracker_table={},
        )

    def fake_run(request, *, work_dir, **_kwargs):
        table = pd.DataFrame(
            {
                "frame": [0, 0, 1],
                "detection_id": [0, 1, 0],
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "confidence": [0.9, 0.8, 0.7],
                "class_id": [0, 0, 1],
                "class_name": ["bee", "bee", "feeder"],
            }
        )
        published = Path(request.output_parquet)
        published.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(published, index=False)
        return InferenceOutcome(
            predictions_path=published, n_frames=2, n_rows=len(table)
        )

    monkeypatch.setattr(tool_env, "probe_environment", fake_probe)
    monkeypatch.setattr(infer_run, "run_point_inference_tool", fake_run)


def test_infer_points_runs_and_bridges(tmp_path, monkeypatch):
    """The whole op over the POLO seam: identity, claim, parquet, bridge, marker."""
    ds = _make_dataset(tmp_path)
    _fake_points_backend(monkeypatch)
    model = tmp_path / "polo.pt"
    model.write_bytes(b"w")

    run_id = run_op(ds, "infer-points", {"model": str(model)})

    from mosaic.core.pipeline.tracks_index import read_tracks_index
    from mosaic.tracking.ops.infer import infer_run_root

    assert run_id.startswith("infer-points.0.2-"), run_id
    run_root = infer_run_root(ds, "infer-points", run_id)
    for sequence in ("vid1", "vid2"):
        # Published by the runner at the path the request named, and read back
        # from there by the op to bridge it.
        assert (run_root / sequence / "predictions.parquet").exists()

    rows = read_tracks_index(ds)
    assert set(rows["sequence"]) == {"vid1", "vid2"}
    assert set(rows["producer"]) == {"infer-points"}
    assert set(rows["producer_run_id"]) == {run_id}


def test_the_predictions_the_runner_published_are_not_rewritten(tmp_path, monkeypatch):
    """A published table is read back, never copied over itself.

    The op writes the parquet only when the caller did not. For the two
    out-of-process ops the runner wrote it atomically at that exact path, so a
    second write would copy a whole table onto itself -- and the localizer, which
    still computes in this process, must keep getting its write.
    """
    ds = _make_dataset(tmp_path)
    _fake_points_backend(monkeypatch)
    model = tmp_path / "polo.pt"
    model.write_bytes(b"w")

    written: list[Path] = []
    import mosaic.tracking.ops.infer as infer_op

    real_write = infer_op.write_parquet_atomic

    def counted(frame, path, *args, **kwargs):
        written.append(Path(path))
        return real_write(frame, path, *args, **kwargs)

    monkeypatch.setattr(infer_op, "write_parquet_atomic", counted)
    _ = run_op(ds, "infer-points", {"model": str(model)})

    assert not [p for p in written if p.name == "predictions.parquet"], (
        f"the op re-wrote a parquet the runner had already published: {written}"
    )


def _fake_pose_model(monkeypatch, tmp_path) -> Path:
    """Patch the pose backend out and return a bare weights path."""
    _fake_pose_backend(monkeypatch)
    model = tmp_path / "m.pt"
    model.write_bytes(b"w")
    return model


def test_a_finished_inference_entry_carries_a_completion_marker(tmp_path, monkeypatch):
    """The sweeper reads markers, not producers.

    Inference was the one thing writing under ``_tracking`` that spoke none of
    the protocol -- so a sweeper would have had to special-case it, or fall back
    to mtime for that root alone, which defeats writing it once.
    """
    from mosaic.core.pipeline.markers import read_inflight, read_phase_marker
    from mosaic.tracking.ops.infer import infer_run_root

    ds = _make_dataset(tmp_path)
    model = _fake_pose_model(monkeypatch, tmp_path)

    run_id = run_op(ds, "infer-pose", {"model": str(model), "convert_to_tracks": True})

    seq_dir = infer_run_root(ds, "infer-pose", run_id) / "vid1"
    marker = read_phase_marker(seq_dir, "infer")
    assert marker is not None
    assert marker.run_id == run_id
    assert marker.completed_at
    assert marker.recorded_output.endswith("predictions.parquet")
    # And the claim is released, or the next run reads a dead directory as busy.
    assert read_inflight(seq_dir) is None


def test_an_entry_held_by_another_execution_is_skipped(tmp_path, monkeypatch):
    """A claim, not a cache -- two writers on one predictions.parquet.

    Asserted by the *absence* of output rather than by a log line: the point is
    that the second execution did not write, not that it said so.
    """
    import shutil

    from mosaic.core.pipeline.markers import new_inflight, write_inflight
    from mosaic.tracking.ops.infer import infer_run_root

    ds = _make_dataset(tmp_path)
    model = _fake_pose_model(monkeypatch, tmp_path)

    # Learn the identifier by running, rather than predicting it: ``model_id``
    # for bare weights is their content digest, and a hand-built prediction that
    # got it wrong would plant the claim in a directory nothing ever visits --
    # passing the "did not write" half for the wrong reason.
    run_id = run_op(ds, "infer-pose", {"model": str(model), "convert_to_tracks": True})
    run_root = infer_run_root(ds, "infer-pose", run_id)
    shutil.rmtree(run_root)

    held = run_root / "vid1"
    held.mkdir(parents=True)
    write_inflight(
        held,
        new_inflight(
            execution_id="someone-else",
            host="other-host",
            pid=1,
            phase="infer",
            idle_seconds=3600.0,
        ),
    )

    assert (
        run_op(ds, "infer-pose", {"model": str(model), "convert_to_tracks": True})
        == run_id
    )

    assert not (held / "predictions.parquet").exists(), "wrote into a held directory"
    assert (run_root / "vid2" / "predictions.parquet").exists(), (
        "an unheld entry must still run"
    )


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
    assert direct.startswith("trex.")


def test_trex_params_exclude_throughput_from_run_id():
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.ops.trex import TrexParams

    a = TrexParams(
        detect_model="m.pt",
        idle_timeout=900,
        max_runtime=None,
        overwrite=False,
        convert_to_tracks=True,
    )
    b = TrexParams(
        detect_model="m.pt",
        idle_timeout=30,
        max_runtime=60,
        overwrite=True,
        convert_to_tracks=False,
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    c = TrexParams(detect_model="other.pt")
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


# --- sleap op (registered; run_id parity with the standalone run_sleap) -----


def _fake_sleap_model(root: Path, name: str = "model") -> Path:
    """A minimal SLEAP model directory with a checkpoint, for identity tests."""
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.ckpt").write_bytes(b"weights")
    return model_dir


def test_sleap_registered_as_gpu_convert_op():
    assert "sleap" in OPS
    d = describe_op("sleap")
    assert d["category"] == "convert"
    assert {"model_paths", "tracker", "entries"} <= set(
        d["params_schema"]["properties"]
    )
    from mosaic.core.pipeline.ops import op_resource_class

    # declared "gpu" despite category "convert" (SLEAP inference wants the GPU)
    assert op_resource_class("sleap") == "gpu"


def test_sleap_op_run_id_matches_standalone_run_sleap(tmp_path):
    # SleapOp must produce the same content run_id as calling run_sleap directly for the
    # same settings. Scope to a missing sequence so the run short-circuits (empty media)
    # after the model resolves but before any sleap binary is used.
    from mosaic.tracking import run_sleap

    ds = _make_dataset(tmp_path)
    model = _fake_sleap_model(tmp_path)
    direct = run_sleap(ds, model_paths=[str(model)], sequences=["nonexistent"])
    via_op = run_op(
        ds, "sleap", {"model_paths": [str(model)], "sequences": ["nonexistent"]}
    )
    assert direct == via_op
    assert direct.startswith("sleap.1.6-")


def test_sleap_params_exclude_throughput_from_run_id():
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.ops.sleap import SleapParams

    a = SleapParams(
        model_paths=["m"],
        batch_size=4,
        device=None,
        idle_timeout=900,
        max_runtime=None,
        overwrite=False,
        convert_to_tracks=True,
    )
    b = SleapParams(
        model_paths=["m"],
        batch_size=16,
        device="cpu",
        idle_timeout=30,
        max_runtime=60,
        overwrite=True,
        convert_to_tracks=False,
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    c = SleapParams(model_paths=["m"], peak_threshold=0.5)
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


def test_sleap_model_identity_is_content_not_path(tmp_path):
    # Two model directories with identical weights mint the same model_id (and so
    # the same run_id); different weights mint a different one. "Name the weights,
    # not the path they sat at."
    from mosaic.tracking.model_refs import resolve_model_set

    a = tmp_path / "a" / "model"
    b = tmp_path / "b" / "model"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "best.ckpt").write_bytes(b"same-weights")
    c = tmp_path / "c" / "model"
    c.mkdir(parents=True)
    (c / "best.ckpt").write_bytes(b"other-weights")

    id_a = resolve_model_set(None, [str(a)], "sleap").model_id
    id_b = resolve_model_set(None, [str(b)], "sleap").model_id
    id_c = resolve_model_set(None, [str(c)], "sleap").model_id
    assert id_a == id_b  # same content, different paths -> same identity
    assert id_a != id_c  # different content -> different identity


def test_sleap_model_order_is_significant(tmp_path):
    # Top-down passes two directories (centroid, then centered-instance); the order
    # is not interchangeable, so it must reach identity.
    from mosaic.tracking.model_refs import resolve_model_set

    d1 = tmp_path / "centroid"
    d2 = tmp_path / "instance"
    d1.mkdir()
    d2.mkdir()
    (d1 / "best.ckpt").write_bytes(b"centroid")
    (d2 / "best.ckpt").write_bytes(b"instance")
    forward = resolve_model_set(None, [str(d1), str(d2)], "sleap").model_id
    reverse = resolve_model_set(None, [str(d2), str(d1)], "sleap").model_id
    assert forward != reverse


def test_sleap_unresolvable_model_raises(tmp_path):
    from mosaic.tracking.model_refs import resolve_model_set

    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(tmp_path / "missing")], "sleap")
    empty = tmp_path / "no_ckpt"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(empty)], "sleap")


# --- litpose op (registered; run_id parity with the standalone run_litpose) -


def _fake_litpose_model(
    root: Path, name: str = "lp_model", weights: bytes = b"weights"
) -> Path:
    """A minimal Lightning Pose model directory (config.yaml + a checkpoint)."""
    model_dir = root / name
    ckpt = model_dir / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(weights)
    (model_dir / "config.yaml").write_text(
        "model:\n  model_type: heatmap\ndata:\n  keypoint_names: [nose, tail]\n"
    )
    return model_dir


def test_litpose_registered_as_gpu_convert_op():
    assert "litpose" in OPS
    d = describe_op("litpose")
    assert d["category"] == "convert"
    assert {"model_path", "litpose_overrides", "entries"} <= set(
        d["params_schema"]["properties"]
    )
    from mosaic.core.pipeline.ops import op_resource_class

    # declared "gpu" despite category "convert" (LP video inference needs the GPU)
    assert op_resource_class("litpose") == "gpu"


def test_litpose_op_run_id_matches_standalone_run_litpose(tmp_path):
    # LitposeOp must produce the same content run_id as calling run_litpose directly
    # for the same settings. Scope to a missing sequence so the run short-circuits
    # (empty media) after the model resolves but before any litpose binary is used.
    from mosaic.tracking import run_litpose

    ds = _make_dataset(tmp_path)
    model = _fake_litpose_model(tmp_path)
    direct = run_litpose(ds, model_path=str(model), sequences=["nonexistent"])
    via_op = run_op(
        ds, "litpose", {"model_path": str(model), "sequences": ["nonexistent"]}
    )
    assert direct == via_op
    assert direct.startswith("litpose.2.3-")


def test_litpose_params_exclude_throughput_from_run_id():
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.ops.litpose import LitposeParams

    a = LitposeParams(
        model_path="m",
        precision="fp32",
        idle_timeout=900,
        max_runtime=None,
        overwrite=False,
        convert_to_tracks=True,
    )
    b = LitposeParams(
        model_path="m",
        precision="fp16",
        idle_timeout=30,
        max_runtime=60,
        overwrite=True,
        convert_to_tracks=False,
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    c = LitposeParams(
        model_path="m", litpose_overrides={"data.image_resize_dims.height": 256}
    )
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


def test_litpose_model_identity_is_content_not_path(tmp_path):
    # Two model directories with identical config + weights mint the same model_id
    # (and so the same run_id); different weights mint a different one.
    from mosaic.tracking.model_refs import resolve_model_set

    a = _fake_litpose_model(tmp_path / "a", weights=b"same-weights")
    b = _fake_litpose_model(tmp_path / "b", weights=b"same-weights")
    c = _fake_litpose_model(tmp_path / "c", weights=b"other-weights")

    id_a = resolve_model_set(None, [str(a)], "litpose").model_id
    id_b = resolve_model_set(None, [str(b)], "litpose").model_id
    id_c = resolve_model_set(None, [str(c)], "litpose").model_id
    assert id_a == id_b  # same content, different paths -> same identity
    assert id_a != id_c  # different weights -> different identity


def test_litpose_config_is_part_of_identity(tmp_path):
    # config.yaml shapes the output (resize dims, keypoint names), so it reaches
    # identity: same weights + different config -> different run.
    from mosaic.tracking.model_refs import resolve_model_set

    a = _fake_litpose_model(tmp_path / "a")
    b = _fake_litpose_model(tmp_path / "b")
    (b / "config.yaml").write_text(
        "model:\n  model_type: heatmap\ndata:\n  keypoint_names: [nose, tail, mid]\n"
    )
    assert (
        resolve_model_set(None, [str(a)], "litpose").model_id
        != resolve_model_set(None, [str(b)], "litpose").model_id
    )


def test_litpose_unresolvable_model_raises(tmp_path):
    from mosaic.tracking.model_refs import resolve_model_set

    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(tmp_path / "missing")], "litpose")
    # a checkpoint but no config.yaml
    no_config = tmp_path / "no_config"
    ckpt = no_config / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"w")
    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(no_config)], "litpose")
    # a config.yaml but no checkpoint
    no_ckpt = tmp_path / "no_ckpt"
    no_ckpt.mkdir()
    (no_ckpt / "config.yaml").write_text("model: {}\n")
    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(no_ckpt)], "litpose")


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
    assert run_id.startswith("convert-points.")

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
    from mosaic.tracking.trex.run import TRexConvertResult, TRexTrackResult

    seen: list[Path] = []

    def fake_convert(video_path, seq_dir, **kw):
        # A tuple of sources now, one element per clip; this entry has one.
        seen.append(Path(video_path[0]))
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
