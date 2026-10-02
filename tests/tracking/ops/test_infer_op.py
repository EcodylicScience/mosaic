"""The inference ops under the job contract, with the model faked.

The bridge from predictions to ``tracks/``, an entry whose bridge fails, the
completion marker, and the claim that keeps two executions off one entry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.tracking.external.runner.ultralytics_protocol import InferPointsRequest
from mosaic.tracking.pose_training.ultralytics_infer import InferenceOutcome
from mosaic.core.pipeline.run_log import read_runs, run_log_dir
from mosaic.tracking import resolve_model

from tests.helpers import (
    install_fake_point_inference,
    install_fake_pose_inference,
    stub_media_dataset,
)


# --- infer-pose op -> tracks bridge (mocked model) -------------------------


def test_infer_pose_bridges_to_tracks(tmp_path, monkeypatch):
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _ = install_fake_pose_inference(monkeypatch)

    # a raw model path (no training run needed)
    model = tmp_path / "m.pt"
    model.write_bytes(b"w")
    run_id = run_op(ds, "infer-pose", {"model": str(model), "convert_to_tracks": True})
    assert run_id.startswith("infer-pose.")

    runs = read_runs(run_log_dir(ds.base_dir), kind="infer-pose")
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
        # The body centre `mosaic_v1` requires, derived from the keypoints the
        # model reported. Neither keypoint's own coordinate, so a bridge that
        # copied one rather than averaging both would fail here.
        assert {"X", "Y"} <= set(tdf.columns)
        assert (tdf["X"] == 3.0).all()  # mean(1.0, 5.0)
        assert (tdf["Y"] == 5.0).all()  # mean(2.0, 8.0)

    # The audit parquet per sequence, under _tracking rather than a root of its
    # own. There is no inference index to assert against any more: the edge from
    # a tracks table back to the run that produced it is ``producer_run_id``,
    # asserted above through the variant directory, and the index this replaced
    # was written, never read, and non-portable.
    from mosaic.tracking.ops.infer import infer_run_root

    run_root = infer_run_root(ds, "infer-pose", run_id)
    assert {p.name for p in run_root.iterdir() if p.is_dir()} == {"vid1", "vid2"}
    for sequence in ("vid1", "vid2"):
        assert (run_root / sequence / "predictions.parquet").exists()


# --- infer under the marker protocol ---------------------------------------


def test_infer_points_runs_and_bridges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole op over the POLO seam: identity, claim, parquet, bridge, marker."""
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _ = install_fake_point_inference(monkeypatch)
    model = tmp_path / "polo.pt"
    model.write_bytes(b"w")

    run_id = run_op(ds, "infer-points", {"model": str(model)})

    from mosaic.core.pipeline.tracks_index import read_tracks_index
    from mosaic.tracking.ops.infer import infer_run_root

    assert run_id.startswith("infer-points.0.4-"), run_id
    run_root = infer_run_root(ds, "infer-points", run_id)
    for sequence in ("vid1", "vid2"):
        # Published by the runner at the path the request named, and read back
        # from there by the op to bridge it.
        assert (run_root / sequence / "predictions.parquet").exists()

    rows = read_tracks_index(ds)
    assert set(rows["sequence"]) == {"vid1", "vid2"}
    assert set(rows["producer"]) == {"infer-points"}
    assert set(rows["producer_run_id"]) == {run_id}

    # The variant records the model it was made with, by the digest of the
    # weights handed in by path, which a search for a model's tracks reads.
    from mosaic.core.pipeline.index_csv import index_records
    from mosaic.core.pipeline.tracks_identity import (
        read_tracks_variant,
        recorded_models,
    )

    (variant,) = {record["run_id"] for record in index_records(rows)}
    sidecar = read_tracks_variant(ds.get_root("tracks"), variant)
    assert sidecar is not None
    digest = resolve_model(ds, str(model), "train-points").digest
    assert recorded_models(sidecar) == (digest,)

    # A point detector already reports the body centre; only its name was
    # wrong. The bridge renames rather than copies, so the lowercase pair the
    # raw predictions use does not survive into the standardized table.
    for _, row in rows.iterrows():
        tdf = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))
        assert {"X", "Y"} <= set(tdf.columns)
        assert not {"x", "y"} & set(tdf.columns)
        assert sorted(tdf["X"].tolist()) == [1.0, 2.0, 3.0]
        assert sorted(tdf["Y"].tolist()) == [4.0, 5.0, 6.0]


def _positionless_predictions_for(
    monkeypatch: pytest.MonkeyPatch, sequences: set[str]
) -> None:
    """Make the fake POLO runner omit positions for *sequences*.

    Such a table cannot be published. The bridge lacks a body center to name, and
    strict validation refuses it. Every other sequence gets the fake's default
    table.
    """
    import mosaic.tracking.pose_training.ultralytics_infer as infer_run

    _ = install_fake_point_inference(monkeypatch)
    whole: Callable[..., InferenceOutcome] = infer_run.run_point_inference_tool

    def fake_run(
        request: InferPointsRequest, *, work_dir: Path, **kwargs: object
    ) -> InferenceOutcome:
        if sequences.isdisjoint(Path(source.path).stem for source in request.sources):
            return whole(request, work_dir=work_dir, **kwargs)
        table = pd.DataFrame({"frame": [0, 1], "confidence": [0.9, 0.8]})
        published = Path(request.output_parquet)
        published.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(published, index=False)
        return InferenceOutcome(
            predictions_path=published, n_frames=2, n_rows=len(table)
        )

    monkeypatch.setattr(infer_run, "run_point_inference_tool", fake_run)


def test_a_failed_inference_bridge_loses_only_its_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is recorded on the attempt, and the next entry still publishes.

    The refused entry keeps its predictions for diagnosis and does not get a
    completion marker, because its output never reached ``tracks/``.
    """
    from mosaic.core.pipeline.markers import read_phase_marker
    from mosaic.core.pipeline.tracks_index import read_tracks_index
    from mosaic.tracking.ops.infer import infer_run_root

    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _positionless_predictions_for(monkeypatch, {"vid1"})
    model = tmp_path / "polo.pt"
    model.write_bytes(b"w")

    run_id = run_op(ds, "infer-points", {"model": str(model)})

    runs = read_runs(run_log_dir(ds.base_dir), kind="infer-points")
    assert [(run["status"], run["entries_failed"]) for run in runs] == [("finished", 1)]
    assert set(read_tracks_index(ds)["sequence"]) == {"vid2"}
    run_root = infer_run_root(ds, "infer-points", run_id)
    assert (run_root / "vid1" / "predictions.parquet").exists()
    assert read_phase_marker(run_root / "vid1", "infer") is None
    assert read_phase_marker(run_root / "vid2", "infer") is not None


def test_an_inference_run_that_publishes_nothing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that loses every entry's table fails instead of finishing."""
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _positionless_predictions_for(monkeypatch, {"vid1", "vid2"})
    model = tmp_path / "polo.pt"
    model.write_bytes(b"w")

    with pytest.raises(AllEntriesFailed, match="vid1, vid2"):
        _ = run_op(ds, "infer-points", {"model": str(model)})

    runs = read_runs(run_log_dir(ds.base_dir), kind="infer-points")
    assert [(run["status"], run["entries_failed"]) for run in runs] == [("failed", 2)]


def test_the_predictions_the_runner_published_are_not_rewritten(tmp_path, monkeypatch):
    """A published table is read back, never copied over itself.

    The op writes the parquet only when the caller did not. For the two
    out-of-process ops the runner wrote it atomically at that exact path, so a
    second write would copy a whole table onto itself -- and the localizer, which
    still computes in this process, must keep getting its write.
    """
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    _ = install_fake_point_inference(monkeypatch)
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
    _ = install_fake_pose_inference(monkeypatch)
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

    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
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

    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
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
