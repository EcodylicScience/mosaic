"""End-to-end ``run_sleap`` reuse and provenance, without a real SLEAP binary.

``sleap-nn track`` and the analysis export are replaced with recording fakes (the shape
established in ``test_trex_run_markers.py``): the fake inference writes a ``.slp``
and the fake export writes a small, converter-readable analysis ``.h5``. This
exercises the Job-Contract machinery -- content ``run_id``, phase-marker reuse,
the analysis-h5 -> tracks bridge, and the two index writers -- with no models.

A run whose last model is an embedding model predicts with ``sleap-nn predict``,
reads the vectors back from its ``.slp``, and publishes an appearance table
beside each tracks table. Its readback is ensured like the export, and a reused
tracks table is republished when its appearance table does not read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.tracking.sleap.dataset_runs as dr
from mosaic.core.dataset import Dataset
from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline._utils import staging_path
from mosaic.core.pipeline.appearance import AppearanceTable, read_appearance
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseName,
    phase_marker_path,
)
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_identity import (
    read_variant_sidecar,
    tracks_variant_root,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.runlog import read_entry_error_keys, run_log_dir
from mosaic.tracking.model_refs import resolve_model_set
from mosaic.tracking.sleap.dataset_runs import sleap_index_path, sleap_run_root
from mosaic.tracking.sleap.params import SleapParams

from tests.helpers import (
    EMBEDDING_TRACKER_SETTINGS,
    FakeSleap,
    MediaClip,
    entry_error_lines,
    install_fake_sleap,
    latest_snapshot,
    make_dataset,
    set_tracks_cell,
    stub_media_dataset,
    write_media_index,
    write_sleap_model,
    write_sleap_model_set,
)


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return stub_media_dataset(tmp_path, ["vid1"])


@pytest.fixture
def model(tmp_path: Path) -> Path:
    return write_sleap_model(tmp_path / "sleap_model")


@pytest.fixture
def sleap(monkeypatch: pytest.MonkeyPatch) -> FakeSleap:
    return install_fake_sleap(monkeypatch)


def _tracks_rows(ds: Dataset) -> pd.DataFrame:
    return read_tracks_index(ds)


# --- a fresh run produces tracks + both index rows -------------------------


def test_a_fresh_run_infers_converts_and_bridges(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    run_id = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

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
    first = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))
    assert len(sleap.tracked) == 1

    second = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))
    assert second == first
    # inference is not re-run: the phase marker proves it is done, and the .h5
    # already exists so the analysis export is not re-run either.
    assert len(sleap.tracked) == 1
    assert len(sleap.converted) == 1

    # The reuse run's index row keeps the true track count -- it re-derives it
    # from the existing parquet rather than replacing the row with a zero.
    sidx = pd.read_csv(sleap_index_path(ds))
    assert len(sidx) == 1
    assert int(sidx.iloc[0]["n_ids"]) == 1


def test_an_interrupted_analysis_export_is_not_trusted(
    ds: Dataset, model: Path, monkeypatch: pytest.MonkeyPatch, sleap: FakeSleap
) -> None:
    """A killed export must not leave a partial .h5 that reuse trusts."""

    def dying_export(slp_path: Path, output_h5: Path, **_kw: object):
        # h5py opens "w" and truncates immediately; a kill leaves the export's
        # partial file, which the export writes beside its output.
        partial = staging_path(Path(output_h5))
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial-not-a-real-h5")
        raise RuntimeError("killed mid-convert")

    monkeypatch.setattr(dr, "run_sleap_analysis_export", dying_export)
    with pytest.raises(RuntimeError):
        dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

    # The entry is reported lost, not written.
    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert snapshot["entries_written"] == 0
    assert snapshot["entries_failed"] == 1
    log_dir = run_log_dir(ds.base_dir)
    keys = read_entry_error_keys(log_dir, snapshot["execution_id"])
    assert keys == [make_entry_key("", "vid1")]

    sleap_root = ds.get_root("sleap")
    # Inference finished (its .slp and marker are valid) but no canonical .h5 was
    # published -- only a *.partial temp, which the ensure step ignores.
    assert list(sleap_root.rglob("*.predictions.slp"))
    assert not list(sleap_root.rglob("*.analysis.h5"))
    assert len(sleap.tracked) == 1

    # A working export now publishes the canonical .h5 and bridges to tracks,
    # WITHOUT re-running the already-complete inference.
    monkeypatch.setattr(dr, "run_sleap_analysis_export", sleap.convert)
    dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))
    assert len(sleap.tracked) == 1  # inference reused
    assert list(sleap_root.rglob("*.analysis.h5"))
    assert len(read_tracks_index(ds)) == 1


# --- overwrite forces a recompute ------------------------------------------


def test_overwrite_forces_a_recompute(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))
    assert len(sleap.tracked) == 1

    dr.run_sleap(ds, SleapParams(model_paths=[str(model)]), overwrite=True)
    assert len(sleap.tracked) == 2  # inference ran again


# --- every tracking setting reaches the tool ---------------------------------


def test_every_tracking_setting_reaches_the_tool(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    """A setting ``run_sleap`` forgot to pass would run at the tool's default.

    ``run_sleap_track`` gives each option a default, so an omission is silent:
    the run_id would name the params while the tracker ran something else. Each
    value here differs from that default.
    """
    params = SleapParams(
        model_paths=[str(model)],
        use_flow=False,
        candidates_method="local_queues",
        features="centroids",
        scoring_method="euclidean_dist",
        track_matching_method="greedy",
        tracking_window_size=9,
        max_tracks=3,
        max_instances=2,
        peak_threshold=0.4,
        batch_size=16,
        device="cpu",
        sleap_extra_settings={"of_img_scale": 0.5},
    )

    dr.run_sleap(ds, params)

    (settings,) = sleap.track_settings
    expected: dict[str, object] = {
        "tracking": True,
        "use_flow": False,
        "candidates_method": "local_queues",
        "features": "centroids",
        "scoring_method": "euclidean_dist",
        "track_matching_method": "greedy",
        "tracking_window_size": 9,
        "max_tracks": 3,
        "max_instances": 2,
        "peak_threshold": 0.4,
        "batch_size": 16,
        "device": "cpu",
        "extra_settings": {"of_img_scale": 0.5},
    }
    assert {key: settings.get(key) for key in expected} == expected


# --- different weights are a different run ---------------------------------


def test_different_weights_are_a_different_run(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    m1 = write_sleap_model(tmp_path / "m1", b"weights-A")
    m2 = write_sleap_model(tmp_path / "m2", b"weights-B")

    a = dr.run_sleap(ds, SleapParams(model_paths=[str(m1)]))
    b = dr.run_sleap(ds, SleapParams(model_paths=[str(m2)]))
    assert a != b
    assert sleap_run_root(ds, a) != sleap_run_root(ds, b)


# --- the reuse comparison is uid-first, with the path as fallback ------------


def test_a_video_replaced_in_place_forces_a_recompute(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    """The case a path comparison cannot see at all.

    Same sequence, same filename, different bytes. Comparing resolved paths
    calls this unchanged, so the second run reused inference over content that
    no longer exists. TREx has compared the uid first since item 8.5; SLEAP
    recorded ``source_uid`` on its markers and never read it back.
    """
    write_media_index(ds, ["vid1"], uids={"vid1": "uid-aaa"})
    run_id = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

    write_media_index(ds, ["vid1"], uids={"vid1": "uid-bbb"})
    second = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

    assert second == run_id, "settings did not change, so neither does the identity"
    assert len(sleap.tracked) == 2, "the replaced video was not re-inferred"


def test_the_same_video_under_a_new_name_is_not_a_recompute(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    """The other direction, and the saving the uid comparison buys.

    A rearrangement changes which file a sequence resolves to without changing
    the bytes. The path comparison calls that a source change and throws away
    the inference; the uid says it is the same video.
    """
    write_media_index(ds, ["vid1"], uids={"vid1": "uid-aaa"})
    run_id = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

    write_media_index(
        ds, ["vid1"], filenames={"vid1": "renamed.mp4"}, uids={"vid1": "uid-aaa"}
    )
    second = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

    assert second == run_id
    assert len(sleap.tracked) == 1, "the same bytes were inferred twice"


def test_an_absent_uid_still_falls_back_to_the_path(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    """Media indexed before the identity columns carries no uid.

    Dropping the path comparison would remove the source guard from exactly the
    datasets that cannot supply a uid.
    """
    run_id = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))
    write_media_index(ds, ["vid1"], filenames={"vid1": "vid2.mp4"})

    second = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

    assert second == run_id
    assert len(sleap.tracked) == 2, "a changed source with no uid must re-infer"


# --- train here, track with it there ---------------------------------------


def _register_training_run(ds: Dataset, model: Path, run_id: str) -> None:
    """Record *model* in ``models/train-sleap/index.csv`` as a finished run.

    Through the registrar a real ``train-sleap`` uses, not a hand-built CSV: the
    claim under test is that the tracker reads back what training wrote, and a
    row assembled here could agree with the reader while disagreeing with the
    writer.
    """
    from mosaic.tracking.ops.train import finalize_training
    from mosaic.tracking.ops.train_sleap import TrainSleapParams
    from mosaic.tracking.sleap.version import TRAIN_SLEAP_KIND

    finalize_training(
        ds,
        TRAIN_SLEAP_KIND,
        run_id,
        model,
        TrainSleapParams(labels="labels.slp"),
        base_model="",
        base_run_id="",
        base_digest="",
        best_model_path=model / "best.ckpt",
        metrics_path=model / "training_log.csv",
        n_epochs=2,
        artifact_shape="directory",
        artifact_path=model,
    )


def test_a_training_run_id_reaches_the_weights_that_run_produced(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    """The closing claim of the model-reference design: train here, track there.

    A reference is a path *or* a registered training ``run_id``, and a run_id
    resolves against ``models/<kind>/index.csv``. The kind that names that index
    is the one that *wrote* the row -- ``train-sleap`` -- so resolving under this
    tracker's own kind sent every run_id to a ``models/sleap/`` index nothing
    writes. Only a path ever resolved, and the handoff could not be spelled by
    name at all.
    """
    training_run_id = "train-sleap.0.1-4b57beb256"
    _register_training_run(ds, model, training_run_id)

    run_id = dr.run_sleap(ds, SleapParams(model_paths=[training_run_id]))

    assert run_id.startswith("sleap.1.6-")
    assert sleap.tracked, "the run resolved its model but never inferred"
    sidx = pd.read_csv(sleap_index_path(ds))
    assert str(sidx.iloc[0]["model_id"]) == training_run_id


def test_naming_the_training_run_records_lineage_a_path_cannot(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    """The two spellings reach one model and are deliberately not one identity.

    ``model_id`` is the training run when there is one and the weights digest
    otherwise, so the same checkpoint tracked by name and by path mints two
    tracker runs. That is the design rather than a collision to fix: a registered
    run names lineage a bare path has none of, and flattening the two would
    either discard the lineage or claim it for weights that carry none.

    Pinned because it is surprising, and because the tempting "both references
    are the same run" reading would be satisfied by dropping exactly the lineage
    the reference exists to carry.
    """
    training_run_id = "train-sleap.0.1-4b57beb256"
    _register_training_run(ds, model, training_run_id)

    by_directory = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))
    by_run_id = dr.run_sleap(ds, SleapParams(model_paths=[training_run_id]))

    assert by_run_id != by_directory
    sidx = pd.read_csv(sleap_index_path(ds))
    recorded = {str(row["model_id"]) for _, row in sidx.iterrows()}
    assert training_run_id in recorded, "the run_id spelling recorded no lineage"
    assert len(recorded) == 2, "one spelling recorded the other's model identity"

    # One model, reached two ways: both runs inferred from the same checkpoint.
    slp_paths = {path.name for path in sleap.tracked}
    assert slp_paths == {"vid1.mp4"}


# --- a run with an embedding model ------------------------------------------


def _embedding_params(tmp_path: Path) -> SleapParams:
    models = write_sleap_model_set(tmp_path / "models", "bottomup", "embedding")
    return SleapParams.model_validate(
        {"model_paths": models, **EMBEDDING_TRACKER_SETTINGS}
    )


def _variant(ds: Dataset) -> str:
    return str(read_tracks_index(ds).iloc[0]["run_id"])


def _appearance(ds: Dataset) -> AppearanceTable:
    table = read_appearance(ds, _variant(ds), ("", "vid1"))
    assert table is not None
    return table


def _tracks_table(ds: Dataset) -> pd.DataFrame:
    row = read_tracks_index(ds).iloc[0]
    return pd.read_parquet(ds.resolve_path(str(row["abs_path"])))


def _observed(ds: Dataset) -> dict[str, str]:
    """Return the ``observed`` block of the variant's ``params.json``."""
    root = tracks_variant_root(ds.get_root("tracks"), _variant(ds))
    sidecar = read_variant_sidecar(root / "params.json")
    assert sidecar is not None
    return sidecar.observed


def _readback(ds: Dataset, run_id: str) -> Path:
    return sleap_run_root(ds, run_id) / "vid1" / "vid1.embeddings.npz"


def _sleap_cells(ds: Dataset, column: str) -> list[str]:
    index = pd.read_csv(sleap_index_path(ds), dtype=str, keep_default_na=False)
    return index[column].tolist()


def test_an_embedding_run_writes_one_vector_per_tracks_row(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    params = _embedding_params(tmp_path)

    run_id = dr.run_sleap(ds, params)

    assert sleap.tracked == []
    assert len(sleap.predicted) == 1
    (settings,) = sleap.predict_settings
    assert settings["model_paths"] == [Path(model) for model in params.model_paths]
    assert sleap.read_back == [
        sleap_run_root(ds, run_id) / "vid1" / "vid1.predictions.slp"
    ]
    table = _appearance(ds)
    tracks = _tracks_table(ds)
    assert list(zip(table.frame.tolist(), table.id.tolist(), strict=True)) == list(
        zip(tracks["frame"].tolist(), tracks["id"].tolist(), strict=True)
    )
    assert table.vectors.shape == (6, sleap.dim)
    expected = np.eye(sleap.dim, dtype=np.float32)[np.arange(6) % sleap.dim]
    np.testing.assert_array_equal(table.vectors, expected)
    assert _sleap_cells(ds, "n_vectors_unmatched") == ["0"]


def test_a_detection_without_keypoints_drops_its_vector_and_counts_it(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export keeps a slot whose keypoints are all NaN, and the converter drops it."""
    _ = install_fake_sleap(monkeypatch, FakeSleap(nan_frames=(2,)))

    _ = dr.run_sleap(ds, _embedding_params(tmp_path))

    assert _tracks_table(ds)["frame"].tolist() == [0, 1, 3, 4, 5]
    assert _appearance(ds).frame.tolist() == [0, 1, 3, 4, 5]
    assert _sleap_cells(ds, "n_vectors_unmatched") == ["1"]


def test_a_tracks_row_without_a_vector_fails_the_entry(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_fake_sleap(monkeypatch, FakeSleap(unembedded_frames=(1, 3)))

    with pytest.raises(AllEntriesFailed):
        _ = dr.run_sleap(ds, _embedding_params(tmp_path), execution_id="unembedded")

    (line,) = entry_error_lines(ds, "unembedded")
    assert "2 tracks rows" in line
    assert "no appearance vector" in line
    assert read_tracks_index(ds).empty
    assert not list(ds.get_root("tracks").rglob("appearance/*.parquet"))


def test_a_rerun_reuses_the_readback_and_both_tables(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    params = _embedding_params(tmp_path)
    _ = dr.run_sleap(ds, params)
    published = _tracks_table(ds)

    _ = dr.run_sleap(ds, params)

    assert len(sleap.predicted) == 1
    assert len(sleap.converted) == 1
    assert len(sleap.read_back) == 1
    pd.testing.assert_frame_equal(_tracks_table(ds), published)
    assert _sleap_cells(ds, "n_vectors_unmatched") == ["0"]


def test_a_rerun_that_reuses_the_table_keeps_the_recorded_unmatched_count(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleap = install_fake_sleap(monkeypatch, FakeSleap(nan_frames=(2,)))
    params = _embedding_params(tmp_path)
    _ = dr.run_sleap(ds, params)

    _ = dr.run_sleap(ds, params)

    assert len(sleap.converted) == 1
    assert _sleap_cells(ds, "n_vectors_unmatched") == ["1"]


def test_the_readback_is_rerun_when_its_file_is_missing(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    params = _embedding_params(tmp_path)
    run_id = dr.run_sleap(ds, params)
    _readback(ds, run_id).unlink()

    _ = dr.run_sleap(ds, params)

    assert len(sleap.predicted) == 1
    assert len(sleap.converted) == 1
    assert len(sleap.read_back) == 2
    assert _readback(ds, run_id).is_file()


def test_a_reused_table_without_its_appearance_file_is_republished(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    params = _embedding_params(tmp_path)
    _ = dr.run_sleap(ds, params)
    cell = str(read_tracks_index(ds).iloc[0]["appearance_abs_path"])
    ds.resolve_path(cell).unlink()

    _ = dr.run_sleap(ds, params)

    assert len(sleap.predicted) == 1
    assert _appearance(ds).frame.tolist() == list(range(6))
    assert _sleap_cells(ds, "n_vectors_unmatched") == ["0"]


def test_a_reused_table_whose_row_records_no_appearance_file_is_republished(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    params = _embedding_params(tmp_path)
    _ = dr.run_sleap(ds, params)
    set_tracks_cell(ds, "appearance_abs_path", "")

    _ = dr.run_sleap(ds, params)

    assert len(sleap.predicted) == 1
    assert _appearance(ds).frame.tolist() == list(range(6))


def test_an_embedding_run_records_its_embedding_model_and_release(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    params = _embedding_params(tmp_path)

    _ = dr.run_sleap(ds, params)

    observed = _observed(ds)
    embedding_model = resolve_model_set(None, [params.model_paths[-1]], "sleap")
    assert observed["embedding_model_id"] == embedding_model.model_id
    assert observed["embedding_dim"] == "128"
    assert observed["sleap_nn_version"] == sleap.version


def test_the_variant_records_every_model_type_of_the_set_as_the_index_row_does(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap
) -> None:
    _ = dr.run_sleap(ds, _embedding_params(tmp_path))

    assert _observed(ds)["model_type"] == "bottomup+embedding"
    assert _sleap_cells(ds, "model_type") == ["bottomup+embedding"]


def test_a_run_without_an_embedding_model_writes_no_appearance(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    _ = dr.run_sleap(ds, SleapParams(model_paths=[str(model)]))

    assert sleap.predicted == []
    assert sleap.read_back == []
    assert str(read_tracks_index(ds).iloc[0]["appearance_abs_path"]) == ""
    assert not {"embedding_model_id", "embedding_dim", "sleap_nn_version"} & set(
        _observed(ds)
    )


@pytest.mark.parametrize("via_op", [True, False], ids=["op", "run_sleap"])
@pytest.mark.parametrize(
    ("embedding", "target"),
    [(True, "sleap-nn predict"), (False, "sleap-nn track")],
    ids=["embedding", "pose"],
)
def test_the_attempt_records_the_sleap_nn_command_it_runs(
    ds: Dataset,
    tmp_path: Path,
    model: Path,
    sleap: FakeSleap,
    *,
    embedding: bool,
    target: str,
    via_op: bool,
) -> None:
    from mosaic.core.pipeline.ops import run_op
    from mosaic.tracking import register_ops

    params = (
        _embedding_params(tmp_path)
        if embedding
        else SleapParams(model_paths=[str(model)])
    )
    if via_op:
        register_ops()
        _ = run_op(ds, "sleap", params)
    else:
        _ = dr.run_sleap(ds, params)

    assert latest_snapshot(ds)["target"] == target


# --- an entry with no detections ---------------------------------------------


def _no_detections_marker(ds: Dataset, run_id: str) -> Path:
    return sleap_run_root(ds, run_id) / "vid1" / "vid1.no-detections.json"


def test_an_entry_with_no_detections_is_written_empty(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_sleap(monkeypatch, FakeSleap(detects=False))

    run_id = dr.run_sleap(ds, _embedding_params(tmp_path))

    assert fake.converted == []
    assert fake.read_back == []
    marker = json.loads(_no_detections_marker(ds, run_id).read_text())
    assert marker == {
        "sleap_nn_version": fake.version,
        "argv": ["sleap-nn", "predict", "-i", str(fake.predicted[0])],
        "message": "No detections to embed",
    }
    row = read_tracks_index(ds).iloc[0]
    assert int(row["n_rows"]) == 0
    assert _tracks_table(ds).empty
    table = _appearance(ds)
    assert table.frame.size == 0
    assert table.vectors.shape == (0, fake.dim)
    assert _sleap_cells(ds, "slp_path") == [""]
    assert _sleap_cells(ds, "analysis_h5_path") == [""]
    assert latest_snapshot(ds)["entries_written"] == 1


def test_an_entry_with_no_detections_is_not_predicted_again(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_sleap(monkeypatch, FakeSleap(detects=False))
    params = _embedding_params(tmp_path)
    _ = dr.run_sleap(ds, params)

    _ = dr.run_sleap(ds, params)

    assert len(fake.predicted) == 1
    assert fake.converted == []
    assert _sleap_cells(ds, "slp_path") == [""]
    assert _sleap_cells(ds, "analysis_h5_path") == [""]
    assert _sleap_cells(ds, "n_vectors_unmatched") == ["0"]
    assert _tracks_table(ds).empty
    assert latest_snapshot(ds)["entries_written"] == 1


def test_a_new_prediction_clears_a_stale_no_detections_marker(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run killed after it wrote the marker but before it recorded the phase."""
    fake = install_fake_sleap(monkeypatch, FakeSleap(detects=False))
    params = _embedding_params(tmp_path)
    run_id = dr.run_sleap(ds, params)
    phase_marker_path(sleap_run_root(ds, run_id) / "vid1", "track").unlink()
    fake.detects = True

    _ = dr.run_sleap(ds, params)

    assert not _no_detections_marker(ds, run_id).exists()
    assert _appearance(ds).frame.tolist() == list(range(6))
    assert _sleap_cells(ds, "slp_path") != [""]


# --- the idle limit of an entry ------------------------------------------------


@pytest.fixture
def claimed(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the idle limit of every claim that the run writes."""
    seen: list[float] = []
    real = dr.claim

    def recording(
        ctx: JobContext, work_dir: Path, phase: PhaseName | None, idle_seconds: float
    ) -> InflightMarker:
        seen.append(idle_seconds)
        return real(ctx, work_dir, phase, idle_seconds)

    monkeypatch.setattr(dr, "claim", recording)
    return seen


def _idle_params(tmp_path: Path, **fields: object) -> SleapParams:
    models = write_sleap_model_set(tmp_path / "models", "bottomup", "embedding")
    return SleapParams.model_validate(
        {"model_paths": models, **EMBEDDING_TRACKER_SETTINGS, **fields}
    )


def test_an_embedding_entry_waits_for_its_frames(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap, claimed: list[float]
) -> None:
    """The 100 frames of the stub video earn 5 s over an idle limit of 1 s."""
    _ = dr.run_sleap(ds, _idle_params(tmp_path, idle_timeout=1))

    assert sleap.predict_settings[0]["idle_timeout"] == pytest.approx(5.0)
    assert claimed == [pytest.approx(5.0)] * 3


def test_an_embedding_entry_under_an_analysis_range_waits_for_the_range(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap, claimed: list[float]
) -> None:
    params = _idle_params(tmp_path, idle_timeout=1, analysis_range=(10, 59))

    _ = dr.run_sleap(ds, params)

    assert sleap.predict_settings[0]["idle_timeout"] == pytest.approx(2.5)
    assert claimed == [pytest.approx(2.5)] * 3


def test_an_embedding_entry_without_a_frame_count_waits_an_hour(
    tmp_path: Path, sleap: FakeSleap, claimed: list[float]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(sequence="vid1", frame_count=0)])

    _ = dr.run_sleap(ds, _idle_params(tmp_path, idle_timeout=1))

    assert sleap.predict_settings[0]["idle_timeout"] == 3600
    assert claimed == [3600] * 3


def test_an_idle_limit_above_the_frame_allowance_is_kept(
    ds: Dataset, tmp_path: Path, sleap: FakeSleap, claimed: list[float]
) -> None:
    _ = dr.run_sleap(ds, _idle_params(tmp_path, idle_timeout=900))

    assert sleap.predict_settings[0]["idle_timeout"] == 900
    assert claimed == [900] * 3


def test_a_pose_entry_waits_for_its_frames(
    ds: Dataset, model: Path, sleap: FakeSleap, claimed: list[float]
) -> None:
    """The 100 frames of the stub video earn 5 s over an idle limit of 1 s."""
    _ = dr.run_sleap(ds, SleapParams(model_paths=[str(model)], idle_timeout=1))

    assert sleap.track_settings[0]["idle_timeout"] == pytest.approx(5.0)
    assert claimed == [pytest.approx(5.0)] * 2


def test_a_pose_entry_without_a_frame_count_waits_an_hour(
    tmp_path: Path, model: Path, sleap: FakeSleap, claimed: list[float]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(sequence="vid1", frame_count=0)])

    _ = dr.run_sleap(ds, SleapParams(model_paths=[str(model)], idle_timeout=1))

    assert sleap.track_settings[0]["idle_timeout"] == 3600
    assert claimed == [3600] * 2
