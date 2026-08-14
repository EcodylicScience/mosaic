"""End-to-end ``run_sleap`` reuse and provenance, without a real SLEAP binary.

``sleap-track`` / ``sleap-convert`` are replaced with recording fakes (the shape
established in ``test_trex_run_markers.py``): the fake inference writes a ``.slp``
and the fake export writes a small, converter-readable analysis ``.h5``. This
exercises the Job-Contract machinery -- content ``run_id``, phase-marker reuse,
the analysis-h5 -> tracks bridge, and the two index writers -- with no models.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.tracking.sleap.dataset_runs as dr
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.tracking.sleap.dataset_runs import sleap_index_path, sleap_run_root
from mosaic.tracking.sleap.run import SleapConvertResult, SleapTrackResult

from tests.helpers import write_media_index

# The bridge reads the analysis HDF5 with h5py (a [recommended] extra); skip the
# whole module when it is absent rather than fail a minimal install.
pytest.importorskip("h5py")


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media_index(dataset, ["vid1"])
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
    assert int(sidx.iloc[0]["n_ids"]) == 1


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
    run_id = dr.run_sleap(ds, model_paths=[str(model)])

    write_media_index(ds, ["vid1"], uids={"vid1": "uid-bbb"})
    second = dr.run_sleap(ds, model_paths=[str(model)])

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
    run_id = dr.run_sleap(ds, model_paths=[str(model)])

    write_media_index(
        ds, ["vid1"], filenames={"vid1": "renamed.mp4"}, uids={"vid1": "uid-aaa"}
    )
    second = dr.run_sleap(ds, model_paths=[str(model)])

    assert second == run_id
    assert len(sleap.tracked) == 1, "the same bytes were inferred twice"


def test_an_absent_uid_still_falls_back_to_the_path(
    ds: Dataset, model: Path, sleap: FakeSleap
) -> None:
    """Media indexed before the identity columns carries no uid.

    Dropping the path comparison would remove the source guard from exactly the
    datasets that cannot supply a uid.
    """
    run_id = dr.run_sleap(ds, model_paths=[str(model)])
    write_media_index(ds, ["vid1"], filenames={"vid1": "vid2.mp4"})

    second = dr.run_sleap(ds, model_paths=[str(model)])

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

    run_id = dr.run_sleap(ds, model_paths=[training_run_id])

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

    by_directory = dr.run_sleap(ds, model_paths=[str(model)])
    by_run_id = dr.run_sleap(ds, model_paths=[training_run_id])

    assert by_run_id != by_directory
    sidx = pd.read_csv(sleap_index_path(ds))
    recorded = {str(row["model_id"]) for _, row in sidx.iterrows()}
    assert training_run_id in recorded, "the run_id spelling recorded no lineage"
    assert len(recorded) == 2, "one spelling recorded the other's model identity"

    # One model, reached two ways: both runs inferred from the same checkpoint.
    slp_paths = {path.name for path in sleap.tracked}
    assert slp_paths == {"vid1.mp4"}
