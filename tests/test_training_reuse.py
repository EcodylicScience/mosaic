"""An identical training resubmission must not retrain.

Every train op computes a content-addressed ``run_id`` over (params, data
fingerprint, base model), mkdirs that run root, and then calls its trainer
**unconditionally**. There is no completed-marker check, unlike the tracker phases,
which gate reuse on exactly such a marker. So resubmitting identical work spends
the GPU hours again and overwrites the artifacts in place.

The index row is not the damage: ``trained_model_index`` dedups on ``run_id``, so a
resubmission replaces its row rather than duplicating it. The damage is the
recompute, and -- because two executions of one ``run_id`` write into one run root
-- the possibility of two trainers interleaving ``best.pt`` / ``last.pt`` /
``results.csv``. That second half is a claim question rather than an inventory one
and is not what this pins; this pins that a *finished* run is not repeated.

The completion evidence is the ``models/<kind>/index.csv`` row, which
``finalize_training`` writes only after the trainer returns. Deliberately not
``best.pt``: Ultralytics writes that progressively, so artifact presence alone
would adopt a half-trained model as finished.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.markers import new_inflight, write_inflight
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.ops import run_op
from mosaic.tracking.ops._common import RunRootHeld, fingerprint_yolo_dataset
from mosaic.tracking.ops.train import trained_model_index
from tests.test_tracking_ops import _make_dataset


class _Counter:
    """A trainer stand-in that records how many times it really trained."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mosaic.tracking.pose_training.train as tr

        def fake_train(
            data_yaml: object,
            *,
            project: str,
            name: str,
            callback: object = None,
            cancel_check: object = None,
            epochs: int = 1,
            **kw: object,
        ) -> None:
            self.calls += 1
            self.last_kwargs = dict(kw)
            run_dir = Path(project) / name
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            _ = (run_dir / "weights" / "best.pt").write_bytes(b"weights")
            _ = (run_dir / "results.csv").write_text("epoch,loss\n0,0.1\n")

        monkeypatch.setattr(tr, "train_pose_model", fake_train)


def _data_yaml(tmp_path: Path) -> Path:
    """A converted-dataset directory holding just the data.yaml.

    Its own directory because that is the layout ``convert-points`` produces, not
    because the fingerprint requires it: ``fingerprint_yolo_dataset`` digests what
    the YAML *declares*, so a data.yaml sharing a directory with anything else --
    including whatever the run itself writes -- fingerprints the same either way.
    ``test_an_unrelated_sibling_does_not_move_the_identity`` is what pins that.
    """
    directory = tmp_path / "converted"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data.yaml"
    _ = path.write_text("kpt_shape: [4, 3]\n")
    return path


def _yolo_dataset(root: Path, *, image: bytes = b"train-image") -> Path:
    """A minimal YOLO dataset: a data.yaml naming two split directories.

    ``path`` is written absolute because that is what ``make_data_yaml`` writes
    (``os.path.abspath(dataset_root)``), which is exactly the spelling the
    fingerprint must not depend on.
    """
    (root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (root / "valid" / "images").mkdir(parents=True, exist_ok=True)
    _ = (root / "train" / "images" / "a.png").write_bytes(image)
    _ = (root / "valid" / "images" / "b.png").write_bytes(b"val-image")
    data_yaml = root / "data.yaml"
    _ = data_yaml.write_text(
        f"path: {root.resolve()}\ntrain: train/images\nval: valid/images\n"
        "nc: 1\nnames: [bee]\n"
    )
    return data_yaml


def test_an_identical_resubmission_does_not_retrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _make_dataset(tmp_path)
    trainer = _Counter()
    trainer.install(monkeypatch)
    params = {"data": str(_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    first = run_op(ds, "train-pose", dict(params))
    second = run_op(ds, "train-pose", dict(params))

    # Same work, so the same content-addressed identifier...
    assert first == second
    # ...and the trainer ran once, not twice.
    assert trainer.calls == 1
    # The row is still there exactly once (the index dedups on run_id).
    rows = trained_model_index(model_index_path(ds, "train-pose")).read(run_id=first)
    assert len(rows) == 1


def test_overwrite_retrains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch: a caller who means it can force the work again."""
    ds = _make_dataset(tmp_path)
    trainer = _Counter()
    trainer.install(monkeypatch)
    params = {"data": str(_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    first = run_op(ds, "train-pose", dict(params))
    second = run_op(ds, "train-pose", {**params, "overwrite": True})

    # `overwrite` must not move the identifier: it is a throughput knob, not a
    # property of the model, so it is excluded from the identity payload.
    assert first == second
    assert trainer.calls == 2


def test_an_incomplete_run_root_retrains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run root without a finished row is not a finished run.

    This is the interrupted-training case: the directory and the identity marker
    are on disk, and even a partially written ``best.pt`` may be, but no row says
    the trainer returned. Adopting that as complete would ship a half-trained
    model, which is why the gate reads the index rather than the artifact.
    """
    ds = _make_dataset(tmp_path)
    trainer = _Counter()
    trainer.install(monkeypatch)
    params = {"data": str(_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    run_id = run_op(ds, "train-pose", dict(params))

    # Drop the completion record, keep the directory and its artifacts.
    index_path = model_index_path(ds, "train-pose")
    index_path.unlink()
    assert model_run_root(ds, "train-pose", run_id).exists()
    assert (
        model_run_root(ds, "train-pose", run_id) / "train" / "weights" / "best.pt"
    ).exists()

    again = run_op(ds, "train-pose", dict(params))

    assert again == run_id
    assert trainer.calls == 2


def test_a_second_execution_cannot_train_into_a_held_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two executions of one identifier would interleave one run root's artifacts.

    For a nondeterministic trainer that is a corrupt model, not a slow one, so the
    second execution fails naming the holder rather than skipping quietly: a
    one-shot op is the whole batch, and returning its run_id would hand the caller
    a model someone else is mid-write.
    """
    ds = _make_dataset(tmp_path)
    trainer = _Counter()
    trainer.install(monkeypatch)
    params = {"data": str(_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    run_id = run_op(ds, "train-pose", dict(params))
    assert trainer.calls == 1

    # A peer holds the root: a live claim whose execution has no terminal run-log.
    write_inflight(
        model_run_root(ds, "train-pose", run_id),
        new_inflight(
            execution_id="SOMEONE-ELSE",
            host="otherhost",
            pid=4242,
            phase=None,
            idle_seconds=3600.0,
        ),
    )

    # overwrite, so the reuse gate does not answer first and the claim is reached.
    with pytest.raises(RunRootHeld, match="SOMEONE-ELSE"):
        _ = run_op(ds, "train-pose", {**params, "overwrite": True})

    assert trainer.calls == 1


# --- What the data fingerprint may and may not notice ----------------------
#
# The reuse gate above can only fire if an identical resubmission mints an
# identical identifier, so these pin the fingerprint the identifier is built on.
# They are filesystem tests on purpose: the golden corpus stubs the fingerprint
# with a literal, so nothing there would notice any of these regressing.


def test_an_unrelated_sibling_does_not_move_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file dropped beside the data.yaml is not part of the training data.

    The fingerprint used to walk the YAML's parent recursively, so anything
    landing there -- a notebook, a log, the run's own output -- re-addressed the
    next training run and made content-addressed reuse unreachable in that layout.
    """
    ds = _make_dataset(tmp_path)
    trainer = _Counter()
    trainer.install(monkeypatch)
    data_yaml = _yolo_dataset(tmp_path / "converted")
    params = {"data": str(data_yaml), "epochs": 2, "device": "cpu"}

    first = run_op(ds, "train-pose", dict(params))
    _ = (data_yaml.parent / "notes.txt").write_text("dropped beside the data.yaml")
    second = run_op(ds, "train-pose", dict(params))

    assert first == second
    assert trainer.calls == 1


def test_the_same_dataset_at_two_locations_fingerprints_alike(tmp_path: Path) -> None:
    """A copied or moved dataset is the same dataset.

    ``make_data_yaml`` writes ``path: os.path.abspath(dataset_root)``, so digesting
    the YAML's raw text made the location part of the model's identity -- two
    identical conversions at two paths were two different models, and an API box
    and a compute pod on different mounts could never agree.
    """
    a = _yolo_dataset(tmp_path / "somewhere")
    b = _yolo_dataset(tmp_path / "elsewhere-entirely")
    assert fingerprint_yolo_dataset(a) == fingerprint_yolo_dataset(b)


def test_a_changed_image_does_move_the_fingerprint(tmp_path: Path) -> None:
    """Location-independence must not cost content-sensitivity.

    This is what rules out the cheap fix of digesting the data.yaml alone: the
    YAML is identical here and the training data is not.
    """
    a = _yolo_dataset(tmp_path / "a", image=b"one")
    b = _yolo_dataset(tmp_path / "b", image=b"a different image entirely")
    assert fingerprint_yolo_dataset(a) != fingerprint_yolo_dataset(b)


def test_a_declaration_free_data_yaml_still_fingerprints(tmp_path: Path) -> None:
    """Identity computation is not the place to refuse an odd dataset."""
    directory = tmp_path / "odd"
    directory.mkdir()
    no_splits = directory / "data.yaml"
    _ = no_splits.write_text("kpt_shape: [4, 3]\n")
    missing = directory / "missing.yaml"
    _ = missing.write_text("train: nowhere/at/all\nval: [also, absent]\n")
    malformed = directory / "malformed.yaml"
    _ = malformed.write_text("{[unclosed")

    digests = {
        fingerprint_yolo_dataset(no_splits),
        fingerprint_yolo_dataset(missing),
        fingerprint_yolo_dataset(malformed),
        fingerprint_yolo_dataset(directory / "does-not-exist.yaml"),
    }
    assert len(digests) == 4  # each is a digest, and none collides with another


# --- train_overrides -------------------------------------------------------


def test_overrides_reach_the_trainer_and_move_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model trained on a different learning rate is a different model.

    So ``train_overrides`` reaches the trainer *and* reaches identity -- the pair
    that lets the deployed POLO hyperparameters run through the op instead of
    around it.
    """
    ds = _make_dataset(tmp_path)
    trainer = _Counter()
    trainer.install(monkeypatch)
    params = {"data": str(_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    plain = run_op(ds, "train-pose", dict(params))
    tuned = run_op(
        ds, "train-pose", {**params, "train_overrides": {"lr0": 0.0044, "lrf": 0.0072}}
    )

    assert plain != tuned
    assert trainer.calls == 2
    assert trainer.last_kwargs["lr0"] == 0.0044
    assert trainer.last_kwargs["lrf"] == 0.0072
