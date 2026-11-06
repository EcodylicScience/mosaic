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
from mosaic.core.pipeline._utils import hash_params
from mosaic.tracking.ops._common import (
    RunRootHeld,
    fingerprint_dataset,
    fingerprint_yolo_dataset,
    remove_label_caches,
)
from mosaic.tracking.external.runner.ultralytics_protocol import TrainRequestBase
from mosaic.tracking.ops.trained_models import trained_model_index
from tests.helpers import FakeTrainer, make_dataset, write_data_yaml


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(content)
    return path


_TRAIN_LABEL = b"0 0.5 0.5 0.2 0.2 0.40 0.50 2 0.60 0.50 2\n"


def _yolo_dataset(
    root: Path, *, image: bytes = b"train-image", label: bytes = _TRAIN_LABEL
) -> Path:
    """A minimal YOLO dataset: a data.yaml naming two split directories.

    Each image has its label file under the sibling ``labels`` directory.
    ``path`` is written absolute because that is what ``make_data_yaml`` writes
    (``os.path.abspath(dataset_root)``), which is exactly the spelling the
    fingerprint must not depend on.
    """
    _ = _write(root / "train" / "images" / "a.png", image)
    _ = _write(root / "valid" / "images" / "b.png", b"val-image")
    _ = _write(root / "train" / "labels" / "a.txt", label)
    _ = _write(root / "valid" / "labels" / "b.txt", b"0 0.5 0.5 0.1 0.1\n")
    data_yaml = root / "data.yaml"
    _ = data_yaml.write_text(
        f"path: {root.resolve()}\ntrain: train/images\nval: valid/images\n"
        "nc: 1\nnames: [bee]\n"
    )
    return data_yaml


def test_an_identical_resubmission_does_not_retrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(write_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    first = run_op(ds, "train-pose", dict(params))
    second = run_op(ds, "train-pose", dict(params))

    # Same work, so the same content-addressed identifier...
    assert first == second
    # ...and the trainer ran once, not twice.
    assert trainer.calls == 1
    # The row is still there exactly once (the index dedups on run_id).
    rows = trained_model_index(model_index_path(ds, "train-pose")).read(run_id=first)
    assert len(rows) == 1


def test_a_reused_training_run_records_the_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An op that reused a trained model says so on its own run-log.

    Until now the reuse was a ``print`` and an early ``return run_id``, so it
    reached nothing: ``run_op`` returns a bare identifier, and widening that
    signature is not available because mosaic-queue and mosaic-api call it too.
    Without this a resubmitted training step is indistinguishable in the ledger
    from one that spent the GPU time again.
    """
    from mosaic.runlog import read_run, run_log_dir

    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(write_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    _ = run_op(ds, "train-pose", dict(params), execution_id="EXECTRAINFIRST")
    _ = run_op(ds, "train-pose", dict(params), execution_id="EXECTRAINSECOND")
    assert trainer.calls == 1

    first = read_run(run_log_dir(ds.base_dir), "EXECTRAINFIRST")
    second = read_run(run_log_dir(ds.base_dir), "EXECTRAINSECOND")
    assert first is not None and second is not None
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_a_label_edit_between_two_runs_trains_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrected label at the same byte length is new data, so a new model."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(data_yaml), "epochs": 1, "device": "cpu"}

    first = run_op(ds, "train-pose", dict(params))

    label = data_yaml.parent / "train" / "labels" / "a.txt"
    moved = _TRAIN_LABEL.replace(b"0.40 0.50", b"0.45 0.55")
    assert len(moved) == len(_TRAIN_LABEL)
    _ = label.write_bytes(moved)
    second = run_op(ds, "train-pose", dict(params))

    assert trainer.calls == 2
    assert second != first


def test_overwrite_retrains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch: a caller who means it can force the work again."""
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(write_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    first = run_op(ds, "train-pose", dict(params))
    second = run_op(ds, "train-pose", dict(params), overwrite=True)

    # `overwrite` must not move the identifier. It describes an attempt rather
    # than the model, and it is passed to the op beside its params.
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
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(write_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

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
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(write_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

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
    with pytest.raises(RunRootHeld, match="SOMEONE-ELSE") as refused:
        _ = run_op(ds, "train-pose", dict(params), overwrite=True)

    assert trainer.calls == 1
    assert refused.value.held is not None
    assert refused.value.held.execution_id == "SOMEONE-ELSE"


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
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
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


def test_a_moved_keypoint_moves_the_fingerprint(tmp_path: Path) -> None:
    """A corrected label is new training data, even at the same byte length."""
    before = fingerprint_yolo_dataset(_yolo_dataset(tmp_path / "a"))
    moved = _TRAIN_LABEL.replace(b"0.40 0.50", b"0.45 0.55")
    assert len(moved) == len(_TRAIN_LABEL)
    after = fingerprint_yolo_dataset(_yolo_dataset(tmp_path / "b", label=moved))
    assert after != before


def test_an_added_instance_moves_the_fingerprint(tmp_path: Path) -> None:
    before = fingerprint_yolo_dataset(_yolo_dataset(tmp_path / "a"))
    added = _TRAIN_LABEL + b"0 0.2 0.2 0.1 0.1 0.15 0.20 2 0.25 0.20 2\n"
    after = fingerprint_yolo_dataset(_yolo_dataset(tmp_path / "b", label=added))
    assert after != before


def test_caches_beside_the_data_leave_the_fingerprint(tmp_path: Path) -> None:
    """What Ultralytics writes while training is not training data."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    before = fingerprint_yolo_dataset(data_yaml)
    _ = _write(data_yaml.parent / "train" / "labels.cache", b"cached labels")
    _ = _write(data_yaml.parent / "train" / "images" / "a.npy", b"cached pixels")
    assert fingerprint_yolo_dataset(data_yaml) == before


@pytest.mark.parametrize("kind", ["train-pose", "train-points"])
def test_empty_label_caches_are_gone_when_the_trainer_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A 0-byte cache left by an earlier crash must not reach Ultralytics."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    caches = [
        _write(data_yaml.parent / "train" / "labels.cache", b""),
        _write(data_yaml.parent / "valid" / "labels.cache", b""),
    ]
    present: list[bool] = []

    def witness(request: TrainRequestBase) -> None:
        del request
        present.extend(cache.exists() for cache in caches)

    trainer = FakeTrainer(on_call=witness)
    trainer.install(monkeypatch)

    _ = run_op(
        make_dataset(tmp_path),
        kind,
        {"data": str(data_yaml), "epochs": 1, "device": "cpu"},
    )

    assert trainer.calls == 1
    assert present == [False, False]


def test_removing_the_label_caches_names_the_file_it_cannot_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_yaml = _yolo_dataset(tmp_path / "data")
    cache = _write(data_yaml.parent / "train" / "labels.cache", b"")
    real_unlink = Path.unlink

    def refuse(self: Path, missing_ok: bool = False) -> None:
        if self == cache:
            raise PermissionError("read-only directory")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", refuse)

    with pytest.raises(RuntimeError, match=r"train[/\\]labels\.cache"):
        remove_label_caches(data_yaml)


def test_only_the_label_caches_of_the_splits_are_removed(tmp_path: Path) -> None:
    """A split's cache is its label directory's sibling; nothing else goes."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    root = data_yaml.parent
    gone = [
        _write(root / "train" / "labels.cache", b"x"),
        _write(root / "valid" / "labels.cache", b"x"),
    ]
    kept = [
        _write(root / "train" / "images" / "a.npy", b"x"),
        _write(root / "train" / "images.cache", b"x"),
        _write(root / "elsewhere" / "labels.cache", b"x"),
        root / "train" / "labels" / "a.txt",
    ]

    remove_label_caches(data_yaml)
    remove_label_caches(data_yaml)

    assert not any(cache.exists() for cache in gone)
    assert all(path.exists() for path in kept)


def test_a_data_yaml_that_does_not_parse_removes_nothing(tmp_path: Path) -> None:
    data_yaml = _yolo_dataset(tmp_path / "data")
    cache = _write(data_yaml.parent / "train" / "labels.cache", b"")
    _ = data_yaml.write_text("train: [train/images\nval: valid/images\n")

    remove_label_caches(data_yaml)

    assert cache.exists()


def test_a_list_split_has_the_label_caches_of_its_images_removed(
    tmp_path: Path,
) -> None:
    data_yaml = _yolo_dataset(tmp_path / "data")
    root = data_yaml.parent
    _ = (root / "train.txt").write_text("./train/images/a.png\n")
    _ = data_yaml.write_text("train: train.txt\nval: valid/images\n")
    listed = _write(root / "train" / "labels.cache", b"")
    valid = _write(root / "valid" / "labels.cache", b"")

    remove_label_caches(data_yaml)

    assert not listed.exists()
    assert not valid.exists()


def test_a_polo_tree_is_portable_and_content_sensitive(tmp_path: Path) -> None:
    """A tree ``make_polo_data_yaml`` writes, with its absolute ``path`` and int keys.

    The integer keys of ``names`` and ``radii`` must parse, so the digest is
    taken from the declared content rather than from the YAML's text, which
    carries the location.
    """
    from mosaic.tracking.pose_training.prep import make_polo_data_yaml

    def tree(
        root: Path, *, image: bytes = b"img", label: bytes = b"0 0.5 0.5\n"
    ) -> Path:
        _ = _write(root / "train" / "images" / "a.png", image)
        _ = _write(root / "train" / "labels" / "a.txt", label)
        _ = _write(root / "valid" / "images" / "b.png", b"val")
        _ = _write(root / "valid" / "labels" / "b.txt", b"0 0.4 0.4\n")
        return Path(make_polo_data_yaml(root, ["bee"], {0: 6.0}))

    here = fingerprint_yolo_dataset(tree(tmp_path / "here"))
    assert fingerprint_yolo_dataset(tree(tmp_path / "there")) == here
    assert fingerprint_yolo_dataset(tree(tmp_path / "image", image=b"other")) != here
    assert (
        fingerprint_yolo_dataset(tree(tmp_path / "label", label=b"0 0.6 0.5\n")) != here
    )


def test_integer_and_text_keys_fingerprint_alike(tmp_path: Path) -> None:
    """Ultralytics reads ``{0: bee}`` and ``{'0': bee}`` as one class map."""
    a = _yolo_dataset(tmp_path / "a")
    b = _yolo_dataset(tmp_path / "b")
    _ = a.write_text("train: train/images\nval: valid/images\nnames: {0: bee}\n")
    _ = b.write_text("train: train/images\nval: valid/images\nnames: {'0': bee}\n")
    assert fingerprint_yolo_dataset(a) == fingerprint_yolo_dataset(b)


def test_a_missing_label_fingerprints_and_adding_it_moves(tmp_path: Path) -> None:
    """An image without a label is a background image, not an error."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    label = data_yaml.parent / "train" / "labels" / "a.txt"
    label.unlink()
    without = fingerprint_yolo_dataset(data_yaml)
    assert without
    _ = _write(label, _TRAIN_LABEL)
    assert fingerprint_yolo_dataset(data_yaml) != without


def test_a_linked_image_reads_the_label_beside_the_link(tmp_path: Path) -> None:
    """Ultralytics does not resolve the files under a split directory."""
    target = _write(tmp_path / "store" / "images" / "a.png", b"pixels")
    _ = _write(tmp_path / "store" / "labels" / "a.txt", b"0 0.1 0.1\n")
    data_yaml = _yolo_dataset(tmp_path / "data")
    image = data_yaml.parent / "train" / "images" / "a.png"
    image.unlink()
    image.symlink_to(target)
    before = fingerprint_yolo_dataset(data_yaml)
    _ = _write(tmp_path / "store" / "labels" / "a.txt", b"0 0.9 0.9\n")
    assert fingerprint_yolo_dataset(data_yaml) == before, "the target's label"
    _ = _write(data_yaml.parent / "train" / "labels" / "a.txt", b"0 0.9 0.9\n")
    assert fingerprint_yolo_dataset(data_yaml) != before, "the link's label"


def test_an_uppercase_suffix_is_an_image(tmp_path: Path) -> None:
    data_yaml = _yolo_dataset(tmp_path / "data")
    before = fingerprint_yolo_dataset(data_yaml)
    _ = _write(data_yaml.parent / "train" / "labels" / "c.txt", b"0 0.3 0.3\n")
    _ = _write(data_yaml.parent / "train" / "images" / "c.JPG", b"upper")
    with_image = fingerprint_yolo_dataset(data_yaml)
    assert with_image != before
    _ = _write(data_yaml.parent / "train" / "labels" / "c.txt", b"0 0.4 0.4\n")
    assert fingerprint_yolo_dataset(data_yaml) != with_image, "its label is read"


@pytest.mark.parametrize("suffix", [".avif", ".heif", ".jp2"])
def test_a_label_edit_beside_a_polo_only_image_moves_the_fingerprint(
    tmp_path: Path, suffix: str
) -> None:
    """The POLO fork's Ultralytics accepts suffixes that 8.4.121 does not."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    _ = _write(data_yaml.parent / "train" / "images" / f"c{suffix}", b"pixels")
    _ = _write(data_yaml.parent / "train" / "labels" / "c.txt", b"0 0.3 0.3\n")
    before = fingerprint_yolo_dataset(data_yaml)
    _ = _write(data_yaml.parent / "train" / "labels" / "c.txt", b"0 0.4 0.4\n")
    assert fingerprint_yolo_dataset(data_yaml) != before


def test_a_parent_relative_split_naming_nothing_drops_its_prefix(
    tmp_path: Path,
) -> None:
    """``../train/images`` that names nothing is read as ``train/images``.

    Ultralytics retries such a split without its ``../`` prefix, so the labels
    under the dataset's own ``train/labels`` are the ones it trains on.
    """
    data_yaml = _yolo_dataset(tmp_path / "data")
    assert not (data_yaml.parent.parent / "train").exists()
    _ = data_yaml.write_text(
        "train: ../train/images\nval: valid/images\nnames: [bee]\n"
    )
    before = fingerprint_yolo_dataset(data_yaml)
    _ = _write(data_yaml.parent / "train" / "labels" / "a.txt", b"0 0.9 0.9\n")
    assert fingerprint_yolo_dataset(data_yaml) != before


def test_a_list_split_derives_its_labels(tmp_path: Path) -> None:
    """A ``.txt`` split names its images, and each image's label is read too."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    root = data_yaml.parent
    _ = (root / "train.txt").write_text("./train/images/a.png\n")
    _ = data_yaml.write_text("train: train.txt\nval: valid/images\nnames: [bee]\n")
    before = fingerprint_yolo_dataset(data_yaml)

    _ = _write(root / "train" / "labels" / "a.txt", b"0 0.9 0.9 0.1 0.1\n")
    edited = fingerprint_yolo_dataset(data_yaml)
    assert edited != before, "a listed image's label"
    _ = _write(root / "train" / "images" / "a.png", b"a longer train image")
    assert fingerprint_yolo_dataset(data_yaml) != edited, "a listed image's size"

    elsewhere = _yolo_dataset(tmp_path / "elsewhere")
    _ = (elsewhere.parent / "train.txt").write_text("./train/images/a.png\n")
    _ = elsewhere.write_text("train: train.txt\nval: valid/images\nnames: [bee]\n")
    assert fingerprint_yolo_dataset(elsewhere) == before, "portable"


def test_a_list_line_replaces_only_its_leading_dot_slash(tmp_path: Path) -> None:
    """``../`` inside a listed path is not another ``./`` to replace."""
    data_yaml = _yolo_dataset(tmp_path / "data")
    root = data_yaml.parent
    _ = (root / "train.txt").write_text("./train/images/../images/a.png\n")
    _ = data_yaml.write_text("train: train.txt\nval: valid/images\nnames: [bee]\n")
    before = fingerprint_yolo_dataset(data_yaml)

    _ = _write(root / "train" / "labels" / "a.txt", b"0 0.9 0.9 0.1 0.1\n")

    assert fingerprint_yolo_dataset(data_yaml) != before


def test_a_dataset_under_glob_characters_fingerprints_its_labels(
    tmp_path: Path,
) -> None:
    """``[``, ``]``, ``*`` and ``?`` in the dataset path are literal."""
    data_yaml = _yolo_dataset(tmp_path / "mice [2024] *?" / "data")
    before = fingerprint_yolo_dataset(data_yaml)

    _ = _write(data_yaml.parent / "train" / "labels" / "a.txt", b"0 0.9 0.9 0.1 0.1\n")

    assert fingerprint_yolo_dataset(data_yaml) != before


def test_an_empty_label_cache_under_glob_characters_is_removed(
    tmp_path: Path,
) -> None:
    data_yaml = _yolo_dataset(tmp_path / "mice [2024] *?" / "data")
    cache = _write(data_yaml.parent / "train" / "labels.cache", b"")

    remove_label_caches(data_yaml)

    assert not cache.exists()


def test_a_data_file_is_its_own_bytes_and_nothing_beside_it(tmp_path: Path) -> None:
    """A ``.slp`` or a CVAT XML is fingerprinted by its bytes alone.

    ``fingerprint_dataset`` used to list every file under a file's parent,
    recursively, so labels at a dataset root moved with each run's own output.
    Each of the three writes below moved it then.
    """
    labels = _write(tmp_path / "workspace" / "session.slp", b"fake slp bytes")
    before = fingerprint_dataset(labels)

    _ = _write(tmp_path / "workspace" / "models" / "index.csv", b"run_id,status\n")
    assert fingerprint_dataset(labels) == before, "a new subfolder beside it"
    _ = _write(tmp_path / "workspace" / "run.log", b"one line\n")
    assert fingerprint_dataset(labels) == before, "a new sibling"
    _ = _write(tmp_path / "workspace" / "run.log", b"one line\nand another\n")
    assert fingerprint_dataset(labels) == before, "a sibling that grew"

    copy = _write(tmp_path / "elsewhere" / "renamed.slp", b"fake slp bytes")
    assert fingerprint_dataset(copy) == before, "same bytes, another place and name"

    _ = _write(labels, b"fake slp bytes, relabelled")
    assert fingerprint_dataset(labels) != before, "changed labels are a new model"


def test_absent_data_still_fingerprints(tmp_path: Path) -> None:
    """Planning for execution must reach the tool that says which file is missing.

    ``run`` plans with ``require_data=False``, so a raise here would replace
    train-sleap's "No SLEAP labels exist" with a bare ``open()`` error.
    """
    assert fingerprint_dataset(tmp_path / "nope.slp")


def test_a_data_directory_digests_as_it_always_has(tmp_path: Path) -> None:
    """The directory form is unchanged, so no localizer or litpose run moves.

    Spelled out against ``hash_params`` rather than a literal, because the
    golden corpus stubs every fingerprint and nothing else would notice.
    """
    directory = tmp_path / "project"
    _ = _write(directory / "labeled-data" / "a.png", b"12345")
    _ = _write(directory / "CollectedData.csv", b"xy")

    assert fingerprint_dataset(directory) == hash_params(
        {"listing": ["CollectedData.csv:2", "labeled-data/a.png:5"]}
    )


# --- train_overrides -------------------------------------------------------


def test_overrides_reach_the_trainer_and_move_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model trained on a different learning rate is a different model.

    So ``train_overrides`` reaches the trainer *and* reaches identity -- the pair
    that lets the deployed POLO hyperparameters run through the op instead of
    around it.
    """
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(write_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    plain = run_op(ds, "train-pose", dict(params))
    tuned = run_op(
        ds, "train-pose", {**params, "train_overrides": {"lr0": 0.0044, "lrf": 0.0072}}
    )

    assert plain != tuned
    assert trainer.calls == 2
    assert trainer.last_request.train_overrides["lr0"] == 0.0044
    assert trainer.last_request.train_overrides["lrf"] == 0.0072


# --- the reuse gate reads the argument, on every op that has one ------------
#
# One gate shape, repeated per op: `if not overwrite and training_is_complete(...)`.
# The value used to be a params field on all six. Each case below asserts both
# directions, because an op that ignored the argument entirely would reuse
# forever and an op whose gate lost its `training_is_complete` half would
# retrain forever, and one direction cannot tell those apart.


def test_train_points_reuses_a_finished_run_unless_overwrite_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.install(monkeypatch)
    params = {"data": str(write_data_yaml(tmp_path)), "epochs": 2, "device": "cpu"}

    first = run_op(ds, "train-points", dict(params))
    assert trainer.calls == 1

    assert run_op(ds, "train-points", dict(params)) == first
    assert trainer.calls == 1, "a finished run must not train again"

    assert run_op(ds, "train-points", dict(params), overwrite=True) == first
    assert trainer.calls == 2, "overwrite must reach the gate"


def _fake_localizer(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Stand in for the PyTorch localizer, recording each run root it wrote."""
    import mosaic.tracking.pose_training.localizer_train as localizer_train

    trained: list[Path] = []

    def train(
        dataset_dir: str | Path,
        *,
        project: str | Path,
        name: str,
        epochs: int = 1,
        **_kwargs: object,
    ) -> localizer_train.TrainingResult:
        run_dir = Path(project) / name
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        weights = run_dir / "weights" / "best.pt"
        _ = weights.write_bytes(b"localizer weights")
        _ = (run_dir / "results.csv").write_text("epoch,loss\n0,0.1\n")
        trained.append(run_dir)
        return localizer_train.TrainingResult(
            best_model_path=weights,
            last_model_path=weights,
            run_dir=run_dir,
            best_epoch=0,
            best_val_loss=0.1,
        )

    monkeypatch.setattr(localizer_train, "train_localizer", train)
    return trained


def test_train_localizer_reuses_a_finished_run_unless_overwrite_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path)
    trained = _fake_localizer(monkeypatch)
    dataset_dir = tmp_path / "patches"
    (dataset_dir / "train").mkdir(parents=True)
    _ = (dataset_dir / "train" / "patches.npy").write_bytes(b"patches")
    params = {"dataset_dir": str(dataset_dir), "epochs": 1, "device": "cpu"}

    first = run_op(ds, "train-localizer", dict(params))
    assert len(trained) == 1

    assert run_op(ds, "train-localizer", dict(params)) == first
    assert len(trained) == 1, "a finished run must not train again"

    assert run_op(ds, "train-localizer", dict(params), overwrite=True) == first
    assert len(trained) == 2, "overwrite must reach the gate"
