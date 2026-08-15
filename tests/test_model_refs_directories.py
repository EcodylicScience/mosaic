"""Model references that are not a single weights file.

A Lightning Pose model is a directory, a SLEAP top-down model is an ordered pair
of them, and TREx's visual-identification weights are an extensionless prefix for
a file that does have an extension. ``resolve_model`` understood exactly one of
those four shapes -- the file -- so the other three were either reimplemented
privately per tracker or, in the prefix case, simply broken.

These cover the shapes themselves. That no *existing* identifier moved while they
were added is a separate claim, pinned by value in ``test_model_digest.py`` and
by the frozen corpus in ``test_op_identity_golden.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.file_digest import file_digest
from mosaic.tracking.model_refs import (
    MODEL_KINDS,
    resolve_model,
    resolve_model_set,
    spec_for,
)

from tests.helpers import make_dataset


def _sleap_model(directory: Path, weights: bytes, head: str = "centroid") -> Path:
    directory.mkdir(parents=True)
    (directory / "best.ckpt").write_bytes(weights)
    (directory / "training_config.yaml").write_text(f"head_configs:\n  {head}: {{}}\n")
    return directory


def _litpose_model(
    directory: Path, weights: bytes = b"lp", model_type: str = "heatmap"
) -> Path:
    checkpoints = directory / "tb_logs" / "run" / "version_0" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (directory / "config.yaml").write_text(f"model:\n  model_type: {model_type}\n")
    (checkpoints / "best.ckpt").write_bytes(weights)
    return directory


# --- directory-shaped artifacts ---------------------------------------------


def test_a_directory_resolves_to_its_declared_files(tmp_path: Path) -> None:
    model = _litpose_model(tmp_path / "lp")
    resolved = resolve_model_set(None, [str(model)], "litpose")

    assert resolved.path == model, "a directory model is handed to the tool whole"
    assert [p.name for p in resolved.significant_files] == ["config.yaml", "best.ckpt"]
    assert resolved.model_type == "heatmap"


def test_a_directory_is_named_by_content_not_location(tmp_path: Path) -> None:
    here = _litpose_model(tmp_path / "here")
    there = _litpose_model(tmp_path / "there")
    other = _litpose_model(tmp_path / "other", weights=b"different")

    assert (
        resolve_model_set(None, [str(here)], "litpose").model_id
        == resolve_model_set(None, [str(there)], "litpose").model_id
    )
    assert (
        resolve_model_set(None, [str(here)], "litpose").model_id
        != resolve_model_set(None, [str(other)], "litpose").model_id
    )


def test_what_a_tool_writes_back_is_not_part_of_the_model(tmp_path: Path) -> None:
    """The rule the whole spec exists to enforce.

    Lightning Pose writes ``video_preds/`` into the directory it was loaded from.
    Under a whole-tree digest the model would stop matching its own cached output
    the moment inference ran, which is the failure mode "identity reads only
    declared roles" is there to make impossible.
    """
    model = _litpose_model(tmp_path / "lp")
    before = resolve_model_set(None, [str(model)], "litpose").model_id

    (model / "video_preds").mkdir()
    (model / "video_preds" / "clip.csv").write_text("scorer,bodypart,coord\n")
    (model / "predictions.csv").write_text("anything\n")
    (model / "tb_logs" / "run" / "version_0" / "events.out.tfevents.1").write_bytes(
        b"\x00\x01"
    )

    assert resolve_model_set(None, [str(model)], "litpose").model_id == before


def test_a_file_handed_to_a_directory_kind_is_refused(tmp_path: Path) -> None:
    weights = tmp_path / "best.ckpt"
    weights.write_bytes(b"w")
    with pytest.raises(NotADirectoryError):
        _ = resolve_model_set(None, [str(weights)], "litpose")


# --- ordered sets -----------------------------------------------------------


def test_an_ordered_pair_carries_both_artifacts(tmp_path: Path) -> None:
    centroid = _sleap_model(tmp_path / "centroid", b"centroid", "centroid")
    instance = _sleap_model(tmp_path / "instance", b"instance", "centered_instance")

    resolved = resolve_model_set(None, [str(centroid), str(instance)], "sleap")
    assert resolved.paths == [centroid, instance], "order is the caller's, preserved"
    assert len(resolved.significant_files) == 2
    assert resolved.model_type == "centroid", "provenance from the first that has one"


def test_the_order_of_an_ordered_pair_is_identity(tmp_path: Path) -> None:
    """Centroid-then-instance is a different model from instance-then-centroid."""
    centroid = _sleap_model(tmp_path / "centroid", b"centroid")
    instance = _sleap_model(tmp_path / "instance", b"instance")

    forward = resolve_model_set(None, [str(centroid), str(instance)], "sleap")
    reverse = resolve_model_set(None, [str(instance), str(centroid)], "sleap")
    assert forward.model_id != reverse.model_id


def test_an_empty_reference_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _ = resolve_model_set(None, [], "sleap")


def test_a_run_id_needs_a_dataset_to_resolve_against() -> None:
    """``ds=None`` is honest for an external model, so it must refuse a run_id."""
    with pytest.raises(FileNotFoundError):
        _ = resolve_model_set(None, ["sleap.0.1-abcdef0123"], "sleap")


# --- the prefix shape -------------------------------------------------------


def test_a_prefix_resolves_the_file_beside_it(tmp_path: Path) -> None:
    """TREx wants ``<root>/identity_model``; the file is ``identity_model.pth``.

    Before the shape was declared this raised: the stem fails ``Path.exists()``,
    so resolution fell through to the index branch and reported a missing
    ``models/train-identity/index.csv``.
    """
    run_root = tmp_path / "run"
    run_root.mkdir()
    weights = run_root / "identity_model.pth"
    weights.write_bytes(b"identity weights")

    resolved = resolve_model(
        make_dataset(tmp_path, roots=("models",)),
        str(run_root / "identity_model"),
        "train-identity",
    )

    assert resolved.path == run_root / "identity_model", "the stem, as TREx wants it"
    assert not resolved.path.exists(), "and it is a stem, not a file"
    assert resolved.significant_files == (weights,)
    assert resolved.digest == file_digest(weights)


def test_a_prefix_naming_nothing_still_raises(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    with pytest.raises(FileNotFoundError, match="names no file"):
        _ = resolve_model(
            make_dataset(tmp_path, roots=("models",)),
            str(run_root / "identity_model"),
            "train-identity",
        )


def test_only_a_prefix_kind_probes_for_siblings(tmp_path: Path) -> None:
    """A run_id-shaped reference must not glob whatever directory it resembles.

    The probe is guarded by the spec rather than by inspection precisely so that
    an unregistered kind cannot accidentally acquire it.
    """
    assert spec_for("train-pose").shape == "file"
    assert spec_for("not-a-registered-kind").shape == "file"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _ = resolve_model(
            make_dataset(tmp_path, roots=("models",)), "train-pose.0.1-abcdef0123", "x"
        )


# --- role resolution --------------------------------------------------------


def test_a_missing_config_is_reported_before_a_missing_checkpoint(
    tmp_path: Path,
) -> None:
    """Role order is error order, and Lightning Pose reports its config first."""
    no_config = tmp_path / "no_config"
    checkpoint = no_config / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"w")

    with pytest.raises(FileNotFoundError, match="config"):
        _ = resolve_model_set(None, [str(no_config)], "litpose")


def test_a_sleap_model_needs_no_config(tmp_path: Path) -> None:
    """SLEAP's config is provenance, so its absence is not an error."""
    model = tmp_path / "sleap"
    model.mkdir()
    (model / "best.ckpt").write_bytes(b"w")

    resolved = resolve_model_set(None, [str(model)], "sleap")
    assert resolved.model_type == "", "nothing to read, so nothing claimed"
    assert len(resolved.significant_files) == 1


def test_the_preferred_checkpoint_wins_over_a_later_sibling(tmp_path: Path) -> None:
    """Lightning Pose writes both ``best`` and ``last``; identity takes ``best``."""
    model = tmp_path / "lp"
    checkpoints = model / "tb_logs" / "run" / "version_0" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (model / "config.yaml").write_text("model: {}\n")
    (checkpoints / "best.ckpt").write_bytes(b"the good one")
    (checkpoints / "last.ckpt").write_bytes(b"the last one")

    resolved = resolve_model_set(None, [str(model)], "litpose")
    assert resolved.significant_files[-1].name == "best.ckpt"


def test_checkpoint_resolution_is_deterministic(tmp_path: Path) -> None:
    """Several candidates and no preference: the same one every time.

    Globs come back in filesystem order, which is not an order. Sorting is what
    makes a fixed directory resolve to fixed weights, and therefore to a fixed
    identifier.
    """
    model = tmp_path / "sleap"
    model.mkdir()
    for name in ("zeta.ckpt", "alpha.ckpt", "middle.ckpt"):
        (model / name).write_bytes(name.encode())

    chosen = {
        resolve_model_set(None, [str(model)], "sleap").significant_files[0].name
        for _ in range(5)
    }
    assert chosen == {"alpha.ckpt"}


# --- registered models ------------------------------------------------------


def test_a_training_kind_inherits_its_frameworks_shape() -> None:
    """``train-litpose`` writes exactly the directory ``litpose`` describes.

    The op kind and the spec kind are different things -- one says which
    ``models/<kind>/index.csv`` a row lives in, the other says what the artifact
    looks like -- so the relationship is derived rather than asked for twice.
    """
    assert spec_for("train-litpose") is spec_for("litpose")
    assert spec_for("train-sleap") is spec_for("sleap")


def test_the_training_prefix_rule_does_not_overreach() -> None:
    """No registered kind inherits a shape by accident of its name.

    ``train-<framework>`` resolving to ``<framework>``'s spec is a rule about
    names, and a rule about names collides. Today's training kinds strip to
    ``pose`` / ``points`` / ``localizer``, which no spec claims, so they land on
    the single-file default that Ultralytics and the localizer actually produce.

    Checked over the whole registry rather than those three, so a future kind
    whose stripped name happens to match a spec fails here -- at registration --
    instead of at its first run, where the symptom is a directory digested as a
    file.
    """
    from mosaic.core.pipeline.ops import OPS
    from mosaic.tracking import register_ops

    register_ops()
    for kind in sorted(OPS):
        if kind in MODEL_KINDS:
            continue  # declared outright, so the fallback never runs
        framework = kind.removeprefix("train-")
        assert framework not in MODEL_KINDS or framework == kind, (
            f"{kind} strips to {framework!r}, which declares a "
            f"{MODEL_KINDS[framework].shape} model it may not produce"
        )
        assert spec_for(kind).shape == "file", kind
        assert spec_for(kind).payload_prefix is None, kind


def test_an_unregistered_kind_is_a_single_weights_file(tmp_path: Path) -> None:
    """The default, and what every Ultralytics-backed training op produces."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"yolo weights")

    resolved = resolve_model(
        make_dataset(tmp_path, roots=("models",)), str(weights), "train-pose"
    )
    assert resolved.path == weights
    assert resolved.significant_files == (weights,)
    assert resolved.digest == file_digest(weights), "the plain file digest, unwrapped"
    assert "train-pose" not in MODEL_KINDS, "no spec needed for the common case"
