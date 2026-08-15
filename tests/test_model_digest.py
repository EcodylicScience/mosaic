"""Tests for model-weight fingerprinting (item 4.6).

A bare weights path is a mutable key: swap the file and every consumer reuses
output produced by different weights, reporting a cache hit. These pin the digest
that closes that, and the rule that a path never names a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline.file_digest import MODEL_DIGEST_HEX, file_digest
from mosaic.tracking.model_refs import (
    ModelArtifact,
    ResolvedModel,
    resolve_model,
    resolve_model_set,
)

from tests.helpers import make_dataset


def test_the_digest_is_stable_and_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"weights" * 1000)
    b.write_bytes(b"weights" * 1000)
    assert file_digest(a) == file_digest(b), "same bytes, same digest"
    assert len(file_digest(a)) == MODEL_DIGEST_HEX

    a.write_bytes(b"different weights entirely")
    assert file_digest(a) != file_digest(b)


def test_the_digest_spans_chunk_boundaries(tmp_path: Path) -> None:
    """Streamed in 1 MiB blocks, so a change past the first block must show."""
    path = tmp_path / "big.pt"
    payload = bytearray(b"\x00" * (3 << 20))
    path.write_bytes(payload)
    before = file_digest(path)

    payload[-1] = 1
    path.write_bytes(payload)
    assert file_digest(path) != before


def test_a_bare_path_is_named_by_its_bytes_not_its_location(tmp_path: Path) -> None:
    """The defect item 4.6 closes, stated as a test.

    Two different weights files at one path used to mint one identifier, and
    unchanged weights at two paths used to mint two.
    """
    ds = make_dataset(tmp_path, roots=("models",))
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"first weights")
    first = resolve_model(ds, str(weights), "train-pose")

    # Same path, different bytes -- must not keep the same identity.
    weights.write_bytes(b"second weights, quite different")
    second = resolve_model(ds, str(weights), "train-pose")
    assert first.model_id != second.model_id

    # Different path, same bytes -- must keep it.
    moved = tmp_path / "elsewhere" / "best.pt"
    moved.parent.mkdir()
    moved.write_bytes(b"second weights, quite different")
    assert resolve_model(ds, str(moved), "train-pose").model_id == second.model_id


def test_a_bare_path_carries_no_lineage_but_is_still_identifiable(
    tmp_path: Path,
) -> None:
    """``run_id`` is honestly empty; ``model_id`` falls back to the digest."""
    ds = make_dataset(tmp_path, roots=("models",))
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")

    resolved = resolve_model(ds, str(weights), "train-pose")
    assert resolved.run_id == ""
    assert resolved.digest == file_digest(weights)
    assert resolved.model_id == resolved.digest


def test_a_registered_model_is_named_by_its_run(tmp_path: Path) -> None:
    """The run when there is one: readable, and stable across a copy or a move."""
    ds = make_dataset(tmp_path, roots=("models",))
    weights = tmp_path / "models" / "train-pose" / "r1" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    index = ds.get_root("models") / "train-pose" / "index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"run_id": "train-pose.0.1-abc", "best_model_path": str(weights)}]
    ).to_csv(index, index=False)

    resolved = resolve_model(ds, "train-pose.0.1-abc", "train-pose")
    assert resolved.model_id == "train-pose.0.1-abc"
    # Measured anyway: it is what the index row records and what an integrity
    # check would compare, even though it never reaches identity here.
    assert resolved.digest == file_digest(weights)


def test_the_model_id_is_never_the_path() -> None:
    """The one rule the property exists to enforce."""
    weights = Path("/models/best.pt")
    resolved = ResolvedModel(
        artifacts=(
            ModelArtifact(
                root=weights,
                files=(("weights", weights),),
                exec_path=weights,
                digest="abc123",
            ),
        ),
        run_id="",
        digest="abc123",
    )
    assert resolved.model_id == "abc123"
    assert "best.pt" not in resolved.model_id


def test_an_unresolvable_reference_still_raises(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, roots=("models",))
    with pytest.raises(FileNotFoundError, match="does not"):
        _ = resolve_model(ds, "train-pose.0.1-nope", "train-pose")


# --- Directory-shaped models: the identity payload, pinned by value ----------
#
# SLEAP and Lightning Pose models are directories, and each spells its identity
# payload its own way -- ``{"sleap_weights": [...]}`` over an ordered list of
# checkpoint digests, ``{"litpose_config": ..., "litpose_weights": ...}`` over a
# pair. Those spellings reach ``hash_params``, so they name every tracks variant
# either tracker has ever written.
#
# Nothing pinned them. The tests in ``test_tracking_ops.py`` are relational --
# they assert two models differ, or that order matters -- and every one of them
# stays green through a rename of a payload key, which would silently re-mint
# identities on disk that no reconcile was asked for.
#
# So: fixed bytes in, exact identifier out. blake2b over fixed bytes is
# machine-independent, and ``hash_params`` sorts keys, so these are stable
# values and not a snapshot to be re-blessed. **A change here is a defect unless
# it is the point of the commit.** Consolidating these resolvers is expected to
# change how they are *spelled* (which function is called, from which module) and
# must not change a single digit below.


def _make_sleap_model(directory: Path, weights: bytes, head: str) -> Path:
    """A minimal SLEAP model directory: the checkpoint, and a config for provenance."""
    directory.mkdir(parents=True)
    (directory / "best.ckpt").write_bytes(weights)
    (directory / "training_config.yaml").write_text(f"head_configs:\n  {head}: {{}}\n")
    return directory


def _make_litpose_model(directory: Path, weights: bytes, model_type: str) -> Path:
    """A minimal Lightning Pose model directory: ``config.yaml`` plus a tb_logs checkpoint."""
    checkpoints = directory / "tb_logs" / "run" / "version_0" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (directory / "config.yaml").write_text(f"model:\n  model_type: {model_type}\n")
    (checkpoints / "best.ckpt").write_bytes(weights)
    return directory


def test_a_sleap_model_directory_mints_a_pinned_identifier(tmp_path: Path) -> None:
    centroid = _make_sleap_model(tmp_path / "centroid", b"centroid weights", "centroid")
    resolved = resolve_model_set(None, [str(centroid)], "sleap")
    assert resolved.model_id == "2bb8be883f"
    assert resolved.model_type == "centroid"


def test_a_sleap_top_down_pair_mints_a_pinned_identifier(tmp_path: Path) -> None:
    """Two directories, one identity -- and the order is part of it."""
    centroid = _make_sleap_model(tmp_path / "centroid", b"centroid weights", "centroid")
    instance = _make_sleap_model(
        tmp_path / "instance", b"instance weights", "centered_instance"
    )

    forward = resolve_model_set(None, [str(centroid), str(instance)], "sleap")
    assert forward.model_id == "9bfb94c526"
    # Reversed is a different model, not the same one described differently.
    reverse = resolve_model_set(None, [str(instance), str(centroid)], "sleap")
    assert reverse.model_id == "ba80135309"


def test_a_litpose_model_directory_mints_a_pinned_identifier(tmp_path: Path) -> None:
    model = _make_litpose_model(tmp_path / "lp", b"lp weights", "heatmap_mhcrnn")
    resolved = resolve_model_set(None, [str(model)], "litpose")
    assert resolved.model_id == "7ebb705dc6"
    assert resolved.model_type == "heatmap_mhcrnn"


def test_a_model_directory_is_named_by_its_declared_files_only(tmp_path: Path) -> None:
    """What a tool writes into a model directory afterwards is not the model.

    Lightning Pose writes ``video_preds/`` into the directory it was loaded from,
    so an identity that read the whole tree would change the moment inference
    ran -- the same model would stop matching its own cached output. The rule
    that prevents it is that identity reads only the declared roles.
    """
    model = _make_litpose_model(tmp_path / "lp", b"lp weights", "heatmap")
    before = resolve_model_set(None, [str(model)], "litpose").model_id

    predictions = model / "video_preds"
    predictions.mkdir()
    (predictions / "some_video.csv").write_text("scorer,bodypart,coord\n")
    (model / "predictions.csv").write_text("anything at all\n")

    assert resolve_model_set(None, [str(model)], "litpose").model_id == before
