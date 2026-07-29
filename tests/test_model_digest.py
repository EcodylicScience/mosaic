"""Tests for model-weight fingerprinting (item 4.6).

A bare weights path is a mutable key: swap the file and every consumer reuses
output produced by different weights, reporting a cache hit. These pin the digest
that closes that, and the rule that a path never names a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.file_digest import MODEL_DIGEST_HEX, file_digest
from mosaic.tracking.model_refs import ResolvedModel, resolve_model


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


def _make_dataset(tmp_path: Path) -> Dataset:
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={"models": str(tmp_path / "models")},
    )
    ds.ensure_roots()
    ds.save()
    return ds


def test_a_bare_path_is_named_by_its_bytes_not_its_location(tmp_path: Path) -> None:
    """The defect item 4.6 closes, stated as a test.

    Two different weights files at one path used to mint one identifier, and
    unchanged weights at two paths used to mint two.
    """
    ds = _make_dataset(tmp_path)
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
    ds = _make_dataset(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")

    resolved = resolve_model(ds, str(weights), "train-pose")
    assert resolved.run_id == ""
    assert resolved.digest == file_digest(weights)
    assert resolved.model_id == resolved.digest


def test_a_registered_model_is_named_by_its_run(tmp_path: Path) -> None:
    """The run when there is one: readable, and stable across a copy or a move."""
    ds = _make_dataset(tmp_path)
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
    resolved = ResolvedModel(path=Path("/models/best.pt"), run_id="", digest="abc123")
    assert resolved.model_id == "abc123"
    assert "best.pt" not in resolved.model_id


def test_an_unresolvable_reference_still_raises(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not"):
        _ = resolve_model(ds, "train-pose.0.1-nope", "train-pose")
