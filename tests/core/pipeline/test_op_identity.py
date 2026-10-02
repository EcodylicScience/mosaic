"""Item 1.2: an op run identifier names its producer's declared version.

``Op.version`` was metadata that reached no identifier, so two producer versions
were indistinguishable on disk: re-run a tracker whose output semantics changed
and the new run landed in the old run's directory, behind an existence check,
reported as a reuse.

The version is a **visible segment**, mirroring ``compute_run_id``'s
``f"{feature.version}-{params_hash}"``, not a hash term -- see
``core/pipeline/op_identity.py`` for why. The consequence tested here is that no
existing digest moved: ``tests/data/op_identity_golden.json`` pins the payloads,
and this file pins the assembled shape around them.

``extract-frames`` is carved out and frozen. That carve-out is tested rather than
commented, because a comment does not survive a refactor by someone who has not
read it.
"""

from __future__ import annotations

import re

import pytest

from pathlib import Path

from mosaic.core.pipeline.identity_scheme import (
    FEATURE_IDENTITY_SCHEME,
    read_identity_scheme,
    write_identity_scheme,
)
from mosaic.core.pipeline.op_identity import (
    OP_IDENTITY_SCHEME,
    OpRunId,
    op_run_id,
    parse_op_run_id,
)
from mosaic.core.pipeline.ops import OPS
from mosaic.tracking.frame_extraction.dataset_runs import (
    ExtractFramesOp,
    ExtractFramesParams,
    frames_run_id,
)
from mosaic.tracking.ops.trex import TrexOp
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION

VERSIONED = re.compile(r"^[a-z0-9-]+\.[0-9]+(?:\.[0-9]+)*-[0-9a-f]{10}$")
FROZEN_FRAMES = re.compile(r"^[a-z]+-[0-9a-f]{10}$")


# --- The format ---------------------------------------------------------------


def test_the_identifier_carries_kind_version_and_digest() -> None:
    minted = op_run_id("train-pose", "0.1", {"a": 1})

    assert VERSIONED.match(minted), minted
    assert minted.startswith("train-pose.0.1-")


def test_the_version_is_a_segment_not_a_hash_term() -> None:
    """A version bump must move the directory without re-deriving the digest.

    This is the whole design: the segment makes two producer versions
    distinguishable *and readable* on disk, and keeping it out of the payload is
    what let this change land without moving a single existing digest.
    """
    old = op_run_id("trex", "0.1", {"a": 1})
    new = op_run_id("trex", "0.2", {"a": 1})

    assert old != new
    assert old.rsplit("-", 1)[1] == new.rsplit("-", 1)[1]


def test_a_changed_payload_moves_the_digest() -> None:
    assert op_run_id("trex", "0.1", {"a": 1}) != op_run_id("trex", "0.1", {"a": 2})


def test_the_kind_separator_survives_a_hyphenated_kind() -> None:
    """Every registered kind contains ``-``, so ``-`` cannot delimit the version."""
    minted = op_run_id("infer-localizer", "0.1", {})

    assert parse_op_run_id(minted) == OpRunId(
        kind="infer-localizer", version="0.1", digest=minted.rsplit("-", 1)[1]
    )


@pytest.mark.parametrize(
    "value",
    [
        "trex-a1b2c3d4e5",  # pre-version, still on disk under migration M1
        "/abs/path/to/best.pt",
        "polo26n.yaml",
        "trex.0.1-NOTHEX0000",
        "",
    ],
)
def test_a_non_identifier_parses_to_none(value: str) -> None:
    """None rather than a guess: a wrong kind resolves to a path that never existed.

    That is exactly what the old ``ref.rsplit("-", 1)[0]`` did once identifiers
    grew a version segment -- it read ``train-points.0.1-<digest>`` as the kind
    ``train-points.0.1``.
    """
    assert parse_op_run_id(value) is None


def test_round_trip() -> None:
    minted = op_run_id("train-points", "0.1", {"params": {}, "data": "d", "base": ""})
    parsed = parse_op_run_id(minted)

    assert parsed is not None
    assert f"{parsed.kind}.{parsed.version}-{parsed.digest}" == minted


# --- Every registered op, and the one carve-out -------------------------------


def test_every_op_declares_a_version() -> None:
    missing = [kind for kind, cls in OPS.items() if not getattr(cls, "version", "")]

    assert not missing, f"ops with no declared version: {sorted(missing)}"


def test_the_trex_op_and_the_standalone_runner_share_one_version() -> None:
    """Two entry points, one integration: they must not name a run two ways."""
    assert TrexOp.version == TREX_VERSION
    assert TrexOp.kind == TREX_KIND


# --- extract-frames: frozen, permanently --------------------------------------


def test_frames_identity_has_no_version_segment() -> None:
    """``<method>-<digest>``, the pre-1.2 shape, preserved deliberately."""
    minted = frames_run_id("uniform", ExtractFramesParams(n_frames=20))

    assert FROZEN_FRAMES.match(minted), minted
    assert not VERSIONED.match(minted)
    assert minted.startswith("uniform-")


def test_frames_identity_ignores_the_op_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The carve-out, made executable.

    ``mosaic-api`` writes this identifier to ``AnnotationFrame.run_id`` -- a
    Dolt-tracked column -- *and embeds it mid-string* in ``image_path`` on rows
    carrying keypoint annotation labor, and finds runs by reading the directory
    name off disk. Moving it orphans every annotated frame path, recoverable
    only by re-annotating. A future refactor that "harmonizes" frames onto
    ``op_run_id`` fails here rather than in a dataset.
    """
    params = ExtractFramesParams(n_frames=20)
    before = frames_run_id("uniform", params)

    monkeypatch.setattr(ExtractFramesOp, "version", "9.9")

    assert frames_run_id("uniform", params) == before


def test_op_runs_are_born_under_a_named_scheme() -> None:
    """Per family, not one global number.

    mosaic has several independent identity functions; bumping one constant for
    an op change would falsely mark every feature run as re-minted, and a marker
    that lies is worse than none. Pinned so a change to the op payload shape has
    to move this line too.
    """
    assert OP_IDENTITY_SCHEME == "1"


def test_an_op_run_root_records_the_scheme_that_minted_it(tmp_path: Path) -> None:
    """Item 0.4's owed reach, and it must land before item 4.6 moves a digest.

    A scheme marker cannot be retrofitted onto identifiers already on disk --
    doing so requires knowing which contract produced each of them, which is
    exactly the provenance that does not exist. So an op run root records its
    family's scheme *before* the train and infer digests move, not after.
    """
    run_root = tmp_path / "train-pose.0.1-abcdef0123"
    run_root.mkdir()
    write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

    assert read_identity_scheme(run_root) == OP_IDENTITY_SCHEME


def test_the_op_and_feature_families_record_different_markers(
    tmp_path: Path,
) -> None:
    """One marker names one family. Sharing a number would make a bump lie.

    A feature-scheme bump that also marked every model and tracker run as
    re-minted would be worse than no marker at all, which is the whole reason
    these constants are per family.
    """
    op_root = tmp_path / "op"
    feature_root = tmp_path / "feature"
    op_root.mkdir()
    feature_root.mkdir()

    write_identity_scheme(op_root, OP_IDENTITY_SCHEME)
    write_identity_scheme(feature_root, FEATURE_IDENTITY_SCHEME)

    assert read_identity_scheme(op_root) == OP_IDENTITY_SCHEME
    assert read_identity_scheme(feature_root) == FEATURE_IDENTITY_SCHEME
    assert OP_IDENTITY_SCHEME != FEATURE_IDENTITY_SCHEME
