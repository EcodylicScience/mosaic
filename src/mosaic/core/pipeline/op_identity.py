"""How a tracking or media op names one of its runs.

``Op.version`` existed as metadata and reached no identifier, so two producer
versions were indistinguishable on disk: re-run a tracker whose output semantics
changed and the new run landed in the old run's directory, behind an existence
check, reported as a reuse.

**The version is a visible segment, not a hash term** -- ``<kind>.<version>-<digest>``,
e.g. ``trex.0.1-a1b2c3d4e5``. Three reasons, in order of weight:

- It is the feature precedent verbatim. ``compute_run_id`` returns
  ``f"{feature.version}-{params_hash}"``, and one rule for what a run
  identifier looks like beats two.
- A hash term would make the version *invisible*, which defeats the point: the
  requirement is that two producers are distinguishable on disk and that a
  version switcher has a readable label. A digest delivers neither.
- Keeping the version out of the payload means no existing digest moves, so the
  golden corpus diff for the commit that introduced this is additions only --
  a very cheap review gate.

The separator between kind and version is ``.`` because kinds already contain
``-`` (``train-pose``, ``convert-points``, ``infer-localizer``), so ``-`` cannot
delimit them.

**Declared, never detected.** The segment is the op's own compatibility number,
bumped by hand when its output semantics change. Deriving it from the installed
tool's build or commit would invalidate every artifact on every upstream release
for bit-identical output -- TREx is updated continuously. What the installed tool
reports is recorded separately, on the index row, as provenance.

**``extract-frames`` is carved out and does not use this.** Its identifier is
frozen permanently: ``mosaic-api`` writes it to ``AnnotationFrame.run_id``, a
Dolt-tracked column, *and embeds it mid-string* in ``image_path`` on rows
carrying keypoint annotation labor, and discovers runs by reading the directory
name off disk. Moving it orphans every annotated frame path, recoverable only by
re-annotating. See ``frames_run_id`` and ``tests/test_op_identity_golden.py``.

**Transcode is also carved out**, for the opposite reason: its output is a media
*filename* with no directory level, so there is no visible slot for a version and
``transcode_recipe_hash`` folds ``TranscodeOp.version`` into the digest instead.
That is a consequence of the artifact's shape, not a competing preference.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from ._utils import hash_params

__all__ = [
    "OP_IDENTITY_SCHEME",
    "OpRunId",
    "op_run_id",
    "parse_op_run_id",
]

OP_IDENTITY_SCHEME: Final = "1"
"""The contract the op minters implement today.

Named per family, like ``FEATURE_IDENTITY_SCHEME``, because mosaic has several
independent identity functions and one number cannot honestly cover them: a bump
for an op change would falsely mark every feature run as re-minted. Scheme 1 is
"``<kind>.<version>-<10 hex sha1 of the payload>``". Op runs are born marked, so
nothing has to be retrofitted.
"""

_RUN_ID = re.compile(
    r"^(?P<kind>[a-z0-9-]+)\.(?P<version>[0-9]+(?:\.[0-9]+)*)-(?P<digest>[0-9a-f]{10})$"
)


@dataclass(frozen=True, slots=True)
class OpRunId:
    """The three parts of an op run identifier."""

    kind: str
    version: str
    digest: str


def op_run_id(kind: str, version: str, payload: Mapping[str, object]) -> str:
    """Mint the run identifier for one op run.

    Args:
        kind: The op's registered kind, e.g. ``"train-pose"``.
        version: The op's *declared* version. A visible segment, deliberately
            absent from *payload* -- see the module docstring.
        payload: Everything that determines the output. Sort any collection in
            it before it gets here: the suite runs under ``PYTHONHASHSEED=random``
            and an unordered term yields a different name per process.
    """
    return f"{kind}.{version}-{hash_params(payload)}"


def parse_op_run_id(run_id: str) -> OpRunId | None:
    """Split *run_id* into kind, version and digest, or None if it is not one.

    None covers both a pre-version identifier (``trex-<digest>``, still on disk
    under migration M1) and a value that is not an identifier at all -- a bare
    weights path handed to ``resolve_model``, for instance. Callers that need a
    kind should fall back to their own default rather than guess, because a
    wrong kind resolves to a path that never existed.
    """
    match = _RUN_ID.match(run_id)
    if match is None:
        return None
    return OpRunId(
        kind=match.group("kind"),
        version=match.group("version"),
        digest=match.group("digest"),
    )
