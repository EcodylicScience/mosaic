"""Resolve a model reference (weights path or prior training run_id) to weights + lineage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.models import model_index_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["MODEL_IDENTITY_SCHEME", "ResolvedModel", "resolve_model"]

MODEL_IDENTITY_SCHEME: Final = "1"
"""The contract turning a model reference into an identity term.

Its own family rather than a bump of ``OP_IDENTITY_SCHEME``. That constant covers
``frames``, ``transcode``, ``convert``, ``train``, ``infer`` and ``trex`` as one
payload-shape contract, and this change touches only how the last three name a
*model*. Bumping the shared number would mark ``frames`` as re-minted -- and
``frames`` is frozen permanently, because mosaic-api embeds its identifier inside
``AnnotationFrame.image_path`` on version-controlled rows carrying keypoint
labour. A marker that lies is worse than none, inside a family as much as across
families.

Born at "1" with the behaviour below, so nothing has to be retrofitted: before
this, a model reference contributed either a run identity or the *path string*,
and neither was recorded as having been produced under any contract.
"""


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Weights, and what names them.

    Returned instead of a widened tuple. There are five unpack sites and the
    added field is a third string beside two that already look like identifiers;
    a positional swap between ``run_id`` and ``digest`` would type-check, run,
    and mint plausible-looking identities from the wrong value.

    ``run_id`` is the training run that produced these weights, or ``""`` for
    weights handed in as a bare path -- there is no run to name. ``digest`` is
    what makes that second case identifiable anyway.
    """

    path: Path
    run_id: str
    digest: str

    @property
    def model_id(self) -> str:
        """What identity calls this model.

        The training run when there is one: readable, stable across a copy or a
        move, and it already names a directory a human can go and look at. The
        weights digest otherwise. **Never the path** -- a path names a location,
        and two locations hold different weights as readily as the same ones,
        which is the whole defect item 4.6 exists to close.
        """
        return self.run_id or self.digest


def resolve_model(ds: "Dataset", ref: str, kind: str) -> ResolvedModel:
    """Resolve a model reference to its weights, lineage and content digest.

    *ref* is either a filesystem path to weights or a prior training ``run_id``
    in ``models/<kind>/index.csv``. This powers retrain-from-existing-model and
    the trained-model -> TREx ``detect_model`` handoff.

    A bare path is still accepted. Refusing it would break the documented
    ``detect_model=/path/to/best.pt`` workflow, and the digest is precisely the
    answer to "this reference carries no lineage", so refusal buys nothing that
    measuring does not.

    The digest is computed on both branches. For a registered model it never
    reaches identity -- ``model_id`` prefers the run -- but it is what the index
    row records and what a future integrity check would compare.
    """
    p = Path(ref)
    if p.exists():
        return ResolvedModel(path=p, run_id="", digest=file_digest(p))

    idx_path = model_index_path(ds, kind)
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Model reference '{ref}' is not a path and {idx_path} does not "
            f"exist; cannot resolve as a run_id."
        )
    df = pd.read_csv(idx_path)
    match = df[df["run_id"].astype(str) == ref]
    if match.empty:
        raise KeyError(f"No model run_id '{ref}' found in {idx_path}")
    best = str(match.iloc[0]["best_model_path"])
    resolved = ds.resolve_path(best)
    return ResolvedModel(path=resolved, run_id=ref, digest=file_digest(resolved))
