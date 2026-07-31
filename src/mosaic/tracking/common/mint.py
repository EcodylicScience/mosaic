"""Naming a tracker run, and recording what it is.

Every integrated tracker opens the same way: ensure its root exists, mint the run
identifier and the tracks variant from one scope-free settings dict, stamp the
identity scheme, and write the settings beside the run so it is explicable from
disk. Three copies of that, differing only in which constants they passed.

**Order matters, and it is the reason this is a function rather than a comment.**
Nothing here may run until every model reference has been resolved to a content
identity. A recorded variant naming weights that could not be found describes a
run that never happened, so an unresolvable model has to abort before the first
directory is created -- which is easy to get right once and easy to get wrong
again in a fourth copy.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.core.pipeline._utils import hash_params, json_ready
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.tracking_roots import tracking_root_default
from mosaic.core.pipeline.tracks_identity import (
    tracker_variant_payload,
    tracks_run_id,
    write_tracks_variant,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["MintedRun", "mint_tracker_run", "tracker_run_root"]


def tracker_run_root(ds: Dataset, kind: str, run_id: str) -> Path:
    """Where one run of tracker *kind* keeps its per-entry working directories."""
    return ds.get_root(kind) / run_id


@dataclass(frozen=True, slots=True)
class MintedRun:
    """What one tracker run is called, and where it writes.

    Attributes:
        run_id: The content-addressed op run identifier, ``<kind>.<version>-<digest>``.
        run_root: The directory holding this run's per-entry working directories.
        params_hash: Digest of the whole settings dict. What a phase marker
            records when the tracker gates every phase on the same parameters; a
            tracker whose phases consume different subsets projects its own.
        tracks_variant: What names the ``tracks/`` tables this run produces.
            Byte-identical to ``run_id``, and separate because they answer
            different questions -- one names the run, the other names the recipe
            its tables belong to.
    """

    run_id: str
    run_root: Path
    params_hash: str
    tracks_variant: str


def mint_tracker_run(
    ds: Dataset,
    *,
    kind: str,
    version: str,
    settings: Mapping[str, object],
    observed: Mapping[str, str] | None = None,
) -> MintedRun:
    """Name a tracker run from its resolved settings, and record what it is.

    Call **after** every model reference has been resolved to a content identity;
    see the module docstring for why that ordering is not negotiable.

    Args:
        ds: The dataset. Its ``_tracking/<kind>`` root is created if absent.
        kind: The tracker's op kind, which is also its root key.
        version: The tracker's declared integration version.
        settings: The scope-free dict that defines the result -- knobs, and the
            model as a content digest. Never a path and never a video, so one
            value names one variant across every sequence the run covers.
        observed: Provenance recorded beside the variant, never identity. The
            model type a config happened to say, for instance: folding it in
            would re-derive every variant when an unrelated tool upgrade changed
            how the value reads.
    """
    if not ds.has_root(kind):
        ds.set_root(kind, tracking_root_default(kind))

    run_id = op_run_id(kind, version, dict(settings))
    run_root = tracker_run_root(ds, kind, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

    tracks_variant = tracks_run_id(kind, version, tracker_variant_payload(settings))
    _ = write_tracks_variant(
        ds.get_root("tracks"),
        tracks_variant,
        kind,
        version,
        settings,
        observed=observed,
    )

    # Best-effort: the settings are already recoverable from the identifier plus
    # the variant sidecar, so failing to write this readable copy must not lose a
    # run that is otherwise fine.
    params_path = run_root / "run_params.json"
    try:
        _ = params_path.write_text(json.dumps(json_ready(settings), indent=2))
    except OSError as exc:
        print(f"[{kind}] failed to save run_params.json: {exc}", file=sys.stderr)

    return MintedRun(
        run_id=run_id,
        run_root=run_root,
        params_hash=hash_params(settings),
        tracks_variant=tracks_variant,
    )
