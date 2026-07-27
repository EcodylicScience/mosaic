"""What names one *variant* of a dataset's standardized tracks.

Three producers write into ``tracks/``: a registered converter reading an
upload, the TREx tracker, and an inference op running a detection model. Until
now none of them recorded *which recipe* produced a table, so two tracker runs
with different settings, or a re-conversion with different parameters, both
targeted one flat ``tracks/<group>__<seq>.parquet`` behind an ``exists()`` skip
-- and the second was discarded with a success return.

The identity here is **params-only and scope-free**, so one value names one
variant across every sequence it covers rather than one per sequence. Source
*content* is tracked separately (item 5.1); this names the recipe, not the
input.

Shape is ``<op>.<version>-<digest>``, the same as an op run identifier and for
the same reasons -- the version is a visible segment so two producer versions
are distinguishable and readable on disk, and it stays out of the digest so a
version bump does not re-derive anything. Stage 3.2 promotes this value into a
directory level (``tracks/<op>.<version>-<hash>/``), which is why it is minted
in full now rather than as something weaker: the migration then moves files and
rewrites ``abs_path``, with no identity to re-derive and no ambiguity about
which variant a pre-existing row belongs to.

**Omit an absent term; never pass it empty.** ``json.dumps(..., sort_keys=True)``
digests an absent key differently from a key whose value is empty, which is the
same mechanism that lets ``compute_run_id`` add ``_scope_entries`` only for
scope-dependent features without disturbing the others. So a producer that
chains from another tracks variant can gain that term later without moving the
identifier of one that does not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ._utils import atomic_write, json_ready
from .op_identity import op_run_id

__all__ = [
    "TRACKS_IDENTITY_SCHEME",
    "convert_variant_payload",
    "converter_op",
    "infer_variant_payload",
    "tracks_run_id",
    "trex_variant_payload",
    "tracks_variant_root",
    "write_tracks_variant",
]

TRACKS_IDENTITY_SCHEME: Final = "1"
"""The contract this module implements today.

Per family, like ``FEATURE_IDENTITY_SCHEME`` and ``OP_IDENTITY_SCHEME``: one
number cannot honestly cover several independent hash functions, and a marker
that lies is worse than none. Tracks variants are born marked -- they did not
exist before this scheme -- so nothing has to be retrofitted.
"""


def tracks_run_id(
    op: str,
    version: str,
    payload: Mapping[str, object],
    upstream: str | None = None,
) -> str:
    """Mint the identity of one tracks variant.

    Args:
        op: What produced it -- ``convert-<src_format>``, ``trex``, or an
            inference kind. Becomes a readable segment of the directory name in
            Stage 3.2, so two producers are distinguishable without decoding a
            digest.
        version: The producer's *declared* version. A visible segment, kept out
            of the digest.
        payload: Everything about the recipe that determines the table. Sort any
            collection in it: the suite runs under ``PYTHONHASHSEED=random``, so
            an unordered term names a different variant in every process.
        upstream: The tracks variant this one was computed from, when a producer
            chains. Omitted from the digest entirely when None -- see the module
            docstring for why that is not the same as hashing an empty value.
    """
    terms: dict[str, object] = dict(payload)
    if upstream is not None:
        terms["upstream"] = upstream
    return op_run_id(op, version, terms)


def convert_variant_payload(params_identity: Mapping[str, object]) -> dict[str, object]:
    """What determines a table produced by converting an upload.

    The three payload builders exist as named functions rather than dict literals
    at their mint sites so the golden corpus can pin the **wrapper**, not just
    ``tracks_run_id``. Renaming this ``"params"`` key would move every tracks
    variant on disk, and a corpus that only called ``tracks_run_id`` with a
    hand-built payload would stay green through it.
    """
    return {"params": dict(params_identity)}


def trex_variant_payload(settings: Mapping[str, object]) -> dict[str, object]:
    """What determines a table bridged from a TREx run: the tracker settings.

    Passed through unwrapped, so the value this mints is byte-identical to the
    tracker's own ``trex_run_id(settings)``. That is deliberate rather than
    incidental: at Stage 3.2 ``tracks/trex.<v>-<digest>/`` and
    ``trex/trex.<v>-<digest>/`` then read as obviously the same run, and no
    existing golden line moves. Wrapping it would mint a second digest for one
    recipe and produce two near-identical directory names.

    The settings are scope-free -- knobs only, no video paths -- so one value
    still names one variant across every sequence the run covered.
    """
    return dict(settings)


def infer_variant_payload(
    params_identity: Mapping[str, object], model_id: str
) -> dict[str, object]:
    """What determines a table bridged from an inference run.

    The op params plus the model that produced the predictions, matching
    ``infer_run_id`` term for term and for the same reason: leaving the model out
    would let two detectors share one identifier.
    """
    return {"params": dict(params_identity), "model": model_id}


def converter_op(src_format: str) -> str:
    """The ``op`` segment naming a conversion from *src_format*.

    Prefixed rather than bare, so a tracks variant produced by converting an
    upload cannot collide with one produced by an op that happens to share the
    format's name, and so the directory says what kind of thing made it.
    """
    return f"convert-{src_format}"


def tracks_variant_root(tracks_root: Path, run_id: str) -> Path:
    """Where a variant's own metadata lives, and where Stage 3.2 will move its
    parquets.

    Created now, holding metadata only, so that migration is a file move plus an
    ``abs_path`` rewrite into a directory that already exists -- rather than a
    rename of something written somewhere else in the meantime.
    """
    return tracks_root / run_id


def write_tracks_variant(
    tracks_root: Path,
    run_id: str,
    op: str,
    version: str,
    params_identity: Mapping[str, object],
    observed: Mapping[str, str] | None = None,
) -> Path:
    """Record what a tracks variant is, beside the tables it names.

    Idempotent: one variant is described once, however many sequences it covers,
    and re-running a conversion rewrites the same content.

    Args:
        tracks_root: The dataset's ``tracks`` root.
        run_id: The variant identity from :func:`tracks_run_id`.
        op: The producer segment.
        version: The producer's declared version.
        params_identity: The hashed payload, persisted so a variant is
            explicable from disk rather than only comparable.
        observed: What the *installed* tooling reported -- a TREx build string,
            an ultralytics release, a model digest. Provenance, never identity:
            folding it in would re-derive every variant on an unrelated upgrade
            of a tool that produced byte-identical output.

    The scheme marker rides inside this file rather than as a sibling dotfile.
    Tracks variants are born under scheme 1, so there is no pre-scheme state to
    distinguish, and one file is one thing to keep consistent.
    """
    root = tracks_variant_root(tracks_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "identity_scheme": TRACKS_IDENTITY_SCHEME,
        "op": op,
        "version": version,
        "params": json_ready(dict(params_identity)),
        "observed": dict(observed or {}),
    }
    path = root / "params.json"
    atomic_write(path, lambda p: p.write_text(json.dumps(record, indent=2)))
    return path
