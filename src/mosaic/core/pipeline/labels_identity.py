"""What names one *variant* of a converted label kind.

The label sibling of :mod:`mosaic.core.pipeline.tracks_identity`. A label kind
used to be written into a flat ``labels/<kind>/`` directory behind an
``exists()`` skip, so a re-conversion with different parameters overwrote the
first and nothing recorded *which recipe* produced the labels a model trained on.
Item 9.3 gives labels the tracks treatment: a params-only, scope-free identity
names one recipe across every sequence it covers, and Stage 9.3 promotes it into
a directory level (``labels/<kind>/<op>.<version>-<digest>/``), so two recipes
coexist rather than clobber.

Shape is ``<op>.<version>-<digest>``, the same as a tracks variant and an op run
identifier, for the same reasons: the version is a visible segment (readable on
disk, out of the digest so a bump re-derives nothing), and the op segment says
what produced the labels.

A **seventh identity family**, born marked ``1``. One number cannot honestly
cover several independent hash functions -- a label variant is a genuinely
different function from a tracks variant -- and the two must never read alike on
disk, which is why the op segment is ``convert-labels-<fmt>`` rather than the
tracks ``convert-<fmt>``: ``calms21_npy`` is registered as both a track and a
label converter, and a shared prefix would let their variants collide.

**Omit an absent term; never pass it empty.** ``json.dumps(..., sort_keys=True)``
digests an absent key differently from an empty one, so a *derived* label kind
computed from a tracks table or feature can gain the ``upstream`` term later
without moving the identifier of a *scored* kind converted from ``labels_raw``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ._utils import atomic_write, json_ready
from .op_identity import op_run_id

__all__ = [
    "LABELS_IDENTITY_SCHEME",
    "label_convert_variant_payload",
    "label_converter_op",
    "labels_run_id",
    "labels_variant_root",
    "write_labels_variant",
]

LABELS_IDENTITY_SCHEME: Final = "1"
"""The contract this module implements today.

Per family, like ``FEATURE_IDENTITY_SCHEME`` and ``TRACKS_IDENTITY_SCHEME``: one
number cannot honestly cover several independent hash functions, and a marker
that lies is worse than none. Label variants are born marked -- they did not
exist before this scheme -- so nothing has to be retrofitted.
"""


def labels_run_id(
    op: str,
    version: str,
    payload: Mapping[str, object],
    upstream: str | None = None,
) -> str:
    """Mint the identity of one label variant.

    Args:
        op: What produced it -- ``convert-labels-<src_format>`` for a scored kind,
            or an upstream op for a derived one. A readable segment of the
            directory name, so two producers are distinguishable without decoding
            a digest.
        version: The converter's *declared* version. A visible segment, kept out
            of the digest.
        payload: Everything about the recipe that determines the labels. Sort any
            collection in it: the suite runs under ``PYTHONHASHSEED=random``, so
            an unordered term names a different variant in every process.
        upstream: The tracks variant or feature run a *derived* kind was computed
            from, when a producer chains. Omitted from the digest entirely when
            None -- see the module docstring for why that is not the same as
            hashing an empty value. No derived-label producer exists yet; this is
            the reserved seam for one (item 9.3's Derived provenance).
    """
    terms: dict[str, object] = dict(payload)
    if upstream is not None:
        terms["upstream"] = upstream
    return op_run_id(op, version, terms)


def label_convert_variant_payload(
    label_kind: str, params_identity: Mapping[str, object]
) -> dict[str, object]:
    """What determines a label kind produced by converting a ``labels_raw`` file.

    The ``kind`` term domain-separates two kinds that share a ``src_format`` --
    the same reason ``source_composition_payload`` carries one -- and the named
    function (rather than a dict literal at the mint site) lets the golden corpus
    pin the wrapper: renaming ``"params"`` or ``"kind"`` here would move every
    label variant on disk, and a corpus that only called ``labels_run_id`` with a
    hand-built payload would stay green through it.
    """
    return {"kind": label_kind, "params": dict(params_identity)}


def label_converter_op(src_format: str) -> str:
    """The ``op`` segment naming a label conversion from *src_format*.

    ``convert-labels-`` rather than the tracks ``convert-`` prefix, so a label
    variant and a tracks variant of the same registered format never read alike
    on disk or in a digest -- ``calms21_npy`` is registered in both registries.
    """
    return f"convert-labels-{src_format}"


def labels_variant_root(labels_kind_root: Path, run_id: str) -> Path:
    """Where a label variant's tables and metadata live: ``labels/<kind>/<run_id>/``.

    *labels_kind_root* is ``labels/<kind>``; the variant is one level below it, so
    two recipes for one kind coexist rather than overwrite, and the kind stays the
    stable namespace a consumer selects by.
    """
    return labels_kind_root / run_id


def write_labels_variant(
    labels_kind_root: Path,
    run_id: str,
    op: str,
    version: str,
    label_kind: str,
    params_identity: Mapping[str, object],
    observed: Mapping[str, str] | None = None,
) -> Path:
    """Record what a label variant is, beside the ``.npz`` files it names.

    Idempotent: one variant is described once, however many sequences it covers,
    and re-running a conversion rewrites the same content.

    Args:
        labels_kind_root: The kind's directory, ``labels/<kind>``.
        run_id: The variant identity from :func:`labels_run_id`.
        op: The producer segment.
        version: The converter's declared version.
        label_kind: The kind produced, recorded so the sidecar is self-describing.
        params_identity: The hashed payload, persisted so a variant is explicable
            from disk rather than only comparable.
        observed: What the installed tooling reported. Provenance, never identity.

    The scheme marker rides inside this file rather than as a sibling dotfile:
    label variants are born under scheme 1, so there is no pre-scheme state to
    distinguish, and one file is one thing to keep consistent.
    """
    root = labels_variant_root(labels_kind_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "identity_scheme": LABELS_IDENTITY_SCHEME,
        "op": op,
        "version": version,
        "kind": label_kind,
        "params": json_ready(dict(params_identity)),
        "observed": dict(observed or {}),
    }
    path = root / "params.json"
    atomic_write(path, lambda p: p.write_text(json.dumps(record, indent=2)))
    return path
