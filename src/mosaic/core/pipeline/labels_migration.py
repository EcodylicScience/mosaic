"""One-shot migrations for item 9.3: the ``labels_raw`` root and the typed
labels index.

Two directions, because 9.3 is a ``[break]`` and the programme's rollback rule
asks every break to ship its reverse:

- :func:`migrate_labels_raw` is additive and safe to re-run. It copies the
  label-format rows out of ``tracks_raw/index.csv`` into a new
  ``labels_raw/index.csv`` and writes the composition. **Files are never moved**
  -- a format registered as both a track and a label converter (``calms21_npy``)
  keeps its one physical file, referenced from both indexes. A dataset that
  predates ``labels_raw`` needs this once before a conversion can read the root.

- :func:`revert_labels` undoes both halves of the break: it flattens each kind's
  resolved variant back to ``labels/<kind>/<entry>.npz``, rewrites the per-kind
  index in the pre-9.3 untyped twelve-column shape, removes the variant
  directories, and drops the ``labels_raw`` index and composition. Lossless for
  the resolved variant, because the ``.npz`` payloads never changed; a second,
  superseded variant's files are the state 9.3 introduced and are removed with
  its directory.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from .index_lock import index_lock
from .labels_index import (
    legacy_labels_view,
    read_labels_index,
    select_label_variant_rows,
)
from .tracks_raw_index import (
    frame_from_rows,
    read_tracks_raw_index,
    write_tracks_raw_index_rows,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

# The pre-9.3 untyped labels index, in the order ``_ensure_labels_index`` wrote.
# Kept here rather than imported because the constant it names was deleted with
# the writer; the reverse migration is the only code that still needs the shape.
_LEGACY_LABEL_COLUMNS = [
    "kind",
    "label_format",
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    "abs_path",
    "source_abs_path",
    "source_md5",
    "n_frames",
    "label_ids",
    "label_names",
]


def _label_kinds(ds: Dataset) -> list[str]:
    """Every ``labels/<kind>/`` directory holding an ``index.csv``, sorted."""
    try:
        root = ds.get_root("labels")
    except KeyError:
        return []
    if not root.exists():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "index.csv").exists()
    )


def migrate_labels_raw(ds: Dataset) -> dict[str, int]:
    """Populate ``labels_raw`` from the label-format rows of ``tracks_raw``.

    Idempotent. Returns ``{"rows": n}`` for the number of label source rows
    projected. Files are referenced where they lie and never copied or moved.
    """
    from mosaic.core.label_converter import registered_label_formats

    # Through the accessor, which fills the registry if nothing has: read raw,
    # an empty registry matches no row, and this returns zero rows migrated --
    # the right-looking answer to a question it never got to ask.
    label_formats = registered_label_formats()
    tracks_raw_index = ds.get_root("tracks_raw") / "index.csv"
    rows = [
        row
        for row in read_tracks_raw_index(tracks_raw_index)
        if str(row.get("src_format", "")) in label_formats
    ]
    if not rows:
        return {"rows": 0}

    out = ds.get_root("labels_raw") / "index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = frame_from_rows(rows)
    with index_lock(out):
        write_tracks_raw_index_rows(out, frame)
    # Same projection the live writer uses, so the migrated composition is
    # byte-identical to one produced by index_labels_raw.
    ds._write_labels_raw_compositions(frame.to_dict("records"))
    return {"rows": len(rows)}


def revert_labels(ds: Dataset) -> dict[str, int]:
    """Undo item 9.3's labels break: flatten variants, restore the untyped index.

    Returns ``{"kinds": k, "labels_raw_removed": 0 or 1}``.
    """
    import pandas as pd

    labels_root = ds.get_root("labels")
    kinds = _label_kinds(ds)
    for kind in kinds:
        kind_root = labels_root / kind
        resolved = legacy_labels_view(
            select_label_variant_rows(read_labels_index(ds, kind), None)
        )
        legacy_rows: list[dict[str, object]] = []
        for _, row in resolved.iterrows():
            stored = str(row.get("abs_path", ""))
            if not stored:
                continue
            source = ds.resolve_path(stored)
            flat = kind_root / source.name
            if source.exists() and source.resolve() != flat.resolve():
                flat.parent.mkdir(parents=True, exist_ok=True)
                _ = shutil.move(str(source), str(flat))
            legacy_rows.append(
                {
                    "kind": str(row.get("label_kind", kind)) or kind,
                    "label_format": str(row.get("label_format", "")),
                    "group": str(row.get("group", "")),
                    "sequence": str(row.get("sequence", "")),
                    "group_safe": str(row.get("group_safe", "")),
                    "sequence_safe": str(row.get("sequence_safe", "")),
                    "abs_path": ds.relative_to_root(flat),
                    "source_abs_path": str(row.get("source_abs_path", "")),
                    "source_md5": str(row.get("source_md5", "")),
                    "n_frames": str(row.get("n_frames", "")),
                    "label_ids": str(row.get("label_ids", "")),
                    "label_names": str(row.get("label_names", "")),
                }
            )
        # Remove the variant directories 9.3 introduced (the flat files have been
        # moved out of them already).
        for child in kind_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
        idx_path = kind_root / "index.csv"
        with index_lock(idx_path):
            pd.DataFrame(legacy_rows, columns=_LEGACY_LABEL_COLUMNS).to_csv(
                idx_path, index=False
            )

    removed = 0
    try:
        labels_raw = ds.get_root("labels_raw")
    except KeyError:
        labels_raw = None
    if labels_raw is not None:
        for name in ("index.csv", "sequences.csv"):
            path = labels_raw / name
            if path.exists():
                path.unlink()
                removed = 1
    return {"kinds": len(kinds), "labels_raw_removed": removed}
