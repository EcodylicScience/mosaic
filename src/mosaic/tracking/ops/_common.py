"""Shared, dependency-light helpers for tracking ops.

Factored out of ``ops/train.py`` so the training ops and the ``convert-points`` op
share one copy of the models-root guard and the copy-stable dataset fingerprint used
in content ``run_id`` computation. Behavior is identical to the original private
helpers -- training ``run_id``s are unchanged by the move.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.markers import (
    InflightMarker,
    clear_inflight,
    inflight_state,
    new_inflight,
    read_inflight,
    try_create_inflight,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def ensure_models_root(ds: Dataset) -> None:
    """Ensure the dataset has a ``models`` root (default ``models/``)."""
    if not ds.has_root("models"):
        ds.set_root("models", "models")


def fingerprint_dataset(path: Path) -> str:
    """Cheap digest of a training/converted dataset (file text + size listing).

    Uses relative paths + file sizes (not mtimes), so a copied or moved *directory*
    fingerprints identically and its run_ids stay deterministic across machines.

    Two limits, both of which bite only the **file** form and are why
    :func:`fingerprint_yolo_dataset` exists for the YOLO / POLO ``data.yaml``:

    - the listing walks ``path.parent`` recursively, so unrelated siblings enter
      the digest;
    - the file's text enters verbatim, so a file naming an absolute path -- as a
      ``data.yaml`` written by ``make_data_yaml`` does -- carries its own location
      into the digest and is not copy-stable after all.

    Callers passing a purpose-built directory (``train-localizer``,
    ``train-litpose``) meet neither.
    """
    path = Path(path)
    parts: dict[str, object] = {}
    if path.is_file():
        parts["file"] = path.name
        try:
            parts["text"] = path.read_text(errors="ignore")
        except Exception:
            parts["text"] = ""
        base = path.parent
    else:
        base = path
    listing: list[str] = []
    if base.exists():
        for f in sorted(base.rglob("*")):
            if f.is_file():
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                listing.append(f"{f.relative_to(base).as_posix()}:{size}")
    parts["listing"] = listing
    return hash_params(parts)


# The keys a YOLO / POLO ``data.yaml`` uses to name its splits. ``path`` is
# deliberately absent: it is a location, not content.
_SPLIT_KEYS: Final = ("train", "val", "test")


_PARSED_DATA_YAML: Final = TypeAdapter(dict[str, JsonValue])
"""Turns whatever the YAML parser produced into a typed mapping, or says why not.

The same device ``core.manifest`` uses on ``dataset.yaml``, and for the same
reason: a parser returns an untyped object, and a data.yaml that is a list, a
bare string, or a mapping with a non-string key is a real thing to find on disk.
Validating once here means the rest of this function works with
``dict[str, JsonValue]`` instead of re-checking at each use.
"""


def _split_roots(declared: JsonValue, base: Path) -> list[Path]:
    """Resolve one split declaration to the roots it names.

    A split is a string or a list of strings, each relative to *base* unless it
    is already absolute.
    """
    if isinstance(declared, str):
        spellings = [declared]
    elif isinstance(declared, list):
        spellings = [item for item in declared if isinstance(item, str)]
    else:
        return []
    return [Path(s) if Path(s).is_absolute() else base / s for s in spellings]


def _listing_under(root: Path) -> list[str]:
    """Relative-path + size listing for one split root, or its text if a file.

    Sizes rather than mtimes, and paths relative to *root* rather than to any
    shared ancestor, so a dataset that moves fingerprints identically. A split
    may point at a ``.txt`` image list instead of a directory, in which case the
    file's text is what names the images.
    """
    if root.is_file():
        try:
            return [root.read_text(errors="ignore")]
        except OSError:
            return []
    if not root.is_dir():
        return []
    entries: list[str] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = -1
        entries.append(f"{f.relative_to(root).as_posix()}:{size}")
    return entries


def fingerprint_yolo_dataset(data_yaml: Path) -> str:
    """Digest a YOLO / POLO training dataset by what its ``data.yaml`` declares.

    :func:`fingerprint_dataset` handed a file digests the file's text and then
    walks its parent **recursively**, which makes a training identity depend on
    whatever else happens to sit beside the YAML -- including anything the run
    itself writes, so an identical resubmission mints a different ``run_id`` and
    content-addressed reuse can never hit.

    This reads the YAML instead and digests two things: the declared *content*
    (class names, keypoint shape, radii, and the relative split spellings) and a
    listing of the files under each declared split root. The absolute ``path``
    the YAML carries is excluded, because a path is a location -- ``make_data_yaml``
    writes ``os.path.abspath(dataset_root)``, so digesting the raw text made the
    same annotations at two locations two different models.

    A YAML that declares no splits, names roots that do not exist yet, or does
    not parse still yields a digest: identity computation is not the place to
    refuse a dataset.

    Args:
        data_yaml: The ``data.yaml`` naming the dataset.

    Returns:
        A 10-character digest, stable across copies and moves.
    """
    import yaml

    data_yaml = Path(data_yaml)
    try:
        text = data_yaml.read_text(errors="ignore")
    except OSError:
        text = ""
    try:
        loaded: object = yaml.safe_load(text)
        parsed = _PARSED_DATA_YAML.validate_python(loaded)
    except (yaml.YAMLError, ValidationError):
        # Unreadable, not a mapping, or carrying something no JSON value covers.
        # Fall back to the file's own text, so two different odd YAMLs still
        # fingerprint differently rather than collapsing onto one digest.
        return hash_params({"file": data_yaml.name, "text": text, "declared": None})

    declared = {key: value for key, value in parsed.items() if key != "path"}
    root = parsed.get("path")
    base = data_yaml.parent
    if isinstance(root, str):
        base = Path(root) if Path(root).is_absolute() else data_yaml.parent / root

    listings: dict[str, list[str]] = {}
    for key in _SPLIT_KEYS:
        declaration = parsed.get(key)
        if declaration is None:
            continue
        entries: list[str] = []
        for split_root in _split_roots(declaration, base):
            entries.extend(_listing_under(split_root))
        listings[key] = entries

    return hash_params({"declared": declared, "splits": listings})


class RunRootHeld(RuntimeError):
    """Another execution is already producing this run, so this one must not."""


def claim_run_root(
    ds: Dataset, ctx: JobContext, run_root: Path, kind: str, idle_seconds: float
) -> InflightMarker:
    """Take *run_root* exclusively for a one-shot op, or raise.

    Per-entry work skips a contended item so one sequence cannot end a batch; a
    one-shot op *is* the batch, and returning its run_id would hand back a model
    another execution is mid-write. Two executions of one identifier are not merely
    wasted: a nondeterministic trainer interleaves ``best.pt`` / ``last.pt`` /
    ``results.csv`` into one root, which is corrupt rather than slow.

    No ``finally`` release -- ``inflight_state`` reads a holder whose run-log went
    terminal as ``orphaned``, so a dead execution frees the root by itself.

    Returns:
        The claim that was taken, so a caller with output to read can keep it
        alive. A claim expires ``idle_seconds`` plus a grace after it is written,
        and a one-shot op that outruns that window has its root read as abandoned
        by the next execution along -- which then clears it and starts writing
        into the same directory. Only a caller with lines arriving from its tool
        can refresh it, through
        :func:`~mosaic.tracking.common.entry.phase_activity`, and it cannot do
        that without the marker. This was minted and dropped on the floor for as
        long as every one-shot op ran in process and had nothing to hang a
        refresh on.
    """
    marker = new_inflight(
        execution_id=ctx.execution_id,
        host=socket.gethostname(),
        pid=os.getpid(),
        phase=None,
        idle_seconds=idle_seconds,
    )
    held: InflightMarker | None = None
    for attempt in (0, 1):
        if try_create_inflight(run_root, marker):
            return marker
        held = read_inflight(run_root)
        state = inflight_state(
            held, run_log_base=ds.base_dir, execution_id=ctx.execution_id
        )
        if state == "mine":
            return marker
        if state in {"expired", "orphaned"} and attempt == 0:
            clear_inflight(run_root)
            continue
        break
    where = f"{held.host}:{held.pid}" if held is not None else "another host"
    raise RunRootHeld(
        f"[{kind}] {run_root.name} is being produced by execution "
        f"{held.execution_id if held else '?'} on {where}; not training it again."
    )
