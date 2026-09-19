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
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.models import (
    PREPARED_DATA_KINDS,
    model_index_path,
    prepared_artifact_cell,
)
from mosaic.core.pipeline.op_identity import parse_op_run_id
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
    """Content digest of the data a training or conversion op reads.

    Two forms, one per shape of input:

    - **A file** is digested by its own bytes (:func:`file_digest`), so a copy
      fingerprints identically wherever it sits. Nothing beside it enters. This
      form used to list every file under the file's *parent*, recursively, which
      made a ``.slp`` or a CVAT XML at a dataset root depend on the whole dataset
      -- including each run's own output, so an identical resubmission minted a
      new identifier and reuse could never hit.
    - **A directory** is digested by a listing of relative paths and sizes (not
      mtimes), so a copied or moved directory fingerprints identically. Sizes
      rather than bytes because a directory of images is too large to read on
      every planning call. ``train-localizer`` and ``train-litpose`` pass one.

    What a file names by path is not followed: a ``.slp`` carries the names of
    the videos its labels sit on, not their pixels. YOLO / POLO ``data.yaml``
    files have :func:`fingerprint_yolo_dataset`, which reads what the YAML
    declares for exactly that reason.

    A missing or unreadable path still yields a digest, the same one for both:
    identity computation is not the place to refuse a dataset, and the tool that
    reads it says which file is wrong and why.
    """
    path = Path(path)
    if path.is_file():
        try:
            return file_digest(path)
        except OSError:
            return hash_params({"listing": []})
    listing: list[str] = []
    if path.exists():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                listing.append(f"{f.relative_to(path).as_posix()}:{size}")
    return hash_params({"listing": listing})


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


def resolve_training_data(ds: Dataset, reference: str) -> Path:
    """What a training op's data argument names: a path, or a preparation run.

    A training op has always taken a path. It now also takes the run identifier
    of a preparation -- ``prepare-training-data`` or ``convert-points`` -- and
    reads what that run wrote, the way a model reference may be a path or the run
    that trained it.

    The difference matters to identity. The reference string is part of a
    training run's parameters, so it is hashed. A path is a location: the same
    data at two places mints two models, and moving a dataset re-mints every one.
    A run identifier is content, so it names the same data from anywhere, and it
    moves exactly when the annotations behind it do.

    A path wins when it exists, as it does for a model reference. A reference
    that names neither is returned as the path it spells, so the caller's own
    "not found" is what is reported rather than a second one from here.
    """
    candidate = Path(ds.resolve_path(reference))
    if candidate.exists():
        return candidate
    parsed = parse_op_run_id(reference)
    if parsed is None or parsed.kind not in PREPARED_DATA_KINDS:
        return candidate
    index_path = model_index_path(ds, parsed.kind)
    if not index_path.exists():
        return candidate
    import pandas as pd

    frame = pd.read_csv(index_path, dtype="string", keep_default_na=False)
    if "run_id" not in frame.columns:
        return candidate
    match = frame[frame["run_id"] == reference]
    if match.empty:
        return candidate
    row = {str(name): str(value) for name, value in match.iloc[-1].items()}
    stored = prepared_artifact_cell(row)
    return Path(ds.resolve_path(stored)) if stored else candidate


def fingerprint_yolo_dataset(data_yaml: Path) -> str:
    """Digest a YOLO / POLO training dataset by what its ``data.yaml`` declares.

    :func:`fingerprint_dataset` digests a file by its bytes, which for a
    ``data.yaml`` is the wrong answer twice over: the images it trains on are not
    in those bytes, and the absolute ``path`` it carries is. (It used to walk the
    YAML's parent recursively instead, folding in whatever sat beside it,
    including the run's own output.)

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
