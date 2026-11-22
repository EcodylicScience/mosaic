"""Copying a finished model's run root into another dataset, with its origin.

A trained model is a row in ``models/<kind>/index.csv`` and the run root
``models/<kind>/<run_id>/`` that the row names. :func:`copy_run_root` copies both
from one dataset into another under the same run id. The run id names the same
model there, because the copied files are the source's byte for byte. It reads
the source and writes the destination alone.

**The origin pin.** The copied run root holds ``.mosaic-origin.json``
(:class:`OriginMarker`), naming the dataset it was copied from. A copied model's
identity cannot be recomputed without its training data, which stays behind, so
the marker keeps any reconciler from moving the run root to another run id. A
second copy from the same dataset is a no-op, and a copy from any other one is
refused.

**The order of a copy.** Every refusal comes before the first write. The files
are copied into a hidden staging directory beside the destination's run root,
``models/<kind>/.copy-<run_id>-<execution id>/``, the marker is written into it,
and the directory is renamed to the run root. Anything raised before the rename
completes removes the staging directory. A rename that fails because another copy
published the run root first is answered from that run root's marker. After the
rename the row is appended, and then its ``started_at`` and ``finished_at`` are
set to the source row's, which an append cannot carry. A model is read as
finished only through its row, so a copy stopped after the rename leaves no
finished model, and the next copy from the same origin completes it: it appends a
missing row, and it sets the times of a row whose ``finished_at`` is blank.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.markers import EXCHANGE_DIRECTORY, INFLIGHT_MARKER_NAME
from mosaic.core.pipeline.models import (
    MODEL_INDEX_PATH_COLUMNS,
    holds_trained_models,
    model_index_path,
    model_index_row,
    model_run_root,
)
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.runlog import new_execution_id, now_iso
from mosaic.tracking.ops.trained_models import (
    TrainedModelIndexRow,
    trained_model_index,
    training_is_complete,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mosaic.core.dataset import Dataset

__all__ = [
    "ORIGIN_MARKER_NAME",
    "CopyOutcome",
    "CopyRefusalReason",
    "CopyRefusedError",
    "CopyReport",
    "OriginMarker",
    "copy_run_root",
    "read_origin_marker",
]

ORIGIN_MARKER_NAME: Final = ".mosaic-origin.json"
"""The file in a copied run root that names the dataset it was copied from."""

_LEFT_BEHIND: Final = frozenset(
    {ORIGIN_MARKER_NAME, INFLIGHT_MARKER_NAME, EXCHANGE_DIRECTORY}
)
"""The names in a run root that a copy does not carry.

The source's own origin, its in-flight claim and its training exchange describe
the source's run root and not the model.
"""

type CopyRefusalReason = Literal[
    "no_uuid",
    "unsupported_root",
    "prepared_data",
    "not_a_run_of_kind",
    "not_finished",
    "run_root_is_a_link",
    "same_dataset",
    "no_models_root",
    "other_origin",
]
"""Why a copy was refused before anything was written."""

type CopyOutcome = Literal["copied", "completed", "unchanged", "planned"]
"""What a copy did: copied the run root, completed an earlier copy's row, found
the copy already done, or, for a dry run, found work to do."""


class CopyRefusedError(ValueError):
    """A copy refused before anything was written.

    Attributes:
        reason: Which refusal this is, for a caller that answers each one
            differently.
        kind: The kind as given.
        run_id: The run identifier as given.
    """

    def __init__(
        self, message: str, *, reason: CopyRefusalReason, kind: str, run_id: str
    ) -> None:
        super().__init__(message)
        self.reason: CopyRefusalReason = reason
        self.kind: str = kind
        self.run_id: str = run_id


class OriginMarker(BaseModel):
    """The contents of ``.mosaic-origin.json`` in a copied run root.

    Attributes:
        source_name: The name of the dataset the run root was copied from.
        source_uuid: That dataset's manifest ``uuid``, which decides whether a
            later copy comes from the same origin.
        source_path: That dataset's resolved base directory at the time of the
            copy. Provenance: a share mounted elsewhere is the same origin.
        run_id: The copied run.
        kind: The training op that wrote it.
        copied_at: When the copy was made, as an ISO 8601 UTC instant.
        content_digest: The blake2b hex digest of the sorted
            ``<relative path>:<file digest>`` lines of every copied file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_name: str
    source_uuid: str
    source_path: str
    run_id: str
    kind: str
    copied_at: str
    content_digest: str


def read_origin_marker(run_root: Path) -> OriginMarker | None:
    """The origin marker of *run_root*, or ``None`` when absent or unreadable."""
    path = run_root / ORIGIN_MARKER_NAME
    if not path.exists():
        return None
    try:
        return OriginMarker.model_validate_json(path.read_text())
    except (OSError, ValidationError):
        return None


@dataclass(frozen=True, slots=True)
class CopyReport:
    """What :func:`copy_run_root` did, or would do.

    Attributes:
        kind: The training op that wrote the model.
        run_id: The model's run identifier, the same in both datasets.
        files: The files copied, or that a dry run would copy, as POSIX paths
            relative to the run root in sorted order. Empty when the run root
            is already in the destination.
        outcome: What the copy did.
    """

    kind: str
    run_id: str
    files: tuple[str, ...]
    outcome: CopyOutcome


type _Destination = Literal["absent", "no_row", "unfinished", "finished"]
"""The state of the run in the destination, when it is not another origin's."""


def copy_run_root(
    src: Dataset,
    dst: Dataset,
    *,
    root_key: str = "models",
    kind: str,
    run_id: str,
    dry_run: bool = True,
) -> CopyReport:
    """Copy the finished model *run_id* of *kind* from *src* into *dst*.

    The run root is copied without the source's origin marker, in-flight marker
    and training exchange, and the copy records *src* in its own origin marker.
    The row appended to *dst*'s index keeps every cell of the source row. A path
    cell under *src*'s ``models`` root names the same place under *dst*'s, and
    any other is kept as the path it resolves to. Each is stored through
    ``dst.relative_to_root``, so a path outside *dst* is stored absolute.

    Args:
        src: The dataset that holds the model.
        dst: The dataset that receives it.
        root_key: The root that holds the run. Only ``"models"`` is copied.
        kind: The training op that wrote the model, such as ``train-pose``.
        run_id: The model's run identifier.
        dry_run: Report what a copy would do and write nothing. A dry run takes
            no lock.

    Returns:
        What the copy did, or would do.

    Raises:
        CopyRefusedError: *src* or *dst* has no ``uuid``, *root_key* is not
            ``"models"``, *kind* holds prepared data, *run_id* is not a run of
            *kind*, *src* holds no finished model of that run or its row
            records an epoch count that is not a whole number, its run root is
            a link, the two are one dataset, *dst* has no ``models`` root, or
            *dst* holds the run from another origin. Nothing is written.
        OSError: Copying a file failed, or the rename failed and left no run
            root. The staging directory is removed.
    """
    _refuse_without_uuid(src, dst, kind, run_id)
    if root_key != "models":
        message = (
            f"Only a run root under 'models' is copied, not one under {root_key!r}."
        )
        raise CopyRefusedError(
            message, reason="unsupported_root", kind=kind, run_id=run_id
        )
    _refuse_unless_a_model_run(kind, run_id)
    source_row = _finished_row(src, kind, run_id)
    source_root = model_run_root(src, kind, run_id)
    if source_root.is_symlink():
        message = f"{source_root} is a link rather than a run root of its own."
        raise CopyRefusedError(
            message, reason="run_root_is_a_link", kind=kind, run_id=run_id
        )
    if src.uuid == dst.uuid:
        message = f"{src.name} and {dst.name} are one dataset (uuid {src.uuid})."
        raise CopyRefusedError(message, reason="same_dataset", kind=kind, run_id=run_id)
    if not dst.has_root("models"):
        message = f"{dst.name} has no models root to copy {run_id} into."
        raise CopyRefusedError(
            message, reason="no_models_root", kind=kind, run_id=run_id
        )
    row = _copied_row(src, dst, kind, run_id, source_row)

    state = _destination(src, dst, kind, run_id)
    if state == "absent":
        files = _copied_files(source_root)
        if dry_run:
            return CopyReport(kind, run_id, files, "planned")
        lost = _publish(src, dst, kind, run_id, files)
        if lost is None:
            _record(dst, row, source_row, append=True)
            return CopyReport(kind, run_id, files, "copied")
        state = _destination(src, dst, kind, run_id)
        if state == "absent":
            raise lost
    if state == "finished":
        return CopyReport(kind, run_id, (), "unchanged")
    if dry_run:
        return CopyReport(kind, run_id, (), "planned")
    _record(dst, row, source_row, append=state == "no_row")
    return CopyReport(kind, run_id, (), "completed")


def _refuse_without_uuid(src: Dataset, dst: Dataset, kind: str, run_id: str) -> None:
    for role, ds in (("source", src), ("destination", dst)):
        if not ds.uuid:
            message = (
                f"The {role} dataset {ds.name} has no uuid, so a copy cannot record "
                "or compare its origin."
            )
            raise CopyRefusedError(message, reason="no_uuid", kind=kind, run_id=run_id)


def _refuse_unless_a_model_run(kind: str, run_id: str) -> None:
    """Raise unless *kind* holds trained models and *run_id* is one of its runs.

    This also keeps both arguments one path component each. A parsed run
    identifier cannot contain a separator or ``..``, and *kind* must be the kind
    it parses to.
    """
    if not holds_trained_models(kind):
        message = (
            f"models/{kind} holds prepared training data, not trained models, so "
            "it has no model to copy."
        )
        raise CopyRefusedError(
            message, reason="prepared_data", kind=kind, run_id=run_id
        )
    parsed = parse_op_run_id(run_id)
    if parsed is None or parsed.kind != kind:
        message = f"{run_id!r} is not a run identifier of {kind}."
        raise CopyRefusedError(
            message, reason="not_a_run_of_kind", kind=kind, run_id=run_id
        )


def _finished_row(src: Dataset, kind: str, run_id: str) -> dict[str, str]:
    """The source's row of the run, refused unless it is a finished model."""
    row = model_index_row(src, kind, run_id) if src.has_root("models") else None
    if (
        row is None
        or not row.get("finished_at", "")
        or not training_is_complete(src, kind, run_id)
    ):
        message = (
            f"{src.name} holds no finished {kind} model {run_id} whose artifact is "
            "on disk."
        )
        raise CopyRefusedError(message, reason="not_finished", kind=kind, run_id=run_id)
    return row


def _destination(src: Dataset, dst: Dataset, kind: str, run_id: str) -> _Destination:
    """The state of the run in *dst*, refused when another origin holds it.

    A run root without a marker naming *src* is another origin's, and so is a
    row without a run root, because a copy publishes the run root before its row.
    """
    run_root = model_run_root(dst, kind, run_id)
    row = model_index_row(dst, kind, run_id)
    if not run_root.exists() and not run_root.is_symlink():
        if row is None:
            return "absent"
        holder = f"{dst.name} registers {run_id} without a run root"
    else:
        marker = read_origin_marker(run_root)
        if marker is not None and marker.source_uuid == src.uuid:
            if row is None:
                return "no_row"
            return "finished" if row.get("finished_at", "") else "unfinished"
        holder = (
            f"{run_root} was not copied from {src.name}"
            if marker is None
            else f"{run_root} was copied from {marker.source_name} "
            f"(uuid {marker.source_uuid})"
        )
    message = f"{holder}; not copying {run_id} from {src.name} over it."
    raise CopyRefusedError(message, reason="other_origin", kind=kind, run_id=run_id)


def _copied_files(source_root: Path) -> tuple[str, ...]:
    """The files of *source_root* that a copy carries, as sorted POSIX paths."""
    found: list[str] = []
    for directory, subdirectories, names in os.walk(source_root):
        here = Path(directory)
        if here == source_root:
            subdirectories[:] = [d for d in subdirectories if d not in _LEFT_BEHIND]
            names = [n for n in names if n not in _LEFT_BEHIND]
        found.extend((here / n).relative_to(source_root).as_posix() for n in names)
    return tuple(sorted(found))


def _content_digest(run_root: Path, files: tuple[str, ...]) -> str:
    lines = sorted(f"{name}:{file_digest(run_root / name)}" for name in files)
    return hashlib.blake2b("\n".join(lines).encode()).hexdigest()


def _publish(
    src: Dataset, dst: Dataset, kind: str, run_id: str, files: tuple[str, ...]
) -> OSError | None:
    """Stage the copy and rename it into place.

    Returns:
        ``None`` once the run root is published, or the error of a failed
        rename. The staging directory is removed either way, and when anything
        before the rename raises.
    """
    source_root = model_run_root(src, kind, run_id)
    run_root = model_run_root(dst, kind, run_id)
    staging = run_root.parent / f".copy-{run_id}-{new_execution_id()}"
    try:
        staging.mkdir(parents=True)
        for name in files:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(source_root / name, target)
        marker = OriginMarker(
            source_name=src.name,
            source_uuid=src.uuid or "",
            source_path=str(src.base_dir.resolve()),
            run_id=run_id,
            kind=kind,
            copied_at=now_iso(),
            content_digest=_content_digest(staging, files),
        )
        text = marker.model_dump_json(indent=2) + "\n"
        atomic_write(staging / ORIGIN_MARKER_NAME, lambda p: p.write_text(text))
        try:
            os.rename(staging, run_root)
        except OSError as exc:
            return exc
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _record(
    dst: Dataset,
    row: TrainedModelIndexRow,
    source_row: Mapping[str, str],
    *,
    append: bool,
) -> None:
    """Append *row* when *append*, then set its times to the source row's."""
    index = trained_model_index(model_index_path(dst, row.kind))
    if append:
        index.append([row])
    _ = index.update_cells(
        row.run_id,
        {
            "started_at": source_row.get("started_at", ""),
            "finished_at": source_row.get("finished_at", ""),
        },
    )


def _copied_row(
    src: Dataset,
    dst: Dataset,
    kind: str,
    run_id: str,
    source_row: Mapping[str, str],
) -> TrainedModelIndexRow:
    """The source row with each path cell stored for *dst*.

    Built before the first write, so a cell that cannot be copied is refused
    before the run root is published.

    Raises:
        CopyRefusedError: The ``n_epochs`` cell is not a whole number, which
            :func:`_epoch_count` refuses as ``not_finished``.
    """
    cells = {
        name: _copied_cell(src, dst, cell)
        if name in MODEL_INDEX_PATH_COLUMNS or name == "abs_path"
        else cell
        for name, cell in source_row.items()
    }
    return TrainedModelIndexRow(
        run_id=run_id,
        kind=kind,
        base_model=cells.get("base_model", ""),
        base_run_id=cells.get("base_run_id", ""),
        best_model_path=cells.get("best_model_path", ""),
        metrics_path=cells.get("metrics_path", ""),
        n_epochs=_epoch_count(cells.get("n_epochs", ""), kind, run_id),
        status=cells.get("status", ""),
        base_digest=cells.get("base_digest", ""),
        artifact_shape=cells.get("artifact_shape", ""),
        artifact_path=cells.get("artifact_path", ""),
        model_type=cells.get("model_type", ""),
        data_path=cells.get("data_path", ""),
        data_fingerprint=cells.get("data_fingerprint", ""),
        base_origin=cells.get("base_origin", ""),
        abs_path=Path(cells["abs_path"]),
    )


def _epoch_count(cell: str, kind: str, run_id: str) -> int:
    """The ``n_epochs`` cell as an integer, ``0`` for a blank cell.

    An older index wrote a whole number as a float, ``100`` as ``100.0``, which
    reads back as the integer. A cell that holds no whole number is not a row
    that a finished training wrote, and is refused as ``not_finished``.
    """
    if not cell:
        return 0
    try:
        value = float(cell)
    except ValueError:
        value = math.nan
    if not value.is_integer():
        message = f"The {kind} row of {run_id} records {cell!r} epochs."
        raise CopyRefusedError(message, reason="not_finished", kind=kind, run_id=run_id)
    return int(value)


def _copied_cell(src: Dataset, dst: Dataset, cell: str) -> str:
    """The cell of *dst* for the path that *cell* names in *src*.

    A path under *src*'s ``models`` root moves to the same subpath under
    *dst*'s, since a run there is named by its run id, which the copy keeps. Any
    other path stays the path it resolves to. Blank stays blank.
    """
    if not cell:
        return ""
    resolved = src.resolve_path(cell).resolve()
    try:
        subpath = resolved.relative_to(src.get_root("models").resolve())
    except ValueError:
        return dst.relative_to_root(resolved)
    return dst.relative_to_root(dst.get_root("models") / subpath)
