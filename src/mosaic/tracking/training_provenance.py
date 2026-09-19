"""From a trained model back to the annotations it saw.

Each link of that chain is written where it is known, by a different writer:

```
models/<train kind>/<run_id>            index row: data_path, data_fingerprint
  -> models/prepare-training-data/<id>  index row: consumed_sets
  -> labels_raw/keypoints/index.csv     the claimed revision, its path
  -> <revision>/manifest.json           origin: what the authoring store recorded
```

Nothing stored the whole chain, on purpose: a copy of it would be a second thing
to keep agreeing with the four rows it summarizes. This walks them instead, and
says where the walk stopped when a link is missing -- a model trained from a bare
path, a revision in a dataset that is archived or unmounted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    read_revision_manifest,
)
from mosaic.core.pipeline.models import PREPARED_DATA_KINDS, model_index_path
from mosaic.core.pipeline.op_identity import parse_op_run_id

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["ConsumedRevision", "TrainingProvenance", "training_provenance"]


def _no_origin() -> dict[str, JsonValue]:
    return {}


class _ConsumedEntry(BaseModel):
    """One entry of a prepared dataset's ``consumed_sets`` cell."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    origin_uuid: str = ""
    set_key: str = ""
    revision: int = 0
    digest: str = ""


_CONSUMED: Final = TypeAdapter(list[_ConsumedEntry])
"""Turns the JSON cell into typed entries, or says it is not one."""


@dataclass(frozen=True, slots=True)
class ConsumedRevision:
    """One annotation revision a model was trained on.

    Attributes:
        origin_uuid: The dataset the revision was saved in.
        set_key: The annotation set.
        revision: Which saved state of it.
        digest: Its content digest, which is what entered the identifiers.
        path: Where the revision is, or ``None`` when no indexed row names it.
        reachable: Whether the revision's file is on disk now.
        origin: What the authoring store recorded with the save -- a database
            commit, for instance. Empty when the revision is not reachable.
    """

    origin_uuid: str
    set_key: str
    revision: int
    digest: str
    path: Path | None = None
    reachable: bool = False
    origin: Mapping[str, JsonValue] = field(default_factory=_no_origin)


@dataclass(frozen=True, slots=True)
class TrainingProvenance:
    """What one trained model was made from, as far as disk can say.

    Attributes:
        kind: The training op.
        run_id: The model.
        served_by: The linked library that holds it, or ``""`` for this dataset.
        data_path: What the run was trained on, as its row recorded it.
        data_fingerprint: The content fingerprint that entered its identifier.
        base_run_id: What it was fine-tuned from, or ``""``.
        prepared_kind: The preparation op behind ``data_path``, or ``""``.
        prepared_run_id: That preparation run, or ``""``.
        sets: The annotation revisions the preparation consumed.
        stopped_at: Why the walk went no further, or ``""`` when it reached
            every revision. Never an error: an incomplete chain is an answer.
    """

    kind: str
    run_id: str
    served_by: str = ""
    data_path: str = ""
    data_fingerprint: str = ""
    base_run_id: str = ""
    prepared_kind: str = ""
    prepared_run_id: str = ""
    sets: tuple[ConsumedRevision, ...] = ()
    stopped_at: str = ""

    def as_json(self) -> dict[str, JsonValue]:
        """A plain mapping, for ``--json`` and for a control plane to store."""
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "served_by": self.served_by,
            "data_path": self.data_path,
            "data_fingerprint": self.data_fingerprint,
            "base_run_id": self.base_run_id,
            "prepared_kind": self.prepared_kind,
            "prepared_run_id": self.prepared_run_id,
            "stopped_at": self.stopped_at,
            "sets": [
                {
                    "origin_uuid": item.origin_uuid,
                    "set_key": item.set_key,
                    "revision": item.revision,
                    "digest": item.digest,
                    "path": str(item.path) if item.path is not None else "",
                    "reachable": item.reachable,
                    "origin": dict(item.origin),
                }
                for item in self.sets
            ],
        }


def _row(index_path: Path, run_id: str) -> dict[str, str] | None:
    if not index_path.exists():
        return None
    frame = pd.read_csv(index_path, dtype="string", keep_default_na=False)
    if "run_id" not in frame.columns:
        return None
    matching = [r for r in index_records(frame) if r.get("run_id", "") == run_id]
    return matching[-1] if matching else None


def _holder(ds: Dataset, kind: str, run_id: str) -> tuple[Dataset, str, dict[str, str]]:
    """The dataset whose index registers the model: this one, then each library."""
    own = _row(model_index_path(ds, kind), run_id)
    if own is not None:
        return ds, "", own
    for linked in ds.linked_libraries():
        found = _row(model_index_path(linked.dataset, kind), run_id)
        if found is not None:
            return linked.dataset, linked.link.id, found
    msg = (
        f"no {kind} run {run_id!r} is registered in this dataset or in a library "
        "it links"
    )
    raise KeyError(msg)


def _prepared_run(data_path: str) -> tuple[str, str]:
    """The preparation a data path lies in, as ``(kind, run_id)``, or two blanks.

    Read from the path: a preparation writes under
    ``models/<kind>/<run_id>/``, so the run is the artifact's directory. A model
    trained from anything else -- a folder somebody exported -- has no
    preparation to name.
    """
    for part in Path(data_path).parts:
        parsed = parse_op_run_id(part)
        if parsed is not None and parsed.kind in PREPARED_DATA_KINDS:
            return parsed.kind, part
    return "", ""


def _consumed(holder: Dataset, cell: str) -> tuple[ConsumedRevision, ...]:
    try:
        entries = _CONSUMED.validate_json(cell or "[]")
    except ValidationError:
        return ()
    indexed = index_records(read_label_series(holder, "keypoints"))
    found: list[ConsumedRevision] = []
    for entry in entries:
        origin_uuid = entry.origin_uuid
        set_key = entry.set_key
        revision = entry.revision
        digest = entry.digest
        row = next(
            (
                record
                for record in indexed
                if record.get("origin_uuid", "") == origin_uuid
                and record.get("key", "") == set_key
                and record.get("revision", "") == str(revision)
            ),
            None,
        )
        path = holder.resolve_path(row["abs_path"]) if row is not None else None
        reachable = path is not None and path.is_file()
        origin: Mapping[str, JsonValue] = {}
        if path is not None and reachable:
            try:
                origin = read_revision_manifest(path).origin
            except (FileNotFoundError, ValueError):
                origin = {}
        found.append(
            ConsumedRevision(
                origin_uuid=origin_uuid,
                set_key=set_key,
                revision=revision,
                digest=digest,
                path=path,
                reachable=reachable,
                origin=origin,
            )
        )
    return tuple(found)


def training_provenance(ds: Dataset, kind: str, run_id: str) -> TrainingProvenance:
    """What the model *run_id* was trained on, walked back as far as disk allows.

    Works from the dataset that trained the model and from any dataset that links
    it, because the rows that answer live with the model.

    Args:
        ds: The dataset asking.
        kind: The training op, such as ``"train-pose"``.
        run_id: The model.

    Raises:
        KeyError: No such run is registered here or in a linked library.
    """
    holder, served_by, row = _holder(ds, kind, run_id)
    data_path = row.get("data_path", "")
    base = TrainingProvenance(
        kind=kind,
        run_id=run_id,
        served_by=served_by,
        data_path=data_path,
        data_fingerprint=row.get("data_fingerprint", ""),
        base_run_id=row.get("base_run_id", ""),
    )
    if not data_path:
        return _stopped(base, "the run recorded no data path; it predates that record")

    prepared_kind, prepared_run_id = _prepared_run(data_path)
    if not prepared_run_id:
        return _stopped(
            base,
            "the data was not written by a preparation run, so nothing records "
            "which annotations it holds",
        )
    prepared = _row(model_index_path(holder, prepared_kind), prepared_run_id)
    if prepared is None:
        return _stopped(
            base,
            f"{prepared_kind} run {prepared_run_id} has no index row",
            prepared_kind=prepared_kind,
            prepared_run_id=prepared_run_id,
        )
    sets = _consumed(holder, prepared.get("consumed_sets", ""))
    unreachable = [item for item in sets if not item.reachable]
    reason = ""
    if not sets:
        reason = f"{prepared_kind} run {prepared_run_id} records no annotation sets"
    elif unreachable:
        named = ", ".join(f"{i.set_key} rev{i.revision}" for i in unreachable)
        reason = (
            f"{len(unreachable)} revision(s) are not on disk ({named}); the "
            "dataset they were saved in may be archived or unmounted. The prepared "
            "data still holds a copy of their labels"
        )
    return TrainingProvenance(
        kind=kind,
        run_id=run_id,
        served_by=served_by,
        data_path=data_path,
        data_fingerprint=base.data_fingerprint,
        base_run_id=base.base_run_id,
        prepared_kind=prepared_kind,
        prepared_run_id=prepared_run_id,
        sets=sets,
        stopped_at=reason,
    )


def _stopped(
    base: TrainingProvenance,
    reason: str,
    *,
    prepared_kind: str = "",
    prepared_run_id: str = "",
) -> TrainingProvenance:
    return TrainingProvenance(
        kind=base.kind,
        run_id=base.run_id,
        served_by=base.served_by,
        data_path=base.data_path,
        data_fingerprint=base.data_fingerprint,
        base_run_id=base.base_run_id,
        prepared_kind=prepared_kind,
        prepared_run_id=prepared_run_id,
        stopped_at=reason,
    )
