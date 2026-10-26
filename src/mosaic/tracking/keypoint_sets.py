"""Keypoint-annotation revisions: which saved state a reference names, and its reader.

The annotator's state reaches a dataset as revisions of the ``keypoints`` label
series. An op that consumes one names it by a :class:`KeypointSetRef`, resolves it
to a file whose digest is measured off disk, and reads it back through
:func:`read_resolved_set`, which is the one place the image root is applied.

An op that reads several revisions orders them by :func:`in_identity_order`,
names each in an error by :func:`set_label`, and records them in an index cell
through :func:`consumed_sets_cell`. A preparation of those revisions records the
keypoints that a model trained on it predicts. :func:`keypoint_schema_cell` writes
that index cell, and :func:`read_keypoint_schema_cell` reads it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.annotations.pose_annotations import (
    PoseAnnotationSet,
    read_pose_annotations,
)
from mosaic.core.annotations.projection import KEYPOINTS_SERIES
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    read_revision_manifest,
)
from mosaic.core.pipeline.ops import IdentityDeferred

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "KeypointSetRef",
    "ResolvedSet",
    "consumed_sets_cell",
    "in_identity_order",
    "keypoint_schema_cell",
    "read_keypoint_schema_cell",
    "read_resolved_set",
    "resolve_keypoint_sets",
    "revision_image_root",
    "set_label",
]


class KeypointSetRef(Params):
    """One annotation set, and which saved state of it to read."""

    set_key: Annotated[str, Declared("The set's key in the keypoints label series.")]
    origin_uuid: Annotated[
        str,
        Declared(
            "The manifest uuid of the dataset the set was saved in. Needed only "
            "when two datasets' sets share a key; empty matches the one that has it."
        ),
    ] = ""
    revision: Annotated[
        int | None,
        Field(ge=1),
        Declared(
            "Which revision to read. Left unset, the latest one indexed when the "
            "run is named. The number selects and is recorded; only the content "
            "it resolves to enters the run identifier."
        ),
    ] = None


@dataclass(frozen=True, slots=True)
class ResolvedSet:
    """A set reference, resolved to one revision on disk."""

    origin_uuid: str
    set_key: str
    revision: int
    digest: str
    payload: Path

    def consumed(self) -> dict[str, str | int]:
        return {
            "origin_uuid": self.origin_uuid,
            "set_key": self.set_key,
            "revision": self.revision,
            "digest": self.digest,
        }


def resolve_keypoint_sets(
    ds: Dataset, refs: tuple[KeypointSetRef, ...], *, kind: str
) -> tuple[ResolvedSet, ...]:
    """Each reference as one revision file, with its digest measured off disk.

    The index says which file a reference means. The file says what is in it:
    the digest is taken from the bytes now, never read from the index column, so
    a revision that was edited in place cannot mint the identifier of the state
    it used to hold.

    Raises:
        KeyError: A reference matches several datasets' sets and names none.
        ValueError: Two references resolve to one set. Two revisions of a set
            name the same images, so both would be placed under one name.
        IdentityDeferred: A reference matches no indexed revision, or the
            revision it matches is not on disk. Both are states a scan or a save
            can still change, which is what deferring means.
    """
    records = index_records(read_label_series(ds, KEYPOINTS_SERIES))
    resolved: list[ResolvedSet] = []
    for ref in refs:
        matching = [
            record
            for record in records
            if record.get("key", "") == ref.set_key
            and (
                not ref.origin_uuid or record.get("origin_uuid", "") == ref.origin_uuid
            )
        ]
        if not matching:
            known = sorted({record.get("key", "") for record in records})
            origin = f" from dataset {ref.origin_uuid}" if ref.origin_uuid else ""
            # Deferred, like data that is not on disk yet: the revision may be
            # claimed by a scan that has not run. The reason names the repair,
            # because at execution this is what the failure will say.
            raise IdentityDeferred(
                kind,
                f"no revision of keypoint set {ref.set_key!r}{origin} is indexed "
                f"here (indexed sets: {known or 'none'}). A set saved in another "
                "dataset has to be claimed first: declare a labels source with "
                "series='keypoints' naming the revision, then scan",
            )
        origins = sorted({record.get("origin_uuid", "") for record in matching})
        if len(origins) > 1:
            msg = (
                f"keypoint set {ref.set_key!r} is indexed from {len(origins)} "
                f"datasets ({', '.join(origins)}); name one with origin_uuid"
            )
            raise KeyError(msg)
        if ref.revision is None:
            chosen = max(matching, key=lambda record: int(record.get("revision", "0")))
        else:
            wanted = [
                record
                for record in matching
                if int(record.get("revision", "0")) == ref.revision
            ]
            if not wanted:
                held = sorted(int(record.get("revision", "0")) for record in matching)
                raise IdentityDeferred(
                    kind,
                    f"revision {ref.revision} of keypoint set {ref.set_key!r} is "
                    f"not indexed here (indexed: {held}); claim it and scan",
                )
            chosen = wanted[0]
        payload = ds.resolve_path(chosen["abs_path"])
        if not payload.is_file():
            raise IdentityDeferred(
                kind,
                f"revision {chosen.get('revision', '')} of keypoint set "
                f"{ref.set_key!r} ({payload}) is not on disk, and this run's "
                "identity covers the content of what it reads",
            )
        named = [
            item
            for item in resolved
            if (item.origin_uuid, item.set_key) == (origins[0], ref.set_key)
        ]
        if named:
            msg = (
                f"keypoint set {ref.set_key!r} is named twice; one dataset reads one "
                "revision of a set, because two revisions show the same images"
            )
            raise ValueError(msg)
        resolved.append(
            ResolvedSet(
                origin_uuid=origins[0],
                set_key=ref.set_key,
                revision=int(chosen.get("revision", "0")),
                digest=file_digest(payload),
                payload=payload,
            )
        )
    return tuple(resolved)


def in_identity_order(sets: Sequence[ResolvedSet]) -> list[ResolvedSet]:
    """*sets* in the order a run identifier sorts them."""
    return sorted(sets, key=lambda item: (item.origin_uuid, item.set_key, item.digest))


def set_label(item: ResolvedSet) -> str:
    """How a set is named in an error: which dataset, which set, which revision."""
    return f"{item.origin_uuid or 'local'}:{item.set_key}@rev{item.revision}"


def consumed_sets_cell(sets: Sequence[ResolvedSet]) -> str:
    """The JSON list that names revisions in an index cell, in identity order."""
    return json.dumps(
        sorted(
            (item.consumed() for item in sets),
            key=lambda entry: (str(entry["origin_uuid"]), str(entry["set_key"])),
        )
    )


def read_resolved_set(item: ResolvedSet) -> PoseAnnotationSet:
    """One revision's saved state, its images anchored where they are.

    A relative image path is relative to the manifest's ``image_root``, which is
    itself relative to the revision's directory.
    """
    manifest = read_revision_manifest(item.payload)
    image_root = (item.payload.parent / (manifest.image_root or ".")).resolve()
    return read_pose_annotations(item.payload, image_root)


def revision_image_root(root: Path | None) -> Path:
    """Return the image root of a set read from a revision.

    :func:`read_resolved_set` always applies one, and narrowing keeps it.

    Raises:
        ValueError: *root* is ``None``, which no set read from a revision has.
    """
    if root is None:
        msg = "annotation set lacks an image root. A revision's set is read with one"
        raise ValueError(msg)
    return root


class _KeypointSchemaCell(BaseModel):
    """A prepared-data row's ``keypoint_schema`` cell."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    keypoints: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...] = ()
    symmetries: tuple[tuple[int, int], ...] = ()


def keypoint_schema_cell(schema: KeypointSchema) -> str:
    """Return the ``keypoint_schema`` cell that records *schema*, as JSON."""
    cell = _KeypointSchemaCell(
        keypoints=schema.names, skeleton=schema.skeleton, symmetries=schema.symmetries
    )
    return json.dumps(cell.model_dump(mode="json"))


def read_keypoint_schema_cell(cell: str) -> KeypointSchema:
    """Return the schema that a ``keypoint_schema`` cell records.

    Raises:
        ValueError: The cell is not that JSON, or its pairs are not a schema's.
    """
    recorded = _KeypointSchemaCell.model_validate_json(cell)
    return KeypointSchema(
        names=recorded.keypoints,
        skeleton=recorded.skeleton,
        symmetries=recorded.symmetries,
    )
