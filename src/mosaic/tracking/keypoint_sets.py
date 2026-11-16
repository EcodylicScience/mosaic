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
from mosaic.core.pipeline.label_series_index import (
    read_revision_manifest,
    resolve_series_refs,
)
from mosaic.core.pipeline.label_series import LabelRevisionRef

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
    the digest is taken from the bytes now and compared with the index row's, so
    a revision that was edited in place cannot mint the identifier of the state
    it used to hold.

    Raises:
        KeyError: A reference matches several datasets' sets and names none.
        ValueError: Two references resolve to one set. Two revisions of a set
            name the same images, so both would be placed under one name.
        IdentityDeferred: A reference matches no indexed revision, or the
            revision it matches is not on disk. Both are states a scan or a save
            can still change, which is what deferring means.
        LabelSeriesTamperedError: A revision's bytes no longer match the digest
            its index row recorded.
    """
    revisions = resolve_series_refs(
        ds,
        KEYPOINTS_SERIES,
        [
            LabelRevisionRef(
                key=ref.set_key, origin_uuid=ref.origin_uuid, revision=ref.revision
            )
            for ref in refs
        ],
        kind=kind,
    )
    resolved: list[ResolvedSet] = []
    for revision in revisions:
        if any(
            (item.origin_uuid, item.set_key) == (revision.origin_uuid, revision.key)
            for item in resolved
        ):
            msg = (
                f"keypoint set {revision.key!r} is named twice; one dataset reads one "
                "revision of a set, because two revisions show the same images"
            )
            raise ValueError(msg)
        resolved.append(
            ResolvedSet(
                origin_uuid=revision.origin_uuid,
                set_key=revision.key,
                revision=revision.revision,
                digest=revision.digest,
                payload=revision.payload,
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
