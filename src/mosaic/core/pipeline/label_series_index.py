"""Writing, indexing and reading the revisions of a label series.

The registry, the layout and the rule a series follows are in
:mod:`mosaic.core.pipeline.label_series`. This module is the half that touches an
index: the typed row, the writer that mints a revision, and the reader that turns
a revision on disk back into a row.

**mosaic owns the layout; a caller hands it content.** Whoever authored the state
-- the annotator through mosaic-api, a notebook -- supplies bytes and a free-form
``origin`` mapping. It chooses no filename and no revision number, so there is one
place that knows how a series is laid out, and one place where two saves landing
together are serialized.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mosaic.core.helpers import validate_entry_name
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.file_digest import MODEL_DIGEST_HEX, file_digest
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    IndexRowBase,
    index_records,
    project_to_schema,
)
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.strict_model import terse
from mosaic.core.pipeline.label_series import (
    MANIFEST_FILENAME,
    SERIES_MARKER,
    SeriesSpec,
    revision_dirname,
    revision_of,
    series_spec,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "LABEL_SERIES_INDEX_COLUMNS",
    "LabelSeriesCollisionError",
    "LabelSeriesIndexRow",
    "LabelSeriesTamperedError",
    "RevisionManifest",
    "SeriesRevision",
    "carried_series_digests",
    "ensure_series_root",
    "label_series_index",
    "latest_revision",
    "payload_digest",
    "read_label_series",
    "read_revision_manifest",
    "row_for_revision_file",
    "series_index_path",
    "series_root",
    "write_label_series_rows",
    "write_series_revision",
]

LABEL_SERIES_ROOT: Final = "labels_raw"
"""The root every series lives under. A series is a kind of authored label."""


_BOOKKEEPING: Final = frozenset({"index.csv", "index.csv.lock"})
"""Files a series directory holds that are mosaic's own and not a squatter's."""


class LabelSeriesCollisionError(ValueError):
    """``labels_raw/<series>/`` exists and is not a series.

    Somebody's folder already has the name. Adopting it would put revisions
    beside files of another meaning and mark the lot as mosaic-managed.
    """


class LabelSeriesTamperedError(ValueError):
    """A revision's bytes no longer match the digest recorded when it was written.

    A revision is immutable: a model may name it as what it was trained on. One
    that changed afterwards is refused rather than re-indexed under its new
    digest, because re-indexing would quietly rewrite what that model saw.
    """


@dataclass(frozen=True, slots=True)
class LabelSeriesIndexRow(IndexRowBase):
    """One revision of one key of one series.

    ``abs_path`` is the payload file, root-relative when it lies inside the
    dataset and absolute when a source claimed it from another one -- the same
    rule the raw indexes follow.

    Attributes:
        series: Which series. Constant within one index file.
        key: What the revision is of: a set key, or an entry key.
        revision: Counted from one, per ``(origin_uuid, key)``.
        digest: Content digest of the payload. What a consumer's identity is
            taken over; the revision number never is.
        size_bytes: Payload size.
        mtime_iso: Payload modification time, for the digest carry-forward.
        n_records: How many records the payload holds: frames for keypoints.
        exported_at: When the revision was written, ISO-8601 UTC.
        origin_uuid: The manifest ``uuid`` of the dataset the revision was
            written in. Survives that dataset's directory moving, and tells two
            datasets' same-named keys apart in a library that claims both.
        origin_ref: A short, caller-supplied pointer to the authoring store's
            version, such as ``dolt:1a2b3c4d5e``. The full mapping is in the
            revision's manifest.
        source_id: The scan source that claimed this row, or ``""`` for a row
            the writer appended in its own dataset.
    """

    series: str
    key: str
    revision: int
    digest: str
    size_bytes: int
    mtime_iso: str
    n_records: int
    exported_at: str
    origin_uuid: str
    origin_ref: str
    source_id: str = ""


LABEL_SERIES_INDEX_COLUMNS: Final[tuple[str, ...]] = (
    "abs_path",
    "series",
    "key",
    "revision",
    "digest",
    "size_bytes",
    "mtime_iso",
    "n_records",
    "exported_at",
    "origin_uuid",
    "origin_ref",
    "source_id",
)


def label_series_index(path: Path) -> IndexCSV[LabelSeriesIndexRow]:
    """The typed index at *path*. One row per ``(origin_uuid, key, revision)``."""
    return IndexCSV(
        path,
        LabelSeriesIndexRow,
        dedup_keys=["origin_uuid", "key", "revision"],
    )


def series_root(ds: Dataset, series: str) -> Path:
    """``labels_raw/<series>/`` in *ds*. Validates that *series* is declared."""
    return ds.get_root(LABEL_SERIES_ROOT) / series_spec(series).name


def series_index_path(ds: Dataset, series: str) -> Path:
    """``labels_raw/<series>/index.csv`` in *ds*."""
    return series_root(ds, series) / "index.csv"


def payload_digest(payload: bytes) -> str:
    """The digest :func:`file_digest` would give a file holding *payload*.

    Computed from bytes so a save that changed nothing is recognized before
    anything is written, and spelled to agree with ``file_digest`` so the value
    recorded at write time is the one a consumer later measures off disk.
    """
    return hashlib.blake2b(payload, digest_size=MODEL_DIGEST_HEX // 2).hexdigest()


@dataclass(frozen=True, slots=True)
class SeriesRevision:
    """A revision, as the writer reports it.

    Attributes:
        series: Which series.
        key: What it is a revision of.
        revision: Its number.
        digest: Content digest of the payload.
        path: The payload file.
        written: ``False`` when the state matched the latest revision and nothing
            was written. The revision named is then the existing one.
    """

    series: str
    key: str
    revision: int
    digest: str
    path: Path
    written: bool


class _RevisionManifestFile(BaseModel):
    """The shape of ``manifest.json`` on disk, for a typed read.

    Tolerant of keys it does not model, so a newer writer's additions do not make
    an older reader refuse a revision it can otherwise use.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    series: str = ""
    key: str = ""
    revision: int = 0
    digest: str = ""
    n_records: int = 0
    exported_at: str = ""
    dataset_uuid: str = ""
    origin_ref: str = ""
    image_root: str = ""
    origin: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RevisionManifest:
    """What a revision's ``manifest.json`` records, read back."""

    series: str
    key: str
    revision: int
    digest: str
    n_records: int
    exported_at: str
    dataset_uuid: str
    origin_ref: str
    image_root: str
    origin: Mapping[str, JsonValue]


def ensure_series_root(root: Path, spec: SeriesSpec) -> None:
    """Create ``labels_raw/<series>/`` and mark it, or refuse a squatter.

    Called by both things that create a series directory: the writer, on a
    dataset's first save, and the scan, on a dataset that only claims other
    datasets' revisions and never saves one itself. Unmarked, that second
    directory would be invisible to the index enumeration and unprotected from
    the per-sequence scan.
    """
    marker = root / SERIES_MARKER
    if marker.is_file():
        return
    # Only somebody else's files count as squatting. Two first saves can land
    # together, and the second then finds a directory holding the first one's
    # lock sidecar, its half-written marker, or an index it has just created --
    # all mosaic's own, and all present a moment before the marker is. Judging
    # those as a squatter refused a perfectly good save. The marker is written
    # before any key directory is, so anything else found here without one was
    # not put there by a series writer.
    if root.exists():
        foreign = sorted(
            child.name
            for child in root.iterdir()
            if not child.name.startswith(".") and child.name not in _BOOKKEEPING
        )
        if foreign:
            msg = (
                f"{root} already exists and is not a label series: it holds "
                f"{foreign[:3]} and no {SERIES_MARKER} marker. The {spec.name!r} "
                "series needs that directory name. Move what is there, then save "
                "again."
            )
            raise LabelSeriesCollisionError(msg)
    root.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {"series": spec.name, "unit": spec.unit, "format": spec.format},
        indent=2,
        sort_keys=True,
    )
    atomic_write(marker, lambda tmp: tmp.write_text(body + "\n", encoding="utf-8"))


def _records(
    frame: pd.DataFrame, *, origin_uuid: str, key: str
) -> list[dict[str, str]]:
    return [
        record
        for record in index_records(frame)
        if record.get("key", "") == key and record.get("origin_uuid", "") == origin_uuid
    ]


def _int_cell(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def latest_revision(
    frame: pd.DataFrame, *, origin_uuid: str, key: str
) -> dict[str, str] | None:
    """The highest-numbered row of ``(origin_uuid, key)`` in *frame*, or ``None``."""
    matching = _records(frame, origin_uuid=origin_uuid, key=key)
    if not matching:
        return None
    return max(matching, key=lambda record: _int_cell(record.get("revision", "")))


def _highest_on_disk(key_root: Path) -> int:
    """The highest ``rev<N>`` directory under *key_root*, or 0.

    Consulted beside the index so a number is never reused. A revision directory
    whose row was never appended -- a crash between the two writes -- still
    occupies its number, and handing that number out again would put two
    different states behind one name.
    """
    if not key_root.is_dir():
        return 0
    return max((revision_of(child.name) for child in key_root.iterdir()), default=0)


def write_series_revision(
    ds: Dataset,
    *,
    series: str,
    key: str,
    payload: bytes,
    origin: Mapping[str, JsonValue],
    n_records: int,
    origin_ref: str = "",
) -> SeriesRevision:
    """Save *payload* as the next revision of *key*, unless nothing changed.

    Safe to call every time an editor is closed. The digest of *payload* is
    compared with the latest revision's: a save that changed nothing writes
    nothing and returns the revision already there. Otherwise the next number is
    taken, the revision directory is created exclusively, and the index row is
    appended -- all inside the series index lock, so two saves landing together
    never share a number and neither loses its row.

    A revision is never overwritten. *payload* must be deterministic for a given
    state: a timestamp inside it would make every save look like a change.
    Timestamps belong in *origin*, which is recorded beside the payload.

    Args:
        ds: The dataset to write into.
        series: A declared series, such as ``"keypoints"``.
        key: What this is a revision of. One path component.
        payload: The exact bytes to store.
        origin: Free-form provenance from the authoring store, recorded verbatim
            in the revision's manifest. Where a database commit belongs.
        n_records: How many records *payload* holds, for the index.
        origin_ref: A short pointer into *origin* for the index row.

    Returns:
        The revision, and whether this call wrote it.

    Raises:
        KeyError: *series* is not declared.
        ValueError: *key* is empty or not one path component.
        LabelSeriesCollisionError: The series directory name is taken.
    """
    spec = series_spec(series)
    if not key or key in {".", ".."} or revision_of(key):
        msg = f"{key!r} cannot be a {spec.name} key: it has to name one directory"
        raise ValueError(msg)
    _ = validate_entry_name(key, f"{spec.name} key")

    root = series_root(ds, spec.name)
    ensure_series_root(root, spec)
    digest = payload_digest(payload)
    origin_uuid = ds.uuid or ""
    key_root = root / key

    index = label_series_index(series_index_path(ds, spec.name))
    index.ensure()
    with index_lock(index.path):
        committed = index.read_holding_lock()
        latest = latest_revision(committed, origin_uuid=origin_uuid, key=key)
        if latest is not None and latest.get("digest", "") == digest:
            existing = ds.resolve_path(latest["abs_path"])
            if existing.is_file():
                return SeriesRevision(
                    series=spec.name,
                    key=key,
                    revision=_int_cell(latest.get("revision", "")),
                    digest=digest,
                    path=existing,
                    written=False,
                )

        indexed = _int_cell(latest.get("revision", "")) if latest is not None else 0
        revision = max(indexed, _highest_on_disk(key_root)) + 1
        revision_root = key_root / revision_dirname(revision)
        # Exclusive: an existing directory here means a number was handed out
        # twice, and writing into it would replace a revision something may name.
        revision_root.mkdir(parents=True, exist_ok=False)

        target = revision_root / spec.payload_filename
        atomic_write(target, lambda tmp: tmp.write_bytes(payload))
        exported_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        document: dict[str, JsonValue] = {
            "series": spec.name,
            "key": key,
            "revision": revision,
            "digest": digest,
            "format": spec.format,
            "payload": spec.payload_filename,
            "n_records": n_records,
            "exported_at": exported_at,
            "dataset_uuid": origin_uuid,
            "dataset_name": ds.name,
            "origin_ref": origin_ref,
            # Relative to this file, so the revision still finds its images after
            # the dataset moves, and a reader in another dataset can find them
            # without being told where this one is.
            "image_root": os.path.relpath(ds.base_dir, revision_root),
            "origin": dict(origin),
        }
        manifest_text = json.dumps(document, indent=2, sort_keys=True) + "\n"
        atomic_write(
            revision_root / MANIFEST_FILENAME,
            lambda tmp: tmp.write_text(manifest_text, encoding="utf-8"),
        )

        stat = target.stat()
        index.append_holding_lock(
            [
                LabelSeriesIndexRow(
                    abs_path=Path(ds.relative_to_root(target)),
                    series=spec.name,
                    key=key,
                    revision=revision,
                    digest=digest,
                    size_bytes=int(stat.st_size),
                    mtime_iso=_mtime_iso(stat.st_mtime),
                    n_records=n_records,
                    exported_at=exported_at,
                    origin_uuid=origin_uuid,
                    origin_ref=origin_ref,
                )
            ]
        )
    return SeriesRevision(
        series=spec.name,
        key=key,
        revision=revision,
        digest=digest,
        path=target,
        written=True,
    )


def _mtime_iso(mtime: float) -> str:
    return datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()


def read_revision_manifest(payload_path: Path) -> RevisionManifest:
    """The manifest beside *payload_path*.

    Raises:
        FileNotFoundError: No manifest is there, so this is not a revision the
            series writer made.
        ValueError: The manifest is not a JSON object.
    """
    manifest_path = payload_path.parent / MANIFEST_FILENAME
    if not manifest_path.is_file():
        msg = (
            f"{payload_path} has no {MANIFEST_FILENAME} beside it, so it is not a "
            "revision of a label series. A series file is written by "
            "write_series_revision, never placed by hand."
        )
        raise FileNotFoundError(msg)
    try:
        document = _RevisionManifestFile.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValidationError as exc:
        msg = f"{manifest_path} is not a revision manifest: {terse(exc)}"
        raise ValueError(msg) from exc
    return RevisionManifest(
        series=document.series,
        key=document.key,
        revision=document.revision,
        digest=document.digest,
        n_records=document.n_records,
        exported_at=document.exported_at,
        dataset_uuid=document.dataset_uuid,
        origin_ref=document.origin_ref,
        image_root=document.image_root,
        origin=document.origin,
    )


def row_for_revision_file(
    ds: Dataset,
    payload_path: Path,
    *,
    series: str,
    source_id: str,
    known_digest: str = "",
) -> LabelSeriesIndexRow:
    """The index row for a revision found on disk, in this dataset or another.

    The manifest says what the revision is; the file says whether it still is.
    The two are compared, and a payload whose bytes moved is refused.

    Args:
        ds: The dataset doing the indexing, which decides how the path is stored.
        payload_path: The revision's payload file.
        series: The series the claiming source declared.
        source_id: The source claiming it.
        known_digest: A digest carried forward from an earlier scan for a file
            whose size and mtime have not moved, so an unchanged revision is not
            re-read. Empty measures it.

    Raises:
        FileNotFoundError: *payload_path*, or its manifest, is not there.
        LabelSeriesTamperedError: The payload's digest disagrees with the manifest.
        ValueError: The manifest names a different series than the source.
    """
    spec = series_spec(series)
    manifest = read_revision_manifest(payload_path)
    if manifest.series != spec.name:
        msg = (
            f"{payload_path} is a revision of the {manifest.series!r} series, but "
            f"source {source_id!r} declares {spec.name!r}"
        )
        raise ValueError(msg)
    measured = known_digest or file_digest(payload_path)
    if measured != manifest.digest:
        msg = (
            f"{payload_path} was written with digest {manifest.digest} and now "
            f"measures {measured}. A revision is immutable -- a model may name it "
            "as what it was trained on -- so it is refused rather than re-indexed. "
            "Restore the file, or save the changed state as a new revision."
        )
        raise LabelSeriesTamperedError(msg)
    stat = payload_path.stat()
    return LabelSeriesIndexRow(
        abs_path=Path(ds.relative_to_root(payload_path)),
        series=spec.name,
        key=manifest.key,
        revision=manifest.revision,
        digest=manifest.digest,
        size_bytes=int(stat.st_size),
        mtime_iso=_mtime_iso(stat.st_mtime),
        n_records=manifest.n_records,
        exported_at=manifest.exported_at,
        origin_uuid=manifest.dataset_uuid,
        origin_ref=manifest.origin_ref,
        source_id=source_id,
    )


def write_label_series_rows(
    index_path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    """Rewrite a series index to exactly *rows*, atomically.

    The caller holds ``index_lock(index_path)``: a scan reads, decides what its
    claim replaces, and writes, and a save landing between the read and the write
    would otherwise lose its row. Projected onto the schema so column order is
    fixed, and sorted so the file is order-stable whatever order the rows were
    found in.
    """
    frame = project_to_schema(pd.DataFrame(list(rows)), LABEL_SERIES_INDEX_COLUMNS)
    if not frame.empty:
        ordering = frame.assign(_revision=frame["revision"].map(_as_int))
        frame = (
            ordering.sort_values(["origin_uuid", "key", "_revision"], kind="stable")
            .drop(columns="_revision")
            .reset_index(drop=True)
        )
    atomic_write(index_path, lambda tmp: frame.to_csv(tmp, index=False))


def _as_int(value: object) -> int:
    return _int_cell(str(value))


def carried_series_digests(ds: Dataset, series: str) -> dict[Path, str]:
    """Digests an earlier scan recorded, for revisions that have not moved since.

    Keyed on the resolved path, and admitted only when the stored size and
    modification time both still match the file. A revision is immutable, so a
    rescan of a library that claims hundreds of them should read none.

    A match on size and time is not proof the bytes are the same, and this is the
    cheap path by design: it spares the *unchanged* file a read. A file edited in
    place shows a new modification time, misses here, is measured, and is
    refused.
    """
    carried: dict[Path, str] = {}
    for record in index_records(read_label_series(ds, series)):
        stored = record.get("abs_path", "").strip()
        digest = record.get("digest", "").strip()
        if not stored or not digest:
            continue
        resolved = ds.resolve_path(stored).resolve()
        try:
            stat = resolved.stat()
        except OSError:
            continue
        if record.get("size_bytes", "") != str(stat.st_size):
            continue
        if record.get("mtime_iso", "") != _mtime_iso(stat.st_mtime):
            continue
        carried[resolved] = digest
    return carried


def read_label_series(ds: Dataset, series: str) -> pd.DataFrame:
    """Every indexed revision of *series* in *ds*. Absent reads as empty."""
    path = series_index_path(ds, series)
    if not path.exists():
        return pd.DataFrame(columns=list(LABEL_SERIES_INDEX_COLUMNS))
    return label_series_index(path).read()
