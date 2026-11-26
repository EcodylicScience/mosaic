"""The rebuilt copy of a Live Mouse Tracker database, and what describes it.

**The layout.** An external program rebuilds a copy of a database's events in
place, and the copy lives in the dataset under the ``rebuild-lmt-events`` root::

    _tracking/rebuild-lmt-events/
      index.csv                  one row per (run_id, entry)
      <run_id>/
        .identity_scheme         the identity scheme marker of the run
        <entry_key>/
          <source file name>     the rebuilt copy, under its source's name
          rebuild.json           the entry manifest (:class:`RebuildManifest`)
          rebuild.log            the rebuild's output
          rebuild-report.json    the rebuild's report
          .mosaic-rebuild.json   the ``rebuild`` phase marker

The copy keeps its source's file name, so a sequence derived from the stem is
the same from the copy as from the source.

**The match by content.** A converter finds the copy of a raw file by the file's
md5, not by its group or name: ``<run_id>/*/rebuild.json`` is read and the
manifest whose ``source_md5`` equals the md5 names the copy. A tracks row and a
labels row of one file, which may carry different groups or names, then find one
copy.

**Resolution.** :func:`rebuilt_copy` finds a run's copy of a raw file by the file's
md5 and refuses, naming the file and the run, a copy that cannot be trusted: one
whose rebuild did not finish, or that was written after its rebuild. The first
copy by entry key is read when several sources are identical, which keeps the
choice independent of the order of a directory listing.

This module reads the root through a lookup that maps a root key to the
dataset's root, and imports nothing from ``tracking``. The converters of the track
library import it, and importing the track library loads no part of
``core.pipeline``, so each function that needs the pipeline imports it when called.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Final, Literal

from pydantic import ConfigDict, ValidationError

from mosaic.core.strict_model import StrictModel, located

__all__ = [
    "REBUILD_KIND",
    "REBUILD_MANIFEST_NAME",
    "REBUILD_RUN_ID_DESCRIPTION",
    "RebuildManifest",
    "copy_unchanged",
    "mtime_seconds",
    "read_rebuild_manifest",
    "rebuilt_copy",
    "refuse_foreign_rebuild_run_id",
]

REBUILD_KIND: Final = "rebuild-lmt-events"
"""The op kind and the key of the root that holds the rebuilt copies."""

REBUILD_MANIFEST_NAME: Final = "rebuild.json"
"""The manifest beside a rebuilt copy."""

REBUILD_RUN_ID_DESCRIPTION: Final = (
    "A rebuild-lmt-events run whose copy of each file is read instead of it; "
    "unset reads the raw file."
)
"""The description of the ``rebuild_run_id`` param of both converters."""


class RebuildManifest(StrictModel):
    """What a rebuild recorded about one copy, written beside it.

    Attributes:
        manifest_format: The layout of this file.
        run_id: The run that made the copy.
        op_version: The version of the op that made it.
        lmt_analysis_commit: The commit of the checkout that rebuilt it.
        builders: The event builders that ran, in order.
        source_md5: The md5 of the raw file the copy was made from.
        source_name: The raw file's name.
        source_size: The raw file's size in bytes.
        copy_name: The copy's file name.
        copy_size: The copy's size in bytes when it was published.
        copy_mtime_ns: The copy's modification time when it was published.
        max_frame: The last frame the rebuild read.
        script_max_frame: The last frame the rebuild script used by default.
        python: The interpreter version that ran the rebuild.
        packages: The versions of the packages the rebuild imported.
        completed_at: When the rebuild finished, as an ISO 8601 instant.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    manifest_format: Literal[1]
    run_id: str
    op_version: str
    lmt_analysis_commit: str
    builders: tuple[str, ...]
    source_md5: str
    source_name: str
    source_size: int
    copy_name: str
    copy_size: int
    copy_mtime_ns: int
    max_frame: int
    script_max_frame: int
    python: str
    packages: dict[str, str]
    completed_at: str

    def write(self, entry_dir: Path) -> None:
        """Write this manifest to ``entry_dir / REBUILD_MANIFEST_NAME`` atomically."""
        from mosaic.core.pipeline._utils import atomic_write

        atomic_write(
            entry_dir / REBUILD_MANIFEST_NAME,
            lambda tmp: tmp.write_text(self.model_dump_json(indent=2)),
        )


def read_rebuild_manifest(copy: Path) -> RebuildManifest:
    """Read the manifest beside *copy*.

    Raises:
        ValueError: If the manifest is missing or does not validate, naming the
            file.
    """
    return _read_manifest(copy.parent / REBUILD_MANIFEST_NAME)


def _read_manifest(path: Path) -> RebuildManifest:
    """Read the manifest at *path*, raising a ``ValueError`` that names it."""
    try:
        text = path.read_text()
    except OSError as exc:
        raise ValueError(f"{path}: the rebuild manifest cannot be read: {exc}") from exc
    try:
        return RebuildManifest.model_validate_json(text)
    except ValidationError as exc:
        raise ValueError(f"{path}: not a rebuild manifest: {located(exc)}") from exc


def mtime_seconds(path: Path) -> int:
    """The modification time of *path* in whole seconds."""
    return path.stat().st_mtime_ns // 10**9


def copy_unchanged(copy: Path, manifest: RebuildManifest) -> bool:
    """Whether *copy* has the size and modification time its manifest recorded.

    The time is compared in whole seconds. A CIFS server stores 100 ns ticks and
    an NFS server its filesystem's precision, and a client can report a finer
    value from its cache than another host reads back from the server. Size and
    whole seconds still catch a copy written after its rebuild, which is the case
    this check guards.
    """
    return (
        copy.stat().st_size == manifest.copy_size
        and mtime_seconds(copy) == manifest.copy_mtime_ns // 10**9
    )


def refuse_foreign_rebuild_run_id(value: str | None) -> str | None:
    """Return *value* when it is unset or the id of a ``rebuild-lmt-events`` run.

    Raises:
        ValueError: *value* is set and is not the id of a ``rebuild-lmt-events``
            run, naming it.
    """
    from mosaic.core.pipeline.op_identity import parse_op_run_id

    if value is None:
        return None
    parsed = parse_op_run_id(value)
    if parsed is None or parsed.kind != REBUILD_KIND:
        raise ValueError(f"{value!r} is not the id of a {REBUILD_KIND} run")
    return value


def rebuilt_copy(root: Path | None, run_id: str, source_md5: str, file: str) -> Path:
    """The copy that run *run_id* made of the raw file with md5 *source_md5*.

    Reads ``<root>/<run_id>/*/rebuild.json`` in the order of the entry keys and
    takes the first manifest whose ``source_md5`` is *source_md5*.

    Args:
        root: The dataset's ``rebuild-lmt-events`` root, ``None`` when it
            declares none.
        run_id: The run whose copy is wanted.
        source_md5: The ``md5`` cell of the raw file's index row.
        file: The raw file, for an error message.

    Raises:
        ValueError: The raw file has no md5, the dataset has no root or the run
            has no directory, the run has no copy of a file with that md5, the
            copy's rebuild did not finish, or the copy was written after its
            rebuild. The message names *file* and *run_id*.
    """

    from mosaic.core.pipeline.markers import read_phase_marker

    def refuse(reason: str) -> ValueError:
        return ValueError(f"{file}: cannot read the copy of run {run_id}: {reason}")

    if not source_md5:
        raise refuse(
            "the index holds no md5 for the file; run `mosaic scan` to hash it"
        )
    if root is None:
        raise refuse(
            f"the dataset has no rebuild-lmt-events root; the {REBUILD_KIND} op "
            "creates it"
        )
    run_dir = root / run_id
    if not run_dir.is_dir():
        raise refuse(
            f"the rebuild-lmt-events root has no directory for the run; the "
            f"{REBUILD_KIND} op makes it"
        )
    manifests = sorted(run_dir.glob(f"*/{REBUILD_MANIFEST_NAME}"))
    for path in manifests:
        try:
            manifest = _read_manifest(path)
        except ValueError as exc:
            raise refuse(str(exc)) from exc
        if manifest.source_md5 != source_md5:
            continue
        entry_dir = path.parent
        if read_phase_marker(entry_dir, "rebuild") is None:
            raise refuse(f"the rebuild in {entry_dir.name} did not finish")
        copy = entry_dir / manifest.copy_name
        if not copy.is_file():
            raise refuse(f"{copy} is missing")
        if not copy_unchanged(copy, manifest):
            raise refuse(f"{copy} was written after its rebuild")
        return copy
    count = len(manifests)
    raise refuse(
        f"the run has no copy of a file with md5 {source_md5} among its {count} "
        f"cop{'y' if count == 1 else 'ies'}. A file added or changed after the run "
        f"needs the {REBUILD_KIND} op again, under a new run id"
    )
