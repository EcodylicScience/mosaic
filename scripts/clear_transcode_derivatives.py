"""Clear a dataset's transcode derivatives: the links, the rows, and the files.

Derivatives are addressed by the source video's identity and the recipe that
produced them, under a transcode kind directory. A dataset transcoded under the
earlier positional scheme holds files no current name resolves to, and clearing
them is the whole migration: the transcode job is idempotent and skippable, so a
re-run rebuilds whatever is wanted rather than repairing what is there.

Scope, and the two things it must not do. The originals index keeps every row
and loses only its two forward-link cells. The derivative index -- a different
file from the originals index only when a distinct ``media_raw`` root holds the
originals -- loses all of its rows. On a single-root dataset the two are one
file and the whole pass is a no-op: clearing there would strip curated
``(group, sequence)`` and ``video_order`` assignments and delete files the
surviving rows point at. The pass touches only the ``transcode/`` kind
directory and the files the derivative rows name; every other child of the media
root is left alone -- ``frames/`` (annotated frame sets, whose extraction
identity is persisted outside this dataset) and the crop visualizers'
``interaction_crops/`` and ``egocentric_crops/`` among them.

Dry run unless ``--apply``. A timestamped backup precedes any write, matching
the re-probe command, because this rewrites a production index in place.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS, read_link_cell
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)
from mosaic.core.pipeline.transcode import TRANSCODE_KIND_DIRECTORY

_LINK_COLUMNS = ("analysis_derivative_path", "playback_derivative_path")


@dataclass(frozen=True)
class ClearReport:
    """What the sweep found, and what it did about it."""

    # considered is False when the dataset has no distinct derivative index, so
    # the pass declined to look at all -- reported apart from a dry run, which
    # looked and would act.
    considered: bool
    applied: bool
    links_cleared: int
    rows_removed: int
    files_removed: int


def clear_transcode_derivatives(dataset: Dataset, *, apply: bool) -> ClearReport:
    """Clear every transcode derivative the dataset records or holds."""
    # No distinct derivative index means no derivatives to clear, and the media
    # index is the originals index -- touching it would destroy curated data.
    if dataset.resolve_media_root() != "media_raw":
        return ClearReport(
            considered=False,
            applied=False,
            links_cleared=0,
            rows_removed=0,
            files_removed=0,
        )

    originals_index = dataset.get_root("media_raw") / "index.csv"
    originals = read_media_index(originals_index) if originals_index.exists() else []

    links_cleared = 0
    cleared_rows: list[dict[str, object]] = []
    for record in originals:
        row: dict[str, object] = {
            column: record.get(column, "") for column in MEDIA_INDEX_COLUMNS
        }
        for column in _LINK_COLUMNS:
            if read_link_cell(row, column):
                row[column] = ""
                links_cleared += 1
        cleared_rows.append(row)

    media_root = dataset.get_root("media")
    derivative_index = media_root / "index.csv"
    derivative_rows = (
        read_media_index(derivative_index) if derivative_index.exists() else []
    )
    files = _derivative_files(dataset, derivative_rows, media_root)
    kind_directory = media_root / TRANSCODE_KIND_DIRECTORY

    if apply:
        if links_cleared:
            _back_up(originals_index)
            write_media_index_rows(originals_index, frame_from_rows(cleared_rows))
        if derivative_rows:
            _back_up(derivative_index)
            write_media_index_rows(derivative_index, frame_from_rows([]))
        for path in files:
            path.unlink(missing_ok=True)
        if kind_directory.exists():
            shutil.rmtree(kind_directory)

    return ClearReport(
        considered=True,
        applied=apply,
        links_cleared=links_cleared,
        rows_removed=len(derivative_rows),
        files_removed=len(files),
    )


def _back_up(index_path: Path) -> None:
    """Copy an index beside itself before rewriting it, as the re-probe does."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # with_name, not with_suffix: with_suffix REPLACES the extension, turning
    # index.csv into index.<stamp>.backup and leaving two backup schemes in one
    # media root. The re-probe writes index.csv.<stamp>.backup.
    _ = shutil.copy2(
        index_path, index_path.with_name(f"{index_path.name}.{stamp}.backup")
    )


def _derivative_files(
    dataset: Dataset, derivative_rows: list[dict[str, str]], media_root: Path
) -> list[Path]:
    """The files the derivative rows name, restricted to the media root.

    A row pointing outside the media root is left alone rather than followed:
    the sweep removes what this dataset produced, not whatever a stray absolute
    path happens to address.
    """
    resolved_media_root = media_root.resolve()
    files: list[Path] = []
    for row in derivative_rows:
        stored = read_link_cell(row, "abs_path")
        if not stored:
            continue
        path = dataset.resolve_path(stored).resolve()
        try:
            _ = path.relative_to(resolved_media_root)
        except ValueError:
            continue
        if path.exists():
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("manifest", type=Path, help="path to dataset.yaml")
    _ = parser.add_argument("--apply", action="store_true", help="write the changes")
    arguments = parser.parse_args()
    dataset = Dataset(manifest_path=arguments.manifest).load()
    report = clear_transcode_derivatives(dataset, apply=arguments.apply)
    if not report.considered:
        # Distinct from a dry run: nothing will ever be cleared here, and a
        # "would clear" line would read as "run it again with --apply".
        print("no distinct derivative index; nothing to clear")
        return
    state = "cleared" if report.applied else "would clear"
    summary = (
        f"{state}: {report.links_cleared} forward link(s), "
        f"{report.rows_removed} derivative row(s), {report.files_removed} file(s)"
    )
    print(summary)


if __name__ == "__main__":
    main()
