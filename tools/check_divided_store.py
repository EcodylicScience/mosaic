#!/usr/bin/env python
"""Check the division of one imgstore against the store itself, reading it only.

Run against a real store before relying on its division:

    python tools/check_divided_store.py STORE --frames 60000 --scratch DIR \\
        --expect 0-5=20210219T192253Z --expect 6-11=20210219T201253Z \\
        --expect-held 12-12

The store is divided twice into fresh datasets under ``DIR``: through a division
``MediaIndexScope`` and through a media scan of a dividing source. The script
asserts that:

- the two paths write the same rows, apart from ``assignment_source``;
- each division covers the chunks and has the name that ``--expect`` gives, and
  ``--expect-held`` names the sequence held back;
- the frame on each side of every boundary between two divisions reads, through
  its division's window, pixel for pixel as the store's own frame;
- the export of the second division holds its frames at the store's rate;
- a listing of the store directory is the same before and after.

The datasets reference the store by its path and write nothing beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Expected:
    """A division the store should make: its chunks and its name."""

    chunk_first: int
    chunk_last: int
    sequence: str


def _chunks(text: str) -> tuple[int, int]:
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def _expected(text: str) -> Expected:
    chunks, _, sequence = text.partition("=")
    first, last = _chunks(chunks)
    return Expected(first, last, sequence)


def listing_digest(store: Path) -> str:
    """The md5 of a long listing of *store*, which a write beside it would change."""
    listed = subprocess.run(
        ["ls", "-laT", str(store)], check=True, capture_output=True
    ).stdout
    return hashlib.md5(listed).hexdigest()


def _without_source(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {key: value for key, value in row.items() if key != "assignment_source"}
        for row in rows
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Divide the store twice, check the divisions, and print a report."""
    parser = argparse.ArgumentParser(
        description="Check the division of one imgstore against the store itself."
    )
    _ = parser.add_argument("store", type=Path)
    _ = parser.add_argument("--frames", type=int, required=True)
    _ = parser.add_argument("--scratch", type=Path, required=True)
    _ = parser.add_argument("--expect", action="append", default=[], type=_expected)
    _ = parser.add_argument("--expect-held", type=_chunks)
    _ = parser.add_argument(
        "--no-export", action="store_true", help="Skip the export of a division."
    )
    args = parser.parse_args(argv)

    from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset
    from mosaic.core.manifest import MediaDivision, MediaScanSource
    from mosaic.core.media.facts_columns import row_division
    from mosaic.core.media.imgstore_native import NativeStore
    from mosaic.core.media.video_io import read_frames_at
    from mosaic.core.pipeline.media_index import MediaIndexScope
    from mosaic.core.pipeline.ops import run_op
    from mosaic.core.pipeline.store_export import StoreExportParams
    from mosaic.core.pipeline.transcode import TRANSCODE_KIND_DIRECTORY
    from mosaic.core.scope import Scope
    from mosaic.tracking import register_ops
    from mosaic_media import probe_media

    store: Path = args.store.resolve()
    scratch: Path = args.scratch.resolve()
    if scratch == store or store in scratch.parents or scratch in store.parents:
        parser.error("the scratch directory must lie outside the store")
    register_ops()
    before = listing_digest(store)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    division = MediaDivision(frames=args.frames)

    def fresh(name: str) -> Dataset:
        return open_dataset(new_dataset_manifest(name, scratch / name))

    scoped = fresh("scope")
    result = scoped.write_media_index(
        [MediaIndexScope(directory=store, group="", sequence="", division=division)]
    )
    scanned = fresh("scan")
    scanned.add_scan_source(
        MediaScanSource(id="store", path=str(store), division=division)
    )
    scan = scanned.scan_media()
    assert _without_source(scoped.read_media_index()) == _without_source(
        scanned.read_media_index()
    ), "the scope and the scan wrote different rows"
    assert [plan.recording for plan in scan.divisions] == [
        plan.recording for plan in result.divisions
    ]

    (plan,) = result.divisions
    print(f"recording {plan.recording}  rule {plan.rule.to_cell()}")
    made = [division_row.row for division_row in plan.divisions]
    for row in made:
        print(
            f"  {row.sequence}  chunks {row.chunk_first}-{row.chunk_last}  "
            f"first_frame {row.first_frame}  frames {row.frame_count}  "
            f"follows_gap {row.follows_gap}  start {row.start_utc.isoformat()} "
            f"({row.start_source})"
        )
    for held in result.held_back:
        print(
            f"  held back: chunks {held.chunk_first}-{held.chunk_last} ({held.reason})"
        )
    for refused in result.refused:
        print(
            f"  refused: chunks {refused.chunk_first}-{refused.chunk_last} ({refused.reason})"
        )
    print(f"continuous_groups {scoped.continuous_groups}")
    if args.expect:
        found = [
            Expected(row.chunk_first, row.chunk_last, row.sequence) for row in made
        ]
        assert found == list(args.expect), f"divisions {found}, expected {args.expect}"
    if args.expect_held is not None:
        held_chunks = [(held.chunk_first, held.chunk_last) for held in result.held_back]
        assert held_chunks == [args.expect_held], f"held back {held_chunks}"
    assert all(row_division(row) for row in scoped.read_media_index())

    with NativeStore(store) as native:
        for before_row, after_row in zip(made, made[1:]):
            for frame in (after_row.first_frame - 1, after_row.first_frame):
                owner = before_row if frame < after_row.first_frame else after_row
                resolved = scoped.resolve_media(plan.recording, owner.sequence)
                local = resolved.sequence_frame(frame)
                window = resolved.windows[0]
                ((_, image),) = list(
                    read_frames_at(
                        resolved.paths[0], [local], resolved.facts[0], window=window
                    )
                )
                ((_, whole),) = list(read_frames_at(store, [frame]))
                with NativeStore(store, window) as windowed:
                    native_local = windowed.frame(local)
                assert np.array_equal(image, whole), f"frame {frame} differs"
                assert np.array_equal(native_local, native.frame(frame)), (
                    f"frame {frame} differs in the store reader"
                )
                print(f"  frame {frame} = local {local} of {owner.sequence}: identical")

    if not args.no_export and len(made) > 1:
        second = made[1]
        _ = run_op(
            scoped,
            "export-store",
            StoreExportParams(),
            scope=Scope(entries=[(plan.recording, second.sequence)]),
        )
        (export,) = sorted(
            (scoped.get_root("media") / TRANSCODE_KIND_DIRECTORY).glob("*.mp4")
        )
        facts = probe_media(export)
        resolved = scoped.resolve_media(plan.recording, second.sequence)
        rate = float(resolved.facts[0].fps)
        print(
            f"  export {export.name}: {facts.frame_count} frames at {facts.fps:.5f} fps"
        )
        assert facts.frame_count == second.frame_count
        assert abs(facts.fps - rate) / rate < 1e-3

    after = listing_digest(store)
    print(f"listing md5 before {before} after {after}")
    assert before == after, "the store's listing changed"
    return 0


if __name__ == "__main__":
    sys.exit(main())
