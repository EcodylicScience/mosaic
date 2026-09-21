"""``mosaic measure-tracks``: fill in what a tracks row never recorded about itself."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import emit_json, fail, stdout_to_stderr


def measure_tracks_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Write the measurements. Default is a dry-run report.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the result as JSON.")
    ] = False,
) -> None:
    """Measure the frame axis of this dataset's tracks tables, and their media's.

    Two passes over ``tracks/index.csv``, both filling only the cells that are
    blank and neither touching a table:

    * the **frame extent** (``frame_min`` / ``frame_max``), read from each
      parquet. Blank refuses ``overlap_frames``, so a dataset converted before
      the columns existed has to be measured once before it can use overlap.
    * the **media length** (``media_frames``), read from the media index by the
      same routing a producer resolves the entry through.

    Then it reports every table where the two disagree, naming its variant:
    an entry re-tracked under a new recipe holds the old table too, and each is
    reported for itself rather than one being resolved to. A tracker that joins a
    session's clips can number fewer frames than the media holds -- TRex does,
    dropping the tail of every clip -- and the result is a table whose ``frame``
    column no longer addresses the video: correct at the start of a sequence and
    progressively wrong through it. Everything computed *inside* such a table is
    unaffected; what breaks is anything that reads a pixel at a track frame.

    This is the only way to ask that question of a table already on disk. A run
    records the comparison as it publishes, but a published table cannot be
    re-bridged without re-tracking, so a session tracked before that existed can
    be measured and never re-reported.

    Dry-run by default. A disagreement it finds is a measurement, not a verdict:
    nothing is rewritten and no table is refused.
    """
    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            extents = ds.measure_frame_extents(dry_run=not apply)
            media = ds.measure_media_frames(dry_run=not apply)
            # After the passes, so a dataset being measured for the first time
            # gets its comparison in the same invocation. In a dry run the cells
            # are unwritten, so this reports only what was already recorded --
            # which is honest, and is why the run is worth repeating with
            # --apply.
            mismatches = ds.frame_axis_mismatches()
    except Exception as exc:  # noqa: BLE001 - surface migration errors cleanly
        fail(f"measure-tracks failed: {exc}")

    rows = [
        {
            "run_id": m.run_id,
            "group": m.group,
            "sequence": m.sequence,
            "tracked_frames": m.tracked,
            "media_frames": m.media,
        }
        for m in mismatches
    ]
    if as_json:
        emit_json(
            {
                "status": "ok",
                "applied": apply,
                "frame_extents_measured": len(extents),
                "media_frames_measured": len(media),
                "frame_axis_mismatch": rows,
            }
        )
        return

    verb = "measured" if apply else "would measure"
    typer.echo(
        f"{verb} {len(extents)} frame extent(s) and {len(media)} media length(s)"
    )
    for row in rows:
        entry = f"{row['group']}/{row['sequence']}" if row["group"] else row["sequence"]
        typer.echo(
            f"frame-axis mismatch {entry} [{row['run_id']}]: the table spans "
            f"{row['tracked_frames']} frames, its media holds "
            f"{row['media_frames']}",
            err=True,
        )
    if rows:
        typer.echo(
            f"{len(rows)} entry(ies) do not address their media. Anything reading "
            "pixels at a track frame is off; everything computed inside the "
            "tables is unaffected.",
            err=True,
        )
    if not apply and (len(extents) or len(media)):
        typer.echo("dry run; pass --apply to write")
