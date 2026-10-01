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

    Three passes over ``tracks/index.csv``, each filling only the cells that are
    blank and none touching a table:

    * the **frame extent** (``frame_min`` / ``frame_max``), read from each
      parquet. Blank refuses ``overlap_frames``, so a dataset converted before
      the columns existed has to be measured once before it can use overlap.
    * the **media length** (``media_frames``): how many frames the tool should
      have read, from the media index, by the rule a producer records it by. A
      table whose run read less on purpose keeps a blank cell: a frame window,
      or a trimmed or decimated media variant. So does a converted or resampled
      table, which no tool made from media.
    * the **frames read** (``frames_read``): how many frames the tool did read,
      from what the run left on disk. That is the ``.pv`` of a TRex
      conversion, the response the Ultralytics runner wrote beside its
      predictions, or a Lightning Pose table, which has a row at every frame
      read. A SLEAP or ``infer-localizer`` table, or one whose run's files are
      gone, keeps a blank cell.

    Then it reports every table whose tool read another number of frames than
    its media holds (``frames_read`` against ``media_frames``), naming its
    variant: an entry re-tracked under a new recipe holds the old table too, and
    each is reported for itself rather than one being resolved to. A tool can
    read fewer frames than the media holds, as TRex does at the end of every
    file it opens. Where the missing frames are not all at the end, the table's
    ``frame`` column no longer addresses the video. Everything computed
    *inside* such a table is unaffected; what breaks is anything that reads a
    pixel at a track frame.

    This is the one way to ask that question of any table already on disk. A run
    records both cells and the comparison as it publishes, and only TRex can
    publish a table again without tracking again (``mosaic track trex
    --republish``). A table published before the cells existed is measured here,
    and reported here, or not at all.

    Dry-run by default. A disagreement it finds is a measurement, not a verdict:
    nothing is rewritten and no table is refused.
    """
    from mosaic.tracking import register_ops

    # A tracker's row is filled by its op's own rule for a frame window.
    register_ops()
    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            extents = ds.measure_frame_extents(dry_run=not apply)
            media = ds.measure_media_frames(dry_run=not apply)
            read = ds.measure_frames_read(dry_run=not apply)
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
            "frames_read": m.read,
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
                "frames_read_measured": len(read),
                "frame_axis_mismatch": rows,
            }
        )
        return

    verb = "measured" if apply else "would measure"
    typer.echo(
        f"{verb} {len(extents)} frame extent(s), {len(media)} media length(s) "
        f"and {len(read)} frame count(s) read"
    )
    for row in rows:
        entry = f"{row['group']}/{row['sequence']}" if row["group"] else row["sequence"]
        typer.echo(
            f"frame-axis mismatch {entry} [{row['run_id']}]: the tool read "
            f"{row['frames_read']} of {row['media_frames']} frames",
            err=True,
        )
    if rows:
        typer.echo(
            f"{len(rows)} entry(ies) were read short of or past their media. "
            "Unless the difference is all at the end, anything reading pixels at "
            "a track frame is off; everything computed inside the tables is "
            "unaffected.",
            err=True,
        )
    if not apply and (len(extents) or len(media) or len(read)):
        typer.echo("dry run; pass --apply to write")
