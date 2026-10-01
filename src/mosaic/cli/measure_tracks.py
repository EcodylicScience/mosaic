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

    Four passes over ``tracks/index.csv``, none touching a table:

    * the **frame extent** (``frame_min`` / ``frame_max``), read from each
      parquet into the cells that are blank. Blank refuses ``overlap_frames``,
      so a dataset converted before the columns existed has to be measured once
      before it can use overlap.
    * the **media length** (``media_frames``): how many frames the tool should
      have read, from the media index, by the rule that a producer records it
      by. A table whose run read less on purpose is given a blank cell: a frame
      window, or a trimmed or decimated media variant. So is a table that no
      tool made from media, such as a converted or resampled one.
    * the **frames read** (``frames_read``): how many frames the tool did read,
      from what the run left on disk. That is the ``.pv`` of a TRex
      conversion, the response the Ultralytics runner wrote beside its
      predictions, or a Lightning Pose table, which has a row at every frame
      read. A table that no tool made from media is given a blank cell.
    * the **known tail loss** (``known_tail_loss``): how many frames short of the
      end of the file it read the tool is known to stop, from the header of the
      file the run recorded reading. Only TRex has one, and its ``.pv`` names
      the file. Several clips that TRex read as a list are allowed none, which
      the TRex run index still shows once the ``.pv`` is swept.

    The last three rewrite each row to what the rule gives: a count over a blank
    or a different value, and a blank where the rule leaves one, clearing what an
    earlier pass filled. A row whose run cannot be established keeps its value
    and is counted as not established: its variant record, media or files are
    missing or have changed, or its producer leaves no file the pass can read,
    as SLEAP and ``infer-localizer`` do.

    Then it reports every table whose tool read a different number of frames
    than its media holds (``frames_read`` against ``media_frames``), naming its
    variant: an entry re-tracked under a new recipe holds the old table too, and
    each is reported for itself rather than one being resolved to. Where the
    missing frames are not all at the end, the table's ``frame`` column does not
    address the video. Everything computed *inside* such a table is
    unaffected; what breaks is anything that reads a pixel at a track frame.

    A shortfall within what the tool is known to lose at the end of the file it
    read is reported apart, as a known tail loss. TRex has one. It does not read
    the frames its decoder holds back to reorder, and reads one fewer when the
    container records no frame count, so it loses none of an AV1 file and 2
    frames of an H.264 file with B-frames. A row whose file's header was not read
    is allowed the most TRex loses on any file, 3, and the report says so. A
    count cannot show that the missing frames are at the end, and the report
    says that too.

    This is the one way to ask that question of any table already on disk. A run
    records the cells and the comparison as it publishes, and only TRex can
    publish a table again without tracking again (``mosaic track trex
    --republish``). A table published before the cells existed is measured here,
    and reported here, or not at all.

    Dry-run by default. A disagreement it finds is a measurement, not a verdict,
    and no table is refused.
    """
    from mosaic.tracking import register_ops

    # The passes judge a tracker's row by its op's frame window and by the
    # readers that its module registers, so every tracking op is imported first.
    register_ops()
    ds = load_dataset(manifest)
    try:
        with stdout_to_stderr():
            extents = ds.measure_frame_extents(dry_run=not apply)
            media = ds.measure_media_frames(dry_run=not apply)
            read = ds.measure_frames_read(dry_run=not apply)
            tail = ds.measure_known_tail_loss(dry_run=not apply)
            # After the passes, so a dataset being measured for the first time
            # gets its comparison in the same invocation. In a dry run the cells
            # are unwritten, so this reports only what was already recorded --
            # which is honest, and is why the run is worth repeating with
            # --apply.
            mismatches = ds.frame_axis_mismatches()
            tail_short = ds.frame_tail_shortfalls()
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
    tail_rows = [
        {
            "run_id": m.run_id,
            "group": m.group,
            "sequence": m.sequence,
            "frames_read": m.read,
            "media_frames": m.media,
            "allowed": m.allowance.frames,
            "allowance_known": m.allowance.known,
        }
        for m in tail_short
    ]
    if as_json:
        emit_json(
            {
                "status": "ok",
                "applied": apply,
                "frame_extents_measured": len(extents),
                "media_frames_measured": len(media.written),
                "media_frames_cleared": len(media.cleared),
                "media_frames_not_established": len(media.not_established),
                "frames_read_measured": len(read.written),
                "frames_read_cleared": len(read.cleared),
                "frames_read_not_established": len(read.not_established),
                "media_frames_unregistered": len(media.unregistered),
                "frames_read_unregistered": len(read.unregistered),
                "known_tail_loss_measured": len(tail.written),
                "known_tail_loss_cleared": len(tail.cleared),
                "known_tail_loss_not_established": len(tail.not_established),
                "known_tail_loss_unregistered": len(tail.unregistered),
                "frame_axis_mismatch": rows,
                "frame_tail_short": tail_rows,
            }
        )
        return

    verb = "measured" if apply else "would measure"
    typer.echo(
        f"{verb} {len(extents)} frame extent(s), {len(media.written)} media "
        f"length(s), {len(read.written)} frame count(s) read and "
        f"{len(tail.written)} known tail loss(es)"
    )
    cleared = len(media.cleared) + len(read.cleared) + len(tail.cleared)
    if cleared:
        typer.echo(
            f"{'cleared' if apply else 'would clear'} {len(media.cleared)} media "
            f"length(s), {len(read.cleared)} frame count(s) read and "
            f"{len(tail.cleared)} known tail loss(es) that the rule leaves blank"
        )
    kept = (
        len(media.not_established)
        + len(read.not_established)
        + len(tail.not_established)
    )
    if kept:
        typer.echo(
            f"{len(media.not_established)} media length(s), "
            f"{len(read.not_established)} frame count(s) read and "
            f"{len(tail.not_established)} known tail loss(es) keep their values: "
            "what their run read cannot be established",
            err=True,
        )
    unread = len(media.unregistered) + len(read.unregistered) + len(tail.unregistered)
    if unread:
        typer.echo(
            f"{len(media.unregistered)} media length(s), "
            f"{len(read.unregistered)} frame count(s) read and "
            f"{len(tail.unregistered)} known tail loss(es) were not checked: "
            "their producer is not a registered op",
            err=True,
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
    for m in tail_short:
        entry = f"{m.group}/{m.sequence}" if m.group else m.sequence
        typer.echo(
            f"known tail loss {entry} [{m.run_id}]: {m.producer} read {m.read} of "
            f"{m.media} frames, within {m.allowance.describe(m.producer)}",
            err=True,
        )
    if tail_short:
        typer.echo(
            f"{len(tail_short)} entry(ies) were read short by a known tail loss. A "
            "frame count cannot show that the missing frames are at the end.",
            err=True,
        )
    if not apply and (
        len(extents)
        or len(media.written)
        or len(read.written)
        or len(tail.written)
        or cleared
    ):
        typer.echo("dry run; pass --apply to write")
