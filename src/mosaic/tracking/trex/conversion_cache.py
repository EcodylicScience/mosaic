"""The shared TREx conversion, addressed by what it is rather than by which run made it.

A TREx run is two gated phases. The conversion is the expensive one -- on a
multi-hour session it is a 28 GB ``.pv`` and hours of detector inference -- and
the tracking that follows it is cheap. But a tracker run root is named by a hash
over *every* setting, so changing one tracking knob moves the whole working
directory and the conversion is redone. Sweeping eight tracking configurations
cost eight conversions.

TRex itself splits the parameters this way: detection settings are baked into
the ``.pv`` and cannot be changed afterwards, while tracking settings are applied
when it is read. ``CONVERT_KEYS`` and ``TRACK_KEYS`` in
:mod:`~mosaic.tracking.trex.dataset_runs` already partition mosaic's settings
along that line, and every convert marker already records the convert-phase
digest and the source content identity. This module is what gives that pair an
address.

**Both terms are in the path, so a published slot is immutable.** A slot is
``<convert run id>/<source uid>``: different detection settings or different
pixels is a *different* slot, never a rebuild of this one. Nothing ever rewrites
28 GB that another run is reading, which is why no claim is needed for
correctness and none of this needs a "may I destroy that" question. The claim is
here only so two runs do not both spend the hours.

**Publish is rename-last, marker-last.** A conversion is built inside
``.incoming-<execution id>`` and renamed into place file by file; the phase
marker is written after every rename. Reuse needs the marker *and* the output, so
a torn publish leaves no marker and the next run simply converts again -- there
is no state in which a half-written ``.pv`` is served.

**The conversion's own ``.results`` never reaches the slot.** TRex writes one
unconditionally at the end of a conversion, ignoring ``auto_no_results``. Nothing
reads it, it is large, and -- the reason this is a rule rather than tidiness -- a
TRex results load with no explicit path falls back to the *input* folder. Leaving
one beside a shared ``.pv`` would put a stale tracking state exactly where a
later run could pick it up.
"""

from __future__ import annotations

import errno
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.core.pipeline.markers import (
    InflightMarker,
    PhaseMarker,
    clear_inflight,
    inflight_state,
    new_inflight,
    read_inflight,
    try_create_inflight,
)
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.tracking_roots import tracking_root_default
from mosaic.tracking.common.entry import reusable_output
from mosaic.tracking.common.index import (
    TrackerRunRowBase,
    tracker_index,
    tracker_index_path,
)
from mosaic.tracking.trex.version import TREX_VERSION

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext
    from mosaic.tracking.common.scope import TrackerWorkItem

__all__ = [
    "CONVERSION_STEM",
    "CONVERT_KIND",
    "SLOT_POLL_SECONDS",
    "ConversionIndexRow",
    "ConversionPublishError",
    "adopt_by_link",
    "claim_slot",
    "conversion_index",
    "conversion_index_path",
    "conversion_run_id",
    "conversion_slot",
    "mint_conversion_run",
    "publish_conversion",
    "release_slot",
    "reusable_slot",
    "slot_marker_is_usable",
    "staging_dir",
]

SLOT_POLL_SECONDS: Final = 30.0
"""How often a run waiting on someone else's conversion looks again.

Waiting rather than converting in parallel is the whole point: two executions
that both want the same slot would otherwise each spend the hours. The wait
terminates on any of four things -- the holder publishes, its claim self-voids
one idle window after its last output line, the run-log proves it dead and the
claim becomes stealable, or this execution is cancelled. The only unbounded case
is a holder genuinely still converting, which is exactly what is worth waiting
for.
"""

CONVERT_KIND: Final = "trex-convert"
"""Root key for the shared conversion cache, and deliberately not an op kind.

Not registered with ``@register_op``: ``mosaic track trex`` still drives both
phases, so a conversion is never something a user asks for separately. Making it
an op would put it in ``cli/track.py``'s ``tracker_kinds()`` and would make the
graph's op declaration read it as writing ``tracks/``, which it does not.
"""

CONVERSION_STEM: Final = "conversion"
"""The stem every file in a slot carries, whatever the source video is called.

A slot is shared, so the source file's name must not decide the names inside it.
Pinning the stem is what makes ``conversion.settings`` derivable from
``conversion.pv`` by one rule at every call site.
"""

_STAGING_PREFIX: Final = ".incoming-"


class ConversionPublishError(RuntimeError):
    """A conversion finished but did not leave both files a slot must hold."""


@dataclass(frozen=True, slots=True)
class ConversionIndexRow(TrackerRunRowBase):
    """One published conversion slot.

    ``group`` is always empty and ``sequence`` is the **source uid**, not an
    entry name. A slot is addressed by the content it was made from rather than
    by the entry that happened to ask for it first -- two entries over one
    recording share a conversion, and an entry renamed between runs must not
    orphan one. The pair is still what the sweeper's row-drop contract takes, so
    the directory name and the row agree by construction.

    ``abs_path`` is the slot directory itself, because
    ``Dataset._rowed_entries`` keys on the basename of that column: pointing it
    at the ``.pv`` would make every slot read as unrowed, and unrowed is refused
    forever.
    """

    pv_path: str = ""
    settings_path: str = ""
    n_source_videos: int = 0


def conversion_run_id(convert_settings: Mapping[str, object]) -> str:
    """What the conversion run root for these convert-phase settings is called.

    Takes the **projected** convert-phase settings, not the whole dict, so this
    module needs no opinion about which keys those are -- ``CONVERT_KEYS`` is the
    one place that is written down. The digest segment is therefore byte-equal to
    the ``params_hash`` every convert marker already records, which is what lets
    the gate and the address be one decision instead of two that can drift.
    """
    return op_run_id(CONVERT_KIND, TREX_VERSION, dict(convert_settings))


def conversion_index_path(ds: Dataset) -> Path:
    """Where the conversion cache keeps its index."""
    return tracker_index_path(ds, CONVERT_KIND)


def conversion_index(path: Path) -> IndexCSV[ConversionIndexRow]:
    """The typed index of published slots, one row per ``(run id, source uid)``."""
    return tracker_index(path, ConversionIndexRow)


def mint_conversion_run(ds: Dataset, convert_settings: Mapping[str, object]) -> Path:
    """Create and stamp the conversion run root for these settings.

    Deliberately **not** :func:`~mosaic.tracking.common.mint.mint_tracker_run`,
    which also writes a ``tracks/<variant>/params.json``. A conversion produces
    no tracks table, so that would leave a phantom recipe in ``tracks/`` naming a
    variant nothing will ever write. Everything else it does applies and is done
    here: ensure the root, create the directory, stamp the identity scheme, and
    leave a readable copy of the settings beside the run.
    """
    import json

    from mosaic.core.pipeline._utils import json_ready

    if not ds.has_root(CONVERT_KIND):
        ds.set_root(CONVERT_KIND, tracking_root_default(CONVERT_KIND))
    run_root = ds.get_root(CONVERT_KIND) / conversion_run_id(convert_settings)
    run_root.mkdir(parents=True, exist_ok=True)
    write_identity_scheme(run_root, OP_IDENTITY_SCHEME)

    # Best-effort, for the same reason the tracker run root's copy is: the
    # settings are recoverable from the identifier, so failing to write a
    # readable duplicate must not lose a conversion that is otherwise fine.
    try:
        _ = (run_root / "run_params.json").write_text(
            json.dumps(json_ready(convert_settings), indent=2)
        )
    except OSError as exc:
        print(
            f"[{CONVERT_KIND}] failed to save run_params.json: {exc}", file=sys.stderr
        )
    return run_root


def conversion_slot(
    ds: Dataset, convert_settings: Mapping[str, object], item: TrackerWorkItem
) -> Path | None:
    """Where this entry's conversion lives, or ``None`` when it cannot be cached.

    ``None`` means the media carries no content identity, and the honest answer
    is then to convert in place. ``item.source_uid`` is the clip's ``video_uuid``
    for a single source and the ordered composition digest for a joined session;
    with neither, the only key left would be a path, and a path is a mutable key
    for a durable shared artifact -- the next dataset move or re-scan would serve
    one recording's conversion for another's.

    Creates the run root but not the slot: an addressable slot that does not
    exist yet is exactly the miss the caller then fills.
    """
    if not item.source_uid:
        return None
    return mint_conversion_run(ds, convert_settings) / item.source_uid


def staging_dir(slot: Path, execution_id: str) -> Path:
    """Where a conversion is built before it is published.

    Per execution rather than one shared ``.incoming``: two executions that both
    got past the claim -- a stolen expired claim, a shared mount with a skewed
    clock -- must not interleave their files into one directory and publish the
    mixture.
    """
    return slot / f"{_STAGING_PREFIX}{execution_id or 'anon'}"


def publish_conversion(staging: Path, slot: Path) -> Path:
    """Move a finished conversion out of *staging* into *slot*, and return its ``.pv``.

    Renames rather than copies, one file at a time, on the same filesystem: each
    is atomic, and a reader already holding the old inode keeps reading it. The
    caller writes the phase marker **after** this returns, so a failure partway
    leaves a slot that reuse will not accept.

    Raises:
        ConversionPublishError: If the conversion did not leave both a ``.pv``
            and a ``.settings``. The settings file is not optional -- it is the
            only thing carrying the detection parameters into a later tracking
            run, so a slot without one would track under defaults and say
            nothing.
    """
    pv = staging / f"{CONVERSION_STEM}.pv"
    settings = staging / f"{CONVERSION_STEM}.settings"
    missing = [p.name for p in (pv, settings) if not p.exists()]
    if missing:
        raise ConversionPublishError(
            f"conversion in {staging} is missing {', '.join(missing)}; "
            f"refusing to publish a slot that cannot be tracked from"
        )

    # TRex writes this at the end of every conversion whatever we ask for. It
    # must not reach the slot: see the module docstring.
    for noise in staging.glob("*.results"):
        noise.unlink(missing_ok=True)
    for noise in staging.glob("*.results.meta"):
        noise.unlink(missing_ok=True)

    slot.mkdir(parents=True, exist_ok=True)
    for name in (
        f"{CONVERSION_STEM}.settings",
        f"average_{CONVERSION_STEM}.png",
        f"{CONVERSION_STEM}.pv",
    ):
        source = staging / name
        if source.exists():
            os.replace(source, slot / name)
    shutil.rmtree(staging, ignore_errors=True)
    return slot / f"{CONVERSION_STEM}.pv"


def adopt_by_link(local_pv: Path, staging: Path) -> bool:
    """Hard-link an already-converted ``.pv`` and its settings into *staging*.

    A hard link is the whole reason adoption is free: both names are one inode,
    so a conversion already on disk appears in the cache at zero bytes, and
    deleting either later leaves the other intact. A symlink would not do -- the
    sweeper reclaiming the tracker directory would silently break every slot
    pointing into it.

    Never copies on failure. ``EXDEV`` across a mount, or a filesystem with no
    link support, means the run simply keeps using the ``.pv`` where it is, which
    is exactly what it did before this cache existed. A 28 GB copy is not a
    graceful degradation.

    Returns:
        Whether both files were linked.
    """
    local_settings = local_pv.with_suffix(".settings")
    if not local_settings.exists():
        return False
    staging.mkdir(parents=True, exist_ok=True)
    try:
        os.link(local_pv, staging / f"{CONVERSION_STEM}.pv")
        os.link(local_settings, staging / f"{CONVERSION_STEM}.settings")
    except OSError as exc:
        reason = errno.errorcode.get(exc.errno or 0, str(exc.errno))
        print(
            f"[{CONVERT_KIND}] could not hard-link {local_pv} into {staging} "
            f"({reason}); keeping the conversion where it is and not caching it.",
            file=sys.stderr,
        )
        shutil.rmtree(staging, ignore_errors=True)
        return False
    return True


def slot_marker_is_usable(marker: PhaseMarker | None) -> bool:
    """Does this local marker *prove* what a durable cache key needs?

    ``reusable_marker`` treats an empty ``params_hash`` or ``source_uid`` as
    "unknown is not mismatched", which is right for reusing a directory where it
    stands and wrong for promoting its contents into a shared address. A marker
    backfilled onto a pre-marker directory records neither by design, so it is
    reused in place exactly as before and never adopted.
    """
    return marker is not None and bool(marker.params_hash) and bool(marker.source_uid)


def reusable_slot(
    ds: Dataset, slot: Path, *, params_hash: str, item: TrackerWorkItem
) -> Path | None:
    """The published ``.pv`` in *slot*, when it is this entry's conversion.

    The same gate a run applies to its own working directory, pointed at the
    shared one -- ``reusable_output`` already resolves a marker's recorded output
    through the dataset and checks it exists, precisely because a phase's
    artifact need not sit inside the directory that ran it.

    A slot whose ``.settings`` has gone is reported as a **miss**, not a hit.
    That file is the only thing carrying the detection parameters into tracking,
    so binding a ``.pv`` without one would track under defaults and record
    nothing about having done so.
    """
    found = reusable_output(
        ds,
        slot,
        "convert",
        params_hash=params_hash,
        video_path=item.video_path,
        video_uid=item.source_uid,
    )
    if found is None:
        return None
    _marker, pv_path = found
    if not pv_path.with_suffix(".settings").exists():
        print(
            f"[{CONVERT_KIND}] {slot} has a .pv but no .settings; "
            f"converting again rather than tracking without the "
            f"detection parameters.",
            file=sys.stderr,
        )
        return None
    return pv_path


def claim_slot(
    ds: Dataset, ctx: JobContext, slot: Path, *, idle_seconds: float
) -> InflightMarker | None:
    """Take *slot* for this execution, or ``None`` while another holds it.

    The same exclusive-create primitive ``open_entry`` uses, and stealing an
    expired or orphaned claim the same way, but without its two other
    behaviours: nothing is ever removed here (a slot is immutable once
    published), and a contended slot is not announced as skipped, because the
    caller waits for it rather than giving up on the entry.
    """
    import os
    import socket

    slot.mkdir(parents=True, exist_ok=True)
    marker = new_inflight(
        execution_id=ctx.execution_id,
        host=socket.gethostname(),
        pid=os.getpid(),
        phase="convert",
        idle_seconds=idle_seconds,
    )
    for attempt in (0, 1):
        if try_create_inflight(slot, marker):
            return marker
        state = inflight_state(
            read_inflight(slot),
            run_log_base=ds.base_dir,
            execution_id=ctx.execution_id,
        )
        if state == "mine":
            return marker
        if state in {"expired", "orphaned"} and attempt == 0:
            clear_inflight(slot)
            continue
        return None
    return None


def release_slot(slot: Path, execution_id: str = "") -> None:
    """Release *our* claim on a slot. Belongs in the caller's ``finally``."""
    clear_inflight(slot, execution_id=execution_id)


register_reconcilable_index(CONVERT_KIND, conversion_index)
