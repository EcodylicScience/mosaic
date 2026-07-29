"""Whether a file's recorded identity still describes the file on disk.

One comparison, three amounts of baseline. The distinction is what this module
exists to keep straight, and it is written here rather than in any one caller
because a later reader will otherwise reconcile them into agreement:

- **Scan** (``index_media``) has no baseline. It derives ``(group, sequence)``
  from the tracks keymap and probes everything, so there is no prior row to
  compare against and nothing it could honestly report.
- **Write** (``write_media_index``) has one, and uses it. It reuses a stored
  measurement when size and mtime both match, and re-probes when they do not --
  which is the moment both values are in hand.
- **Audit** (``reprobe-media``) re-probes unconditionally. It must: a cache that
  skips the probe skips the comparison that *is* the check.

The comparison itself lives here rather than in ``reprobe`` so that the write
path can perform it without importing a command module -- one that carries
backups, an abort exception and CLI-shaped report types. Neither side owns the
other.

**What escapes, stated rather than implied.** A replacement preserving size
*and* mtime exactly is invisible to the write path, because the cache serves the
stored measurement and never probes. ``touch -r`` after a re-encode produces
exactly that. The audit path is the only detector, which is why its no-cache rule
must never be optimized into agreement with the write path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .facts_columns import ProbeMetadata, read_link_cell

__all__ = [
    "IdentityChange",
    "MediaDrift",
    "classify_identity",
]

IdentityChange = Literal[
    "minted",
    "unchanged",
    "content_digest_changed",
    "video_uuid_changed",
    "unmintable",
]

MeasurementOrigin = Literal["injected", "cached", "probed"]
"""Where the measurement a row is about to be written from came from.

Load-bearing for drift, not diagnostic. The same inequality means two different
things: against an *injected* measurement it is a caller describing a different
file than the row did -- an ordinary re-upload -- and against a *probed* one it
is the bytes on disk having moved under a stable path, which is drift. A
*cached* measurement is the stored one, so it is tautologically equal and can
only ever report a false positive.
"""


@dataclass(frozen=True)
class MediaDrift:
    """A file whose recorded identity no longer describes the bytes on disk.

    Carries both sides. The fresh measurement wins and the row is rewritten, so
    once the write completes this report is the only surviving record that the
    old identity ever existed -- a caller that wants to keep it has exactly one
    chance to.
    """

    stored_path: str
    resolved_path: Path
    change: IdentityChange
    recorded_uuid: str
    measured_uuid: str
    recorded_digest: str
    measured_digest: str


def classify_identity(
    row: Mapping[str, object], probe: ProbeMetadata
) -> IdentityChange:
    """Compare a row's recorded identity against the freshly probed one.

    Absence is tested before divergence and wins: a row that recorded no identity
    has nothing for the measurement to diverge from, so it is minted, not
    drifted. Treating an absent value as a differing one would report every row
    of a pre-identity media index as drift on the one run where a clean report
    matters.

    **The absence test is split by what the row's regime actually mints**, and
    that split is not optional. A store carries a uuid (its ``__store.uuid``,
    open item O5) and never a ``content_digest`` -- there is no elementary stream
    to hash. Under a single "either is empty" test a store would stay permanently
    ``unmintable``, so ``reprobe --apply`` could never backfill the uuid onto an
    already-indexed row. Under a naive fix it would classify ``"minted"`` on
    every run, because ``recorded_digest`` is empty forever, and ``_needs_patch``
    would then rewrite the index and drop a backup on every run without bound --
    the failure that function's docstring already warns about. Branching on
    whether the *probe* produced a digest gives each regime its own settled
    state.
    """
    if not probe["video_uuid"]:
        # No identity at all: a store whose metadata carries no uuid, or a probe
        # that produced nothing. Neither has anything to mint or compare.
        return "unmintable"
    recorded_uuid = read_link_cell(row, "video_uuid")
    if not probe["content_digest"]:
        # A store: uuid-only. It settles on the uuid alone, and a re-recorded
        # store (a fresh mint in the same directory) is detected. Chunks edited
        # in place are not -- see STORE_IDENTITY_SCHEME, which exists to say so.
        if not recorded_uuid:
            return "minted"
        if recorded_uuid != probe["video_uuid"]:
            return "video_uuid_changed"
        return "unchanged"
    recorded_digest = read_link_cell(row, "content_digest")
    if not recorded_uuid or not recorded_digest:
        return "minted"
    if recorded_digest != probe["content_digest"]:
        return "content_digest_changed"
    if recorded_uuid != probe["video_uuid"]:
        return "video_uuid_changed"
    return "unchanged"
