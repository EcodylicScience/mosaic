"""The identity-scheme marker: which hashing contract produced a run.

``run_id`` carries the *feature's* declared version, never the version of the
hashing contract that built it. So two runs whose identifiers differ because the
payload shape changed are indistinguishable from two runs whose parameters
differed -- and nothing, script or human or control plane, can tell a
pre-change directory from a post-change one.

That gap has to be closed *before* the first term is added, not after.
Retrofitting a marker onto identifiers already on disk requires knowing which
contract produced each of them, which is exactly the provenance that does not
exist. So every run root records the scheme it was minted under from the start,
and a migration pass becomes idempotent and resumable rather than guesswork.

**The marker is never hashed, and never appears in a path.** Folding it into
``compute_run_id`` would make the marker itself move every identifier -- the
detector would cause the event it exists to detect. It is provenance, in the
same sense as ``mosaic-media``'s ``identity_scheme`` and ``prober_version``
beside a ``video_uuid``.

**Scope: feature runs, and now op runs.** The retrofit hazard applies to an
identifier that already exists *and* is about to move. That was feature identity
alone until item 4.6, which moves the ``train`` and ``infer`` digests by folding
a weights digest into them -- so an op run root now records its family's scheme
too, and it records it *before* the digest moves rather than after, because a
marker cannot be retrofitted onto identifiers already on disk.

The constants stay per family (``FEATURE_IDENTITY_SCHEME``,
``OP_IDENTITY_SCHEME``, ``TRACKS_IDENTITY_SCHEME``, the two composition ones)
rather than collapsing into one number. Six independent hash functions cannot
honestly share a marker: bumping it for a feature change would falsely mark every
model and tracker run as re-minted, and a marker that lies is worse than none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from ._utils import atomic_write

MARKER_NAME: Final = ".identity_scheme"

FEATURE_IDENTITY_SCHEME: Final = "3"
"""The contract ``compute_run_id`` implements today.

Scheme 3 (item 3.3): the payload gains ``_tracks``, the tracks recipes behind
the tables a run reads, present only when the index names any. The other input
kind was already covered -- a ``Result`` carries its upstream ``run_id`` -- so
this closes the last consumed artifact whose identity the digest omitted.
Because the term is omitted when absent, a dataset whose tracks predate variant
identities digests exactly as it did under scheme 2; the marker still moves,
because what the digest *covers* changed even where its value did not.

Scheme 2 (item 1.1): every ``Result``-shaped reference is pinned to a concrete
``run_id`` before the payload is built, so ``_inputs`` carries the upstream run
rather than a ``None`` standing for "latest, whichever that turns out to be".
The payload's *shape* is unchanged -- one field's value stopped being null --
but a scheme is about what the digest covers, and it now covers something it
did not. Scheme 1 runs stay on disk and keep resolving (migration M1).

A bare monotone counter, matching ``mosaic-media``'s ``IDENTITY_SCHEME``, so a
reader who knows one knows the other. Typed ``str`` rather than ``int`` on
purpose: an integer column round-trips as ``1.0`` through a ``pd.concat``
against an all-NaN column and reads back as the string ``"1.0"``, which is two
on-disk spellings of one scheme and silently defeats the detector.

Named for features rather than bare ``IDENTITY_SCHEME`` because mosaic has six
independent identity functions and this covers one. The bare name would assert
a coverage that does not exist.

Bump it when the *shape of the hashed payload* changes -- a new term, a
resolved reference replacing a literal, a different digest width. Never for a
feature's own ``version``, which ``run_id`` already carries.
"""

# Recorded alongside the scheme so a later repair pass can read what produced a
# directory without having to know what scheme "1" meant at the time.
_ALGORITHM: Final = "sha1"
_DIGEST_BYTES: Final = 5


def identity_scheme_payload(scheme: str) -> dict[str, str | int]:
    """The marker's contents for one identity family."""
    return {
        "scheme": scheme,
        "algo": _ALGORITHM,
        "bytes": _DIGEST_BYTES,
    }


def write_identity_scheme(run_root: Path, scheme: str) -> None:
    """Record the scheme that minted *run_root*.

    Atomic, and deliberately not best-effort: a silently missing marker is the
    exact state the detector must catch, so failing loudly beats writing
    nothing. Idempotent -- rewriting with the same contents is harmless, and a
    run root is rewritten on every invocation including a pure cache hit.

    *scheme* is required rather than defaulted to the feature one. mosaic has
    several independent identity functions and a marker names exactly one of
    them; a default would let a new family inherit another's number silently,
    which is the "a marker that lies is worse than none" failure this module
    exists to avoid. There is one caller per family and each states its own.
    """
    payload = json.dumps(identity_scheme_payload(scheme), indent=2, sort_keys=True)
    atomic_write(run_root / MARKER_NAME, lambda p: p.write_text(payload + "\n"))


def read_identity_scheme(run_root: Path) -> str:
    """The scheme *run_root* was minted under.

    Returns:
        The recorded scheme, or ``""`` for a run root written before the marker
        existed. An honest empty is the point: it means "predates the scheme",
        which is a different and more useful answer than a confident wrong one.
    """
    marker = run_root / MARKER_NAME
    if not marker.exists():
        return ""
    try:
        recorded: object = json.loads(marker.read_text()).get("scheme", "")
    except (json.JSONDecodeError, OSError, AttributeError):
        return ""
    return str(recorded) if recorded else ""
