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

**Scope: feature runs only, for now.** The retrofit hazard applies to an
identifier that already exists *and* is about to move, which today is feature
identity alone -- items 1.1, stage 3 and 4.4 all change that payload's shape.
Tracks variants do not exist yet and can be born marked; no dataset holds any
transcode derivative; and the model, tracker, frames and inference digests are
untouched by any planned stage. Widening later means per-family constants
(``FRAMES_IDENTITY_SCHEME`` and so on) rather than one global number, because
six independent hash functions cannot honestly share one: bumping it for a
feature change would falsely mark every model and tracker row as re-minted, and
a marker that lies is worse than none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from ._utils import atomic_write

MARKER_NAME: Final = ".identity_scheme"

FEATURE_IDENTITY_SCHEME: Final = "2"
"""The contract ``compute_run_id`` implements today.

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


def identity_scheme_payload() -> dict[str, str | int]:
    """The marker's contents."""
    return {
        "scheme": FEATURE_IDENTITY_SCHEME,
        "algo": _ALGORITHM,
        "bytes": _DIGEST_BYTES,
    }


def write_identity_scheme(run_root: Path) -> None:
    """Record the scheme that minted *run_root*.

    Atomic, and deliberately not best-effort: a silently missing marker is the
    exact state the detector must catch, so failing loudly beats writing
    nothing. Idempotent -- rewriting with the same contents is harmless, and a
    run root is rewritten on every invocation including a pure cache hit.
    """
    payload = json.dumps(identity_scheme_payload(), indent=2, sort_keys=True)
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
