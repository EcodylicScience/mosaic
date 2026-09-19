"""Versioned label series: which exist, how they are laid out, how they are recognized.

``labels_raw`` holds what a person authored about the data. Most of it follows the
**truth** rule every source root follows: one current state per sequence, where a
change moves a composition hash and blocks while derivatives exist.

A **series** follows a different rule, on purpose. It is the committed state of an
editor -- the keypoint annotator, later the behavior scoring tool -- projected to
disk every time that editor is closed. Such a state is *versioned*: each save that
changed something is a new immutable revision, revisions coexist, nothing is ever
replaced, and a consumer names the revision it read. That is what lets a trained
model be tied to the exact annotations it saw, and it is why a new revision must
never block anything: the revision a model consumed is still there.

```
labels_raw/
  index.csv                      uploaded label files, per sequence (truth)
  <series>/                      e.g. keypoints
    .mosaic-series               what marks the directory as a series
    index.csv                    one row per revision
    <key>/rev1/ ... rev<N>/      immutable
      <payload>                  e.g. annotations.coco.json
      manifest.json              origin, revision, digest, counts
```

Two rules under one root is a hazard if either is inferred, so **the rule is
declared here, per series, and never read off a path**. Code that needs to know
whether something is versioned asks this registry.

This module is the dependency-light half: the registry, the layout, and the
recognition test. It imports no pandas, because ``core.manifest`` reads the
registry to validate a declared source and must stay cheap to import. The index,
the writer and the scan live in :mod:`mosaic.core.pipeline.label_series_index`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

__all__ = [
    "LABEL_SERIES",
    "MANIFEST_FILENAME",
    "RESERVED_SERIES",
    "SERIES_MARKER",
    "SeriesSpec",
    "SeriesUnit",
    "is_under_label_series",
    "revision_dirname",
    "revision_of",
    "series_spec",
]

SERIES_MARKER: Final = ".mosaic-series"
"""The file that marks a directory as a label series.

Written once, when the series directory is created, and never removed. It is how
a series is recognized, rather than its name: a scan is handed arbitrary
directories in arbitrary datasets, and a folder a user happened to call
``keypoints`` must stay an ordinary folder.
"""

MANIFEST_FILENAME: Final = "manifest.json"
"""The record beside every revision's payload."""

SeriesUnit = Literal["set", "sequence"]
"""What one key of a series names.

A ``set`` spans sequences -- an annotation set holds frames from many -- so it has
no ``(group, sequence)`` and can never enter a per-sequence composition. A
``sequence`` key names one entry.
"""


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """One declared series.

    Attributes:
        name: The directory under ``labels_raw`` and the value a source declares.
        unit: What a key names. See :data:`SeriesUnit`.
        payload_filename: The file every revision directory holds.
        format: What that file is, for a reader choosing how to parse it.
    """

    name: str
    unit: SeriesUnit
    payload_filename: str
    format: str


LABEL_SERIES: Final = MappingProxyType(
    {
        "keypoints": SeriesSpec(
            name="keypoints",
            unit="set",
            payload_filename="annotations.coco.json",
            format="coco_keypoints",
        ),
    }
)
"""Every series mosaic can write and read, by name."""

RESERVED_SERIES: Final[frozenset[str]] = frozenset({"behavior"})
"""Names held for a series that is designed and not yet specified.

``behavior`` is the scoring tool's. It will use this same mechanism, keyed by
sequence, once its payload format is settled. Reserving the name now means no
dataset grows a ``labels_raw/behavior/`` of some other meaning in the meantime.
"""

_REVISION_RE: Final = re.compile(r"^rev([1-9][0-9]*)$")


def series_spec(name: str) -> SeriesSpec:
    """The declared series called *name*.

    Raises:
        KeyError: If *name* is not a series. A reserved name says so, because
            "unknown" would send someone looking for a typo that is not there.
    """
    found = LABEL_SERIES.get(name)
    if found is not None:
        return found
    known = ", ".join(sorted(LABEL_SERIES))
    if name in RESERVED_SERIES:
        msg = (
            f"label series {name!r} is reserved and not specified yet; "
            f"the series that exist are: {known}"
        )
        raise KeyError(msg)
    msg = f"unknown label series {name!r}; the series that exist are: {known}"
    raise KeyError(msg)


def revision_dirname(revision: int) -> str:
    """The directory name of *revision*: ``rev<N>``, counted from one."""
    if revision < 1:
        msg = f"a revision is counted from 1, got {revision}"
        raise ValueError(msg)
    return f"rev{revision}"


def revision_of(dirname: str) -> int:
    """Which revision a directory name is, or 0 if it is not one."""
    match = _REVISION_RE.match(dirname)
    return int(match.group(1)) if match else 0


def is_under_label_series(path: Path) -> bool:
    """Does *path* lie inside a label series?

    Asked of a file a per-sequence scan is about to index. A series file is not
    an uploaded label file: indexed as one, a projected scoring export would be
    read as a second, conflicting source for its sequence, and its checksum
    would move that sequence's composition on every save.

    Recognized by the marker in an ancestor, never by a directory's name. The
    marker sits at most three levels up -- ``<series>/<key>/rev<N>/<file>`` -- and
    one level up for the series index itself, so the walk is bounded and a file
    anywhere else costs three failed ``stat`` calls.

    This cannot be an ``exclude_patterns`` entry for the reason ``_tracking``
    cannot: those match basenames, and a directory is not a basename.
    """
    parent = path.parent
    for _ in range(3):
        if (parent / SERIES_MARKER).is_file():
            return True
        if parent.parent == parent:
            return False
        parent = parent.parent
    return False
