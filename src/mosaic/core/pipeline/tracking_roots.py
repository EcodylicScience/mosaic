"""Where tracker intermediates live, and what may be done with them -- item 8.1.

One root for every tracking-stage intermediate, and one place that says so. Item
8.1 asked for the root literal collapsed into a single constant "so the
relocation is a single edit"; by the time three trackers had landed there were
**six** copies of it -- one per tool in ``default_roots``, one per tool in each
runner's self-creating ``set_root`` -- and the add-a-tracker recipe had written
the duplication down as a checklist item rather than removing it.

**A table rather than a constant, because the sweeper needs more than a path.**
Item 8.4's retention is *per artifact class, not per root*: a ``.pv`` and its
settings are expensive and reusable, inference output is audit-only. Written as a
branch per tool that is three branches and then four; written as a column here it
is data, and a fifth tracker joins by adding a row.

**Naming: ``_tracking``, not ``tmp``.** The contents are generated and safe to
delete, but nothing evicts them automatically and no code may assume they are
ephemeral. The leading underscore marks the root as machine-generated without
hiding it, so a user browsing the dataset can see what is safe to remove.

**This module deliberately holds no index-row classes.** ``core`` does not import
``tracking`` -- the constraint ``provenance.py`` states and the reason its walk
leaves extracted frames out -- so a table naming ``TRexIndexRow`` would invert
the layering for the sake of a type annotation. What the sweeper and the
reconciler need from a row is ``prune_missing`` and ``drop_entries``, which every
``IndexCSV`` has whatever it holds; the row classes stay where they are written
and reach this table through registration, not import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "TRACKING_ROOT",
    "TRACKING_ROOTS",
    "RetentionClass",
    "TrackingRoot",
    "is_under_tracking_root",
    "tracking_root_default",
]

TRACKING_ROOT: Final = "_tracking"
"""The parent root, and the one string this programme wants written once.

Also the name a user-content scan must never descend into (item 8.1's exclusion
clause), which is why it is a bare component name rather than a path: the check
is against ``Path.parts``, since a directory exclusion cannot be expressed as a
basename pattern.
"""

RetentionClass = Literal["tracker", "inference"]
"""Which retention window an intermediate falls under (item 8.4).

``tracker`` is the expensive, reusable, correctable output -- a ``.pv`` and its
settings, a ``.slp``, a predictions CSV. Its window is long and is ended by
promotion (item 8.6) rather than by age. ``inference`` is audit-only: neither
reused nor edited, kept so someone can see what a detector emitted before schema
coercion, and evicted on a shorter clock.

A closed alias rather than a bare ``str``, because the window a class maps to is
a policy decision and an unrecognized value must not silently fall through to the
longer one.
"""


@dataclass(frozen=True, slots=True)
class TrackingRoot:
    """One tool's intermediate root, and what the sweeper needs to know about it.

    ``outputs`` are the globs that identify *real* output inside an entry working
    directory. They are not a completeness test -- a completion marker is, and
    that is item 8.2's -- but they are what distinguishes a directory a tracker
    wrote from a directory something else left behind.

    ``path_columns`` are this root's path-bearing index columns *beyond*
    ``abs_path``. They live here rather than in a table beside ``default_roots``
    because that table is what a new tracker forgets: a column missing from it
    silently stops being portable, and the add-a-tracker recipe had to carry a
    checklist item asking people to remember. One row per tracker, and the
    portability passes read it.
    """

    key: str
    retention: RetentionClass
    outputs: tuple[str, ...]
    path_columns: tuple[str, ...] = ()

    @property
    def default_path(self) -> str:
        """This root's location, relative to the dataset base directory."""
        return f"{TRACKING_ROOT}/{self.key}"


TRACKING_ROOTS: Final[dict[str, TrackingRoot]] = {
    root.key: root
    for root in (
        # `.pv` + settings from the convert phase, `.results` + per-individual
        # `data/*.npz` from the track phase.
        TrackingRoot(
            key="trex",
            retention="tracker",
            outputs=("*.pv", "*.settings", "*.results", "data/*.npz"),
            path_columns=("video_abs_path", "pv_path"),
        ),
        TrackingRoot(
            key="sleap",
            retention="tracker",
            outputs=("*.predictions.slp", "*.analysis.h5"),
            path_columns=("video_abs_path", "slp_path", "analysis_h5_path"),
        ),
        TrackingRoot(
            key="litpose",
            retention="tracker",
            outputs=("*.predictions.csv",),
            path_columns=("video_abs_path", "csv_path"),
        ),
        # Model inference (item 8.7). Audit-only: the parquet is what a detector
        # emitted *before* schema coercion, which is what you want when debugging
        # a bad model -- and nothing reads it back, so it is a byproduct on a
        # shorter clock rather than a cache. One root per inference kind, because
        # each is a separate op with its own identifiers.
        TrackingRoot(
            key="infer-pose", retention="inference", outputs=("predictions.parquet",)
        ),
        TrackingRoot(
            key="infer-points", retention="inference", outputs=("predictions.parquet",)
        ),
        TrackingRoot(
            key="infer-localizer",
            retention="inference",
            outputs=("predictions.parquet",),
        ),
    )
}
"""Every root under ``_tracking``, keyed by root key.

The keys are the op kinds, so ``ds.get_root(kind)`` answers for a tracker and an
inference op alike and no caller has to know which it is holding.
"""


def tracking_root_default(key: str) -> str:
    """The default location of tracker root *key*, relative to ``base_dir``.

    Raises on an unregistered key rather than composing a path for it. A tracker
    that has not joined the table is one the sweeper cannot see, and minting its
    root here would put output somewhere nothing reclaims.
    """
    if key not in TRACKING_ROOTS:
        known = ", ".join(sorted(TRACKING_ROOTS))
        raise KeyError(f"unknown tracking root {key!r}; registered roots are {known}")
    return TRACKING_ROOTS[key].default_path


def is_under_tracking_root(parts: tuple[str, ...]) -> bool:
    """Does a path with these components pass through ``_tracking``?

    Component-wise, never by prefix string: a scan is handed arbitrary search
    directories, and ``str.startswith`` would both miss a match below the search
    root and fire on a sibling named ``_tracking_backup``.
    """
    return TRACKING_ROOT in parts
