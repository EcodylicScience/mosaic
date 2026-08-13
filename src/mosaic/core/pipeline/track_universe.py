"""Which track-shaped artifact to read -- item 9.4's widening.

M2 built the narrow half: ``tracks_run_id`` selects one variant *within*
``tracks/``, routed through the single decider ``select_variant_rows``. What was
left is that a track-shaped table can also live in ``features/`` -- a smoothed or
subsampled track is still a track -- and nothing enumerated those.

**The default is the leaf of the chain, never the newest by modification time.**
That phrase names a live default rather than a hypothetical:
``latest_feature_run_root`` sorts on the ``finished_at`` / ``started_at`` strings,
which is wall-clock ordering and is not reproducible across a synced dataset --
two machines that ran the same work in a different order disagree about which
table is current. The leaf is a property of the *data*: the track-shaped node no
other run consumed.

**Two leaves is a refusal, not a tiebreak.** Two smoothing chains off one tracks
variant is a legitimate arrangement and there is no defensible way to pick
between them -- exactly the position ``select_variant_rows`` already takes when
one entry has two genuine recipes. Guessing would serve a silent wrong answer;
raising names both and asks.

**Membership is truth-based, not registry-based.** A run is track-shaped iff its
parquet carries the position columns, which is the test ``build_manifest`` already
performs -- so no producer registry is needed (``core`` must not import
``behavior``) and a new track producer joins by producing. Two consequences worth
knowing: it needs an output *materialized* on disk, so a cold run cannot be
classified at all; and it reads one file per run rather than all of them, so a
run whose outputs disagree about their schema is described by whichever the walk
saw first. Neither is a change from how ``build_manifest`` has always decided.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow.parquet as pq

from mosaic.core.pipeline.index import (
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.dataset_indexes import feature_storages
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.types import COLUMNS

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "AmbiguousTrackLeaf",
    "TrackSource",
    "is_track_shaped",
    "track_leaf",
    "track_universe",
]


class AmbiguousTrackLeaf(RuntimeError):
    """Two or more track-shaped artifacts are leaves, so there is no default.

    Its own type rather than a bare ``ValueError``: a caller offering a picker --
    a CLI prompt, an endpoint returning the options -- needs to distinguish "you
    must choose" from every other bad argument, and matching on a message is how
    that goes wrong later.
    """


@dataclass(frozen=True, slots=True)
class TrackSource:
    """One track-shaped artifact a consumer could read.

    ``storage`` is ``"tracks"`` for a variant under the tracks root, else the
    feature storage directory. ``consumed`` is the track-shaped artifacts this one
    was built from, which is what makes the leaf computable.
    """

    storage: str
    run_id: str
    consumed: frozenset[str] = frozenset()

    @property
    def is_tracks(self) -> bool:
        """Is this a variant under ``tracks/`` rather than a feature output?"""
        return self.storage == "tracks"


def is_track_shaped(path: Path) -> bool:
    """Does this parquet carry the position columns?

    The boolean form of ``manifest._ensure_track_shaped``, which raises. Extracted
    rather than reimplemented so the enumerator and the validator cannot disagree
    about what a track is -- the failure mode being a table this offers and
    ``build_manifest`` then rejects.
    """
    if not path.exists():
        return False
    try:
        # pyarrow ships no type stubs here, so read_schema().names is Unknown.
        names = cast("list[str]", pq.read_schema(path).names)  # pyright: ignore[reportUnknownMemberType]
    except (OSError, ValueError):
        return False
    return {COLUMNS.x_col, COLUMNS.y_col} <= set(names)


def _sample_output(ds: Dataset, storage: str, run_id: str) -> Path | None:
    """One parquet from this run, or ``None`` when it wrote none."""
    root = feature_run_root(ds, storage, run_id)
    if not root.exists():
        return None
    for candidate in sorted(root.glob("*.parquet")):
        return candidate
    return None


def _consumed_track_runs(ds: Dataset, storage: str, run_id: str) -> frozenset[str]:
    """The track-shaped run identifiers this run read, from its ``params.json``.

    Both edges live in the same record and both matter: ``inputs[tracks]`` names a
    tracks variant, and a ``Result`` input names an upstream feature run. Reading
    only the first would make every feature-on-feature chain look like a leaf.

    ``params.json`` rather than an index column because there is no index column:
    ``FeatureIndexRow`` records no upstream. ``provenance._consumed_variants``
    reads the same file for the blast-radius walk; this reads more of it.

    **An empty ``run_id`` on a tracks edge is kept, and that is not an oversight.**
    Item 3.4 defines an unlabelled tracks row -- every row written before
    variants existed -- as *unknown recipe*, and its ``run_id`` is ``""``. A
    feature that consumed one records ``""`` here, so discarding falsy values
    would leave the unlabelled variant looking unconsumed and therefore a leaf
    forever, on exactly the legacy datasets where a chain is most likely to
    already exist. The tracks edge is identified by its ``where`` marker instead,
    which is what distinguishes "consumed the unlabelled variant" from "recorded
    no upstream at all".
    """
    path = feature_run_root(ds, storage, run_id) / "params.json"
    try:
        parsed: object = json.loads(path.read_text())
    except (OSError, ValueError):
        return frozenset()
    if not isinstance(parsed, dict):
        return frozenset()
    payload: Mapping[object, object] = parsed
    entries = payload.get("_resolved")
    if not isinstance(entries, list):
        return frozenset()
    listed: list[object] = entries
    consumed: set[str] = set()
    for item in listed:
        if not isinstance(item, dict):
            continue
        reference: Mapping[object, object] = item
        upstream = reference.get("run_id")
        if not isinstance(upstream, str):
            continue
        if upstream or reference.get("where") == "inputs[tracks]":
            consumed.add(upstream)
    return frozenset(consumed)


def track_universe(ds: Dataset) -> list[TrackSource]:
    """Every track-shaped artifact in the dataset, tracks and features alike.

    Sorted for a stable answer: filesystem order is not, and a report or a
    refusal message built on this must not vary between runs.
    """
    found: list[TrackSource] = []

    variants = read_tracks_index(ds)
    if not variants.empty and "run_id" in variants.columns:
        for run_id in sorted({str(value) for value in variants["run_id"]}):
            found.append(TrackSource(storage="tracks", run_id=run_id))

    for storage in feature_storages(ds):
        index_path = feature_index_path(ds, storage)
        if not index_path.exists():
            continue
        runs = feature_index(index_path).list_runs()
        if runs.empty or "run_id" not in runs.columns:
            continue
        for run_id in sorted({str(value) for value in runs["run_id"]}):
            sample = _sample_output(ds, storage, run_id)
            if sample is None or not is_track_shaped(sample):
                continue
            found.append(
                TrackSource(
                    storage=storage,
                    run_id=run_id,
                    consumed=_consumed_track_runs(ds, storage, run_id),
                )
            )
    return sorted(found, key=lambda source: (source.storage, source.run_id))


def track_leaf(ds: Dataset) -> TrackSource:
    """The track-shaped artifact nothing else consumed -- item 9.4's default.

    A dataset with only converted tables has exactly one leaf per variant, so on
    the ordinary dataset this answers the tracks variant and the widening costs
    nothing.

    Raises:
        AmbiguousTrackLeaf: when two or more are leaves, naming them. Two
            smoothing chains off one variant is legitimate and there is no
            defensible tiebreak -- the same position ``select_variant_rows``
            takes for two recipes on one entry.
        LookupError: when the dataset holds no track-shaped artifact at all.
    """
    universe = track_universe(ds)
    if not universe:
        raise LookupError(
            "this dataset holds no track-shaped artifact. Convert tracks first "
            "(`mosaic convert-tracks`), or run a track-producing feature."
        )
    consumed = {run_id for source in universe for run_id in source.consumed}
    leaves = [source for source in universe if source.run_id not in consumed]
    if len(leaves) == 1:
        return leaves[0]
    if not leaves:
        # Every node is consumed by another, which means a cycle -- impossible
        # through `run_feature`, and evidence of a hand-edited `params.json`
        # rather than of an arrangement to pick between.
        raise AmbiguousTrackLeaf(
            "every track-shaped artifact is consumed by another; the recorded "
            "chain has a cycle and cannot be read. Check the params.json files "
            f"of: {', '.join(f'{s.storage}/{s.run_id}' for s in universe)}"
        )
    named = ", ".join(f"{source.storage}/{source.run_id}" for source in leaves)
    raise AmbiguousTrackLeaf(
        f"{len(leaves)} track-shaped artifacts are leaves of the chain, so there "
        f"is no default: {named}. Name one explicitly -- pass tracks_run_id= for "
        "a tracks variant, or the feature Result for a derived track table."
    )


def current_run_id(ds: Dataset, feature_name: str) -> str:
    """Which run of *feature_name* an unpinned reference reads. The single rule.

    There were three. ``resolve._latest_run_id`` sorted on the recorded timestamps
    and its docstring claimed to use "the same rule as every consumer"; the query
    path used a leaf walk restricted to track-shaped runs; and that walk fell back
    to the clock twice. The claim was false, so pinning a reference could change
    which run a run would have read -- the exact defect pinning exists to prevent.

    Chain-aware rather than leaf-always. When this storage's runs have edges among
    themselves the leaf is meaningful and is used, and two leaves refuse instead of
    tiebreaking. When they have none they are siblings -- the ordinary "re-ran it
    with one parameter changed" state -- and siblings are not an ambiguous chain, so
    recorded time answers. Leaf-always would make every such dataset raise.

    Unlike the old walk this does not require a run to be track-shaped or even
    materialised: the edges come from ``params.json``, which every run writes.
    """
    index_path = feature_index_path(ds, feature_name)
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    index = feature_index(index_path)
    runs = {str(value) for value in index.list_runs().get("run_id", [])}
    if not runs:
        raise ValueError(f"No runs found in {index_path}")
    consumed: set[str] = set()
    for run_id in runs:
        consumed |= _consumed_track_runs(ds, feature_name, run_id)
    if not consumed & runs:
        # Siblings, or a dataset whose runs predate the ``_resolved`` record.
        return index.latest_run_id()
    leaves = sorted(runs - consumed)
    if len(leaves) == 1:
        return leaves[0]
    if not leaves:
        return index.latest_run_id()
    raise AmbiguousTrackLeaf(
        f"{len(leaves)} runs of {feature_name!r} are leaves of the chain, so there "
        f"is no default: {', '.join(leaves)}. Name one by passing its run_id in the "
        "Result."
    )
