"""What every integrated tracker must implement, asserted for all of them at once.

A tracker is not one file. It is a registered op, a row in ``TRACKING_ROOTS``, a
reconcilable index, a typed row class, an entry in the golden corpus, and a set
of accessors -- and the way this repository grew a tracker twice was by copying
one that already had all of that, so a piece left out was a piece nobody noticed
was missing. Two of the copies did leave something out: neither read back the
``source_uid`` it recorded, and neither had a golden case pinning its settings
builder.

So these are parametrized over ``TRACKING_ROOTS`` rather than written per
tracker: a fourth tracker inherits every assertion the day its row lands, and a
half-implementation fails by name instead of by silence at the point the
omission finally matters -- an index nothing reclaims, a path column that stops
being portable, an identifier that moves with a green suite.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import Field
from pathlib import Path

import pytest

from mosaic.core.pipeline.dataset_indexes import reconcilable_index
from mosaic.core.pipeline.markers import PhaseName
from mosaic.core.pipeline.ops import OPS
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS, TrackingRoot
from mosaic.tracking import register_ops
from mosaic.tracking.common.index import TrackerRunRowBase
from mosaic.tracking.common.params import TrackerOpParams
from mosaic.tracking.litpose.dataset_runs import LitposeIndexRow
from mosaic.tracking.sleap.dataset_runs import SleapIndexRow
from mosaic.tracking.trex.dataset_runs import TRexIndexRow

register_ops()

TRACKERS: list[str] = sorted(
    key for key, root in TRACKING_ROOTS.items() if root.retention == "tracker"
)

# Named rather than reached through the index registry, because that registry's
# protocol is deliberately the two methods the reconciler and the sweeper call
# and nothing else -- widening it so a test could read a row class would be the
# tail wagging the dog. Naming them here costs one line per tracker and is
# policed by the coverage assertion below.
ROW_CLASSES: dict[str, type[TrackerRunRowBase]] = {
    "trex": TRexIndexRow,
    "sleap": SleapIndexRow,
    "litpose": LitposeIndexRow,
}

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "op_identity_golden.json").read_text()
)

# Types IndexCSV can map onto a CSV column. A row field of any other type is
# written by str() and read back as a string, which is a silent corruption
# rather than an error.
_STORABLE: tuple[type, ...] = (str, int, float, bool, Path)


def test_every_tracker_has_a_row_class_here() -> None:
    """The one place this file names trackers, so it cannot silently miss one."""
    assert sorted(ROW_CLASSES) == TRACKERS


@pytest.mark.parametrize("kind", TRACKERS)
def test_it_is_a_registered_op(kind: str) -> None:
    """Without this there is no ``mosaic run --kind``, and no queued execution."""
    assert kind in OPS
    op = OPS[kind]
    assert op.domain == "tracking"
    assert op.resource_class, f"{kind} declares no resource class, so it routes nowhere"


@pytest.mark.parametrize("kind", TRACKERS)
def test_its_params_share_the_scope_and_execution_contract(kind: str) -> None:
    """Scope and the throughput knobs are the same question for every tracker."""
    assert issubclass(OPS[kind].Params, TrackerOpParams)


@pytest.mark.parametrize("kind", TRACKERS)
def test_it_registers_a_reconcilable_index(kind: str) -> None:
    """An unregistered index is reached by no reindex or prune pass.

    A working directory deleted by hand keeps its row forever, and
    ``mosaic sweep-tracking`` silently reclaims nothing.
    """
    assert reconcilable_index(kind) is not None


@pytest.mark.parametrize("kind", TRACKERS)
def test_it_declares_at_least_one_phase_with_unique_names(kind: str) -> None:
    """The sweeper needs every phase before it calls a directory finished."""
    root: TrackingRoot = TRACKING_ROOTS[kind]

    assert root.phase_outputs, f"{kind} declares no gated phase"
    names: list[PhaseName] = [phase.name for phase in root.phase_outputs]
    assert len(names) == len(set(names)), f"{kind} declares a phase twice"
    for phase in root.phase_outputs:
        assert phase.clear_globs, f"{kind}'s {phase.name} declares nothing to clear"


@pytest.mark.parametrize("kind", TRACKERS)
def test_every_path_column_is_declared_both_places(kind: str) -> None:
    """The trap the roots table exists to close.

    A path column on the row but missing from ``path_columns`` silently stops
    being rewritten by the portability passes, so the index stops surviving a
    move or a sync between machines -- and nothing fails until it does.
    """
    root = TRACKING_ROOTS[kind]
    fields: dict[str, Field[object]] = {
        f.name: f for f in dataclasses.fields(ROW_CLASSES[kind])
    }

    for column in root.path_columns:
        assert column in fields, f"{kind} declares path column {column!r} with no field"

    # The other direction, by the naming convention the rows already follow.
    # ``abs_path`` is handled by the generic pass, so it is not declared here.
    looks_like_a_path = {
        name
        for name in fields
        if (name.endswith("_path") or name.endswith("_abs_path")) and name != "abs_path"
    }
    assert looks_like_a_path <= set(root.path_columns), (
        f"{kind} has path-shaped columns missing from path_columns: "
        f"{sorted(looks_like_a_path - set(root.path_columns))}"
    )


@pytest.mark.parametrize("kind", TRACKERS)
def test_every_row_field_can_be_stored_in_a_csv(kind: str) -> None:
    row_cls = ROW_CLASSES[kind]

    for field in dataclasses.fields(row_cls):
        assert field.type in _STORABLE or isinstance(field.type, str), (
            f"{kind}'s {field.name} is {field.type!r}, which IndexCSV cannot map"
        )


@pytest.mark.parametrize("kind", TRACKERS)
def test_its_identity_is_pinned_by_the_golden_corpus(kind: str) -> None:
    """Both halves, because they fail in different ways.

    ``<kind>/run-id-settings`` pins the settings builder's *key set*: rename a
    key inside it and every run root and tracks variant on disk moves. The
    ``tracks/<kind>-variant`` case pins the payload wrapper. Two of the three
    trackers shipped without the first, so a rename would have moved every one of
    their runs with a fully green suite.
    """
    assert f"{kind}/run-id-settings" in GOLDEN
    assert f"tracks/{kind}-variant" in GOLDEN


@pytest.mark.parametrize("kind", TRACKERS)
def test_its_op_run_id_and_tracks_variant_share_one_digest(kind: str) -> None:
    """The invariant that keeps a run and the tables it produced legible together.

    Both are minted from the same settings passed through unwrapped, so
    ``_tracking/<kind>/<id>/`` and ``tracks/<id>/`` name the same run. Reading it
    off the corpus rather than recomputing means a change to either minter that
    broke the correspondence would be visible in the data file.
    """
    variant = GOLDEN[f"tracks/{kind}-variant"]

    assert variant.startswith(f"{kind}."), (
        f"{kind}'s tracks variant is not named for its producer: {variant}"
    )
