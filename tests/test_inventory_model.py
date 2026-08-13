"""Coverage arithmetic and the one status rule.

Pure types, no dataset and no I/O, so every transition is a table row rather
than a scenario. That is the point of keeping the rule in one function: the
cases that matter -- a partial run, a run still writing, a superseded one -- are
otherwise only reachable through a live filesystem.
"""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.inventory.model import (
    ArtifactStatus,
    Coverage,
    DatasetInventory,
    Entry,
    FeatureRunRef,
    InventoryScope,
    MediaDerivativeRef,
    classify,
)

A: Entry = ("", "seq_a")
B: Entry = ("", "seq_b")


def _coverage(target: set[Entry], present: set[Entry], **kw: bool) -> Coverage[Entry]:
    return Coverage(target=frozenset(target), present=frozenset(present), **kw)


# --- coverage arithmetic ------------------------------------------------------


def test_covered_and_missing_are_derived_not_stored() -> None:
    """Three fields that must agree are one field and two views of it."""
    coverage = _coverage({A, B}, {A})

    assert coverage.covered == frozenset({A})
    assert coverage.missing == frozenset({B})


def test_entries_present_but_unwanted_are_not_covered() -> None:
    """Kept on the record, because a widened scope will find them computed."""
    coverage = _coverage({A}, {A, B})

    assert coverage.covered == frozenset({A})
    assert coverage.missing == frozenset()
    assert coverage.is_satisfied


def test_an_empty_target_is_satisfied_by_anything_at_all() -> None:
    """The rule the completeness predicate this replaced already applied."""
    assert _coverage(set(), {A}).is_satisfied
    assert not _coverage(set(), set()).is_satisfied


def test_covers_all_answers_for_every_key() -> None:
    """A global fit writes one output, so asking which entries it covers is the
    wrong question -- and answering it by counting would say zero of ninety."""
    coverage = _coverage({A, B}, set(), covers_all=True)

    assert coverage.is_satisfied


# --- the status rule ----------------------------------------------------------

_BASE = {
    "satisfied": False,
    "any_covered": False,
    "orphan_rows": False,
    "orphan_files": False,
    "drifted": False,
    "finished": True,
}


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "absent"),
        ({"any_covered": True}, "partial"),
        ({"satisfied": True, "any_covered": True}, "complete"),
        (
            {"satisfied": True, "any_covered": True, "drifted": True},
            "complete-but-drifted",
        ),
        ({"satisfied": True, "any_covered": True, "orphan_rows": True}, "inconsistent"),
        # A row naming a file that is gone outranks everything: it is damage,
        # and it is damage whether or not the run finished.
        ({"orphan_rows": True, "finished": False}, "inconsistent"),
        # Files ahead of rows on a finished run is the disagreement worth naming.
        (
            {"satisfied": True, "any_covered": True, "orphan_files": True},
            "inconsistent",
        ),
        # ... but on an unfinished one it is just a run in progress. Outputs are
        # written before their index rows, so calling this damage would make
        # every live run red.
        (
            {
                "satisfied": True,
                "any_covered": True,
                "orphan_files": True,
                "finished": False,
            },
            "complete",
        ),
    ],
)
def test_status_precedence(changes: dict[str, bool], expected: ArtifactStatus) -> None:
    assert classify(**{**_BASE, **changes}) == expected


def test_partial_is_not_absent() -> None:
    """The distinction the four-value vocabulary could not make.

    "Nothing has run" and "89 of 90" call for different actions, and reporting
    the second as the first is a lie a coverage bar cannot recover from.
    """
    assert classify(**{**_BASE, "any_covered": True}) == "partial"
    assert classify(**_BASE) == "absent"


# --- lookups never raise ------------------------------------------------------


def test_an_artifact_the_dataset_does_not_hold_reads_absent(tmp_path) -> None:
    """Absence has to be something a caller can act on, not something it has to
    guard every lookup against."""
    inventory = DatasetInventory(
        dataset_root=tmp_path,
        scope=InventoryScope(kinds=frozenset({"feature"})),
    )
    ref = FeatureRunRef(name="speed-angvel__from__tracks", run_id="0.1-abcdef0123")

    assert inventory.record(ref) is None
    assert inventory.coverage(ref).target == frozenset()
    assert inventory.status(ref) == "absent"


def test_every_ref_kind_is_a_declared_artifact_kind() -> None:
    """A ref whose kind is not in the vocabulary would group under nothing."""
    from typing import get_args

    from mosaic.core.pipeline.inventory.model import ArtifactKind

    declared = set(get_args(ArtifactKind))
    refs = [
        FeatureRunRef("f", "r"),
        MediaDerivativeRef("analysis"),
    ]
    assert {ref.kind for ref in refs} <= declared
