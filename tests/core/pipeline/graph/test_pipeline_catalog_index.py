"""The catalog index and one step type described whole.

``catalog_index`` summarizes every entry of ``editor_catalog`` without its params
schema, and ``describe_step_type`` returns one entry with it. A prerequisite that
limits use supplies the entry's license note, which ``kpms`` declares.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from mosaic.behavior.feature_library import FEATURES
from mosaic.behavior.feature_library.kpms import KPMS_LICENSE_ENV
from mosaic.core.pipeline.graph import (
    EXCLUDED_KINDS,
    TRACKS_INPUT,
    CatalogEntry,
    StepTypeNotFound,
    catalog_index,
    declaration_catalog,
    describe_step_type,
    editor_catalog,
    lane_for_step,
)
from mosaic.core.pipeline.ops import OPS
from mosaic.tracking import register_ops


@pytest.fixture
def fresh_catalogs() -> Iterator[None]:
    """Both memoized catalogs are rebuilt inside the test and after it."""
    declaration_catalog.cache_clear()
    editor_catalog.cache_clear()
    try:
        yield
    finally:
        declaration_catalog.cache_clear()
        editor_catalog.cache_clear()


def test_the_index_has_a_row_per_registered_step_type(fresh_catalogs: None) -> None:
    """Every feature, every op a recipe may run, and ``tracks``, sorted by name."""
    register_ops()
    ops = set(OPS) - set(EXCLUDED_KINDS)
    names = [entry.name for entry in catalog_index().entries]

    assert len(names) == len(FEATURES) + len(ops) + 1
    assert set(names) == {cls.name for cls in FEATURES.values()} | ops | {TRACKS_INPUT}
    assert names == sorted(names)
    assert names == [entry.name for entry in editor_catalog().entries]


def test_the_index_holds_no_params_schema() -> None:
    dumped = catalog_index().model_dump_json()

    assert "params_schema" not in dumped
    assert '"$defs"' not in dumped
    assert len(dumped) * 10 < len(editor_catalog().model_dump_json())


def test_each_row_summarizes_the_entry_that_describe_returns() -> None:
    for row in catalog_index().entries:
        entry = describe_step_type(row.name)
        provides = entry.provides

        assert (row.type, row.version, row.category, row.summary) == (
            entry.type,
            entry.version,
            entry.category,
            entry.summary,
        )
        assert row.emits == entry.declaration.emits
        assert (row.unavailable, row.license_note) == (
            entry.unavailable,
            entry.license_note,
        )
        if entry.requires is None:
            assert row.requires is None
        else:
            assert row.requires == tuple(names.label for names in entry.requires)
        if provides is None:
            assert (row.provides, row.passthrough, row.payload_families) == (
                None,
                False,
                False,
            )
        else:
            assert row.provides == tuple(names.label for names in provides.adds)
            assert row.passthrough == provides.passthrough
            assert row.payload_families == provides.payload_families
        expected_lane = None if row.type == "tracks" else lane_for_step(row.name)
        assert row.lane == expected_lane


def test_a_described_step_type_round_trips_through_json() -> None:
    entry = describe_step_type("speed-angvel")
    served = CatalogEntry.model_validate_json(entry.model_dump_json())

    assert served == entry
    assert entry.params_schema is not None
    assert "step_size" in json.dumps(entry.params_schema)
    assert entry.requires is not None
    assert entry.provides is not None


@pytest.mark.parametrize(
    ("name", "costly"),
    [
        ("kpms", True),
        ("overlay", True),
        ("global-kmeans", False),
        ("speed-angvel", False),
    ],
)
def test_the_declaration_carries_costly_fit(name: str, costly: bool) -> None:
    entry = describe_step_type(name)
    served = CatalogEntry.model_validate_json(entry.model_dump_json())

    assert entry.declaration.costly_fit is costly
    assert served.declaration.costly_fit is costly


@pytest.mark.parametrize(
    ("name", "fits"),
    [
        ("extract-labeled-templates", True),
        ("xgboost", True),
        ("global-scaler", True),
        ("sequence-classifier", True),
        ("evaluate-labels", False),
        ("speed-angvel", False),
        ("tracks", False),
    ],
)
def test_the_declaration_carries_fits(name: str, fits: bool) -> None:
    entry = describe_step_type(name)
    served = CatalogEntry.model_validate_json(entry.model_dump_json())

    assert entry.declaration.fits is fits
    assert served.declaration.fits is fits


@pytest.mark.parametrize("name", ["no-such-step", *sorted(EXCLUDED_KINDS)])
def test_a_name_outside_the_catalog_is_not_found(name: str) -> None:
    with pytest.raises(StepTypeNotFound) as raised:
        _ = describe_step_type(name)

    assert raised.value.error_kind == "not_found"
    assert raised.value.error_detail() == {"name": name}


@pytest.mark.parametrize("accepted", ["1", ""])
def test_kpms_is_restricted_whether_or_not_its_license_is_accepted(
    accepted: str, monkeypatch: pytest.MonkeyPatch, fresh_catalogs: None
) -> None:
    monkeypatch.setenv(KPMS_LICENSE_ENV, accepted)
    rows = {row.name: row for row in catalog_index().entries}
    note = rows["kpms"].license_note

    assert note is not None
    assert "non-commercial" in note
    assert "fee-for-service" in note
    assert describe_step_type("kpms").license_note == note
    assert {name for name, row in rows.items() if row.license_note} == {"kpms"}
