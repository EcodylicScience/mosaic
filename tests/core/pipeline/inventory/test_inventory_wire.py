"""An inventory as bounded JSON: narrowed, filtered, paged and read in depth."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import get_args

import pytest

from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory import (
    ARTIFACT_KINDS,
    ARTIFACT_STATUSES,
    ArtifactKind,
    ArtifactNotFound,
    ArtifactRef,
    ArtifactStatus,
    Coverage,
    FeatureRunRef,
    FrameRunRef,
    InventoryQueryInvalid,
    InventoryWire,
    LabelSeriesRef,
    MediaDerivativeRef,
    ResolvedWire,
    TracksVariantRef,
    TrainedModelRef,
    coverage_wire,
    inventory,
    inventory_wire,
    key_sample,
    ref_address,
)
from mosaic.core.scope import Scope
from mosaic.core.pipeline.preprocess_layout import media_variant_path
from tests.helpers import (
    add_media_variant,
    add_track_sequences,
    add_tracks_variant,
    finish_media_variant,
    make_dataset,
    stub_media_dataset,
)

STORAGE = "speed-angvel__from__tracks"
VARIANT = "convert-trex.0.2-1111111111"
MEDIA_VARIANT = "preprocess.0.1-aaaaaaaaaa"
SEQ_A = ("", "seq_a")


def _run(ds: Dataset, *, scope: Scope | None = None) -> str:
    feature = build_feature("speed-angvel", None, None)
    return str(ds.run_feature(feature, scope=scope).run_id)


def _keys(wire: InventoryWire) -> list[tuple[str, str, str, str]]:
    return [(a.kind, a.name, a.run_id, a.ref) for a in wire.artifacts]


@pytest.fixture
def computed(tmp_path: Path) -> tuple[Dataset, str]:
    """Two tracks tables of one variant, and one feature run over both."""
    ds = make_dataset(tmp_path / "computed")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds, _run(ds)


# --- the projection -----------------------------------------------------------


def test_with_no_query_every_record_crosses_as_the_scan_read_it(
    computed: tuple[Dataset, str],
) -> None:
    ds, _ = computed
    held = inventory(ds)

    wire = inventory_wire(ds, held)

    assert wire.total == len(held.records) == len(wire.artifacts)
    assert wire.next_cursor is None
    statuses = {a.ref: a.status for a in wire.artifacts}
    assert statuses == {ref_address(r.ref): r.status for r in held.records}
    assert _keys(wire) == sorted(_keys(wire))


def test_a_feature_run_crosses_with_its_params_and_tracks_variants(
    computed: tuple[Dataset, str],
) -> None:
    ds, run_id = computed

    [feature] = inventory_wire(ds, inventory(ds), kinds=["feature"]).artifacts

    assert feature.ref == f"feature/{STORAGE}/{run_id}"
    assert feature.status == "complete"
    assert feature.params_state == "present"
    assert feature.params is not None
    assert feature.tracks_variants == (VARIANT,)
    assert feature.coverage.covered == feature.coverage.target == 2
    assert feature.detail is None


def test_a_kind_the_inventory_did_not_read_is_unavailable(
    computed: tuple[Dataset, str],
) -> None:
    """An empty list would read as "this dataset holds none of those"."""
    ds, _ = computed
    held = inventory(ds, kinds=["feature"])

    wire = inventory_wire(ds, held, kinds=["feature", "tracks-variant"])

    assert wire.unavailable_kinds == ("tracks-variant",)
    assert {a.kind for a in wire.artifacts} == {"feature"}


# --- narrowing ----------------------------------------------------------------


def test_a_scope_narrows_an_entry_keyed_target(computed: tuple[Dataset, str]) -> None:
    ds, _ = computed

    wire = inventory_wire(ds, inventory(ds), scope=Scope(entries=[SEQ_A]))

    by_kind = {a.kind: a for a in wire.artifacts}
    assert by_kind["feature"].coverage.target == 1
    assert by_kind["tracks-variant"].coverage.target == 1


def test_a_narrowed_status_is_rederived_over_the_narrowed_target(
    scenario_dataset: Dataset,
) -> None:
    """A run over one of two entries is partial for the dataset and complete for
    a scope naming that entry."""
    _ = _run(scenario_dataset, scope=Scope(entries=[SEQ_A]))
    held = inventory(scenario_dataset, kinds=["feature"])

    [whole] = inventory_wire(scenario_dataset, held).artifacts
    [narrowed] = inventory_wire(
        scenario_dataset, held, scope=Scope(entries=[SEQ_A])
    ).artifacts

    assert whole.status == "partial"
    assert narrowed.status == "complete"


def test_a_scope_naming_every_entry_keeps_every_status(
    computed: tuple[Dataset, str],
) -> None:
    ds, run_id = computed
    (feature_run_root(ds, STORAGE, run_id) / "seq_b.parquet").unlink()
    held = inventory(ds)
    every = Scope(entries=[SEQ_A, ("", "seq_b")])

    narrowed = inventory_wire(ds, held, scope=every)

    assert [a.status for a in narrowed.artifacts] == [
        a.status for a in inventory_wire(ds, held).artifacts
    ]
    assert "inconsistent" in {a.status for a in narrowed.artifacts}


def test_tracks_run_id_keeps_the_feature_runs_that_read_it(
    computed: tuple[Dataset, str],
) -> None:
    ds, run_id = computed
    held = inventory(ds)

    kept = inventory_wire(ds, held, kinds=["feature"], tracks_run_id=VARIANT)
    other = inventory_wire(
        ds, held, kinds=["feature"], tracks_run_id="convert-trex.0.2-2222222222"
    )

    assert [(a.run_id, a.tracks_variants) for a in kept.artifacts] == [
        (run_id, (VARIANT,))
    ]
    assert other.artifacts == ()


def test_the_empty_tracks_run_id_keeps_the_runs_over_the_unlabeled_tables(
    tmp_path: Path,
) -> None:
    """``run_feature`` records no tracks edge for the unlabeled tables, so a
    run over them has no variants and ``""`` keeps it."""
    ds = make_dataset(tmp_path / "unlabeled")
    add_track_sequences(ds, SEQ_A, ("", "seq_b"))
    run_id = _run(ds)
    held = inventory(ds)

    kept = inventory_wire(ds, held, kinds=["feature"], tracks_run_id="")
    labeled = inventory_wire(ds, held, kinds=["feature"], tracks_run_id=VARIANT)

    assert [(a.run_id, a.tracks_variants) for a in kept.artifacts] == [(run_id, ())]
    assert labeled.artifacts == ()


def test_tracks_run_id_measures_a_run_over_that_variants_entries(
    tmp_path: Path,
) -> None:
    """A run over the one entry of its variant is complete for that variant,
    and partial for the dataset's two entries."""
    other = "convert-trex.0.2-2222222222"
    ds = make_dataset(tmp_path / "two-variants")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    add_tracks_variant(ds, other, "seq_b", std_format="mosaic_v1")
    feature = build_feature("speed-angvel", None, None)
    _ = ds.run_feature(feature, tracks_run_id=other)
    held = inventory(ds, kinds=["feature"])

    [whole] = inventory_wire(ds, held).artifacts
    [narrowed] = inventory_wire(ds, held, tracks_run_id=other).artifacts

    assert (whole.coverage.target, whole.status) == (2, "partial")
    assert (narrowed.coverage.target, narrowed.status) == (1, "complete")


def test_a_narrowed_projection_matches_the_scan_under_the_same_scope(
    computed: tuple[Dataset, str],
) -> None:
    """A scan narrows a variant by its entries alone and a feature run by every
    part of the selector, and a projection narrows each kind the same way."""
    ds, _ = computed
    elsewhere = Scope(groups=["elsewhere"])

    projected = inventory_wire(ds, inventory(ds), scope=elsewhere)
    scanned = inventory_wire(ds, inventory(ds, scope=elsewhere))

    def summary(wire: InventoryWire) -> dict[str, tuple[str, int]]:
        return {a.ref: (a.status, a.coverage.target) for a in wire.artifacts}

    assert summary(projected) == summary(scanned)
    assert summary(projected)[f"tracks-variant/{VARIANT}"] == ("complete", 2)


def test_an_entry_scope_narrows_a_media_variants_orphans_as_the_scan_does(
    tmp_path: Path,
) -> None:
    """A media variant's scan reads only the rows in scope, so an orphaned row
    outside the scope leaves a narrowed projection complete."""
    ds = stub_media_dataset(tmp_path / "media", ["s1", "s2"])
    _ = add_media_variant(ds, MEDIA_VARIANT, "s1")
    _ = add_media_variant(ds, MEDIA_VARIANT, "s2")
    media_variant_path(ds, MEDIA_VARIANT, "", "s2", "").unlink()
    finish_media_variant(ds, MEDIA_VARIANT)
    only_s1 = Scope(entries=[("", "s1")])
    kinds: list[ArtifactKind] = ["media-variant"]

    whole = inventory_wire(ds, inventory(ds, kinds=kinds))
    projected = inventory_wire(ds, inventory(ds, kinds=kinds), scope=only_s1)
    scanned = inventory_wire(ds, inventory(ds, kinds=kinds, scope=only_s1))

    assert whole.artifacts[0].status == "inconsistent"
    assert projected.artifacts[0].status == scanned.artifacts[0].status == "complete"


def test_status_not_leaves_out_those_statuses(computed: tuple[Dataset, str]) -> None:
    ds, _ = computed
    held = inventory(ds)
    everything = inventory_wire(ds, held)

    rest = inventory_wire(ds, held, status_not=["complete"])

    assert "complete" not in {a.status for a in rest.artifacts}
    assert rest.total == sum(a.status != "complete" for a in everything.artifacts)
    assert rest.total < everything.total


def test_the_statuses_as_a_value_match_their_declaration() -> None:
    assert set(ARTIFACT_STATUSES) == set(get_args(ArtifactStatus))
    assert len(ARTIFACT_STATUSES) == len(set(ARTIFACT_STATUSES))


# --- paging -------------------------------------------------------------------


def test_pages_concatenate_to_the_whole_listing(computed: tuple[Dataset, str]) -> None:
    ds, _ = computed
    held = inventory(ds)
    whole = inventory_wire(ds, held)

    paged: list[tuple[str, str, str, str]] = []
    cursor: str | None = None
    while True:
        page = inventory_wire(ds, held, limit=1, cursor=cursor)
        assert page.total == whole.total
        paged.extend(_keys(page))
        cursor = page.next_cursor
        if cursor is None:
            break

    assert paged == _keys(whole)
    assert len(paged) > 2


def test_a_cursor_is_the_same_on_every_read(computed: tuple[Dataset, str]) -> None:
    ds, _ = computed

    first = inventory_wire(ds, inventory(ds), limit=2)
    again = inventory_wire(ds, inventory(ds), limit=2)

    assert first.next_cursor is not None
    assert (first.next_cursor, first.artifacts) == (again.next_cursor, again.artifacts)


def test_a_page_starts_after_its_cursor_when_earlier_artifacts_leave(
    computed: tuple[Dataset, str],
) -> None:
    """The cursor names an artifact, not a position, so a listing that loses an
    earlier artifact between pages skips nothing."""
    ds, _ = computed
    held = inventory(ds)
    first = inventory_wire(ds, held, limit=1)
    second = inventory_wire(ds, held, limit=1, cursor=first.next_cursor)
    gone = first.artifacts[0].kind
    assert second.artifacts[0].kind != gone

    narrower = inventory_wire(
        ds,
        held,
        kinds=[kind for kind in ARTIFACT_KINDS if kind != gone],
        limit=1,
        cursor=first.next_cursor,
    )

    assert narrower.artifacts == second.artifacts


@pytest.mark.parametrize(
    "cursor",
    [
        "not a cursor!",
        base64.urlsafe_b64encode(b'["feature"]').decode(),
        base64.urlsafe_b64encode(b"\xff\xfe").decode(),
    ],
)
def test_a_cursor_this_module_did_not_write_is_invalid(
    computed: tuple[Dataset, str], cursor: str
) -> None:
    ds, _ = computed

    with pytest.raises(InventoryQueryInvalid) as raised:
        _ = inventory_wire(ds, inventory(ds), cursor=cursor)

    assert raised.value.error_kind == "invalid"
    assert raised.value.error_detail() == {"field": "cursor"}


def test_a_limit_below_one_is_invalid(computed: tuple[Dataset, str]) -> None:
    ds, _ = computed

    with pytest.raises(InventoryQueryInvalid):
        _ = inventory_wire(ds, inventory(ds), limit=0)


# --- one artifact in depth ----------------------------------------------------


def test_a_ref_returns_one_artifact_in_depth(computed: tuple[Dataset, str]) -> None:
    ds, run_id = computed

    wire = inventory_wire(ds, inventory(ds), ref=f"feature/{STORAGE}/{run_id}")

    [artifact] = wire.artifacts
    detail = artifact.detail
    assert detail is not None
    assert detail.params_finding == ""
    assert ResolvedWire(where="inputs[tracks]", feature="tracks", run_id=VARIANT) in (
        detail.resolved
    )
    assert detail.scope_entries == 2
    assert detail.rows == 2
    assert detail.orphan_rows.count == 0
    assert detail.drift.count == 0
    assert detail.identity_scheme


def test_a_ref_reports_the_orphans_behind_an_inconsistent_status(
    computed: tuple[Dataset, str],
) -> None:
    ds, run_id = computed
    (feature_run_root(ds, STORAGE, run_id) / "seq_b.parquet").unlink()

    [artifact] = inventory_wire(
        ds, inventory(ds), ref=f"feature/{STORAGE}/{run_id}"
    ).artifacts

    assert artifact.status == "inconsistent"
    assert artifact.detail is not None
    assert artifact.detail.orphan_rows.sample == (("", "seq_b"),)


def test_an_unknown_ref_is_not_found(computed: tuple[Dataset, str]) -> None:
    ds, _ = computed

    with pytest.raises(ArtifactNotFound) as raised:
        _ = inventory_wire(ds, inventory(ds), ref="feature/nothing/0.1-0000000000")

    assert raised.value.error_kind == "not_found"


# --- keys and addresses -------------------------------------------------------


def test_a_key_crosses_as_its_parts() -> None:
    cameras = Coverage(
        target=frozenset({("g", "s", "cam1"), ("g", "s", "cam0")}),
        present=frozenset({("g", "s", "cam0")}),
    )
    revisions = Coverage(target=frozenset({"2", "1"}), present=frozenset[str]())

    assert coverage_wire(cameras).missing_sample == (("g", "s", "cam1"),)
    assert coverage_wire(revisions).missing_sample == (("1",), ("2",))


def test_a_sample_lists_twenty_keys_and_says_there_are_more() -> None:
    sample = key_sample({("", f"s{index:02}") for index in range(25)})

    assert sample.count == 25
    assert sample.sample[0] == ("", "s00")
    assert len(sample.sample) == 20
    assert sample.has_more


@pytest.mark.parametrize(
    ("ref", "address"),
    [
        (
            FeatureRunRef(name="nn__from__tracks", run_id="0.1-a"),
            "feature/nn__from__tracks/0.1-a",
        ),
        (TracksVariantRef(run_id=""), "tracks-variant/"),
        (LabelSeriesRef(series="keypoints", key="set"), "label-series/keypoints/set/"),
        (FrameRunRef(method="uniform", run_id="r"), "frame-run/uniform/r"),
        (
            TrainedModelRef(op_kind="train-pose", run_id="r"),
            "trained-model/train-pose/r",
        ),
        (MediaDerivativeRef(target="analysis"), "media-derivative/analysis"),
    ],
)
def test_an_address_is_the_kind_and_the_key_parts(
    ref: ArtifactRef, address: str
) -> None:
    assert ref_address(ref) == address
