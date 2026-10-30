"""The parameters and identity of an assign-identities run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

import mosaic.tracking.ops.assign_identities as assign_identities
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import OPS, run_op
from mosaic.core.pipeline.tracks_identity import (
    chained_variant_payload,
    tracks_run_id,
    tracks_variant_root,
)
from mosaic.core.pipeline.tracks_index import tracks_index_path, write_tracks_row
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.runlog import read_run, run_log_dir
from mosaic.tracking.ops.assign_identities import (
    AssignIdentitiesOp,
    AssignIdentitiesParams,
)
from tests.helpers import make_dataset

KIND = "assign-identities"

SOURCE = "sleap.1.6-abcdef0123"

type Reference = tuple[str, str, str, int | None]
"""An identity, the group and sequence of its entry, and its source id."""


def _params(references: list[Reference], **overrides: object) -> AssignIdentitiesParams:
    return AssignIdentitiesParams.model_validate(
        {
            "source_tracks_run_id": SOURCE,
            "references": [
                {
                    "identity": identity,
                    "entry": {"group": group, "sequence": sequence},
                    "id": source_id,
                }
                for identity, group, sequence, source_id in references
            ],
            **overrides,
        }
    )


TWO: list[Reference] = [("b", "", "s1", 1), ("a", "", "s1", 0)]


def _stored(params: AssignIdentitiesParams) -> list[Reference]:
    return [
        (ref.identity, ref.entry.group, ref.entry.sequence, ref.id)
        for ref in params.references
    ]


# --- params ----------------------------------------------------------------------


def test_the_defaults_are_the_declared_ones() -> None:
    params = _params(TWO)

    assert params.min_similarity == 0.5
    assert params.min_margin == 0.05
    assert params.max_gap == 15


def test_the_registered_op_declares_the_params_alone_as_its_identity() -> None:
    op = OPS[KIND]

    assert op is AssignIdentitiesOp
    assert op.version == "0.1"
    assert op.category == "convert"
    assert op.domain == "tracking"
    assert op.writes_tracks is True
    assert op.scope_takes == "any"
    assert op.scope_dependent is False
    assert op.Params is AssignIdentitiesParams


def test_references_are_stored_sorted_with_a_whole_entry_first() -> None:
    params = _params(
        [
            ("b", "", "s1", 2),
            ("a", "g", "s1", 3),
            ("a", "", "s2", None),
            ("a", "", "s2", 1),
            ("a", "", "s1", 4),
        ]
    )

    assert _stored(params) == [
        ("a", "", "s1", 4),
        ("a", "", "s2", None),
        ("a", "", "s2", 1),
        ("a", "g", "s1", 3),
        ("b", "", "s1", 2),
    ]


def test_the_order_given_does_not_change_the_identity() -> None:
    assert _params(TWO).identity_dump() == _params(list(reversed(TWO))).identity_dump()


def test_a_repeated_reference_is_refused() -> None:
    with pytest.raises(ValidationError, match="more than once"):
        _ = _params([*TWO, ("a", "", "s1", 0)])


def test_an_empty_identity_name_is_refused() -> None:
    with pytest.raises(ValidationError, match="identity"):
        _ = _params([*TWO, ("", "", "s2", 0)])


def test_one_track_named_for_two_identities_is_refused() -> None:
    with pytest.raises(
        ValidationError, match=r"\('', 's1'\) id 0 is named for both 'a' and 'c'"
    ):
        _ = _params([*TWO, ("c", "", "s1", 0)])


def test_one_whole_entry_named_for_two_identities_is_refused() -> None:
    with pytest.raises(ValidationError, match="id None is named for both"):
        _ = _params([("a", "", "s1", None), ("b", "", "s1", None)])


@pytest.mark.parametrize(
    ("whole", "by_id"),
    [pytest.param("a", "b", id="whole-first"), pytest.param("b", "a", id="id-first")],
)
def test_a_track_of_an_entry_named_whole_for_another_identity_is_refused(
    whole: str, by_id: str
) -> None:
    with pytest.raises(ValidationError, match="named whole and by id for both"):
        _ = _params([(whole, "", "s2", None), (by_id, "", "s2", 3), *TWO])


def test_fewer_than_two_identities_are_refused() -> None:
    with pytest.raises(ValidationError, match="at least two identities"):
        _ = _params([("a", "", "s1", 0), ("a", "", "s2", 0)])


def test_a_negative_gap_is_refused() -> None:
    with pytest.raises(ValidationError, match="max_gap"):
        _ = _params(TWO, max_gap=-1)


# --- identity --------------------------------------------------------------------


def test_the_identity_is_the_params_and_the_source(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    params = _params(TWO)

    identity = AssignIdentitiesOp().plan_identity(ds, params, ResolvedScope())

    assert chained_variant_payload(params.identity_dump()) == {
        "params": params.identity_dump()
    }
    assert identity.run_id == op_run_id(
        KIND, "0.1", {"params": params.identity_dump(), "upstream": SOURCE}
    )
    assert identity.tracks_variant == tracks_run_id(
        KIND,
        "0.1",
        chained_variant_payload(params.identity_dump()),
        upstream=SOURCE,
    )


def test_the_identity_answers_without_reading_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    assert not tracks_index_path(ds).exists()

    def refuse(_ds: Dataset) -> pd.DataFrame:
        raise AssertionError("plan_identity read the tracks index")

    monkeypatch.setattr(assign_identities, "read_tracks_index", refuse)

    identity = AssignIdentitiesOp().plan_identity(ds, _params(TWO), ResolvedScope())

    assert identity.run_id.startswith(f"{KIND}.0.1-")
    assert identity.tracks_variant.startswith(f"{KIND}.0.1-")


def test_another_source_names_another_run_and_variant(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    op = AssignIdentitiesOp()

    first = op.plan_identity(ds, _params(TWO), ResolvedScope())
    second = op.plan_identity(
        ds, _params(TWO, source_tracks_run_id="sleap.1.6-0000000000"), ResolvedScope()
    )

    assert first.run_id != second.run_id
    assert first.tracks_variant != second.tracks_variant


# --- the source ------------------------------------------------------------------


def _seed_other_variant(ds: Dataset) -> None:
    """A tracks row of a variant other than the source."""
    variant = "trex.0.1-bbbbbbbbbb"
    out_path = tracks_variant_root(ds.get_root("tracks"), variant) / "s1.parquet"
    table = pd.DataFrame({"frame": [0], "id": [0], "X": [1.0], "Y": [2.0]})
    _ = write_parquet_atomic(table, out_path)
    write_tracks_row(
        ds,
        run_id=variant,
        group="",
        sequence="s1",
        out_path=out_path,
        producer="trex",
        std_format="trex_v1",
        n_rows=len(table),
        consumed_source_roots=("tracks_raw",),
    )


def test_a_source_with_no_row_in_scope_is_refused_after_recording_it(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_other_variant(ds)

    with pytest.raises(ClassifiedError, match=SOURCE) as raised:
        _ = run_op(ds, KIND, _params(TWO), execution_id="exec-missing")

    assert raised.value.error_kind == "not_found"
    snapshot = read_run(run_log_dir(ds.base_dir), "exec-missing")
    assert snapshot is not None
    assert snapshot["tracks_variant"] == SOURCE
    assert snapshot["entries_written"] == 0
