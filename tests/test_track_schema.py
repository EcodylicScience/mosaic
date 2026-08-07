"""The track-schema registry refuses a name it does not know.

An unregistered schema name used to return an empty report and validate nothing.
That is worse than it sounds: the strict check sits *below* the early return, so
naming a schema that does not exist also silently disarmed ``strict=True``. A
dataset whose ``standard_format`` held a typo got no validation, no warning, and
an exit code of zero -- the failure surfaced much later as a missing column deep
inside a feature, with nothing pointing back at the name.

The exception subclasses ``TrackSchemaError`` deliberately. Both conversion loops
catch that class ahead of their bare ``except Exception`` and re-raise it, so an
unknown name aborts the batch. As a plain ``ValueError`` it would be swallowed
per entry by the warn-and-continue handler, and the run would report success over
a dataset nothing had validated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.types import Inputs
from mosaic.core.schema import (
    DERIVED_COLUMNS,
    TRACK_SCHEMAS,
    ForbiddenTrackColumnError,
    TrackSchema,
    TrackSchemaError,
    UnknownTrackSchemaError,
    ensure_track_schema,
    register_track_schema,
    schema_family,
)

from .conftest import add_tracks_variant


def _minimal_trex_v1_frame() -> pd.DataFrame:
    """A table satisfying every ``trex_v1`` requirement."""
    return pd.DataFrame(
        {
            "frame": [0, 1],
            "time": [0.0, 0.04],
            "id": [0, 0],
            "group": ["", ""],
            "sequence": ["seq", "seq"],
            "poseX0": [1.0, 2.0],
            "poseY0": [3.0, 4.0],
        }
    )


def test_an_unregistered_schema_name_raises() -> None:
    with pytest.raises(UnknownTrackSchemaError):
        _ = ensure_track_schema(_minimal_trex_v1_frame(), "no_such_schema")


def test_an_unregistered_name_raises_even_when_not_strict() -> None:
    """The whole point: ``strict=False`` never meant "accept an unknown name"."""
    with pytest.raises(UnknownTrackSchemaError):
        _ = ensure_track_schema(
            _minimal_trex_v1_frame(), "no_such_schema", strict=False
        )


def test_the_unknown_name_error_is_a_track_schema_error() -> None:
    """So the conversion loops re-raise it instead of warning and continuing."""
    assert issubclass(UnknownTrackSchemaError, TrackSchemaError)


def test_the_unknown_name_error_names_the_registered_schemas() -> None:
    """A typo is only actionable if the message says what was available."""
    with pytest.raises(UnknownTrackSchemaError) as excinfo:
        _ = ensure_track_schema(_minimal_trex_v1_frame(), "trex_v0")
    message = str(excinfo.value)
    assert "trex_v0" in message
    for name in TRACK_SCHEMAS:
        assert name in message


def test_a_registered_schema_still_validates() -> None:
    _, report = ensure_track_schema(_minimal_trex_v1_frame(), "trex_v1", strict=True)
    assert report["missing_required"] == []
    assert report["missing_prefixes"] == []


def test_a_missing_required_prefix_still_refuses_under_strict() -> None:
    frame = _minimal_trex_v1_frame().drop(columns=["poseX0", "poseY0"])
    with pytest.raises(TrackSchemaError):
        _ = ensure_track_schema(frame, "trex_v1", strict=True)


def test_a_non_string_column_label_does_not_break_prefix_matching() -> None:
    """An integer label survives a parquet round trip and must not crash a check.

    ``startswith`` on a non-string label would raise rather than simply not
    matching, which would turn an unusual-but-legal table into a validation
    error naming the wrong cause.
    """
    frame = _minimal_trex_v1_frame()
    frame[7] = [0.0, 0.0]
    _, report = ensure_track_schema(frame, "trex_v1", strict=True)
    assert report["missing_prefixes"] == []


def test_the_optional_column_sets_default_to_empty() -> None:
    """They were annotated ``Set[str] = None``, a default the annotation forbids."""
    schema = TrackSchema(name="bare", required={"frame"})
    assert schema.required_prefixes == frozenset()
    assert schema.recommended == frozenset()


def _minimal_mosaic_v1_frame() -> pd.DataFrame:
    """A table satisfying every ``mosaic_v1`` requirement."""
    frame = _minimal_trex_v1_frame()
    frame["X"] = [1.0, 2.0]
    frame["Y"] = [3.0, 4.0]
    return frame


def test_mosaic_v1_requires_the_body_centre() -> None:
    """``trex_v1`` only recommended it, so a table without X/Y validated clean."""
    with pytest.raises(TrackSchemaError):
        _ = ensure_track_schema(_minimal_trex_v1_frame(), "mosaic_v1", strict=True)


def test_mosaic_v1_forbids_a_derived_column() -> None:
    frame = _minimal_mosaic_v1_frame()
    frame["SPEED"] = [0.0, 1.0]
    with pytest.raises(ForbiddenTrackColumnError):
        _ = ensure_track_schema(frame, "mosaic_v1")


def test_a_forbidden_column_refuses_even_when_not_strict() -> None:
    """A wrong table must not be written just because nobody asked for strict.

    Every tracker write path calls this with ``strict=False``, so a forbidden
    column that only warned would reach disk on all of them.
    """
    frame = _minimal_mosaic_v1_frame()
    frame["ANGLE"] = [0.0, 0.1]
    with pytest.raises(ForbiddenTrackColumnError):
        _ = ensure_track_schema(frame, "mosaic_v1", strict=False)


def test_the_forbidden_error_is_a_track_schema_error() -> None:
    """So a conversion batch aborts rather than warning past a wrong table."""
    assert issubclass(ForbiddenTrackColumnError, TrackSchemaError)


def test_trex_v2_accepts_what_mosaic_v1_forbids() -> None:
    """TREx measures these, so for its schema they are not derived at all."""
    frame = _minimal_mosaic_v1_frame()
    for column in sorted(DERIVED_COLUMNS):
        frame[column] = [0.0, 1.0]
    _, report = ensure_track_schema(frame, "trex_v2", strict=True)
    assert report["forbidden_present"] == []
    assert report["missing_required"] == []


def test_trex_v2_inherits_the_base_requirements() -> None:
    """It declares only its additions; the base contract is not restated."""
    assert TRACK_SCHEMAS["mosaic_v1"].required <= TRACK_SCHEMAS["trex_v2"].required
    assert TRACK_SCHEMAS["trex_v2"].forbidden == frozenset()


def test_an_unlisted_extra_column_is_still_accepted() -> None:
    """The additive-columns promise survives: forbidding is a closed list.

    ``camera`` is the live case -- a multi-camera recording carries one, and
    nothing about that is derived.
    """
    frame = _minimal_mosaic_v1_frame()
    frame["camera"] = ["cam0", "cam0"]
    frame["some_future_column"] = [1, 2]
    _, report = ensure_track_schema(frame, "mosaic_v1", strict=True)
    assert report["forbidden_present"] == []


def test_extending_an_unregistered_schema_raises_at_registration() -> None:
    """Named at import, where the traceback says which schema is at fault."""
    with pytest.raises(UnknownTrackSchemaError):
        register_track_schema(
            TrackSchema(name="derived_from_nothing", extends="no_such_base")
        )
    assert "derived_from_nothing" not in TRACK_SCHEMAS


def test_schema_family_groups_a_superset_with_its_base() -> None:
    assert schema_family("trex_v2") == schema_family("mosaic_v1")
    assert schema_family("trex_v1") != schema_family("mosaic_v1")


def test_schema_family_classifies_an_unknown_name_rather_than_raising() -> None:
    """An index row predating the column carries ``""``; it must still classify."""
    assert schema_family("") == ""
    assert schema_family("never_registered") == "never_registered"


def _dataset_mixing(tmp_path: Path, first: str, second: str) -> Dataset:
    """A dataset whose two entries claim the named schemas."""
    manifest = new_dataset_manifest(name="mixed", base_dir=tmp_path / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    add_tracks_variant(dataset, "convert-a.0.1-aaaaaaaaaa", "seq_a", std_format=first)
    add_tracks_variant(dataset, "convert-b.0.1-bbbbbbbbbb", "seq_b", std_format=second)
    return dataset


def test_a_scope_spanning_two_schema_families_refuses(tmp_path: Path) -> None:
    """The mixture ``select_variant_rows`` cannot see, because it is per entry.

    Two entries, two variants, one recipe each -- legal by every rule that
    existed. But one table is centimetres with ``X`` at the head and the other
    is pixels with ``X`` at the body centre, so a feature reading both compares
    quantities that are not the same quantity.
    """
    ds = _dataset_mixing(tmp_path, "trex_v1", "mosaic_v1")
    with pytest.raises(ValueError, match="incompatible schemas"):
        _ = build_manifest(ds, Inputs(("tracks",)))


def test_the_refusal_names_both_families_and_their_entries(tmp_path: Path) -> None:
    ds = _dataset_mixing(tmp_path, "trex_v1", "mosaic_v1")
    with pytest.raises(ValueError) as excinfo:
        _ = build_manifest(ds, Inputs(("tracks",)))
    message = str(excinfo.value)
    assert "trex_v1" in message
    assert "mosaic_v1" in message
    assert "seq_a" in message
    assert "seq_b" in message


def test_mosaic_v1_and_trex_v2_mix_freely(tmp_path: Path) -> None:
    """They share a base, so what ``mosaic_v1`` guarantees means one thing."""
    ds = _dataset_mixing(tmp_path, "mosaic_v1", "trex_v2")
    _, scope = build_manifest(ds, Inputs(("tracks",)))
    assert scope.entries == {("", "seq_a"), ("", "seq_b")}


def test_narrowing_the_scope_avoids_the_refusal(tmp_path: Path) -> None:
    """Scoped, not dataset-wide: a migration in progress stays usable.

    Refusing on the whole index would make a dataset unusable from the first
    reconverted entry until the last. What must not happen is one *run*
    spanning both.
    """
    ds = _dataset_mixing(tmp_path, "trex_v1", "mosaic_v1")
    _, scope = build_manifest(ds, Inputs(("tracks",)), sequences={"seq_a"})
    assert scope.entries == {("", "seq_a")}


def test_one_schema_across_the_scope_is_untouched(tmp_path: Path) -> None:
    ds = _dataset_mixing(tmp_path, "trex_v1", "trex_v1")
    _, scope = build_manifest(ds, Inputs(("tracks",)))
    assert scope.entries == {("", "seq_a"), ("", "seq_b")}


def test_an_unrecorded_schema_pairs_with_the_legacy_one(tmp_path: Path) -> None:
    """A blank cell is ``trex_v1`` stated by omission, not a third family.

    Every dataset converted before the column existed carries blanks, and most
    carry them beside rows written since. Counting blank as its own family
    refuses every one of those -- the same trap ``select_variant_rows``
    documents for an unlabelled ``run_id``.
    """
    ds = _dataset_mixing(tmp_path, "trex_v1", "")
    _, scope = build_manifest(ds, Inputs(("tracks",)))
    assert scope.entries == {("", "seq_a"), ("", "seq_b")}


def test_an_unrecorded_schema_still_refuses_beside_a_pixel_native_one(
    tmp_path: Path,
) -> None:
    """The case that matters: blank is centimetres, ``mosaic_v1`` is pixels."""
    ds = _dataset_mixing(tmp_path, "", "mosaic_v1")
    with pytest.raises(ValueError, match="incompatible schemas"):
        _ = build_manifest(ds, Inputs(("tracks",)))
