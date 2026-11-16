"""A feature that reads the ``tags`` series reads one pinned revision of it."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.annotations.tag_table import (
    TagRead,
    TagTable,
    read_tag_table,
    sequence_tags_projection,
    tag_table_payload,
)
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline import label_series_index
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.label_series_index import payload_digest
from mosaic.core.pipeline.ops import IdentityDeferred
from mosaic.core.pipeline.storage_names import derive_storage_name
from mosaic.core.pipeline.run import (
    build_output_path,
    resolve_feature_identity,
    run_feature,
)
from mosaic.core.pipeline.types import DependencyLookup, InputStream, TrackInputs
from mosaic.core.pipeline.types.feature import ReadsTags
from mosaic.core.pipeline.label_series import LabelRevisionRef
from mosaic.core.params import Params
from mosaic.core.scope import Scope
from mosaic.core.typed_attribute import TypedAttributeType, TypedAttributeValue
from tests.helpers import add_track_sequences, make_dataset, tag_table, write_tags

ENTRIES: tuple[Entry, ...] = (("g", "s1"), ("g", "s2"))
TYPES: dict[str, TypedAttributeType] = {"treatment": "categorical", "site": "text"}
WRITES = frozenset({"group", "sequence", "id", "frame", "time"})


class _Stamp:
    """Writes the value of each field's tag for each entry, from the lookup."""

    name = "test-tag-stamp"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        fields: list[str] = []
        field_renames: dict[str, str] | None = None

    def __init__(
        self,
        fields: tuple[str, ...] = ("treatment",),
        field_renames: dict[str, str] | None = None,
        *,
        while_binding: Mapping[str, TagTable] | None = None,
    ) -> None:
        self.inputs = self.Inputs(("tracks",))
        self.params = self.Params(fields=list(fields), field_renames=field_renames)
        self._tags: dict[Entry, Mapping[str, TypedAttributeValue]] = {}
        self._while_binding = while_binding
        self.lookup: DependencyLookup = {}
        self.bound: Dataset | None = None

    def tag_read(self) -> TagRead:
        return TagRead(
            sequence_tags=tuple(self.params.fields),
            writes=WRITES,
            renames=tuple(sorted((self.params.field_renames or {}).items())),
        )

    def bind_dataset(self, ds: Dataset) -> None:
        self.bound = ds
        if self._while_binding is not None:
            _ = write_tags(ds, self._while_binding["next"])

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self.lookup = dependency_lookups.get("_tags", {})
        paths = set(self.lookup.values())
        self._tags = {}
        for path in paths:
            for sequence in read_tag_table(path).sequences:
                self._tags[(sequence.group, sequence.sequence)] = sequence.tags
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        entry = (str(df["group"].iloc[0]), str(df["sequence"].iloc[0]))
        tags = self._tags.get(entry, {})
        out = pd.DataFrame({"frame": df["frame"]})
        for name in self.params.fields:
            renamed = (self.params.field_renames or {}).get(name, name)
            out[renamed] = str(tags.get(name, ""))
        return out


class _Plain:
    """The same feature without the hook."""

    name = "test-tag-plain"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        pass

    def __init__(self) -> None:
        self.inputs = self.Inputs(("tracks",))
        self.params = self.Params()
        self.lookups: dict[str, DependencyLookup] = {}

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self.lookups = dependency_lookups
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"frame": df["frame"], "value": df["X"]})


def _table(treatment: str = "a", site: str = "north", *, entries: int = 2) -> TagTable:
    return tag_table(
        {entry: {"treatment": treatment, "site": site} for entry in ENTRIES[:entries]},
        TYPES,
    )


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    dataset = make_dataset(tmp_path / "ds")
    add_track_sequences(dataset, *ENTRIES)
    return dataset


def _run_root(ds: Dataset, feature: _Stamp, run_id: str) -> Path:
    storage = derive_storage_name(feature.name, feature.inputs.storage_suffix())
    return feature_run_root(ds, storage, run_id)


def _output(ds: Dataset, feature: _Stamp, run_id: str, entry: Entry) -> pd.DataFrame:
    run_root = _run_root(ds, feature, run_id)
    return pd.read_parquet(build_output_path(entry[0], entry[1], run_root))


def test_the_hook_is_a_protocol_a_feature_declares() -> None:
    assert isinstance(_Stamp(), ReadsTags)
    assert not isinstance(_Plain(), ReadsTags)


def test_a_run_reads_the_written_revision_and_records_its_term(ds: Dataset) -> None:
    table = _table()
    revision = write_tags(ds, table)
    feature = _Stamp()

    result = run_feature(ds, feature)

    assert set(feature.lookup) == set(ENTRIES)
    assert (
        _output(ds, feature, result.run_id or "", ("g", "s1"))["treatment"].tolist()
        == ["a"] * 40
    )
    params = json.loads(
        (_run_root(ds, feature, result.run_id or "") / "params.json").read_text()
    )
    (recorded,) = [
        entry for entry in params["_resolved"] if entry["where"] == "series[tags]"
    ]
    assert recorded == {
        "where": "series[tags]",
        "feature": "",
        "run_id": None,
        "series": "tags",
        "key": "dataset",
        "origin_uuid": ds.uuid or "",
        "revision": revision.revision,
        "digest": payload_digest(sequence_tags_projection(table, ["treatment"])),
        "payload_digest": payload_digest(tag_table_payload(table)),
    }


def test_an_edit_of_the_named_tag_gives_a_new_run_id(ds: Dataset) -> None:
    _ = write_tags(ds, _table(treatment="a"))
    first = run_feature(ds, _Stamp())

    _ = write_tags(ds, _table(treatment="b"))
    second = run_feature(ds, _Stamp())

    assert second.run_id != first.run_id
    assert second.cache_hit is False


def test_an_edit_of_another_tag_keeps_the_run_id_and_serves_the_cache(
    ds: Dataset,
) -> None:
    _ = write_tags(ds, _table(site="north"))
    first = run_feature(ds, _Stamp())

    _ = write_tags(ds, _table(site="south"))
    second = run_feature(ds, _Stamp())

    assert second.run_id == first.run_id
    assert second.cache_hit is True


def test_a_revision_that_defines_another_tag_keeps_the_run_id(ds: Dataset) -> None:
    _ = write_tags(ds, _table())
    first = run_feature(ds, _Stamp())

    extended = tag_table(
        {entry: {"treatment": "a", "site": "north", "batch": "x"} for entry in ENTRIES},
        {**TYPES, "batch": "text"},
    )
    _ = write_tags(ds, extended)
    second = run_feature(ds, _Stamp())

    assert second.run_id == first.run_id
    assert second.cache_hit is True


def test_a_new_entry_in_the_table_gives_a_new_run_id(ds: Dataset) -> None:
    _ = write_tags(ds, _table(entries=1))
    first = run_feature(ds, _Stamp())

    _ = write_tags(ds, _table(entries=2))
    second = run_feature(ds, _Stamp())

    assert second.run_id != first.run_id


def test_a_named_revision_is_read_after_a_later_one_exists(ds: Dataset) -> None:
    _ = write_tags(ds, _table(treatment="a"))
    first = run_feature(ds, _Stamp())
    _ = write_tags(ds, _table(treatment="b"))

    pinned = _Stamp()
    result = run_feature(
        ds, pinned, tag_revision=LabelRevisionRef(key="dataset", revision=1)
    )

    assert result.run_id == first.run_id
    assert result.cache_hit is True
    assert (
        _output(ds, pinned, result.run_id or "", ("g", "s1"))["treatment"].tolist()
        == ["a"] * 40
    )


def test_a_revision_written_after_the_identity_is_not_read(ds: Dataset) -> None:
    _ = write_tags(ds, _table(treatment="a"))
    raced = _Stamp(while_binding={"next": _table(treatment="b")})

    result = run_feature(ds, raced)

    assert (
        _output(ds, raced, result.run_id or "", ("g", "s1"))["treatment"].tolist()
        == ["a"] * 40
    )
    assert run_feature(ds, _Stamp()).run_id != result.run_id


def test_a_run_without_a_revision_raises_and_names_the_series(ds: Dataset) -> None:
    with pytest.raises(IdentityDeferred, match="no revision of tags"):
        _ = run_feature(ds, _Stamp())


@pytest.mark.parametrize(
    ("fields", "renames", "message"),
    [
        ((), None, "fields"),
        (("treatment", "nothing"), None, "'nothing'"),
        (
            ("treatment",),
            {"treatment": "group"},
            "rename 'treatment' to 'group'",
        ),
        (
            ("treatment", "site"),
            {"treatment": "x", "site": "x"},
            "fields 'treatment' and 'site' both write the column 'x'",
        ),
        (
            ("treatment", "site"),
            {"treatment": "site"},
            "fields 'treatment' and 'site' both write the column 'site'",
        ),
    ],
)
def test_a_problem_with_the_fields_refuses_the_run(
    ds: Dataset,
    fields: tuple[str, ...],
    renames: dict[str, str] | None,
    message: str,
) -> None:
    _ = write_tags(ds, _table())

    with pytest.raises(ValueError, match=message):
        _ = run_feature(ds, _Stamp(fields, renames))


def test_a_tag_named_as_a_column_the_step_writes_refuses_the_run(ds: Dataset) -> None:
    named_frame = tag_table(
        {entry: {"frame": "x", "treatment": "a"} for entry in ENTRIES},
        {"frame": "text", "treatment": "categorical"},
    )
    _ = write_tags(ds, named_frame)

    with pytest.raises(
        ValueError,
        match=(
            "tag 'frame' has the name of a column that this step writes; "
            "rename the tag to read it"
        ),
    ):
        _ = run_feature(ds, _Stamp())


def test_an_unsaved_table_gives_the_run_id_of_the_table_once_written(
    ds: Dataset,
) -> None:
    table = _table()
    scope = Scope(entries=list(ENTRIES))

    planned, _ = resolve_feature_identity(
        ds, _Stamp(), scope=scope, tag_payload=tag_table_payload(table)
    )
    _ = write_tags(ds, table)
    from_revision, _ = resolve_feature_identity(ds, _Stamp(), scope=scope)
    result = run_feature(ds, _Stamp())

    assert planned == from_revision == result.run_id


def test_an_identity_of_a_reader_without_a_table_is_deferred(ds: Dataset) -> None:
    with pytest.raises(IdentityDeferred, match="tags"):
        _ = resolve_feature_identity(ds, _Stamp(), scope=Scope(entries=list(ENTRIES)))


def test_a_feature_without_the_hook_resolves_no_table(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        msg = "the series index was read"
        raise AssertionError(msg)

    monkeypatch.setattr(label_series_index, "read_label_series", refuse)
    feature = _Plain()

    _ = run_feature(ds, feature)
    _ = resolve_feature_identity(ds, _Plain(), scope=Scope(entries=list(ENTRIES)))

    assert "_tags" not in feature.lookups
    assert not (ds.base_dir / "tags").exists()
