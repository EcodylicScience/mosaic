"""The typed per-kind labels index, its resolver, and the conversion flow (item 9.3)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, override

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior import label_library
from mosaic.core import dataset as dataset_module
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import place_label_payload
from mosaic.core.label_converter import (
    LABEL_CONVERTERS,
    LabelConverter,
    LabelConvertParams,
    LabelEntry,
)
from mosaic.core.pipeline.composition import SourceMember, labels_raw_composition
from mosaic.core.pipeline.labels_index import (
    LABELS_INDEX_COLUMNS,
    empty_labels_frame,
    read_labels_index,
    select_label_variant_rows,
    write_labels_row,
)
from mosaic.core.pipeline.sequence_index import write_sequence_compositions
from tests.helpers import (
    add_track_sequences,
    add_tracks_table,
    calms21_dataset,
    functions_named,
    make_dataset,
    names_read,
    source_tree,
    track_table,
    write_painted_recording,
)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(["group", "sequence", "run_id"]))


# --- select_label_variant_rows -------------------------------------------------


def test_absent_index_reads_as_empty_full_schema(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    frame = read_labels_index(ds, "behavior")
    assert frame.empty
    assert list(frame.columns) == LABELS_INDEX_COLUMNS


def test_select_single_variant(tmp_path: Path) -> None:
    df = _frame([{"group": "", "sequence": "s", "run_id": "v1"}])
    out = select_label_variant_rows(df)
    assert list(out["run_id"]) == ["v1"]


def test_labelled_supersedes_unlabelled(tmp_path: Path) -> None:
    # A migrated dataset holds an empty-run_id row; the first re-conversion writes
    # a labelled one beside it. Resolving with None must take the labelled one,
    # not raise on the pair.
    df = _frame(
        [
            {"group": "", "sequence": "s", "run_id": ""},
            {"group": "", "sequence": "s", "run_id": "v1"},
        ]
    )
    out = select_label_variant_rows(df)
    assert list(out["run_id"]) == ["v1"]


def test_two_genuine_recipes_raise(tmp_path: Path) -> None:
    df = _frame(
        [
            {"group": "", "sequence": "s", "run_id": "v1"},
            {"group": "", "sequence": "s", "run_id": "v2"},
        ]
    )
    with pytest.raises(ValueError, match="labels_run_id"):
        select_label_variant_rows(df)


def test_explicit_run_id_selects_exactly(tmp_path: Path) -> None:
    df = _frame(
        [
            {"group": "", "sequence": "s", "run_id": "v1"},
            {"group": "", "sequence": "s", "run_id": "v2"},
        ]
    )
    out = select_label_variant_rows(df, "v2")
    assert list(out["run_id"]) == ["v2"]


def test_empty_run_id_selects_unlabelled(tmp_path: Path) -> None:
    df = _frame(
        [
            {"group": "", "sequence": "s", "run_id": ""},
            {"group": "", "sequence": "s", "run_id": "v1"},
        ]
    )
    out = select_label_variant_rows(df, "")
    assert list(out["run_id"]) == [""]


def test_empty_frame_stays_empty() -> None:
    out = select_label_variant_rows(empty_labels_frame())
    assert out.empty


# --- write_labels_row: consumed_composition wiring ----------------------------


def test_write_labels_row_records_labels_raw_composition(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    # A labels_raw composition for the entry, as index_labels_raw would project.
    members = [SourceMember(name="s.csv", digest="deadbeef", algo="md5")]
    comp = labels_raw_composition(members)
    _ = write_sequence_compositions(ds, "labels_raw", compositions={("", "s"): comp})
    # A label table for the same entry, converted from labels_raw.
    out_path = ds.get_root("labels") / "behavior" / "v1" / "s.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, labels=np.array([0, 1]))
    write_labels_row(
        ds,
        run_id="v1",
        group="",
        sequence="s",
        out_path=out_path,
        producer="convert-labels-boris_aggregated_csv",
        label_kind="behavior",
        label_format="individual_pair_v1",
        n_frames=2,
        source="s.csv",
        source_md5="deadbeef",
        consumed_source_roots=("labels_raw",),
    )
    row = read_labels_index(ds, "behavior").iloc[0]
    assert row["consumed_source_roots"] == "labels_raw"
    assert row["consumed_composition"] == comp.digest
    assert row["producer"] == "convert-labels-boris_aggregated_csv"


def test_authored_row_records_no_source_root(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    out_path = ds.get_root("labels") / "id_tags" / "s.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=np.array([0]))
    write_labels_row(
        ds,
        run_id="",
        group="",
        sequence="s",
        out_path=out_path,
        producer="authored",
        label_kind="id_tags",
        label_format="id_tags_v1",
        n_frames=1,
    )
    row = read_labels_index(ds, "id_tags").iloc[0]
    assert row["consumed_source_roots"] == ""
    assert row["consumed_composition"] == ""
    assert row["run_id"] == ""


def test_an_absent_group_is_recorded_as_empty(tmp_path: Path) -> None:
    """The labels writer takes ``object`` for the same reason the tracks one does.

    A caller reading a cell off a pandas ``Series`` gets a float NaN for a blank,
    and ``str()`` of that is the word -- which passes ``validate_entry_name`` and
    then names an entry nothing else knows about.
    """
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    out_path = ds.get_root("labels") / "behavior" / "v1" / "s.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, labels=np.array([0]))

    write_labels_row(
        ds,
        run_id="v1",
        group=float("nan"),
        sequence="s",
        out_path=out_path,
        producer="convert-labels-x",
        label_kind="behavior",
        label_format="individual_pair_v1",
        n_frames=1,
    )

    row = read_labels_index(ds, "behavior").iloc[0]
    assert str(row["group"]) == ""
    assert str(row["sequence"]) == "s"


# --- convert_all_labels end to end --------------------------------------------


def test_convert_all_labels_hash_dir_and_reconvert(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    src = tmp_path / "uploads"
    src.mkdir()
    np.save(
        src / "f1.npy",
        {"g": {"s1": {"annotations": [0, 1, 2, 3]}}},
        allow_pickle=True,
    )
    ds.index_labels_raw([src], patterns=["*.npy"], src_format="calms21_npy")
    ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    idx = read_labels_index(ds, "behavior")
    variant_a = next(str(r) for r in idx["run_id"] if str(r))
    assert set(idx["consumed_source_roots"]) == {"labels_raw"}
    vdir = ds.get_root("labels") / "behavior" / variant_a
    assert (vdir / "g__s1.npz").exists()
    assert (vdir / "params.json").exists()
    original = (vdir / "g__s1.npz").read_bytes()

    # Re-convert with different params: a new variant beside the old, old untouched.
    ds.convert_all_labels(
        kind="behavior",
        source_format="calms21_npy",
        params={"resident_id": 7, "intruder_id": 8},
    )
    idx2 = read_labels_index(ds, "behavior")
    variants = sorted(set(idx2["run_id"]))
    assert len(variants) == 2
    assert (vdir / "g__s1.npz").read_bytes() == original

    # Two labelled variants for one entry -> resolving without a selector raises.
    with pytest.raises(ValueError, match="labels_run_id"):
        select_label_variant_rows(idx2)
    picked = select_label_variant_rows(idx2, variant_a)
    assert set(picked["run_id"]) == {variant_a}


def test_load_labels_resolves_and_reads(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    src = tmp_path / "uploads"
    src.mkdir()
    np.save(
        src / "f1.npy",
        {"g": {"s1": {"annotations": [0, 1, 2, 3, 0]}}},
        allow_pickle=True,
    )
    ds.index_labels_raw([src], patterns=["*.npy"], src_format="calms21_npy")
    ds.convert_all_labels(kind="behavior", source_format="calms21_npy")
    data = ds.load_labels("g", "s1", "behavior")
    assert data["labels"].tolist() == [0, 1, 2, 3, 0]
    assert ds.get_label_map("behavior")[0] == "attack"


# --- the registry a caller has no reason to know they must fill ---------------


def test_an_unregistered_label_format_names_the_ones_that_exist(
    tmp_path: Path,
) -> None:
    """Refused at indexing, where the format is chosen.

    Allowed through, the row lands in the index and ``convert_all_labels``'
    ``src_format`` filter skips it -- forever, and without a report naming the
    rows no converter can reach.
    """
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    src = tmp_path / "uploads"
    src.mkdir()
    (src / "f1.csv").write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="boris_aggregated_csv"):
        _ = ds.index_labels_raw([src], patterns=["*.csv"], src_format="typo_format")

    assert not (ds.get_root("labels_raw") / "index.csv").exists()


def test_a_track_only_format_is_not_a_label_format(tmp_path: Path) -> None:
    """The registries are separate, and the check reads the right one.

    ``deeplabcut`` is a registered *track* converter and no label converter at
    all, so it is exactly the case a track-registry lookup would wave through.
    """
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    src = tmp_path / "uploads"
    src.mkdir()
    (src / "f1.csv").write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="deeplabcut"):
        _ = ds.index_labels_raw([src], patterns=["*.csv"], src_format="deeplabcut")


def test_a_registered_label_format_still_indexes(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", save=False, ensure_roots=False)
    src = tmp_path / "uploads"
    src.mkdir()
    np.save(src / "f1.npy", {"g": {"s1": {"annotations": [0]}}}, allow_pickle=True)

    out = ds.index_labels_raw([src], patterns=["*.npy"], src_format="calms21_npy")

    assert out.exists()
    assert list(pd.read_csv(out)["src_format"]) == ["calms21_npy"]


_FILL_FROM_CORE_ALONE = """
import sys
from mosaic.core.label_converter import registered_label_formats

assert "mosaic.behavior" not in sys.modules, "the label library was already imported"
formats = registered_label_formats()
assert "calms21_npy" in formats, formats
assert "boris_aggregated_csv" in formats, formats
print("OK")
"""


def test_the_label_registry_fills_without_an_explicit_import() -> None:
    """A subprocess, because in this one the registry is already full.

    Every other test here reaches a converter through a Dataset, and any sibling
    module importing ``mosaic.behavior`` fills the registry for the whole
    session -- so an in-process assertion would pass without the loading it
    claims to test. The property is about a *fresh* interpreter that has only
    ever imported ``mosaic.core``, which is what the control plane does.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", _FILL_FROM_CORE_ALONE],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    assert "OK" in proc.stdout


@pytest.mark.xfail(
    strict=True,
    reason="LabelConvertParams.strict_schema is declared but never read",
)
def test_the_label_conversion_reads_strict_schema() -> None:
    """The record for a parameter the label path declares and never consults.

    ``TrackConvertParams.strict_schema`` is read three times, each time as
    ``ensure_track_schema(strict=...)``. ``LabelConvertParams`` declares a field
    of the same name, and no label path validates a schema at all, so nothing
    reads it -- a client drawing a form from the schema offers a strictness
    toggle that changes nothing. Whether labels should have a schema to be
    strict about is the maintainer's call, so the field stays and carries an
    ``unwired`` record.

    Scanned across both places a reader could plausibly sit -- the conversion
    method and every registered converter -- rather than asserted at one call
    site, so that wiring it anywhere retires the marker. The scan counts a read
    only off the names a converter's params are bound to, since ``strict_schema``
    is reached as ``conv_params.strict_schema`` on the tracks side and every
    converter here names its own ``params``. ``strict`` is what retires it: the
    day something reads the field this xpasses and the suite fails until the
    marker is deleted.
    """
    trees = functions_named(dataset_module, ["convert_all_labels"])
    library = Path(label_library.__file__ or "").parent
    for module_path in sorted(library.glob("*.py")):
        trees.append(source_tree(module_path))

    reads = names_read(
        trees,
        owners={"params", "conv_params", "label_params", "p", "self.params"},
        destructured_classes={"LabelConvertParams"},
    )

    assert "strict_schema" in reads


def test_convert_all_labels_reports_written_and_skipped(tmp_path: Path) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 3}}})

    first = ds.convert_all_labels(kind="behavior", source_format="calms21_npy")
    again = ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    assert (first.written, first.skipped) == (2, 0)
    assert (again.written, again.skipped) == (0, 2)
    assert first.variant == again.variant


def test_convert_all_labels_registers_through_mutate_manifest(
    tmp_path: Path,
) -> None:
    """Another writer's manifest change survives the registration."""
    from mosaic.core.dataset import Dataset

    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})
    other = Dataset(manifest_path=ds.manifest_path).load()
    other.meta["note"] = "written by another process"
    other.save()

    outcome = ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    reread = Dataset(manifest_path=ds.manifest_path).load()
    assert reread.meta["note"] == "written by another process"
    labels = reread.meta["labels"]
    assert isinstance(labels, dict)
    behavior = labels["behavior"]
    assert isinstance(behavior, dict)
    assert behavior["run_id"] == outcome.variant


# --- labels on the recording frame axis -------------------------------------


def _pair_payload() -> dict[str, object]:
    return {
        "label_format": "individual_pair_v1",
        "frames": np.array([0, 5]),
        "labels": np.array([1, 2]),
        "individual_ids": np.array([[0, 1], [0, 1]]),
    }


def test_event_labels_move_by_the_sequence_s_first_frame() -> None:
    placed = place_label_payload(
        _pair_payload(), first_frame=100, sequence_start=100, entry="rec:s2"
    )

    assert np.asarray(placed["frames"]).tolist() == [100, 105]
    assert np.asarray(placed["labels"]).tolist() == [1, 2]


def test_labels_at_frame_zero_are_kept_as_converted() -> None:
    payload = _pair_payload()

    placed = place_label_payload(
        payload, first_frame=0, sequence_start=100, entry="rec:s2"
    )

    assert placed is payload


def test_dense_labels_are_kept_for_a_sequence_at_frame_zero() -> None:
    dense = {"labels": np.array([0, 1, 1, 0])}

    assert (
        place_label_payload(dense, first_frame=0, sequence_start=0, entry="rec:s1")
        is dense
    )


def test_dense_labels_are_refused_for_a_sequence_past_frame_zero() -> None:
    dense = {"labels": np.array([0, 1, 1, 0])}

    with pytest.raises(ValueError, match="'rec:s2'.*dense.*starts at frame 5"):
        _ = place_label_payload(dense, first_frame=0, sequence_start=5, entry="rec:s2")


def test_dense_labels_are_refused_for_a_sequence_of_unknown_start() -> None:
    dense = {"labels": np.array([0, 1, 1, 0])}

    with pytest.raises(
        ValueError, match="'rec:s2'.*lacks a known first frame.*no media"
    ):
        _ = place_label_payload(
            dense, first_frame=0, sequence_start="no media", entry="rec:s2"
        )


def test_labels_of_a_continuous_group_convert_at_frame_zero(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    src = tmp_path / "uploads"
    src.mkdir()
    annotations = {"g": {"s1": {"annotations": [0, 1, 2, 3, 0]}}}
    np.save(src / "f1.npy", np.array(annotations, dtype=object), allow_pickle=True)
    ds.index_labels_raw([src], patterns=["*.npy"], src_format="calms21_npy")
    add_track_sequences(ds, ("g", "s1"))
    ds.set_continuous_groups(["g"])

    ds.convert_all_labels(kind="behavior", source_format="calms21_npy")

    (path,) = read_labels_index(ds, "behavior")["abs_path"].tolist()
    with np.load(ds.resolve_path(str(path)), allow_pickle=True) as npz:
        frames = np.asarray(npz["frames"], dtype=np.int64)
    assert frames.tolist() == [0, 1, 2, 3, 4]


class _DenseLabels(LabelConverter):
    """Dense labels from one file for ``a`` of the empty group and each of ``members``."""

    src_format: ClassVar[str] = "dense_test_npy"
    label_kind: ClassVar[str] = "behavior"
    label_format: ClassVar[str] = "dense"
    members: ClassVar[tuple[str, ...]] = ("s1", "s2")

    @override
    def convert(
        self,
        src_path: Path,
        params: LabelConvertParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        labels = np.array([0, 1, 1, 0])
        return [
            LabelEntry(group="", sequence="a", payload={"labels": labels}, n_frames=4),
            *(
                LabelEntry(
                    group="g", sequence=member, payload={"labels": labels}, n_frames=4
                )
                for member in self.members
            ),
        ]


class _DenseFirstMember(_DenseLabels):
    """The dense labels of ``a`` and of ``g``'s first sequence alone."""

    src_format: ClassVar[str] = "dense_first_test_npy"
    members: ClassVar[tuple[str, ...]] = ("s1",)


def _dense_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    converter: type[_DenseLabels],
    *,
    media: bool,
) -> Dataset:
    """A dataset declaring ``g``, with *converter*'s upload indexed.

    With *media*, ``g`` records ``s1`` in 5 frames and ``s2`` in 4, so ``s2``
    starts at frame 5. Without it, ``g`` has tracks and no media.
    """
    monkeypatch.setitem(LABEL_CONVERTERS, (converter.src_format, "behavior"), converter)
    ds = make_dataset(tmp_path / "ds")
    src = tmp_path / "uploads"
    src.mkdir()
    np.save(src / "f1.npy", np.zeros(1))
    ds.index_labels_raw([src], patterns=["*.npy"], src_format=converter.src_format)
    if media:
        _ = write_painted_recording(ds, "g", [("s1", [(5, 30.0)]), ("s2", [(4, 30.0)])])
    else:
        add_track_sequences(ds, ("g", "s1"), ("g", "s2"))
    ds.set_continuous_groups(["g"])
    return ds


def _assert_no_label_written(ds: Dataset) -> None:
    assert read_labels_index(ds, "behavior").empty
    assert not list((ds.get_root("labels") / "behavior").rglob("*.npz"))


@pytest.mark.media
def test_dense_labels_of_a_continuous_group_s_first_sequence_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requires_ffmpeg: None
) -> None:
    """The first sequence starts at frame 0, where each position is its frame."""
    ds = _dense_dataset(tmp_path, monkeypatch, _DenseFirstMember, media=True)

    outcome = ds.convert_all_labels(
        kind="behavior", source_format=_DenseFirstMember.src_format
    )

    assert outcome.written == 2
    entries = read_labels_index(ds, "behavior")[["group", "sequence"]]
    assert sorted(map(tuple, entries.to_numpy().tolist())) == [("", "a"), ("g", "s1")]


@pytest.mark.media
def test_a_refused_dense_entry_leaves_no_label_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requires_ffmpeg: None
) -> None:
    """``g:s2`` starts at frame 5, and the refusal comes before the first file."""
    ds = _dense_dataset(tmp_path, monkeypatch, _DenseLabels, media=True)

    with pytest.raises(ValueError, match="'g:s2'.*dense.*starts at frame 5"):
        _ = ds.convert_all_labels(
            kind="behavior", source_format=_DenseLabels.src_format
        )

    _assert_no_label_written(ds)


def test_dense_labels_of_a_first_sequence_without_media_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``s1`` sorts first in ``g`` and starts at frame 0 without media."""
    ds = _dense_dataset(tmp_path, monkeypatch, _DenseFirstMember, media=False)

    outcome = ds.convert_all_labels(
        kind="behavior", source_format=_DenseFirstMember.src_format
    )

    assert outcome.written == 2
    entries = read_labels_index(ds, "behavior")[["group", "sequence"]]
    assert sorted(map(tuple, entries.to_numpy().tolist())) == [("", "a"), ("g", "s1")]


def test_dense_labels_of_a_later_sequence_without_media_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without media, the first frame of ``s2``, after ``s1``, is unknown."""
    ds = _dense_dataset(tmp_path, monkeypatch, _DenseLabels, media=False)

    with pytest.raises(
        ValueError, match="'g:s2'.*lacks a known first frame.*media is not indexed"
    ):
        _ = ds.convert_all_labels(
            kind="behavior", source_format=_DenseLabels.src_format
        )

    _assert_no_label_written(ds)


def test_dense_labels_of_a_first_sequence_that_tracks_number_later_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracks variant numbers ``s2`` before ``s1``, so ``s1`` may not start at 0."""
    monkeypatch.setitem(
        LABEL_CONVERTERS, (_DenseFirstMember.src_format, "behavior"), _DenseFirstMember
    )
    ds = make_dataset(tmp_path / "ds")
    src = tmp_path / "uploads"
    src.mkdir()
    np.save(src / "f1.npy", np.zeros(1))
    ds.index_labels_raw(
        [src], patterns=["*.npy"], src_format=_DenseFirstMember.src_format
    )
    add_tracks_table(ds, "convert.a", track_table("g", "s1", n_rows=5, start_frame=4))
    add_tracks_table(ds, "convert.a", track_table("g", "s2", n_rows=4))
    ds.set_continuous_groups(["g"])

    with pytest.raises(ValueError, match="'g:s1'.*dense.*Rename the sequences"):
        _ = ds.convert_all_labels(
            kind="behavior", source_format=_DenseFirstMember.src_format
        )

    _assert_no_label_written(ds)
