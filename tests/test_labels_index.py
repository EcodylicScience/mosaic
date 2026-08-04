"""The typed per-kind labels index, its resolver, and the conversion flow (item 9.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.composition import SourceMember, labels_raw_composition
from mosaic.core.pipeline.labels_index import (
    LABELS_INDEX_COLUMNS,
    empty_labels_frame,
    read_labels_index,
    select_label_variant_rows,
    write_labels_row,
)
from mosaic.core.pipeline.sequence_index import write_sequence_compositions


def _dataset(tmp_path: Path) -> Dataset:
    return Dataset(new_dataset_manifest("t", tmp_path / "ds")).load()


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(["group", "sequence", "run_id"]))


# --- select_label_variant_rows -------------------------------------------------


def test_absent_index_reads_as_empty_full_schema(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
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
    ds = _dataset(tmp_path)
    src = tmp_path / "uploads"
    src.mkdir()
    (src / "f1.csv").write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="deeplabcut"):
        _ = ds.index_labels_raw([src], patterns=["*.csv"], src_format="deeplabcut")


def test_a_registered_label_format_still_indexes(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
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
