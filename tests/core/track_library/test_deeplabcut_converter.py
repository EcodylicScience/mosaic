"""Tests for the generic DeepLabCut track converter."""

from __future__ import annotations

import math
from pathlib import Path
import pickle
import struct
import tracemalloc
from typing import Literal
import warnings

import h5py
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.deeplabcut import DlcConverter, DlcParams, load_dlc
from mosaic.core.track_library.safe_load import (
    TRUST_VARIABLE,
    PickledFormatNotTrustedError,
    UnsafeRawFileError,
    check_hdf5_file,
    read_pandas_hdf_frame,
)

from tests.helpers import unpickling_member, write_dlc_csv

_BODYPARTS = ["snout", "midbody", "tailtip"]
_BLOSC_FILTER = 32001
# The bytes one decoded label takes besides its stored bytes.
_PYTHON_OBJECT_BYTES = 64


def test_load_dlc_single_animal(tmp_path: Path) -> None:
    csv = tmp_path / "single.csv"
    vals = write_dlc_csv(csv, _BODYPARTS, n_frames=20)

    individuals = load_dlc(csv)
    assert len(individuals) == 1
    indiv = individuals[0]
    assert indiv.bodyparts == _BODYPARTS
    assert indiv.x.shape == (20, len(_BODYPARTS))
    assert indiv.likelihood is not None
    np.testing.assert_allclose(indiv.x[:, 0], vals[:, 0, 0], rtol=1e-5)
    np.testing.assert_allclose(indiv.likelihood[:, 1], vals[:, 1, 2], rtol=1e-5)


def test_dlc_converter_single_animal_schema(tmp_path: Path) -> None:
    csv = tmp_path / "single.csv"
    write_dlc_csv(csv, _BODYPARTS, n_frames=20)

    df = DlcConverter().convert(
        csv, DlcParams(fps=50.0), EntryHints(group="g1", sequence="s1")
    )

    # Required trex_v1 columns + pose prefixes present.
    _, report = ensure_track_schema(df, "trex_v1", strict=True, source=str(csv))
    assert report["missing_required"] == []
    assert report["missing_prefixes"] == []

    assert len(df) == 20
    assert set(df["id"]) == {0}
    assert (df["group"] == "g1").all()
    assert (df["sequence"] == "s1").all()
    # 3 keypoints -> poseX0..2, poseY0..2, poseP0..2
    for k in range(len(_BODYPARTS)):
        assert f"poseX{k}" in df.columns
        assert f"poseY{k}" in df.columns
        assert f"poseP{k}" in df.columns
    # time derived from fps
    np.testing.assert_allclose(df["time"].to_numpy(), np.arange(20) / 50.0)
    # centroid equals mean of keypoint x's
    pose_x = df[[f"poseX{k}" for k in range(len(_BODYPARTS))]].to_numpy()
    np.testing.assert_allclose(
        df["X"].to_numpy(), np.nanmean(pose_x, axis=1), rtol=1e-6
    )


def test_dlc_converter_multi_animal(tmp_path: Path) -> None:
    csv = tmp_path / "multi.csv"
    individuals = ("fish0", "fish1")
    _ = write_dlc_csv(csv, _BODYPARTS, n_frames=15, individuals=individuals)
    n = len(individuals)

    assert len(load_dlc(csv)) == n

    df = DlcConverter().convert(
        csv, DlcParams(fps=30.0), EntryHints(group="g", sequence="rec")
    )
    assert set(df["id"]) == set(range(n))
    # Each individual contributes n_frames rows.
    assert len(df) == 15 * n


def test_dlc_converter_csv_h5_roundtrip(tmp_path: Path) -> None:
    """A DLC export saved as HDF5 yields the same poses as the CSV form."""
    csv = tmp_path / "single.csv"
    write_dlc_csv(csv, _BODYPARTS, n_frames=12)

    indiv = load_dlc(csv)[0]
    # Build the equivalent maDLC-less HDF5 with a (scorer, bodypart, coord) index.
    cols = pd.MultiIndex.from_tuples(
        [("DLC_model", bp, c) for bp in _BODYPARTS for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    flat = np.empty((12, len(_BODYPARTS) * 3))
    for b in range(len(_BODYPARTS)):
        flat[:, b * 3 + 0] = indiv.x[:, b]
        flat[:, b * 3 + 1] = indiv.y[:, b]
        assert indiv.likelihood is not None
        flat[:, b * 3 + 2] = indiv.likelihood[:, b]
    h5 = tmp_path / "single.h5"
    pd.DataFrame(flat, columns=cols).to_hdf(h5, key="df", mode="w")

    converter = DlcConverter()
    df_csv = converter.convert(csv, DlcParams(fps=30.0), EntryHints())
    df_h5 = converter.convert(h5, DlcParams(fps=30.0), EntryHints())
    np.testing.assert_allclose(
        df_csv[["poseX0", "poseY1", "poseP2"]].to_numpy(),
        df_h5[["poseX0", "poseY1", "poseP2"]].to_numpy(),
        rtol=1e-6,
    )


def _dlc_frame(individuals: tuple[str, ...]) -> pd.DataFrame:
    """A DeepLabCut pose table with a missing value, as DeepLabCut writes it."""
    levels: list[list[str]] = [["DLC_model"]]
    names = ["scorer"]
    if individuals:
        levels.append(list(individuals))
        names.append("individuals")
    levels += [_BODYPARTS, ["x", "y", "likelihood"]]
    names += ["bodyparts", "coords"]
    columns = pd.MultiIndex.from_product(levels, names=names)
    values = np.random.default_rng(0).uniform(0, 100, (7, len(columns)))
    values[2, 1] = np.nan
    return pd.DataFrame(values, columns=columns)


def _pandas_frame(path: Path) -> pd.DataFrame:
    """The frame that pandas itself reads from *path*."""
    frame = pd.read_hdf(path)
    assert isinstance(frame, pd.DataFrame)
    return frame


@pytest.fixture
def untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the trust flag, so a layout the reader refuses cannot fall back."""
    monkeypatch.delenv(TRUST_VARIABLE, raising=False)


@pytest.mark.usefixtures("untrusted")
@pytest.mark.parametrize("fmt", ["fixed", "table"])
@pytest.mark.parametrize("individuals", [(), ("fish0", "fish1")])
def test_a_pandas_hdf_frame_reads_as_pandas_reads_it(
    tmp_path: Path, fmt: Literal["fixed", "table"], individuals: tuple[str, ...]
) -> None:
    path = tmp_path / "pose.h5"
    _dlc_frame(individuals).to_hdf(path, key="df_with_missing", mode="w", format=fmt)

    assert_frame_equal(read_pandas_hdf_frame(path), _pandas_frame(path))


@pytest.mark.usefixtures("untrusted")
@pytest.mark.parametrize("fmt", ["fixed", "table"])
def test_a_frame_of_several_dtypes_reads_in_its_column_order(
    tmp_path: Path, fmt: Literal["fixed", "table"]
) -> None:
    path = tmp_path / "mixed.h5"
    frame = pd.DataFrame(
        {"a": [1.5, 2.5], "b": np.array([1, 2], dtype=np.int32), "c": [0.5, np.nan]},
        index=pd.Index([10, 20], name="frame"),
    )
    frame.to_hdf(path, key="pose", mode="w", format=fmt)

    assert_frame_equal(read_pandas_hdf_frame(path), _pandas_frame(path))


def test_an_object_column_is_refused_before_anything_unpickles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "marker"
    path = tmp_path / "hostile.h5"
    frame = pd.DataFrame({"x": [1.0], "payload": unpickling_member(marker)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        frame.to_hdf(path, key="df_with_missing", mode="w", format="fixed")
    monkeypatch.setenv(TRUST_VARIABLE, "1")

    with pytest.raises(UnsafeRawFileError, match="variable-length"):
        read_pandas_hdf_frame(path)

    assert not marker.exists(), "the refusal unpickled the payload"
    _ = pd.read_hdf(path)
    assert marker.exists(), "the fixture must unpickle when pandas reads it"


def _pose_group(path: Path) -> h5py.File:
    """An HDF5 file holding one group with a small dataset, open for writing."""
    handle = h5py.File(path, "w")
    handle.create_group("pose").create_dataset("values", data=np.zeros(4))
    return handle


def test_a_soft_link_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "soft.h5"
    with _pose_group(path) as handle:
        handle["pose/alias"] = h5py.SoftLink("/pose/values")

    with pytest.raises(UnsafeRawFileError, match="link") as caught:
        read_pandas_hdf_frame(path)

    assert caught.value.member == "pose/alias"


def test_an_external_link_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as handle:
        handle["data"] = np.zeros(4)
    path = tmp_path / "external_link.h5"
    with _pose_group(path) as handle:
        handle["pose/elsewhere"] = h5py.ExternalLink(str(other), "/data")

    with pytest.raises(UnsafeRawFileError, match="link"):
        read_pandas_hdf_frame(path)


def test_a_dataset_stored_in_another_file_is_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"\0" * 32)
    path = tmp_path / "external_storage.h5"
    with _pose_group(path) as handle:
        handle.create_dataset(
            "pose/outside", shape=(4,), dtype="<f8", external=[(str(raw), 0, 32)]
        )

    with pytest.raises(UnsafeRawFileError, match="external") as caught:
        read_pandas_hdf_frame(path)

    assert caught.value.member == "pose/outside"


def test_a_virtual_dataset_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "virtual.h5"
    with _pose_group(path) as handle:
        layout = h5py.VirtualLayout(shape=(4,), dtype="<f8")
        layout[:] = h5py.VirtualSource(".", "pose/values", shape=(4,))
        handle.create_virtual_dataset("pose/mapped", layout)

    with pytest.raises(UnsafeRawFileError, match="virtual"):
        read_pandas_hdf_frame(path)


def test_a_huge_declared_shape_is_refused_without_allocating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "huge.h5"
    with _pose_group(path) as handle:
        handle.create_dataset("pose/huge", shape=(2**40,), dtype="<f8", chunks=(1024,))
    monkeypatch.setenv(TRUST_VARIABLE, "1")

    with pytest.raises(UnsafeRawFileError, match="MOSAIC_RAW_ARRAY_LIMIT_BYTES"):
        read_pandas_hdf_frame(path)


@pytest.mark.usefixtures("untrusted")
def test_a_file_must_hold_exactly_one_top_level_group(tmp_path: Path) -> None:
    path = tmp_path / "two.h5"
    with _pose_group(path) as handle:
        handle.create_group("second")

    with pytest.raises(PickledFormatNotTrustedError, match="one top-level group"):
        read_pandas_hdf_frame(path)


def test_an_unknown_layout_loads_only_when_pickled_files_are_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dated.h5"
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2)
    )
    frame.to_hdf(path, key="df_with_missing", mode="w", format="fixed")
    monkeypatch.delenv(TRUST_VARIABLE, raising=False)

    with pytest.raises(PickledFormatNotTrustedError, match="CSV"):
        read_pandas_hdf_frame(path)

    monkeypatch.setenv(TRUST_VARIABLE, "1")
    assert_frame_equal(read_pandas_hdf_frame(path), _pandas_frame(path))


def test_the_converter_reads_h5_through_the_safe_reader(tmp_path: Path) -> None:
    path = tmp_path / "soft.h5"
    with _pose_group(path) as handle:
        handle["pose/alias"] = h5py.SoftLink("/pose/values")

    with pytest.raises(UnsafeRawFileError) as caught:
        load_dlc(path)

    assert caught.value.member == "pose/alias"


def _with_attribute(tmp_path: Path, name: str, payload: bytes) -> Path:
    """A DeepLabCut table-format export whose group attribute *name* is *payload*."""
    written = tmp_path / "written.h5"
    _dlc_frame(()).to_hdf(written, key="df_with_missing", mode="w", format="table")
    # A group made in the latest format stores an attribute past the 64 KiB
    # that a node's header holds, which the group PyTables writes cannot. The
    # table's own index is left behind, since the reader does not read it.
    path = tmp_path / "pose.h5"
    with (
        h5py.File(written, "r") as source,
        h5py.File(path, "w", libver="latest") as handle,
    ):
        original = source["df_with_missing"]
        table = source["df_with_missing/table"]
        assert isinstance(original, h5py.Group)
        assert isinstance(table, h5py.Dataset)
        group = handle.create_group("df_with_missing")
        for key, value in original.attrs.items():
            group.attrs[key] = value
        group.copy(table, "table")
        group.attrs[name] = np.bytes_(payload)
    return path


def _calling_payloads(marker: Path) -> dict[str, bytes]:
    """Pickles that call code when unpickled: through a global and a persistent id."""
    return {
        "global": pickle.dumps(unpickling_member(marker)[0], protocol=0),
        "persistent id": b"Pmarker\n.",
    }


@pytest.mark.usefixtures("untrusted")
@pytest.mark.parametrize("payload_name", ["global", "persistent id"])
@pytest.mark.parametrize("attribute", ["pandas_type", "info"])
def test_a_pickled_attribute_that_calls_code_is_refused_without_running(
    tmp_path: Path, payload_name: str, attribute: str
) -> None:
    marker = tmp_path / "marker"
    payload = _calling_payloads(marker)[payload_name]
    path = _with_attribute(tmp_path, attribute, payload)

    with pytest.raises(PickledFormatNotTrustedError, match=attribute):
        read_pandas_hdf_frame(path)

    assert not marker.exists(), "the refusal ran the pickle"


def _memo_bomb(text_bytes: int, references: int) -> bytes:
    """A pickle of one string, memoized, then a list naming it *references* times."""
    text = b"x" * text_bytes
    head = b"\x80\x04X" + struct.pack("<I", len(text)) + text + b"\x94("
    return head + b"h\x00" * references + b"l."


@pytest.mark.usefixtures("untrusted")
@pytest.mark.parametrize(
    ("payload", "refusal"),
    [
        (_memo_bomb(30_000, 15_000), "pandas"),
        (b"\x80\x02" + b"]" * 20_000 + b"a" * 19_999 + b".", "nests"),
        (b"\x80\x02]q\x00h\x00a.", "nests"),
    ],
    ids=["memo-bomb", "deep-nesting", "self-reference"],
)
@pytest.mark.parametrize("attribute", ["pandas_type", "index_cols"])
def test_a_pickled_attribute_that_expands_is_refused_in_bounded_memory(
    tmp_path: Path, payload: bytes, refusal: str, attribute: str
) -> None:
    path = _with_attribute(tmp_path, attribute, payload)

    tracemalloc.start()
    try:
        with pytest.raises(PickledFormatNotTrustedError, match=refusal) as caught:
            read_pandas_hdf_frame(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(str(caught.value)) < 1_000
    assert peak < 20 * 1024**2


def _with_group_attribute(
    tmp_path: Path, name: str, value: object, dtype: np.dtype[np.generic] | None = None
) -> Path:
    """A DeepLabCut fixed-format export whose group attribute *name* is *value*."""
    path = tmp_path / "pose.h5"
    _dlc_frame(()).to_hdf(path, key="df_with_missing", mode="w", format="fixed")
    with h5py.File(path, "a") as handle:
        group = handle["df_with_missing"]
        del group.attrs[name]
        group.attrs.create(name, value, dtype=dtype)
    return path


@pytest.mark.parametrize(
    "attribute",
    [
        pytest.param(_memo_bomb(100_000, 10_000), id="pickle-over-cap"),
        pytest.param(b"x" * 100_000, id="text-over-cap"),
    ],
)
def test_an_attribute_past_the_cap_is_refused_unread(
    tmp_path: Path, attribute: bytes
) -> None:
    path = _with_attribute(tmp_path, "pandas_type", attribute)

    with pytest.raises(UnsafeRawFileError, match="attribute pandas_type") as caught:
        read_pandas_hdf_frame(path)

    assert not isinstance(caught.value, PickledFormatNotTrustedError)


def test_a_variable_length_attribute_is_refused_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _with_group_attribute(
        tmp_path, "pandas_type", ["frame", "frame"], dtype=h5py.string_dtype()
    )
    read_names: list[str] = []
    read_attribute = h5py.AttributeManager.__getitem__

    def recording_read(manager: h5py.AttributeManager, name: str) -> object:
        read_names.append(name)
        return read_attribute(manager, name)

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", recording_read)
    monkeypatch.setenv(TRUST_VARIABLE, "1")

    tracemalloc.start()
    try:
        with pytest.raises(UnsafeRawFileError, match="variable-length") as caught:
            read_pandas_hdf_frame(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert caught.value.member == "df_with_missing"
    assert peak < 20 * 1024**2
    assert "pandas_type" not in read_names


@pytest.mark.usefixtures("untrusted")
def test_an_array_where_a_scalar_attribute_belongs_is_refused(tmp_path: Path) -> None:
    path = _with_group_attribute(
        tmp_path, "pandas_type", np.array([b"frame", b"frame"])
    )

    with pytest.raises(PickledFormatNotTrustedError, match="pandas_type"):
        read_pandas_hdf_frame(path)


def test_a_string_index_that_expands_past_the_limit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labeled.h5"
    rows = 500
    frame = pd.DataFrame(
        {"x": np.zeros(rows)},
        index=pd.Index([f"{row % 100:02d}" for row in range(rows)]),
    )
    frame.to_hdf(path, key="df_with_missing", mode="w", format="fixed")
    monkeypatch.setenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", "10000")
    with h5py.File(path, "r") as handle:
        labels = handle["df_with_missing/axis1"]
        assert isinstance(labels, h5py.Dataset)
        assert labels.dtype.itemsize * rows < 10_000

    with pytest.raises(
        UnsafeRawFileError, match="MOSAIC_RAW_ARRAY_LIMIT_BYTES"
    ) as caught:
        read_pandas_hdf_frame(path)

    assert not isinstance(caught.value, PickledFormatNotTrustedError)


@pytest.mark.usefixtures("untrusted")
def test_a_string_level_of_a_fixed_multiindex_is_charged_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "levels.h5"
    columns = pd.MultiIndex.from_product(
        [[f"animal{number:03d}" for number in range(40)], ["x", "y"]],
        names=["individuals", "coords"],
    )
    pd.DataFrame(np.zeros((3, len(columns))), columns=columns).to_hdf(
        path, key="pose", mode="w", format="fixed"
    )
    with h5py.File(path, "r") as handle:
        declared = check_hdf5_file(path, handle)
        pose = handle["pose"]
        assert isinstance(pose, h5py.Group)
        decoded = 0
        for name in pose.keys():
            node = pose[name]
            if not isinstance(node, h5py.Dataset):
                continue
            if any(part in name for part in ("_level", "_label", "_items")):
                per_element = _PYTHON_OBJECT_BYTES
                if node.dtype.kind == "S":
                    per_element += 4 * node.dtype.itemsize
                decoded += math.prod(node.shape or ()) * per_element
    limit = declared + decoded

    monkeypatch.setenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", str(limit))
    assert_frame_equal(read_pandas_hdf_frame(path), _pandas_frame(path))

    monkeypatch.setenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", str(limit - 1))
    with pytest.raises(UnsafeRawFileError, match="MOSAIC_RAW_ARRAY_LIMIT_BYTES"):
        read_pandas_hdf_frame(path)


@pytest.mark.skipif(
    h5py.h5z.filter_avail(_BLOSC_FILTER), reason="h5py reads blosc here"
)
def test_a_filter_only_pytables_reads_loads_only_when_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "blosc.h5"
    _dlc_frame(()).to_hdf(
        path,
        key="df_with_missing",
        mode="w",
        format="fixed",
        complib="blosc",
        complevel=5,
    )
    monkeypatch.delenv(TRUST_VARIABLE, raising=False)

    with pytest.raises(PickledFormatNotTrustedError, match="filter"):
        read_pandas_hdf_frame(path)

    monkeypatch.setenv(TRUST_VARIABLE, "1")
    assert_frame_equal(read_pandas_hdf_frame(path), _pandas_frame(path))


def test_a_datatype_without_a_numpy_equivalent_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "time.h5"
    with _pose_group(path) as handle:
        group = handle["pose"]
        assert isinstance(group, h5py.Group)
        space = h5py.h5s.create_simple((4,))
        h5py.h5d.create(group.id, b"stamps", h5py.h5t.UNIX_D32LE, space)

    with pytest.raises(UnsafeRawFileError, match="datatype") as caught:
        read_pandas_hdf_frame(path)

    assert caught.value.member == "pose/stamps"
