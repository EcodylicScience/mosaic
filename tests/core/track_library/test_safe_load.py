"""Raw files a user supplies load without unpickling and within a size limit.

Each hostile file here is built from its header alone, written by hand into a zip
member, so a test of an oversized declaration never allocates the array it
declares.
"""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from mosaic.core.track_library.safe_load import (
    PickledFormatNotTrustedError,
    UnsafeRawFileError,
    check_npz,
    pickled_files_trusted,
    raw_array_limit_bytes,
    require_pickled_trust,
)
from mosaic.core.schema import TrackSchemaError
from tests.helpers import unpickling_member

_GIB = 1024**3


def _header_member(
    shape: tuple[int, ...],
    *,
    dtype: str | list[tuple[str, str]] = "<f8",
    version: tuple[int, int] = (1, 0),
    payload: bytes = b"",
) -> bytes:
    """The bytes of a `.npy` member: a header declaring *shape*, then *payload*."""
    buffer = io.BytesIO()
    header = {"descr": dtype, "fortran_order": False, "shape": shape}
    if version == (1, 0):
        np.lib.format.write_array_header_1_0(buffer, header)
    else:
        np.lib.format.write_array_header_2_0(buffer, header)
    raw = bytearray(buffer.getvalue())
    raw[6:8] = bytes(version)
    return bytes(raw) + payload


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_a_plain_npz_passes(tmp_path: Path) -> None:
    path = tmp_path / "plain.npz"
    np.savez(path, x=np.arange(10.0), y=np.zeros((3, 4), dtype=np.int32))

    check_npz(path)


def test_an_object_member_is_refused_without_being_unpickled(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    path = tmp_path / "hostile.npz"
    payload = unpickling_member(marker)
    np.savez(path, allow_pickle=True, x=np.arange(3.0), payload=payload)

    with pytest.raises(UnsafeRawFileError, match="object") as caught:
        check_npz(path)

    assert caught.value.path == path
    assert caught.value.member == "payload.npy"
    assert not marker.exists(), "the refusal unpickled the member"
    with np.load(path, allow_pickle=True) as data:
        _ = data["payload"]
    assert marker.exists(), "the fixture must unpickle when it is allowed to"


def test_an_oversized_declaration_is_refused_without_allocating_it(
    tmp_path: Path,
) -> None:
    path = _write_zip(tmp_path / "big.npz", {"x.npy": _header_member((3 * _GIB,))})

    with pytest.raises(UnsafeRawFileError, match="declares"):
        check_npz(path)


def test_a_shape_whose_int64_product_wraps_is_refused(tmp_path: Path) -> None:
    shape = (2**32, 2**32)
    assert np.prod(np.array(shape, dtype=np.int64)) == 0, "the product must wrap"
    member = _header_member(shape, dtype="|i1")
    path = _write_zip(tmp_path / "wrap.npz", {"x.npy": member})

    with pytest.raises(UnsafeRawFileError, match="declares"):
        check_npz(path)


def test_a_header_declaring_more_bytes_than_its_member_holds_is_refused(
    tmp_path: Path,
) -> None:
    member = _header_member((100,), payload=b"\0" * 8 * 10)
    path = _write_zip(tmp_path / "short.npz", {"x.npy": member})

    with pytest.raises(UnsafeRawFileError, match="declares") as caught:
        check_npz(path)

    assert caught.value.member == "x.npy"


def test_arrays_past_the_limit_together_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "two.npz"
    np.savez(path, x=np.zeros(8), y=np.zeros(8))
    monkeypatch.setenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", "100")

    with pytest.raises(UnsafeRawFileError, match="MOSAIC_RAW_ARRAY_LIMIT_BYTES"):
        check_npz(path)

    monkeypatch.setenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", "128")
    check_npz(path)


def test_a_member_that_is_not_npy_is_refused(tmp_path: Path) -> None:
    path = _write_zip(
        tmp_path / "extra.npz",
        {"x.npy": _header_member((1,), payload=b"\0" * 8), "notes.pkl": b"\x80\x04."},
    )

    with pytest.raises(UnsafeRawFileError, match=r"\.npy") as caught:
        check_npz(path)

    assert caught.value.member == "notes.pkl"


def test_too_many_members_are_refused(tmp_path: Path) -> None:
    empty = _header_member((0,))
    path = _write_zip(
        tmp_path / "many.npz", {f"m{index}.npy": empty for index in range(4097)}
    )

    with pytest.raises(UnsafeRawFileError, match="4096"):
        check_npz(path)


def test_a_file_that_is_not_a_zip_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "plain.npz"
    with path.open("wb") as stream:
        np.save(stream, np.arange(3.0))

    with pytest.raises(UnsafeRawFileError, match="zip"):
        check_npz(path)


def test_an_npy_with_a_zip_appended_is_refused(tmp_path: Path) -> None:
    """`np.load` decides by the first bytes, so the leading `.npy` would load."""
    archive = _write_zip(
        tmp_path / "inner.npz", {"x.npy": _header_member((1,), payload=b"\0" * 8)}
    )
    leading = _header_member((4,), payload=b"\0" * 32)
    path = tmp_path / "polyglot.npz"
    path.write_bytes(leading + archive.read_bytes())
    assert zipfile.is_zipfile(path), "the fixture must look like a zip from its end"

    with pytest.raises(UnsafeRawFileError, match="zip"):
        check_npz(path)


def test_an_empty_archive_passes(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    np.savez(path)

    check_npz(path)


def test_a_negative_dimension_cannot_offset_an_oversized_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", "1000")
    path = _write_zip(
        tmp_path / "negative.npz",
        {
            "big.npy": _header_member((400,), payload=b"\0" * 8 * 400),
            "offset.npy": _header_member((-1, 10**6)),
        },
    )

    with pytest.raises(UnsafeRawFileError, match="negative") as caught:
        check_npz(path)

    assert caught.value.member == "offset.npy"


@pytest.mark.parametrize("dtype", ["|S0", "|V0"])
def test_a_zero_width_dtype_with_a_huge_shape_is_refused(
    tmp_path: Path, dtype: str
) -> None:
    path = _write_zip(
        tmp_path / "zero.npz", {"x.npy": _header_member((10**7,), dtype=dtype)}
    )

    with pytest.raises(UnsafeRawFileError, match="declares"):
        check_npz(path)


def test_an_empty_structured_dtype_with_a_huge_shape_is_refused(
    tmp_path: Path,
) -> None:
    assert np.dtype([]).itemsize == 0
    member = _header_member((10**7,), dtype=[])
    path = _write_zip(tmp_path / "empty_fields.npz", {"x.npy": member})

    with pytest.raises(UnsafeRawFileError, match="declares"):
        check_npz(path)


def test_format_version_3_is_refused(tmp_path: Path) -> None:
    member = _header_member((1,), version=(3, 0), payload=b"\0" * 8)
    path = _write_zip(tmp_path / "v3.npz", {"x.npy": member})

    with pytest.raises(UnsafeRawFileError, match="version"):
        check_npz(path)


def test_format_version_2_passes(tmp_path: Path) -> None:
    member = _header_member((1,), version=(2, 0), payload=b"\0" * 8)
    path = _write_zip(tmp_path / "v2.npz", {"x.npy": member})

    check_npz(path)


def test_the_errors_reach_the_per_file_failure_handler() -> None:
    """A `TrackSchemaError` is re-raised past it, so these must not be one."""
    assert not issubclass(UnsafeRawFileError, TrackSchemaError)
    assert issubclass(PickledFormatNotTrustedError, UnsafeRawFileError)


def test_the_limit_defaults_to_two_gib_and_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", raising=False)
    assert raw_array_limit_bytes() == 2 * _GIB

    monkeypatch.setenv("MOSAIC_RAW_ARRAY_LIMIT_BYTES", "4096")
    assert raw_array_limit_bytes() == 4096


def test_pickled_files_are_untrusted_unless_the_flag_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOSAIC_TRUST_PICKLED_FILES", raising=False)
    assert not pickled_files_trusted()

    monkeypatch.setenv("MOSAIC_TRUST_PICKLED_FILES", "1")
    assert pickled_files_trusted()


def test_the_trust_refusal_names_the_file_the_variable_and_the_safe_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOSAIC_TRUST_PICKLED_FILES", raising=False)
    path = tmp_path / "calms21.npy"

    with pytest.raises(PickledFormatNotTrustedError) as caught:
        require_pickled_trust(path, "pickled .npy", "JSON")

    message = str(caught.value)
    assert str(path) in message
    assert "MOSAIC_TRUST_PICKLED_FILES=1" in message
    assert "JSON" in message
    assert caught.value.path == path

    monkeypatch.setenv("MOSAIC_TRUST_PICKLED_FILES", "1")
    require_pickled_trust(path, "pickled .npy", "JSON")
