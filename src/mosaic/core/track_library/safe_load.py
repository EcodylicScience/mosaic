"""Load raw files that a user, or an upload, supplied.

A supplied file is untrusted. A loader here never unpickles one, so the file
cannot run code, and refuses arrays whose declared sizes add up past
`raw_array_limit_bytes`, so it cannot exhaust memory. A format that can only be
read by unpickling loads only when `MOSAIC_TRUST_PICKLED_FILES=1` says the
files are trusted.

The refusals are `UnsafeRawFileError`, not a `TrackSchemaError`, so a batch
conversion records the refused file as failed and goes on to the next one.
"""

from __future__ import annotations

import codecs
from collections.abc import Mapping
from dataclasses import dataclass
import io
import math
from pathlib import Path
import pickle
import posixpath
import reprlib
from typing import TYPE_CHECKING, NoReturn
import zipfile

import numpy as np
import pandas as pd

from mosaic.core.narrowing import (
    is_object_list,
    is_object_mapping,
    is_object_set,
    is_object_tuple,
)
from mosaic.env_settings import flag_env, int_env

if TYPE_CHECKING:
    import h5py

__all__ = [
    "LIMIT_VARIABLE",
    "TRUST_VARIABLE",
    "PickledFormatNotTrustedError",
    "UnsafeRawFileError",
    "check_hdf5_file",
    "check_hdf5_member",
    "check_npz",
    "pickled_files_trusted",
    "raw_array_limit_bytes",
    "read_pandas_hdf_frame",
    "require_pickled_trust",
    "require_within_limit",
]

LIMIT_VARIABLE = "MOSAIC_RAW_ARRAY_LIMIT_BYTES"
"""The environment variable that sets the size limit, in bytes."""

TRUST_VARIABLE = "MOSAIC_TRUST_PICKLED_FILES"
"""The environment variable that, set to ``1``, trusts pickled files."""

_DEFAULT_LIMIT_BYTES = 2 * 1024**3
_MAX_NPZ_MEMBERS = 4096
_ZIP_LOCAL_HEADER = b"PK\x03\x04"
_ZIP_SIGNATURES = (_ZIP_LOCAL_HEADER, b"PK\x05\x06")
_HEADER_READERS = {
    (1, 0): np.lib.format.read_array_header_1_0,
    (2, 0): np.lib.format.read_array_header_2_0,
}


class UnsafeRawFileError(ValueError):
    """A supplied raw file is refused because loading it would be unsafe.

    Attributes:
        path: The refused file.
        reason: Why the file is refused.
        member: The member refused, or ``None`` for the whole file.
    """

    def __init__(self, path: Path, reason: str, *, member: str | None = None) -> None:
        where = f"{path} (member {member})" if member is not None else str(path)
        super().__init__(f"{where}: {reason}")
        self.path = path
        self.reason = reason
        self.member = member


class PickledFormatNotTrustedError(UnsafeRawFileError):
    """A file in a format that loads by unpickling, without the trust flag."""


def raw_array_limit_bytes() -> int:
    """The most bytes the arrays of one supplied file may declare in total.

    Returns:
        ``MOSAIC_RAW_ARRAY_LIMIT_BYTES``, or 2 GiB when it is unset.
    """
    return int_env(LIMIT_VARIABLE, _DEFAULT_LIMIT_BYTES)


def pickled_files_trusted() -> bool:
    """Whether ``MOSAIC_TRUST_PICKLED_FILES=1`` trusts files that unpickle."""
    return flag_env(TRUST_VARIABLE)


def require_pickled_trust(path: Path, fmt: str, safe_form: str) -> None:
    """Refuse *path*, a file that loads by unpickling, unless pickles are trusted.

    Args:
        path: The file about to be loaded.
        fmt: The file's format, as the refusal names it.
        safe_form: A form of the same data that loads without unpickling, such
            as JSON or CSV.

    Raises:
        PickledFormatNotTrustedError: If `pickled_files_trusted` is false.
    """
    if pickled_files_trusted():
        return
    reason = (
        f"a {fmt} file loads by unpickling, which can run code. Convert it to "
        f"{safe_form}, or set {TRUST_VARIABLE}=1 if you trust where it came from."
    )
    raise PickledFormatNotTrustedError(path, reason)


def check_npz(path: Path) -> None:
    """Refuse an ``.npz`` that cannot be loaded safely, from its headers alone.

    No member's data is read. The file must start as a zip archive does, the
    test ``np.load`` makes. Each member must be a ``.npy`` of format 1.0 or 2.0
    whose dtype holds no Python objects, which ``np.load`` would unpickle, and
    whose shape, with no negative dimension, declares a size that fits in the
    member. The declared sizes together must fit in `raw_array_limit_bytes`.

    Args:
        path: The file to check.

    Raises:
        UnsafeRawFileError: If the file fails any of these checks.
    """
    with path.open("rb") as stream:
        leading = stream.read(len(_ZIP_LOCAL_HEADER))
    # `np.load` decides from the leading bytes, and `is_zipfile` from the end
    # record, so a file must pass both to be read as the archive checked here.
    if leading not in _ZIP_SIGNATURES or not zipfile.is_zipfile(path):
        raise UnsafeRawFileError(path, "not a zip archive, so not an .npz")
    total = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > _MAX_NPZ_MEMBERS:
            reason = f"holds {len(members)} members, more than {_MAX_NPZ_MEMBERS}"
            raise UnsafeRawFileError(path, reason)
        for info in members:
            total += _declared_bytes(path, archive, info)
    require_within_limit(path, total, "its arrays declare")


def require_within_limit(path: Path, total: int, measure: str) -> None:
    """Refuse *path* if *total* bytes exceed `raw_array_limit_bytes`.

    Args:
        path: The file, as the refusal names it.
        total: The bytes the file's arrays take.
        measure: How the refusal says they take them, such as ``"its arrays
            declare"``.

    Raises:
        UnsafeRawFileError: If *total* exceeds the limit.
    """
    limit = raw_array_limit_bytes()
    if total > limit:
        reason = (
            f"{measure} {total} bytes, more than the {limit} that "
            f"{LIMIT_VARIABLE} allows"
        )
        raise UnsafeRawFileError(path, reason)


def _declared_bytes(path: Path, archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> int:
    """The bytes that member *info* declares, refusing an unsafe header."""
    member = info.filename
    if not member.endswith(".npy"):
        raise UnsafeRawFileError(path, "not an .npy array", member=member)
    with archive.open(info) as stream:
        try:
            version = np.lib.format.read_magic(stream)
        except ValueError as exc:
            raise UnsafeRawFileError(
                path, f"not an .npy array: {exc}", member=member
            ) from exc
        read_header = _HEADER_READERS.get(version)
        if read_header is None:
            reason = f"format version {version[0]}.{version[1]} is not 1.0 or 2.0"
            raise UnsafeRawFileError(path, reason, member=member)
        try:
            shape, _, dtype = read_header(stream)
        except ValueError as exc:
            reason = f"unreadable .npy header: {exc}"
            raise UnsafeRawFileError(path, reason, member=member) from exc
    if dtype.hasobject:
        reason = "holds Python objects, which load only by unpickling"
        raise UnsafeRawFileError(path, reason, member=member)
    if any(dimension < 0 for dimension in shape):
        reason = f"declares a negative dimension in shape {shape}"
        raise UnsafeRawFileError(path, reason, member=member)
    # A zero-width dtype costs memory per element once loaded, so each element
    # counts as at least one byte.
    declared = math.prod(shape) * max(dtype.itemsize, 1)
    if declared > info.file_size:
        reason = f"declares {declared} bytes but holds {info.file_size}"
        raise UnsafeRawFileError(path, reason, member=member)
    return declared


def check_hdf5_member(path: Path, group: h5py.Group, name: str) -> int:
    """Refuse member *name* of *group* if reading it is unsafe, from its metadata.

    No data is read. The member must be a hard link, since a soft or external
    link can point outside the file. A dataset must keep its data in the file,
    neither in external storage nor mapped from other datasets, and must hold
    no variable-length, reference or object values, which h5py returns as
    Python objects.

    Args:
        path: The file that *group* belongs to, as the refusal names it.
        group: The group holding the member.
        name: The member's path relative to *group*.

    Returns:
        The bytes the member declares: its shape times its item size for a
        dataset, and 0 for a group.

    Raises:
        UnsafeRawFileError: If the member fails any of these checks, or has a
            datatype that h5py cannot read.
    """
    import h5py

    member = posixpath.join(group.name or "/", name).lstrip("/")
    link = group.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        kind = "an external" if isinstance(link, h5py.ExternalLink) else "a soft"
        reason = f"is {kind} link, which can point outside the file"
        raise UnsafeRawFileError(path, reason, member=member)
    target = group[name]
    if not isinstance(target, h5py.Dataset):
        return 0
    if target.is_virtual:
        reason = "is a virtual dataset, which maps data from other datasets"
        raise UnsafeRawFileError(path, reason, member=member)
    if target.external:
        reason = "keeps its data in external files"
        raise UnsafeRawFileError(path, reason, member=member)
    try:
        dtype = target.dtype
    except TypeError as exc:
        reason = "has a datatype with no numpy equivalent"
        raise UnsafeRawFileError(path, reason, member=member) from exc
    if dtype.hasobject:
        reason = "holds variable-length, reference or object values"
        raise UnsafeRawFileError(path, reason, member=member)
    if target.shape is None:
        return 0
    return math.prod(target.shape) * max(dtype.itemsize, 1)


def check_hdf5_file(path: Path, handle: h5py.File) -> int:
    """Refuse an open HDF5 file whose members cannot be read safely.

    Each link in the file passes `check_hdf5_member`, and the bytes its
    datasets declare together fit in `raw_array_limit_bytes`. No data is read.

    Args:
        path: The file, as the refusal names it.
        handle: The file, open for reading.

    Returns:
        The bytes the file's datasets declare.

    Raises:
        UnsafeRawFileError: If a member fails `check_hdf5_member`, or the
            declared sizes add up past the limit.
    """
    names: list[str] = []
    handle.visit_links(names.append)
    total = sum(check_hdf5_member(path, handle, name) for name in names)
    require_within_limit(path, total, "its datasets declare")
    return total


def read_pandas_hdf_frame(path: Path) -> pd.DataFrame:
    """Read a DataFrame that pandas wrote to HDF5, without unpickling.

    The file passes `check_hdf5_file` and holds exactly one top-level group, a
    DataFrame in pandas' ``fixed`` or ``table`` format, whatever its key. The
    frame's values are numbers or booleans, and its index and column levels
    numbers or strings. The pickled attributes of the ``table`` format decode
    to builtin containers and scalars only, nested a few levels deep at most.
    The Python objects that index labels decode to count toward the limit with
    the declared sizes.

    A file in another layout loads through `pd.read_hdf`, which unpickles,
    only when `pickled_files_trusted` says so. A file refused as unsafe never
    does.

    Args:
        path: The ``.h5`` file to read.

    Returns:
        The frame, equal to what `pd.read_hdf` returns for the file.

    Raises:
        UnsafeRawFileError: If the file fails `check_hdf5_file` or is not HDF5.
        PickledFormatNotTrustedError: If the layout is another one, without the
            trust flag.
        ValueError: If the trusted fallback reads an object other than a
            DataFrame.
    """
    # Deferred for the reason the SLEAP reader in ``sleap.py`` gives.
    import h5py

    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise UnsafeRawFileError(path, f"not a readable HDF5 file: {exc}") from exc
    with handle:
        declared = check_hdf5_file(path, handle)
        try:
            return _PandasGroupReader.open(path, handle, declared).frame()
        except _UnrecognizedLayoutError as exc:
            refusal = exc
    if not pickled_files_trusted():
        reason = (
            f"{refusal.reason}. Only pandas reads that layout, and it unpickles, "
            f"which can run code. Convert the file to CSV, or set "
            f"{TRUST_VARIABLE}=1 if you trust where it came from."
        )
        raise PickledFormatNotTrustedError(path, reason) from refusal
    frame = pd.read_hdf(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{path}: holds a {type(frame).__name__}, not a DataFrame")
    return frame


class _UnrecognizedLayoutError(UnsafeRawFileError):
    """A file laid out in a way the reader does not know, but not unsafe."""


class _ForbiddenPickleError(Exception):
    """A pickle that asks for an object other than a builtin one."""


class _BuiltinUnpickler(pickle.Unpickler):
    """An unpickler that builds builtin containers and scalars only.

    A pickle calls code only through a callable it gets from `find_class`,
    which also resolves the extension registry, or from `persistent_load`. Both
    raise, so the pickle can call nothing. A pickle that needs a callable, such
    as a numpy scalar's, is a layout the reader refuses.
    """

    def find_class(self, module_name: str, global_name: str, /) -> NoReturn:
        raise _ForbiddenPickleError(f"{module_name}.{global_name}")

    def persistent_load(self, pid: object, /) -> NoReturn:
        raise _ForbiddenPickleError("a persistent object")


_UNREADABLE_PICKLE = (
    pickle.UnpicklingError,
    EOFError,
    ValueError,
    OverflowError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
)
# PyTables stores an attribute in its node's header, which holds 64 KiB, so an
# attribute pandas writes, a pickle included, is smaller unless the file uses
# the latest format.
_MAX_ATTRIBUTE_BYTES = 64 * 1024
# pandas nests its pickled attributes four levels deep at most:
# ``non_index_axes`` is a list of tuples of a list of tuples.
_MAX_PICKLE_DEPTH = 8
# A decoded label is a Python object held through an 8-byte pointer: a short
# str takes about 50 bytes and an int 28. A str takes up to four bytes per
# character, and each stored byte decodes to one character at most.
_PYTHON_OBJECT_BYTES = 64
_STRING_BYTES_PER_STORED_BYTE = 4
_BRIEF = reprlib.Repr(
    maxlevel=2,
    maxtuple=4,
    maxlist=4,
    maxdict=4,
    maxset=4,
    maxfrozenset=4,
    maxstring=60,
    maxlong=40,
    maxother=60,
)
_NUMERIC_KINDS = {"integer": "iu", "float": "f", "bool": "b"}
_VALUE_KINDS = "biuf"


def _as_list(value: object) -> list[object] | None:
    """*value* if it is a list, which is what a pickled list decodes to."""
    return value if is_object_list(value) else None


def _as_mapping(value: object) -> Mapping[object, object] | None:
    """*value* if it is a mapping, which is what a pickled dict decodes to."""
    return value if is_object_mapping(value) else None


def _brief(value: object) -> str:
    """A short repr of *value*, which a refusal can show whatever its size."""
    return _BRIEF.repr(value)


def _children(value: object) -> list[object]:
    """The items of a container that a pickle decodes to, keys included."""
    if is_object_list(value) or is_object_tuple(value) or is_object_set(value):
        return list(value)
    if is_object_mapping(value):
        return [*value.keys(), *value.values()]
    return []


def _nests_within(value: object, depth: int) -> bool:
    """Whether *value* nests containers *depth* levels deep at most.

    A container that holds itself nests without end, so it never does. Each
    level visits each container once, however many times the pickle refers to
    it.
    """
    level = [value]
    for _ in range(depth):
        nested = {id(child): child for item in level for child in _children(item)}
        level = [child for child in nested.values() if _children(child)]
        if not level:
            return True
    return False


def _column_labels(non_index_axes: object) -> list[object] | None:
    """The column labels of a table's ``non_index_axes``, ``[(1, labels)]``."""
    axes = _as_list(non_index_axes)
    if axes is None or len(axes) != 1:
        return None
    axis = axes[0]
    if not is_object_tuple(axis) or len(axis) != 2 or axis[0] != 1:
        return None
    return _as_list(axis[1])


@dataclass
class _PandasGroupReader:
    """Reads the one pandas group of a file the way pandas' own reader does.

    Attributes:
        path: The file, as a refusal names it.
        group: The file's one top-level group.
        encoding: The codec the group's strings are stored in.
        errors: The codec's error handler.
        charged: The bytes the file's datasets declare, plus those of the
            Python objects decoded from them so far.
    """

    path: Path
    group: h5py.Group
    encoding: str
    errors: str
    charged: int

    @classmethod
    def open(cls, path: Path, handle: h5py.File, declared: int) -> _PandasGroupReader:
        import h5py

        keys = list(handle.keys())
        group = handle[keys[0]] if len(keys) == 1 else None
        if not isinstance(group, h5py.Group):
            reason = f"holds {len(keys)} top-level members, not one top-level group"
            raise _UnrecognizedLayoutError(path, reason)
        reader = cls(path, group, "UTF-8", "strict", declared)
        encoding = reader.text(group, "encoding", "UTF-8")
        errors = reader.text(group, "errors", "strict")
        try:
            _ = codecs.lookup(encoding)
            _ = codecs.lookup_error(errors)
        except LookupError:
            reader.fail(f"stores strings as {encoding} ({errors})")
        return cls(path, group, encoding, errors, declared)

    def frame(self) -> pd.DataFrame:
        pandas_type = self.attribute(self.group, "pandas_type")
        if pandas_type == "frame":
            return self.fixed_frame()
        if pandas_type == "frame_table":
            return self.table_frame()
        return self.fail(f"holds a pandas object of type {_brief(pandas_type)}")

    def fixed_frame(self) -> pd.DataFrame:
        """A frame in the fixed format, as `BlockManagerFixed.read` reads it."""
        if self.integer(self.group, "ndim") != 2:
            self.fail("holds a fixed-format object that is not two-dimensional")
        columns = self.fixed_index("axis0")
        index = self.fixed_index("axis1")
        blocks: list[pd.DataFrame] = []
        for number in range(self.integer(self.group, "nblocks")):
            items = self.fixed_index(f"block{number}_items")
            values = self.array(f"block{number}_values", _VALUE_KINDS).T
            positions = columns.get_indexer(items)
            if values.shape != (len(index), len(items)) or (positions < 0).any():
                self.fail(f"block {number} does not match the frame's axes")
            block = pd.DataFrame(
                np.asfortranarray(values),
                columns=columns[positions],
                index=index,
                copy=False,
            )
            blocks.append(block)
        if not blocks:
            return pd.DataFrame(columns=columns, index=index)
        return pd.concat(blocks, axis=1).reindex(columns=columns)

    def fixed_index(self, key: str) -> pd.Index:
        """An axis of the fixed format, as `GenericFixed.read_index` reads it."""
        variety = self.attribute(self.group, f"{key}_variety")
        if variety == "regular":
            return self.index_node(key)
        if variety != "multi":
            return self.fail(f"axis {key} is of the variety {_brief(variety)}")
        levels: list[pd.Index] = []
        codes: list[list[int]] = []
        for level in range(self.integer(self.group, f"{key}_nlevels")):
            levels.append(self.index_node(f"{key}_level{level}"))
            self.charge_objects(f"{key}_level{level}")
            level_codes = self.array(f"{key}_label{level}", "iu")
            self.charge_objects(f"{key}_label{level}")
            codes.append([int(code) for code in level_codes])
        try:
            return pd.MultiIndex(
                levels=[level.tolist() for level in levels],
                codes=codes,
                names=[level.name for level in levels],
                verify_integrity=True,
            )
        except (TypeError, ValueError) as exc:
            return self.fail(
                f"axis {key} is not a valid MultiIndex: {_brief(str(exc))}"
            )

    def index_node(self, key: str) -> pd.Index:
        """One index node, as `GenericFixed.read_index_node` reads it."""
        node = self.dataset(key)
        for special in ("index_class", "tz", "shape", "value_type"):
            if special in node.attrs:
                self.fail(f"index {key} carries {special}")
        if self.attribute(node, "freq") is not None:
            self.fail(f"index {key} carries a frequency")
        name = self.attribute(node, "name")
        if name is not None and not isinstance(name, str | int | float):
            self.fail(f"index {key} has a name of type {type(name).__name__}")
        kind = self.attribute(node, "kind")
        if isinstance(kind, str) and kind in _NUMERIC_KINDS:
            values = self.array(key, _NUMERIC_KINDS[kind])
        elif kind == "string":
            values = self.strings(key, self.nan_rep(node))
        else:
            return self.fail(f"index {key} is of the kind {_brief(kind)}")
        return pd.Index(values, name=name)

    def table_frame(self) -> pd.DataFrame:
        """A frame in the table format, as `AppendableFrameTable.read` reads it."""
        group = self.group
        if self.attribute(group, "table_type") != "appendable_frame":
            self.fail("holds a table that is not an appendable frame")
        if self.attribute(group, "index_cols") != [(0, "index")]:
            self.fail("indexes its table by other than one index column")
        if self.attribute(group, "data_columns") not in (None, []):
            self.fail("holds data columns")
        if not isinstance(self.attribute(group, "levels"), int | None):
            self.fail("has a MultiIndex for its rows")
        values_cols = _as_list(self.attribute(group, "values_cols"))
        info = _as_mapping(self.attribute(group, "info"))
        if values_cols is None or info is None:
            return self.fail("does not describe its table's columns")
        labels = _column_labels(self.attribute(group, "non_index_axes"))
        if labels is None:
            return self.fail("does not describe its table's column axis")
        table = self.dataset("table")
        if table.dtype.names != ("index", *values_cols):
            self.fail("has a table whose fields are not its index and value blocks")
        rows = table[()]
        index = pd.Index(
            self.table_index(table, rows, info), name=self.index_name(info)
        )
        column_info = _as_mapping(info.get(1, {}))
        if column_info is None:
            return self.fail("describes its column axis with a non-mapping")
        blocks = [
            self.table_block(table, rows, block, column_info, index)
            for block in values_cols
        ]
        frame = blocks[0] if len(blocks) == 1 else pd.concat(blocks, axis=1)
        order = pd.Index(labels)
        if not order.equals(frame.columns):
            frame = frame.loc[:, order.unique()]
        return frame

    def table_index(
        self, table: h5py.Dataset, rows: np.ndarray, info: Mapping[object, object]
    ) -> np.ndarray:
        kind = self.attribute(table, "index_kind")
        if not isinstance(kind, str) or kind not in _NUMERIC_KINDS:
            return self.fail(f"has a table index of the kind {_brief(kind)}")
        if self.attribute(table, "index_meta") is not None:
            self.fail("has a table index with metadata")
        values = rows["index"]
        if values.dtype.kind not in _NUMERIC_KINDS[kind]:
            self.fail(f"has a table index of {values.dtype}, not {kind}")
        return np.asarray(values)

    def index_name(self, info: Mapping[object, object]) -> str | int | float | None:
        index_info = _as_mapping(info.get("index", {}))
        if index_info is None:
            return self.fail("describes its table index with a non-mapping")
        if any(index_info.get(key) is not None for key in ("freq", "tz", "ordered")):
            self.fail("has a table index with a frequency, zone or order")
        name = index_info.get("index_name")
        if name is not None and not isinstance(name, str | int | float):
            return self.fail(f"names its table index with a {type(name).__name__}")
        return name

    def table_block(
        self,
        table: h5py.Dataset,
        rows: np.ndarray,
        block: object,
        column_info: Mapping[object, object],
        index: pd.Index,
    ) -> pd.DataFrame:
        """One value block of the table, as `DataCol.convert` reads it."""
        if not isinstance(block, str):
            return self.fail("names a value block with a non-string")
        values = rows[block]
        dtype_name = self.attribute(table, f"{block}_dtype")
        if values.ndim != 2 or values.dtype.kind not in _VALUE_KINDS:
            self.fail(f"block {block} holds {values.dtype} values, not numbers")
        if dtype_name != values.dtype.name:
            self.fail(
                f"block {block} declares {_brief(dtype_name)} but holds {values.dtype}"
            )
        if self.attribute(table, f"{block}_meta") is not None:
            self.fail(f"block {block} carries metadata")
        labels = _as_list(self.attribute(table, f"{block}_kind"))
        if labels is None or len(labels) != values.shape[1]:
            return self.fail(f"block {block} does not label each of its columns")
        multi = column_info.get("type") == "MultiIndex"
        if multi and not all(is_object_tuple(label) for label in labels):
            self.fail(f"block {block} labels a MultiIndex with non-tuples")
        try:
            # A list of tuples builds the MultiIndex that `from_tuples` does.
            columns = pd.Index(labels)
            names = column_info.get("names")
            if names is not None:
                columns = columns.set_names(names)
        except (TypeError, ValueError) as exc:
            return self.fail(
                f"block {block} has unusable column labels: {_brief(str(exc))}"
            )
        return pd.DataFrame(values, columns=columns, index=index, copy=False)

    def dataset(self, key: str) -> h5py.Dataset:
        import h5py

        node = self.group.get(key)
        if not isinstance(node, h5py.Dataset):
            return self.fail(f"has no dataset {key}")
        plist = node.id.get_create_plist()
        for index in range(plist.get_nfilters()):
            code, _, _, name = plist.get_filter(index)
            if not h5py.h5z.filter_avail(code):
                filter_name = _brief(name.decode(errors="replace"))
                self.fail(f"dataset {key} needs the {filter_name} filter")
        return node

    def charge_objects(self, key: str) -> None:
        """Count the Python objects that dataset *key* decodes to toward the limit.

        Raises:
            UnsafeRawFileError: If the count passes the limit.
        """
        node = self.dataset(key)
        elements = math.prod(node.shape or ())
        per_element = _PYTHON_OBJECT_BYTES
        if node.dtype.kind == "S":
            per_element += _STRING_BYTES_PER_STORED_BYTE * node.dtype.itemsize
        self.charged += elements * per_element
        measure = "its datasets, with the Python objects their labels decode to, take"
        require_within_limit(self.path, self.charged, measure)

    def array(self, key: str, kinds: str) -> np.ndarray:
        """A dataset's values, as `GenericFixed.read_array` reads them."""
        node = self.dataset(key)
        for special in ("value_type", "shape"):
            if special in node.attrs:
                self.fail(f"array {key} carries {special}")
        if node.dtype.kind not in kinds:
            self.fail(f"array {key} holds {node.dtype} values")
        values = np.asarray(node[()])
        return values.T if self.attribute(node, "transposed") else values

    def strings(self, key: str, nan_rep: str | None) -> np.ndarray:
        """A dataset of encoded strings, as `_unconvert_string_array` decodes them."""
        node = self.dataset(key)
        if node.dtype.kind != "S":
            self.fail(f"strings {key} are stored as {node.dtype}")
        self.charge_objects(key)
        stored = node[()]
        decoded = np.empty(stored.shape, dtype=object)
        try:
            for position, raw in np.ndenumerate(stored):
                if not isinstance(raw, bytes):
                    return self.fail(f"strings {key} hold a {type(raw).__name__}")
                decoded[position] = raw.decode(self.encoding, self.errors)
        except UnicodeError as exc:
            return self.fail(f"strings {key} do not decode: {_brief(str(exc))}")
        if nan_rep is not None:
            decoded[decoded == nan_rep] = np.nan
        return decoded

    def nan_rep(self, node: h5py.Dataset) -> str | None:
        """The string a string index stores for a missing value, if any."""
        if "nan_rep" not in node.attrs:
            return "nan"
        nan_rep = self.attribute(node, "nan_rep")
        if nan_rep is not None and not isinstance(nan_rep, str):
            return self.fail("stores a missing string as a non-string")
        return nan_rep or None

    def integer(self, node: h5py.Group | h5py.Dataset, name: str) -> int:
        value = self.attribute(node, name)
        if not isinstance(value, int) or isinstance(value, bool):
            return self.fail(f"attribute {name} is not an integer")
        return value

    def text(self, node: h5py.Group, name: str, default: str) -> str:
        value = self.attribute(node, name)
        if value is None:
            return default
        if not isinstance(value, str):
            return self.fail(f"attribute {name} is not a string")
        return value

    def attribute(self, node: h5py.Group | h5py.Dataset, name: str) -> object:
        """Attribute *name* of *node* as PyTables reads it, or ``None``.

        PyTables stores a value it cannot store natively as a pickle, and
        unpickles any byte string that ends with a period, keeping the bytes
        when that fails. This does the same with `_BuiltinUnpickler`.
        """
        import h5py

        if name not in node.attrs:
            return None
        self.check_attribute(node, name)
        value = node.attrs[name]
        if isinstance(value, h5py.Empty):
            return ""
        scalar = np.asarray(value)
        if scalar.ndim == 0 and scalar.dtype.kind in _VALUE_KINDS:
            if scalar.dtype.kind == "b":
                return bool(scalar)
            if scalar.dtype.kind == "f":
                return float(scalar)
            return int(scalar)
        if not isinstance(value, bytes):
            return value
        value = bytes(value)
        if value.endswith(b".") and value not in (b"0", b"0."):
            member = (node.name or "").lstrip("/")
            try:
                decoded = self.unpickle(value)
            except (_ForbiddenPickleError, RecursionError) as exc:
                reason = f"attribute {name} cannot be unpickled: {_brief(str(exc))}"
                raise _UnrecognizedLayoutError(
                    self.path, reason, member=member
                ) from exc
            except _UNREADABLE_PICKLE:
                pass
            else:
                if not _nests_within(decoded, _MAX_PICKLE_DEPTH):
                    reason = (
                        f"attribute {name} nests more than {_MAX_PICKLE_DEPTH} "
                        f"levels deep"
                    )
                    raise _UnrecognizedLayoutError(self.path, reason, member=member)
                return decoded
        try:
            return value.decode()
        except UnicodeDecodeError:
            return value

    def check_attribute(self, node: h5py.Group | h5py.Dataset, name: str) -> None:
        """Refuse attribute *name* of *node* from its datatype and shape, unread.

        h5py builds a Python object for each element of a variable-length or
        reference attribute, and the elements of one can all name a single
        large object in the file, so such an attribute is refused before it is
        read, as is one past the size pandas writes. pandas writes each
        attribute that the reader reads as a scalar.

        Raises:
            UnsafeRawFileError: If the attribute holds variable-length,
                reference or object values, has a datatype with no numpy
                equivalent, or declares more than ``_MAX_ATTRIBUTE_BYTES``.
            _UnrecognizedLayoutError: If the attribute is not a scalar.
        """
        member = (node.name or "").lstrip("/")
        attribute = node.attrs.get_id(name)
        try:
            dtype = attribute.dtype
        except TypeError as exc:
            reason = f"attribute {name} has a datatype with no numpy equivalent"
            raise UnsafeRawFileError(self.path, reason, member=member) from exc
        if dtype.hasobject:
            reason = f"attribute {name} holds variable-length or reference values"
            raise UnsafeRawFileError(self.path, reason, member=member)
        declared = math.prod(attribute.shape or ()) * max(dtype.itemsize, 1)
        if declared > _MAX_ATTRIBUTE_BYTES:
            reason = (
                f"attribute {name} declares {declared} bytes, more than the "
                f"{_MAX_ATTRIBUTE_BYTES} pandas writes"
            )
            raise UnsafeRawFileError(self.path, reason, member=member)
        if attribute.shape not in (None, ()):
            self.fail(f"attribute {name} is an array of shape {attribute.shape}")

    @staticmethod
    def unpickle(value: bytes) -> object:
        try:
            return _BuiltinUnpickler(io.BytesIO(value)).load()
        except UnicodeDecodeError:
            # A pickle that Python 2 wrote, which PyTables decodes as Latin-1.
            return _BuiltinUnpickler(io.BytesIO(value), encoding="latin1").load()

    def fail(self, reason: str) -> NoReturn:
        raise _UnrecognizedLayoutError(
            self.path, f"pandas layout not recognized: {reason}"
        )
