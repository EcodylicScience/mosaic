"""Type stub for h5py, which ships none.

It declares only the names that mosaic uses. The signatures follow h5py 3.16.
"""

from collections.abc import Callable, ItemsView, KeysView
from types import TracebackType
from typing import Literal, Self, overload

import numpy as np
import numpy.typing as npt

from . import h5a as h5a
from . import h5d as h5d
from . import h5s as h5s
from . import h5t as h5t
from . import h5z as h5z
from .h5d import DatasetID
from .h5g import GroupID

type _Node = Group | Dataset | Datatype
type _Link = HardLink | SoftLink | ExternalLink

class Empty:
    dtype: np.dtype[np.generic]
    def __init__(self, dtype: npt.DTypeLike) -> None: ...

def string_dtype(
    encoding: str = "utf-8", length: int | None = None
) -> np.dtype[np.object_]: ...

class HardLink: ...

class SoftLink:
    path: str
    def __init__(self, path: str) -> None: ...

class ExternalLink:
    filename: str
    path: str
    def __init__(self, filename: str, path: str) -> None: ...

class AttributeManager:
    def __getitem__(self, name: str) -> object: ...
    def __setitem__(self, name: str, value: object) -> None: ...
    def __delitem__(self, name: str) -> None: ...
    def __contains__(self, name: str) -> bool: ...
    def create(
        self, name: str, data: object, dtype: npt.DTypeLike | None = None
    ) -> None: ...
    def get(self, name: str, default: object = None) -> object: ...
    def get_id(self, name: str) -> h5a.AttrID: ...
    def keys(self) -> KeysView[str]: ...
    def items(self) -> ItemsView[str, object]: ...

class Datatype:
    name: str | None
    attrs: AttributeManager
    dtype: np.dtype[np.generic]

class Dataset:
    id: DatasetID
    name: str | None
    attrs: AttributeManager
    dtype: np.dtype[np.generic]
    shape: tuple[int, ...] | None
    is_virtual: bool
    external: list[tuple[str, int, int]] | None
    def __getitem__(self, key: object) -> npt.NDArray[np.generic]: ...
    def __array__(
        self, dtype: npt.DTypeLike | None = None
    ) -> npt.NDArray[np.generic]: ...
    def __len__(self) -> int: ...

class VirtualSource:
    def __init__(
        self,
        path_or_dataset: str | Dataset,
        name: str | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: npt.DTypeLike | None = None,
    ) -> None: ...

class VirtualLayout:
    def __init__(
        self, shape: tuple[int, ...], dtype: npt.DTypeLike | None = None
    ) -> None: ...
    def __setitem__(self, key: object, source: VirtualSource) -> None: ...

class Group:
    id: GroupID
    name: str | None
    attrs: AttributeManager
    def __getitem__(self, name: str) -> _Node: ...
    def __setitem__(self, name: str, value: object) -> None: ...
    def __contains__(self, name: str) -> bool: ...
    def __iter__(self) -> object: ...
    def __len__(self) -> int: ...
    @overload
    def get(self, name: str, default: None = None) -> _Node | None: ...
    @overload
    def get(
        self, name: str, default: None = None, *, getlink: Literal[True]
    ) -> _Link | None: ...
    def keys(self) -> KeysView[str]: ...
    def create_group(self, name: str) -> Group: ...
    def create_dataset(
        self,
        name: str,
        shape: tuple[int, ...] | None = None,
        dtype: npt.DTypeLike | None = None,
        data: npt.ArrayLike | None = None,
        *,
        chunks: tuple[int, ...] | bool | None = None,
        external: list[tuple[str, int, int]] | None = None,
    ) -> Dataset: ...
    def copy(self, source: Group | Dataset, dest: str) -> None: ...
    def create_virtual_dataset(self, name: str, layout: VirtualLayout) -> Dataset: ...
    def visit_links(self, func: Callable[[str], object]) -> None: ...
    def visititems(self, func: Callable[[str, _Node], object]) -> None: ...

class File(Group):
    def __init__(
        self, name: str | bytes | object, mode: str = "r", libver: str | None = None
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...
