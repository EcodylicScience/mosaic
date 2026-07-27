"""The track-converter contract: turn one raw tracks file into a standard table.

A converter used to be a bare ``Callable[[Path, dict], pd.DataFrame]`` handed an
untyped dict that the caller had **mutated with the entry's group and sequence**
just before the call. Two consequences, both fatal to item 3.1's tracks hash:

- Hashing "the conversion params" would hash the sequence name, so one recipe
  applied to two hundred sequences would mint two hundred variants -- exactly
  what P2d says an identifier must not contain.
- The dict was untyped, so validation and debug flags sat beside the parameters
  that genuinely determine the output, with nothing to tell them apart.

So entry identity moves to :class:`EntryHints` -- a frozen dataclass that is
deliberately **not** a ``Params``. The type itself says "this is not hashable",
and a sequence name cannot reach a digest because it is not in the object being
digested. Throughput and validation knobs stay in ``Params`` but are marked
``HASH_EXCLUDE``, the same migration the tracking ops already completed.

``group_from`` does not appear here at all. No converter ever read it -- only the
*label* converters do -- and it is a caller-side policy about which group string
to write, resolved into a concrete group before ``convert`` is called. Left on
the interface it would put a display policy into the tracks hash, minting two
variants for bit-identical output.

**Why this module rather than ``dataset.py``.** The registry lived there, so
every converter imported ``dataset`` to register itself while ``dataset``
imported the registry to dispatch -- the cycle noted in ``dataset.py``'s own
comments, which says closing it needs the registry moved out. It is moved out
here: this module imports nothing from ``mosaic.core.dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Generic

import pandas as pd
from typing_extensions import TypeVar

from mosaic.core.pipeline.types import HASH_EXCLUDE, Params

__all__ = [
    "TRACK_CONVERTERS",
    "EntryHints",
    "TrackConvertParams",
    "TrackConverter",
    "get_track_converter",
    "register_track_converter",
]


@dataclass(frozen=True, slots=True)
class EntryHints:
    """Which ``(group, sequence)`` the caller is asking the converter to produce.

    Passed beside the params, never merged into them, and never hashed. A
    converter may write these into its output columns and may use them to select
    one sequence out of a multi-sequence file; what it must not do is let them
    change *how* it converts, because then one recipe would be many recipes.

    Both default to empty, which means "the converter decides" -- typically from
    the filename stem.
    """

    group: str = ""
    sequence: str = ""


class TrackConvertParams(Params):
    """Base for every converter's parameters.

    Carries only what every converter shares. ``strict_schema`` is validation
    strictness, not a property of the output table, so it is excluded from
    identity: flipping it must not mint a second variant of the same tracks.
    """

    strict_schema: Annotated[bool, HASH_EXCLUDE] = False


P = TypeVar("P", bound=TrackConvertParams, default=TrackConvertParams)


class TrackConverter(Generic[P]):
    """Convert one raw tracks file into a schema-valid standard table.

    Mirrors ``core.pipeline.ops.Op``: a class with declared class variables and
    a typed ``Params``, registered by decorator. Subclass, declare
    ``src_format`` / ``Params``, implement ``convert``.

    Class variables:
        src_format: The raw format this handles, and the registry key. One
            format per class -- a converter that reads two file formats declares
            a thin subclass per format, so a tracks variant identity names
            exactly one producer.
        version: Declared compatibility version, a visible segment of the tracks
            variant identity. **Declared, never detected**: bump it by hand when
            the output semantics change, so a bit-identical conversion keeps its
            identity across an unrelated release.
        enumerable: Whether this format can hold several sequences in one file,
            i.e. whether ``enumerate_sequences`` is implemented. A flag rather
            than a second registry keyed on the same string -- two dicts
            populated by one module is how they drift apart.
        Params: The parameter model. Everything in it determines the output
            except fields marked ``HASH_EXCLUDE``.
    """

    src_format: ClassVar[str]
    version: ClassVar[str] = "0.1"
    enumerable: ClassVar[bool] = False
    Params: ClassVar[type[TrackConvertParams]] = TrackConvertParams

    def convert(self, path: Path, params: P, hints: EntryHints) -> pd.DataFrame:
        """Read *path* and return one schema-valid table.

        Args:
            path: The raw tracks file.
            params: Typed conversion parameters. These, plus ``src_format`` and
                ``version``, are what the tracks variant identity is made of.
            hints: Which entry to produce. Never hashed -- see :class:`EntryHints`.
        """
        raise NotImplementedError

    def enumerate_sequences(self, path: Path) -> list[tuple[str, str]]:
        """The ``(group, sequence)`` pairs *path* contains.

        Only meaningful when ``enumerable`` is True. Used to expand one
        multi-sequence file into one output per sequence.
        """
        raise NotImplementedError


TRACK_CONVERTERS: dict[str, type[TrackConverter[TrackConvertParams]]] = {}


def register_track_converter(
    cls: type[TrackConverter[TrackConvertParams]],
) -> type[TrackConverter[TrackConvertParams]]:
    """Register a converter class under its declared ``src_format``."""
    src_format = getattr(cls, "src_format", "")
    if not src_format:
        raise ValueError(f"{cls.__name__} must declare a non-empty src_format")
    TRACK_CONVERTERS[src_format] = cls
    return cls


def get_track_converter(src_format: str) -> TrackConverter[TrackConvertParams]:
    """Instantiate the converter registered for *src_format*.

    Raises:
        KeyError: with the registered formats listed, because "no converter for
            X" is almost always a typo or a missing import of the module that
            would have registered it.
    """
    cls = TRACK_CONVERTERS.get(src_format)
    if cls is None:
        known = ", ".join(sorted(TRACK_CONVERTERS)) or "(none registered)"
        raise KeyError(
            f"No converter registered for src_format={src_format!r}. Known: {known}"
        )
    return cls()
