"""The track-converter contract: turn one raw tracks file into a standard table.

A converter used to be a bare ``Callable[[Path, dict], pd.DataFrame]`` handed an
untyped dict that the caller had **mutated with the entry's group and sequence**
just before the call. Two consequences, both fatal to a tracks hash:

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

**One declaration of ``keep_empty_frames``.** :data:`KeepEmptyFrames` is the
annotation that every converter's ``keep_empty_frames`` field uses, so its prose
is written once. Each converter chooses its default and drops the field from its
``identity_dump`` at that default through :meth:`Params.drop_at_default`, so a
conversion that never sets it keeps its variant.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Generic, TypeVar, override

import pandas as pd
from pydantic import Field

from mosaic.core.entry import validate_entry_name
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)
from mosaic.core.root_lookup import RootLookup
from mosaic.entry_points import (
    EntryPointRefused,
    InstalledOrigin,
    installed_entry_points,
    load_entry_point,
)

__all__ = [
    "INSTALLED_STATE",
    "TRACK_CONVERTERS",
    "TRACK_CONVERTER_GROUP",
    "EntryHints",
    "FrameRateUnknownError",
    "InstalledConverterState",
    "KeepEmptyFrames",
    "TimedTrackConvertParams",
    "TrackConvertParams",
    "TrackConverter",
    "builtin_track_converters",
    "conversion_rate",
    "ensure_track_converters_registered",
    "get_track_converter",
    "merge_on_column_union",
    "refused_track_converters",
    "register_track_converter",
    "track_converter_origin",
]

TRACK_CONVERTER_GROUP: Final = "mosaic.track_converters"
"""The entry point group in which an installed package declares track converters."""

_INSTALLED_NAME: Final = re.compile(r"[a-z0-9]+(_[a-z0-9]+)*__[a-z0-9_]+")
"""An installed ``src_format``: a namespace and a name joined by ``__``."""

_VERSION: Final = re.compile(r"[0-9]+(\.[0-9]+)*")
"""A converter version, the segment that a conversion's run id carries."""

_STRICT_SCHEMA_DESCRIPTION = (
    "Raise when the converted table is missing a required column or prefix, "
    "instead of printing a validation report."
)


@dataclass(frozen=True, slots=True)
class EntryHints:
    """Which ``(group, sequence)`` the caller is asking the converter to produce.

    Passed beside the params, never merged into them, and never hashed. A
    converter may write these into its output columns and may use them to select
    one sequence out of a multi-sequence file; what it must not do is let them
    change *how* it converts, because then one recipe would be many recipes.

    Both default to empty, which means "the converter decides" -- typically from
    the filename stem.

    ``fps`` is the measured frame rate of the entry's media: the media the tool
    read, or the media indexed for an imported file. It is a fact about the entry's
    input, as the file's bytes are, and not a choice of recipe: a converter that
    reads it applies one rule to every entry, and the rate decides only the
    numbers. ``None`` means not measured. A converter of a tool that times its own
    output keeps that timing then, and a converter of a file that records no rate
    takes the rate from its params (:func:`conversion_rate`).

    ``first_frame`` is the sequence's first frame on its recording's frame axis,
    0 for an undivided recording, and like ``fps`` a fact about the entry. A
    converter numbers its table from the sequence's own first frame, and the
    caller moves it onto the recording's axis
    (:func:`~mosaic.core.pipeline.placement.place_converted`). ``time_offset``
    is the sequence's first frame on the recording's time axis, in seconds,
    which a division's rows record (``Dataset.division_facts``); ``None`` takes
    ``first_frame`` at ``fps``.

    Validated on construction, so a converter structurally cannot emit a name
    that is not usable as one path component. This is the earliest of the three
    write boundaries: catching it here names the converter in the traceback,
    rather than the index writer three frames later.
    """

    group: str = ""
    sequence: str = ""
    fps: float | None = None
    first_frame: int = 0
    time_offset: float | None = None

    def __post_init__(self) -> None:
        _ = validate_entry_name(self.group, "group")
        _ = validate_entry_name(self.sequence, "sequence")
        if self.fps is not None and not (math.isfinite(self.fps) and self.fps > 0):
            msg = (
                f"fps={self.fps} is not a measured frame rate. Pass None when "
                "the media's rate is unknown."
            )
            raise ValueError(msg)
        if self.first_frame < 0:
            msg = (
                f"first_frame={self.first_frame} is before the recording's first "
                "frame, 0."
            )
            raise ValueError(msg)


class FrameRateUnknownError(ValueError):
    """A conversion has no frame rate to time its table by."""


def conversion_rate(
    explicit: float | None, fallback: float | None, hints: EntryHints, source: str
) -> float:
    """Return the frame rate that times a table whose file records none.

    A SLEAP analysis file, a DeepLabCut export and an Ultralytics predictions file
    store frame numbers only, so ``time`` is ``frame / rate`` for a rate that the
    caller supplies. The rate is *explicit*, the converter's ``fps`` parameter,
    else the measured rate of the entry's media (``hints.fps``), else *fallback*,
    the dataset's ``fps_default``. Every such converter calls this function.

    Args:
        explicit: The converter's ``fps`` parameter, ``None`` when unset.
        fallback: The converter's ``fallback_fps`` parameter, ``None`` when unset.
        hints: The entry, whose ``fps`` is the measured rate of its media.
        source: The file being converted, which the refusal names.

    Returns:
        The positive rate.

    Raises:
        FrameRateUnknownError: If none of the three rates is known.
    """
    for rate in (explicit, hints.fps, fallback):
        if rate is not None:
            return rate
    message = (
        f"Cannot time {source}: the file records no frame rate, no fps was given, "
        "the entry's media has no measured rate, and meta.fps_default is not set. "
        "Pass fps=, index the entry's media, or set meta.fps_default."
    )
    raise FrameRateUnknownError(message)


_KEEP_EMPTY_FRAMES_DESCRIPTION = (
    "Write a row for every frame and individual that the source records, with "
    "NaN positions, keypoints and scores where it has none. Unset keeps the "
    "converter's default."
)

KeepEmptyFrames = Annotated[bool, Declared(_KEEP_EMPTY_FRAMES_DESCRIPTION)]
"""The type of every converter's ``keep_empty_frames`` field."""


class TrackConvertParams(Params):
    """Base for every converter's parameters.

    Carries only what every converter shares. ``strict_schema`` is validation
    strictness, not a property of the output table, so it is excluded from
    identity: flipping it must not mint a second variant of the same tracks.
    """

    strict_schema: Annotated[
        bool, HASH_EXCLUDE, Declared(_STRICT_SCHEMA_DESCRIPTION)
    ] = False


_FPS_DESCRIPTION = (
    "Frame rate that derives time, taken before the entry's measured media rate and "
    "the dataset's fps_default. Leave unset to time each entry at its media's "
    "measured rate."
)

_FALLBACK_FPS_DESCRIPTION = (
    "Frame rate for an entry whose media has no measured rate. The dataset fills "
    "it from meta.fps_default."
)


class TimedTrackConvertParams(TrackConvertParams):
    """Base of a converter whose file records no frame rate.

    ``time`` is ``frame / rate`` (:func:`conversion_rate`), and the rate is
    ``fps``, else the entry's measured media rate, else ``fallback_fps``. A field
    that is unset stays out of ``identity_dump()``, and the measured rate is a fact
    about the entry (:class:`EntryHints`) and never enters it.
    """

    fps: Annotated[
        float | None, Field(gt=0.0, allow_inf_nan=False), Declared(_FPS_DESCRIPTION)
    ] = None
    fallback_fps: Annotated[
        float | None,
        Field(gt=0.0, allow_inf_nan=False),
        Declared(_FALLBACK_FPS_DESCRIPTION),
    ] = None

    @override
    def identity_dump(self) -> dict[str, object]:
        """The hashed params, with each rate left out while it is unset."""
        return self.drop_at_default(super().identity_dump(), ("fps", "fallback_fps"))


P = TypeVar("P", bound=TrackConvertParams, default=TrackConvertParams)


class TrackConverter(Generic[P]):
    """Convert one raw tracks file into a schema-valid standard table.

    Mirrors ``core.pipeline.ops.Op``: a class with declared class variables and
    a typed ``Params``, registered by decorator. Subclass, declare
    ``src_format`` / ``Params``, implement ``convert``. The hooks are
    ``tracklets``, ``input_file``, ``convert``, ``frames_read``,
    ``enumerate_sequences`` and ``sequence_from_stem``.

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
        merges_per_sequence: The mirror of ``enumerable``: whether several files
            of this format make up one sequence, as TRex's one ``.npz`` per
            individual does, so ``convert_all_tracks`` concatenates them on the
            union of their columns instead of writing a table each. Declared
            here for the reason ``enumerable`` is declared here -- the
            alternative was ``core`` comparing ``src_format`` to a literal,
            which is that second registry spelled as an ``if``, and it drifts
            from this one the first time a format is added. Not compatible with
            ``enumerable``: they are opposite claims about the same
            file/sequence relationship, and the machinery can act on only one.
        output_schema: The track schema this converter's tables answer to, and
            the single place that answer is spelled. It used to live in two
            places at once: each ``convert`` hard-coded a name while
            ``Dataset`` validated the very same frame against
            ``meta.tracks.standard_format``, so one table was checked twice
            under two independently chosen names and the index row recorded
            only the second. A dataset could carry two labels for one schema
            and no test asserted what any converter wrote.

            A ClassVar rather than a ``Params`` field on purpose: ClassVars
            never enter ``identity_dump()``, so declaring one moves no tracks
            variant. Declaring a *different* schema is a change of output
            semantics and takes a ``version`` bump like any other.

            The default is ``trex_v1``, the legacy schema whose spatial columns
            are centimeters and whose ``X`` is a head position, and
            :func:`register_track_converter` does not check it. A converter that
            omits the declaration is validated and recorded under that schema.
            Every new converter declares one, ``mosaic_v1`` for a pixel table.
        carries_tracklets: Whether the converted ``id`` follows one animal over
            time: ``True`` for a format whose ids name animals, ``False`` for one
            whose ids restart or number detections, and ``None`` when the
            format cannot say and the user is asked (:meth:`tracklets`). A
            ClassVar, so it never enters ``identity_dump()``. Every converter
            declares it, and the tracks index row records the answer.
        Params: The parameter model. Everything in it determines the output
            except fields marked ``HASH_EXCLUDE``.
    """

    src_format: ClassVar[str]
    version: ClassVar[str] = "0.1"
    enumerable: ClassVar[bool] = False
    merges_per_sequence: ClassVar[bool] = False
    output_schema: ClassVar[str] = "trex_v1"
    carries_tracklets: ClassVar[bool | None] = None
    Params: ClassVar[type[TrackConvertParams]] = TrackConvertParams

    def tracklets(self, params: P) -> bool | None:
        """Return the tracklets answer that a conversion under *params* records.

        The class's :attr:`carries_tracklets`, unless a format lets the user
        answer through a parameter and the user did. ``None`` records a blank.
        """
        _ = params
        return self.carries_tracklets

    def input_file(
        self, path: Path, params: P, source_md5: str, root_of: RootLookup
    ) -> Path:
        """Return the file that :meth:`convert` reads for the raw file *path*.

        The default is *path* itself. A converter that reads another file in its
        place, such as a copy that a tool made of the raw file, returns that
        file's path. A file it cannot find or trust is a refusal: a
        ``ValueError`` naming the file, which fails the entry.

        Args:
            path: The raw tracks file.
            params: Typed conversion parameters.
            source_md5: The ``md5`` cell of the raw file's index row, ``""`` when
                the scan did not hash it.
            root_of: Finds a dataset root by its key, and returns ``None`` for a
                root the dataset does not declare.
        """
        _ = params, source_md5, root_of
        return path

    def convert(self, path: Path, params: P, hints: EntryHints) -> pd.DataFrame:
        """Read *path* and return one schema-valid table.

        Args:
            path: The file that :meth:`input_file` returned for the raw file.
            params: Typed conversion parameters. These, plus ``src_format`` and
                ``version``, are what the tracks variant identity is made of.
            hints: Which entry to produce. Never hashed -- see :class:`EntryHints`.
        """
        raise NotImplementedError

    def frames_read(self, path: Path) -> int | None:
        """Return the number of frames the file *path* records, or ``None``.

        The base returns ``None``: the converter records no count. An override
        returns the count from the file's shape where it can, and raises
        ``OSError`` or ``ValueError`` on a file it cannot read. An enumerable
        converter keeps the base, because one file's count is not a sequence's.
        A conversion that merges several files records the count of its first
        file, the one its tracks row names as its source. The count is
        provenance and enters no identifier.

        Args:
            path: The raw tracks file.
        """
        _ = path
        return None

    def enumerate_sequences(self, path: Path) -> list[tuple[str, str]]:
        """The ``(group, sequence)`` pairs *path* contains.

        Only meaningful when ``enumerable`` is True. Used to expand one
        multi-sequence file into one output per sequence.
        """
        raise NotImplementedError

    def sequence_from_stem(self, stem: str) -> str:
        """Which sequence a file with this stem belongs to.

        The default is the stem itself: one file is one sequence. A format that
        declares ``merges_per_sequence`` overrides this to drop whatever names
        the *individual* rather than the entry, so that its several files are
        recognized as one sequence.

        Asked of a stem rather than of a file because ``index_tracks_raw`` has
        to group the files it found into entries before it opens any of them.
        It describes the format's own naming rather than a knob, so it is not a
        ``Params`` field and cannot reach a digest: two datasets whose files are
        named differently must not be two recipes.
        """
        return stem


TRACK_CONVERTERS: dict[str, type[TrackConverter[Any]]] = {}


def _qualified_name(cls: type) -> str:
    """Return *cls* spelled as ``module.qualname``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def register_track_converter[C: type[TrackConverter[Any]]](cls: C) -> C:
    """Register a converter class under its declared ``src_format``.

    A format names one converter for the life of the process, because a variant's
    run id carries the format and the converter's version and nothing of the
    class: a second class under the same format would write a different table
    under an id that already names another. A class of another module or
    qualified name is therefore refused.

    A class of the same module and qualified name replaces the registered one,
    so a notebook cell that defines and registers its converter keeps working
    when it is run again, and so does a module reloaded by ``%autoreload``, each
    of which builds a new class object of the same name.

    Raises:
        ValueError: when *cls* declares no ``src_format``, or when another class
            holds it, naming both.
    """
    src_format = getattr(cls, "src_format", "")
    if not src_format:
        raise ValueError(f"{cls.__name__} must declare a non-empty src_format")
    registered = TRACK_CONVERTERS.get(src_format)
    if registered is not None and _qualified_name(registered) != _qualified_name(cls):
        raise ValueError(
            f"src_format={src_format!r} is registered to "
            f"{_qualified_name(registered)}, and {_qualified_name(cls)} cannot "
            "take it. Give the converter a src_format of its own."
        )
    TRACK_CONVERTERS[src_format] = cls
    return cls


def get_track_converter(src_format: str) -> TrackConverter[TrackConvertParams]:
    """Instantiate the converter registered for *src_format*.

    A converter that is registered answers first, so an installed entry point
    refused for a built-in's name leaves the built-in.

    Raises:
        ValueError: when no converter holds *src_format*. The message is the
            recorded refusal of an installed entry point of that name, else the
            registered formats listed and the group in which an installed package
            declares a converter, because "no converter for X" is almost always a
            typo, a missing import of the module that would have registered it, or
            a package that is not installed. A ``ValueError`` rather than the
            ``KeyError`` a registry lookup suggests: the argument is a caller's
            choice, not a key they were handed, and the CLI interpolates the
            exception into a message where ``KeyError`` renders its own quotes.
    """
    ensure_track_converters_registered()
    cls = TRACK_CONVERTERS.get(src_format)
    if cls is not None:
        return cls()
    refusal = INSTALLED_STATE.refusals.get(src_format)
    if refusal is not None:
        raise ValueError(refusal)
    known = ", ".join(sorted(TRACK_CONVERTERS)) or "(none registered)"
    raise ValueError(
        f"No converter registered for src_format={src_format!r}. Known: {known}. "
        f"An installed package declares a converter in the group "
        f"{TRACK_CONVERTER_GROUP!r}."
    )


@dataclass(slots=True)
class InstalledConverterState:
    """What the load of the ``mosaic.track_converters`` group has found.

    One instance, :data:`INSTALLED_STATE`, is mutated in place for the life of the
    process.

    Attributes:
        origins: The entry point of each accepted installed converter, by format.
        refusals: The message of each refused entry point, by the name it declares.
        loaded: Whether the load has run.
        loading: Whether a thread is running the load now.
        lock: Held for the whole load, so a second thread waits for it.
    """

    origins: dict[str, InstalledOrigin] = field(
        default_factory=dict[str, InstalledOrigin]
    )
    refusals: dict[str, str] = field(default_factory=dict[str, str])
    loaded: bool = False
    loading: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


INSTALLED_STATE: Final = InstalledConverterState()


def ensure_track_converters_registered() -> None:
    """Register every built-in converter, then every installed one, once.

    A converter registers only as a side effect of importing the module that
    defines it. ``mosaic.core`` used to perform that import on every import of
    anything beneath it, so no caller had to ask, and every consumer of every
    leaf under ``core`` paid for pandas and for h5py. It imports nothing now, so
    a caller reading the registry asks first.

    The built-ins are imported on every call, and deliberately not guarded on the
    registry being empty. The built-ins were guaranteed present before this file
    stopped importing them, and a guard on contents reverses that guarantee for
    the one caller who registers a converter of their own first: theirs fills the
    registry, the guard then returns early, and every built-in is missing for the
    rest of the process. That is the failure this function exists to prevent,
    arriving through the guard instead of a missing call. A repeat call costs one
    :data:`sys.modules` lookup, which is what the guard saved.

    The converters that installed packages declare in the group
    ``mosaic.track_converters`` load on the first call of the process, after the
    built-ins, and never again: :func:`_load_installed_converters` explains what
    each entry point must be, and an entry point that is not is recorded in
    :func:`refused_track_converters` and not registered. This call does not raise
    for one, so an installed package that fails does not take the built-ins down
    with it.

    A lock serializes the load, so threads that share the registry, which is every
    worker of thread mode, wait for it and find every installed converter. A
    process worker has a registry of its own and loads on its own first call. A
    call made while the load runs on the same thread, from a plugin module that
    looks up a converter as it is imported, returns after the built-ins without
    loading again.

    Read the registry through :func:`get_track_converter`, which calls this, or
    call this before reading :data:`TRACK_CONVERTERS` directly. Reading it
    without asking gives an empty mapping rather than an error, which answers
    "no such format" for a format that is registered.
    """
    _ = import_module("mosaic.core.track_library")
    state = INSTALLED_STATE
    if state.loaded:
        return
    with state.lock:
        if state.loaded or state.loading:
            return
        state.loading = True
        try:
            _load_installed_converters()
        finally:
            state.loading = False
            state.loaded = True


class _InstalledLoad:
    """One load of the group: the entry points, and which keys are provisional.

    A plugin module may decorate several converters, so importing it for one
    entry point can register keys that other entry points name. A key is
    provisional while the entry point of that name, which names its class, has not
    been decided. A key whose name is decided already, or that no entry point of
    its own name names, is refused: a name refused without an import, because it
    is declared twice, breaks the namespace rule or is a built-in's, stays refused
    with its own message when a later module decorates its class.
    """

    def __init__(self) -> None:
        self.builtin = frozenset(builtin_track_converters())
        self.declared = installed_entry_points(TRACK_CONVERTER_GROUP)
        self.provisional: set[str] = set()
        self.undecided = set(self.declared)

    def run(self) -> None:
        """Decide each entry point in the order of its name.

        An error that no check foresaw refuses the entry point it came from, with
        the keys it added removed, and the load goes on with the next.
        """
        for name, found in self.declared.items():
            self.undecided.discard(name)
            before = frozenset(TRACK_CONVERTERS)
            try:
                self._decide(name, found)
            except Exception as error:
                owners = " and ".join(
                    InstalledOrigin.from_entry_point(each).describe() for each in found
                )
                message = (
                    f"{owners} could not be decided: {type(error).__name__}: {error}"
                )
                added = frozenset(TRACK_CONVERTERS) - before
                self.refuse(name, message, added | self._own_keys(name))

    def _decide(self, name: str, found: tuple[EntryPoint, ...]) -> None:
        """Accept or refuse the entry points that declare *name*."""
        first = found[0]
        if len(found) > 1:
            self.refuse(name, self._twice(name, found), self._own_keys(name))
        elif name in self.builtin:
            self.refuse(name, self._refusal(first, "names a built-in format"))
        elif not _INSTALLED_NAME.fullmatch(name):
            self.refuse(
                name,
                self._refusal(
                    first,
                    "does not name a namespace and a name joined by '__' "
                    "(lowercase letters, digits and '_')",
                ),
                self._own_keys(name),
            )
        else:
            self._load(name, first)

    def refuse(
        self, name: str, message: str, discard: frozenset[str] = frozenset()
    ) -> None:
        """Record *message* under *name* and remove the keys of *discard*."""
        INSTALLED_STATE.refusals[name] = message
        _ = INSTALLED_STATE.origins.pop(name, None)
        for key in discard:
            if key in self.builtin:
                continue
            _ = TRACK_CONVERTERS.pop(key, None)
            _ = INSTALLED_STATE.origins.pop(key, None)
            self.provisional.discard(key)

    def _own_keys(self, name: str) -> frozenset[str]:
        """The key *name* when an earlier module left it provisional."""
        return frozenset({name}) & self.provisional

    @staticmethod
    def _refusal(found: EntryPoint, reason: str) -> str:
        return f"{InstalledOrigin.from_entry_point(found).describe()} {reason}."

    @staticmethod
    def _twice(name: str, found: tuple[EntryPoint, ...]) -> str:
        origins = " and ".join(
            InstalledOrigin.from_entry_point(each).describe() for each in found
        )
        return (
            f"The format {name!r} is declared twice in the group "
            f"{TRACK_CONVERTER_GROUP!r}, by {origins}. Neither is registered."
        )

    def _declared_class(self, key: str, registered: type[object]) -> bool:
        """Does the entry point named *key* name *registered*?"""
        return any(
            each.module == registered.__module__
            and each.attr == registered.__qualname__
            for each in self.declared.get(key, ())
        )

    def _load(self, name: str, found: EntryPoint) -> None:
        origin = InstalledOrigin.from_entry_point(found)
        before = frozenset(TRACK_CONVERTERS)
        try:
            cls = load_entry_point(found, TrackConverter, subclass=True)
        except EntryPointRefused as error:
            added = frozenset(TRACK_CONVERTERS) - before
            self.refuse(name, str(error), added | self._own_keys(name))
            return
        added = frozenset(TRACK_CONVERTERS) - before
        problem = self._problem(origin, name, cls)
        if problem is None:
            try:
                _ = register_track_converter(cls)
            except ValueError as error:
                problem = self._refusal(
                    found, f"cannot register {cls.__name__}: {error}"
                )
        if problem is not None:
            self.refuse(name, problem, added | self._own_keys(name))
            return
        INSTALLED_STATE.origins[name] = origin
        _ = INSTALLED_STATE.refusals.pop(name, None)
        self.provisional.discard(name)
        for key in sorted(added - {name}, key=str):
            registered = TRACK_CONVERTERS[key]
            if key in self.undecided and self._declared_class(key, registered):
                self.provisional.add(key)
                continue
            message = (
                f"{origin.describe()} registered the format {key!r}, which no "
                f"entry point of the group {TRACK_CONVERTER_GROUP!r} names. "
                "It is not registered."
            )
            if key not in self.undecided:
                message = INSTALLED_STATE.refusals.get(key, message)
            self.refuse(key, message, frozenset({key}))

    @staticmethod
    def _problem(
        origin: InstalledOrigin, name: str, cls: type[TrackConverter[Any]]
    ) -> str | None:
        """Why *cls* cannot be the converter of *name*, or ``None``."""
        src_format = getattr(cls, "src_format", "")
        if not isinstance(src_format, str) or src_format != name:
            return (
                f"{origin.describe()} names {cls.__name__}, whose src_format is "
                f"{src_format!r}, not {name!r}."
            )
        version = getattr(cls, "version", "")
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            return (
                f"{origin.describe()} names {cls.__name__}, whose version "
                f"{version!r} does not match {_VERSION.pattern!r}."
            )
        return None


def _load_installed_converters() -> None:
    """Load every entry point of ``mosaic.track_converters`` in the order of its name.

    The entry point's name is the converter's ``src_format``, and its object is a
    ``TrackConverter`` subclass. An entry point is refused, with its message
    recorded under its name, when:

    - two entry points declare the name, or the name is a built-in's or breaks the
      namespace rule, which gives an installed format a ``__`` that no built-in
      holds;
    - it cannot be loaded, names an object that is not a class, or a class that is
      not a ``TrackConverter`` subclass;
    - the class's ``src_format`` differs from the name, or its ``version`` does not
      match the pattern that a conversion's run id carries;
    - its registration is refused, because a class of another module holds the
      format.

    A plugin module may decorate several converters with
    :func:`register_track_converter`. A key that the import of an entry point's
    module adds is therefore kept when the entry point of the same name names the
    same class and has not been decided yet, and still passes that entry point's
    checks when its name comes up. Any other key is refused and removed, and so is
    every key that the import of a refused entry point added. A key whose name was
    refused earlier keeps that refusal's message. An error that none of these
    checks foresaw refuses only the entry point it came from. The outcome depends
    on the entry points alone, not on which of a module's entry points loads
    first. An accepted converter is registered explicitly, so a key removed earlier
    returns when its own entry point is accepted, although its module is cached.
    """
    _InstalledLoad().run()


def track_converter_origin(src_format: str) -> InstalledOrigin | None:
    """Return the entry point and distribution of an installed converter.

    ``None`` for a built-in, and for a converter that a caller registered in Python.
    """
    ensure_track_converters_registered()
    return INSTALLED_STATE.origins.get(src_format)


def refused_track_converters() -> dict[str, str]:
    """Return each refused entry point's name and the message that says why."""
    ensure_track_converters_registered()
    return dict(INSTALLED_STATE.refusals)


def builtin_track_converters() -> dict[str, type[TrackConverter[Any]]]:
    """Return the converters that ship with mosaic, by ``src_format``.

    A converter counts as built in when its class lives under ``mosaic.``, so a
    converter that a notebook or a test registers is not one. The generated
    reference and the README counts read this set, and so do not change with
    what a process has registered.
    """
    ensure_track_converters_registered()
    return {
        src_format: cls
        for src_format, cls in TRACK_CONVERTERS.items()
        if cls.__module__.startswith("mosaic.")
    }


def merge_on_column_union(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the several files of one sequence, aligned on every column.

    What ``merges_per_sequence`` means, as an operation. The files are one per
    individual and need not agree on their columns -- TRex exports the fields
    its ``output_fields`` asked for, per individual -- so a column present for
    one and absent for another survives as NaN rather than dropping for the
    whole sequence.

    It does not read ``frame`` or ``id``. Files that are clips of one sequence,
    each numbered from frame 0, merge into a table in which one frame number
    names a moment in each clip. No check refuses that table.

    Two callers reach it: ``convert_all_tracks``, for files a user uploaded, and
    the TRex bridge, for files the tracker has just written. They had a copy
    each, which is the drift a declared capability exists to avoid.

    Reindexing rather than assigning the absent columns in, as both copies did:
    that assignment mutated the frames the converter had just returned, so a
    caller holding one saw columns it never produced appear in it.
    """
    seen: set[str] = set()
    for frame in frames:
        seen.update(map(str, frame.columns))
    all_columns = sorted(seen)
    return pd.concat(
        [frame.reindex(columns=all_columns) for frame in frames],
        ignore_index=True,
    )
