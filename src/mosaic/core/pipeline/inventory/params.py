"""The one reader of a feature run's ``params.json``.

Three parsers read this document before this module, each shaped for its own
caller and none of them public: a typed pydantic model in ``reconcile_features``
that read every key, and two raw ``json.loads`` walks -- in ``provenance`` and in
``track_universe`` -- that read ``_resolved`` and nothing else. Three readers of
one document is three answers about what it says, and two of them had already
drifted apart on a case that matters (see :meth:`RunParams.consumed_run_ids`).

**Absent and unreadable are different answers.** The parser this replaces
returned ``None`` for both, so "this run predates provenance" and "this file is
corrupt" arrived identically and were reported with one message naming both.
:func:`read_run_params` distinguishes them, which is what lets an inventory say
``params: unknown`` rather than presenting an empty dict as the run's params.
The distinction is real rather than pedantic: the write sits inside a bare
``except Exception`` that prints and continues, so a run root can legitimately
exist with no sidecar at all.

**A block that cannot be read is dropped, never fatal.** The two raw walks
tolerated a document whose ``_resolved`` was valid and whose other keys were
junk, because they never looked at the other keys; validating the whole document
strictly would have silently emptied both provenance walks on such a file. So
:func:`read_run_params` validates, removes whatever the errors blame, and
validates again, until the document reads or nothing more can be removed. The
piece removed is the deepest one the error addresses, so one malformed
``_resolved`` entry costs that entry rather than the block. Nothing here raises
on content.

The removal is driven by pydantic's own error locations rather than a table of
which key should hold what: a table is a second statement of the schema, and it
drifts from the fields silently.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from mosaic.core.filter_bounds import TimeWindow
from mosaic.core.helpers import make_entry_key
from mosaic.core.narrowing import is_str_mapping
from mosaic.core.pipeline.index import DatasetRoots, feature_run_root

__all__ = [
    "InputSource",
    "ParamsState",
    "ResolvedRef",
    "RunKey",
    "RunParams",
    "RunParamsRead",
    "ScopeBlock",
    "REPORT_MAX_CHARS",
    "ReportTooLarge",
    "TracksVariantWalk",
    "features_roots",
    "named_references",
    "read_run_params",
    "read_run_report",
    "run_key_of",
    "run_tracks_variants",
    "upstream_runs",
]

ParamsState = Literal["present", "absent", "unreadable"]
"""Whether a run's sidecar was read, was never written, or could not be parsed."""

TRACKS_EDGE: Final = "inputs[tracks]"
"""The synthetic ``_resolved`` edge naming the tracks variants a run consumed."""

LABELS_EDGE: Final = "inputs[labels]"
"""The synthetic ``_resolved`` edge naming the label variants a run consumed."""

type RunKey = tuple[str, str]
"""A feature run: its storage directory and its run id."""

type InputSource = Literal["tracks"] | RunKey
"""One input of a feature run: the tracks tables, or the feature run it read."""


class ResolvedRef(BaseModel):
    """One ``_resolved`` entry: which concrete upstream a reference pinned to.

    ``run_id`` is ``None`` when the document recorded a JSON ``null``, and ``""``
    when it recorded an unlabelled variant. The two are not the same thing and
    are not collapsed -- see :meth:`RunParams.consumed_run_ids`.
    """

    model_config = ConfigDict(extra="ignore")

    where: str = ""
    feature: str = ""
    run_id: str | None = None


class ScopeBlock(BaseModel):
    """The ``_scope`` block: the invocation's resolved scope and compositions.

    ``entries`` is the scope of the latest invocation. ``asked`` is the union of
    the resolved entries of every invocation of the run, and ``None`` in a file
    written before the key existed. ``asked_all`` records that an invocation of
    a scope-free run asked for every entry of the dataset, including entries
    added later. An invocation without a selector asks for every entry, and so
    does a recipe step that neither its request nor a step scope narrows.
    ``stacks_inputs`` records that the run read its inputs as runs over
    disjoint entries rather than over the entries they all hold, and is
    ``None`` in a file written before the key existed.
    """

    model_config = ConfigDict(extra="ignore")

    scope_dependent: bool = False
    consumed_roots: list[str] = Field(default_factory=list)
    entries: list[list[str]] = Field(default_factory=list)
    asked: list[list[str]] | None = None
    asked_all: bool = False
    stacks_inputs: bool | None = None
    compositions: dict[str, dict[str, str]] = Field(default_factory=dict)


def _pairs(entries: list[list[str]]) -> set[tuple[str, str]]:
    """Return the ``(group, sequence)`` pairs of *entries*, skipping malformed ones."""
    return {(pair[0], pair[1]) for pair in entries if len(pair) == 2}


class RunParams(BaseModel):
    """A feature run's recorded ``params.json``, typed.

    Read through a model rather than raw ``json.loads`` navigation, which keeps
    the ``Any``/``Unknown`` that untyped JSON access spreads out of every caller.
    ``extra="ignore"``, so a file written by a later scheme with new keys reads
    here unchanged.

    This is **provenance, never the digest**. Nothing on it is hashed, and
    ``compute_run_id`` builds its own payload; reconstructing an identity from
    these fields would be the second hash site rule P2e forbids.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    params: dict[str, object] = Field(default_factory=dict, alias="_params")
    inputs: list[object] = Field(default_factory=list, alias="_inputs")
    frame_range: list[object] = Field(default_factory=list, alias="_frame_range")
    overlap_frames: int = Field(0, alias="_overlap_frames")
    time_range: list[object] = Field(default_factory=list, alias="_time_range")
    scope: ScopeBlock = Field(default_factory=ScopeBlock, alias="_scope")
    resolved: list[ResolvedRef] = Field(default_factory=list, alias="_resolved")
    execution_id: str = Field("", alias="_execution_id")
    mosaic_version: str = Field("", alias="_mosaic_version")

    @property
    def records_resolutions(self) -> bool:
        """Whether the file carried a ``_resolved`` block at all.

        ``run_feature`` writes the key unconditionally, empty list included, so
        an absent one dates the file to before the block existed rather than
        saying the run resolved nothing. The distinction is load-bearing and the
        default erases it: both spellings arrive here as ``[]``.
        """
        return "resolved" in self.model_fields_set

    def frames(self) -> tuple[int | None, int | None]:
        """The recorded ``(frame_start, frame_end)``, or ``(None, None)``.

        Coerced here rather than typed on the field, so a garbage
        ``_frame_range`` costs this one answer instead of the whole document.
        """
        if len(self.frame_range) != 2:
            return None, None
        start, end = self.frame_range
        return (
            start if isinstance(start, int) else None,
            end if isinstance(end, int) else None,
        )

    def time_window(self) -> TimeWindow:
        """The recorded window of ``time`` in seconds, open when none is recorded.

        A document written before time windows were hashed as seconds records
        none, and its ``_frame_range`` already holds what its time filter became.
        """
        if len(self.time_range) != 2:
            return TimeWindow()
        start, end = self.time_range
        return TimeWindow(
            float(start) if isinstance(start, (int, float)) else None,
            float(end) if isinstance(end, (int, float)) else None,
        )

    def entry_scope(self) -> set[tuple[str, str]]:
        """The ``(group, sequence)`` pairs this invocation's scope covered."""
        return _pairs(self.scope.entries)

    def asked_scope(self) -> set[tuple[str, str]]:
        """The ``(group, sequence)`` pairs every invocation of this run asked for.

        A file written before ``asked`` existed returns :meth:`entry_scope`, the
        latest invocation's scope.
        """
        if self.scope.asked is None:
            return self.entry_scope()
        return _pairs(self.scope.asked)

    def entry_compositions(
        self, entries: set[tuple[str, str]]
    ) -> dict[tuple[str, str], dict[str, str]]:
        """Rebuild ``ResolvedScope.compositions`` from the recorded ``_scope`` block.

        The block is keyed by ``make_entry_key(group, sequence)``; *entries*
        gives the inverse map back to the tuple ``ResolvedScope`` wants, so no
        key parsing is needed.
        """
        by_key = {make_entry_key(group, seq): (group, seq) for (group, seq) in entries}
        out: dict[tuple[str, str], dict[str, str]] = {}
        for entry_key, per_root in self.scope.compositions.items():
            entry = by_key.get(entry_key)
            if entry is not None:
                out[entry] = dict(per_root)
        return out

    def variant_ids(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The recorded ``(tracks, labels)`` variant ids, in document order.

        ``compute_run_id`` reads these off ``ResolvedScope``; the document
        records them as synthetic ``inputs[tracks]`` / ``inputs[labels]``
        entries in ``_resolved``, because they are ``ResolvedScope`` terms
        rather than ``_inputs`` fields.
        """
        tracks = [
            r.run_id for r in self.resolved if r.where == TRACKS_EDGE and r.run_id
        ]
        labels = [
            r.run_id for r in self.resolved if r.where == LABELS_EDGE and r.run_id
        ]
        return tuple(tracks), tuple(labels)

    def consumed_variants(self, where: str = TRACKS_EDGE) -> frozenset[str]:
        """Named upstream variants on edge *where*.

        A falsy ``run_id`` is **dropped**: this answers a blast-radius question,
        and an unnamed variant is not a member of one.

        Deliberately not the same rule as :meth:`consumed_run_ids`. The two
        differ on exactly one case and both spellings are correct for their own
        caller, so they are two named methods rather than one with a flag.
        """
        return frozenset(
            ref.run_id for ref in self.resolved if ref.where == where and ref.run_id
        )

    def consumed_run_ids(self) -> frozenset[str]:
        """Every upstream run identifier this run read, tracks and features alike.

        **An empty ``run_id`` on a tracks edge is kept**, where
        :meth:`consumed_variants` drops it. That empty string names the
        *unlabelled* tracks variant -- every table written before variants
        existed -- and it is a real upstream. Dropped, such a run looks as though
        it consumed no tracks at all, so it never appears as another run's
        upstream and reads as a leaf of its chain forever.

        A JSON ``null`` is a different thing and is skipped: it records a
        reference that was never pinned, not a reference to the unlabelled
        variant.
        """
        return frozenset(
            ref.run_id
            for ref in self.resolved
            if ref.run_id is not None and (ref.run_id or ref.where == TRACKS_EDGE)
        )

    def consumed_tracks_variants(self) -> frozenset[str]:
        """The tracks variants on this run's tracks edge.

        The rule of :meth:`consumed_run_ids` for the tracks edge: an empty
        ``run_id`` is kept as the unlabelled variant, and a JSON ``null`` is
        skipped. ``run_feature`` records labelled variants only, so a run over the
        unlabelled tables alone records none.
        """
        return frozenset(
            ref.run_id
            for ref in self.resolved
            if ref.where == TRACKS_EDGE and ref.run_id is not None
        )

    def input_sources(self) -> tuple[InputSource, ...]:
        """The run's inputs in input order, each ``"tracks"`` or the feature run it read.

        A feature input's run is the ``inputs[<i>]`` pin of ``_resolved`` when the
        file records one, else the run id ``_inputs`` names, else ``""``.
        """
        pins = {ref.where: ref.run_id for ref in self.resolved if ref.run_id}
        sources: list[InputSource] = []
        for position, item in enumerate(self.inputs):
            if item == "tracks":
                sources.append("tracks")
                continue
            named = _named_run(item)
            if named is None:
                continue
            feature, run_id = named
            sources.append((feature, pins.get(f"inputs[{position}]") or run_id))
        return tuple(sources)

    def upstream_references(self) -> tuple[frozenset[RunKey], frozenset[str]]:
        """The feature runs this run read, as ``(pairs, storages)``.

        ``pairs`` holds the runs named with a run id, and ``storages`` the
        storages of references without one, which resolve to whichever run of
        the storage is current. A file with ``_resolved`` names them in its
        entries, the tracks and labels edges excluded. A file written before
        ``_resolved`` existed (:attr:`records_resolutions`) names them as the
        :func:`named_references` among its inputs and top-level params.
        """
        if not self.records_resolutions:
            return named_references(
                {
                    **{f"inputs[{i}]": item for i, item in enumerate(self.inputs)},
                    **self.params,
                }
            )
        pairs: set[RunKey] = set()
        storages: set[str] = set()
        for ref in self.resolved:
            if ref.where in (TRACKS_EDGE, LABELS_EDGE) or not ref.feature:
                continue
            if ref.run_id:
                pairs.add((ref.feature, ref.run_id))
            else:
                storages.add(ref.feature)
        return frozenset(pairs), frozenset(storages)


def _named_run(value: object) -> RunKey | None:
    """The ``(feature, run_id)`` that a recorded ``{feature, run_id}`` names."""
    if not is_str_mapping(value):
        return None
    feature = value.get("feature")
    run_id = value.get("run_id")
    if not isinstance(feature, str) or not feature:
        return None
    return feature, run_id if isinstance(run_id, str) else ""


def named_references(
    values: Mapping[str, object],
) -> tuple[frozenset[RunKey], frozenset[str]]:
    """The ``{feature, run_id}`` references among the values of *values*.

    Returns:
        ``(pairs, storages)``: the runs named by references with a run id, and
        the storages of references without one, which resolve to whichever run
        of the storage is current.
    """
    pairs: set[RunKey] = set()
    storages: set[str] = set()
    for value in values.values():
        named = _named_run(value)
        if named is None:
            continue
        if named[1]:
            pairs.add(named)
        else:
            storages.add(named[0])
    return frozenset(pairs), frozenset(storages)


@dataclass(frozen=True, slots=True)
class RunParamsRead:
    """The outcome of reading one run's sidecar: what it said, or why it did not."""

    state: ParamsState
    params: RunParams | None
    error: str = ""

    @property
    def finding(self) -> str:
        """A one-line reason for a report, empty when the document was read.

        Two messages where the parser this replaces had one naming both states,
        because the remedies differ: a missing sidecar is a run that predates
        provenance and cannot be recovered, an unreadable one is a file to look at.
        """
        if self.state == "absent":
            return "params.json is missing"
        if self.state == "unreadable":
            return f"params.json is unreadable: {self.error}"
        return ""


_DOCUMENT: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
"""Parses the sidecar into a fully-typed JSON tree.

``json.loads`` returns ``Any``, and narrowing that with ``isinstance`` yields a
container whose members stay unknown however it is annotated afterwards -- so
every read off it spreads the unknown outward. Typing the document once at the
boundary is what makes the walk below ordinary typed code.
"""


def _without(
    value: JsonValue, location: tuple[int | str, ...]
) -> tuple[JsonValue, bool]:
    """*value* with the deepest element *location* addresses removed.

    Pure rather than a mutating walk: the caller re-validates the result, and a
    rebuilt tree is easier to reason about than a document edited underneath a
    model that already rejected it.

    **Deepest, not outermost.** A location of ``("_scope", "compositions")``
    costs that one key and leaves the entries beside it readable; a location of
    ``("_resolved", 3, ...)`` costs entry 3, where dropping ``_resolved`` whole
    would make the run look dependency-free -- the reading that moves an artifact
    under a lineage it never had. A step the document does not have stops the
    descent, and the last step that resolved is the one removed.

    Returns:
        ``(rebuilt, changed)``. ``changed`` is ``False`` when the location
        addresses nothing, which is how the caller knows to stop rather than loop.
    """
    if not location:
        return value, False
    key, rest = location[0], location[1:]
    if isinstance(value, dict):
        if not isinstance(key, str) or key not in value:
            return value, False
        if rest:
            inner, changed = _without(value[key], rest)
            if changed:
                return {**value, key: inner}, True
        return {k: v for k, v in value.items() if k != key}, True
    if isinstance(value, list):
        if not isinstance(key, int) or not 0 <= key < len(value):
            return value, False
        if rest:
            inner, changed = _without(value[key], rest)
            if changed:
                return [*value[:key], inner, *value[key + 1 :]], True
        return [item for position, item in enumerate(value) if position != key], True
    return value, False


def _without_unreadable(
    document: JsonValue, error: ValidationError
) -> tuple[JsonValue, bool]:
    """*document* with every block the errors blame removed.

    Driven by pydantic's own error locations rather than a hand-written table of
    which key should hold what. A table is a second statement of the schema and
    drifts from the fields silently: a field renamed later would keep passing the
    table check and quietly stop being protected.
    """
    current: JsonValue = document
    dropped = False
    for entry in error.errors():
        current, changed = _without(current, entry["loc"])
        dropped = dropped or changed
    return current, dropped


REPORT_MAX_CHARS: Final = 40_000
"""The longest report :func:`read_run_report` returns, in bytes on disk.

The read budget of the MCP server's results, which serves a report whole and
could not narrow a longer one.
"""


class ReportTooLarge(ValueError):
    """A run's report is longer than :data:`REPORT_MAX_CHARS`."""

    def __init__(self, path: Path, size: int) -> None:
        self.path: Path = path
        self.size: int = size
        super().__init__(
            f"the report at {path} holds {size} characters, more than the "
            f"{REPORT_MAX_CHARS} a reader returns"
        )


def read_run_report(run_root: Path, file_name: str) -> JsonValue | None:
    """Read the JSON report a feature run saved as *file_name*, or ``None``.

    A feature that writes a report names its file in a ``report_file`` class
    attribute, as ``evaluate-labels`` names ``metrics.json``. Reads only.

    Args:
        run_root: A run directory, which need not exist.
        file_name: The report's file name inside it, one path component.

    Returns:
        The report, or ``None`` when the run has none.

    Raises:
        ValueError: *file_name* is not one path component, or the file is not
            JSON.
        ReportTooLarge: The file is longer than :data:`REPORT_MAX_CHARS`.
    """
    if not file_name or Path(file_name).name != file_name:
        raise ValueError(f"a report file is one path component, not {file_name!r}")
    path = run_root / file_name
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return None
    if size > REPORT_MAX_CHARS:
        raise ReportTooLarge(path, size)
    report: JsonValue = _DOCUMENT.validate_json(path.read_text())
    return report


def read_run_params(run_root: Path) -> RunParamsRead:
    """Read ``<run_root>/params.json``. Never raises.

    A block the model cannot read is dropped and the rest is returned, because
    the two provenance walks this reader replaced only ever looked at
    ``_resolved`` and so tolerated junk elsewhere in the document. Validating
    strictly would have silently emptied both on such a file.

    Args:
        run_root: A run directory, which need not exist.

    Returns:
        A :class:`RunParamsRead` whose ``state`` distinguishes a sidecar that was
        read, one that was never written, and one that could not be parsed at
        all -- which now means only a document that is not a mapping.
    """
    path = run_root / "params.json"
    try:
        text = path.read_text()
    except FileNotFoundError:
        return RunParamsRead("absent", None)
    except OSError as exc:
        return RunParamsRead("unreadable", None, str(exc))
    try:
        document: JsonValue = _DOCUMENT.validate_json(text)
    except ValidationError as exc:
        return RunParamsRead("unreadable", None, str(exc))
    if not isinstance(document, dict):
        return RunParamsRead("unreadable", None, "document is not a JSON object")
    # Bounded by the work done per pass: every iteration removes at least one
    # piece or stops, so this cannot spin on a document that keeps failing.
    while True:
        try:
            return RunParamsRead("present", RunParams.model_validate(document))
        except ValidationError as exc:
            document, dropped = _without_unreadable(document, exc)
            if not dropped:
                return RunParamsRead("unreadable", None, str(exc))


class TracksVariantWalk:
    """The tracks variants that feature runs read, reading each ``params.json`` once.

    A run that reads ``tracks`` records the variants on its tracks edge
    (:data:`TRACKS_EDGE`). A run that reads another feature's output records no
    tracks edge, because its upstream hashed the variant it read. The walk
    follows every other ``_resolved`` entry to that upstream's ``params.json``.
    A run over the unlabeled tables records no tracks edge, because
    ``run_feature`` records only labeled variants, and its walk returns an empty
    set.

    Args:
        ds: The dataset whose ``features`` root contains the runs.
        known: Params already read, keyed by run. A run listed here is not read
            from disk, so a held inventory's records answer for the runs that
            it holds.
    """

    def __init__(
        self, ds: DatasetRoots, known: Mapping[RunKey, RunParams | None] | None = None
    ) -> None:
        self._ds: DatasetRoots = ds
        self._params: dict[RunKey, RunParams | None] = dict(known or {})

    def __call__(self, storage_name: str, run_id: str) -> frozenset[str] | None:
        """Return the tracks variants that one run read, directly or through its upstreams.

        Args:
            storage_name: The run's feature storage directory.
            run_id: The run.

        Returns:
            The variants, or ``None`` when the answer is unknown: a
            ``params.json`` on the walk is absent or unreadable, or predates the
            ``_resolved`` block (:attr:`RunParams.records_resolutions`).
        """
        variants: set[str] = set()
        start = (storage_name, run_id)
        for _, params in upstream_runs(self._ds, [start], self._params):
            if params is None or not params.records_resolutions:
                return None
            variants.update(params.consumed_tracks_variants())
        return frozenset(variants)


def upstream_runs(
    ds: DatasetRoots,
    starts: Iterable[RunKey],
    cache: dict[RunKey, RunParams | None] | None = None,
) -> Iterator[tuple[RunKey, RunParams | None]]:
    """Yield each run of *starts* and every feature run upstream of them, once.

    The starts come first, then their upstreams breadth first, each with the
    params :func:`read_run_params` reads: ``None`` when its ``params.json`` is
    absent or unreadable. A run leads to the runs named in its
    :meth:`RunParams.upstream_references`, in sorted order, and a run whose
    params are ``None`` is yielded and not followed. A step bound to an earlier
    run is that run, because its downstream recorded the bound run's id.

    Args:
        ds: The dataset whose ``features`` root holds the runs.
        starts: The runs to walk from.
        cache: Params already read, read and filled in place, so a caller that
            walks many times reads each ``params.json`` once. Without one the
            call keeps its own.

    Yields:
        ``(run, params)`` for each run reached.
    """
    params_of: dict[RunKey, RunParams | None] = {} if cache is None else cache
    seen: set[RunKey] = set()
    pending: deque[RunKey] = deque(starts)
    while pending:
        node = pending.popleft()
        if node in seen:
            continue
        seen.add(node)
        if node not in params_of:
            params_of[node] = read_run_params(feature_run_root(ds, *node)).params
        params = params_of[node]
        yield node, params
        if params is not None:
            pending.extend(sorted(params.upstream_references()[0]))


@dataclass(frozen=True, slots=True)
class _FeaturesRoot:
    """A dataset's ``features`` root, and no other root."""

    features: Path

    def get_root(self, key: str, /) -> Path:
        """Return the ``features`` root, and raise ``KeyError`` for any other."""
        if key != "features":
            raise KeyError(f"only the features root is known here, not {key!r}")
        return self.features


def features_roots(run_root: Path) -> DatasetRoots:
    """The roots of the dataset holding the feature run at *run_root*.

    For a caller that holds a run root and no dataset, such as a feature's
    ``load_state``. The ``features`` root is ``run_root.parent.parent``, and
    every other root raises ``KeyError``.
    """
    return _FeaturesRoot(run_root.parent.parent)


def run_key_of(run_root: Path) -> RunKey:
    """The ``(storage, run_id)`` of the feature run at *run_root*.

    Args:
        run_root: A run directory, ``<features>/<storage>/<run_id>``.
    """
    return run_root.parent.name, run_root.name


def run_tracks_variants(
    ds: DatasetRoots, storage_name: str, run_id: str
) -> frozenset[str] | None:
    """Return the tracks variants that one feature run read, as :class:`TracksVariantWalk` walks them."""
    return TracksVariantWalk(ds)(storage_name, run_id)
