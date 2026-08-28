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

from mosaic.core.helpers import make_entry_key

__all__ = [
    "ParamsState",
    "ResolvedRef",
    "RunParams",
    "RunParamsRead",
    "ScopeBlock",
    "read_run_params",
]

ParamsState = Literal["present", "absent", "unreadable"]
"""Whether a run's sidecar was read, was never written, or could not be parsed."""

TRACKS_EDGE: Final = "inputs[tracks]"
"""The synthetic ``_resolved`` edge naming the tracks variants a run consumed."""

LABELS_EDGE: Final = "inputs[labels]"
"""The synthetic ``_resolved`` edge naming the label variants a run consumed."""


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
    """The ``_scope`` block: the invocation's resolved scope and compositions."""

    model_config = ConfigDict(extra="ignore")

    scope_dependent: bool = False
    consumed_roots: list[str] = Field(default_factory=list)
    entries: list[list[str]] = Field(default_factory=list)
    compositions: dict[str, dict[str, str]] = Field(default_factory=dict)


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

    def entry_scope(self) -> set[tuple[str, str]]:
        """The ``(group, sequence)`` pairs this invocation's scope covered."""
        return {(pair[0], pair[1]) for pair in self.scope.entries if len(pair) == 2}

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
