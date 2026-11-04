"""Where a dataset keeps the recipes applied to it, and the requests against them.

A recipe is a file that lives wherever its author keeps files -- a git
repository, a shared folder, an attachment. On first use against a dataset it is
**copied in**, so the dataset records which pipelines were applied to it and can
be handed to someone else intact, with no reference to a location the recipient
cannot reach.

Addressed by digest rather than by name, which makes the copy idempotent: the
same recipe applied twice is one file, and two recipes that happen to share a
name are two.

``base_dir`` rather than a ``Dataset`` throughout, matching ``runlog``'s own
path helpers. Reading a request must stay cheap enough for a release gate, and
constructing a dataset to learn a path is the opposite of that. For the same
reason, importing this module leaves pandas, cv2, av and the feature library
unloaded (``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

import datetime
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import TypeAdapter

from mosaic.core.json_object import JSON_OBJECT
from mosaic.runlog import (
    changed_at_or_after,
    parse_instant,
    reduce_run_log,
    run_log_dir,
    written_since,
)

from ..error_kind import ClassifiedError, ErrorKind
from ..index_lock import index_lock
from .digest import canonical_recipe, recipe_digest
from .model import ComputedRun, Recipe, Request
from .recipe_store import (
    load_recipe,
    pipelines_root,
    read_json,
    recipe_path,
)
from .status import (
    RequestPageWire,
    RequestSummaryWire,
    request_summary,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from mosaic.core.json_value import JsonValue
    from mosaic.core.scope import Scope

__all__ = [
    "DUPLICATE_STEP_SAMPLE",
    "OpenDuplicate",
    "describe_duplicate",
    "RequestUnreadable",
    "computes_path",
    "computes_root",
    "current_digest",
    "list_requests",
    "load_computes",
    "load_request",
    "open_duplicates",
    "request_ids",
    "request_path",
    "requests_matching",
    "requests_root",
    "save_computes",
    "save_recipe",
    "save_request",
    "submission_lock",
]


class RequestUnreadable(ClassifiedError, ValueError):
    """A request file that is not valid JSON or not a valid request.

    Attributes:
        request_id: The request read.
        path: The request file.
    """

    error_kind: ClassVar[ErrorKind] = "unavailable"

    def __init__(self, request_id: str, path: Path, problem: str) -> None:
        super().__init__(f"request file {path} does not read as a request: {problem}")
        self.request_id: str = request_id
        self.path: Path = path

    def error_reason(self) -> str:
        """Return ``request_unreadable``."""
        return "request_unreadable"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the request and its file."""
        return {"request_id": self.request_id, "path": str(self.path)}


def requests_root(base_dir: Path | str) -> Path:
    """``<base_dir>/.mosaic/pipelines/requests`` -- one file per submission."""
    return pipelines_root(base_dir) / "requests"


def computes_root(base_dir: Path | str) -> Path:
    """``<base_dir>/.mosaic/pipelines/computes`` -- the runs each request computes."""
    return pipelines_root(base_dir) / "computes"


def computes_path(base_dir: Path | str, request_id: str) -> Path:
    """Where the runs that one submission computes are recorded."""
    return computes_root(base_dir) / f"{request_id}.json"


def request_path(base_dir: Path | str, request_id: str) -> Path:
    """Where one submission's request file sits."""
    return requests_root(base_dir) / f"{request_id}.json"


def save_recipe(base_dir: Path | str, recipe: Recipe) -> Path:
    """Copy *recipe* into the dataset, and return where it landed.

    Writes the **canonical** document rather than whatever the author's file
    said, so the bytes on disk are the bytes the digest names and a reader can
    verify the one against the other. Idempotent by construction: the same recipe
    canonicalizes identically, so re-saving rewrites the same content at the same
    path.
    """
    digest = recipe_digest(recipe)
    path = recipe_path(base_dir, digest)
    _write_json(path, canonical_recipe(recipe))
    return path


def save_request(base_dir: Path | str, request: Request) -> Path:
    """Write one submission's request file, and return where it landed.

    A request without step scopes, a tracks variant, pins, an applied request,
    locked steps, a cleanup, or a cleanup that retires no recipe is written
    without those keys, so its file is what an older mosaic wrote and reads
    there.
    """
    path = request_path(base_dir, request.request_id)
    exclude: dict[str, bool | dict[str, bool]] = {}
    if not request.step_scopes:
        exclude["step_scopes"] = True
    if request.tracks_run_id is None:
        exclude["tracks_run_id"] = True
    if not request.pins:
        exclude["pins"] = True
    if request.applies_request_id is None:
        exclude["applies_request_id"] = True
    if not request.locked:
        exclude["locked"] = True
    if request.cleanup is None:
        exclude["cleanup"] = True
    elif not request.cleanup.retired:
        exclude["cleanup"] = {"retired": True}
    _write_json(path, request.model_dump(mode="json", exclude=exclude or None))
    return path


_COMPUTED_RUNS: Final = TypeAdapter(list[ComputedRun])


def save_computes(
    base_dir: Path | str, request_id: str, runs: Sequence[ComputedRun]
) -> Path:
    """Record the runs that one submission's plan left to compute.

    A file of its own under :func:`computes_root` and not a key of the request
    file, because an older mosaic refuses a request file with a key it does not
    know. An empty list records a request that computes nothing, which differs
    from a request that has no such file. Nothing removes the file. One whose
    request file is gone is never read, since :func:`open_duplicates` reads it
    by the id of a request file.
    """
    path = computes_path(base_dir, request_id)
    _write_json(path, _COMPUTED_RUNS.dump_python(list(runs), mode="json"))
    return path


def load_computes(base_dir: Path | str, request_id: str) -> list[ComputedRun] | None:
    """The runs a submission recorded with :func:`save_computes`, or ``None``.

    ``None`` for a request submitted before mosaic recorded them, and for a
    file that does not read, with a ``UserWarning`` naming it.
    """
    path = computes_path(base_dir, request_id)
    if not path.is_file():
        return None
    try:
        return _COMPUTED_RUNS.validate_python(read_json(path, "computed runs"))
    except (OSError, ValueError) as error:
        warnings.warn(
            f"skipped computed runs file {path}: {error}", UserWarning, stacklevel=3
        )
        return None


def load_request(base_dir: Path | str, request_id: str) -> Request:
    """Read one submission's request file.

    Raises:
        FileNotFoundError: There is no request file *request_id*.
        RequestUnreadable: The file is not valid JSON or not a valid request.
    """
    path = request_path(base_dir, request_id)
    try:
        return Request.model_validate(read_json(path, "request"))
    except ValueError as error:
        raise RequestUnreadable(request_id, path, str(error)) from error


def list_requests(
    base_dir: Path | str,
    *,
    limit: int | None = None,
    after: str | None = None,
    changed_since: datetime.datetime | None = None,
) -> RequestPageWire:
    """List one page of a dataset's pipeline requests, newest first.

    The request file names are listed and sorted, and only the requests on the page
    are read and summarized, one run-log per step. A request id minted by
    ``submit_request`` is a ULID, and a descending sort of the names is creation-time
    descending. A request submitted after a page was read sorts before that page and
    leaves the next page unchanged.

    Args:
        base_dir: The dataset root.
        limit: The most requests to return, at least 1. ``None`` returns all.
        after: The ``next_after`` of the previous page. The page starts at the first
            request id that sorts before it.
        changed_since: Keep only the requests created at or after this instant,
            or one of whose step attempts was created, started or ended then
            (:func:`~mosaic.runlog.changed_at_or_after`). Every request file is
            read, and the run-logs written since. A file that does not read as
            a request is kept, and raises where the page reaches it.

    Returns:
        The page, and the ``after`` of the next page while unread requests remain.

    Raises:
        ValueError: *limit* is below 1.
        RequestUnreadable: A request file on the page is not valid JSON or not a
            valid request.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    names = request_ids(base_dir)
    if after is not None:
        names = [name for name in names if name < after]
    if changed_since is not None:
        names = [
            name for name in names if _changed_since(base_dir, name, changed_since)
        ]
    page = names if limit is None else names[:limit]
    return RequestPageWire(
        items=tuple(
            request_summary(base_dir, load_request(base_dir, name)) for name in page
        ),
        next_after=page[-1] if len(page) < len(names) else None,
    )


def _changed_since(
    base_dir: Path | str, request_id: str, instant: datetime.datetime
) -> bool:
    """Whether a request was created, or one of its attempts changed, at or after *instant*.

    An unreadable request file counts as changed, so the page that reaches it
    reports it as an unfiltered listing does.
    """
    try:
        request = load_request(base_dir, request_id)
    except RequestUnreadable:
        return True
    created = parse_instant(request.created_at) if request.created_at else None
    if created is not None and created >= instant:
        return True
    logs = run_log_dir(base_dir)
    for execution_id in request.step_executions.values():
        path = logs / f"{execution_id}.jsonl"
        if not written_since(path, instant):
            continue
        snap = reduce_run_log(path)
        if snap is not None and changed_at_or_after(snap, instant):
            return True
    return False


DUPLICATE_STEP_SAMPLE: Final = 3
"""The shared steps that a duplicate names in a message or a JSON object.

The rest are counted, so a recipe of many steps keeps a refusal, a warning and
a ``submit_pipeline`` result short.
"""


@dataclass(frozen=True, slots=True)
class OpenDuplicate:
    """An open request that computes a run that a submission computes.

    Attributes:
        request_id: The open request.
        steps: The submission's steps whose runs the open request also
            computes, in the submission's order. Empty for a request that
            recorded no runs (:func:`load_computes` is ``None``), which is
            matched by its recipe and scope instead.
    """

    request_id: str
    steps: tuple[str, ...] = ()

    @property
    def named_steps(self) -> tuple[str, ...]:
        """The first :data:`DUPLICATE_STEP_SAMPLE` shared steps."""
        return self.steps[:DUPLICATE_STEP_SAMPLE]

    @property
    def steps_omitted(self) -> int:
        """How many shared steps :attr:`named_steps` leaves out."""
        return max(len(self.steps) - DUPLICATE_STEP_SAMPLE, 0)

    def as_json(self) -> dict[str, JsonValue]:
        """Return the request id, the named shared steps and the count of the rest."""
        return {
            "request_id": self.request_id,
            "steps": list(self.named_steps),
            "steps_omitted": self.steps_omitted,
        }


def describe_duplicate(request_id: str, steps: Sequence[str], omitted: int) -> str:
    """Name one open request, the *steps* it shares and a count of *omitted* more.

    A request that shares no named step is named as one of the same recipe.
    """
    if not steps:
        return f"{request_id} (the same recipe)"
    named = ", ".join(map(repr, steps))
    more = f" and {omitted} more" if omitted else ""
    return f"{request_id} (steps {named}{more})"


def open_duplicates(
    base_dir: Path | str,
    recipe_digest: str,
    scope: Scope,
    computes: Sequence[ComputedRun],
    *,
    step_scopes: Mapping[str, Scope] | None = None,
) -> tuple[OpenDuplicate, ...]:
    """Return the open requests that compute a run that a submission computes.

    A request of any recipe that recorded the runs it computes
    (:func:`load_computes`) duplicates the submission where both name one run
    (storage and ``run_id``). A run that either side computes whole
    (``ComputedRun.whole``) is duplicated over any entries. Any other run is
    duplicated only when the two steps' entries can meet: each side's scope
    and its step's scope, compared pairwise with
    :meth:`~mosaic.core.scope.Scope.overlaps`. A request that recorded no runs
    duplicates the submission when it is of the same recipe and its scope
    overlaps *scope*, as :func:`requests_matching` reads it.

    ``run_feature`` runs a feature without claiming it, so two requests that
    compute one run at once both compute it. A request is open while its rollup
    is ``running``, including before its first step starts. A request whose
    driver lease is stale and none of whose step processes may still run is
    ``orphaned`` (:func:`~.status.request_verdict_of`) and is not open.

    Every request file is read, and the run-logs of each request that shares a
    run. A request file that cannot be read, or is not a valid request, is
    skipped with a ``UserWarning`` naming it. A damaged request file therefore
    leaves the dataset open to submissions.

    Args:
        base_dir: The dataset root.
        recipe_digest: The digest of the recipe to be submitted.
        scope: The scope to be submitted.
        computes: The runs the submission's plan leaves to compute.
        step_scopes: The scopes narrowing the submission's steps, by step id.

    Returns:
        The open requests, in descending id order. That order is creation
        order for the ids that ``submit_request`` mints.
    """
    narrowed = step_scopes or {}
    found: list[OpenDuplicate] = []
    current: dict[str, str | None] = {}
    for name in request_ids(base_dir):
        path = request_path(base_dir, name)
        document = _request_document(path)
        request = None if document is None else _validated(path, document)
        if request is None:
            continue
        theirs = load_computes(base_dir, request.request_id)
        if theirs is None:
            shared: tuple[str, ...] = ()
            if not request.scope.overlaps(scope) or not _names_recipe(
                base_dir, request.recipe_digest, recipe_digest, current
            ):
                continue
        else:
            shared = _shared_steps(computes, scope, narrowed, request, theirs)
            if not shared:
                continue
        if request_summary(base_dir, request).verdict == "running":
            found.append(OpenDuplicate(request.request_id, shared))
    return tuple(found)


def _shared_steps(
    computes: Sequence[ComputedRun],
    scope: Scope,
    step_scopes: Mapping[str, Scope],
    request: Request,
    recorded: Sequence[ComputedRun],
) -> tuple[str, ...]:
    """Return the steps of *computes* whose runs *request* also computes.

    *recorded* is what *request* recorded that it computes.
    """
    theirs: dict[tuple[str, str], list[ComputedRun]] = {}
    for run in recorded:
        theirs.setdefault((run.storage, run.run_id), []).append(run)
    shared: list[str] = []
    for run in computes:
        if run.step in shared:
            continue
        mine = _selectors(scope, step_scopes.get(run.step))
        if any(
            run.whole
            or other.whole
            or all(
                one.overlaps(two)
                for one in mine
                for two in _selectors(request.scope, request.step_scope(other.step))
            )
            for other in theirs.get((run.storage, run.run_id), ())
        ):
            shared.append(run.step)
    return tuple(shared)


def _selectors(scope: Scope, step_scope: Scope | None) -> tuple[Scope, ...]:
    """The selectors that narrow one step: its request's, and its own if any."""
    return (scope,) if step_scope is None else (scope, step_scope)


def requests_matching(
    base_dir: Path | str,
    recipe_digest: str,
    scope: Scope,
) -> tuple[RequestSummaryWire, ...]:
    """Summarize the requests of one recipe over an overlapping scope that have not ended.

    The requests are those whose scope shares an entry with *scope*
    (:meth:`~mosaic.core.scope.Scope.overlaps`), with a verdict of ``running``
    or ``orphaned``. An ``orphaned`` request is not open, since nothing drives
    it, but its unfinished work is the plan's to resume or cancel, so a plan
    lists it beside the open ones.

    A request saved under an earlier canonical form of the recipe names it by
    an older digest; it is matched by what its saved recipe digests to now. A
    request file that cannot be read, or is not a valid request of the recipe,
    is skipped with a ``UserWarning`` naming it.

    Args:
        base_dir: The dataset root.
        recipe_digest: The digest of the recipe.
        scope: The scope to be planned or submitted.

    Returns:
        One summary per such request (:func:`~.status.request_summary`), in
        descending id order.
    """
    found: list[RequestSummaryWire] = []
    current: dict[str, str | None] = {}
    for name in request_ids(base_dir):
        request = _request_of_recipe(
            base_dir, request_path(base_dir, name), recipe_digest, current
        )
        if request is None or not request.scope.overlaps(scope):
            continue
        summary = request_summary(base_dir, request)
        if summary.verdict in ("running", "orphaned"):
            found.append(summary)
    return tuple(found)


def _request_of_recipe(
    base_dir: Path | str,
    path: Path,
    recipe_digest: str,
    current: dict[str, str | None],
) -> Request | None:
    """Read the request at *path* when it is a request of *recipe_digest*, else ``None``.

    The digest is compared on the raw document, so a request file of another
    recipe is not validated. A file that cannot be read or validated is
    ``None`` with a ``UserWarning`` naming it.
    """
    document = _request_document(path)
    if document is None:
        return None
    named = document.get("recipe_digest")
    if not isinstance(named, str) or not _names_recipe(
        base_dir, named, recipe_digest, current
    ):
        return None
    return _validated(path, document)


def _names_recipe(
    base_dir: Path | str,
    named: str,
    recipe_digest: str,
    current: dict[str, str | None],
) -> bool:
    """Whether a request naming *named* is a request of *recipe_digest*.

    A request names its recipe by the digest it had when it was submitted. A
    later canonical form can name the same recipe differently, so a digest that
    differs is compared by what its saved recipe digests to now. *current*
    caches that answer per digest across one scan. A request whose saved recipe
    cannot be read is not a request of *recipe_digest*; the status tools report
    it.
    """
    return named == recipe_digest or (
        current_digest(base_dir, named, current) == recipe_digest
    )


def _request_document(path: Path) -> dict[str, JsonValue] | None:
    """The JSON object at *path*, or ``None`` with a ``UserWarning`` naming it."""
    try:
        return JSON_OBJECT.validate_python(read_json(path, "request"))
    except (OSError, ValueError) as error:
        warnings.warn(
            f"skipped request file {path}: {error}", UserWarning, stacklevel=4
        )
        return None


def _validated(path: Path, document: dict[str, JsonValue]) -> Request | None:
    """*document* as a request, or ``None`` with a ``UserWarning`` naming *path*."""
    try:
        return Request.model_validate(document)
    except ValueError as error:
        warnings.warn(
            f"skipped request file {path}: {error}", UserWarning, stacklevel=4
        )
        return None


def current_digest(
    base_dir: Path | str, named: str, current: dict[str, str | None]
) -> str | None:
    """What the recipe saved under *named* digests to now, or ``None`` if unreadable."""
    if named not in current:
        try:
            current[named] = recipe_digest(load_recipe(recipe_path(base_dir, named)))
        except (OSError, ValueError):
            current[named] = None
    return current[named]


@contextmanager
def submission_lock(base_dir: Path | str) -> Generator[None]:
    """Serialize the duplicate check and the request write of one submission.

    Takes :func:`~mosaic.core.pipeline.index_lock.index_lock` on
    ``requests/.submit``. The lock creates that empty file and its
    ``.submit.lock`` sidecar, and neither is a request file. Two submissions on
    one mount check and write one after the other. Submissions from two hosts
    to one synced folder are not serialized.
    """
    root = requests_root(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    with index_lock(root / ".submit"):
        yield


def request_ids(base_dir: Path | str) -> list[str]:
    """Return the ids of a dataset's request files in descending order.

    A request id minted by ``submit_request`` is a ULID, so that order is creation
    time descending for those ids. A caller-supplied id sorts by its text.
    """
    root = requests_root(base_dir)
    if not root.is_dir():
        return []
    return sorted((path.stem for path in root.glob("*.json")), reverse=True)


def _write_json(path: Path, payload: object) -> None:
    """Write *payload* as indented JSON, atomically.

    Indented rather than compact: these are files a human reads and diffs in a
    review, and the canonical compact form exists for the digest rather than for
    the copy. Through ``atomic_write`` so a reader never sees a partial document
    and an interrupted write leaves a whole one in place. ``atomic_write`` is
    imported in the call because its module loads pandas, which a reader of this
    module leaves unloaded.
    """
    from .._utils import atomic_write

    atomic_write(path, lambda temp: temp.write_text(json.dumps(payload, indent=2)))
