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

import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..error_kind import ClassifiedError, ErrorKind
from ..index_lock import index_lock
from .digest import canonical_recipe, recipe_digest
from .model import Recipe, Request
from .status import (
    JSON_OBJECT,
    RequestPageWire,
    RequestSummaryWire,
    request_summary,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from mosaic.core.json_value import JsonValue
    from mosaic.core.scope import Scope

__all__ = [
    "RequestUnreadable",
    "list_requests",
    "load_recipe",
    "load_request",
    "open_requests_matching",
    "pipelines_root",
    "read_recipe_document",
    "recipe_path",
    "request_path",
    "requests_matching",
    "requests_root",
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


def pipelines_root(base_dir: Path | str) -> Path:
    """``<base_dir>/.mosaic/pipelines`` -- the recipes applied to this dataset."""
    return Path(base_dir) / ".mosaic" / "pipelines"


def requests_root(base_dir: Path | str) -> Path:
    """``<base_dir>/.mosaic/pipelines/requests`` -- one file per submission."""
    return pipelines_root(base_dir) / "requests"


def recipe_path(base_dir: Path | str, digest: str) -> Path:
    """Where the recipe with this digest sits once copied in."""
    return pipelines_root(base_dir) / f"{digest}.json"


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


def read_recipe_document(path: Path | str) -> dict[str, JsonValue]:
    """Read a recipe file as the JSON object it holds, without validating it.

    The bytes a saved recipe holds are what its file name digests, so a reader
    that verifies a file against its name needs the document as written, before
    pydantic fills defaults or normalizes a value.

    Raises:
        FileNotFoundError: There is no file there.
        ValueError: It is not valid JSON, or not a JSON object.
    """
    return JSON_OBJECT.validate_python(_read_json(Path(path), "recipe"))


def load_recipe(path: Path | str) -> Recipe:
    """Read a recipe file.

    Raises:
        FileNotFoundError: There is no file there.
        ValueError: It is not valid JSON, or not a valid recipe -- including a
            ``schema_version`` newer than this mosaic understands, which is
            refused rather than read under the wrong rules.
    """
    return Recipe.model_validate(read_recipe_document(path))


def save_request(base_dir: Path | str, request: Request) -> Path:
    """Write one submission's request file, and return where it landed."""
    path = request_path(base_dir, request.request_id)
    _write_json(path, request.model_dump(mode="json"))
    return path


def load_request(base_dir: Path | str, request_id: str) -> Request:
    """Read one submission's request file.

    Raises:
        FileNotFoundError: There is no request file *request_id*.
        RequestUnreadable: The file is not valid JSON or not a valid request.
    """
    path = request_path(base_dir, request_id)
    try:
        return Request.model_validate(_read_json(path, "request"))
    except ValueError as error:
        raise RequestUnreadable(request_id, path, str(error)) from error


def list_requests(
    base_dir: Path | str, *, limit: int | None = None, after: str | None = None
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

    Returns:
        The page, and the ``after`` of the next page while unread requests remain.

    Raises:
        ValueError: *limit* is below 1.
        RequestUnreadable: A request file on the page is not valid JSON or not a
            valid request.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    names = _request_ids(base_dir)
    if after is not None:
        names = [name for name in names if name < after]
    page = names if limit is None else names[:limit]
    return RequestPageWire(
        items=tuple(
            request_summary(base_dir, load_request(base_dir, name)) for name in page
        ),
        next_after=page[-1] if len(page) < len(names) else None,
    )


def open_requests_matching(
    base_dir: Path | str,
    recipe_digest: str,
    scope: Scope,
) -> tuple[str, ...]:
    """Return the ids of the open requests of one recipe over an overlapping scope.

    A request of the same recipe whose scope shares an entry with *scope*
    (:meth:`~mosaic.core.scope.Scope.overlaps`) runs the same steps over that
    entry. ``run_feature`` runs a feature without claiming it. Two such requests
    that run at once both compute it. A request is open while its rollup is
    ``running``, including before its first step starts. A request whose
    driver lease is stale and none of whose step processes may still run is
    ``orphaned`` (:func:`~.status.request_verdict_of`) and is not open.

    A request saved under an earlier canonical form of the recipe names it by
    an older digest; it is matched by what its saved recipe digests to now.

    Every request file is read, and the run-logs of each request of the recipe.
    A request file that cannot be read, or is not a valid request of the recipe,
    is skipped with a ``UserWarning`` naming it. A damaged request file therefore
    leaves the dataset open to submissions.

    Args:
        base_dir: The dataset root.
        recipe_digest: The digest of the recipe to be submitted.
        scope: The scope to be submitted.

    Returns:
        The ids of the open requests, in descending id order. That order is
        creation order for the ids that ``submit_request`` mints.
    """
    return tuple(
        summary.request_id
        for summary in requests_matching(base_dir, recipe_digest, scope)
        if summary.verdict == "running"
    )


def requests_matching(
    base_dir: Path | str,
    recipe_digest: str,
    scope: Scope,
) -> tuple[RequestSummaryWire, ...]:
    """Summarize the requests of one recipe over an overlapping scope that have not ended.

    The requests are those :func:`open_requests_matching` reads, with a verdict
    of ``running`` or ``orphaned``. An ``orphaned`` request is not open, since
    nothing drives it, but its unfinished work is the plan's to resume or
    cancel, so a plan lists it beside the open ones.

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
    for name in _request_ids(base_dir):
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

    A request names its recipe by the digest it had when it was submitted. A
    later canonical form can name the same recipe differently, so a request
    whose digest differs is matched by what its saved recipe digests to now.
    *current* caches that answer per digest across one scan. The digest is
    compared on the raw document, so a request file of another recipe is not
    validated. A file that cannot be read or validated is ``None`` with a
    ``UserWarning`` naming it. A request whose saved recipe cannot be read is
    not a request of *recipe_digest*; the status tools report it.
    """
    try:
        document = JSON_OBJECT.validate_python(_read_json(path, "request"))
        named = document.get("recipe_digest")
        if not isinstance(named, str):
            return None
        if named != recipe_digest and (
            _current_digest(base_dir, named, current) != recipe_digest
        ):
            return None
        return Request.model_validate(document)
    except (OSError, ValueError) as error:
        warnings.warn(
            f"skipped request file {path}: {error}", UserWarning, stacklevel=3
        )
        return None


def _current_digest(
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


def _request_ids(base_dir: Path | str) -> list[str]:
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


def _read_json(path: Path, what: str) -> object:
    """Read one JSON document, naming the file when it does not parse."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"no {what} file at {path}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
