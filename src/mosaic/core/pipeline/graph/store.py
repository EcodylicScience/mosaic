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
from pathlib import Path

from .digest import recipe_digest
from .model import Recipe, Request
from .status import RequestPageWire, request_summary

__all__ = [
    "list_requests",
    "load_recipe",
    "load_request",
    "pipelines_root",
    "recipe_path",
    "request_path",
    "requests_root",
    "save_recipe",
    "save_request",
]


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
    _write_json(path, recipe.model_dump(mode="json"))
    return path


def load_recipe(path: Path | str) -> Recipe:
    """Read a recipe file.

    Raises:
        FileNotFoundError: There is no file there.
        ValueError: It is not valid JSON, or not a valid recipe -- including a
            ``schema_version`` newer than this mosaic understands, which is
            refused rather than read under the wrong rules.
    """
    return Recipe.model_validate(_read_json(Path(path), "recipe"))


def save_request(base_dir: Path | str, request: Request) -> Path:
    """Write one submission's request file, and return where it landed."""
    path = request_path(base_dir, request.request_id)
    _write_json(path, request.model_dump(mode="json"))
    return path


def load_request(base_dir: Path | str, request_id: str) -> Request:
    """Read one submission's request file."""
    path = request_path(base_dir, request_id)
    return Request.model_validate(_read_json(path, "request"))


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
        ValueError: *limit* is below 1, or a request file on the page is not valid
            JSON or not a valid request.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    root = requests_root(base_dir)
    names = (
        sorted((path.stem for path in root.glob("*.json")), reverse=True)
        if root.is_dir()
        else []
    )
    if after is not None:
        names = [name for name in names if name < after]
    page = names if limit is None else names[:limit]
    return RequestPageWire(
        items=tuple(
            request_summary(base_dir, load_request(base_dir, name)) for name in page
        ),
        next_after=page[-1] if len(page) < len(names) else None,
    )


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
