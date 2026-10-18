"""Retiring a saved recipe, so a cleanup reaches the runs its requests wrote.

A cleanup judges the runs of the storages that a plan's feature steps write
(:mod:`~.cleanup`). A recipe that was rebuilt leaves its old storages behind, and
their runs are outside every later plan. A submission with a cleanup may therefore
**retire** a saved recipe by its digest: the feature runs that the recipe's requests
*wrote* become candidates wherever they sit, and stop being protected as another
recipe's (``recipe_uses``). A retirement belongs to that one submission. It is
recorded in the request's cleanup and in the plan's fingerprint, and a later plan
judges as if nothing had been retired.

This module holds what retiring reads and refuses, and none of the judging:

- :func:`validate_retirement` refuses a digest that no saved recipe has, the
  submitted recipe's own, and a recipe with a request that has not ended.
- :func:`retired_digest_of`, :func:`written_runs` and :func:`exclusively_written` say
  which requests are retired and which runs they wrote. The judging in
  :mod:`~.cleanup` calls them from the loop over the requests it already reads.
- :func:`retirable_recipes` lists the recipes a person or an agent may retire.

A run is **written** by a request when one of its attempts named the run, was not a
full cache hit, and ran in a process of its own. A stalled step records the run it
would have used and a cached step records the run it found, and neither wrote it.
A run that any feature attempt outside every request also names was used by someone
else, and is not exclusively the retired recipe's.

The module reads request files, rollups and run-logs only. It imports no planner, so
that ``cleanup`` and ``request`` can import it, and it leaves pandas, cv2, av and
the feature library unloaded (``tests/meta/test_status_imports.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Final

from mosaic.core.scope import Scope
from mosaic.runlog import list_runs, run_log_dir

from ..error_kind import ClassifiedError, ErrorKind
from ..markers import parse_instant
from .digest import DIGEST_LENGTH, recipe_digest
from .rollup import request_rollup
from .status import (
    RETIRABLE_REQUEST_IDS,
    RetirableRecipeWire,
    request_verdict_of,
)
from .store import (
    RequestUnreadable,
    current_digest,
    load_recipe,
    load_request,
    pipelines_root,
    recipe_path,
    request_ids,
    requests_matching,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Container, Iterable
    from pathlib import Path

    from mosaic.core.json_value import JsonValue

    from .model import Recipe, Request
    from .rollup import RequestRollup

__all__ = [
    "RetireNeedsCleanup",
    "RetireOpenRecipe",
    "RetireOwnRecipe",
    "RetireUnknownRecipe",
    "exclusively_written",
    "retirable_recipes",
    "retired_digest_of",
    "retired_recipe_names",
    "validate_retirement",
    "written_runs",
]

_DIGEST: Final = re.compile(rf"[0-9a-f]{{{DIGEST_LENGTH}}}")

NAME_LIMIT: Final = 80
"""The most characters of a recipe's ``name`` that a listing or a question shows.

A name is dataset content of any length, and a plan or a confirmation holds a few.
"""


class RetireNeedsCleanup(ClassifiedError):
    """Refuse ``retire`` without a cleanup, which is what removes the runs."""

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self) -> None:
        super().__init__(
            "retire names recipes whose runs a cleanup removes, so it needs a "
            "cleanup: plan or submit with cleanup 'superseded' "
            "(--cleanup superseded)"
        )

    def error_reason(self) -> str:
        """Return ``retire_needs_cleanup``."""
        return "retire_needs_cleanup"


class RetireUnknownRecipe(ClassifiedError):
    """Refuse a digest that no saved recipe of the dataset has.

    Attributes:
        digest: The digest as given.
    """

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, digest: str, problem: str) -> None:
        super().__init__(f"cannot retire recipe {digest!r}: {problem}")
        self.digest: str = digest

    def error_reason(self) -> str:
        """Return ``retire_unknown_recipe``."""
        return "retire_unknown_recipe"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the digest."""
        return {"digest": self.digest}


class RetireOwnRecipe(ClassifiedError):
    """Refuse retiring the recipe that is being planned or submitted.

    Attributes:
        digest: The recipe's digest.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self, digest: str) -> None:
        super().__init__(
            f"recipe {digest!r} is the one being planned, so it cannot be "
            f"retired by its own submission"
        )
        self.digest: str = digest

    def error_reason(self) -> str:
        """Return ``retire_own_recipe``."""
        return "retire_own_recipe"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the digest."""
        return {"digest": self.digest}


class RetireOpenRecipe(ClassifiedError):
    """Refuse retiring a recipe with a request that has not ended.

    Attributes:
        digest: The recipe's digest.
        request_ids: Its running or orphaned requests.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"

    def __init__(self, digest: str, request_ids: tuple[str, ...]) -> None:
        listed = ", ".join(request_ids)
        super().__init__(
            f"recipe {digest!r} has request(s) {listed} that have not ended, and "
            f"they may still write its runs. Let them finish with 'mosaic "
            f"pipeline run --request <id>', or stop each with 'mosaic cancel "
            f"--request <id>', then retire the recipe."
        )
        self.digest: str = digest
        self.request_ids: tuple[str, ...] = request_ids

    def error_reason(self) -> str:
        """Return ``retire_open_recipe``."""
        return "retire_open_recipe"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the digest and the requests that have not ended."""
        return {"digest": self.digest, "request_ids": list(self.request_ids)}


def validate_retirement(
    base_dir: Path | str,
    own_digest: str,
    retire: Iterable[str],
    *,
    cleanup: bool,
) -> tuple[str, ...]:
    """Check the recipes a submission asks to retire, and name them canonically.

    Each digest is read as the digest of a saved recipe, in any canonical form the
    dataset holds, and returned as what that recipe digests to now, so a request
    saved under an earlier form still matches.

    Args:
        base_dir: The dataset root.
        own_digest: The digest of the recipe being planned or submitted, in the
            current canonical form.
        retire: The digests given.
        cleanup: Whether the plan or submission asks for a cleanup.

    Returns:
        The recipes' current digests, sorted and without repeats; empty when
        *retire* is.

    Raises:
        RetireNeedsCleanup: *retire* is given without a cleanup.
        RetireUnknownRecipe: A digest is not that of a saved recipe that reads.
        RetireOwnRecipe: A digest is *own_digest*.
        RetireOpenRecipe: A recipe has a running or orphaned request.
    """
    given = list(retire)
    if not given:
        return ()
    if not cleanup:
        raise RetireNeedsCleanup()
    retired: set[str] = set()
    for digest in given:
        current = _saved_digest(base_dir, digest)
        if current == own_digest:
            raise RetireOwnRecipe(current)
        retired.add(current)
    for digest in sorted(retired):
        unended = tuple(
            summary.request_id
            for summary in requests_matching(base_dir, digest, Scope())
        )
        if unended:
            raise RetireOpenRecipe(digest, unended)
    return tuple(sorted(retired))


def _saved_digest(base_dir: Path | str, digest: str) -> str:
    """What the saved recipe named *digest* digests to now.

    *digest* may name the file as it was saved or the recipe in the current
    canonical form, which ``mosaic pipeline show`` and the plan's ``retirable``
    list print (:func:`_saved_recipe`).
    """
    if not _DIGEST.fullmatch(digest):
        raise RetireUnknownRecipe(
            digest, f"a recipe digest is {DIGEST_LENGTH} lowercase hex characters"
        )
    path = recipe_path(base_dir, digest)
    if path.is_file():
        try:
            return recipe_digest(load_recipe(path))
        except (OSError, ValueError) as error:
            raise RetireUnknownRecipe(
                digest, f"its saved recipe does not read: {error}"
            ) from error
    if _saved_recipe(base_dir, digest) is None:
        raise RetireUnknownRecipe(digest, "this dataset has no saved recipe with it")
    return digest


def _saved_recipe(base_dir: Path | str, digest: str) -> Recipe | None:
    """The saved recipe that *digest* names in either canonical form, or ``None``.

    A recipe saved by an earlier mosaic stays in the file named for the form it
    was saved in, and nothing saves it again under its current digest. So a
    digest with no file of its own is looked for among the recipes that read.
    """
    path = recipe_path(base_dir, digest)
    try:
        if path.is_file():
            return load_recipe(path)
    except (OSError, ValueError):
        return None
    root = pipelines_root(base_dir)
    for candidate in sorted(root.glob("*.json")) if root.is_dir() else ():
        if not _DIGEST.fullmatch(candidate.stem):
            continue
        try:
            recipe = load_recipe(candidate)
        except (OSError, ValueError):
            continue
        if recipe_digest(recipe) == digest:
            return recipe
    return None


def retired_digest_of(
    base_dir: Path | str,
    request: Request,
    retire: Collection[str],
    current: dict[str, str | None],
    *,
    before: str = "",
) -> str:
    """The retired digest that *request* belongs to, ``""`` when it belongs to none.

    A request belongs to a recipe when it names the digest, or when the recipe it
    names digests to it now (:func:`~.store.current_digest`).

    Args:
        base_dir: The dataset root.
        request: The request.
        retire: The retired digests, in the current canonical form.
        current: What each digest a request names digests to now, filled as the
            requests of one scan are read.
        before: The ``created_at`` of the request being cleaned up, or ``""``
            when it is not written yet. A request created at or after it is not
            retired: it ran the recipe again after the retirement was reviewed,
            and its runs keep the protection of any request of another recipe.
    """
    if not retire:
        return ""
    cutoff = parse_instant(before) if before else None
    if cutoff is not None:
        made = parse_instant(request.created_at)
        if made is not None and made >= cutoff:
            return ""
    if request.recipe_digest in retire:
        return request.recipe_digest
    named = current_digest(base_dir, request.recipe_digest, current)
    return named if named is not None and named in retire else ""


def written_runs(rollup: RequestRollup) -> frozenset[tuple[str, str]]:
    """The ``(storage, run_id)`` pairs that *rollup*'s attempts wrote.

    An attempt wrote a run when it named it, found some of its work to do, and ran
    in a process of its own. A full cache hit and a stalled step are recorded
    without a process.
    """
    return frozenset(
        (attempt.target, attempt.run_id)
        for attempt in rollup.steps
        if attempt.run_id and attempt.target and not attempt.cache_hit and attempt.pid
    )


def exclusively_written(
    base_dir: Path | str,
    written: Iterable[tuple[str, str]],
    executions: Container[str],
) -> frozenset[tuple[str, str]]:
    """The runs of *written* that no feature attempt outside the requests names.

    A feature run called directly (a notebook, ``Pipeline``) records an attempt
    that belongs to no request, whether it wrote the run or found it done. A run it
    names was used outside the retired recipe, and is left to the other rules.

    Args:
        base_dir: The dataset root.
        written: The ``(storage, run_id)`` pairs the retired requests wrote.
        executions: The attempts that belong to some request.
    """
    pairs = frozenset(written)
    if not pairs:
        return pairs
    outside = {
        (snapshot["target"], snapshot["run_id"])
        for snapshot in list_runs(run_log_dir(base_dir))["items"]
        if snapshot["kind"] == "feature"
        and snapshot["run_id"]
        and snapshot["execution_id"] not in executions
    }
    return pairs - outside


def retirable_recipes(
    base_dir: Path | str,
    *,
    exclude_digest: str = "",
    limit: int | None = None,
    present: Callable[[str, str], bool] | None = None,
) -> tuple[tuple[RetirableRecipeWire, ...], int]:
    """The saved recipes whose requests wrote runs, newest activity first.

    A recipe is listed under its digest in the current canonical form, which
    ``retire`` takes whichever form its requests name. Requests that do not read
    and recipes that do not read are left out.

    Args:
        base_dir: The dataset root.
        exclude_digest: A recipe to leave out, the one being planned.
        limit: The most recipes to return, ``None`` for all.
        present: Whether the feature run ``(storage, run_id)`` is still on disk.
            The listing counts only those, so a recipe whose runs were removed, or
            that only ran ops, is not offered. ``None`` counts every run written.

    Returns:
        The recipes, and how many more there were.
    """
    current: dict[str, str | None] = {}
    groups: dict[str, _Group] = {}
    for request_id in request_ids(base_dir):
        try:
            request = load_request(base_dir, request_id)
        except (RequestUnreadable, OSError):
            continue
        digest = current_digest(base_dir, request.recipe_digest, current)
        if digest is None or digest == exclude_digest:
            continue
        group = groups.setdefault(digest, _Group(named=request.recipe_digest))
        rollup = request_rollup(base_dir, request)
        group.request_ids.append(request_id)
        group.runs |= written_runs(rollup)
        group.open = group.open or (
            request_verdict_of(base_dir, rollup) in ("running", "orphaned")
        )
    found = [
        RetirableRecipeWire(
            digest=digest,
            name=_recipe_name(base_dir, group.named),
            requests=len(group.request_ids),
            request_ids=tuple(group.request_ids[:RETIRABLE_REQUEST_IDS]),
            runs=len(kept),
            open=group.open,
        )
        for digest, group in groups.items()
        if (kept := _present(group.runs, present))
    ]
    shown = found if limit is None else found[:limit]
    return tuple(shown), len(found) - len(shown)


def retired_recipe_names(
    base_dir: Path | str, digests: Iterable[str]
) -> dict[str, str]:
    """Each digest's recipe ``name``, for a sentence that names the recipes.

    A recipe that does not read has the name ``""``. A name is cut to
    :data:`NAME_LIMIT` characters.
    """
    return {digest: _recipe_name(base_dir, digest) for digest in digests}


def _present(
    runs: Collection[tuple[str, str]],
    present: Callable[[str, str], bool] | None,
) -> set[tuple[str, str]]:
    """The runs of *runs* that *present* finds, all of them without it."""
    if present is None:
        return set(runs)
    return {pair for pair in runs if present(*pair)}


@dataclass(slots=True)
class _Group:
    """The requests of one recipe, as :func:`retirable_recipes` folds them."""

    named: str
    request_ids: list[str] = field(default_factory=list[str])
    runs: set[tuple[str, str]] = field(default_factory=set[tuple[str, str]])
    open: bool = False


def _recipe_name(base_dir: Path | str, digest: str) -> str:
    """The ``name`` of the recipe *digest* names, cut to :data:`NAME_LIMIT`.

    ``""`` when no saved recipe that reads has the digest.
    """
    recipe = _saved_recipe(base_dir, digest)
    return recipe.name[:NAME_LIMIT] if recipe is not None else ""
