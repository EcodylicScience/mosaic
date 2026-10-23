"""A saved recipe named by its digest or its name, in place of the document.

A recipe that a dataset holds in ``.mosaic/pipelines/`` can be named instead of
sent again. ``{"digest": "<digest>"}`` names the saved recipe with that digest,
in the form it was saved or the current canonical form, and ``{"saved":
"<name>"}`` names the one saved recipe whose ``name`` is that name. A name that
several saved recipes share is refused with their digests, since the versions of
one recipe share a name.

Like ``store``, this module leaves pandas, cv2, av and the feature library
unloaded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from mosaic.core.narrowing import is_str_mapping

from ..error_kind import ClassifiedError, ErrorKind
from .digest import DIGEST_LENGTH, recipe_digest
from .store import load_recipe, pipelines_root, recipe_path

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

    from .model import Recipe

__all__ = [
    "RecipeNameAmbiguous",
    "RecipeNotSaved",
    "RecipeReference",
    "find_saved_recipe",
    "recipe_reference",
    "resolve_saved_recipe",
]

_DIGEST: Final = re.compile(rf"[0-9a-f]{{{DIGEST_LENGTH}}}")
"""A recipe digest: :data:`DIGEST_LENGTH` lowercase hex characters."""

type RecipeReference = tuple[Literal["digest", "saved"], str]
"""How a document names a saved recipe: by digest, or by its ``name``."""


class RecipeNotSaved(ClassifiedError):
    """No saved recipe of the dataset has the digest or name given.

    Attributes:
        reference: How the recipe was named.
    """

    error_kind: ClassVar[ErrorKind] = "not_found"

    def __init__(self, reference: RecipeReference, problem: str) -> None:
        how, value = reference
        super().__init__(
            f"no saved recipe with {how} {value!r}: {problem}. A recipe is saved "
            f"when it is submitted or patched; send the recipe itself, or save it "
            f"with patch_recipe and no ops"
        )
        self.reference: RecipeReference = reference

    def error_reason(self) -> str:
        """Return ``recipe_not_saved``."""
        return "recipe_not_saved"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return how the recipe was named."""
        how, value = self.reference
        return {how: value}


class RecipeNameAmbiguous(ClassifiedError):
    """Several saved recipes share the name given.

    Attributes:
        name: The name.
        digests: The digests of the recipes that have it, newest saved first.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(self, name: str, digests: tuple[str, ...]) -> None:
        super().__init__(
            f"{len(digests)} saved recipes are named {name!r}; name one by "
            f"digest. Newest saved first: {', '.join(digests)}"
        )
        self.name: str = name
        self.digests: tuple[str, ...] = digests

    def error_reason(self) -> str:
        """Return ``recipe_name_ambiguous``."""
        return "recipe_name_ambiguous"

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the name and the digests that share it."""
        return {"saved": self.name, "digests": list(self.digests)}


def recipe_reference(document: object) -> RecipeReference | None:
    """How *document* names a saved recipe, or ``None`` when it is a recipe itself.

    A reference is an object with one key, ``digest`` or ``saved``, holding a
    string. A recipe never has either key, so the two cannot be mistaken.
    """
    if not is_str_mapping(document) or len(document) != 1:
        return None
    for how in ("digest", "saved"):
        value = document.get(how)
        if isinstance(value, str):
            return ("digest", value) if how == "digest" else ("saved", value)
    return None


def find_saved_recipe(base_dir: Path | str, digest: str) -> Recipe | None:
    """The saved recipe that *digest* names in either canonical form, or ``None``.

    A recipe saved by an earlier mosaic stays in the file named for the form it
    was saved in, and nothing saves it again under its current digest. So a
    digest with no file of its own is looked for among the recipes that read.
    """
    if not _DIGEST.fullmatch(digest):
        return None
    path = recipe_path(base_dir, digest)
    try:
        if path.is_file():
            return load_recipe(path)
    except (OSError, ValueError):
        return None
    for candidate in _saved_files(base_dir):
        try:
            recipe = load_recipe(candidate)
        except (OSError, ValueError):
            continue
        if recipe_digest(recipe) == digest:
            return recipe
    return None


def resolve_saved_recipe(base_dir: Path | str, reference: RecipeReference) -> Recipe:
    """The saved recipe that *reference* names.

    Raises:
        RecipeNotSaved: No saved recipe has the digest or the name.
        RecipeNameAmbiguous: Several saved recipes have the name.
    """
    how, value = reference
    if how == "digest":
        if not _DIGEST.fullmatch(value):
            raise RecipeNotSaved(
                reference, f"a digest is {DIGEST_LENGTH} lowercase hex characters"
            )
        found = find_saved_recipe(base_dir, value)
        if found is None:
            raise RecipeNotSaved(reference, "this dataset has no saved recipe with it")
        return found
    if not value.strip():
        raise RecipeNotSaved(reference, "a name is not empty")
    named: dict[str, tuple[float, Recipe]] = {}
    for candidate in _saved_files(base_dir):
        try:
            recipe = load_recipe(candidate)
            saved_at = candidate.stat().st_mtime
        except (OSError, ValueError):
            continue
        if recipe.name != value:
            continue
        digest = recipe_digest(recipe)
        if digest not in named or named[digest][0] < saved_at:
            named[digest] = (saved_at, recipe)
    if not named:
        raise RecipeNotSaved(reference, "this dataset has no saved recipe with it")
    if len(named) > 1:
        newest = sorted(named, key=lambda digest: (-named[digest][0], digest))
        raise RecipeNameAmbiguous(value, tuple(newest))
    return next(iter(named.values()))[1]


def _saved_files(base_dir: Path | str) -> list[Path]:
    """The saved recipe files of the dataset, sorted by name."""
    root = pipelines_root(base_dir)
    if not root.is_dir():
        return []
    return [
        path for path in sorted(root.glob("*.json")) if _DIGEST.fullmatch(path.stem)
    ]
