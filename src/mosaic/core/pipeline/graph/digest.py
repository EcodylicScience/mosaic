"""What names a recipe, and what that name does *not* answer.

A digest over the JSON as written moves with key order, whitespace and omitted
defaults, so two semantically identical graphs become two pipelines and the
dataset records the same analysis twice. So the document is canonicalized first
-- defaults filled by pydantic, keys sorted, separators pinned -- and the digest
is taken over that.

**What the digest identifies is the recipe, not what ran.** The same digest
resolves to different ``run_id``s once a mosaic upgrade bumps a feature's
``version``, because a version is a visible segment of every feature identifier.
So the request beside the recipe records the resolved ``{step -> feature
version}`` map, and *that* is what identifies what ran. Recording it is not
optional book-keeping: without it an upgrade during an open request resolves
early steps under the old versions and later ones under the new, and the
downstream steps then read as absent rather than complete.
"""

from __future__ import annotations

import hashlib
import json

from mosaic.core.json_value import JsonValue

from .model import Recipe

__all__ = ["DIGEST_LENGTH", "canonical_json", "canonical_recipe", "recipe_digest"]

DIGEST_LENGTH: int = 16
"""How many hex characters of the digest name a recipe.

Longer than the ten a ``run_id`` carries, and for a different reason. A ``run_id``
is scoped to one feature's directory in one dataset; a recipe digest names a file
that travels between datasets, repositories and people, so its namespace is
everything anyone has ever written rather than one directory's contents.
"""


def canonical_recipe(recipe: Recipe) -> dict[str, JsonValue]:
    """*recipe* as the document its digest is taken over.

    ``mode="json"`` so every value is already JSON-representable and no encoder
    hook decides anything; defaults are filled by pydantic, which is what makes
    an omitted field and an explicitly-written default one recipe rather than
    two.
    """
    return recipe.model_dump(mode="json")


def canonical_json(recipe: Recipe) -> str:
    """The canonical document as text: sorted keys, pinned separators, no spaces.

    Both arguments are load-bearing rather than tidy. ``sort_keys`` removes the
    authoring order of a mapping, which carries no meaning; the pinned separators
    remove the encoder's default spaces, which would otherwise make the digest
    depend on the standard library's formatting choices.
    """
    return json.dumps(canonical_recipe(recipe), sort_keys=True, separators=(",", ":"))


def recipe_digest(recipe: Recipe) -> str:
    """Name *recipe* by its content.

    Deliberately not ``hash_params``: that hashes an *identity payload* through
    ``identity_ready`` and is pinned by the golden corpus at ten characters of
    SHA-1. This hashes a whole document to name a file, so it says what it does
    rather than borrowing a function whose contract is about something else, and
    the two cannot be changed by accident together.
    """
    text = canonical_json(recipe).encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:DIGEST_LENGTH]
