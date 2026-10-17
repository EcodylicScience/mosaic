"""What names a recipe, and what that name does *not* answer.

A digest over the JSON as written moves with key order, whitespace and omitted
defaults, so two semantically identical graphs become two pipelines and the
dataset records the same analysis twice. So the document is canonicalized first
-- defaults filled by pydantic, keys sorted, separators pinned -- and the digest
is taken over that. A whole number in a step's ``params`` is written as an
integer (:func:`~.model.canonical_params`, the number rule of RFC 8785), so
``5.0`` and ``5`` name one recipe.

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

from .model import Recipe, canonical_params

__all__ = [
    "DIGEST_LENGTH",
    "canonical_json",
    "canonical_recipe",
    "canonical_text",
    "document_digest",
    "recipe_digest",
]

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
    two. Parsing already normalizes each step's ``params``; doing it again here
    keeps the digest right for a recipe built without validation.
    """
    document: dict[str, JsonValue] = recipe.model_dump(mode="json")
    steps = document.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                params = step.get("params")
                if isinstance(params, dict):
                    step["params"] = canonical_params(params)
    return document


def canonical_text(document: JsonValue) -> str:
    """*document* as text with sorted keys and pinned separators, without spaces.

    Both arguments are load-bearing rather than tidy. ``sort_keys`` removes the
    authoring order of a mapping, which carries no meaning; the pinned separators
    remove the encoder's default spaces, which would otherwise make the digest
    depend on the standard library's formatting choices.
    """
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def canonical_json(recipe: Recipe) -> str:
    """The canonical document of *recipe* as text."""
    return canonical_text(canonical_recipe(recipe))


def document_digest(document: JsonValue) -> str:
    """Name *document* by the SHA-256 of its canonical text.

    The name is the first :data:`DIGEST_LENGTH` hex characters.
    """
    text = canonical_text(document).encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:DIGEST_LENGTH]


def recipe_digest(recipe: Recipe) -> str:
    """Name *recipe* by its content.

    Deliberately not ``hash_params``: that hashes an *identity payload* through
    ``identity_ready`` and is pinned by the golden corpus at ten characters of
    SHA-1. This hashes a whole document to name a file, so it says what it does
    rather than borrowing a function whose contract is about something else, and
    the two cannot be changed by accident together.
    """
    return document_digest(canonical_recipe(recipe))
