"""The directory name a feature run stores under, and the digest that shortens it.

A storage name is ``<slug>`` or ``<slug>__from__<suffix>``, and a name too long
for one path component is abbreviated to a head and a digest. The claims store
abbreviates its own path components by the same rule. This module imports only
the standard library, because the status reads reach it through the claims
store and load neither numpy nor pandas.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Final

__all__ = [
    "NAME_MAX_BYTES",
    "SHORTENED_MARKER",
    "SHORTENED_NAME_BYTES",
    "abbreviate",
    "derive_storage_name",
    "text_digest",
]

NAME_MAX_BYTES: Final = 255
"""The longest single path component, in UTF-8 bytes, that Linux and macOS accept."""

SHORTENED_NAME_BYTES: Final = 160
"""The most UTF-8 bytes in a name that :func:`derive_storage_name` shortened.

It leaves room under :data:`NAME_MAX_BYTES` for the steps that read the name, and
keeps the name within a 255-character column that records a run's target.
"""

SHORTENED_MARKER: Final = "~"
"""The character that separates an abbreviated name from its digest.

Feature slugs and literal input names do not contain it. A storage name that
contains it was shortened or reads a shortened storage.
"""

_LINEAGE_BOUNDARIES: Final = ("+", "__from__")


def text_digest(text: str) -> str:
    """The first 10 hex characters of the SHA-1 of *text* in UTF-8.

    :func:`~mosaic.core.pipeline._utils.hash_params` applies it to canonical
    JSON, and its docstring explains the fixed width.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def abbreviate(text: str, fits: Callable[[str], bool]) -> str:
    """Return ``<head>~<digest>``, the abbreviation of *text*.

    ``<digest>`` is :func:`text_digest` of *text* encoded as JSON, which equals
    :func:`~mosaic.core.pipeline._utils.hash_params` of *text*. ``<head>`` is the
    longest prefix of *text* whose result *fits* accepts, cut where a ``+`` or
    ``__from__`` begins. The head is empty when every cut is too long.
    """
    marked = f"{SHORTENED_MARKER}{text_digest(json.dumps(text))}"
    cuts = sorted(
        {
            index
            for boundary in _LINEAGE_BOUNDARIES
            for index in range(len(text))
            if text.startswith(boundary, index)
        },
        reverse=True,
    )
    for cut in cuts:
        if fits(text[:cut] + marked):
            return text[:cut] + marked
    return marked


def derive_storage_name(feature_name: str, inputs_suffix: str | None) -> str:
    """Return the directory name under ``features/`` for a run of *feature_name*.

    A feature that reads inputs stores under ``<slug>__from__<suffix>``, for
    example ``speed__from__tracks``. The suffix is the ``+``-joined storage names
    of the inputs. A name longer than :data:`NAME_MAX_BYTES` UTF-8 bytes cannot be
    a directory. Its suffix is abbreviated (:func:`abbreviate`) to keep the whole
    name within :data:`SHORTENED_NAME_BYTES`: ``<slug>__from__<head>~<10 hex>``.
    The digest is of the full suffix, and two lineages that share a head get
    different names. Each run's ``params.json`` records the full lineage under
    ``_inputs``.

    Names that fit are returned unchanged. A step reading a shortened name
    inherits the short form, and names stop growing at a shortened step.
    """
    if inputs_suffix is None:
        return feature_name
    prefix = f"{feature_name}__from__"
    name = prefix + inputs_suffix
    if _byte_length(name) <= NAME_MAX_BYTES:
        return name
    return prefix + abbreviate(
        inputs_suffix,
        lambda suffix: _byte_length(prefix + suffix) <= SHORTENED_NAME_BYTES,
    )


def _byte_length(text: str) -> int:
    return len(text.encode("utf-8"))
