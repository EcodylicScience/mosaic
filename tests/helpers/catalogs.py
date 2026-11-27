"""Declaration catalogs that a graph test validates or plans against.

The real catalog, changed only where a test names a step type. A test about a
feature that does not declare a column contract strips the contract of a declared
feature. Every registered feature is to declare one, and a test written against an
undeclared feature stops testing anything once that feature is declared.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace

from mosaic.core.pipeline.graph import DeclarationCatalog, declaration_catalog
from mosaic.core.pipeline.types import Feature

RESULT_READING_FEATURES: frozenset[str] = frozenset(
    {
        "arhmm",
        "global-kmeans",
        "global-ward",
        "global-tsne",
        "global-scaler",
        "extract-templates",
        "extract-labeled-templates",
        "temporal-stack",
    }
)
"""Eight of the steps that read only results and compute in every space.

The clustering, embedding, template, scaler, AR-HMM and stacking steps. The label and
classifier steps that read only results are not listed.
"""


def catalog_without_contract(*names: str) -> DeclarationCatalog:
    """The registries' catalog, with no ``requires`` or ``provides`` for *names*.

    Raises:
        KeyError: A name that the catalog does not declare.
    """
    catalog = declaration_catalog()
    missing = [name for name in names if name not in catalog]
    if missing:
        raise KeyError(f"the catalog declares no {missing}")
    return DeclarationCatalog(
        entries={
            name: replace(declared, requires=None, provides=None)
            if name in names
            else declared
            for name, declared in catalog.entries.items()
        }
    )


@contextmanager
def registered_features(*classes: type[Feature]) -> Generator[None]:
    """Register *classes* in ``FEATURES`` under their ``__name__`` for the block.

    The declaration catalog is memoized, so a feature registered after anything
    asked for one is invisible to validation and planning. The catalog is cleared
    on the way in and on the way out, so a test neither reads a catalog built
    without the classes nor leaves one built with them. The classes are removed
    in a ``finally``.
    """
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.graph.resolve import (
        builtin_declaration_catalog,
        builtin_editor_catalog,
    )

    for registered in classes:
        FEATURES[registered.__name__] = registered
    builtin_declaration_catalog.cache_clear()
    builtin_editor_catalog.cache_clear()
    try:
        yield
    finally:
        for registered in classes:
            del FEATURES[registered.__name__]
        builtin_declaration_catalog.cache_clear()
        builtin_editor_catalog.cache_clear()
