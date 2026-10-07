"""Declaration catalogs that a graph test validates or plans against.

The real catalog, changed only where a test names a step type. A test about a
feature that does not declare a column contract strips the contract of a declared
feature. Every registered feature is to declare one, and a test written against an
undeclared feature stops testing anything once that feature is declared.
"""

from __future__ import annotations

from dataclasses import replace

from mosaic.core.pipeline.graph import DeclarationCatalog, declaration_catalog


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
