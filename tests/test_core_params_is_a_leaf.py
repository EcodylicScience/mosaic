"""``mosaic.core.params`` and ``mosaic.core.strict_model`` declare the
vocabulary every parameter model in mosaic shares, and each stays importable
without pulling in the rest of mosaic.

The check below reads each file's own ast rather than measuring an import at
run time. ``core/__init__.py`` imports ``Dataset`` eagerly, so pandas and the
whole pipeline machinery already sit in ``sys.modules`` by the time either leaf
module is reachable, whatever either module itself imports -- a runtime probe
would report the same pollution regardless of which module earned it. Walking
the full tree, not just the module body, also catches an import deferred
inside a function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mosaic.core.params as params_module
import mosaic.core.strict_model as strict_model_module


def _mosaic_imports(path: Path) -> set[str]:
    """Every mosaic module named by an import anywhere in the file.

    Walks the whole tree rather than the module body alone, so an import
    deferred inside a function counts the same as a top-level one. A relative
    import counts too, spelled with its dot count, since a module living
    inside the mosaic package reaches back into it at any level.
    """
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name == "mosaic" or alias.name.startswith("mosaic.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.add("." * node.level + (node.module or ""))
            elif node.module and (
                node.module == "mosaic" or node.module.startswith("mosaic.")
            ):
                found.add(node.module)
    return found


def test_strict_model_has_no_mosaic_imports() -> None:
    """``StrictModel`` imports pydantic and the standard library only."""
    assert _mosaic_imports(Path(strict_model_module.__file__)) == set()


def test_params_imports_only_strict_model() -> None:
    """``Params`` imports exactly one mosaic module: ``mosaic.core.strict_model``."""
    assert _mosaic_imports(Path(params_module.__file__)) == {"mosaic.core.strict_model"}
