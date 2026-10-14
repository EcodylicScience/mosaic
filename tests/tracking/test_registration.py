"""``register_ops`` imports every module that registers something at import."""

from __future__ import annotations

import ast
from pathlib import Path

from mosaic.tracking.registration import REGISTERING_MODULES
from tests.helpers import REPO_ROOT

PACKAGE = REPO_ROOT / "src" / "mosaic" / "tracking"


def _callee_name(node: ast.expr) -> str | None:
    """The name a call or decorator refers to: ``f`` for ``f`` and ``mod.f``."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_registering_call(node: ast.expr | None) -> bool:
    """Is *node* a call to a ``register_*`` function?"""
    if not isinstance(node, ast.Call):
        return False
    name = _callee_name(node)
    return name is not None and name.startswith("register_")


def _registers_at_import(source: Path) -> bool:
    """Does *source* call a ``register_*`` function or apply ``@register_op`` at module scope?

    A call counts as a statement of its own or as the value of an assignment, and a
    callee counts by its bare or attribute name.
    """
    for node in ast.parse(source.read_text()).body:
        if isinstance(
            node, ast.Expr | ast.Assign | ast.AnnAssign
        ) and _is_registering_call(node.value):
            return True
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and any(
            _callee_name(decorator) == "register_op"
            for decorator in node.decorator_list
        ):
            return True
    return False


def test_the_listed_modules_are_the_ones_that_register_at_import() -> None:
    """Read from the source: a module left off the list registers nothing until
    something else happens to import it."""
    registering = {
        ".".join(source.relative_to(PACKAGE.parent.parent).with_suffix("").parts)
        for source in sorted(PACKAGE.rglob("*.py"))
        if "external" not in source.relative_to(PACKAGE).parts
        and _registers_at_import(source)
    }

    assert registering == set(REGISTERING_MODULES)
