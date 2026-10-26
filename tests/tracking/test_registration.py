"""``register_ops`` imports every module that registers something at import."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mosaic.tracking.registration import REGISTERING_MODULES
from tests.helpers import REPO_ROOT, run_fresh

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


@pytest.mark.parametrize("root_key", ["frame_predictions", "selections"])
def test_register_ops_registers_the_index_of_a_derived_root(root_key: str) -> None:
    """A fresh interpreter, because an earlier test may have imported the index."""
    code = (
        "from mosaic.core.pipeline.dataset_indexes import reconcilable_index\n"
        "from mosaic.tracking import register_ops\n"
        "register_ops()\n"
        f"print(reconcilable_index({root_key!r}) is not None)"
    )

    assert run_fresh(code) == "True"
