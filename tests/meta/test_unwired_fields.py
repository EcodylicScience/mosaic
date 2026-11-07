"""Ties every parameter field declaring ``unwired=`` to a test recording it.

A field is published into its params schema, and a client drawing a form from
that schema renders a control for it. A field whose ``Declared(...)`` passes
``unwired="<why>"`` publishes ``x-mosaic-unwired`` beside it, which is how that
client knows to leave the control out. Whether to wire or withdraw such a field
is the maintainer's call, and until that call is made the field gets a
``strict`` xfail test naming the wiring it lacks, listed in ``_RECORDED_BY``.
The day someone supplies that wiring the test xpasses and the suite fails until
the marker is deleted.

A recording test asserts across each place a reader could plausibly sit, never
at one call site, and comes with an unmarked control asserting a field the same
region does read, because a ``strict`` xfail reports every failure as an xfail.

``test_every_unwired_field_is_recorded_by_a_test`` ties the declarations and the
records together in both directions. No field declares ``unwired=`` today.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mosaic

from tests.helpers import (
    TESTS_ROOT,
    runs_in_an_external_environment,
    source_tree,
)

# --- the registry: every unwired field is recorded --------------------------

_SRC = Path(mosaic.__file__ or "").resolve().parent

# Every ``unwired=`` declaration under ``src/mosaic/``, keyed by its module path
# and field name, mapped to the test that records it. A test is named by its
# file's name and its own, so the record survives a move between directories.
_RECORDED_BY: dict[tuple[str, str], str] = {}


def _declares_unwired(annotation: ast.expr) -> bool:
    """Whether *annotation* declares a ``Declared(...)`` passing ``unwired=``."""
    for node in ast.walk(annotation):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else ""
        if isinstance(function, ast.Attribute):
            name = function.attr
        if name != "Declared":
            continue
        if any(keyword.arg == "unwired" for keyword in node.keywords):
            return True
    return False


def _unwired_declarations() -> set[tuple[str, str]]:
    """Every ``unwired=`` field under the package, as (module path, field name).

    The external-environment trees are skipped. A program under one of them
    takes no import from ``mosaic`` at all, so it cannot reach ``Declared``, and
    each is where the user builds a virtualenv holding thousands of files.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(_SRC.rglob("*.py")):
        if runs_in_an_external_environment(path, _SRC):
            continue
        tree = source_tree(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue
            target = node.target
            if isinstance(target, ast.Name) and _declares_unwired(node.annotation):
                found.add((path.relative_to(_SRC).as_posix(), target.id))
    return found


def _functions_defined_in(test_file: str) -> set[str]:
    """The test function names defined by the test file named *test_file*.

    The file is found by its name anywhere under ``tests/``, and exactly one may
    carry it.
    """
    matches = sorted(TESTS_ROOT.rglob(test_file))
    assert len(matches) == 1, f"{test_file} names {len(matches)} files: {matches}"
    tree = source_tree(matches[0])
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_every_unwired_field_is_recorded_by_a_test() -> None:
    """No ``unwired=`` field exists without a test naming the wiring it lacks.

    ``core/params.py`` promises that record, and prose does not fail. This does,
    in both directions. A new declaration with no entry means a field publishing
    ``x-mosaic-unwired`` that nothing will notice being wired. An entry with no
    declaration means a field that was wired and retired, leaving a row here
    pointing at a test that no longer records anything.

    The registry also names each recording test, and the test has to exist. A
    renamed one would leave the row resolving to nothing.
    """
    assert _unwired_declarations() == set(_RECORDED_BY)

    for location in sorted(set(_RECORDED_BY.values())):
        test_file, _, test_name = location.partition("::")
        assert test_name in _functions_defined_in(test_file), location
