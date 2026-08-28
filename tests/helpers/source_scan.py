"""Reads a module's source as a tree, to assert what a code path reads and calls.

A test recording an unwired parameter field asserts an absence, and an absence
has to be looked for everywhere a reader could sit. These functions parse that
region and report the field names it reads. :func:`names_called_by` answers the
neighboring question about calls, and a negative assertion about a call belongs
beside the positive one: a body that calls neither of two functions passes an
"it does not call the wrong one" test without doing the work.

A read counts only off the names that hold the object in the scanned region.
``np.load``, ``joblib.load`` and ``json.load`` are attribute reads named
``load``; counting them by name reports a labels source's ``load`` field as read
by a loader that never saw a labels source. Adding a binding to the region
therefore means adding its name -- with one exception the caller pays nothing
for: a function parameter annotated with a class in *destructured_classes* is
picked up on its own, so a helper naming its own parameter is covered.

Five spellings count as a read, because a field is wired by any of them:

- an attribute access off one of *owners*,
- ``getattr(<owner>, "name")`` with a literal name,
- a keyword pattern in a ``match`` case, for the classes named in
  *destructured_classes*,
- a constant-key subscript of a payload derived from an owner, the spelling a
  ``model_dump()`` dispatch uses,
- ``Field(discriminator="name")`` on a declaration that also names one of
  *destructured_classes*, this repository's spelling for a discriminator tag.
"""

from __future__ import annotations

import ast
from collections.abc import Collection, Iterable
from pathlib import Path
from types import ModuleType


def source_tree(path: Path) -> ast.AST:
    """Parses the file at *path*."""
    return ast.parse(path.read_text(encoding="utf-8"))


def module_tree(module: ModuleType) -> ast.AST:
    """Parses the whole of *module*, for a scan with no function to narrow to."""
    return source_tree(Path(module.__file__ or ""))


def functions_named(module: ModuleType, names: Collection[str]) -> list[ast.AST]:
    """Returns the bodies of *names* in *module*, one tree per definition.

    Every definition of a name is returned, so an overload or a nested helper
    sharing a name is scanned too. Raises when a name is absent: a renamed
    function empties the scan and leaves the assertion agreeing with itself.
    """
    found: list[ast.AST] = []
    seen: set[str] = set()
    for node in ast.walk(module_tree(module)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in names:
                found.append(node)
                seen.add(node.name)
    missing = sorted(set(names) - seen)
    assert not missing, f"{missing} renamed in {module.__name__}; this scan misses them"
    return found


def names_called_by(module: ModuleType, function_name: str) -> set[str]:
    """Returns every function name the body of *function_name* in *module* calls.

    A call through an object is collected by its final name, which is what makes
    ``ds.resolve_scope(...)`` visible as ``resolve_scope``. Read out of the
    module's source rather than off an imported function object, which reaches a
    module-private function without asking it to be exported. An aliased import
    stays out of reach of any source-level read of a body.

    Read the parsed body rather than the source text: a comment naming a
    function is not a call to it, and a substring search cannot tell them apart.
    """
    called: set[str] = set()
    for tree in functions_named(module, {function_name}):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (name := _trailing_name(node.func)):
                called.add(name)
    return called


def _dotted_name(node: ast.expr) -> str:
    """Returns ``a.b.c`` for a chain of names, or empty for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _base_name(node: ast.expr) -> str:
    """Returns the leftmost name a chain of accesses, calls and indexes starts at."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            return ""


def _trailing_name(node: ast.expr) -> str:
    """Returns the last name of a dotted expression.

    Names the class a ``match`` case destructures, and the function a call
    invokes.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _constant_string(node: ast.expr) -> str:
    """Returns the value of a string literal, or empty for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _mentions_a_class(node: ast.AST, class_names: Collection[str]) -> bool:
    """Whether *node* names one of *class_names* anywhere beneath it."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Name, ast.Attribute)):
            if _trailing_name(child) in class_names:
                return True
    return False


def _annotated_parameters(tree: ast.AST, class_names: Collection[str]) -> set[str]:
    """Returns the parameters *tree* annotates with one of *class_names*.

    A helper beside a dispatcher names its own parameter, so the caller cannot
    list every binding up front. An annotation is the declaration that the
    parameter holds the class, union members included.
    """
    owners: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args
        every = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        ]
        for argument in every:
            if argument is None or argument.annotation is None:
                continue
            if _mentions_a_class(argument.annotation, class_names):
                owners.add(argument.arg)
    return owners


def _discriminator_reads(tree: ast.AST, class_names: Collection[str]) -> set[str]:
    """Returns the tags ``Field(discriminator=...)`` names on *class_names*.

    Read off the declaration rather than off the bare call, because a
    discriminated union of unrelated models in the same module says nothing
    about this one. The declaration has to name the class as well as the tag.
    """
    read: set[str] = set()
    for node in ast.walk(tree):
        declaration = isinstance(node, (ast.Assign, ast.AnnAssign))
        annotated = isinstance(node, ast.Subscript) and _trailing_name(node.value) in {
            "Annotated"
        }
        if not declaration and not annotated:
            continue
        if not _mentions_a_class(node, class_names):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or _trailing_name(child.func) != "Field":
                continue
            for keyword in child.keywords:
                if keyword.arg == "discriminator":
                    if tag := _constant_string(keyword.value):
                        read.add(tag)
    return read


def _getattr_read(node: ast.Call, owners: Collection[str]) -> set[str]:
    """Returns the literal name a ``getattr(<owner>, "name")`` call reads."""
    if _trailing_name(node.func) != "getattr" or len(node.args) < 2:
        return set()
    if _base_name(node.args[0]) not in owners:
        return set()
    return {name} if (name := _constant_string(node.args[1])) else set()


def names_read(
    trees: Iterable[ast.AST],
    *,
    owners: Collection[str],
    destructured_classes: Collection[str],
) -> set[str]:
    """Returns the field names *trees* read off *owners*, by every spelling.

    *owners* are the names the object is bound to in the scanned region, dotted
    where it is reached through an attribute (``self.params``). Include
    ``"self"`` wherever the region declares the model, since a validator or a
    computed field reads its own fields through it. *destructured_classes* are
    the classes whose ``match`` keyword patterns and discriminator declarations
    count, and whose annotated parameters join *owners* per tree.
    """
    read: set[str] = set()
    for tree in trees:
        here = set(owners) | _annotated_parameters(tree, destructured_classes)
        read |= _discriminator_reads(tree, destructured_classes)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if _dotted_name(node.value) in here:
                    read.add(node.attr)
            elif isinstance(node, ast.Subscript):
                if _base_name(node.value) in here:
                    if key := _constant_string(node.slice):
                        read.add(key)
            elif isinstance(node, ast.MatchClass):
                if _trailing_name(node.cls) in destructured_classes:
                    read.update(node.kwd_attrs)
            elif isinstance(node, ast.Call):
                read |= _getattr_read(node, here)
    return read
