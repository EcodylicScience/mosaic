"""Reads a module's source as a tree, to assert what a code path reads and calls.

A test recording an unwired parameter field asserts an absence, and an absence
has to be looked for everywhere a reader could sit. These functions parse that
region and report the field names it reads. :func:`names_called_by` answers the
neighboring question about calls, and a negative assertion about a call belongs
beside the positive one: a body that calls neither of two functions passes an
"it does not call the wrong one" test without doing the work. :func:`imports_in`
and :func:`imports_of` answer the question about imports, for the guards over
which modules a file imports. :func:`imports_reaching` resolves each import to
the modules it reaches, and also counts ``import_module`` and ``__import__``
calls and literal module names, for the layering guards.

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
import re
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path
from types import ModuleType
from typing import Final, Literal


def source_tree(path: Path) -> ast.Module:
    """Parses the file at *path*."""
    return ast.parse(path.read_text(encoding="utf-8"))


ImportScope = Literal["module", "anywhere"]
"""Where :func:`imports_in` reads import statements.

``"module"`` reads the statements of the module body. ``"anywhere"`` reads every
scope, function bodies and ``TYPE_CHECKING`` blocks included.
"""


@dataclass(frozen=True, slots=True)
class ImportedName:
    """One name that an import statement imports.

    ``import a.b as c`` imports module ``a.b`` as ``c``, and ``from a import b as
    c`` imports ``b`` from module ``a`` as ``c``.

    Attributes:
        line: The line of the statement.
        module: The module imported, or imported from. A relative module is
            spelled with its leading dots.
        name: The name imported from :attr:`module`, or ``None`` for an
            ``import`` statement.
        asname: The name it is bound to, or ``None`` when the statement gives
            none.
    """

    line: int
    module: str
    name: str | None
    asname: str | None

    def in_package(self, package: str) -> bool:
        """Whether :attr:`module` is *package* or a module inside it.

        The dot counts. ``ultralytics_track`` is a mosaic module apart from
        ``ultralytics``, and ``mosaic_media`` is a distribution apart from
        ``mosaic``.
        """
        return self.module == package or self.module.startswith(f"{package}.")


def imports_in(
    tree: ast.Module, *, scope: ImportScope, relative: bool
) -> list[ImportedName]:
    """Returns every name that *tree* imports, in the order of lines.

    Each name of a statement is reported, so ``from a import b, c`` is two. A
    caller that also reads other nodes of the source parses it once and passes
    the tree here.

    Args:
        tree: The parsed source.
        scope: Where to read import statements.
        relative: Whether to report a relative import. ``False`` passes over
            one, since it names a module inside the importing package.
    """
    statements: Iterable[ast.AST] = tree.body if scope == "module" else ast.walk(tree)
    found: list[ImportedName] = []
    for node in statements:
        if isinstance(node, ast.Import):
            found.extend(
                ImportedName(node.lineno, alias.name, None, alias.asname)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and (relative or node.level == 0):
            module = "." * node.level + (node.module or "")
            found.extend(
                ImportedName(node.lineno, module, alias.name, alias.asname)
                for alias in node.names
            )
    return sorted(found, key=lambda imported: imported.line)


def imports_of(path: Path, *, scope: ImportScope, relative: bool) -> list[ImportedName]:
    """Returns :func:`imports_in` of the file at *path*."""
    return imports_in(source_tree(path), scope=scope, relative=relative)


DYNAMIC_IMPORTS: Final = frozenset({"import_module", "__import__"})
"""The functions whose call with a literal module name counts as an import."""

_MODULE_LITERAL: Final = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*(:\w+)?")
"""A string naming a module, with the ``:<attribute>`` a ``lazy_export`` source
may end in."""


@dataclass(frozen=True, slots=True)
class ReachedImport:
    """One name a file imports, by a statement, a call or a literal module name.

    Attributes:
        line: The line of the statement, the call or the literal.
        scope: The functions, classes and ``if`` branches it sits in, outermost
            first.
        statement: The import as written, narrowed to this one name.
        reaches: The absolute modules it may import. ``from a import b`` reaches
            ``a`` and ``a.b``, since ``b`` may be a submodule.
    """

    line: int
    scope: tuple[str, ...]
    statement: str
    reaches: tuple[str, ...]

    def into(self, package: str) -> str | None:
        """The first module of :attr:`reaches` inside *package*, or ``None``.

        The dot counts, so ``mosaic.trackings`` is not inside ``mosaic.tracking``.
        """
        return next(
            (
                module
                for module in self.reaches
                if module == package or module.startswith(f"{package}.")
            ),
            None,
        )


def _scoped(
    nodes: Iterable[ast.AST], scope: tuple[str, ...]
) -> Iterator[tuple[ast.AST, tuple[str, ...]]]:
    """Yields every node of *nodes* and beneath them, with the scope it sits in.

    A function or a class adds its name. An ``if`` adds its test to its body and
    the negated test to its ``else`` branch. A node comes before its children.
    """
    for node in nodes:
        yield node, scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from _scoped(ast.iter_child_nodes(node), (*scope, node.name))
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test)
            yield from _scoped((node.test,), scope)
            yield from _scoped(node.body, (*scope, f"if {test}"))
            yield from _scoped(node.orelse, (*scope, f"if not {test}"))
        else:
            yield from _scoped(ast.iter_child_nodes(node), scope)


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """The argument *call* passes at *position* or as *keyword*."""
    if len(call.args) > position:
        return call.args[position]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def _called(call: ast.Call, package: str) -> tuple[str, ...]:
    """The modules a call of ``import_module`` or ``__import__`` reaches.

    Empty unless the call names its module with a literal. ``import_module``
    resolves a relative name against its ``package`` argument, given as a
    literal or as ``__package__``. ``__import__`` also reaches each literal name
    of its ``fromlist``.
    """
    called = _trailing_name(call.func)
    if called not in DYNAMIC_IMPORTS:
        return ()
    name = _constant_string(_argument(call, 0, "name"))
    if not name:
        return ()
    if called == "import_module":
        if not name.startswith("."):
            return (name,)
        anchor = _argument(call, 1, "package")
        if isinstance(anchor, ast.Name) and anchor.id == "__package__":
            return (resolve_name(name, package),)
        anchor_name = _constant_string(anchor)
        return (resolve_name(name, anchor_name),) if anchor_name else ()
    fromlist = _argument(call, 3, "fromlist")
    if not isinstance(fromlist, (ast.List, ast.Tuple)):
        return (name,)
    submodules = [_constant_string(element) for element in fromlist.elts]
    return (name, *(f"{name}.{sub}" for sub in submodules if sub))


def imports_reaching(
    tree: ast.Module, package: str, *, literal_packages: Collection[str] = ()
) -> list[ReachedImport]:
    """Returns every name that *tree* imports, resolved, in the order of lines.

    Reads every scope, as :func:`imports_in` does with ``scope="anywhere"``,
    and reports each name of a statement, so ``from a import b, c`` is two. A
    relative import resolves against *package*, the package of the scanned
    file. A call of ``import_module`` or ``__import__`` that names its module
    with a literal counts as the import it performs. A module name built at run
    time is out of reach of any static scan.

    A string whose whole value names a module inside one of *literal_packages*
    counts too, with any ``:<attribute>`` suffix dropped. That is the spelling
    of a ``lazy_export`` table and of a tuple of modules imported in a loop,
    both of which name their modules as literals.

    Args:
        tree: The parsed source.
        package: The absolute package of the scanned file: its directory for an
            ``__init__.py`` and for a module alike.
        literal_packages: The packages a string literal is matched against.
    """
    found: list[ReachedImport] = []
    consumed: set[int] = set()
    for node, scope in _scoped(tree.body, ()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                statement = f"import {alias.name}" + (
                    f" as {alias.asname}" if alias.asname else ""
                )
                found.append(
                    ReachedImport(node.lineno, scope, statement, (alias.name,))
                )
        elif isinstance(node, ast.ImportFrom):
            written = "." * node.level + (node.module or "")
            module = resolve_name(written, package)
            for alias in node.names:
                statement = f"from {written} import {alias.name}" + (
                    f" as {alias.asname}" if alias.asname else ""
                )
                reaches = (
                    (module,)
                    if alias.name == "*"
                    else (module, f"{module}.{alias.name}")
                )
                found.append(ReachedImport(node.lineno, scope, statement, reaches))
        elif isinstance(node, ast.Call) and (reaches := _called(node, package)):
            consumed.update(id(child) for child in ast.walk(node))
            found.append(ReachedImport(node.lineno, scope, ast.unparse(node), reaches))
        elif (
            literal_packages
            and id(node) not in consumed
            and isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _MODULE_LITERAL.fullmatch(node.value)
        ):
            module = node.value.partition(":")[0]
            if any(
                module == named or module.startswith(f"{named}.")
                for named in literal_packages
            ):
                found.append(
                    ReachedImport(node.lineno, scope, repr(node.value), (module,))
                )
    return sorted(found, key=lambda imported: imported.line)


def module_tree(module: ModuleType) -> ast.Module:
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


def _constant_string(node: ast.expr | None) -> str:
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
