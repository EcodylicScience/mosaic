"""Parse the import statements of a Python source."""

import ast
import warnings
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ImportStatement:
    """Record one `import` or `from` statement.

    `module` is None for a plain `import` and for `from . import x`. `names`
    holds the imported names, not their `as` names, and is `('*',)` for a star
    import.
    """

    kind: Literal["import", "from"]
    module: str | None
    names: tuple[str, ...]
    level: int


@dataclass(frozen=True)
class ParsedModule:
    """Record a module's import statements and the strings of its module-level
    `pytest_plugins = [...]` list, empty for any other form."""

    statements: tuple[ImportStatement, ...]
    plugins: tuple[str, ...]


def plugin_names(tree: ast.Module) -> tuple[str, ...]:
    """Return the string elements of the last module-level `pytest_plugins`
    list assignment in `tree`."""
    plugins: tuple[str, ...] = ()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        if any(isinstance(target, ast.Name) and target.id == "pytest_plugins" for target in node.targets):
            plugins = tuple(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return plugins


def parse_module(source: bytes) -> ParsedModule | None:
    """Return the import statements of `source`, in `ast.walk` order, and its
    pytest plugins, or None when it does not parse, including a source nested
    too deeply for the parser.

    The parser's warnings, such as an invalid escape sequence, are dropped.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    statements: list[ImportStatement] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = tuple(alias.name for alias in node.names)
        if isinstance(node, ast.Import):
            statements.append(ImportStatement(kind="import", module=None, names=names, level=0))
        else:
            statements.append(ImportStatement(kind="from", module=node.module, names=names, level=node.level))
    return ParsedModule(statements=tuple(statements), plugins=plugin_names(tree))
