"""Find the sibling modules that a program imports, to install or ship beside
it."""

from pathlib import Path

from python_imports import parse_module


def imported_names(source: bytes) -> list[str]:
    """Return the modules that `source` imports, none for a non-Python
    file."""
    parsed = parse_module(source)
    if parsed is None:
        return []
    names: list[str] = []
    for statement in parsed.statements:
        if statement.kind == "import":
            names.extend(statement.names)
        elif statement.module is not None:
            names.append(statement.module)
    return names


def library_modules(program: Path) -> list[Path]:
    """Return the modules beside `program` that it imports, directly or through
    another.

    For a program invoked through a symbolic link, return the modules beside
    the link's target, where Python looks for them.
    """
    directory = program.resolve().parent
    found: list[Path] = []
    pending = [program]
    while pending:
        for name in imported_names(pending.pop().read_bytes()):
            module = directory / f"{name}.py"
            if module.is_file() and module not in found:
                found.append(module)
                pending.append(module)
    return found
