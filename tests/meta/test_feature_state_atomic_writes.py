"""Every write in a feature's ``save_state`` is atomic.

``load_state`` decides that a stored state exists by testing for one file, so
``save_state`` writes that file last and writes every file through a helper that
renames a finished temporary file into place. A reader that is not under the run
claim, the first ``load_state`` of a run or a process worker, then never sees a
file that is half written or a state that is missing its last file. The check
reads each feature module's syntax tree and flags a write inside a ``save_state``
that is not an argument of ``atomic_write``, ``atomic_savez``, ``write_bundle`` or
``write_parquet_atomic``. It cannot see inside a callee, and it cannot see the
order of the writes. The behavior tests of the fitters cover the order.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mosaic.behavior.feature_library as feature_library

WRITER_CALLS = frozenset(
    {
        "dump",
        "save",
        "savez",
        "savez_compressed",
        "write_text",
        "write_bytes",
        "to_csv",
        "to_json",
        "to_parquet",
    }
)
ATOMIC_CALLS = frozenset(
    {"atomic_write", "atomic_savez", "write_bundle", "write_parquet_atomic"}
)
WRITE_MODE_LETTERS = frozenset("wxa")
MODE_LETTERS = frozenset("rwxabt+")


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def _opens_for_writing(node: ast.Call) -> bool:
    modes = [
        argument.value
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    ]
    return any(
        mode and set(mode) <= MODE_LETTERS and set(mode) & WRITE_MODE_LETTERS
        for mode in modes
    )


def _write_in_call(node: ast.Call) -> str | None:
    name = _call_name(node)
    if name in WRITER_CALLS:
        return f"calls {name}"
    if name == "open" and _opens_for_writing(node):
        return "opens a file for writing"
    return None


def _unwrapped_writes(node: ast.AST, *, wrapped: bool) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    if isinstance(node, ast.Call):
        what = None if wrapped else _write_in_call(node)
        if what is not None:
            found.append((node.lineno, what))
        arguments_wrapped = wrapped or _call_name(node) in ATOMIC_CALLS
        found.extend(_unwrapped_writes(node.func, wrapped=wrapped))
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            found.extend(_unwrapped_writes(argument, wrapped=arguments_wrapped))
        return found
    for child in ast.iter_child_nodes(node):
        found.extend(_unwrapped_writes(child, wrapped=wrapped))
    return found


def unwrapped_state_writes(source: str) -> list[str]:
    """What in a ``save_state`` of *source* writes outside an atomic helper."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "save_state":
                found.extend(_unwrapped_writes(node, wrapped=False))
    return [f"{line}: {what}" for line, what in sorted(set(found))]


def _in_save_state(body: str) -> str:
    return f"class F:\n    def save_state(self, run_root):\n        {body}\n"


def test_no_save_state_writes_outside_an_atomic_helper() -> None:
    root = Path(feature_library.__file__).parent
    offenders = [
        f"{path.name}:{write}"
        for path in sorted(root.glob("*.py"))
        for write in unwrapped_state_writes(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_check_refuses_each_way_to_write_outside_a_helper() -> None:
    snippets = {
        "joblib dump": "joblib.dump(bundle, path)",
        "json dump": "json.dump(payload, handle)",
        "torch save": "torch.save(state, path)",
        "np savez": "np.savez(path, a=a)",
        "np savez compressed": "np.savez_compressed(path, a=a)",
        "write_text": "path.write_text('x')",
        "write_bytes": "path.write_bytes(b'x')",
        "to_csv": "frame.to_csv(path)",
        "to_json": "frame.to_json(path)",
        "to_parquet": "frame.to_parquet(path)",
        "open for writing": "open(path, 'w')",
        "open for appending": "open(path, mode='a')",
        "path open for writing": "path.open('wb')",
        "write after a helper": "atomic_write(a, lambda t: 1); joblib.dump(bundle, b)",
    }
    for label, body in snippets.items():
        assert unwrapped_state_writes(_in_save_state(body)), label


def test_the_check_passes_a_write_inside_a_helper() -> None:
    bodies = [
        "atomic_write(path, lambda t: joblib.dump(bundle, t))",
        "atomic_savez(path, a=a)",
        "write_bundle(path, bundle)",
        "write_parquet_atomic(frame, path)",
        "atomic_write(path, lambda t: json.dump(payload, t.open('w')))",
        "atomic_write(path, lambda t: frame.to_csv(t, index=False))",
        "payload = json.loads(path.read_text())",
        "handle = open(path, 'r')",
        "handle = open(path)",
    ]
    for body in bodies:
        assert unwrapped_state_writes(_in_save_state(body)) == [], body


def test_the_check_reads_only_save_state() -> None:
    source = "def fit(self):\n    joblib.dump(1, p)\n"
    assert unwrapped_state_writes(source) == []
