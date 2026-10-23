"""The agent tools read a group's role and never write it.

``splits`` marks the groups a model may not be fit on. An agent that could set a
role could unseal what the role protects, so only the operator's ``mosaic splits``
verbs write the manifest key. The check reads each adapter module's syntax tree and
refuses what writes the key however it is spelled: an attribute or item store, the
two writers on ``Dataset``, a ``model_copy`` that updates it, a ``setattr``, and any
call that writes the manifest at all (``mutate_manifest``, ``write_manifest``,
``save``). A reader such as ``plan.splits`` passes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mosaic.agent as agent_package

KEY = "splits"
WRITER_CALLS = frozenset(
    {"set_splits", "remove_splits", "mutate_manifest", "write_manifest", "save"}
)
# A call named here is a reader of the manifest in the adapter, not a writer.
ALLOWED_CALLS: frozenset[str] = frozenset()


def _is_key(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == KEY


def _stores(target: ast.expr) -> list[ast.expr]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return [leaf for element in target.elts for leaf in _stores(element)]
    if isinstance(target, ast.Starred):
        return _stores(target.value)
    return [target]


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def _writes_in_call(node: ast.Call) -> list[str]:
    name = _call_name(node)
    found: list[str] = []
    if name in WRITER_CALLS and name not in ALLOWED_CALLS:
        found.append(f"calls {name}")
    if name == "setattr" and len(node.args) >= 2 and _is_key(node.args[1]):
        found.append("setattr of splits")
    if name == "model_copy":
        updates = [node.args[0]] if node.args else []
        updates += [
            keyword.value for keyword in node.keywords if keyword.arg == "update"
        ]
        for update in updates:
            if not isinstance(update, ast.Dict):
                found.append("model_copy with an update the check cannot read")
            elif any(_is_key(key) for key in update.keys):
                found.append("model_copy updating splits")
    return found


def _store_targets(node: ast.stmt) -> list[ast.expr]:
    if isinstance(node, (ast.Assign, ast.Delete)):
        return [leaf for target in node.targets for leaf in _stores(target)]
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return [node.target]
    return []


def splits_writes(source: str) -> list[str]:
    """What in *source* writes the manifest's ``splits``, as ``line: what``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.stmt):
            for target in _store_targets(node):
                if isinstance(target, ast.Attribute) and target.attr == KEY:
                    found.append((node.lineno, "stores an attribute named splits"))
                if isinstance(target, ast.Subscript) and _is_key(target.slice):
                    found.append((node.lineno, "stores the item 'splits'"))
        elif isinstance(node, ast.Call):
            found.extend((node.lineno, what) for what in _writes_in_call(node))
    return [f"{line}: {what}" for line, what in sorted(found)]


def test_no_agent_module_writes_splits() -> None:
    root = Path(agent_package.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{write}"
        for path in sorted(root.rglob("*.py"))
        for write in splits_writes(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_check_refuses_each_way_to_write_the_key() -> None:
    snippets = {
        "attribute store": "m.splits = {}",
        "item store": "m['splits'] = {}",
        "unpacking store": "a, m.splits = 1, {}",
        "delete": "del m.splits",
        "augmented store": "m.splits |= {}",
        "set_splits": "ds.set_splits('test', ['a'])",
        "remove_splits": "ds.remove_splits(['a'])",
        "model_copy update": "m.model_copy(update={'splits': {}})",
        "model_copy positional": "m.model_copy({'splits': {}})",
        "model_copy opaque": "m.model_copy(update=changes)",
        "setattr": "setattr(m, 'splits', {})",
        "mutate_manifest": "ds.mutate_manifest(fn)",
        "write_manifest": "write_manifest(path, m)",
        "save": "ds.save()",
    }
    for label, source in snippets.items():
        assert splits_writes(source), label


def test_the_check_passes_a_read() -> None:
    reads = "roles = plan.splits\nrole = dataset.manifest.splits.get(g)\nx = {'a': 1}"
    assert splits_writes(reads) == []
