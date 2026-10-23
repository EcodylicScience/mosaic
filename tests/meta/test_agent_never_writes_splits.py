"""The agent tools read a group's role and never write it.

``splits`` marks the groups a model may not be fit on. An agent that could set a
role could unseal what the role protects, so only the operator's ``mosaic splits``
verbs write the manifest key. The check is by name: the two writers on ``Dataset``
and the manifest key itself must not appear in the adapter.
"""

from __future__ import annotations

from pathlib import Path

import mosaic.agent as agent_package

WRITERS = ("set_splits", "remove_splits", '"splits"]', "splits=")


def test_no_agent_module_writes_splits() -> None:
    root = Path(agent_package.__file__).parent
    offenders = [
        f"{path.relative_to(root)}: {writer}"
        for path in sorted(root.rglob("*.py"))
        for writer in WRITERS
        if writer in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
