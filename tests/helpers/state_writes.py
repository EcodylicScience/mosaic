"""What a feature's ``save_state`` writes, and in what order.

``atomic_write`` publishes each file with one ``os.replace``, so the replaces
under a run root are the order in which its files appear. A fit that dies at one
of them is a kill at that point, and the files that were already replaced stay.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

__all__ = ["load_results_after_each_failed_write", "record_published_files"]


def record_published_files(
    monkeypatch: pytest.MonkeyPatch, root: Path, *, fail_at: int | None = None
) -> list[str]:
    """Record the files published under *root*, and optionally fail one publish.

    Args:
        monkeypatch: The test's patcher, which restores ``os.replace``.
        root: The directory whose publishes are recorded.
        fail_at: The position of the publish that raises ``OSError``. ``None``
            lets every publish through.

    Returns:
        A list that gains the name of each file as it is published.
    """
    published: list[str] = []
    replace = os.replace

    def recording(source: str | Path, destination: str | Path) -> None:
        target = Path(destination)
        if root in target.parents:
            if fail_at is not None and len(published) == fail_at:
                raise OSError(f"the write of {target.name} failed")
            published.append(target.name)
        replace(source, destination)

    monkeypatch.setattr(os, "replace", recording)
    return published


def load_results_after_each_failed_write(
    monkeypatch: pytest.MonkeyPatch,
    parent: Path,
    save: Callable[[Path], object],
    load: Callable[[Path], bool],
) -> tuple[list[str], dict[str, bool]]:
    """Save once, then save again and fail at each publish in turn.

    Args:
        monkeypatch: The test's patcher.
        parent: A directory that holds one run root per save.
        save: Runs ``save_state`` of the fitted feature into the given root.
        load: Runs ``load_state`` of a fresh feature on the given root and
            returns whether it found a stored state.

    Returns:
        The files of the complete save, in publish order, and for each file
        whose publish failed whether ``load`` then found a state.
    """
    published = record_published_files(monkeypatch, parent / "complete")
    save(parent / "complete")
    loaded: dict[str, bool] = {}
    for position, name in enumerate(list(published)):
        root = parent / f"failed-at-{position}"
        record_published_files(monkeypatch, root, fail_at=position)
        with pytest.raises(OSError, match=name):
            save(root)
        loaded[name] = load(root)
    return list(published), loaded
