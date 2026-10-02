"""Telling "this mount will not lock" from "someone else holds it".

The two look identical from a poll loop -- both surface as OSError from a
non-blocking lock attempt -- so a loop that treats refusal as contention spins
the whole 60s timeout on every index write and then reports a competing writer
that does not exist, sending a reader to look for a wedged process.

What must never happen is the opposite mistake: concluding "unsupported" and
writing unlocked, which is the lost update the whole module exists to prevent.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from mosaic.core.pipeline import index_lock as lock_module
from mosaic.core.pipeline.index_lock import (
    IndexLockTimeout,
    IndexLockUnsupported,
    index_lock,
    probe_lock_support,
)


def _refusing(code: int):
    def _refuse(fd: int) -> None:
        raise OSError(code, "refused")

    return _refuse


def test_an_unlockable_mount_fails_fast_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not after the timeout, and not blaming another writer."""
    monkeypatch.setattr(lock_module, "_lock_exclusive_nb", _refusing(errno.ENOLCK))

    with pytest.raises(IndexLockUnsupported, match="does not support locking"):
        with index_lock(tmp_path / "index.csv", timeout=30.0):
            pass


@pytest.mark.parametrize(
    "code", [errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL]
)
def test_every_unsupported_errno_is_recognised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    monkeypatch.setattr(lock_module, "_lock_exclusive_nb", _refusing(code))

    with pytest.raises(IndexLockUnsupported):
        with index_lock(tmp_path / "index.csv", timeout=0.05):
            pass


def test_contention_is_still_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EAGAIN means someone holds it, and waiting is exactly the right response.

    Misreading this as "unsupported" would turn every busy moment into a refusal
    to write at all.
    """
    monkeypatch.setattr(lock_module, "_lock_exclusive_nb", _refusing(errno.EAGAIN))

    with pytest.raises(IndexLockTimeout):
        with index_lock(tmp_path / "index.csv", timeout=0.05):
            pass


def test_an_unfamiliar_errno_keeps_waiting_rather_than_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conservative direction. Guessing "unsupported" from an error nobody
    classified, and then proceeding, is how this check would cause the very
    corruption it was added to detect."""
    monkeypatch.setattr(lock_module, "_lock_exclusive_nb", _refusing(errno.EIO))

    with pytest.raises(IndexLockTimeout):
        with index_lock(tmp_path / "index.csv", timeout=0.05):
            pass


def test_a_working_mount_probes_clean(tmp_path: Path) -> None:
    probe_lock_support(tmp_path)


def test_the_probe_refuses_an_unlockable_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For a caller that wants to know at load rather than minutes into a run."""
    monkeypatch.setattr(lock_module, "_lock_exclusive_nb", _refusing(errno.ENOLCK))

    with pytest.raises(IndexLockUnsupported):
        probe_lock_support(tmp_path)


def test_the_probe_leaves_nothing_behind(tmp_path: Path) -> None:
    """Safe to remove where a live index lock is not: nothing else knows the name."""
    before = set(tmp_path.iterdir())

    probe_lock_support(tmp_path)

    assert set(tmp_path.iterdir()) == before


def test_locking_still_works_normally(tmp_path: Path) -> None:
    """The negative half: none of this changes a working mount."""
    index = tmp_path / "index.csv"
    with index_lock(index):
        with index_lock(index):  # re-entrant within one thread
            pass
    assert index.exists()
