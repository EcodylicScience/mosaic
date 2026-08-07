"""A claim must be taken exclusively, and released only by its holder.

``open_entry`` said it "takes" the entry and took nothing: it mkdir'd, *read* the
in-flight marker, and returned. The caller's ``claim()`` landed hundreds of lines
later inside per-phase code, so two executions could both read the directory as
free and both proceed -- and the write was ``os.replace``, last-writer-wins, with no
``O_EXCL`` anywhere in the toolkit.

Worse, the loser did not need to win the race to do damage. On the reuse-hit path no
claim was ever written, yet ``release_entry`` ran in the driver's unconditional
``finally`` and unlinked whatever marker was present -- so an execution that claimed
nothing deleted a live peer's claim, after which a third execution read the
directory as free and cleared it, ``overwrite``'s ``rmtree`` included.

Three properties pinned here: the create is exclusive, the release is
ownership-checked, and the refresh distinguishes *absent* (restore mine) from
*foreign* (leave it). That last asymmetry is not a preference -- the activity
callback legitimately restores a claim someone deleted mid-phase, which
``test_trex_run_markers`` asserts.
"""

from __future__ import annotations

from pathlib import Path

from mosaic.core.pipeline.markers import (
    clear_inflight,
    inflight_marker_path,
    new_inflight,
    read_inflight,
    refresh_inflight,
    try_create_inflight,
    write_inflight,
)

_MINE = "EXEC-MINE"
_THEIRS = "EXEC-THEIRS"


def _marker(execution_id: str, idle_seconds: float = 60.0):
    return new_inflight(
        execution_id=execution_id,
        host="h",
        pid=1,
        phase="track",
        idle_seconds=idle_seconds,
    )


def test_two_executions_racing_one_entry_leave_one_holder(tmp_path: Path) -> None:
    """The create is exclusive, so the second attempt loses rather than clobbering."""
    assert try_create_inflight(tmp_path, _marker(_MINE)) is True
    assert try_create_inflight(tmp_path, _marker(_THEIRS)) is False

    held = read_inflight(tmp_path)
    assert held is not None
    assert held.execution_id == _MINE


def test_a_release_does_not_delete_a_foreign_claim(tmp_path: Path) -> None:
    """The reuse-hit path wrote no claim, and its release deleted a peer's."""
    write_inflight(tmp_path, _marker(_THEIRS))

    clear_inflight(tmp_path, execution_id=_MINE)

    survivor = read_inflight(tmp_path)
    assert survivor is not None
    assert survivor.execution_id == _THEIRS


def test_a_release_removes_our_own_claim(tmp_path: Path) -> None:
    write_inflight(tmp_path, _marker(_MINE))

    clear_inflight(tmp_path, execution_id=_MINE)

    assert read_inflight(tmp_path) is None


def test_a_refresh_restores_an_absent_claim_but_not_a_foreign_one(
    tmp_path: Path,
) -> None:
    """Absent means ours vanished; foreign means someone else holds it now.

    The activity callback fires on every output line for the whole phase, so it is
    the one thing that must put back a claim deleted underneath a running tool --
    while never re-stamping a directory another execution has taken over.
    """
    mine = _marker(_MINE)
    write_inflight(tmp_path, mine)
    inflight_marker_path(tmp_path).unlink()

    restored = refresh_inflight(tmp_path, mine, 60.0)
    assert restored is not None
    held = read_inflight(tmp_path)
    assert held is not None and held.execution_id == _MINE

    # A peer took it over: leave theirs alone and report that we no longer hold it.
    write_inflight(tmp_path, _marker(_THEIRS))
    assert refresh_inflight(tmp_path, mine, 60.0) is None
    still_theirs = read_inflight(tmp_path)
    assert still_theirs is not None and still_theirs.execution_id == _THEIRS


def test_a_claim_leaves_no_temp_sibling(tmp_path: Path) -> None:
    """The exclusive create must not litter: the sweeper enumerates these dirs."""
    assert try_create_inflight(tmp_path, _marker(_MINE)) is True

    assert [p.name for p in tmp_path.iterdir()] == [inflight_marker_path(tmp_path).name]
