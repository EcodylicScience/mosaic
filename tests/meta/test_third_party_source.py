"""A source walk skips third-party code, and a checkout of it, and reads mosaic's own.

A user builds the lmt-analysis environment beside a clone of lmt-analysis, under
``tracking/external/lmt-env/``. The clone has no ``site-packages`` and no
``pyvenv.cfg``, and each of its scripts imports ``lmtanalysis``, which the
separation and layering guards allow only in the runner. A walk that read the
clone as mosaic's source would fail those guards for every user who built the
environment as its README says.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import is_third_party_source


def _source(root: Path, *parts: str) -> Path:
    source = root.joinpath(*parts)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("")
    return source


@pytest.mark.parametrize(
    "git_entry_is_a_file", [False, True], ids=["clone", "worktree"]
)
def test_a_file_in_a_git_checkout_under_the_package_is_third_party(
    tmp_path: Path, git_entry_is_a_file: bool
) -> None:
    """A checkout's ``.git`` is a directory, and a linked worktree's is a file."""
    checkout = tmp_path / "tracking" / "external" / "lmt-env" / "lmt-analysis"
    _source(checkout, "LMT", "scripts", "Rebuild_All_Events.py")
    if git_entry_is_a_file:
        (checkout / ".git").write_text("gitdir: ../elsewhere\n")
    else:
        (checkout / ".git").mkdir()

    assert is_third_party_source(
        checkout / "LMT" / "scripts" / "Rebuild_All_Events.py", tmp_path
    )


def test_a_file_of_the_package_is_not_third_party_though_the_root_is_a_checkout(
    tmp_path: Path,
) -> None:
    """The package root's own ``.git`` is above the walked range."""
    (tmp_path / ".git").write_text("gitdir: ../elsewhere\n")
    own = _source(tmp_path, "tracking", "external", "runner", "lmt_rebuild_runner.py")

    assert not is_third_party_source(own, tmp_path)


def test_a_file_beside_a_checkout_is_not_third_party(tmp_path: Path) -> None:
    (tmp_path / "tracking" / "external" / "lmt-env" / "lmt-analysis" / ".git").mkdir(
        parents=True
    )
    beside = _source(tmp_path, "tracking", "external", "lmt-env", "other.py")

    assert not is_third_party_source(beside, tmp_path)


def test_installed_packages_and_virtualenvs_are_still_third_party(
    tmp_path: Path,
) -> None:
    installed = _source(tmp_path, "env", "lib", "site-packages", "pandas", "frame.py")
    environment = tmp_path / "tracking" / "external" / "any-name"
    environment.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("")
    inside = _source(environment, "lib", "module.py")

    assert is_third_party_source(installed, tmp_path)
    assert is_third_party_source(inside, tmp_path)
