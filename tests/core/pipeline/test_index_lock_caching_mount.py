"""A CIFS mount whose client caches loses index rows across machines.

The lock excludes the other machines, yet a client that caches attributes reads
an index another machine replaced as the old version, and one that keeps closed
handles open fails the open with EINVAL. ``index_lock`` warns once per such
mount.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from mosaic.core.pipeline import index_lock as lock_module
from mosaic.core.pipeline.index_lock import (
    CachingMountWarning,
    cifs_caching_options,
    index_lock,
)

_ROOT = "22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw"
_SHARE_OPTIONS = "rw,vers=3.0,cache=strict,soft,nounix,serverino"


def _share(mount_point: str, options: str) -> str:
    return (
        f"512 22 0:61 / {mount_point} rw,relatime shared:300 - cifs "
        f"//server/share {_SHARE_OPTIONS},{options}"
    )


def test_a_caching_cifs_mount_names_both_options() -> None:
    mountinfo = "\n".join([_ROOT, _share("/mnt/share", "actimeo=1,closetimeo=1")])

    found = cifs_caching_options(Path("/mnt/share/data/features/x"), mountinfo)

    assert found == ("/mnt/share", ["actimeo=1", "closetimeo=1"])


def test_a_cifs_mount_without_caching_is_not_named() -> None:
    mountinfo = "\n".join([_ROOT, _share("/mnt/share", "actimeo=0,closetimeo=0")])

    assert cifs_caching_options(Path("/mnt/share/data"), mountinfo) is None


def test_a_separate_file_attribute_timeout_is_read() -> None:
    """``acregmax`` replaces ``actimeo`` in the options when it differs."""
    mountinfo = _share("/mnt/share", "acdirmax=30,acregmax=0,closetimeo=0")

    assert cifs_caching_options(Path("/mnt/share/data"), mountinfo) is None


def test_a_kernel_without_closetimeo_is_judged_on_actimeo() -> None:
    mountinfo = _share("/mnt/share", "actimeo=1")

    found = cifs_caching_options(Path("/mnt/share"), mountinfo)

    assert found == ("/mnt/share", ["actimeo=1"])


def test_the_longest_mount_point_holds_the_directory() -> None:
    """A local disk mounted under the share is not the share."""
    mountinfo = "\n".join(
        [
            _ROOT,
            _share("/mnt/share", "actimeo=1,closetimeo=1"),
            "600 512 259:3 / /mnt/share/local rw - ext4 /dev/sdb1 rw",
        ]
    )

    assert cifs_caching_options(Path("/mnt/share/local/data"), mountinfo) is None
    assert cifs_caching_options(Path("/mnt/shared/data"), mountinfo) is None


def test_an_escaped_mount_point_is_matched() -> None:
    mountinfo = _share("/mnt/my\\040share", "actimeo=1,closetimeo=0")

    found = cifs_caching_options(Path("/mnt/my share/data"), mountinfo)

    assert found == ("/mnt/my share", ["actimeo=1"])


def test_a_local_disk_is_not_named() -> None:
    assert cifs_caching_options(Path("/home/data"), _ROOT) is None


@pytest.fixture
def caching_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Report *tmp_path* as a CIFS mount that caches."""
    mountinfo = "\n".join(
        [_ROOT, _share(str(tmp_path.resolve()), "actimeo=1,closetimeo=1")]
    )
    monkeypatch.setattr(lock_module, "_read_mountinfo", lambda: mountinfo)
    checked: set[Path] = set()
    warned: set[str] = set()
    monkeypatch.setattr(lock_module, "_checked_dirs", checked)
    monkeypatch.setattr(lock_module, "_warned_mounts", warned)
    return tmp_path


def test_index_lock_warns_once_per_caching_mount(caching_share: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with index_lock(caching_share / "a" / "index.csv"):
            pass
        with index_lock(caching_share / "a" / "index.csv"):
            pass
        with index_lock(caching_share / "b" / "index.csv"):
            pass

    found = [w for w in caught if issubclass(w.category, CachingMountWarning)]
    assert len(found) == 1
    assert "actimeo=0,closetimeo=0" in str(found[0].message)


def test_index_lock_is_silent_where_mountinfo_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: set[Path] = set()
    monkeypatch.setattr(lock_module, "_read_mountinfo", lambda: None)
    monkeypatch.setattr(lock_module, "_checked_dirs", checked)

    with warnings.catch_warnings():
        warnings.simplefilter("error", CachingMountWarning)
        with index_lock(tmp_path / "index.csv"):
            pass
