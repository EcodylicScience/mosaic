"""The memory limit that applies to a process: a cgroup's or the host's."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mosaic.core.pipeline.memory_limit import (
    MemoryLimit,
    read_memory_limit,
    usable_memory_bytes,
)

GIB = 1024**3
HOST_TOTAL = 64 * GIB
HOST_AVAILABLE = 40 * GIB


@pytest.fixture(autouse=True)
def _host(monkeypatch: pytest.MonkeyPatch) -> None:
    host = types.SimpleNamespace(total=HOST_TOTAL, available=HOST_AVAILABLE)
    module = types.SimpleNamespace(virtual_memory=lambda: host)
    monkeypatch.setitem(sys.modules, "psutil", module)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _v2_level(
    directory: Path,
    *,
    max_text: str,
    current: int = 0,
    inactive_file: int | None = None,
    peak: int | None = None,
) -> None:
    _write(directory / "memory.max", max_text + "\n")
    _write(directory / "memory.current", f"{current}\n")
    if inactive_file is not None:
        _write(directory / "memory.stat", f"anon 1\ninactive_file {inactive_file}\n")
    if peak is not None:
        _write(directory / "memory.peak", f"{peak}\n")


def _proc(tmp_path: Path, text: str) -> Path:
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(text)
    return proc


def _read(tmp_path: Path) -> MemoryLimit:
    limit = read_memory_limit(
        cgroup_root=tmp_path / "cgroup", proc_self_cgroup=tmp_path / "proc_self_cgroup"
    )
    assert limit is not None
    return limit


def test_a_namespaced_v2_root_reads_the_limit_beside_the_process(
    tmp_path: Path,
) -> None:
    _proc(tmp_path, "0::/\n")
    _v2_level(tmp_path / "cgroup", max_text=str(8 * GIB), current=3 * GIB, peak=5 * GIB)

    limit = _read(tmp_path)

    assert limit == MemoryLimit(
        limit_bytes=8 * GIB,
        source="cgroup_v2",
        in_use_bytes=3 * GIB,
        peak_bytes=5 * GIB,
        host_total_bytes=HOST_TOTAL,
        host_available_bytes=HOST_AVAILABLE,
    )
    assert limit.available_bytes == 5 * GIB


def test_a_limit_two_levels_up_applies_with_its_own_figures(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/job/step/task\n")
    root = tmp_path / "cgroup"
    _v2_level(root / "job" / "step" / "task", max_text="max", current=1 * GIB)
    _v2_level(root / "job" / "step", max_text="max", current=2 * GIB)
    _v2_level(root / "job", max_text=str(16 * GIB), current=6 * GIB, peak=9 * GIB)

    limit = _read(tmp_path)

    assert (limit.limit_bytes, limit.source) == (16 * GIB, "cgroup_v2")
    assert (limit.in_use_bytes, limit.peak_bytes) == (6 * GIB, 9 * GIB)


def test_the_smallest_limit_on_the_way_up_applies(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/job/step\n")
    root = tmp_path / "cgroup"
    _v2_level(root / "job" / "step", max_text=str(4 * GIB), current=1 * GIB)
    _v2_level(root / "job", max_text=str(16 * GIB), current=6 * GIB)

    limit = _read(tmp_path)

    assert (limit.limit_bytes, limit.in_use_bytes) == (4 * GIB, 1 * GIB)


def test_max_at_every_level_is_the_host(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/job/step\n")
    root = tmp_path / "cgroup"
    _v2_level(root / "job" / "step", max_text="max")
    _v2_level(root / "job", max_text="max")
    _v2_level(root, max_text="max")

    limit = _read(tmp_path)

    assert limit.source == "host"
    assert limit.limit_bytes == HOST_TOTAL
    assert limit.in_use_bytes == HOST_TOTAL - HOST_AVAILABLE
    assert limit.peak_bytes == 0
    assert limit.available_bytes == HOST_AVAILABLE


def test_a_limit_at_or_above_the_host_total_is_no_limit(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/\n")
    _v2_level(tmp_path / "cgroup", max_text=str(HOST_TOTAL), current=GIB)

    limit = _read(tmp_path)

    assert limit.source == "host"
    assert limit.available_bytes == HOST_AVAILABLE


def test_a_malformed_limit_skips_its_level(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/job/step\n")
    root = tmp_path / "cgroup"
    _v2_level(root / "job" / "step", max_text="lots", current=GIB)
    _v2_level(root / "job", max_text=str(16 * GIB), current=6 * GIB)

    limit = _read(tmp_path)

    assert (limit.limit_bytes, limit.source) == (16 * GIB, "cgroup_v2")


def test_a_missing_peak_reads_as_zero(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/\n")
    _v2_level(tmp_path / "cgroup", max_text=str(8 * GIB), current=GIB)

    assert _read(tmp_path).peak_bytes == 0


def test_the_inactive_file_cache_is_not_in_use(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/\n")
    _v2_level(
        tmp_path / "cgroup",
        max_text=str(8 * GIB),
        current=5 * GIB,
        inactive_file=2 * GIB,
    )

    limit = _read(tmp_path)

    assert limit.in_use_bytes == 3 * GIB
    assert limit.available_bytes == 5 * GIB


def test_version_1_reads_the_memory_controller(tmp_path: Path) -> None:
    _proc(tmp_path, "12:cpu,cpuacct:/x\n4:memory:/\n")
    memory = tmp_path / "cgroup" / "memory"
    _write(memory / "memory.limit_in_bytes", f"{8 * GIB}\n")
    _write(memory / "memory.usage_in_bytes", f"{5 * GIB}\n")
    _write(memory / "memory.max_usage_in_bytes", f"{7 * GIB}\n")
    _write(memory / "memory.stat", f"cache 1\ntotal_inactive_file {2 * GIB}\n")

    limit = _read(tmp_path)

    assert limit.source == "cgroup_v1"
    assert limit.limit_bytes == 8 * GIB
    assert (limit.in_use_bytes, limit.peak_bytes) == (3 * GIB, 7 * GIB)


def test_a_version_1_path_the_mount_lacks_reads_the_mount_root(tmp_path: Path) -> None:
    """A container whose cgroup namespace is not its own sees a host-relative path."""
    _proc(tmp_path, "4:memory:/kubepods/pod1/abc\n")
    memory = tmp_path / "cgroup" / "memory"
    _write(memory / "memory.limit_in_bytes", f"{8 * GIB}\n")
    _write(memory / "memory.usage_in_bytes", f"{5 * GIB}\n")

    limit = _read(tmp_path)

    assert (limit.source, limit.limit_bytes) == ("cgroup_v1", 8 * GIB)
    assert limit.in_use_bytes == 5 * GIB


def test_a_version_2_path_the_mount_lacks_reads_the_mount_root(tmp_path: Path) -> None:
    _proc(tmp_path, "0::/system.slice/x\n")
    _v2_level(tmp_path / "cgroup", max_text=str(8 * GIB), current=2 * GIB)

    limit = _read(tmp_path)

    assert (limit.source, limit.limit_bytes) == ("cgroup_v2", 8 * GIB)
    assert limit.in_use_bytes == 2 * GIB


def test_a_path_with_a_parent_component_reads_the_mount_root(tmp_path: Path) -> None:
    """The path is not followed out of the tree its own names describe."""
    _proc(tmp_path, "0::/a/../b\n")
    root = tmp_path / "cgroup"
    _v2_level(root, max_text=str(8 * GIB), current=2 * GIB)
    _v2_level(root / "b", max_text=str(2 * GIB), current=GIB)
    (root / "a").mkdir()

    limit = _read(tmp_path)

    assert limit.limit_bytes == 8 * GIB


def test_a_version_1_limit_that_is_unlimited_is_the_host(tmp_path: Path) -> None:
    _proc(tmp_path, "4:memory:/\n")
    memory = tmp_path / "cgroup" / "memory"
    _write(memory / "memory.limit_in_bytes", "9223372036854771712\n")
    _write(memory / "memory.usage_in_bytes", f"{GIB}\n")

    assert _read(tmp_path).source == "host"


def test_a_host_without_cgroup_files_is_the_host(tmp_path: Path) -> None:
    limit = _read(tmp_path)

    assert limit.source == "host"
    assert (limit.limit_bytes, limit.available_bytes) == (HOST_TOTAL, HOST_AVAILABLE)


def test_a_host_without_psutil_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert (
        read_memory_limit(
            cgroup_root=tmp_path / "cgroup",
            proc_self_cgroup=tmp_path / "proc_self_cgroup",
        )
        is None
    )
    assert usable_memory_bytes() == (0, 0)


def test_the_usable_memory_is_the_limit_and_what_is_left_under_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = MemoryLimit(
        limit_bytes=8 * GIB,
        source="cgroup_v2",
        in_use_bytes=3 * GIB,
        peak_bytes=0,
        host_total_bytes=HOST_TOTAL,
        host_available_bytes=HOST_AVAILABLE,
    )
    monkeypatch.setattr(
        "mosaic.core.pipeline.memory_limit.read_memory_limit", lambda: limit
    )

    assert usable_memory_bytes() == (8 * GIB, 5 * GIB)
