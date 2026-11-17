"""The facts of the machine a job runs on, and the thread count they derive."""

from __future__ import annotations

import os
from pathlib import Path

import psutil
import pytest

from mosaic.core.pipeline import machine
from mosaic.core.pipeline.machine import MACHINE_CLASS_ENV, current_machine
from mosaic.core.pipeline.parallelism import threads_per_core


def _fake_proc(root: Path, cpuinfo: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _ = (root / "cpuinfo").write_text(cpuinfo, encoding="utf-8")
    return root


def test_the_cpu_model_comes_from_cpuinfo(tmp_path: Path) -> None:
    proc = _fake_proc(
        tmp_path / "proc",
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        "model name\t: Example Xeon 9000 @ 3.00GHz\n"
        "processor\t: 1\n"
        "model name\t: Some Other Model\n",
    )

    facts = current_machine(proc_root=proc)

    assert facts.cpu_model == "Example Xeon 9000 @ 3.00GHz"
    assert facts.logical_cpus == os.cpu_count()
    assert facts.memory_bytes == psutil.virtual_memory().total


def test_a_missing_cpuinfo_falls_back_to_the_platform_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(machine.platform, "processor", lambda: "arm")

    assert current_machine(proc_root=tmp_path / "absent").cpu_model == "arm"


def test_a_cpuinfo_without_a_model_name_and_no_platform_processor_reads_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _fake_proc(tmp_path / "proc", "processor\t: 0\n")
    monkeypatch.setattr(machine.platform, "processor", lambda: "")

    assert current_machine(proc_root=proc).cpu_model == ""


def test_the_machine_class_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MACHINE_CLASS_ENV, "gpu-a100")

    assert current_machine(proc_root=tmp_path).machine_class == "gpu-a100"


def test_no_machine_class_reads_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MACHINE_CLASS_ENV, raising=False)

    assert current_machine(proc_root=tmp_path).machine_class == ""


def test_the_threads_per_core_are_the_logical_over_the_physical_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def count(*, logical: bool = True) -> int | None:
        return 40 if logical else 20

    monkeypatch.setattr(psutil, "cpu_count", count)

    assert threads_per_core() == 2
    assert current_machine().threads_per_core == 2


@pytest.mark.parametrize(("logical", "physical"), [(8, None), (None, 4), (0, 0)])
def test_threads_per_core_is_none_when_physical_cores_are_unreported(
    monkeypatch: pytest.MonkeyPatch, logical: int | None, physical: int | None
) -> None:
    def count(*, logical: bool = True) -> int | None:
        return reported_logical if logical else physical

    reported_logical = logical
    monkeypatch.setattr(psutil, "cpu_count", count)

    assert threads_per_core() is None
    assert current_machine().threads_per_core is None
