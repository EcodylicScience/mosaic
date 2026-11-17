"""Names the machine a job runs on, for the ``started`` event of its run-log.

Imports the standard library, psutil and :mod:`mosaic.runlog`. psutil and ``/proc``
are read when :func:`current_machine` runs, never at import.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Final

import psutil

from mosaic.core.pipeline.parallelism import threads_per_core
from mosaic.runlog import MachineFacts

__all__ = ["MACHINE_CLASS_ENV", "current_machine"]

MACHINE_CLASS_ENV: Final = "MOSAIC_MACHINE_CLASS"
"""The environment variable an operator sets to name the kind of machine a job runs on."""

_MODEL_KEY: Final = "model name"


def current_machine(*, proc_root: Path = Path("/proc")) -> MachineFacts:
    """Return the facts of the machine this process runs on.

    Args:
        proc_root: The directory that holds ``cpuinfo``. A host without one reports
            the platform's processor name instead.

    Returns:
        The machine, a fact it cannot learn left empty.
    """
    return MachineFacts(
        cpu_model=_cpu_model(proc_root),
        logical_cpus=os.cpu_count() or 0,
        threads_per_core=threads_per_core(),
        memory_bytes=psutil.virtual_memory().total,
        machine_class=os.environ.get(MACHINE_CLASS_ENV, ""),
    )


def _cpu_model(proc_root: Path) -> str:
    """The first ``model name`` of ``cpuinfo``, else the platform's processor, else ``""``."""
    try:
        text = (proc_root / "cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == _MODEL_KEY and value.strip():
            return value.strip()
    return platform.processor()
