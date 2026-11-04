"""Stand in for ``nvidia-smi`` in the tests of GPU card leases.

The lease module finds ``nvidia-smi`` through one lookup,
``gpu_lease._nvidia_smi``. :func:`isolate_card_leases`, which the root conftest
applies to every test, makes that lookup find nothing and points the lease
directory at a directory of the test's own, so a test on a GPU machine leases no
real card. :func:`install_fake_nvidia_smi` writes a program that prints the rows
a test chooses and makes the lookup return it. A test that starts a child process
which queries the cards puts the program's directory first on the child's
``PATH``, because the lookup stub does not reach a child.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from mosaic.core.pipeline.graph import gpu_lease

TWO_CARDS: tuple[str, ...] = (
    "0, GPU-11111111-1111-1111-1111-111111111111, NVIDIA RTX A4000, Disabled",
    "1, GPU-22222222-2222-2222-2222-222222222222, NVIDIA RTX A4000, Disabled",
)
"""The query rows of a host with two cards of one model and MIG off."""


def isolate_card_leases(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Lease cards in *directory*, and find no ``nvidia-smi`` in this process.

    Child processes inherit the lease directory through the environment.
    """
    monkeypatch.setenv(gpu_lease.LEASE_DIR_VAR, str(directory))
    monkeypatch.setattr(gpu_lease, "_nvidia_smi", lambda: None)


def install_fake_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rows: Sequence[str],
    *,
    exit_code: int = 0,
    hang_seconds: float = 0.0,
) -> Path:
    """Write a fake ``nvidia-smi`` and make the lease module's lookup return it.

    The program sleeps for *hang_seconds*, prints each of *rows* on its own line,
    and exits with *exit_code*. It is a shell script that execs this test's
    interpreter, so a timeout that kills it kills the sleep too.

    Args:
        monkeypatch: The test's monkeypatch, which restores the lookup.
        tmp_path: The directory under which the program is written, in
            ``nvidia-smi-bin/``.
        rows: The lines the query prints.
        exit_code: The exit status of every run.
        hang_seconds: How long each run sleeps before it prints.

    Returns:
        The path of the program.
    """
    directory = tmp_path / "nvidia-smi-bin"
    directory.mkdir(parents=True, exist_ok=True)
    body = directory / "nvidia_smi_body.py"
    _ = body.write_text(
        "import sys, time\n"
        f"time.sleep({hang_seconds!r})\n"
        f"for row in {list(rows)!r}:\n"
        "    print(row)\n"
        f"sys.exit({exit_code!r})\n"
    )
    program = directory / "nvidia-smi"
    _ = program.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(body))} "$@"\n'
    )
    program.chmod(0o755)
    monkeypatch.setattr(gpu_lease, "_nvidia_smi", lambda: str(program))
    return program
