"""The keypoint-MoSeq runner program, run in mosaic's environment, without JAX.

The program imports keypoint-moseq and JAX at its first command. It therefore
imports, and serves ``Ready`` and ``shutdown``, in an environment that has neither.
"""

from __future__ import annotations

import sys
from typing import Final, get_args

import numpy as np

from mosaic.behavior.feature_library.external.kpms_protocol import Command
from mosaic.core.pipeline.toolenv import subprocess_env
from tests.helpers import (
    KPMS_SERVER,
    SOURCE_ROOT,
    LineChildren,
    assert_ready_then_shutdown,
    import_runner_program,
)

_SERVER: Final = SOURCE_ROOT / KPMS_SERVER
"""The runner program."""

_SIBLINGS: Final = ("kpms_protocol", "runner_wire")
"""The modules that the runner program imports by their bare names."""


def test_the_program_imports_without_its_libraries() -> None:
    """Importing the program neither imports JAX nor patches numpy."""
    bincount = np.bincount

    with import_runner_program(_SERVER, _SIBLINGS) as program:
        assert callable(program.main)

    assert np.bincount is bincount
    assert "jax" not in sys.modules
    assert "keypoint_moseq" not in sys.modules


def test_the_dispatch_serves_exactly_the_commands_of_the_protocol() -> None:
    with import_runner_program(_SERVER, _SIBLINGS) as program:
        dispatch = program.dispatch(program.KpmsServer())

    assert sorted(dispatch) == sorted(get_args(Command))


def test_the_program_answers_ready_and_then_shutdown(
    line_children: LineChildren,
) -> None:
    """The program runs under this interpreter, without keypoint-moseq."""
    server = line_children.spawn([sys.executable, str(_SERVER)], env=subprocess_env())

    ready = assert_ready_then_shutdown(server)

    assert ready.tool == "keypoint-moseq"
