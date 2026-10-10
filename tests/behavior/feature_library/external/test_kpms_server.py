"""The keypoint-MoSeq runner program, run in mosaic's environment, without JAX.

The program imports keypoint-moseq and JAX at its first command. It therefore
imports, and serves ``Ready`` and ``shutdown``, in an environment that has neither.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final, get_args

import numpy as np
import pytest

from mosaic.behavior.feature_library.external.kpms_protocol import Command
from mosaic.behavior.feature_library.external.runner_wire import (
    REPLIES,
    SHUTDOWN,
    Done,
    Ready,
    Request,
)
from mosaic.core.pipeline.subprocess_util import LineChild
from mosaic.core.pipeline.toolenv import subprocess_env
from tests.helpers import SOURCE_ROOT, import_runner_program, wait_for

_EXTERNAL: Final = SOURCE_ROOT / "behavior" / "feature_library" / "external"
"""The directory of the runner program and its siblings."""

_WAIT: Final = 60.0
"""How long a test waits for a line, which covers a runner's start under load."""


@pytest.fixture
def server(tmp_path: Path) -> Iterator[LineChild]:
    """The runner program, started under this interpreter and closed at the end."""
    child = LineChild(
        [sys.executable, str(_EXTERNAL / "kpms_server.py")],
        log_path=tmp_path / "kpms_server.log",
        env=subprocess_env(),
    )
    yield child
    child.close(grace=1.0)


def test_the_program_imports_without_its_libraries() -> None:
    """Importing the program neither imports JAX nor patches numpy."""
    bincount = np.bincount

    with import_runner_program(
        _EXTERNAL, "kpms_server", ("kpms_protocol", "runner_wire")
    ) as program:
        assert callable(program.main)

    assert np.bincount is bincount
    assert "jax" not in sys.modules
    assert "keypoint_moseq" not in sys.modules


def test_the_dispatch_serves_exactly_the_commands_of_the_protocol() -> None:
    with import_runner_program(
        _EXTERNAL, "kpms_server", ("kpms_protocol", "runner_wire")
    ) as program:
        dispatch = program.dispatch(program.KpmsServer())

    assert sorted(dispatch) == sorted(get_args(Command))


def test_the_program_answers_ready_and_then_shutdown(server: LineChild) -> None:
    ready = server.next_line(timeout=_WAIT)
    assert ready is not None, server.log_path.read_text()
    reply = REPLIES.validate_json(ready)
    assert isinstance(reply, Ready)
    assert reply.tool == "keypoint-moseq"

    server.send(Request(command=SHUTDOWN).model_dump_json())

    done = server.next_line(timeout=_WAIT)
    assert done is not None
    assert REPLIES.validate_json(done) == Done()
    wait_for(lambda: not server.alive, _WAIT, "the runner did not exit")
    assert server.exit_code == 0
