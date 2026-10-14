"""Mosaic's side of the runner protocol: one runner process, started when first asked.

A feature that drives a tool in a separate environment creates one
:class:`RunnerProcess`. The process runs a runner program, which serves
requests through ``runner_wire``. It starts at the first request, survives
between requests, and starts again after it exits, whatever the reason. The
feature stops it with :meth:`RunnerProcess.close`, which also removes the
directory of the files that the feature and the runner exchange.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, final

from pydantic import BaseModel, JsonValue, ValidationError

from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.job import CancelCheck, Cancelled
from mosaic.core.pipeline.subprocess_util import LineChild, ProcessCancelled
from mosaic.core.pipeline.toolenv import (
    PROBE_DEADLINE_FLOOR_SECONDS,
    ToolEnv,
    ToolExitError,
    ToolInvocation,
    interpreter_refusal,
    locate_tool,
    missing_interpreter,
    subprocess_env,
)

from .external.runner_wire import (
    REPLIES,
    SHUTDOWN,
    Done,
    Event,
    Failed,
    Ready,
    Reply,
    Request,
)

__all__ = [
    "OnStart",
    "RunnerCommandError",
    "RunnerInterpreter",
    "RunnerProcess",
    "RunnerVersionError",
]

log = logging.getLogger(__name__)

_CLOSE_GRACE_SECONDS: Final = 10.0
"""How long :meth:`RunnerProcess.close` waits for a runner, and then for each signal."""

_PROTOCOL_BREAK: Final = (
    "Mosaic stopped the runner on reading the line above, which breaks the runner "
    "protocol. The exit code is that of the stopped runner."
)
"""Ends the standard output of an error for a line that breaks the protocol."""

OnStart: TypeAlias = Callable[["RunnerProcess", CancelCheck | None], None]
"""Called each time that a runner is ready.

It receives the :class:`RunnerProcess` and the cancel check of the request that
started the runner.
"""


@final
@dataclass(frozen=True, slots=True)
class RunnerInterpreter:
    """A prerequisite of a feature that drives a runner: an interpreter for it.

    Resolved through *env*, the ladder that starting the runner walks, without
    starting the interpreter. The catalog does not read a step's interpreter
    param. Only the machine's rungs answer: the conda environment variable, the
    interpreter variable and the environment under the runner's directory.

    Attributes:
        env: The tool's location ladder.
    """

    env: ToolEnv

    def missing(self) -> str | None:
        """How to provide the environment, or ``None`` when one is found."""
        return missing_interpreter(self.env)


class RunnerCommandError(RuntimeError):
    """Raised when a runner's handler raised, with the traceback from the runner.

    Attributes:
        tool: The tool that the runner drives.
        command: The command that failed.
        message: The exception's type and text, as the runner reported them.
        traceback: The traceback in the runner's process.
    """

    def __init__(self, tool: str, command: str, message: str, traceback: str) -> None:
        self.tool = tool
        self.command = command
        self.message = message
        self.traceback = traceback
        text = f"The {tool} runner's command '{command}' failed: {message}"
        if traceback:
            text += f"\n\nRunner traceback:\n{traceback}"
        super().__init__(text)


class RunnerVersionError(RuntimeError):
    """Raised when a runner reports a tool version other than the one required.

    Attributes:
        required: The version that mosaic requires.
        found: The version that the runner reported.
    """

    def __init__(
        self, env: ToolEnv, invocation: ToolInvocation, required: str, found: str
    ) -> None:
        self.required = required
        self.found = found
        claim = f"runs {env.tool} {found}, and mosaic requires {required}"
        super().__init__(
            interpreter_refusal(env, invocation.argv[-1], invocation.source, claim)
        )


class RunnerProcess:
    """One runner program in a tool's environment, started by the first request.

    The interpreter is located through *env*'s ladder and runs *script* with
    :func:`~mosaic.core.pipeline.toolenv.subprocess_env`. Before the runner's
    ``Ready``, a line that is not an envelope is logged and skipped, since an
    activation script run by ``conda run`` may print. ``Ready`` is awaited for
    :data:`~mosaic.core.pipeline.toolenv.PROBE_DEADLINE_FLOOR_SECONDS`, its
    version is checked, and *on_start* is called. A runner that exited, for any
    reason, is started again by the next request, and *on_start* runs again. A
    request made by *on_start* does not start a runner. When the runner that
    *on_start* prepares has exited, such a request raises *failure*.

    The runner's standard error is appended to a temporary log file, which
    :meth:`log` reads and :meth:`close` returns and removes. :meth:`keep_log`
    names a file that receives the log, such as the log in a run root.
    :meth:`exchange_dir` creates the directory of the files exchanged with the
    runner, which :meth:`close` removes.

    Call the methods from one thread. On Linux the runner receives SIGTERM when
    the thread that started it exits.

    Args:
        env: The tool's placement.
        script: The runner program.
        failure: The error raised when the runner exits during a request or
            breaks the protocol, for example with a non-envelope line after
            ``Ready``.
        version: The tool version that the runner must report, or ``None`` to
            accept any.
        on_start: Called with this process and the request's cancel check once
            each runner is ready. A feature sends ``load_model`` from it, with
            that cancel check.
        scratch_dir: The directory in which :meth:`exchange_dir` creates the
            exchange directory, or ``None`` for the system's temporary directory.
    """

    def __init__(
        self,
        env: ToolEnv,
        script: Path,
        *,
        failure: type[ToolExitError],
        version: str | None,
        on_start: OnStart | None = None,
        scratch_dir: str | Path | None = None,
    ) -> None:
        self._env = env
        self._script = script
        self._failure = failure
        self._version = version
        self._on_start = on_start
        self._child: LineChild | None = None
        # The runner that on_start prepares, while on_start runs.
        self._starting: LineChild | None = None
        self._log_path: Path | None = None
        # The file that keep_log names, and the characters of the log appended
        # to it.
        self._kept_log: Path | None = None
        self._kept = 0
        self._scratch_dir = scratch_dir
        self._exchange: Path | None = None

    def request(
        self,
        command: str,
        payload: BaseModel | None = None,
        *,
        on_event: Callable[[Event], None] | None = None,
        cancel: CancelCheck | None = None,
    ) -> JsonValue:
        """Run *command* in the runner, and return its result.

        Args:
            command: The command to run.
            payload: The command's payload, or ``None`` for an empty one.
            on_event: Called with each event of the command, in order.
            cancel: Asked when each line arrives, and every 0.5 s while none
                does, including while a runner starts. *on_start* receives it.
                When it returns True, the runner's process group is sent
                SIGTERM, then SIGKILL if it runs 5 s later.

        Returns:
            The result of the runner's ``Done``, as JSON.

        Raises:
            ToolNotFoundError: The subclass that *env* names, when the
                interpreter cannot be located or started.
            RunnerVersionError: The runner reported another tool version.
            Cancelled: *cancel* returned True, or the runner reported the
                command as cancelled.
            RunnerCommandError: The command raised in the runner.
            ToolExitError: The *failure* subclass, when the runner exited or
                broke the protocol, or when the runner prepared by *on_start*
                was stopped. Its message ends with the tail of the runner's log.
            subprocess.TimeoutExpired: No ``Ready`` arrived in time.
        """
        child = self._running(cancel)
        body = {} if payload is None else payload.model_dump(mode="json")
        child.send(Request(command=command, payload=body).model_dump_json())
        try:
            reply = self._reply(child, on_event, cancel)
        except BaseException:
            # The runner may be inside the command, and its later lines would
            # answer the next request.
            self._discard()
            raise
        if isinstance(reply, Done):
            return reply.result
        if reply.kind == "cancelled":
            raise Cancelled()
        raise RunnerCommandError(
            self._env.tool, command, reply.message, reply.traceback
        )

    def locate(self) -> ToolInvocation:
        """Resolve the next runner's interpreter, without starting it.

        Raises:
            ToolNotFoundError: The subclass that *env* names, when its ladder
                does not locate an interpreter.
        """
        return locate_tool(self._env, executable="python")

    def exchange_dir(self) -> Path:
        """Return the directory of the files exchanged with the runner.

        The first call, and the first call after :meth:`close`, create it under
        *scratch_dir*. Every runner started before the next :meth:`close` shares
        it.
        """
        if self._exchange is None:
            prefix = f"mosaic-{self._env.tool.lower()}-"
            self._exchange = Path(
                tempfile.mkdtemp(prefix=prefix, dir=self._scratch_dir)
            )
        return self._exchange

    def log(self) -> str:
        """Return the standard error of the runners since the last :meth:`close`."""
        if self._log_path is None:
            return ""
        return self._log_path.read_text(encoding="utf-8", errors="replace")

    def keep_log(self, path: Path) -> None:
        """Append the standard error of the runners to *path*, now and at close.

        The standard error since the last close that no call appended is
        appended now, and :meth:`close` appends the rest. Each append replaces
        *path* atomically. A call with another path appends the rest there.
        """
        self._kept_log = path
        self._append_kept_log()

    def close(self, grace: float = _CLOSE_GRACE_SECONDS) -> str:
        """Stop the runner, and return its log.

        The rest of the log is appended to the file of :meth:`keep_log`, and the
        log file and the exchange directory are removed. A running runner is
        sent ``shutdown`` and its standard input is closed. A runner still
        running *grace* seconds later has its process group sent SIGTERM, and
        SIGKILL *grace* seconds after that. Once the runner has exited, any
        process left in its group is sent SIGKILL. A second call returns an empty
        log. A later request starts a new runner.

        Returns:
            The standard error of the runners since the last close.
        """
        child, self._child = self._child, None
        if child is not None:
            if child.alive:
                child.send(Request(command=SHUTDOWN).model_dump_json())
            child.close(grace)
        self._append_kept_log()
        self._kept_log = None
        self._kept = 0
        text = self.log()
        if self._log_path is not None:
            self._log_path.unlink(missing_ok=True)
            self._log_path = None
        if self._exchange is not None:
            shutil.rmtree(self._exchange, ignore_errors=True)
            self._exchange = None
        return text

    def _append_kept_log(self) -> None:
        """Append the rest of the log to the file that :meth:`keep_log` names."""
        path = self._kept_log
        if path is None:
            return
        text = self.log()
        rest = text[self._kept :]
        self._kept = len(text)
        if not rest:
            return
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        atomic_write(path, lambda target: target.write_text(previous + rest, "utf-8"))

    def _running(self, cancel: CancelCheck | None) -> LineChild:
        """Return the running runner, after starting one if none runs.

        Raises:
            ToolExitError: The *failure* subclass, when a request of *on_start*
                finds that the runner it prepares has exited, or when
                *on_start* returns after that runner was stopped.
        """
        if self._child is not None and self._child.alive:
            return self._child
        if self._starting is not None:
            # The request is on_start's. Each runner started for it would run
            # on_start again.
            raise self._exited(self._starting, [])
        self._discard()
        invocation = self.locate()
        child = self._spawn(invocation)
        self._child = child
        try:
            ready = self._ready(child, cancel)
            if self._version is not None and ready.version != self._version:
                raise RunnerVersionError(
                    self._env, invocation, self._version, ready.version
                )
            if self._on_start is not None:
                self._starting = child
                try:
                    self._on_start(self, cancel)
                finally:
                    self._starting = None
            if self._child is not child:
                # on_start caught the error of a request that stopped the runner.
                raise self._exited(child, [])
        except BaseException:
            self._discard()
            raise
        return child

    def _spawn(self, invocation: ToolInvocation) -> LineChild:
        if self._log_path is None:
            descriptor, name = tempfile.mkstemp(prefix="mosaic-runner-", suffix=".log")
            os.close(descriptor)
            self._log_path = Path(name)
        try:
            return LineChild(
                [*invocation.argv, str(self._script)],
                log_path=self._log_path,
                env=subprocess_env(),
            )
        except FileNotFoundError as missing:
            message = interpreter_refusal(
                self._env,
                invocation.argv[-1],
                invocation.source,
                "could not be started",
            )
            raise self._env.not_found(message) from missing

    def _ready(self, child: LineChild, cancel: CancelCheck | None) -> Ready:
        """Return the runner's ``Ready``, and skip the lines before it."""
        deadline = time.monotonic() + PROBE_DEADLINE_FLOOR_SECONDS
        skipped: list[str] = []
        while True:
            line = self._line(child, cancel, max(0.0, deadline - time.monotonic()))
            if line is None:
                raise self._exited(child, skipped)
            try:
                reply = REPLIES.validate_json(line)
            except ValidationError:
                log.info("The %s runner printed before Ready: %s", self._env.tool, line)
                skipped.append(line)
                continue
            if isinstance(reply, Ready):
                return reply
            raise self._broken(child, [*skipped, line])

    def _reply(
        self,
        child: LineChild,
        on_event: Callable[[Event], None] | None,
        cancel: CancelCheck | None,
    ) -> Done | Failed:
        """Pass the command's events to *on_event*, and return its reply."""
        while True:
            line = self._line(child, cancel, None)
            if line is None:
                raise self._exited(child, [])
            reply = self._parse(child, line)
            if isinstance(reply, Event):
                if on_event is not None:
                    on_event(reply)
            elif isinstance(reply, (Done, Failed)):
                return reply
            else:
                raise self._broken(child, [line])

    def _parse(self, child: LineChild, line: str) -> Reply:
        try:
            return REPLIES.validate_json(line)
        except ValidationError:
            raise self._broken(child, [line]) from None

    def _line(
        self,
        child: LineChild,
        cancel: CancelCheck | None,
        timeout: float | None,
    ) -> str | None:
        """Return the runner's next line, converting a cancel to :class:`Cancelled`."""
        try:
            return child.next_line(timeout, cancel_check=cancel)
        except ProcessCancelled as cancelled:
            self._discard()
            raise Cancelled() from cancelled

    def _exited(self, child: LineChild, stdout: list[str]) -> ToolExitError:
        """Stop *child*, and return the *failure* error for it.

        Args:
            child: The runner that exited or broke the protocol.
            stdout: The lines on standard output that were not envelopes.
        """
        self._discard()
        # The status of a process that survived SIGKILL is unknown.
        returncode = child.exit_code if child.exit_code is not None else -1
        return self._failure(child.argv, returncode, "\n".join(stdout), self.log())

    def _broken(self, child: LineChild, stdout: list[str]) -> ToolExitError:
        """Stop *child*, and return *failure* for a line that breaks the protocol.

        Args:
            child: The runner that wrote the line.
            stdout: The runner's skipped lines on standard output, then the line
                that breaks the protocol.
        """
        return self._exited(child, [*stdout, _PROTOCOL_BREAK])

    def _discard(self) -> None:
        """Stop the runner, if one was started, without asking it to shut down."""
        child, self._child = self._child, None
        if child is not None:
            child.close(_CLOSE_GRACE_SECONDS)
