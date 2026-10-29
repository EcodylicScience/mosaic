"""Run tmux commands, read their answers, and build the claude command that
a tmux pane runs."""

import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from lossless_text import decoded, nonempty_lines
from program_start import Stream, run_program


def tmux(arguments: list[str], stdout: Stream = None, stderr: Stream = None) -> subprocess.CompletedProcess[bytes]:
    return run_program(["tmux", *arguments], stdout=stdout, stderr=stderr)


def tmux_output(arguments: list[str]) -> bytes:
    """Return the standard output of a tmux query, empty when it fails, as it
    does without a server."""
    query = tmux(arguments, subprocess.PIPE, subprocess.DEVNULL)
    return query.stdout if query.returncode == 0 else b""


def tmux_lines(arguments: list[str]) -> list[str]:
    """Return the lines that a tmux query prints, none when it fails, as with
    no server."""
    return nonempty_lines(decoded(tmux_output(arguments)))


def tmux_session_exists(name: str) -> bool:
    """Return whether a tmux session is named exactly `name`."""
    return tmux(["has-session", "-t", f"={name}"], stderr=subprocess.DEVNULL).returncode == 0


def caller_session() -> str | None:
    """Return the tmux session name of the pane in TMUX_PANE, or None outside
    tmux and for a pane that tmux does not know.

    An empty TMUX_PANE counts as outside tmux. tmux reads an empty target as
    the most recent session, which need not be the caller's.
    """
    pane = os.environ.get("TMUX_PANE")
    if not pane:
        return None
    names = tmux_lines(["display-message", "-p", "-t", pane, "#{session_name}"])
    return names[0] if names else None


def move_clients(source: str, destination: str) -> None:
    """Switch every client attached to the tmux session named exactly
    `source` to the session named exactly `destination`.

    A client that leaves between the query and its switch fails the switch,
    and the loop ignores that failure.
    """
    for client in tmux_lines(["list-clients", "-t", f"={source}", "-F", "#{client_tty}"]):
        _ = tmux(["switch-client", "-c", client, "-t", f"={destination}"], stderr=subprocess.DEVNULL)


def kill_session(name: str, destination: str | None) -> bool:
    """Kill the tmux session named exactly `name`, after moving its clients to
    the session named exactly `destination` when one is given. Return whether
    tmux killed it."""
    if destination is not None:
        move_clients(name, destination)
    return tmux(["kill-session", "-t", f"={name}"], stderr=subprocess.DEVNULL).returncode == 0


def kill_window(target: str) -> bool:
    """Kill the tmux window that `target` names. Return whether tmux killed
    it."""
    return tmux(["kill-window", "-t", target], stderr=subprocess.DEVNULL).returncode == 0


def claude_command(
    environment: Sequence[tuple[str, str]], agent: str, name: str, prompt: Path | str, options: Sequence[str] = ()
) -> str:
    """Return the shell command that runs `claude` as `agent` under `name`
    with `options` and `environment` assigned.

    A Path prompt is the content of that file and a str prompt is the text
    itself.
    """
    assignments = [f"{variable}={shlex.quote(value)}" for variable, value in environment]
    words = [*assignments, "claude", *(shlex.quote(word) for word in ["--agent", agent, "-n", name, *options])]
    if isinstance(prompt, Path):
        words.append(f'"$(cat {shlex.quote(str(prompt))})"')
    else:
        words.append(shlex.quote(prompt))
    return " ".join(words)
