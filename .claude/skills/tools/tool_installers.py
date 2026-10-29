"""Map each skill installer to the commands that it installs, in a repository
that the skills are synced into."""

from pathlib import PurePosixPath

# Each installer, as a path from the skills directory.
INSTALLERS = {
    "heavy-task/scripts/install-heavy": ("heavy",),
    "remote-task/scripts/install-remote": ("remote",),
    "mutate/scripts/install-mutate": ("mutate",),
    "concise-prose/scripts/install-prose-check": ("prose-check",),
    "review-prose/scripts/install-review-prose": ("launch-gate", "teardown-gate"),
    "rewrap/scripts/install-rewrap": ("wrap-check", "rewrap-comment", "rewrap-md"),
}


def installer(command: str) -> str:
    """Return the installer of `command`, as a path from the repository
    root."""
    path = next(path for path, commands in INSTALLERS.items() if command in commands)
    return f".claude/skills/{path}"


def installed_commands(installer_name: str) -> tuple[str, ...]:
    """Return the commands that the installer named `installer_name`
    installs, empty when no installer has that name."""
    return next(
        (commands for path, commands in INSTALLERS.items() if PurePosixPath(path).name == installer_name), ()
    )
