"""Name the per-user state that hooks keep under the system temp directory."""

import os
import re
import tempfile
from pathlib import Path
from typing import TypeGuard

from json_value import Json

# Matches a session or agent id that the harness provides. Hooks use only a
# matching id in a file path or a shell command.
SAFE_ID = re.compile(r"[\w.-]+", re.ASCII)


def safe_id(value: Json) -> TypeGuard[str]:
    """Return whether `value` is an id that names one path component."""
    return (
        isinstance(value, str)
        and value not in (".", "..")
        and SAFE_ID.fullmatch(value) is not None
    )


def hook_state_directory(name: str) -> Path:
    """Return the current user's state directory for `name`, without creating
    it."""
    return Path(tempfile.gettempdir()) / f"{name}-{os.getuid()}"
