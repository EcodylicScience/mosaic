"""Environment variable readers shared by `core`, `behavior` and `tracking`.

Top level rather than under `mosaic.core` for the reason `mosaic.media_probe_config`
is: a consumer that needs one variable should not import the dataset model, pandas
or cv2 to read it. The module imports only the standard library.
"""

from __future__ import annotations

import os


def int_env(name: str, default: int) -> int:
    """The integer value of environment variable *name*, or *default*.

    An unset or blank value falls back; a value that is not an integer raises
    naming the variable, because a silently ignored typo would produce a verdict
    nobody could explain.

    Args:
        name: The environment variable to read.
        default: The value for an unset or blank variable.

    Returns:
        The parsed integer, or *default*.

    Raises:
        ValueError: If the value is set and is not an integer.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        message = f"{name} must be an integer, got {raw!r}"
        raise ValueError(message) from exc


def flag_env(name: str) -> bool:
    """Whether environment variable *name* is exactly ``1``, whitespace aside.

    Surrounding whitespace is ignored: a value carried in from a ``.env`` file or
    a YAML ``env:`` block picks up a trailing space easily. Nothing else counts as
    set. Widening to ``true`` or ``yes`` would widen the ways to set a flag
    *accidentally*, as a job matrix that sets every flag to ``true`` would. An
    unset, blank, ``0`` or malformed value is one answer, ``False``, so a flag
    has no error of its own.

    Args:
        name: The environment variable to read.

    Returns:
        ``True`` only when the stripped value equals ``"1"``.
    """
    return os.environ.get(name, "").strip() == "1"
