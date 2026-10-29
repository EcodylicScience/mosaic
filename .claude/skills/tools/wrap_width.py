"""Find the column that a file's prose wraps at, from its language's
`REWRAP_WIDTH_*`."""

import os
import re
from pathlib import Path

from claude_settings import SETTINGS_LOCAL_PATH, SettingsError, environment_block
from git_in_directory import repository_toplevel
from lexical_paths import normal_join, working_directory
from quiet_paths import present

DEFAULT_WIDTH = 79
WIDTH_FORM = re.compile(r"[0-9]+")
MARKDOWN_SUFFIXES = (".md", ".markdown")

# Maps path suffixes to keys, in order. Any other file reads the markdown key.
SUFFIX_KEYS = (
    (MARKDOWN_SUFFIXES, "REWRAP_WIDTH_MD"),
    ((".py",), "REWRAP_WIDTH_PY"),
    ((".ts", ".tsx", ".js", ".jsx", ".mjs"), "REWRAP_WIDTH_TS"),
    ((".sh", ".bash", ".awk"), "REWRAP_WIDTH_SH"),
)


def width_key(path: Path) -> str:
    for suffixes, key in SUFFIX_KEYS:
        if path.name.endswith(suffixes):
            return key
    return "REWRAP_WIDTH_MD"


def readable_width(value: object) -> int | None:
    text = str(value)
    return int(text) if WIDTH_FORM.fullmatch(text) else None


def repository_root(directory: Path) -> Path:
    """Return the root of the repository that contains `directory`, or else
    `directory`.

    Roots that do not contain the directory are discarded.
    """
    root = repository_toplevel(directory)
    if root is None or not directory.is_relative_to(root):
        return directory
    return root


def configured_width(path: Path) -> int | None:
    """Return the width that the nearest settings file up to the repository
    root sets, or else the environment's.

    The walk stops at the root and does not read the settings of a directory
    above the checkout. The first settings file found ends the walk. Files
    without the key fall through to the environment. The environment contains
    the env block of the settings that the session loaded. Worktrees need the
    fallback, because their root lacks a settings file.
    """
    key = width_key(path)
    directory = normal_join(working_directory(), path.parent).resolve()
    root = repository_root(directory)
    while True:
        settings = directory / SETTINGS_LOCAL_PATH
        if present(settings.is_file):
            try:
                found = readable_width(environment_block(settings).get(key, ""))
            except SettingsError:
                found = None
            if found is not None:
                return found
            break
        if directory in (root, Path("/")):
            break
        directory = directory.parent
    return readable_width(os.environ.get(key, ""))


def wrap_width(path: Path) -> int:
    width = configured_width(path)
    return DEFAULT_WIDTH if width is None else width
