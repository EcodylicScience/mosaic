"""Read a Claude settings file and its env block."""

from pathlib import Path

from json_value import Json, parse_json

SETTINGS_PATH = Path(".claude") / "settings.json"
SETTINGS_LOCAL_PATH = Path(".claude") / "settings.local.json"


class SettingsError(Exception):
    """Signal a settings file that cannot be used.

    The message names the reason and omits the values in the file.
    """


class MissingSettingsError(SettingsError):
    """Signal a settings file that does not exist."""


def settings_object(path: Path) -> dict[str, Json]:
    """Return the JSON object in the settings file at `path`.

    Parse the file as JSON instead of matching a pattern. The file is JSON with
    escapes and nesting, and its `permissions.allow` array contains command
    strings that a loose pattern matches inside.
    """
    try:
        settings = parse_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        kind = MissingSettingsError if isinstance(error, FileNotFoundError) else SettingsError
        raise kind(f"cannot open {path}: {error.strerror}") from error
    except ValueError as error:
        raise SettingsError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(settings, dict):
        raise SettingsError(f"{path} does not hold a JSON object")
    return settings


def environment_of(settings: dict[str, Json], path: Path) -> dict[str, Json]:
    """Return the env block of the settings read from `path`, empty when they
    have none."""
    block = settings.get("env", {})
    if not isinstance(block, dict):
        raise SettingsError(f"{path}: env is not a JSON object")
    return block


def environment_block(path: Path) -> dict[str, Json]:
    """Return the env block of the settings file at `path`, empty when the file
    has none."""
    return environment_of(settings_object(path), path)
