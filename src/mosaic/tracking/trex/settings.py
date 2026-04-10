"""Generate T-Rex .settings files programmatically."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_settings_file(
    settings: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Write a T-Rex .settings file from a dictionary.

    Parameters
    ----------
    settings : dict
        Parameter name -> value pairs.  Values are serialized as follows:

        - ``bool``  -> ``true`` / ``false``
        - ``list``/``tuple`` -> ``[v0, v1, ...]``
        - ``str``   -> ``"value"`` (quoted)
        - numbers   -> bare literal

    output_path : path
        Destination file.  Parent directories are created automatically.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for key, value in settings.items():
        lines.append(f"{key} = {_format_value(value)}")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def _format_value(value: Any) -> str:
    """Serialize a Python value to T-Rex settings format."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        inner = ",".join(_format_value(v) for v in value)
        return f"[{inner}]"
    return str(value)
