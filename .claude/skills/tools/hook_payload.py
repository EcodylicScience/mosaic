"""Read a hook's JSON payload and print the hook's answer."""

import json
import sys

from json_value import Json, parse_json


def read_payload() -> dict[str, Json] | None:
    """Return the payload on standard input, None when it is not a JSON
    object."""
    try:
        payload = parse_json(sys.stdin.read())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def print_answer(event: str, fields: dict[str, Json]) -> None:
    """Print the hook's answer with `event` as the event name and `fields`."""
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, **fields}}))
