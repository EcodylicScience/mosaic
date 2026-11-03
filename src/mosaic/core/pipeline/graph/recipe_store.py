"""Reading the recipes a dataset holds, one file per digest.

A leaf below the request store (:mod:`~.store`) and the rollup
(:mod:`~.rollup`). The rollup reads a request's recipe through it without
importing the request listing or the status wires that the store imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from mosaic.core.json_object import JSON_OBJECT
from mosaic.core.json_value import JsonValue

from .model import Recipe

__all__ = [
    "load_recipe",
    "pipelines_root",
    "read_json",
    "read_recipe_document",
    "recipe_path",
]


def pipelines_root(base_dir: Path | str) -> Path:
    """``<base_dir>/.mosaic/pipelines``: the recipes applied to this dataset."""
    return Path(base_dir) / ".mosaic" / "pipelines"


def recipe_path(base_dir: Path | str, digest: str) -> Path:
    """Where the recipe with this digest sits once copied in."""
    return pipelines_root(base_dir) / f"{digest}.json"


def read_json(path: Path, what: str) -> object:
    """Read one JSON document, naming the file when it does not parse.

    Raises:
        FileNotFoundError: There is no file there, named as a *what* file.
        ValueError: It is not valid JSON.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"no {what} file at {path}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def read_recipe_document(path: Path | str) -> dict[str, JsonValue]:
    """Read a recipe file as the JSON object it holds, without validating it.

    The bytes a saved recipe holds are what its file name digests. A reader
    that verifies a file against its name needs the document as written, before
    pydantic fills defaults or normalizes a value.

    Raises:
        FileNotFoundError: There is no file there.
        ValueError: It is not valid JSON, or not a JSON object.
    """
    return JSON_OBJECT.validate_python(read_json(Path(path), "recipe"))


def load_recipe(path: Path | str) -> Recipe:
    """Read a recipe file.

    Raises:
        FileNotFoundError: There is no file there.
        ValueError: It is not valid JSON, or not a valid recipe. A
            ``schema_version`` newer than this mosaic understands is refused
            instead of read under the wrong rules.
    """
    return Recipe.model_validate(read_recipe_document(path))
