"""The file a fitting feature's ``model`` param names, or a refusal.

A feature that fits and takes a ``model`` applies that model instead. Its run id
then hashes the model's run and no scope, so a step over more sequences is the
same run. That holds only while a missing model stops the run: a step that fitted
a new network in its place would store it under an id that names the old one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

__all__ = ["named_model_file"]


class _ModelReference(Protocol):
    """The three fields of an ``ArtifactSpec`` that a refusal names."""

    @property
    def feature(self) -> str: ...

    @property
    def run_id(self) -> str | None: ...

    @property
    def pattern(self) -> str: ...


def named_model_file(
    feature: str, model: _ModelReference, artifact_paths: Mapping[str, Path]
) -> Path:
    """Return the file that ``model`` resolved to.

    Args:
        feature: The name of the feature that applies the model, for the message.
        model: The ``model`` param, which the engine has pinned to a run.
        artifact_paths: The files the engine resolved, by params field.

    Returns:
        The path of the model file.

    Raises:
        FileNotFoundError: The named run holds no such file. The step does not
            fit a model of its own in its place.
    """
    path = artifact_paths.get("model")
    if path is None:
        msg = (
            f"[{feature}] The model {model.feature!r} run {model.run_id!r} holds no "
            f"{model.pattern!r}. A step given a model applies it and fits nothing, "
            f"so run the fit that writes it first."
        )
        raise FileNotFoundError(msg)
    return path
