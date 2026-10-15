"""What ``mosaic agent serve`` reads before it starts: its datasets and its SDK.

:class:`ServerConfig` is the YAML file that ``--config`` names::

    datasets:               # manifests, or the directories that hold them
      - ~/data/study1
    discover:               # directories searched for datasets, three levels deep
      - ~/data/projects
    allow_managed_writes: false

A relative path is read from the file's directory. :func:`served_manifests`
joins the file's datasets with the ones named on the command line.
:data:`MCP_SDK` declares the MCP SDK as the ``mcp`` extra, which the server
loads. The module imports neither the SDK nor pandas.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

from pydantic import Field, ValidationError

from mosaic.core.strict_model import StrictModel
from mosaic.optional_dependency import OptionalModule

__all__ = [
    "MCP_SDK",
    "ServerConfig",
    "ServerConfigInvalid",
    "read_server_config",
    "served_manifests",
]

MCP_SDK: Final = OptionalModule(
    module="mcp", extra="mcp", purpose="the MCP server, mosaic agent serve"
)
"""The MCP SDK, which ``mosaic.agent.mcp.server`` loads before binding the server."""


class ServerConfig(StrictModel):
    """The datasets that a server serves and whether it may change a managed one.

    Attributes:
        datasets: Manifest files, or the directories that hold them.
        discover: Directories searched for datasets, as ``discover_datasets``
            searches them.
        allow_managed_writes: Whether ``submit_pipeline`` and ``cancel_request``
            may change a dataset whose manifest names a managing service.
    """

    datasets: list[Path] = Field(default_factory=list)
    discover: list[Path] = Field(default_factory=list)
    allow_managed_writes: bool = False


class ServerConfigInvalid(ValueError):
    """A server configuration file that does not read or does not fit the model."""


def read_server_config(path: Path) -> ServerConfig:
    """Read the configuration file *path*, its relative paths read from its directory.

    Raises:
        ServerConfigInvalid: The file does not read, is not YAML, or does not fit
            :class:`ServerConfig`.
    """
    import yaml

    from mosaic.user_paths import user_path

    try:
        document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        msg = f"cannot read the server configuration {path}: {error}"
        raise ServerConfigInvalid(msg) from error
    try:
        config = ServerConfig.model_validate({} if document is None else document)
    except ValidationError as error:
        msg = f"the server configuration {path} is invalid: {error}"
        raise ServerConfigInvalid(msg) from error
    base = path.resolve().parent
    return config.model_copy(
        update={
            "datasets": [base / user_path(given) for given in config.datasets],
            "discover": [base / user_path(given) for given in config.discover],
        }
    )


def served_manifests(datasets: Iterable[Path], config: ServerConfig) -> list[Path]:
    """Return the manifests to serve: *datasets*, the file's, then those discovered.

    A dataset named twice is served once, which ``LocalBackend`` decides by uuid.

    Args:
        datasets: The manifests or dataset directories named on the command line.
        config: The configuration file's contents.
    """
    from .local import discovered_manifests

    manifests = [*datasets, *config.datasets]
    for root in config.discover:
        manifests.extend(discovered_manifests(root))
    return manifests
