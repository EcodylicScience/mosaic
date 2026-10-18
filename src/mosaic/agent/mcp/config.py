"""What ``mosaic agent serve`` reads before it starts: its datasets and its SDK.

:class:`ServerConfig` is the YAML file that ``--config`` names::

    datasets:               # manifests, or the directories that hold them
      - ~/data/study1
    discover:               # directories searched for datasets, three levels deep
      - ~/data/projects
    allow_managed_writes: false
    allow_cleanup: []       # [feature-runs] lets submit_pipeline remove superseded runs
    cleanup:                # which superseded runs a cleanup keeps
      keep_newest: 1
      min_age_hours: 24
    max_workers: 1          # most entries a feature step may compute at once
    allow_process_mode: false   # let a submission ask for process workers
    min_free_memory_fraction: 0.10   # share of host memory a parallel run keeps free

A relative path is read from the file's directory. :func:`served_manifests`
joins the file's datasets with the ones named on the command line.
:data:`MCP_SDK` declares the MCP SDK as the ``mcp`` extra, which the server
loads. The module imports neither the SDK nor pandas.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, ValidationError

from mosaic.core.pipeline.graph.model import CleanupPolicy
from mosaic.core.pipeline.parallelism import MAX_WORKERS_CEILING, ParallelLimits
from mosaic.core.strict_model import StrictModel
from mosaic.optional_dependency import OptionalModule

__all__ = [
    "CLEANUP_FEATURE_RUNS",
    "MCP_SDK",
    "CleanupClass",
    "ServerConfig",
    "ServerConfigInvalid",
    "read_server_config",
    "served_manifests",
]

MCP_SDK: Final = OptionalModule(
    module="mcp", extra="mcp", purpose="the MCP server, mosaic agent serve"
)
"""The MCP SDK, which ``mosaic.agent.mcp.server`` loads before binding the server."""


type CleanupClass = Literal["feature-runs"]
"""A class of artifact a server may let its tools remove."""

CLEANUP_FEATURE_RUNS: Final[CleanupClass] = "feature-runs"
"""The superseded feature runs that ``submit_pipeline`` removes with ``cleanup``."""


class ServerConfig(StrictModel):
    """The datasets that a server serves and whether it may change a managed one.

    Attributes:
        datasets: Manifest files, or the directories that hold them.
        discover: Directories searched for datasets, as ``discover_datasets``
            searches them.
        allow_managed_writes: Whether ``submit_pipeline`` and ``cancel_request``
            may change a dataset whose manifest names a managing service.
        allow_cleanup: The artifacts the tools may remove. ``feature-runs``
            lets ``submit_pipeline`` remove superseded feature runs with
            ``cleanup: superseded``.
        cleanup: Which superseded runs a cleanup keeps by recency and age. The
            operator sets it here; no tool call does.
        max_workers: The most entries a parallelizable feature step computes at
            once, and the count a submission that names none receives. 1 computes
            them in turn. A submission may ask for fewer and never for more.
        allow_process_mode: Whether a submission may ask for ``process`` workers,
            which are faster for work that does not release the interpreter lock
            and use much more memory than threads. A submission that names no
            mode receives ``thread``.
        min_free_memory_fraction: The share of the host's memory that a run with
            several workers keeps available. The run waits for its in-flight
            entries while the share is lower, and fails when it is still lower
            with none in flight, ending in a state a later run resumes. 0 turns
            the check off.
    """

    datasets: list[Path] = Field(default_factory=list)
    discover: list[Path] = Field(default_factory=list)
    allow_managed_writes: bool = False
    allow_cleanup: list[CleanupClass] = Field(default_factory=list)
    cleanup: CleanupPolicy = Field(default_factory=lambda: CleanupPolicy())
    max_workers: int = Field(default=1, ge=1, le=MAX_WORKERS_CEILING)
    allow_process_mode: bool = False
    min_free_memory_fraction: float = Field(default=0.10, ge=0.0, lt=1.0)

    @property
    def parallel_limits(self) -> ParallelLimits:
        """The limits that a submission's parallelism is held to."""
        return ParallelLimits(
            max_workers=self.max_workers,
            allow_process_mode=self.allow_process_mode,
            min_free_memory_fraction=self.min_free_memory_fraction,
        )


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
