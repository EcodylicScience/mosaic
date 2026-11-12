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
    max_workers: 8          # most entries at once; unset: physical cores
    allow_process_mode: true    # false: refuse process workers
    min_free_memory_fraction: 0.10   # share of host memory a parallel run keeps free
    gpu_cards: [0, 1]       # cards GPU steps may lease; unset leases every visible card
    dataset_roots: []       # where manage_dataset may create datasets, served too
    source_roots: []        # where manage_dataset may declare sources
    recipe_roots: []        # where a recipe argument {path: ...} may be read

A relative path is read from the file's directory. :func:`served_manifests`
joins the file's datasets with the ones named on the command line.
:data:`MCP_SDK` declares the MCP SDK as the ``mcp`` extra, which the server
loads. The module imports neither the SDK nor pandas.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BeforeValidator, Field, ValidationError

from mosaic.core.pipeline.graph.model import CleanupPolicy
from mosaic.core.pipeline.parallelism import (
    DEFAULT_MIN_FREE_MEMORY_FRACTION,
    MAX_WORKERS_CEILING,
    ParallelLimits,
    default_max_workers,
)
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
"""The feature runs that ``submit_pipeline`` removes with ``cleanup``.

Superseded ones, and with ``retire`` the runs of retired recipes.
"""


def _index_as_name(value: object) -> object:
    """Turn a non-negative YAML integer into the card name it writes.

    A ``bool`` is an ``int`` to Python and stays as it is, so the field refuses it.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    if value < 0:
        msg = "a card index is not negative"
        raise ValueError(msg)
    return str(value)


type CardName = Annotated[str, BeforeValidator(_index_as_name)]
"""A card as ``nvidia-smi`` lists it: an index or a UUID."""


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
            ``cleanup: superseded``, and with ``retire`` the runs that retired
            recipes wrote.
        cleanup: Which superseded runs a cleanup keeps by recency and age. The
            operator sets it here; no tool call does.
        max_workers: The most entries a parallelizable feature step computes at
            once, and the count a submission that names none receives. 1 computes
            them in turn. A submission may ask for fewer and never for more.
            Unset: the host's physical cores, at most 32.
        allow_process_mode: Whether a submission may ask for ``process`` workers,
            which each hold an interpreter and their own copy of an entry's
            tables. A submission that names no mode receives ``thread``. Set it
            false on a host whose memory cannot hold that per worker.
        min_free_memory_fraction: The share of the host's memory that a run with
            several workers keeps available. The run waits for its in-flight
            entries while the share is lower, and fails when it is still lower
            with none in flight, ending in a state a later run resumes. 0 turns
            the check off.
        gpu_cards: The cards the server's drivers may lease to GPU steps, as
            ``nvidia-smi`` indexes (PCI bus order, which ``CUDA_VISIBLE_DEVICES``
            integers follow under ``CUDA_DEVICE_ORDER=PCI_BUS_ID``) or UUIDs, among the
            cards ``CUDA_VISIBLE_DEVICES`` leaves visible. An index may be written as a
            YAML integer. ``None`` leases every visible card, and the server refuses to
            start when a name is not visible. It belongs to the machine and is part of
            no recipe, request or run id.
        dataset_roots: Directories inside which ``manage_dataset`` may create a
            dataset. The datasets below them are served, as ``discover``
            serves them, and one created while the server runs is served at
            once. Empty: no dataset is created.
        source_roots: Directories inside which ``manage_dataset`` may declare a
            source. Keep held-out data outside them. Empty: no source is
            declared.
        recipe_roots: Directories inside which a recipe argument named
            ``{"path": ...}`` may be read. A relative path is read from the
            first. Empty: a recipe is sent whole or named by digest or name.
    """

    datasets: list[Path] = Field(default_factory=list)
    discover: list[Path] = Field(default_factory=list)
    allow_managed_writes: bool = False
    allow_cleanup: list[CleanupClass] = Field(default_factory=list)
    cleanup: CleanupPolicy = Field(default_factory=lambda: CleanupPolicy())
    max_workers: int = Field(
        default_factory=default_max_workers, ge=1, le=MAX_WORKERS_CEILING
    )
    allow_process_mode: bool = True
    min_free_memory_fraction: float = Field(
        default=DEFAULT_MIN_FREE_MEMORY_FRACTION, ge=0.0, lt=1.0
    )
    gpu_cards: list[CardName] | None = Field(default=None, min_length=1)
    dataset_roots: list[Path] = Field(default_factory=list)
    source_roots: list[Path] = Field(default_factory=list)
    recipe_roots: list[Path] = Field(default_factory=list)

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
            "dataset_roots": [
                (base / user_path(given)).resolve() for given in config.dataset_roots
            ],
            "source_roots": [
                (base / user_path(given)).resolve() for given in config.source_roots
            ],
            "recipe_roots": [
                (base / user_path(given)).resolve() for given in config.recipe_roots
            ],
        }
    )


def served_manifests(datasets: Iterable[Path], config: ServerConfig) -> list[Path]:
    """Return the manifests to serve: *datasets*, the file's, then those discovered.

    The datasets below ``dataset_roots`` are discovered as those below
    ``discover`` are. A dataset root that does not exist yet holds none, and
    ``manage_dataset`` makes it when it creates a dataset there. A ``discover``
    directory that does not exist raises.

    A dataset named twice is served once, which ``LocalBackend`` decides by uuid.

    Args:
        datasets: The manifests or dataset directories named on the command line.
        config: The configuration file's contents.
    """
    from .local import discovered_manifests

    manifests = [*datasets, *config.datasets]
    for root in config.discover:
        manifests.extend(discovered_manifests(root))
    for root in config.dataset_roots:
        if root.is_dir():
            manifests.extend(discovered_manifests(root))
    return manifests
