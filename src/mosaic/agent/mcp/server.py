"""mosaic's MCP server over standard input and output, for one local client.

:func:`build_server` binds the shared tools, resources and prompts to a
:class:`~mosaic.agent.mcp.local.LocalBackend`, and :func:`serve` runs them on
the process's standard streams, as ``mosaic agent serve`` does. Importing this
module imports no MCP SDK. Each function loads it through
:data:`~mosaic.agent.mcp.config.MCP_SDK` first, so that without the ``mcp``
extra it raises :class:`~mosaic.optional_dependency.MissingOptionalDependency`
naming the extra. :mod:`~mosaic.agent.mcp._binding` holds the server itself.

Standard output carries the transport only. :func:`serve` enters the SDK's
``stdio_server``, which points descriptor 1 at standard error, before it imports
the binding and the library behind it. A print during those imports, a
library's print during a call and a child's output are written to standard
error.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from .config import MCP_SDK

if TYPE_CHECKING:
    from mcp.server import Server

    from mosaic.core.pipeline.graph.model import CleanupPolicy
    from mosaic.core.pipeline.parallelism import ParallelLimits

    from .config import CleanupClass

    from .local import LocalBackend

__all__ = ["build_server", "serve"]

_LOG = logging.getLogger(__name__)


def build_server(backend: LocalBackend) -> Server[None]:
    """Bind the shared tools, resources and prompts to *backend*.

    Args:
        backend: What answers the tools and reads the resources.

    Returns:
        The low-level server, which ``Server.run`` serves over a pair of
        streams and the SDK's ``Client`` connects to in process.

    Raises:
        MissingOptionalDependency: The ``mcp`` extra is not installed.
    """
    _ = MCP_SDK.load()
    from ._binding import bind_server

    return bind_server(backend)


def serve(
    manifests: Sequence[Path],
    *,
    allow_managed: bool = False,
    allow_cleanup: frozenset[CleanupClass] = frozenset(),
    cleanup_policy: CleanupPolicy | None = None,
    parallel_limits: ParallelLimits | None = None,
) -> None:
    """Serve the datasets of *manifests* to one client over standard input and output.

    Standard output is claimed for the transport before the server and the
    library behind it are imported, and the backend opens the datasets and
    builds the step-type catalog on a background thread. Returns when the
    client closes standard input.

    Args:
        manifests: The manifest files, or the directories holding them.
        allow_managed: Whether ``submit_pipeline`` and ``cancel_request`` may
            change a dataset whose manifest names a managing service.
        allow_cleanup: The artifacts the tools may remove. ``feature-runs``
            lets ``submit_pipeline`` remove superseded feature runs, and with
            ``retire`` the runs of retired recipes.
        cleanup_policy: Which superseded runs a cleanup keeps, the default
            policy when ``None``.
        parallel_limits: The parallelism a submission may ask for. One worker, in
            threads, when ``None``.

    Raises:
        MissingOptionalDependency: The ``mcp`` extra is not installed.
    """
    _ = MCP_SDK.load()
    import anyio

    anyio.run(
        partial(
            _serve,
            tuple(manifests),
            allow_managed=allow_managed,
            allow_cleanup=allow_cleanup,
            cleanup_policy=cleanup_policy,
            parallel_limits=parallel_limits,
        )
    )


async def _serve(
    manifests: tuple[Path, ...],
    *,
    allow_managed: bool,
    allow_cleanup: frozenset[CleanupClass],
    cleanup_policy: CleanupPolicy | None,
    parallel_limits: ParallelLimits | None,
) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        from ._binding import bind_server
        from .local import LocalBackend

        backend = LocalBackend(
            manifests,
            allow_managed=allow_managed,
            allow_cleanup=allow_cleanup,
            cleanup_policy=cleanup_policy,
            parallel_limits=parallel_limits,
        )
        server = bind_server(backend)
        _LOG.info("serving %d dataset manifest(s) over stdio", len(manifests))
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )
