"""mosaic's MCP server, for one client over standard streams or one user over HTTP.

:func:`build_server` binds the shared tools, resources and prompts to a
:class:`~mosaic.agent.mcp.local.LocalBackend`. :func:`serve` runs them on the
process's standard streams, as ``mosaic agent serve`` does, and
:func:`serve_http` on streamable HTTP behind a bearer token, as ``mosaic agent
serve --http`` does. Importing this
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
    from mcp.server.transport_security import TransportSecuritySettings

    from mosaic.core.pipeline.graph.model import CleanupPolicy
    from mosaic.core.pipeline.parallelism import ParallelLimits

    from .config import CleanupClass
    from .http_auth import HttpAddress
    from .local import LocalBackend

__all__ = ["build_server", "serve", "serve_http"]

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


def serve_http(
    manifests: Sequence[Path],
    *,
    address: HttpAddress,
    token: str,
    allowed_hosts: Sequence[str] = (),
    tls_cert: Path | None = None,
    tls_key: Path | None = None,
    allow_model_confirm: bool = False,
    allow_managed: bool = False,
    allow_cleanup: frozenset[CleanupClass] = frozenset(),
    cleanup_policy: CleanupPolicy | None = None,
    parallel_limits: ParallelLimits | None = None,
) -> None:
    """Serve the datasets of *manifests* to one user over streamable HTTP.

    The endpoint is ``/mcp``. Every request must carry ``Authorization: Bearer
    <token>``. Sessions are stateful, so a client that declares elicitation is
    asked to confirm a costly call during the call itself. Returns when the
    process is interrupted.

    Args:
        manifests: The manifest files, or the directories holding them.
        address: The host and port to listen on.
        token: The bearer token every request must carry.
        allowed_hosts: The host names, beside a loopback address, that a
            request's ``Host`` header may name. Empty on a non-loopback address
            turns the DNS rebinding check off.
        tls_cert: A certificate file to serve HTTPS with, beside *tls_key*.
        tls_key: The certificate's private key file.
        allow_model_confirm: Whether a client that cannot ask the user may pass
            ``confirm: true`` set by its model. Off, such a call is refused
            with ``confirmation_unavailable``.
        allow_managed: As for :func:`serve`.
        allow_cleanup: As for :func:`serve`.
        cleanup_policy: As for :func:`serve`.
        parallel_limits: As for :func:`serve`.

    Raises:
        MissingOptionalDependency: The ``mcp`` extra is not installed.
    """
    _ = MCP_SDK.load()
    import uvicorn

    from ._binding import bind_server
    from .http_auth import BearerGuard
    from .local import LocalBackend

    backend = LocalBackend(
        tuple(manifests),
        allow_managed=allow_managed,
        allow_cleanup=allow_cleanup,
        cleanup_policy=cleanup_policy,
        parallel_limits=parallel_limits,
    )
    server = bind_server(backend, model_confirm=allow_model_confirm)
    app = server.streamable_http_app(
        transport_security=_transport_security(address, allowed_hosts),
        host=address.host,
    )
    config = uvicorn.Config(
        BearerGuard(app, token),
        host=address.host,
        port=address.port,
        ssl_certfile=None if tls_cert is None else str(tls_cert),
        ssl_keyfile=None if tls_key is None else str(tls_key),
        log_level="info",
        # A client's address is the peer's, never a header it could forge.
        proxy_headers=False,
    )
    _LOG.info(
        "serving %d dataset manifest(s) over HTTP at %s:%d/mcp",
        len(manifests),
        address.host,
        address.port,
    )
    uvicorn.Server(config).run()


def _transport_security(
    address: HttpAddress, allowed_hosts: Sequence[str]
) -> TransportSecuritySettings | None:
    """The DNS rebinding check: the hosts a request may name, or ``None`` for none.

    A loopback address is checked against the loopback names and the address
    itself, beside *allowed_hosts*. Any other address is checked against
    *allowed_hosts* alone, and not checked when there are none.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    from .http_auth import is_loopback

    hosts = [*allowed_hosts]
    if is_loopback(address.host):
        bound = f"[{address.host}]" if ":" in address.host else address.host
        hosts.extend(
            name
            for name in (bound, "127.0.0.1", "localhost", "[::1]")
            if name not in hosts
        )
    if not hosts:
        _LOG.warning(
            "no --allowed-host: requests to %s are not checked for DNS "
            "rebinding; name the host clients use",
            address.host,
        )
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[pattern for host in hosts for pattern in (host, f"{host}:*")],
        allowed_origins=[
            f"{scheme}://{host}{port}"
            for host in hosts
            for scheme in ("http", "https")
            for port in ("", ":*")
        ],
    )
