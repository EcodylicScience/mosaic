"""The bearer token of mosaic's MCP server over HTTP, and the address it binds.

``mosaic agent serve --http`` serves one user on a trusted network. Every
request carries ``Authorization: Bearer <token>``, compared in constant time
with the token the operator keeps in a file only they can read. The SDK's OAuth
settings describe an authorization server, which a static token does not have,
so the check is a small ASGI wrapper instead.

Imports the standard library and :mod:`mosaic.core.narrowing` only.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import stat
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mosaic.core.narrowing import is_object_list, is_object_tuple

__all__ = [
    "MIN_TOKEN_CHARS",
    "BearerGuard",
    "HttpAddress",
    "TokenFileInvalid",
    "is_loopback",
    "parse_http_address",
    "read_token_file",
]

MIN_TOKEN_CHARS: Final = 32
"""The fewest characters a token has: about 192 bits as URL-safe base64."""

type _Scope = MutableMapping[str, object]
type _Message = MutableMapping[str, object]
type _Receive = Callable[[], Awaitable[_Message]]
type _Send = Callable[[_Message], Awaitable[None]]
type AsgiApp = Callable[[_Scope, _Receive, _Send], Awaitable[None]]
"""An ASGI application, as uvicorn calls one."""


class TokenFileInvalid(ValueError):
    """A token file that is missing, readable by others, or too short."""


@dataclass(frozen=True, slots=True)
class HttpAddress:
    """Where the server listens.

    Attributes:
        host: The host name or IP address, without brackets.
        port: The TCP port.
    """

    host: str
    port: int


def parse_http_address(text: str) -> HttpAddress:
    """Read ``HOST:PORT``, with an IPv6 host in brackets (``[::1]:8765``).

    Raises:
        ValueError: The text names no host or no port from 1 to 65535.
    """
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        msg = f"{text!r} is not HOST:PORT, such as 127.0.0.1:8765"
        raise ValueError(msg)
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
        if not host:
            msg = f"{text!r} names no host between the brackets"
            raise ValueError(msg)
    elif ":" in host:
        msg = f"{text!r}: write an IPv6 host in brackets, as [::1]:8765"
        raise ValueError(msg)
    if not port.isdigit() or not 0 < int(port) < 65536:
        msg = f"{text!r}: the port is a number from 1 to 65535"
        raise ValueError(msg)
    return HttpAddress(host=host, port=int(port))


def is_loopback(host: str) -> bool:
    """Whether *host* is this machine only: ``localhost`` or a loopback address."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def read_token_file(path: Path) -> str:
    """Return the token that *path* holds, stripped of surrounding whitespace.

    Raises:
        TokenFileInvalid: The file does not read, other users may read or
            write it (it must be mode 0600 or stricter), or the token is
            shorter than :data:`MIN_TOKEN_CHARS`.
    """
    try:
        mode = path.stat().st_mode
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"the token file {path} does not read: {error.strerror or error}"
        raise TokenFileInvalid(msg) from error
    except UnicodeDecodeError as error:
        msg = f"the token file {path} is not UTF-8 text"
        raise TokenFileInvalid(msg) from error
    if os.name == "posix" and stat.S_IMODE(mode) & 0o077:
        msg = (
            f"the token file {path} is mode {stat.S_IMODE(mode):o}, so other users "
            f"may read it. Run: chmod 600 {path}"
        )
        raise TokenFileInvalid(msg)
    token = text.strip()
    if len(token) < MIN_TOKEN_CHARS or any(char.isspace() for char in token):
        msg = (
            f"the token in {path} must be one word of at least {MIN_TOKEN_CHARS} "
            f"characters. Make one with: python -c 'import secrets; "
            f"print(secrets.token_urlsafe(32))' > {path}"
        )
        raise TokenFileInvalid(msg)
    return token


class BearerGuard:
    """An ASGI wrapper that answers 401 to an HTTP request without the token.

    A lifespan event passes through, so the wrapped app starts and stops as it
    would alone.
    """

    def __init__(self, app: AsgiApp, token: str) -> None:
        self._app: AsgiApp = app
        self._expected: bytes = token.encode()

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        """Pass an authorized request to the app, and refuse any other."""
        if scope.get("type") == "http" and not self._authorized(scope):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b'Bearer realm="mosaic"'),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error":"unauthorized","error_description":'
                    b'"send Authorization: Bearer <token>"}',
                }
            )
            return
        await self._app(scope, receive, send)

    def _authorized(self, scope: _Scope) -> bool:
        headers = scope.get("headers")
        pairs: list[object] = (
            list(headers)
            if is_object_list(headers)
            else list(headers)
            if is_object_tuple(headers)
            else []
        )
        given = b""
        for pair in pairs:
            if is_object_tuple(pair) and len(pair) == 2 and pair[0] == b"authorization":
                value = pair[1]
                given = value if isinstance(value, bytes) else b""
                break
        scheme, _, credentials = given.partition(b" ")
        # The scheme is case-insensitive (RFC 7235), the token is not.
        return scheme.lower() == b"bearer" and hmac.compare_digest(
            credentials.strip(), self._expected
        )
