"""A local stand-in for Hugging Face, and the small examples it serves.

:class:`ExampleServer` is a ``ThreadingHTTPServer`` on ``127.0.0.1`` that answers the
way Hugging Face does for a dataset repository: an archive's ``resolve`` URL answers
302 with an absolute ``Location`` on a second path of the same server, standing in for
the CDN, and ``SHA256SUMS`` answers 307 with a relative ``Location``. Both final paths
honor ``Range``. :meth:`ExampleServer.script` queues replies for a final path that
are served before the file, which is how a test makes a response a 429, a 500, a cut
body or a 200 to a ``Range`` request. :attr:`ExampleServer.requests` records every
request.

:func:`example_fixture` builds a small :class:`~mosaic.examples.catalog.Example`
whose sizes are those of the generated archives, with the archives' bytes and a matching
``SHA256SUMS`` body, and :func:`tar_gz` builds one archive. :func:`tar_gz_with` adds
members that are not regular files, which :func:`symlink_member`,
:func:`hardlink_member` and :func:`device_member` make. :func:`serve_example`
publishes a fixture on a server, and :func:`serve_example_server` runs the server.
The ``example_server`` fixture of ``tests/conftest.py`` starts one and points
``HF_ENDPOINT`` at it.
"""

from __future__ import annotations

import hashlib
import io
import socket
import struct
import tarfile
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from mosaic.examples.catalog import Example, ExampleArchive

__all__ = [
    "REVISION",
    "ExampleFixture",
    "ExampleServer",
    "GeneratedArchive",
    "RecordedRequest",
    "Reply",
    "archive_path",
    "cdn_path",
    "checksums_path",
    "example_fixture",
    "serve_example",
    "serve_example_server",
    "device_member",
    "hardlink_member",
    "sums_text",
    "symlink_member",
    "tar_gz",
    "tar_gz_with",
]

REVISION: Final = "0123456789abcdef0123456789abcdef01234567"
"""The revision of the generated example."""

REPOSITORY: Final = "EcodylicScience/mosaic-example-tiny"
"""The repository of the generated example."""


@dataclass(frozen=True)
class Reply:
    """A scripted response.

    Attributes:
        status: The status code.
        body: The body to send.
        headers: Extra response headers.
        cut_after: When set, the headers declare the whole body but only this many
            bytes are sent before the connection closes.
        reset: Whether the connection ends with a reset instead of a close, which the
            client sees as a connection error while it reads the body.
    """

    status: int = 200
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    cut_after: int | None = None
    reset: bool = False


@dataclass(frozen=True)
class RecordedRequest:
    """A request the server received.

    Attributes:
        path: The request path with its query.
        range: The ``Range`` header, or ``None``.
        user_agent: The ``User-Agent`` header, or ``None``.
    """

    path: str
    range: str | None
    user_agent: str | None


@dataclass(frozen=True)
class GeneratedArchive:
    """One generated archive of an example.

    Attributes:
        file: The file name in the repository.
        unpacks_to: The top-level directory its members lie under.
        data: The bytes of the ``.tar.gz``.
        dataset: Whether the directory is a mosaic dataset.
    """

    file: str
    unpacks_to: str
    data: bytes
    dataset: bool


@dataclass(frozen=True)
class ExampleFixture:
    """A generated example with the bytes the server serves for it.

    Attributes:
        example: The catalog entry, with the real sizes of the archives.
        archives: Each archive's bytes, keyed by file name.
        sums: The ``SHA256SUMS`` body.
    """

    example: Example
    archives: Mapping[str, bytes]
    sums: str


def tar_gz(members: Mapping[str, bytes]) -> bytes:
    """Build a ``.tar.gz`` holding regular files.

    Args:
        members: The file contents, keyed by member name.

    Returns:
        The bytes of the compressed archive.
    """
    return tar_gz_with(members, [])


def tar_gz_with(
    members: Mapping[str, bytes], extra: Sequence[tarfile.TarInfo]
) -> bytes:
    """Build a ``.tar.gz`` of regular files followed by members of other kinds.

    Args:
        members: The regular files' contents, keyed by member name.
        extra: Further members, such as :func:`symlink_member`'s, added after the
            files and carrying no data.

    Returns:
        The bytes of the compressed archive.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        for info in extra:
            archive.addfile(info)
    return buffer.getvalue()


def symlink_member(name: str, target: str) -> tarfile.TarInfo:
    """A symbolic link named *name* that points at *target*."""
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info


def hardlink_member(name: str, target: str) -> tarfile.TarInfo:
    """A hard link named *name* to the archive member or path *target*."""
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    return info


def device_member(name: str) -> tarfile.TarInfo:
    """A character device named *name*."""
    info = tarfile.TarInfo(name)
    info.type = tarfile.CHRTYPE
    return info


def sums_text(files: Mapping[str, bytes], *, omit: Sequence[str] = ()) -> str:
    """The body of a ``SHA256SUMS`` file for *files*, less the names in *omit*."""
    return "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in files.items()
        if name not in omit
    )


def example_fixture(
    archives: Sequence[GeneratedArchive] | None = None,
    *,
    name: str = "tiny",
    omit_from_sums: Sequence[str] = (),
) -> ExampleFixture:
    """Build an example from generated archives.

    Args:
        archives: The archives in fetch order. By default one dataset archive,
            ``tiny-tracks.tar.gz``, holding ``tiny-tracks/dataset.yaml`` and one
            track file.
        name: The example's name.
        omit_from_sums: Files left out of ``SHA256SUMS``.

    Returns:
        The example, the archives' bytes and the ``SHA256SUMS`` body.
    """
    if archives is None:
        archives = [
            GeneratedArchive(
                file="tiny-tracks.tar.gz",
                unpacks_to="tiny-tracks",
                data=tar_gz(
                    {
                        "tiny-tracks/dataset.yaml": b"name: tiny\n",
                        "tiny-tracks/tracks/a.txt": b"track a\n" * 1000,
                    }
                ),
                dataset=True,
            )
        ]
    example = Example(
        name=name,
        repository=REPOSITORY,
        revision=REVISION,
        license="cc0-1.0",
        summary="A generated example.",
        notebook="tiny.ipynb",
        archives=tuple(
            ExampleArchive(
                file=archive.file,
                unpacks_to=archive.unpacks_to,
                download_bytes=len(archive.data),
                dataset=archive.dataset,
            )
            for archive in archives
        ),
    )
    files = {archive.file: archive.data for archive in archives}
    return ExampleFixture(example, files, sums_text(files, omit=omit_from_sums))


class _Handler(BaseHTTPRequestHandler):
    """Answers a GET from the server's files, redirects and scripted replies."""

    def log_message(self, format: str, *args: object) -> None:
        """Keep the server quiet."""

    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, ExampleServer)
        server.record(self)
        server.answer(self)


class ExampleServer(ThreadingHTTPServer):
    """The scripted server, started by :func:`serve_example_server`."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[RecordedRequest] = []
        self._files: dict[str, bytes] = {}
        self._redirects: dict[str, tuple[int, str]] = {}
        self._scripts: dict[str, list[Reply]] = {}
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        """The server's base URL, which ``HF_ENDPOINT`` is set to."""
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"

    def add_file(self, path: str, data: bytes) -> None:
        """Serve *data* at *path*, honoring ``Range``."""
        self._files[path] = data

    def add_redirect(self, path: str, status: int, location: str) -> None:
        """Answer *path* with *status* and *location*, which may be relative."""
        self._redirects[path] = (status, location)

    def script(self, path: str, *replies: Reply) -> None:
        """Serve *replies* in order at the file path *path*, before its file."""
        with self._lock:
            self._scripts.setdefault(path, []).extend(replies)

    def count(self, path_prefix: str) -> int:
        """The number of requests whose path starts with *path_prefix*."""
        return sum(
            1 for request in self.requests if request.path.startswith(path_prefix)
        )

    def record(self, handler: BaseHTTPRequestHandler) -> None:
        """Log the request in :attr:`requests`."""
        with self._lock:
            self.requests.append(
                RecordedRequest(
                    path=handler.path,
                    range=handler.headers.get("Range"),
                    user_agent=handler.headers.get("User-Agent"),
                )
            )

    def answer(self, handler: BaseHTTPRequestHandler) -> None:
        """Write the response to *handler*'s request."""
        path = handler.path.split("?", 1)[0]
        if path in self._redirects:
            status, location = self._redirects[path]
            handler.send_response(status)
            handler.send_header("Location", location)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        with self._lock:
            queued = self._scripts.get(path)
            reply = queued.pop(0) if queued else None
        if reply is not None:
            self._send(handler, reply)
        elif path in self._files:
            self._send_file(handler, self._files[path])
        else:
            self._send(handler, Reply(404, b"not found"))

    def _send_file(self, handler: BaseHTTPRequestHandler, data: bytes) -> None:
        requested = handler.headers.get("Range")
        if requested is None:
            self._send(handler, Reply(200, data))
            return
        start = int(requested.removeprefix("bytes=").split("-", 1)[0])
        if start >= len(data):
            self._send(
                handler, Reply(416, headers={"Content-Range": f"bytes */{len(data)}"})
            )
            return
        content_range = f"bytes {start}-{len(data) - 1}/{len(data)}"
        self._send(handler, Reply(206, data[start:], {"Content-Range": content_range}))

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, reply: Reply) -> None:
        handler.send_response(reply.status)
        for key, value in reply.headers.items():
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(reply.body)))
        handler.end_headers()
        body = reply.body if reply.cut_after is None else reply.body[: reply.cut_after]
        handler.wfile.write(body)
        if reply.cut_after is not None:
            handler.wfile.flush()
            if reply.reset:
                handler.connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
            handler.close_connection = True


@contextmanager
def serve_example_server() -> Generator[ExampleServer]:
    """Run an :class:`ExampleServer` on a thread and stop it on exit."""
    server = ExampleServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def archive_path(example: Example, file: str) -> str:
    """The ``resolve`` path under which the repository serves *file*."""
    return f"/datasets/{example.repository}/resolve/{example.revision}/{file}"


def cdn_path(example: Example, file: str) -> str:
    """The path the ``resolve`` URL of *file* redirects to, standing in for the CDN."""
    return f"/cdn/{example.repository}/{file}"


def checksums_path(example: Example) -> str:
    """The path that the ``SHA256SUMS`` ``resolve`` URL redirects to, less its query."""
    return (
        f"/api/resolve-cache/datasets/{example.repository}"
        f"/{example.revision}/SHA256SUMS"
    )


def serve_example(server: ExampleServer, fixture: ExampleFixture) -> None:
    """Publish *fixture* on *server* as Hugging Face does.

    Each archive's ``resolve`` URL answers 302 with an absolute ``Location`` at
    :func:`cdn_path`. ``SHA256SUMS`` answers 307 with a relative ``Location``.
    """
    example = fixture.example
    for file, data in fixture.archives.items():
        server.add_file(cdn_path(example, file), data)
        server.add_redirect(
            archive_path(example, file), 302, server.url + cdn_path(example, file)
        )
    server.add_file(checksums_path(example), fixture.sums.encode())
    server.add_redirect(
        archive_path(example, "SHA256SUMS"),
        307,
        checksums_path(example) + "?download=true",
    )
