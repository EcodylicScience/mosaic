"""Fetch an example: download its archives, check them, unpack them and record it.

:func:`fetch_example` is the entry point. It downloads each archive that is not yet
in the directory, unpacks it under the checks of :func:`_extract`, and writes a
completion record naming the archives that are in place. A later fetch of the same
example reuses what the record names and makes no request.

An archive is read from ``<endpoint>/datasets/<repository>/resolve/<revision>/<file>``
with the standard library's ``urllib``, so a notebook can fetch before anything
imports pandas. :func:`download_archive` streams one archive to a ``.part`` file and
verifies its size and SHA-256 against the repository's ``SHA256SUMS`` at the pinned
revision, which :func:`fetch_checksums` reads. The checksum detects a corrupt or
truncated transfer. Read from the same host as the archive, it does not defend
against that host.

The download resumes and retries:

* An existing ``.part`` is requested from its size on with a ``Range`` header. A 206
  appends, a 200 restarts, and a 416 means the part is whole and goes to
  verification.
* Hugging Face answers 429 with ``Retry-After``. The wait is that many seconds, at
  most 300, or 60 without the header.
* A 5xx, a reset connection, a TLS or other socket error, a timeout or a short body
  is retried after 2, 4, 8 and 16 seconds, resuming from the bytes received.
* A request gets five attempts, the rate-limited ones included. Any other 4xx raises
  at once.

The directory holds the unpacked archives, ``.mosaic-example.json`` (the completion
record), ``.mosaic-example.lock`` (present while a fetch runs) and
``.mosaic-example-partial/`` (downloads and staging, removed when the fetch ends). An
archive is unpacked into staging and moved into place with one ``os.rename``, and
only then recorded, so an interruption never leaves a recorded directory that is
incomplete.

This module imports the standard library, ``mosaic.byte_sizes``, ``mosaic.user_paths``
and ``mosaic.version`` only.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import IO, Final, Literal, TypeGuard, TypeVar
from urllib.parse import urljoin, urlsplit

from mosaic.byte_sizes import format_size
from mosaic.examples import catalog as catalog_module
from mosaic.examples.catalog import Example, ExampleArchive
from mosaic.user_paths import user_path
from mosaic.version import installed_version

__all__ = [
    "ArchiveWithoutDataset",
    "ChecksumMismatch",
    "ChecksumMissing",
    "DownloadFailed",
    "ExampleFetchError",
    "FetchConflict",
    "FetchedExample",
    "RedirectHandler",
    "UnknownExample",
    "UnsafeArchive",
    "archive_url",
    "default_directory",
    "download_archive",
    "fetch_checksums",
    "fetch_example",
    "find_example",
    "parse_checksums",
]

DEFAULT_ENDPOINT: Final = "https://huggingface.co"
"""The Hugging Face endpoint, used when ``HF_ENDPOINT`` is not set."""

CHECKSUMS_FILE: Final = "SHA256SUMS"
"""The file in each repository that lists its files' SHA-256 digests."""

_TIMEOUT_SECONDS: Final = 60
_ATTEMPTS: Final = 5
_BACKOFF_SECONDS: Final = (2, 4, 8, 16)
_MAX_RETRY_AFTER_SECONDS: Final = 300
_DEFAULT_RETRY_AFTER_SECONDS: Final = 60
_CHUNK_BYTES: Final = 64 * 1024
_TERMINAL_INTERVAL_SECONDS: Final = 0.5
_CHECKSUM_LINE: Final = re.compile(r"([0-9a-fA-F]{64}) [ *](.+)")

RECORD_FILE: Final = ".mosaic-example.json"
"""The completion record, in the example's directory."""

LOCK_FILE: Final = ".mosaic-example.lock"
"""Present in the example's directory while a fetch runs."""

PARTIAL_DIRECTORY: Final = ".mosaic-example-partial"
"""Downloads and staging, removed when a fetch ends."""

_STAGING_DIRECTORY: Final = "unpack"

_NETWORK_ERRORS: Final = (OSError, http.client.HTTPException)
"""What a request or a read of the response can raise when the network fails.

``OSError`` covers ``URLError``, timeouts, resets and ``ssl.SSLError``. Only a call
that touches the response uses it, never one that writes a file.
"""

_sleep: Callable[[float], None] = time.sleep
"""Waits between attempts. Tests replace it to record the waits."""

_monotonic: Callable[[], float] = time.monotonic
"""The clock that spaces a terminal's progress updates."""

_Result = TypeVar("_Result")


class ExampleFetchError(Exception):
    """Fetching an example failed."""


class UnknownExample(ExampleFetchError):
    """The name is not in the catalog."""


class FetchConflict(ExampleFetchError):
    """The directory holds something fetching must not replace, or a fetch runs."""


class DownloadFailed(ExampleFetchError):
    """A request failed for good: a refused redirect, a 4xx, or five failed attempts."""


class ChecksumMissing(ExampleFetchError):
    """``SHA256SUMS`` does not parse or has no line for the file."""


class ChecksumMismatch(ExampleFetchError):
    """The downloaded bytes differ in size or digest from the repository's record."""


class UnsafeArchive(ExampleFetchError):
    """An archive member would write outside its top-level directory."""


class ArchiveWithoutDataset(ExampleFetchError):
    """The dataset archive does not hold a mosaic dataset."""


def _endpoint() -> str:
    """The Hugging Face endpoint, read when called so a mirror or a test can set it."""
    return os.environ.get("HF_ENDPOINT", "").rstrip("/") or DEFAULT_ENDPOINT


def default_directory(name: str) -> Path:
    """The directory an example unpacks into when none is given.

    Args:
        name: The example's catalog name.

    Returns:
        ``~/mosaic-examples/<name>``, with the ``~`` expanded.
    """
    return user_path(f"~/mosaic-examples/{name}")


def archive_url(example: Example, file: str) -> str:
    """The URL at which the repository serves *file* at the example's revision.

    Args:
        example: The catalog entry.
        file: A file name in its repository.

    Returns:
        ``<endpoint>/datasets/<repository>/resolve/<revision>/<file>``, with the
        endpoint from ``HF_ENDPOINT`` when that is set.
    """
    return (
        f"{_endpoint()}/datasets/{example.repository}/resolve/{example.revision}/{file}"
    )


class RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follows 301, 302, 303, 307 and 308 up to ten hops, keeping the request's headers.

    ``urllib`` resolves a relative ``Location`` against the request URL and copies
    the headers, ``Range`` included. This handler adds one rule: when the endpoint is
    ``https``, a redirect to anything but ``https`` raises :class:`DownloadFailed` at
    whatever hop it occurs. An ``http`` endpoint, a local mirror or a test's server,
    is followed as given.
    """

    def __init__(self, endpoint: str) -> None:
        """Remember whether the endpoint requires ``https`` of every hop."""
        super().__init__()
        self._require_https = urlsplit(endpoint).scheme == "https"

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Refuse a downgrade to ``http``, and otherwise build the next request."""
        target = urljoin(req.full_url, newurl)
        if self._require_https and urlsplit(target).scheme != "https":
            raise DownloadFailed(
                f"refusing the redirect from {req.full_url} to {target}: "
                "it does not use https"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _Transient(Exception):
    """One attempt failed in a way that another attempt may not.

    Attributes:
        reason: What failed, for the last-failure message.
        wait: Seconds to wait before the next attempt, or ``None`` for the backoff.
        rate_limited: Whether the server answered 429.
    """

    def __init__(
        self, reason: str, *, wait: float | None = None, rate_limited: bool = False
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.wait = wait
        self.rate_limited = rate_limited


class _RangeNotSatisfiable(Exception):
    """The server answered 416 to a ``Range`` request."""


def _retry_after_seconds(value: str | None) -> float:
    """The wait a ``Retry-After`` header asks for, a delta or an HTTP date, capped."""
    if value is None:
        return _DEFAULT_RETRY_AFTER_SECONDS
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return _DEFAULT_RETRY_AFTER_SECONDS
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        seconds = (date - datetime.now(timezone.utc)).total_seconds()
    return min(max(seconds, 0), _MAX_RETRY_AFTER_SECONDS)


def _where(example: Example, file: str) -> str:
    return f"{file} from {example.repository} at revision {example.revision}"


def _open(
    example: Example, file: str, headers: Mapping[str, str]
) -> http.client.HTTPResponse:
    """Open *file* of the example's repository for one attempt.

    Raises:
        _Transient: A 429, a 5xx or a network failure.
        _RangeNotSatisfiable: A 416 to a ``Range`` request.
        DownloadFailed: Any other 4xx, a refused redirect or too many redirects.
    """
    request = urllib.request.Request(
        archive_url(example, file),
        headers={"User-Agent": f"mosaic-behavior/{installed_version()}", **headers},
    )
    opener = urllib.request.build_opener(RedirectHandler(_endpoint()))
    try:
        response = opener.open(request, timeout=_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        status = error.code
        retry_after = error.headers.get("Retry-After") if error.headers else None
        error.close()
        if status == 429:
            raise _Transient(
                "HTTP 429",
                wait=_retry_after_seconds(retry_after),
                rate_limited=True,
            ) from error
        if status == 416 and "Range" in headers:
            raise _RangeNotSatisfiable from error
        if status >= 500:
            raise _Transient(f"HTTP {status}") from error
        if status == 404:
            raise DownloadFailed(
                f"{file} is not in {example.repository} at revision "
                f"{example.revision} (HTTP 404): the revision or the file is missing"
            ) from error
        raise DownloadFailed(
            f"HTTP {status} {error.reason} fetching {_where(example, file)}"
        ) from error
    except _NETWORK_ERRORS as error:
        raise _Transient(f"{type(error).__name__}: {error}") from error
    if not isinstance(response, http.client.HTTPResponse):
        raise DownloadFailed(f"unexpected response fetching {_where(example, file)}")
    return response


def _read_chunk(
    response: http.client.HTTPResponse, size: int | None = _CHUNK_BYTES
) -> bytes:
    """Read from *response*, turning a network failure into a retryable one.

    Args:
        response: The open response.
        size: Bytes to read, or ``None`` for the rest of the body.

    Raises:
        _Transient: The read failed, including by a TLS error or a reset.
    """
    try:
        return response.read(size)
    except _NETWORK_ERRORS as error:
        raise _Transient(f"{type(error).__name__}: {error}") from error


def _with_attempts(
    example: Example, file: str, attempt: Callable[[], _Result]
) -> _Result:
    """Run *attempt* up to five times, waiting between failed ones.

    Raises:
        DownloadFailed: Every attempt failed transiently.
    """
    last = ""
    for number in range(1, _ATTEMPTS + 1):
        try:
            return attempt()
        except _Transient as failure:
            last = failure.reason
            if number == _ATTEMPTS:
                break
            wait = (
                failure.wait
                if failure.wait is not None
                else _BACKOFF_SECONDS[number - 1]
            )
            if failure.rate_limited:
                print(
                    f"rate limited by Hugging Face; waiting {math.ceil(wait)} s",
                    file=sys.stderr,
                )
            _sleep(wait)
    raise DownloadFailed(
        f"could not fetch {_where(example, file)} after {_ATTEMPTS} attempts; "
        f"last failure: {last}"
    )


def parse_checksums(text: str, revision: str) -> dict[str, str]:
    """Parse a ``SHA256SUMS`` file.

    Args:
        text: The file's text, lines of ``<64 hex digits> <space or *><file>``.
        revision: The revision it was read at, for the error message.

    Returns:
        The lowercase digest of each file.

    Raises:
        ChecksumMissing: A line that is neither blank nor in that form.
    """
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ChecksumMissing(f"SHA256SUMS at {revision} does not parse")
        checksums[match[2]] = match[1].lower()
    return checksums


def fetch_checksums(example: Example) -> dict[str, str]:
    """Read the repository's ``SHA256SUMS`` at the example's revision.

    Args:
        example: The catalog entry.

    Returns:
        The lowercase SHA-256 digest of each file the repository lists.

    Raises:
        DownloadFailed: The request failed.
        ChecksumMissing: The file does not parse.
    """

    def attempt() -> bytes:
        with _open(example, CHECKSUMS_FILE, {}) as response:
            return _read_chunk(response, None)

    body = _with_attempts(example, CHECKSUMS_FILE, attempt)
    return parse_checksums(body.decode("utf-8", errors="replace"), example.revision)


class _Progress:
    """Progress lines on standard error for one archive.

    On a terminal a carriage-return line updates the percentage at most every half
    second. Elsewhere, as in Jupyter, a line is printed at each tenth.
    """

    def __init__(self, file: str, total: int, *, enabled: bool) -> None:
        self._file = file
        self._total = max(total, 1)
        self._enabled = enabled
        self._terminal = sys.stderr.isatty()
        self._last_tenth = 0
        self._last_update = float("-inf")
        self._open_line = False

    def begin(self, received: int) -> None:
        """Announce the archive and its size."""
        if not self._enabled:
            return
        print(f"fetching {self._file} ({format_size(self._total)})", file=sys.stderr)
        self._last_tenth = self._percent(received) // 10 * 10

    def restart(self) -> None:
        """Count from zero again, after the server restarted the download."""
        self._last_tenth = 0

    def update(self, received: int) -> None:
        """Report that *received* bytes have arrived."""
        if not self._enabled:
            return
        percent = self._percent(received)
        if self._terminal:
            now = _monotonic()
            if now - self._last_update >= _TERMINAL_INTERVAL_SECONDS:
                self._last_update = now
                self._write_line(percent)
        elif percent // 10 * 10 > self._last_tenth:
            self._last_tenth = percent // 10 * 10
            print(f"  {self._file} {self._last_tenth}%", file=sys.stderr)

    def close(self, received: int) -> None:
        """End a terminal line, showing the final percentage."""
        if self._enabled and self._terminal and self._open_line:
            self._write_line(self._percent(received))
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._open_line = False

    def _percent(self, received: int) -> int:
        return min(100, received * 100 // self._total)

    def _write_line(self, percent: int) -> None:
        sys.stderr.write(f"\r  {self._file} {percent}%")
        sys.stderr.flush()
        self._open_line = True


class _Transfer:
    """One archive's partial file, its running digest, and the attempts that grow it."""

    def __init__(
        self, example: Example, archive: ExampleArchive, part: Path, progress: _Progress
    ) -> None:
        self._example = example
        self._file = archive.file
        self._part = part
        self._progress = progress
        self.digest = hashlib.sha256()
        self.size = 0
        if part.exists():
            with part.open("rb") as existing:
                while chunk := existing.read(_CHUNK_BYTES):
                    self.digest.update(chunk)
                    self.size += len(chunk)

    def attempt(self) -> None:
        """Request the rest of the file and append it, or restart on a 200."""
        headers = {"Range": f"bytes={self.size}-"} if self.size else {}
        try:
            response = _open(self._example, self._file, headers)
        except _RangeNotSatisfiable:
            return
        with response:
            if response.status == 206:
                mode = "ab"
            else:
                mode = "wb"
                self.digest = hashlib.sha256()
                self.size = 0
                self._progress.restart()
            declared = response.getheader("Content-Length")
            expected = int(declared) if declared and declared.isdigit() else None
            written = 0
            with self._part.open(mode) as out:
                while chunk := _read_chunk(response):
                    out.write(chunk)
                    self.digest.update(chunk)
                    self.size += len(chunk)
                    written += len(chunk)
                    self._progress.update(self.size)
            if expected is not None and written < expected:
                raise _Transient(f"short body: {written} of {expected} bytes")


def download_archive(
    example: Example,
    archive: ExampleArchive,
    partial_directory: Path,
    checksums: Mapping[str, str],
    *,
    quiet: bool = False,
) -> Path:
    """Download one archive to ``<partial_directory>/<file>.part`` and verify it.

    An existing ``.part`` is resumed. The file stays after a successful verification
    for the caller to unpack and delete.

    Args:
        example: The catalog entry.
        archive: The archive to fetch.
        partial_directory: The directory for the ``.part`` file, created if missing.
        checksums: The repository's digests, from :func:`fetch_checksums`.
        quiet: Whether to leave standard error without progress lines. The notice
            that a 429 rate-limited the request is still printed.

    Returns:
        The path of the verified ``.part`` file.

    Raises:
        ChecksumMissing: *checksums* has no line for the file. Nothing is requested.
        DownloadFailed: The request failed for good.
        ChecksumMismatch: The size or the digest differs from the record. The
            ``.part`` file is deleted.
    """
    expected = checksums.get(archive.file)
    if expected is None:
        raise ChecksumMissing(
            f"SHA256SUMS at {example.revision} has no line for {archive.file}"
        )
    partial_directory.mkdir(parents=True, exist_ok=True)
    part = partial_directory / f"{archive.file}.part"
    progress = _Progress(archive.file, archive.download_bytes, enabled=not quiet)
    transfer = _Transfer(example, archive, part, progress)
    progress.begin(transfer.size)
    try:
        _with_attempts(example, archive.file, transfer.attempt)
    finally:
        progress.close(transfer.size)
    received = transfer.digest.hexdigest()
    if transfer.size != archive.download_bytes or received != expected:
        part.unlink(missing_ok=True)
        raise ChecksumMismatch(
            f"{archive.file} at revision {example.revision} does not match "
            f"SHA256SUMS: expected {expected} ({archive.download_bytes} bytes), "
            f"received {received} ({transfer.size} bytes)"
        )
    return part


FetchAction = Literal["downloaded", "reused"]


@dataclass(frozen=True)
class FetchedExample:
    """An example in a directory, as :func:`fetch_example` returns it.

    Attributes:
        name: The catalog name.
        directory: The directory that holds the unpacked archives.
        dataset: The mosaic dataset's directory.
        manifest: The dataset's ``dataset.yaml``.
        paths: Each archive's unpacked directory, keyed by its ``unpacks_to``.
        actions: Whether each archive was ``downloaded`` or ``reused``, keyed by file.
    """

    name: str
    directory: Path
    dataset: Path
    manifest: Path
    paths: Mapping[str, Path]
    actions: Mapping[str, FetchAction]


@dataclass(frozen=True)
class _RecordedArchive:
    sha256: str
    unpacks_to: str


@dataclass(frozen=True)
class _Record:
    """The completion record: which example, and the archives that are in place."""

    name: str
    repository: str
    revision: str
    archives: Mapping[str, _RecordedArchive]

    def to_json(self) -> str:
        """The record's file content."""
        return (
            json.dumps(
                {
                    "name": self.name,
                    "repository": self.repository,
                    "revision": self.revision,
                    "archives": {
                        file: {"sha256": item.sha256, "unpacks_to": item.unpacks_to}
                        for file, item in self.archives.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    """Whether *value* is a JSON object, whose keys are always strings."""
    return isinstance(value, dict)


def _parse_record(data: object) -> _Record | None:
    """The record in *data*, or ``None`` when *data* is not one."""
    if not _is_object(data):
        return None
    name, repository, revision = (
        data.get("name"),
        data.get("repository"),
        data.get("revision"),
    )
    listed = data.get("archives")
    if not (
        isinstance(name, str)
        and isinstance(repository, str)
        and isinstance(revision, str)
        and _is_object(listed)
    ):
        return None
    archives: dict[str, _RecordedArchive] = {}
    for file, item in listed.items():
        if not _is_object(item):
            return None
        sha256, unpacks_to = item.get("sha256"), item.get("unpacks_to")
        if not (isinstance(sha256, str) and isinstance(unpacks_to, str)):
            return None
        archives[file] = _RecordedArchive(sha256, unpacks_to)
    return _Record(name, repository, revision, archives)


def _read_record(directory: Path) -> _Record | None:
    """The completion record in *directory*, or ``None`` when there is none.

    Raises:
        FetchConflict: The record cannot be read.
    """
    path = directory / RECORD_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError:
        data = None
    record = _parse_record(data)
    if record is None:
        raise FetchConflict(
            f"{path} is not a completion record that mosaic examples fetch wrote; "
            "move it or pass another directory (--dir)"
        )
    return record


def _write_record(directory: Path, record: _Record) -> None:
    """Replace the completion record through a temporary file and ``os.replace``."""
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=".mosaic-example-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as out:
            out.write(record.to_json())
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, directory / RECORD_FILE)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _lock_holder(lock: Path) -> str:
    """``pid N on HOST`` from *lock*, or ``an unknown process`` when it is unreadable."""
    try:
        data: object = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "an unknown process"
    if _is_object(data):
        return f"pid {data.get('pid')} on {data.get('host')}"
    return "an unknown process"


def _extract(archive: Path, file: str, unpacks_to: str, staging: Path) -> None:
    """Unpack *archive* into *staging* under the checks that make it safe.

    ``tarfile.data_filter`` refuses absolute names, names that leave the destination,
    links that leave it and device or FIFO members, and clears set-user-id and
    group or other write bits. Two refusals follow it: a member whose first path
    component is not *unpacks_to*, and a link whose target lies outside
    ``<staging>/<unpacks_to>``.

    Raises:
        UnsafeArchive: A member is refused. The caller removes the staging.
    """
    root = Path(os.path.realpath(staging / unpacks_to))

    def refuse(member: str, reason: str) -> UnsafeArchive:
        return UnsafeArchive(f"{file} holds the unsafe member {member!r}: {reason}")

    def check(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
        original = member.name
        try:
            checked = tarfile.data_filter(member, path)
        except tarfile.FilterError as error:
            raise refuse(original, str(error)) from error
        parts = PurePosixPath(checked.name).parts
        if not parts or parts[0] != unpacks_to:
            raise refuse(original, f"it is not under {unpacks_to}/")
        if checked.issym() or checked.islnk():
            base = (
                os.path.join(path, os.path.dirname(checked.name))
                if checked.issym()
                else path
            )
            target = Path(os.path.realpath(os.path.join(base, checked.linkname)))
            if not target.is_relative_to(root):
                raise refuse(
                    original, f"it links to {checked.linkname!r}, outside {unpacks_to}/"
                )
        return checked

    with tarfile.open(archive, "r:gz") as opened:
        opened.extractall(staging, filter=check)


def _discard(staging: Path, part: Path, partial: Path) -> None:
    """Remove the staging, the archive and then the partial directory when empty."""
    shutil.rmtree(staging, ignore_errors=True)
    part.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        partial.rmdir()


def _plan(
    example: Example, directory: Path, record: _Record | None
) -> list[ExampleArchive]:
    """The archives to download, after refusing every conflict.

    Raises:
        FetchConflict: A top directory exists that no record names.
    """
    needed: list[ExampleArchive] = []
    for archive in example.archives:
        exists = (directory / archive.unpacks_to).exists()
        recorded = record is not None and archive.file in record.archives
        if exists and not recorded:
            raise FetchConflict(
                f"{directory / archive.unpacks_to} exists and was not unpacked by "
                "mosaic examples fetch; move it or pass another directory (--dir)"
            )
        if not (exists and recorded):
            needed.append(archive)
    return needed


def _unpack(
    example: Example,
    archive: ExampleArchive,
    directory: Path,
    part: Path,
    sha256: str,
    record: _Record,
) -> _Record:
    """Unpack the verified *part*, move it into place and record it.

    Returns:
        The record with *archive* added.

    Raises:
        UnsafeArchive: A member is refused.
        ArchiveWithoutDataset: The dataset archive holds no ``dataset.yaml``.
    """
    partial = part.parent
    staging = partial / _STAGING_DIRECTORY
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        _extract(part, archive.file, archive.unpacks_to, staging)
        if archive.dataset and not (staging / example.manifest_relative).is_file():
            raise ArchiveWithoutDataset(
                f"{archive.file} unpacked without {example.manifest_relative}; "
                "it holds no mosaic dataset"
            )
    except (UnsafeArchive, ArchiveWithoutDataset):
        _discard(staging, part, partial)
        raise
    os.rename(staging / archive.unpacks_to, directory / archive.unpacks_to)
    recorded = _Record(
        record.name,
        record.repository,
        record.revision,
        {**record.archives, archive.file: _RecordedArchive(sha256, archive.unpacks_to)},
    )
    _write_record(directory, recorded)
    part.unlink()
    shutil.rmtree(staging, ignore_errors=True)
    return recorded


def _fetch_locked(example: Example, directory: Path, quiet: bool) -> FetchedExample:
    """Refuse conflicts, reuse or download each archive and return the result.

    The lock is held by the caller.
    """
    record = _read_record(directory)
    if record is not None and (
        record.name != example.name or record.revision != example.revision
    ):
        raise FetchConflict(
            f"{directory} holds the example {record.name!r} at revision "
            f"{record.revision}, not {example.name!r} at revision {example.revision}; "
            "it may hold your derived work, so it is not replaced: pass another "
            "directory (--dir)"
        )
    needed = _plan(example, directory, record)
    current = record or _Record(example.name, example.repository, example.revision, {})
    actions: dict[str, FetchAction] = {}
    for archive in example.archives:
        if archive not in needed:
            actions[archive.file] = "reused"
            if not quiet:
                print(f"have {directory / archive.unpacks_to}", file=sys.stderr)
    partial = directory / PARTIAL_DIRECTORY
    if needed:
        checksums = fetch_checksums(example)
        for archive in needed:
            part = download_archive(example, archive, partial, checksums, quiet=quiet)
            current = _unpack(
                example, archive, directory, part, checksums[archive.file], current
            )
            actions[archive.file] = "downloaded"
    shutil.rmtree(partial, ignore_errors=True)
    dataset = directory / example.dataset_archive.unpacks_to
    return FetchedExample(
        name=example.name,
        directory=directory,
        dataset=dataset,
        manifest=directory / example.manifest_relative,
        paths={a.unpacks_to: directory / a.unpacks_to for a in example.archives},
        actions={a.file: actions[a.file] for a in example.archives},
    )


def find_example(name: str, catalog: Sequence[Example] | None = None) -> Example:
    """The catalog entry called *name*.

    Args:
        name: The example's catalog name.
        catalog: The entries to look the name up in. ``None`` reads the catalog's
            ``CATALOG`` when called.

    Returns:
        The entry.

    Raises:
        UnknownExample: *name* is not in the catalog. The message lists the names.
    """
    entries = catalog_module.CATALOG if catalog is None else catalog
    for entry in entries:
        if entry.name == name:
            return entry
    names = ", ".join(entry.name for entry in entries)
    raise UnknownExample(f"unknown example {name!r}; the examples are: {names}")


def fetch_example(
    name: str,
    directory: str | Path | None = None,
    *,
    quiet: bool = False,
    catalog: Sequence[Example] | None = None,
) -> FetchedExample:
    """Download, verify and unpack an example, or reuse an earlier fetch of it.

    A second fetch of the same example into the same directory makes no request and
    does not check the unpacked files again, because the dataset may have been
    changed since, as the demo converts its tracks. The completion record names the
    archives that are in place. A directory holding another example, another
    revision of this one, or a top directory that no record names is never replaced.

    Args:
        name: The example's catalog name.
        directory: Where to unpack, with a leading ``~`` expanded. ``None`` is
            ``~/mosaic-examples/<name>/``.
        quiet: Whether to leave standard error without progress lines.
        catalog: The entries to look the name up in. ``None`` reads the catalog's
            ``CATALOG`` when called. This is the seam tests use to serve generated
            examples.

    Returns:
        The example's paths and what was done to each archive.

    Raises:
        UnknownExample: *name* is not in the catalog.
        FetchConflict: Another fetch holds the lock, or the directory holds
            something fetching must not replace.
        DownloadFailed: A request failed for good.
        ChecksumMissing: ``SHA256SUMS`` does not parse or lacks an archive.
        ChecksumMismatch: A download differs from ``SHA256SUMS``.
        UnsafeArchive: An archive has a member that would leave its directory.
        ArchiveWithoutDataset: The dataset archive holds no ``dataset.yaml``.
    """
    example = find_example(name, catalog)
    target = (
        default_directory(name) if directory is None else user_path(directory)
    ).absolute()
    target.mkdir(parents=True, exist_ok=True)
    lock = target / LOCK_FILE
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise FetchConflict(
            f"another fetch into {target} is running ({_lock_holder(lock)}); "
            f"if none is, delete {lock}"
        ) from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as out:
            out.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
        return _fetch_locked(example, target, quiet)
    finally:
        lock.unlink(missing_ok=True)
