"""Downloading an example's archive from Hugging Face, and verifying it.

Each test runs against a local server that answers as Hugging Face does, with
``HF_ENDPOINT`` pointed at it, so none reaches the network. The waits between attempts
go through the module's ``_sleep``, which the tests record instead of sleeping.
"""

from __future__ import annotations

import dataclasses
import errno
import http.client
import io
import itertools
import json
import os
import shutil
import socket
import ssl
import tarfile
import time
import urllib.request
from collections.abc import Buffer, Callable, Mapping
from email.utils import formatdate
from http.client import HTTPMessage
from pathlib import Path
from typing import IO

import pytest

from mosaic.examples import catalog, fetch
from mosaic.examples.catalog import Example, ExampleArchive
from mosaic.examples.fetch import (
    ArchiveWithoutDataset,
    ChecksumMismatch,
    ChecksumMissing,
    DownloadFailed,
    ExampleFetchError,
    FetchConflict,
    RedirectHandler,
    UnknownExample,
    UnsafeArchive,
    archive_url,
    default_directory,
    download_archive,
    fetch_checksums,
    fetch_example,
    find_example,
    parse_checksums,
)
from mosaic.version import installed_version
from tests.helpers import (
    REVISION,
    ExampleFixture,
    ExampleServer,
    GeneratedArchive,
    Reply,
    archive_path,
    cdn_path,
    checksums_path,
    device_member,
    example_fixture,
    hardlink_member,
    serve_example,
    sums_text,
    symlink_member,
    tar_gz,
    tar_gz_with,
)

ARCHIVE_FILE = "tiny-tracks.tar.gz"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits the fetch module asked for, which it no longer sleeps."""
    recorded: list[float] = []
    monkeypatch.setattr(fetch, "_sleep", recorded.append)
    return recorded


@pytest.fixture
def fixture(example_server: ExampleServer) -> ExampleFixture:
    """The generated example, published on the server."""
    published = example_fixture()
    serve_example(example_server, published)
    return published


def download(published: ExampleFixture, directory: Path, **options: bool) -> Path:
    """Download *published*'s first archive into *directory* and return its path."""
    example = published.example
    return download_archive(
        example,
        example.archives[0],
        directory,
        fetch_checksums(example),
        **options,
    )


def part_of(directory: Path) -> Path:
    """The partial file of the generated archive in *directory*."""
    return directory / f"{ARCHIVE_FILE}.part"


class TestUrl:
    def test_the_url_names_repository_revision_and_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        example = example_fixture().example

        assert archive_url(example, "a.tar.gz") == (
            f"https://huggingface.co/datasets/{example.repository}"
            f"/resolve/{REVISION}/a.tar.gz"
        )

    def test_hf_endpoint_is_read_at_call_time_without_a_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        example = example_fixture().example
        monkeypatch.setenv("HF_ENDPOINT", "http://mirror.test:8080/")

        assert archive_url(example, "a").startswith("http://mirror.test:8080/datasets/")

        monkeypatch.setenv("HF_ENDPOINT", "https://other.test")

        assert archive_url(example, "a").startswith("https://other.test/datasets/")


class TestDownload:
    def test_the_archive_arrives_whole_through_a_302_with_an_absolute_location(
        self, fixture: ExampleFixture, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        part = download(fixture, tmp_path / "partial")

        assert part == part_of(tmp_path / "partial")
        assert part.read_bytes() == fixture.archives[ARCHIVE_FILE]
        assert example_server.count(cdn_path(fixture.example, ARCHIVE_FILE)) == 1

    def test_the_checksums_arrive_through_a_307_with_a_relative_location(
        self, fixture: ExampleFixture, example_server: ExampleServer
    ) -> None:
        checksums = fetch_checksums(fixture.example)

        assert set(checksums) == {ARCHIVE_FILE}
        assert example_server.requests[0].path == archive_path(
            fixture.example, "SHA256SUMS"
        )
        assert example_server.requests[1].path.startswith("/api/resolve-cache/")

    def test_every_request_names_mosaic_in_its_user_agent(
        self, fixture: ExampleFixture, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        download(fixture, tmp_path)

        agents = {request.user_agent for request in example_server.requests}
        assert agents == {f"mosaic-behavior/{installed_version()}"}

    def test_more_than_ten_redirects_fail(
        self, fixture: ExampleFixture, example_server: ExampleServer
    ) -> None:
        checksums_path = archive_path(fixture.example, "SHA256SUMS")
        example_server.add_redirect(checksums_path, 302, "/hop/0")
        for hop in range(12):
            example_server.add_redirect(f"/hop/{hop}", 302, f"/hop/{hop + 1}")

        with pytest.raises(DownloadFailed):
            fetch_checksums(fixture.example)

    def test_the_partial_file_is_kept_after_a_verified_download(
        self, fixture: ExampleFixture, tmp_path: Path
    ) -> None:
        assert download(fixture, tmp_path).exists()


class TestRedirectHandler:
    @staticmethod
    def redirect(handler: RedirectHandler, source: str, target: str) -> object:
        request = urllib.request.Request(source, headers={"Range": "bytes=5-"})
        return handler.redirect_request(
            request, io.BytesIO(), 302, "Found", HTTPMessage(), target
        )

    def test_an_https_endpoint_refuses_an_http_target_at_the_first_hop(self) -> None:
        handler = RedirectHandler("https://huggingface.co")

        with pytest.raises(DownloadFailed, match="http://cdn.test/x"):
            self.redirect(
                handler,
                "https://huggingface.co/datasets/r/resolve/v/f",
                "http://cdn.test/x",
            )

    def test_an_https_endpoint_refuses_an_http_target_at_a_later_hop(self) -> None:
        handler = RedirectHandler("https://huggingface.co")

        with pytest.raises(DownloadFailed):
            self.redirect(handler, "https://cdn.test/a", "http://cdn.test/b")

    def test_an_https_endpoint_follows_an_https_target_and_keeps_range(self) -> None:
        handler = RedirectHandler("https://huggingface.co")

        followed = self.redirect(
            handler, "https://huggingface.co/a", "https://cdn.test/b"
        )

        assert isinstance(followed, urllib.request.Request)
        assert followed.full_url == "https://cdn.test/b"
        assert followed.get_header("Range") == "bytes=5-"

    def test_an_http_endpoint_follows_an_http_target(self) -> None:
        handler = RedirectHandler("http://127.0.0.1:8000")

        followed = self.redirect(
            handler, "http://127.0.0.1:8000/a", "http://127.0.0.1:8000/b"
        )

        assert isinstance(followed, urllib.request.Request)
        assert followed.full_url == "http://127.0.0.1:8000/b"


class TestRateLimit:
    def test_a_429_with_retry_after_zero_then_a_200_succeeds(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        example_server.script(
            cdn_path(fixture.example, ARCHIVE_FILE),
            Reply(429, headers={"Retry-After": "0"}),
        )

        part = download(fixture, tmp_path)

        assert part.read_bytes() == fixture.archives[ARCHIVE_FILE]
        assert sleeps == [0]
        assert "rate limited by Hugging Face; waiting 0 s" in capsys.readouterr().err

    def test_a_429_that_never_ends_raises_after_five_attempts(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        example_server.script(path, *[Reply(429, headers={"Retry-After": "1"})] * 6)

        with pytest.raises(DownloadFailed, match="429") as raised:
            download(fixture, tmp_path)

        assert example_server.count(path) == 5
        assert sleeps == [1, 1, 1, 1]
        message = str(raised.value)
        assert fixture.example.repository in message
        assert ARCHIVE_FILE in message
        assert REVISION in message

    def test_a_retry_after_above_the_window_waits_the_window(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        example_server.script(
            cdn_path(fixture.example, ARCHIVE_FILE),
            Reply(429, headers={"Retry-After": "9999"}),
        )

        download(fixture, tmp_path)

        assert sleeps == [300]

    def test_a_429_without_retry_after_waits_sixty_seconds(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        example_server.script(cdn_path(fixture.example, ARCHIVE_FILE), Reply(429))

        download(fixture, tmp_path)

        assert sleeps == [60]

    def test_a_retry_after_date_waits_until_then(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        date = formatdate(time.time() + 120, usegmt=True)
        example_server.script(
            cdn_path(fixture.example, ARCHIVE_FILE),
            Reply(429, headers={"Retry-After": date}),
        )

        download(fixture, tmp_path)

        (waited,) = sleeps
        assert 100 <= waited <= 120

    def test_the_checksums_request_is_rate_limited_too(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
    ) -> None:
        example_server.script(
            checksums_path(fixture.example), Reply(429, headers={"Retry-After": "3"})
        )

        assert set(fetch_checksums(fixture.example)) == {ARCHIVE_FILE}
        assert sleeps == [3]


class TestResume:
    def test_a_body_cut_part_way_resumes_with_range_and_verifies(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        data = fixture.archives[ARCHIVE_FILE]
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        example_server.script(path, Reply(200, data, cut_after=100))

        part = download(fixture, tmp_path)

        assert part.read_bytes() == data
        ranges = [r.range for r in example_server.requests if r.path == path]
        assert ranges == [None, "bytes=100-"]
        assert sleeps == [2]

    def test_an_existing_partial_file_is_resumed_and_appended_to(
        self, fixture: ExampleFixture, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        data = fixture.archives[ARCHIVE_FILE]
        part_of(tmp_path).write_bytes(data[:50])

        part = download(fixture, tmp_path)

        assert part.read_bytes() == data
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        assert [r.range for r in example_server.requests if r.path == path] == [
            "bytes=50-"
        ]

    def test_a_200_to_a_range_request_restarts_the_download(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        tmp_path: Path,
    ) -> None:
        data = fixture.archives[ARCHIVE_FILE]
        part_of(tmp_path).write_bytes(b"not the start of the archive")
        example_server.script(cdn_path(fixture.example, ARCHIVE_FILE), Reply(200, data))

        assert download(fixture, tmp_path).read_bytes() == data

    def test_a_complete_partial_file_goes_to_verification_on_a_416(
        self, fixture: ExampleFixture, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        data = fixture.archives[ARCHIVE_FILE]
        part_of(tmp_path).write_bytes(data)

        part = download(fixture, tmp_path)

        assert part.read_bytes() == data
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        assert [r.range for r in example_server.requests if r.path == path] == [
            f"bytes={len(data)}-"
        ]


class TestFailures:
    def test_a_500_then_a_200_succeeds_after_a_two_second_wait(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        example_server.script(cdn_path(fixture.example, ARCHIVE_FILE), Reply(500))

        assert download(fixture, tmp_path).exists()
        assert sleeps == [2]

    def test_a_server_that_keeps_failing_is_tried_five_times_with_growing_waits(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        example_server.script(path, *[Reply(503)] * 6)

        with pytest.raises(DownloadFailed, match="503"):
            download(fixture, tmp_path)

        assert example_server.count(path) == 5
        assert sleeps == [2, 4, 8, 16]

    def test_a_404_raises_at_once_naming_file_and_revision(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        example_server.script(path, Reply(404))

        with pytest.raises(DownloadFailed) as raised:
            download(fixture, tmp_path)

        assert example_server.count(path) == 1
        assert sleeps == []
        assert ARCHIVE_FILE in str(raised.value)
        assert REVISION in str(raised.value)

    def test_another_4xx_raises_at_once(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        example_server.script(path, Reply(403))

        with pytest.raises(DownloadFailed, match="403"):
            download(fixture, tmp_path)

        assert example_server.count(path) == 1
        assert sleeps == []

    def test_a_refused_connection_is_retried_then_fails(
        self,
        fixture: ExampleFixture,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        checksums = fetch_checksums(fixture.example)
        monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:1")

        with pytest.raises(DownloadFailed):
            download_archive(
                fixture.example, fixture.example.archives[0], tmp_path, checksums
            )

        assert sleeps == [2, 4, 8, 16]

    def test_every_failure_is_an_example_fetch_error(
        self, fixture: ExampleFixture, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        example_server.script(cdn_path(fixture.example, ARCHIVE_FILE), Reply(404))

        with pytest.raises(ExampleFetchError):
            download(fixture, tmp_path)


def fail_first_read(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    """Make the first body read of a response raise *error*; the rest read normally."""
    original = http.client.HTTPResponse.read
    attempts: list[int] = []

    def read(self: http.client.HTTPResponse, size: int | None = None) -> bytes:
        attempts.append(1)
        if len(attempts) == 1:
            raise error
        return original(self, size)

    monkeypatch.setattr(http.client.HTTPResponse, "read", read)


class TestBodyReadFailures:
    def test_a_reset_while_reading_the_body_is_retried(
        self,
        fixture: ExampleFixture,
        example_server: ExampleServer,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        data = fixture.archives[ARCHIVE_FILE]
        path = cdn_path(fixture.example, ARCHIVE_FILE)
        example_server.script(path, Reply(200, data, cut_after=100, reset=True))

        assert download(fixture, tmp_path).read_bytes() == data
        assert sleeps == [2]

    @pytest.mark.parametrize(
        "error",
        [ssl.SSLError("[SSL: DECRYPTION_FAILED]"), OSError(errno.EIO, "I/O error")],
        ids=["ssl", "oserror"],
    )
    def test_an_archive_read_that_fails_is_retried(
        self,
        fixture: ExampleFixture,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        tmp_path: Path,
        error: OSError,
    ) -> None:
        checksums = fetch_checksums(fixture.example)
        fail_first_read(monkeypatch, error)

        part = download_archive(
            fixture.example, fixture.example.archives[0], tmp_path, checksums
        )

        assert part.read_bytes() == fixture.archives[ARCHIVE_FILE]
        assert sleeps == [2]

    @pytest.mark.parametrize(
        "error",
        [ssl.SSLError("[SSL: DECRYPTION_FAILED]"), OSError(errno.EIO, "I/O error")],
        ids=["ssl", "oserror"],
    )
    def test_a_checksums_read_that_fails_is_retried(
        self,
        fixture: ExampleFixture,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        error: OSError,
    ) -> None:
        fail_first_read(monkeypatch, error)

        assert set(fetch_checksums(fixture.example)) == {ARCHIVE_FILE}
        assert sleeps == [2]

    def test_a_read_that_keeps_failing_ends_as_a_download_failure(
        self,
        fixture: ExampleFixture,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        checksums = fetch_checksums(fixture.example)

        def read(self: http.client.HTTPResponse, size: int | None = None) -> bytes:
            raise ssl.SSLError("[SSL: DECRYPTION_FAILED]")

        monkeypatch.setattr(http.client.HTTPResponse, "read", read)

        with pytest.raises(DownloadFailed, match="SSLError"):
            download_archive(
                fixture.example, fixture.example.archives[0], tmp_path, checksums
            )

        assert sleeps == [2, 4, 8, 16]

    def test_a_failure_to_write_the_partial_file_is_not_retried(
        self,
        fixture: ExampleFixture,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        tmp_path: Path,
    ) -> None:
        checksums = fetch_checksums(fixture.example)

        class FullDisk(io.BytesIO):
            def write(self, buffer: Buffer, /) -> int:
                raise OSError(errno.ENOSPC, "No space left on device")

        original = Path.open

        def open_part(
            self: Path,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> IO[bytes]:
            if self.suffix == ".part" and mode in {"wb", "ab"}:
                return FullDisk()
            return original(self, mode, buffering, encoding, errors, newline)

        monkeypatch.setattr(Path, "open", open_part)

        with pytest.raises(OSError, match="No space left") as raised:
            download_archive(
                fixture.example, fixture.example.archives[0], tmp_path, checksums
            )

        assert not isinstance(raised.value, ExampleFetchError)
        assert sleeps == []


class TestChecksums:
    def test_the_parser_reads_text_and_binary_lines_and_skips_blank_ones(self) -> None:
        digest_a, digest_b = "a" * 64, "B" * 64
        text = f"{digest_a}  one.tar.gz\n\n{digest_b} *two.tar.gz\n"

        assert parse_checksums(text, REVISION) == {
            "one.tar.gz": digest_a,
            "two.tar.gz": digest_b.lower(),
        }

    @pytest.mark.parametrize(
        "text",
        ["not a checksum line\n", f"{'a' * 63}  short.tar.gz\n", f"{'a' * 64} one\n"],
    )
    def test_any_other_line_raises(self, text: str) -> None:
        with pytest.raises(ChecksumMissing, match="does not parse"):
            parse_checksums(text, REVISION)

    def test_a_file_without_a_line_raises_before_the_archive_is_requested(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = example_fixture(omit_from_sums=[ARCHIVE_FILE])
        serve_example(example_server, published)

        with pytest.raises(ChecksumMissing, match=ARCHIVE_FILE):
            download(published, tmp_path)

        assert example_server.count(archive_path(published.example, ARCHIVE_FILE)) == 0
        assert example_server.count(cdn_path(published.example, ARCHIVE_FILE)) == 0

    def test_a_wrong_digest_raises_and_deletes_the_partial_file(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = example_fixture()
        wrong = dataclasses.replace(
            published, sums=sums_text({ARCHIVE_FILE: b"other bytes"})
        )
        serve_example(example_server, wrong)

        with pytest.raises(ChecksumMismatch, match=ARCHIVE_FILE):
            download(wrong, tmp_path)

        assert not part_of(tmp_path).exists()

    def test_a_wrong_size_raises_and_deletes_the_partial_file(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = example_fixture()
        (archive,) = published.example.archives
        larger = dataclasses.replace(
            published.example,
            archives=(
                dataclasses.replace(archive, download_bytes=archive.download_bytes + 1),
            ),
        )
        serve_example(example_server, published)

        with pytest.raises(ChecksumMismatch, match=str(archive.download_bytes + 1)):
            download_archive(
                larger, larger.archives[0], tmp_path, fetch_checksums(larger)
            )

        assert not part_of(tmp_path).exists()

    def test_the_next_run_after_a_mismatch_starts_clean(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = example_fixture()
        serve_example(example_server, published)
        part_of(tmp_path).write_bytes(b"x" * len(published.archives[ARCHIVE_FILE]))

        with pytest.raises(ChecksumMismatch):
            download(published, tmp_path)

        assert (
            download(published, tmp_path).read_bytes()
            == published.archives[ARCHIVE_FILE]
        )


class TestProgress:
    @staticmethod
    def large_fixture() -> ExampleFixture:
        """An example whose archive spans several read chunks."""
        data = os.urandom(300_000)
        return example_fixture(
            [
                GeneratedArchive(
                    "big.tar.gz", "big", tar_gz({"big/dataset.yaml": data}), True
                )
            ]
        )

    def test_a_quiet_download_writes_nothing(
        self,
        fixture: ExampleFixture,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        download(fixture, tmp_path, quiet=True)

        assert capsys.readouterr().err == ""

    def test_without_a_terminal_a_line_is_printed_at_each_tenth(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        published = self.large_fixture()
        serve_example(example_server, published)

        download_archive(
            published.example,
            published.example.archives[0],
            tmp_path,
            fetch_checksums(published.example),
        )

        lines = capsys.readouterr().err.splitlines()
        assert lines[0].startswith("fetching big.tar.gz (")
        assert "\r" not in "".join(lines)
        assert lines[-1].endswith("100%")
        percentages = [int(line.split()[-1].rstrip("%")) for line in lines[1:]]
        assert percentages == sorted(set(percentages))
        assert all(percent % 10 == 0 for percent in percentages)

    def test_on_a_terminal_updates_within_half_a_second_give_one_update(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ticks = itertools.count()

        def clock() -> float:
            return next(ticks) * 0.05

        output = self.terminal_output(example_server, tmp_path, monkeypatch, clock)

        assert output.startswith("fetching big.tar.gz (")
        assert output.count("\r") == 2
        assert output.endswith("100%\n")

    def test_on_a_terminal_every_half_second_allows_another_update(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ticks = itertools.count()

        def clock() -> float:
            return next(ticks) * 0.5

        output = self.terminal_output(example_server, tmp_path, monkeypatch, clock)

        assert output.count("\r") > 2

    def terminal_output(
        self,
        server: ExampleServer,
        directory: Path,
        monkeypatch: pytest.MonkeyPatch,
        clock: Callable[[], float],
    ) -> str:
        """What a terminal's standard error held while the large archive downloaded."""
        published = self.large_fixture()
        serve_example(server, published)
        checksums = fetch_checksums(published.example)
        terminal = _Terminal()
        monkeypatch.setattr("sys.stderr", terminal)
        monkeypatch.setattr(fetch, "_monotonic", clock)

        download_archive(
            published.example, published.example.archives[0], directory, checksums
        )

        return terminal.getvalue()


class _Terminal(io.StringIO):
    """A standard error that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


RAW_FILE = "tiny-raw.tar.gz"
DATASET_MEMBERS = {
    "tiny-tracks/dataset.yaml": b"name: tiny\n",
    "tiny-tracks/tracks/a.txt": b"track a\n" * 100,
}


def two_archives() -> ExampleFixture:
    """An example with a dataset archive and a raw sample, published separately."""
    return example_fixture(
        [
            GeneratedArchive(
                ARCHIVE_FILE, "tiny-tracks", tar_gz(DATASET_MEMBERS), True
            ),
            GeneratedArchive(
                RAW_FILE,
                "tiny-raw",
                tar_gz({"tiny-raw/clip.txt": b"clip\n" * 50}),
                False,
            ),
        ]
    )


def serve(example_server: ExampleServer, published: ExampleFixture) -> ExampleFixture:
    """Publish *published* and return it."""
    serve_example(example_server, published)
    return published


def fetch_tiny(
    published: ExampleFixture, directory: Path | None, *, quiet: bool = True
) -> fetch.FetchedExample:
    """Fetch the generated example into *directory*."""
    return fetch_example(
        published.example.name, directory, quiet=quiet, catalog=[published.example]
    )


def names_in(directory: Path) -> set[str]:
    """The names directly inside *directory*."""
    return {entry.name for entry in directory.iterdir()}


def write_record(
    directory: Path,
    *,
    name: str = "tiny",
    revision: str = REVISION,
    files: dict[str, str],
) -> None:
    """Write a completion record naming *files*, as ``{file: unpacks_to}``."""
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "repository": "EcodylicScience/mosaic-example-tiny",
        "revision": revision,
        "archives": {
            file: {"sha256": "0" * 64, "unpacks_to": top} for file, top in files.items()
        },
    }
    (directory / ".mosaic-example.json").write_text(json.dumps(record))


class TestFirstFetch:
    def test_it_writes_the_top_directory_the_record_and_the_manifest(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"

        fetched = fetch_tiny(published, directory)

        assert fetched.name == "tiny"
        assert fetched.directory == directory
        assert fetched.dataset == directory / "tiny-tracks"
        assert fetched.manifest == directory / "tiny-tracks" / "dataset.yaml"
        assert fetched.manifest.read_text() == "name: tiny\n"
        assert fetched.paths == {"tiny-tracks": directory / "tiny-tracks"}
        assert fetched.actions == {ARCHIVE_FILE: "downloaded"}
        assert names_in(directory) == {"tiny-tracks", ".mosaic-example.json"}
        record = json.loads((directory / ".mosaic-example.json").read_text())
        digest = sums_text(published.archives).split()[0]
        assert record == {
            "name": "tiny",
            "repository": published.example.repository,
            "revision": REVISION,
            "archives": {ARCHIVE_FILE: {"sha256": digest, "unpacks_to": "tiny-tracks"}},
        }

    def test_it_unpacks_every_archive_of_an_example(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, two_archives())

        fetched = fetch_tiny(published, tmp_path / "tiny")

        assert fetched.paths == {
            "tiny-tracks": tmp_path / "tiny" / "tiny-tracks",
            "tiny-raw": tmp_path / "tiny" / "tiny-raw",
        }
        assert fetched.dataset == tmp_path / "tiny" / "tiny-tracks"
        assert fetched.actions == {ARCHIVE_FILE: "downloaded", RAW_FILE: "downloaded"}
        record = json.loads((tmp_path / "tiny" / ".mosaic-example.json").read_text())
        assert set(record["archives"]) == {ARCHIVE_FILE, RAW_FILE}
        assert (tmp_path / "tiny" / "tiny-raw" / "clip.txt").is_file()

    def test_nothing_but_the_results_remains_after_it(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, two_archives())

        fetch_tiny(published, tmp_path / "tiny")

        assert names_in(tmp_path / "tiny") == {
            "tiny-tracks",
            "tiny-raw",
            ".mosaic-example.json",
        }

    def test_the_checksums_are_requested_once_for_every_archive(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, two_archives())

        fetch_tiny(published, tmp_path / "tiny")

        sums = [r for r in example_server.requests if r.path.endswith("SHA256SUMS")]
        assert len(sums) == 1

    def test_the_default_directory_is_mosaic_examples_under_the_home_directory(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        published = serve(example_server, example_fixture())

        fetched = fetch_tiny(published, None)

        assert fetched.directory == tmp_path / "mosaic-examples" / "tiny"
        assert fetched.manifest.is_file()

    def test_default_directory_names_the_example_under_mosaic_examples(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

        assert default_directory("tiny") == tmp_path / "mosaic-examples" / "tiny"

    def test_a_directory_with_a_tilde_is_expanded(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        published = serve(example_server, example_fixture())

        fetched = fetch_example(
            "tiny", "~/elsewhere", quiet=True, catalog=[published.example]
        )

        assert fetched.directory == tmp_path / "elsewhere"

    def test_the_catalog_is_read_at_call_time_by_default(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        published = serve(example_server, example_fixture())
        monkeypatch.setattr(catalog, "CATALOG", (published.example,))

        fetched = fetch_example("tiny", tmp_path / "tiny", quiet=True)

        assert fetched.actions == {ARCHIVE_FILE: "downloaded"}

    def test_an_unknown_name_lists_the_names_and_makes_no_request(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())

        with pytest.raises(UnknownExample, match="tiny"):
            fetch_example("nope", tmp_path / "x", catalog=[published.example])

        assert example_server.requests == []
        assert not (tmp_path / "x").exists()

    def test_find_example_returns_the_entry_or_lists_the_names(self) -> None:
        first = example_fixture(name="first").example
        second = example_fixture(name="second").example

        assert find_example("second", [first, second]) is second
        with pytest.raises(UnknownExample, match=r"'nope'.*first, second"):
            find_example("nope", [first, second])

    def test_find_example_reads_the_catalog_when_given_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = example_fixture(name="only").example
        monkeypatch.setattr(catalog, "CATALOG", (entry,))

        assert find_example("only") is entry

    def test_progress_goes_to_standard_error_unless_quiet(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        published = serve(example_server, example_fixture())

        fetch_tiny(published, tmp_path / "quiet", quiet=True)
        assert capsys.readouterr().err == ""

        fetch_tiny(published, tmp_path / "loud", quiet=False)
        captured = capsys.readouterr()
        assert f"fetching {ARCHIVE_FILE}" in captured.err
        assert captured.out == ""

    def test_the_lock_is_gone_after_a_fetch(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())

        fetch_tiny(published, tmp_path / "tiny")

        assert not (tmp_path / "tiny" / ".mosaic-example.lock").exists()


class TestReuse:
    def test_a_second_fetch_makes_no_request_after_the_dataset_was_changed(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        fetch_tiny(published, directory)
        converted = directory / "tiny-tracks" / "tracks" / "a.txt"
        converted.write_text("converted\n")
        requests = len(example_server.requests)

        again = fetch_tiny(published, directory)

        assert len(example_server.requests) == requests
        assert again.actions == {ARCHIVE_FILE: "reused"}
        assert again.manifest == directory / "tiny-tracks" / "dataset.yaml"
        assert converted.read_text() == "converted\n"

    def test_a_reused_archive_is_reported_on_standard_error_as_had(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        fetch_tiny(published, directory)
        capsys.readouterr()

        fetch_tiny(published, directory, quiet=False)

        assert f"have {directory / 'tiny-tracks'}" in capsys.readouterr().err

    def test_only_the_archive_the_record_lacks_is_downloaded(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, two_archives())
        directory = tmp_path / "tiny"
        (directory / "tiny-tracks").mkdir(parents=True)
        (directory / "tiny-tracks" / "dataset.yaml").write_text("name: tiny\n")
        write_record(directory, files={ARCHIVE_FILE: "tiny-tracks"})

        fetched = fetch_tiny(published, directory)

        assert fetched.actions == {ARCHIVE_FILE: "reused", RAW_FILE: "downloaded"}
        assert example_server.count(cdn_path(published.example, ARCHIVE_FILE)) == 0
        record = json.loads((directory / ".mosaic-example.json").read_text())
        assert set(record["archives"]) == {ARCHIVE_FILE, RAW_FILE}
        assert record["archives"][ARCHIVE_FILE]["sha256"] == "0" * 64

    def test_a_reuse_removes_a_partial_directory_an_interruption_left(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        fetch_tiny(published, directory)
        partial = directory / ".mosaic-example-partial"
        partial.mkdir()
        part_of(partial).write_bytes(published.archives[ARCHIVE_FILE])
        requests = len(example_server.requests)

        again = fetch_tiny(published, directory)

        assert again.actions == {ARCHIVE_FILE: "reused"}
        assert len(example_server.requests) == requests
        assert names_in(directory) == {"tiny-tracks", ".mosaic-example.json"}

    def test_an_archive_whose_directory_was_deleted_is_downloaded_again(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        fetch_tiny(published, directory)
        shutil.rmtree(directory / "tiny-tracks")

        fetched = fetch_tiny(published, directory)

        assert fetched.actions == {ARCHIVE_FILE: "downloaded"}
        assert fetched.manifest.is_file()


class TestUnsafeArchive:
    @staticmethod
    def unsafe(extra: str) -> ExampleFixture:
        """An example whose dataset archive holds a good member then a bad one."""
        good = dict(DATASET_MEMBERS)
        archive: bytes
        if extra == "parent":
            archive = tar_gz({**good, "../x": b"x"})
        elif extra == "absolute":
            archive = tar_gz({**good, "/tmp/evil": b"x"})
        elif extra == "outside":
            archive = tar_gz({**good, "other/x": b"x"})
        elif extra == "symlink":
            archive = tar_gz_with(
                good, [symlink_member("tiny-tracks/link", "../other")]
            )
        elif extra == "hardlink":
            archive = tar_gz_with(
                good, [hardlink_member("tiny-tracks/passwd", "/etc/passwd")]
            )
        else:
            archive = tar_gz_with(good, [device_member("tiny-tracks/tty")])
        return example_fixture(
            [GeneratedArchive(ARCHIVE_FILE, "tiny-tracks", archive, True)]
        )

    @pytest.mark.parametrize(
        ("extra", "member"),
        [
            ("parent", "../x"),
            ("absolute", "tmp/evil"),
            ("outside", "other/x"),
            ("symlink", "tiny-tracks/link"),
            ("hardlink", "tiny-tracks/passwd"),
            ("device", "tiny-tracks/tty"),
        ],
    )
    def test_an_unsafe_member_raises_and_leaves_the_directory_as_it_was(
        self,
        extra: str,
        member: str,
        example_server: ExampleServer,
        tmp_path: Path,
    ) -> None:
        published = serve(example_server, self.unsafe(extra))
        directory = tmp_path / "tiny"
        directory.mkdir()

        with pytest.raises(UnsafeArchive) as raised:
            fetch_tiny(published, directory)

        assert ARCHIVE_FILE in str(raised.value)
        assert member in str(raised.value)
        assert names_in(directory) == set()
        assert not (tmp_path / "x").exists()

    def test_a_symlink_inside_the_top_directory_is_extracted(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        archive = tar_gz_with(
            DATASET_MEMBERS, [symlink_member("tiny-tracks/latest", "tracks/a.txt")]
        )
        published = serve(
            example_server,
            example_fixture(
                [GeneratedArchive(ARCHIVE_FILE, "tiny-tracks", archive, True)]
            ),
        )

        fetched = fetch_tiny(published, tmp_path / "tiny")

        link = fetched.dataset / "latest"
        assert link.is_symlink()
        assert link.read_text() == "track a\n" * 100

    def test_an_unsafe_second_archive_keeps_the_first_that_is_recorded(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        raw = tar_gz({"tiny-raw/ok": b"ok", "../x": b"x"})
        published = serve(
            example_server,
            example_fixture(
                [
                    GeneratedArchive(
                        ARCHIVE_FILE, "tiny-tracks", tar_gz(DATASET_MEMBERS), True
                    ),
                    GeneratedArchive(RAW_FILE, "tiny-raw", raw, False),
                ]
            ),
        )
        directory = tmp_path / "tiny"

        with pytest.raises(UnsafeArchive):
            fetch_tiny(published, directory)

        assert names_in(directory) == {"tiny-tracks", ".mosaic-example.json"}
        record = json.loads((directory / ".mosaic-example.json").read_text())
        assert set(record["archives"]) == {ARCHIVE_FILE}


class TestInterruptedExtraction:
    @pytest.mark.parametrize("failure", [OSError("disk full"), KeyboardInterrupt()])
    def test_a_failure_during_extraction_leaves_no_top_directory_and_no_lock(
        self,
        failure: BaseException,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        calls = itertools.count()
        original = tarfile.data_filter

        def failing(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
            if next(calls) == 1:
                raise failure
            return original(member, path)

        monkeypatch.setattr(tarfile, "data_filter", failing)

        with pytest.raises(type(failure)):
            fetch_tiny(published, directory)

        assert not (directory / "tiny-tracks").exists()
        assert not (directory / ".mosaic-example.json").exists()
        assert not (directory / ".mosaic-example.lock").exists()
        staged = directory / ".mosaic-example-partial" / "unpack"
        assert staged.is_dir()

    def test_the_next_fetch_removes_the_staging_and_succeeds(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        original = tarfile.data_filter
        calls = itertools.count()

        def failing(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
            if next(calls) == 1:
                raise OSError("disk full")
            return original(member, path)

        with monkeypatch.context() as patched:
            patched.setattr(tarfile, "data_filter", failing)
            with pytest.raises(OSError, match="disk full"):
                fetch_tiny(published, directory)

        fetched = fetch_tiny(published, directory)

        assert fetched.actions == {ARCHIVE_FILE: "downloaded"}
        assert fetched.manifest.is_file()
        assert names_in(directory) == {"tiny-tracks", ".mosaic-example.json"}
        assert (fetched.dataset / "tracks" / "a.txt").read_bytes() == (
            b"track a\n" * 1000
        )


class TestConflicts:
    def test_a_top_directory_without_a_record_is_refused(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        (directory / "tiny-tracks").mkdir(parents=True)
        (directory / "tiny-tracks" / "mine.txt").write_text("mine")

        with pytest.raises(
            FetchConflict, match="not unpacked by mosaic examples fetch"
        ):
            fetch_tiny(published, directory)

        assert example_server.requests == []
        assert names_in(directory) == {"tiny-tracks"}
        assert names_in(directory / "tiny-tracks") == {"mine.txt"}

    def test_a_conflict_on_a_later_archive_writes_nothing_for_an_earlier_one(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, two_archives())
        directory = tmp_path / "tiny"
        (directory / "tiny-raw").mkdir(parents=True)

        with pytest.raises(FetchConflict):
            fetch_tiny(published, directory)

        assert example_server.requests == []
        assert names_in(directory) == {"tiny-raw"}

    def test_a_record_of_another_name_is_refused_naming_both(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        write_record(directory, name="other", files={ARCHIVE_FILE: "tiny-tracks"})

        with pytest.raises(FetchConflict, match="other.*tiny|tiny.*other"):
            fetch_tiny(published, directory)

        assert example_server.requests == []
        assert names_in(directory) == {".mosaic-example.json"}

    def test_a_record_of_another_revision_is_refused_naming_both(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        older = "f" * 40
        write_record(directory, revision=older, files={ARCHIVE_FILE: "tiny-tracks"})
        (directory / "tiny-tracks").mkdir()

        with pytest.raises(FetchConflict) as raised:
            fetch_tiny(published, directory)

        assert older in str(raised.value)
        assert REVISION in str(raised.value)
        assert "--dir" in str(raised.value)
        assert example_server.requests == []
        assert names_in(directory) == {".mosaic-example.json", "tiny-tracks"}

    def test_a_record_that_cannot_be_read_is_refused(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        directory.mkdir()
        (directory / ".mosaic-example.json").write_text("{not json")

        with pytest.raises(FetchConflict, match="completion record"):
            fetch_tiny(published, directory)

        assert example_server.requests == []

    def test_an_existing_lock_is_refused_and_left_in_place(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        directory.mkdir()
        lock = directory / ".mosaic-example.lock"
        lock.write_text(json.dumps({"pid": 4242, "host": "otherhost"}))

        with pytest.raises(FetchConflict) as raised:
            fetch_tiny(published, directory)

        message = str(raised.value)
        assert "another fetch" in message
        assert "4242" in message
        assert "otherhost" in message
        assert str(lock) in message
        assert example_server.requests == []
        assert names_in(directory) == {".mosaic-example.lock"}
        assert json.loads(lock.read_text()) == {"pid": 4242, "host": "otherhost"}

    def test_the_lock_holds_this_process_and_host_while_the_fetch_runs(
        self,
        example_server: ExampleServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        published = serve(example_server, example_fixture())
        directory = tmp_path / "tiny"
        seen: list[object] = []
        original = fetch.download_archive

        def watching(
            example: Example,
            archive: ExampleArchive,
            partial_directory: Path,
            checksums: Mapping[str, str],
            *,
            quiet: bool = False,
        ) -> Path:
            seen.append(json.loads((directory / ".mosaic-example.lock").read_text()))
            return original(example, archive, partial_directory, checksums, quiet=quiet)

        monkeypatch.setattr(fetch, "download_archive", watching)

        fetch_tiny(published, directory)

        assert seen == [{"pid": os.getpid(), "host": socket.gethostname()}]


class TestArchiveWithoutDataset:
    def test_a_dataset_archive_without_dataset_yaml_is_refused_before_any_rename(
        self, example_server: ExampleServer, tmp_path: Path
    ) -> None:
        archive = tar_gz({"tiny-tracks/tracks/a.txt": b"a"})
        published = serve(
            example_server,
            example_fixture(
                [GeneratedArchive(ARCHIVE_FILE, "tiny-tracks", archive, True)]
            ),
        )
        directory = tmp_path / "tiny"
        directory.mkdir()

        with pytest.raises(ArchiveWithoutDataset) as raised:
            fetch_tiny(published, directory)

        assert ARCHIVE_FILE in str(raised.value)
        assert "tiny-tracks/dataset.yaml" in str(raised.value)
        assert names_in(directory) == set()
