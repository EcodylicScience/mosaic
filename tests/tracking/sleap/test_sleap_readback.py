"""The readback of tracked instances and their appearance vectors from a ``.slp``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mosaic.tracking.sleap.run as sleap_run
from mosaic.core.track_library.sleap import read_appearance_readback
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapError, read_back_embeddings
from tests.helpers import (
    ReadbackRecord,
    StubInstance,
    install_fake_tool_python,
    write_appearance_readback,
    write_stub_sleap_io,
    write_stub_slp,
)


def _answering(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    writes: bool = True,
    malformed: bool = False,
    published: Path | None = None,
) -> list[list[str]]:
    """Stand in for the SLEAP interpreter: write a readback to the last argument.

    A *malformed* readback has two rows of vectors for one record.
    """
    commands: list[list[str]] = []

    def invocation(*_args: object, **_kwargs: object) -> list[str]:
        return ["/fake/bin/python"]

    def run(cmd: list[str], **_kwargs: object) -> tuple[str, str, int]:
        commands.append(cmd)
        if writes:
            records = [ReadbackRecord(0, "track_0", vector=(1.0,))]
            _ = write_appearance_readback(Path(cmd[-1]), records, dim=1)
        if malformed:
            with Path(cmd[-1]).open("wb") as handle:
                np.savez(
                    handle,
                    frame_idx=np.zeros(1, dtype=np.int64),
                    track_name=np.asarray(["track_0"], dtype=np.str_),
                    is_user=np.zeros(1, dtype=np.bool_),
                    has_vector=np.ones(1, dtype=np.bool_),
                    vectors=np.zeros((2, 1), dtype=np.float32),
                )
        if published is not None:
            assert not published.exists()
        return ("", "the snippet failed", returncode)

    monkeypatch.setattr(sleap_run, "tool_invocation", invocation)
    monkeypatch.setattr(sleap_run, "run_supervised", run)
    return commands


def test_the_readback_takes_its_paths_in_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predicted = tmp_path / "e1.predictions.slp"
    out = tmp_path / "e1.embeddings.npz"
    commands = _answering(monkeypatch)

    published = read_back_embeddings(predicted, out)

    (cmd,) = commands
    assert cmd[:2] == ["/fake/bin/python", "-c"]
    assert cmd[3:] == [str(predicted), str(tmp_path / "e1.embeddings.partial.npz")]
    assert str(tmp_path) not in cmd[2]
    assert published == out


def test_the_file_is_published_only_once_the_snippet_has_written_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "e1.embeddings.npz"
    _ = out.write_bytes(b"stale")
    _ = _answering(monkeypatch, published=out)

    _ = read_back_embeddings(tmp_path / "e1.predictions.slp", out)

    assert read_appearance_readback(out).track_name.tolist() == ["track_0"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["e1.embeddings.npz"]


def test_a_failed_readback_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "e1.embeddings.npz"
    _ = out.write_bytes(b"stale")
    _ = _answering(monkeypatch, returncode=1)

    with pytest.raises(SleapError, match="the snippet failed"):
        _ = read_back_embeddings(tmp_path / "e1.predictions.slp", out)

    assert list(tmp_path.iterdir()) == []


def test_a_clean_exit_without_a_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "e1.embeddings.npz"
    _ = _answering(monkeypatch, writes=False)

    with pytest.raises(FileNotFoundError):
        _ = read_back_embeddings(tmp_path / "e1.predictions.slp", out)

    assert list(tmp_path.iterdir()) == []


def test_a_malformed_readback_raises_and_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "e1.embeddings.npz"
    _ = out.write_bytes(b"stale")
    _ = _answering(monkeypatch, writes=False, malformed=True)

    with pytest.raises(ValueError, match="vectors has shape"):
        _ = read_back_embeddings(tmp_path / "e1.predictions.slp", out)

    assert list(tmp_path.iterdir()) == []


def _stub_sleap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imports = write_stub_sleap_io(tmp_path / "stub")
    _ = install_fake_tool_python(
        monkeypatch, SLEAP_ENV, tmp_path / "bin", imports=imports
    )


def test_the_snippet_reads_every_tracked_instance_in_file_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    predicted = write_stub_slp(
        tmp_path / "work" / "e1.predictions.slp",
        [
            (
                5,
                [
                    StubInstance("b", embedding=(0.5, 0.5, 0.0)),
                    StubInstance(None, embedding=(9.0, 9.0, 9.0)),
                    StubInstance("a", predicted=False),
                ],
            ),
            (2, [StubInstance("a", embedding=(1.0, 0.0, -1.0))]),
        ],
    )
    out = tmp_path / "work" / "e1.embeddings.npz"

    _ = read_back_embeddings(predicted, out)

    readback = read_appearance_readback(out)
    assert readback.frame_idx.tolist() == [5, 5, 2]
    assert readback.track_name.tolist() == ["b", "a", "a"]
    assert readback.is_user.tolist() == [False, True, False]
    assert readback.has_vector.tolist() == [True, False, True]
    assert readback.vectors.tolist() == [
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, -1.0],
    ]
    assert not (tmp_path / "work" / "e1.embeddings.partial.npz").exists()


def test_the_snippet_writes_zero_columns_for_a_file_without_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    predicted = write_stub_slp(
        tmp_path / "e1.predictions.slp",
        [(0, [StubInstance("a"), StubInstance("b", predicted=False)])],
    )
    out = tmp_path / "e1.embeddings.npz"

    _ = read_back_embeddings(predicted, out)

    readback = read_appearance_readback(out)
    assert readback.track_name.tolist() == ["a", "b"]
    assert readback.vectors.shape == (2, 0)
    assert not readback.has_vector.any()


def test_the_snippet_refuses_vectors_of_two_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sleap(monkeypatch, tmp_path)
    predicted = write_stub_slp(
        tmp_path / "e1.predictions.slp",
        [
            (
                0,
                [
                    StubInstance("a", embedding=(1.0,)),
                    StubInstance("b", embedding=(1.0, 2.0)),
                ],
            )
        ],
    )
    out = tmp_path / "e1.embeddings.npz"

    with pytest.raises(SleapError, match="lengths"):
        _ = read_back_embeddings(predicted, out)

    assert not out.exists()
