"""Index writes are serialized, across threads and across processes.

Rule P7 of the hashing contract: two writers to one index never lose a row
between them, never leave a file mixing two writers' sets, and never write
unlocked when the lock cannot be taken. The lock lives on a sidecar file, so
it survives the atomic writes it serializes.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.index_csv import IndexCSV, IndexRowBase
from mosaic.core.pipeline.index_lock import IndexLockTimeout, index_lock, lock_path_for


# --- P7: index writes are serialized ------------------------------------------


@dataclass(frozen=True)
class _Row(IndexRowBase):
    key: str


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
def test_concurrent_index_appends_do_not_lose_rows(
    tmp_path: Path, existing: bool
) -> None:
    """Two writers whose reads interleave must not silently drop one's write.

    Also run on an absent index, where the first write creates the file.
    """
    index: IndexCSV[_Row] = IndexCSV(tmp_path / "index.csv", _Row)
    if existing:
        index.ensure()

    ready = threading.Barrier(2)

    def writer(name: str) -> None:
        ready.wait(timeout=5)
        index.append([_Row(abs_path=Path(f"{name}.parquet"), key=name)])

    threads = [threading.Thread(target=writer, args=(n,)) for n in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # Read back through csv rather than the pandas accessor: the assertion is
    # about which rows survived, and stdlib csv keeps it typed.
    with (tmp_path / "index.csv").open(newline="") as handle:
        written = {row["key"] for row in csv.DictReader(handle)}
    assert written == {"first", "second"}, f"a concurrent append was lost: {written}"


_APPEND_PROBE = """
import sys
from pathlib import Path
from dataclasses import dataclass
from mosaic.core.pipeline.index_csv import IndexCSV, IndexRowBase

@dataclass(frozen=True)
class Row(IndexRowBase):
    key: str

path, name, barrier = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
index = IndexCSV(path, Row)
# Read the whole file first and only then append, which is the interleaving a
# lock has to prevent -- without one, both writers compute a merged frame from
# the same starting state and the second erases the first. An absent index has
# nothing to read, and the first write creates it.
if path.exists():
    _ = index.read()
barrier.write_text("ready")
while len(list(barrier.parent.glob("*.ready"))) < 2:
    pass
index.append([Row(abs_path=Path(name + ".parquet"), key=name)])
"""


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "absent"])
def test_concurrent_index_appends_across_processes_do_not_lose_rows(
    tmp_path: Path, existing: bool
) -> None:
    """The real contention is between processes, not threads.

    The thread test above is satisfied by a plain ``threading.Lock``, which
    leaves the case that actually happens -- two queue workers, or two
    ``mosaic run`` invocations, in separate interpreters -- completely
    unprotected. Only a file lock covers this one.
    """
    index_path = tmp_path / "index.csv"
    index: IndexCSV[_Row] = IndexCSV(index_path, _Row)
    if existing:
        index.ensure()

    gate = tmp_path / "gate"
    gate.mkdir()

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _APPEND_PROBE,
                str(index_path),
                name,
                str(gate / f"{name}.ready"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for name in ("first", "second")
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err.decode()[-800:]

    with index_path.open(newline="") as handle:
        written = {row["key"] for row in csv.DictReader(handle)}
    assert written == {"first", "second"}, f"a concurrent append was lost: {written}"


_RIVAL_APPEND = """
import sys
from pathlib import Path
from dataclasses import dataclass
from mosaic.core.pipeline.index_csv import IndexCSV, IndexRowBase
from mosaic.core.pipeline.index_lock import IndexLockTimeout, index_lock

@dataclass(frozen=True)
class Row(IndexRowBase):
    key: str

path = Path(sys.argv[1])
# One line tells the writer that started this one when to resume: "waiting" when
# the lock is held, else "appended" once the row is written.
try:
    with index_lock(path, timeout=0):
        pass
except IndexLockTimeout:
    print("waiting", flush=True)
IndexCSV(path, Row).append([Row(abs_path=Path("rival.parquet"), key="rival")])
print("appended", flush=True)
"""


@pytest.mark.parametrize(
    "ensure_first", [False, True], ids=["append", "ensure-then-append"]
)
def test_a_first_write_cannot_erase_a_racing_writers_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ensure_first: bool
) -> None:
    """Creating an index is a write, and the lock serializes it with the rest.

    The first write to a fresh index starts a rival writer in another process
    and resumes on the rival's first line: it is waiting for the lock, or it has
    appended. The interleaving therefore does not depend on timing. A creation
    outside the lock lets the rival append inside that window, and the header
    written afterwards replaces the rival's row.
    """
    index_path = tmp_path / "index.csv"
    index: IndexCSV[_Row] = IndexCSV(index_path, _Row)
    rivals: list[subprocess.Popen[str]] = []

    def write_after_a_rival(
        final_path: Path, write_fn: Callable[[Path], object]
    ) -> None:
        if not rivals:
            rival = subprocess.Popen(
                [sys.executable, "-c", _RIVAL_APPEND, str(index_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            rivals.append(rival)
            assert rival.stdout is not None
            _ = rival.stdout.readline()
        atomic_write(final_path, write_fn)

    monkeypatch.setattr(
        "mosaic.core.pipeline.index_csv.atomic_write", write_after_a_rival
    )
    if ensure_first:
        index.ensure()
    index.append([_Row(abs_path=Path("first.parquet"), key="first")])

    (rival,) = rivals
    _, err = rival.communicate(timeout=60)
    assert rival.returncode == 0, err[-800:]
    with index_path.open(newline="") as handle:
        written = {row["key"] for row in csv.DictReader(handle)}
    assert written == {"first", "rival"}, f"the first write erased a row: {written}"


def test_a_failed_lock_acquisition_raises_rather_than_writing_unlocked(
    tmp_path: Path,
) -> None:
    """Contention must fail loudly, never degrade to an unlocked write.

    Degrading is what the lock exists to prevent, and it would degrade exactly
    when the system is busiest.
    """
    index_path = tmp_path / "index.csv"
    index: IndexCSV[_Row] = IndexCSV(index_path, _Row)
    index.ensure()

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pathlib import Path\n"
            "from mosaic.core.pipeline.index_lock import index_lock\n"
            "with index_lock(Path(sys.argv[1])):\n"
            "    Path(sys.argv[2]).write_text('held')\n"
            "    time.sleep(30)\n",
            str(index_path),
            str(tmp_path / "held"),
        ]
    )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "held").exists():
            assert time.monotonic() < deadline, "holder never acquired the lock"
            time.sleep(0.02)

        with pytest.raises(IndexLockTimeout):
            with index_lock(index_path, timeout=0.5):
                pass  # pragma: no cover - the raise is the assertion
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_the_index_lock_is_reentrant_within_a_thread() -> None:
    """``IndexCSV.append`` calls ``ensure``; both take the lock.

    A non-re-entrant file lock would deadlock against itself here, with no
    error and no timeout distinguishable from real contention.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.csv"
        with index_lock(path, timeout=5):
            with index_lock(path, timeout=5):
                pass


# --- P7: the lock survives the write it serializes ---------------------------
#
# The property the sidecar exists for, and the one the previous design silently
# lacked. A subprocess rather than a thread throughout: ``index_lock`` serializes
# threads with its own ``RLock`` and is re-entrant within one, so a thread would
# block whether or not the *file* lock survived -- which is the thing measured.

_LOCK_PROBE = """
import sys
from pathlib import Path
from mosaic.core.pipeline.index_lock import IndexLockTimeout, index_lock

try:
    with index_lock(Path(sys.argv[1]), timeout=float(sys.argv[2])):
        pass
except IndexLockTimeout:
    sys.exit(0)   # still held by the parent
sys.exit(3)       # acquired: the parent's grip is gone
"""


def _locked_out(index_path: Path, timeout: float = 1.0) -> bool:
    """Does a separate process fail to acquire *index_path*'s lock right now?"""
    proc = subprocess.run(
        [sys.executable, "-c", _LOCK_PROBE, str(index_path), str(timeout)],
        capture_output=True,
        timeout=120,
    )
    if proc.returncode not in (0, 3):
        raise AssertionError(proc.stderr.decode()[-800:])
    return proc.returncode == 0


def test_the_lock_survives_the_atomic_writes_it_serializes(tmp_path: Path) -> None:
    """A locked block may rewrite its index as often as it likes.

    The regression the sidecar exists for, and it reproduces on any POSIX
    machine. Held on the *index* inode, the first ``atomic_write`` renamed a new
    inode over the path and the block silently lost the lock it thought it had:
    a second process opened the new inode, flocked it uncontended, and
    interleaved. Two writes here, and the probe must still be shut out after
    both -- with the final assertion proving the earlier one is not vacuous.
    """
    index_path = tmp_path / "index.csv"

    with index_lock(index_path, timeout=10):
        lock_file = lock_path_for(index_path.resolve())
        held_inode = lock_file.stat().st_ino

        atomic_write(index_path, lambda p: p.write_text("key\nfirst\n"))
        atomic_write(index_path, lambda p: p.write_text("key\nsecond\n"))

        # Same inode, so nothing replaced the file the lock is on...
        assert lock_file.stat().st_ino == held_inode
        # ...and, decisively, another process still cannot take it.
        assert _locked_out(index_path), (
            "an atomic_write dropped the block's lock: a concurrent writer can "
            "now interleave with a block that believes it holds it"
        )

    assert index_path.read_text() == "key\nsecond\n"
    # The probe can succeed when the lock is free, so the assertion above is
    # about the lock and not about the probe never working.
    assert not _locked_out(index_path), "the lock outlived its block"


def test_the_lock_file_is_created_once_and_never_removed(tmp_path: Path) -> None:
    """Unlinking a lock file reopens the race the sidecar closed.

    A holder's inode would go away while it still held it, the next process
    would create a fresh one at the same name and flock it uncontended, and both
    would write. Pinned as an *inode*, not merely as a name: a
    delete-and-recreate leaves the name in place and is the same bug.
    """
    index_path = tmp_path / "index.csv"
    lock_file = lock_path_for(index_path.resolve())

    with index_lock(index_path, timeout=10):
        assert lock_file.exists()
        first_inode = lock_file.stat().st_ino

    assert lock_file.exists(), "the lock file was removed on release"
    with index_lock(index_path, timeout=10):
        pass
    assert lock_file.stat().st_ino == first_inode, "the lock file was recreated"


def test_acquiring_still_creates_the_index_empty(tmp_path: Path) -> None:
    """The side effect that index readers and writers are built around.

    ``mark_finished`` checks for the index before the lock because acquiring
    creates the file. ``IndexCSV`` reads a zero-byte index as empty, and
    ``IndexCSV.ensure`` writes its header under the lock.
    ``load_media_index_frame`` and ``Dataset._read_media_index`` read a zero-byte
    index as empty rather than raising. Moving the lock off the index inode
    removed the ``O_CREAT`` that produced it. The side effect is now deliberate,
    and this test pins it.
    """
    index_path = tmp_path / "nested" / "index.csv"
    assert not index_path.exists()

    with index_lock(index_path, timeout=10):
        pass

    assert index_path.exists() and index_path.stat().st_size == 0
    assert sorted(p.name for p in index_path.parent.iterdir()) == [
        "index.csv",
        "index.csv.lock",
    ]


def test_two_routes_to_one_index_take_one_lock(tmp_path: Path) -> None:
    """The sidecar is derived from the resolved path, not the path as written.

    Two callers reaching one index by different routes must name one lock file.
    Derived from the unresolved path instead, ``sub/../index.csv`` and
    ``index.csv`` would each get their own and serialize against nothing -- the
    exact failure a temp-directory lock keyed on ``$TMPDIR`` has, reintroduced
    locally. ``..`` rather than a symlink so this needs no privilege on Windows.
    """
    (tmp_path / "sub").mkdir()
    direct = tmp_path / "index.csv"
    roundabout = tmp_path / "sub" / ".." / "index.csv"

    with index_lock(direct, timeout=10):
        assert _locked_out(roundabout)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "index.csv",
        "index.csv.lock",
        "sub",
    ]


# --- P7: a replace never tears the file ---------------------------------------

_REPLACE_PROBE = """
import sys
from pathlib import Path
from dataclasses import dataclass
from mosaic.core.pipeline.index_csv import IndexCSV, SchemaRowBase

@dataclass(frozen=True)
class Row(SchemaRowBase):
    key: str = ""
    value: str = ""

path, name, barrier = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
index = IndexCSV(path, Row)
index.ensure()
barrier.write_text("ready")
while len(list(barrier.parent.glob("*.ready"))) < 2:
    pass
# Each writer declares a whole set of its own, several rows wide, so a torn or
# interleaved write shows up as a file mixing the two.
index.replace([Row(key=name, value=name) for _ in range(200)])
"""


def test_concurrent_index_replaces_never_tear_the_file(tmp_path: Path) -> None:
    """``replace`` serializes, so a reader always sees one writer's whole set.

    Deliberately NOT a lost-update test. ``rows`` is computed by the caller
    before the lock is taken, so two processes each declaring a different whole
    set have the last one win, entire -- which is what a projection means. What
    the lock guarantees is that the file is never a mixture of the two, and never
    half-written: every surviving row belongs to one writer, and the file parses.
    """
    index_path = tmp_path / "index.csv"
    gate = tmp_path / "gate"
    gate.mkdir()

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _REPLACE_PROBE,
                str(index_path),
                name,
                str(gate / f"{name}.ready"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for name in ("first", "second")
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err.decode()[-800:]

    with index_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Every row carries the writer that produced it, so a mixture is detectable.
    writers = {row["value"] for row in rows}
    assert len(writers) == 1, f"the file mixes two writers' sets: {writers}"
    assert {row["key"] for row in rows} == writers
    assert len(rows) == 200, "a partial set survived"
