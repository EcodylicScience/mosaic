"""The read choke point, and the seam the ops half arrives through.

``core`` does not import ``tracking``, so tracker runs, frame runs and trained
models reach an inventory by registration rather than by import. And every index
a scan reads is read once: mosaic holds no caching decorators at all, which is
why one ``reconcile`` call makes three full passes over every index it has.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.entry import CameraEntry, Entry
from mosaic.core.pipeline.inventory._read import IndexReader
from mosaic.core.pipeline.inventory._runs import keyed_run_records, run_keyed_record
from mosaic.core.pipeline.inventory.contributors import (
    inventory_contributor,
    register_inventory_contributor,
    registered_inventory_kinds,
)
from mosaic.core.pipeline.inventory.model import (
    ArtifactRecord,
    FrameRunRef,
    InventoryScope,
    PreparedDatasetRef,
)
from mosaic.core.scope import Scope

from tests.helpers import imports_of, make_dataset


# --- one read per index per scan ---------------------------------------------


def test_one_index_is_read_once_however_many_callers_want_it(tmp_path: Path) -> None:
    """The defect this exists to stop: three passes over one file in one call."""
    index = tmp_path / "index.csv"
    index.write_text("run_id\nabc\n")
    calls: list[int] = []

    def _read() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"run_id": ["abc"]})

    reader = IndexReader()
    first = reader.frame(index, _read)
    second = reader.frame(index, _read)

    assert len(calls) == 1
    assert first.equals(second)


def test_two_routes_to_one_file_share_the_read(tmp_path: Path) -> None:
    """Keyed on the resolved path, so a symlinked root does not read twice --
    and, worse, does not disagree with itself within one scan."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "index.csv").write_text("run_id\nabc\n")
    link = tmp_path / "link"
    link.symlink_to(real)
    calls: list[int] = []

    def _read() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"run_id": ["abc"]})

    reader = IndexReader()
    _ = reader.frame(real / "index.csv", _read)
    _ = reader.frame(link / "index.csv", _read)

    assert len(calls) == 1


def test_a_reader_that_raises_reads_as_empty(tmp_path: Path) -> None:
    """Each pass decides what an absent index means, and for all of them it is
    "nothing here" rather than an exception out of the middle of a scan."""

    def _explode() -> pd.DataFrame:
        raise OSError("gone")

    assert IndexReader().frame(tmp_path / "index.csv", _explode).empty


def test_reading_stamps_what_it_touched(tmp_path: Path) -> None:
    """The stamps a long-lived holder revalidates against, collected as a
    by-product rather than by a second walk over the same files."""
    index = tmp_path / "index.csv"
    index.write_text("run_id\nabc\n")
    reader = IndexReader()

    _ = reader.frame(index, lambda: pd.DataFrame())

    stamp = reader.stamps()[index.resolve()]
    assert stamp.exists
    assert stamp.size == len("run_id\nabc\n")


# --- the registry -------------------------------------------------------------


def test_a_contributor_registers_and_reads_back() -> None:
    def _none(ds: object, scope: object, reader: object) -> list[object]:
        return []

    register_inventory_contributor("tracker-run", _none)

    assert inventory_contributor("tracker-run") is _none
    assert "tracker-run" in registered_inventory_kinds()


def test_a_kind_nobody_registered_reads_none() -> None:
    """Reported by the scan as unavailable, never as zero artifacts: a caller
    that imported only ``mosaic.core`` has not imported the producers, and
    answering "no tracker runs" would be wrong rather than merely unhelpful."""
    assert inventory_contributor("labels-variant") is None


# --- and core still does not import tracking ---------------------------------


def test_the_inventory_names_no_tracking_module() -> None:
    """The layering constraint this whole seam exists to keep.

    Checked over the parse tree at every scope, ``TYPE_CHECKING`` included: an
    import that only the checker sees is still a statement that ``core`` knows
    what lives in ``tracking``, and it is the one that quietly becomes a runtime
    import later.
    """
    import mosaic.core.pipeline.inventory as package

    package_root = Path(package.__file__).parent
    offenders = sorted(
        {
            f"{source.name}:{imported.line}"
            for source in package_root.rglob("*.py")
            for imported in imports_of(source, scope="anywhere", relative=False)
            if imported.in_package("mosaic.tracking")
            or imported.in_package("mosaic.behavior")
        }
    )

    assert offenders == [], (
        f"the inventory reaches up into a domain package: {offenders}"
    )


@pytest.fixture(autouse=True)
def _restore_registry() -> object:
    """Registration is process-global, so a test that registers must undo it."""
    from mosaic.core.pipeline.inventory import contributors

    saved = dict(contributors._CONTRIBUTORS)
    yield
    contributors._CONTRIBUTORS.clear()
    contributors._CONTRIBUTORS.update(saved)


# --- the shared record builders -----------------------------------------------


def _write_index(base: Path, kind: str, rows: list[dict[str, str]]) -> None:
    folder = base / "things" / kind
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "index.csv", index=False)


def _keyed_rows(base: Path) -> list[dict[str, str]]:
    (base / "out").mkdir()
    (base / "out" / "a.bin").write_text("x")
    return [
        {
            "run_id": "r1",
            "group": "g",
            "sequence": "s1",
            "camera": "c1",
            "abs_path": "out/a.bin",
            "started_at": "t0",
            "finished_at": "t1",
        },
        {
            "run_id": "r1",
            "group": "g",
            "sequence": "s2",
            "camera": "c1",
            "abs_path": "out/missing.bin",
            "started_at": "t0",
            "finished_at": "",
        },
    ]


def _build_keyed(
    base: Path,
    drifted: Callable[[str, pd.DataFrame], tuple[Entry, ...]],
    selector: Scope | None = None,
) -> list[ArtifactRecord[CameraEntry]]:
    scope = InventoryScope(kinds=frozenset(), selector=selector or Scope())
    return keyed_run_records(
        make_dataset(base, roots=["things"]),
        scope,
        IndexReader(),
        root_key="things",
        kinds=None,
        read_index=lambda ds, kind: pd.read_csv(
            ds.get_root("things") / kind / "index.csv",
            dtype="string",
            keep_default_na=False,
        ),
        key_of=lambda row: (row["group"], row["sequence"], row["camera"]),
        make_ref=lambda kind, run_id, _rows: FrameRunRef(method=kind, run_id=run_id),
        drifted=drifted,
    )


def test_keyed_run_records_reads_coverage_from_rows_and_files(tmp_path: Path) -> None:
    _write_index(tmp_path, "m", _keyed_rows(tmp_path))

    (record,) = _build_keyed(tmp_path, lambda _run_id, _rows: ())

    assert record.run_id == "r1"
    assert record.name == "m"
    assert record.coverage.present == {("g", "s1", "c1")}
    assert record.orphan_rows == {("g", "s2", "c1")}
    assert record.drift == ()
    assert record.status == "inconsistent"
    assert record.started_at == "t0"
    assert record.finished_at == "t1"


def test_keyed_run_records_sets_drift_and_status_from_the_callable(
    tmp_path: Path,
) -> None:
    rows = _keyed_rows(tmp_path)
    rows[1]["abs_path"] = "out/a.bin"
    _write_index(tmp_path, "m", rows)
    seen: list[tuple[str, int]] = []

    def _drifted(run_id: str, run_rows: pd.DataFrame) -> tuple[Entry, ...]:
        seen.append((run_id, len(run_rows)))
        return (("g", "s2"),)

    (record,) = _build_keyed(tmp_path, _drifted)

    assert seen == [("r1", 2)]
    assert record.drift == (("g", "s2"),)
    assert record.status == "complete-but-drifted"


def test_keyed_run_records_narrows_to_the_scope_before_drift(tmp_path: Path) -> None:
    _write_index(tmp_path, "m", _keyed_rows(tmp_path))
    seen: list[int] = []

    def _drifted(_run_id: str, run_rows: pd.DataFrame) -> tuple[Entry, ...]:
        seen.append(len(run_rows))
        return ()

    (record,) = _build_keyed(tmp_path, _drifted, Scope(entries=[("g", "s1")]))

    assert seen == [1]
    assert record.rows == {("g", "s1", "c1")}


def test_keyed_run_records_reads_an_absent_root_as_nothing(tmp_path: Path) -> None:
    assert _build_keyed(tmp_path, lambda _run_id, _rows: ()) == []


def test_run_keyed_record_is_one_unit_of_its_own_run(tmp_path: Path) -> None:
    row = {"started_at": "t0", "finished_at": "t1", "status": "finished"}
    ref = PreparedDatasetRef(op_kind="k", run_id="r9")

    done = run_keyed_record(
        ref=ref,
        name="k",
        run_id="r9",
        row=row,
        recorded_finished=True,
        complete=True,
        index_path=tmp_path / "index.csv",
        run_root=tmp_path / "r9",
    )
    lost = run_keyed_record(
        ref=ref,
        name="k",
        run_id="r9",
        row=row,
        recorded_finished=True,
        complete=False,
        index_path=tmp_path / "index.csv",
        run_root=tmp_path / "r9",
    )

    assert done.coverage.present == {"r9"}
    assert done.status == "complete"
    assert done.finished_at == "t1"
    assert lost.coverage.present == frozenset()
    assert lost.rows == frozenset({"r9"})
    assert lost.status == "inconsistent"
