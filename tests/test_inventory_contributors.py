"""The read choke point, and the seam the ops half arrives through.

``core`` does not import ``tracking``, so tracker runs, frame runs and trained
models reach an inventory by registration rather than by import. And every index
a scan reads is read once: mosaic holds no caching decorators at all, which is
why one ``reconcile`` call makes three full passes over every index it has.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline.inventory._read import IndexReader, IndexStamp
from mosaic.core.pipeline.inventory.contributors import (
    inventory_contributor,
    register_inventory_contributor,
    registered_inventory_kinds,
)


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


def test_an_absent_file_still_stamps(tmp_path: Path) -> None:
    """Its appearing later is a change, and recording nothing hides that."""
    stamp = IndexStamp.of(tmp_path / "never-written.csv")

    assert not stamp.exists
    assert stamp.size == 0


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
    offenders: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                ("mosaic.tracking", "mosaic.behavior")
            ):
                offenders.append(f"{source.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("mosaic.tracking", "mosaic.behavior")):
                        offenders.append(f"{source.name}:{node.lineno}")

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
