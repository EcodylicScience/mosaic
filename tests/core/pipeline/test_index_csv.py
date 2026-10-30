from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline import index_csv as index_csv_module
from mosaic.core.pipeline.index import FeatureIndexRow, feature_index
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    SchemaRowBase,
    finished_row_file,
    finished_rows,
    row_file_exists,
)
from mosaic.core.scope import Scope
from tests.helpers import index_locked_out, make_dataset


@dataclass(frozen=True, slots=True)
class SampleRow(RunIndexRowBase):
    name: str = ""
    status: str = ""
    value: int = 0


def _sample_row(tmp_path: Path, **overrides: object) -> SampleRow:
    """Create a SampleRow with a real abs_path in tmp_path."""
    p = tmp_path / "data.parquet"
    p.touch(exist_ok=True)
    defaults: dict[str, object] = {
        "run_id": "r1",
        "abs_path": str(p),
        "name": "",
        "status": "",
        "value": 0,
    }
    defaults.update(overrides)
    return SampleRow(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    return tmp_path / "index.csv"


# --- Schema ---


class TestEnsure:
    def test_creates_file_with_columns(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        assert tmp_csv.exists()
        df = pd.read_csv(tmp_csv)
        expected = [f.name for f in dataclasses.fields(SampleRow)]
        assert list(df.columns) == expected
        assert len(df) == 0

    def test_idempotent(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        idx.ensure()
        df = pd.read_csv(tmp_csv)
        assert len(df) == 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "index.csv"
        idx = IndexCSV(deep, SampleRow)
        idx.ensure()
        assert deep.exists()

    def test_headers_the_zero_byte_file_the_lock_creates(self, tmp_csv: Path) -> None:
        """``index_lock`` creates an absent index empty, and ``ensure`` headers it."""
        tmp_csv.touch()
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        assert list(pd.read_csv(tmp_csv, nrows=0).columns) == list(idx.schema)

    def test_leaves_an_index_with_content_untouched(
        self, tmp_csv: Path, tmp_path: Path
    ) -> None:
        """A rewrite replaces the inode even when the bytes come out the same."""
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.append([_sample_row(tmp_path, name="kept")])
        inode, content = tmp_csv.stat().st_ino, tmp_csv.read_bytes()
        idx.ensure()
        assert (tmp_csv.stat().st_ino, tmp_csv.read_bytes()) == (inode, content)


class TestAppend:
    def test_append_creates_if_missing(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.append([_sample_row(tmp_path, name="foo", value=1, status="ok")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "foo"

    def test_append_adds_rows(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        idx.append([_sample_row(tmp_path, name="a", value=1, status="ok")])
        idx.append([_sample_row(tmp_path, name="b", value=2, status="ok")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 2

    def test_append_fills_missing_keys(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.append([_sample_row(tmp_path, name="x")])
        df = pd.read_csv(tmp_csv)
        assert df.iloc[0]["name"] == "x"

    def test_append_is_atomic_on_failure(
        self, tmp_csv: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """A to_csv that raises mid-append leaves the prior CSV and no temp."""
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.append([_sample_row(tmp_path, name="a", value=1, status="ok")])

        def boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
        with pytest.raises(RuntimeError):
            idx.append([_sample_row(tmp_path, name="b", value=2, status="ok")])
        monkeypatch.undo()
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1  # prior content intact
        assert df.iloc[0]["name"] == "a"
        assert list(tmp_csv.parent.glob("*.tmp")) == []


class TestDedup:
    def test_dedup_by_keys(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow, dedup_keys=["name"])
        idx.append([_sample_row(tmp_path, name="a", value=1, status="v1")])
        idx.append([_sample_row(tmp_path, name="a", value=2, status="v2")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1
        assert df.iloc[0]["value"] == 2

    def test_dedup_composite_key(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow, dedup_keys=["name", "status"])
        idx.append([_sample_row(tmp_path, name="a", value=1, status="v1")])
        idx.append([_sample_row(tmp_path, name="a", value=2, status="v2")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 2  # different composite key, both kept

    def test_dedup_no_keys_means_no_dedup(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.append([_sample_row(tmp_path, name="a", value=1, status="ok")])
        idx.append([_sample_row(tmp_path, name="a", value=2, status="ok")])
        df = pd.read_csv(tmp_csv)
        assert len(df) == 2


class TestRead:
    def test_read_returns_dataframe(self, tmp_csv: Path, tmp_path: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.append([_sample_row(tmp_path, name="a", value=1, status="ok")])
        df = idx.read()
        assert len(df) == 1

    def test_read_empty(self, tmp_csv: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        idx.ensure()
        df = idx.read()
        assert len(df) == 0

    def test_read_missing_file_raises(self, tmp_csv: Path) -> None:
        idx = IndexCSV(tmp_csv, SampleRow)
        with pytest.raises(FileNotFoundError):
            idx.read()

    def test_filter_ext(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "index.csv"
        idx = IndexCSV(csv_path, SampleRow)

        parquet_file = tmp_path / "data.parquet"
        parquet_file.touch()
        npz_file = tmp_path / "data.npz"
        npz_file.touch()

        idx.append(
            [
                _sample_row(tmp_path, name="pq", abs_path=str(parquet_file)),
                _sample_row(tmp_path, name="nz", abs_path=str(npz_file)),
            ]
        )
        df_all = idx.read()
        assert len(df_all) == 2

        df_pq = idx.read(filter_ext=".parquet")
        assert len(df_pq) == 1
        assert df_pq.iloc[0]["name"] == "pq"

    def test_stale_path_raises_when_validated(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "index.csv"
        idx = IndexCSV(csv_path, SampleRow)
        p = tmp_path / "will_delete.parquet"
        p.touch()
        idx.append([_sample_row(tmp_path, name="x", abs_path=str(p))])
        p.unlink()
        # Opt-in validation still raises...
        with pytest.raises(FileNotFoundError, match="Stale index"):
            idx.read(validate_paths=True)
        # ...but the default (validate_paths=False) does not: dataset-aware
        # callers resolve + check existence themselves via resolve_path.
        assert len(idx.read()) == 1

    def test_prune_missing(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "index.csv"
        idx = IndexCSV(csv_path, SampleRow)
        present = tmp_path / "present.parquet"
        present.touch()
        missing = tmp_path / "missing.parquet"
        missing.touch()
        idx.append(
            [
                _sample_row(tmp_path, name="keep", abs_path=str(present)),
                _sample_row(tmp_path, name="drop", abs_path=str(missing)),
            ]
        )
        missing.unlink()

        # dry_run reports the drop without rewriting.
        dropped = idx.prune_missing(Path, dry_run=True)
        assert len(dropped) == 1
        assert dropped.iloc[0]["name"] == "drop"
        assert len(idx.read()) == 2  # unchanged on disk

        # apply: only the missing row is removed; the present one survives.
        dropped = idx.prune_missing(Path)
        assert len(dropped) == 1
        remaining = idx.read()
        assert len(remaining) == 1
        assert remaining.iloc[0]["name"] == "keep"

    def test_prune_missing_keeps_relocated(self, tmp_path: Path) -> None:
        # A resolver that maps a stored (relative) path under a *different*
        # root simulates a moved/synced dataset: the file exists there, so the
        # row must be kept, not pruned.
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir(parents=True)
        (root_b / "features").mkdir(parents=True)
        real = root_b / "features" / "x.parquet"
        real.touch()
        csv_path = root_a / "index.csv"
        idx = IndexCSV(csv_path, SampleRow)
        # Store a relative path; the file does NOT exist under root_a.
        idx.append([_sample_row(root_a, name="reloc", abs_path="features/x.parquet")])

        dropped = idx.prune_missing(lambda p: root_b / p)
        assert len(dropped) == 0
        assert len(idx.read()) == 1


# --- Feature Index ---


class TestFeatureIndex:
    def test_schema_has_required_columns(self) -> None:
        names = {f.name for f in dataclasses.fields(FeatureIndexRow)}
        assert "feature" in names
        assert "run_id" in names
        assert "n_rows" in names
        assert "finished_at" in names

    def test_factory_returns_index_csv(self, tmp_path: Path) -> None:
        idx = feature_index(tmp_path / "index.csv")
        assert isinstance(idx, IndexCSV)

    def test_ensure_creates_with_all_columns(self, tmp_path: Path) -> None:
        idx = feature_index(tmp_path / "index.csv")
        idx.ensure()
        df = pd.read_csv(tmp_path / "index.csv")
        assert "feature" in df.columns

    def test_dedup_by_run_group_sequence(self, tmp_path: Path) -> None:
        idx = feature_index(tmp_path / "index.csv")
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="v1-abc",
            feature="speed",
            version="v1",
            group="a",
            sequence="s1",
            abs_path=str(p),
            params_hash="h",
            n_rows=10,
        )
        idx.append([row])
        row2 = FeatureIndexRow(
            run_id="v1-abc",
            feature="speed",
            version="v1",
            group="a",
            sequence="s1",
            abs_path=str(p),
            params_hash="h",
            n_rows=20,
        )
        idx.append([row2])
        df = idx.read()
        assert len(df) == 1
        assert df.iloc[0]["n_rows"] == 20


class TestFeatureIndexRow:
    def test_fields_match_schema(self, tmp_path: Path) -> None:
        p = tmp_path / "G1__S1.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="abc123",
            feature="speed",
            version="0.1",
            group="G1",
            sequence="S1",
            abs_path=str(p),
            n_rows=100,
            params_hash="deadbeef",
        )
        df = pd.DataFrame([row])
        assert set(df.columns) == {f.name for f in dataclasses.fields(FeatureIndexRow)}
        assert df.iloc[0]["feature"] == "speed"
        assert df.iloc[0]["n_rows"] == 100
        assert df.iloc[0]["finished_at"] == ""

    def test_finished_at_default(self, tmp_path: Path) -> None:
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="r",
            feature="f",
            version="v",
            group="",
            sequence="s",
            abs_path=str(p),
            n_rows=0,
            params_hash="h",
        )
        assert row.finished_at == ""

    def test_started_at_auto_populated(self, tmp_path: Path) -> None:
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="r",
            feature="f",
            version="v",
            group="",
            sequence="s",
            abs_path=str(p),
            params_hash="h",
        )
        assert row.started_at != ""
        assert "T" in row.started_at  # ISO format

    def test_empty_abs_path_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            FeatureIndexRow(
                run_id="r",
                feature="f",
                version="v",
                group="",
                sequence="s",
                abs_path="",
                params_hash="h",
            )

    def test_nonexistent_abs_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            FeatureIndexRow(
                run_id="r",
                feature="f",
                version="v",
                group="",
                sequence="s",
                abs_path="/no/such/file.parquet",
                params_hash="h",
            )

    def test_relative_abs_path_accepted(self) -> None:
        # Relative paths are the portable storage form; they carry no dataset
        # context here, so the existence check is skipped (validated later at
        # resolve time). Construction must NOT raise even though the relative
        # path does not exist relative to CWD.
        row = FeatureIndexRow(
            run_id="r",
            feature="f",
            version="v",
            group="",
            sequence="s",
            abs_path="features/f/v-hash/g__s.parquet",
            params_hash="h",
        )
        assert not row.abs_path.is_absolute()

    def test_appendable_to_feature_index(self, tmp_path: Path) -> None:
        idx = feature_index(tmp_path / "index.csv")
        p = tmp_path / "data.parquet"
        p.touch()
        row = FeatureIndexRow(
            run_id="abc",
            feature="speed",
            version="0.1",
            group="G1",
            sequence="S1",
            abs_path=str(p),
            n_rows=10,
            params_hash="hash",
        )
        idx.append([row])
        df = idx.read()
        assert len(df) == 1
        assert df.iloc[0]["feature"] == "speed"


# --- latest_run_id ---


def _write_run_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal index CSV with explicit timestamps for sorting tests."""
    # Use FeatureIndexRow schema but write raw CSV to control timestamps
    p = path.parent / "data.parquet"
    p.touch(exist_ok=True)
    base_fields = {
        "feature": "f",
        "version": "v",
        "group": "g",
        "sequence": "s",
        "abs_path": str(p),
        "params_hash": "h",
        "n_rows": "0",
    }
    all_rows = [{**base_fields, **r} for r in rows]
    pd.DataFrame(all_rows).to_csv(path, index=False)


class TestLatestRunId:
    def test_returns_latest_finished(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "index.csv"
        _write_run_csv(
            csv_path,
            [
                {
                    "run_id": "old",
                    "started_at": "2025-01-01T00:00:00",
                    "finished_at": "2025-01-01T01:00:00",
                },
                {
                    "run_id": "new",
                    "started_at": "2025-01-02T00:00:00",
                    "finished_at": "2025-01-02T01:00:00",
                },
            ],
        )
        idx = IndexCSV(csv_path, FeatureIndexRow)
        assert idx.latest_run_id() == "new"

    def test_prefers_finished_over_unfinished(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "index.csv"
        _write_run_csv(
            csv_path,
            [
                {
                    "run_id": "finished",
                    "started_at": "2025-01-01T00:00:00",
                    "finished_at": "2025-01-01T01:00:00",
                },
                {
                    "run_id": "in_progress",
                    "started_at": "2025-06-01T00:00:00",
                    "finished_at": "",
                },
            ],
        )
        idx = IndexCSV(csv_path, FeatureIndexRow)
        assert idx.latest_run_id() == "finished"

    def test_falls_back_to_started_at(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "index.csv"
        _write_run_csv(
            csv_path,
            [
                {
                    "run_id": "old",
                    "started_at": "2025-01-01T00:00:00",
                    "finished_at": "",
                },
                {
                    "run_id": "new",
                    "started_at": "2025-06-01T00:00:00",
                    "finished_at": "",
                },
            ],
        )
        idx = IndexCSV(csv_path, FeatureIndexRow)
        assert idx.latest_run_id() == "new"

    def test_empty_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "index.csv"
        idx = IndexCSV(csv_path, FeatureIndexRow)
        idx.ensure()
        with pytest.raises(ValueError, match="No runs found"):
            idx.latest_run_id()


# --- string columns stay strings -------------------------------------------
#
# Every write path here reads the whole CSV and writes it back, so a column the
# row class declares a ``str`` but pandas infers numerically is not merely read
# wrong -- it is *rewritten* wrong, and the corrected value is what the next
# reader sees.


def _feature_row(
    tmp_path: Path,
    *,
    run_id: str = "0.1-aaaaaaaaaa",
    version: str = "0.1",
    group: str = "",
    sequence: str = "s",
    params_hash: str = "aaaaaaaaaa",
) -> FeatureIndexRow:
    """A FeatureIndexRow whose abs_path exists, with the string fields overridable."""
    path = tmp_path / f"{sequence}.parquet"
    path.touch(exist_ok=True)
    return FeatureIndexRow(
        abs_path=path,
        run_id=run_id,
        feature="feat",
        version=version,
        group=group,
        sequence=sequence,
        params_hash=params_hash,
        n_rows=5,
    )


class TestStringColumnsStayStrings:
    def test_zero_padded_names_survive_append_mark_finished_read(
        self, tmp_path: Path
    ) -> None:
        """A padded name must not be renumbered by a round trip through pandas.

        ``mark_finished`` is included deliberately: it rewrites the file too, so
        fixing only ``read``/``_append_locked`` would leave the corruption to a
        later call rather than removing it.
        """
        idx = feature_index(tmp_path / "index.csv")
        idx.append([_feature_row(tmp_path, group="01", sequence="001")])
        idx.mark_finished("0.1-aaaaaaaaaa")

        # Asserted against the file, not the frame: the defect is that the
        # rewrite persists the re-inferred value, so what is on disk is the claim.
        assert ",01,001," in (tmp_path / "index.csv").read_text()

    def test_a_dotted_version_is_not_rounded_on_rewrite(self, tmp_path: Path) -> None:
        """``0.10`` must not become ``0.1`` -- and collide with a real ``0.1``.

        The row's own ``run_id`` embeds the version, so an inferred rewrite makes
        the index contradict its own identifier.
        """
        idx = feature_index(tmp_path / "index.csv")
        idx.append(
            [
                _feature_row(
                    tmp_path, version="0.10", run_id="0.10-aaaaaaaaaa", sequence="a"
                )
            ]
        )
        idx.append([_feature_row(tmp_path, version="0.2", sequence="b")])

        text = (tmp_path / "index.csv").read_text()
        assert ",0.10-aaaaaaaaaa," in text and ",0.10," in text

    def test_a_leading_zero_hash_is_not_truncated_on_rewrite(
        self, tmp_path: Path
    ) -> None:
        """``params_hash`` is a SHA1 prefix; ~15% of them start with a zero."""
        idx = feature_index(tmp_path / "index.csv")
        idx.append(
            [
                _feature_row(
                    tmp_path,
                    params_hash="0123456789",
                    run_id="0.1-0123456789",
                    sequence="a",
                )
            ]
        )
        idx.append([_feature_row(tmp_path, sequence="b")])

        text = (tmp_path / "index.csv").read_text()
        assert ",0.1-0123456789," in text and ",0123456789," in text

    def test_an_identical_rerun_replaces_a_numeric_named_row(
        self, tmp_path: Path
    ) -> None:
        """Dedup must fire for numeric names -- the CalMS21/MABe convention.

        Inferred, the existing cell is ``int64 1`` and the incoming one is
        ``"1"``, so every dedup key comparison is False and the index grows by a
        row per re-run instead of staying put.
        """
        idx = feature_index(tmp_path / "index.csv")
        for _ in range(3):
            idx.append(
                [
                    _feature_row(tmp_path, sequence="1"),
                    _feature_row(tmp_path, sequence="2"),
                ]
            )

        assert len(idx.read()) == 2

    def test_numeric_names_still_match_the_scope_filters(self, tmp_path: Path) -> None:
        """``isin`` and the entry-tuple filter compare against Python strings."""
        idx = feature_index(tmp_path / "index.csv")
        idx.append([_feature_row(tmp_path, group="01", sequence="001")])

        assert len(idx.read(scope=Scope(sequences=["001"]))) == 1
        assert len(idx.read(scope=Scope(groups=["01"]))) == 1
        assert len(idx.read(scope=Scope(entries=[("01", "001")]))) == 1
        assert idx.ordered_entries() == [("01", "001")]


# --- the adopt hook ---------------------------------------------------------
#
# An index whose on-disk schema predates its row class has to be brought forward
# somewhere. The only safe place is inside the write lock, in memory -- see
# ``index_lock``'s one-atomic_write-per-block invariant.


@dataclass(frozen=True, slots=True)
class WideRow(RunIndexRowBase):
    """A row class with a column an older on-disk file would not have."""

    group: str = ""
    sequence: str = ""
    producer: str = ""


def _wide_row(tmp_path: Path, sequence: str, producer: str = "p") -> WideRow:
    path = tmp_path / f"{sequence}.parquet"
    path.touch(exist_ok=True)
    return WideRow(
        abs_path=path, run_id="r1", group="", sequence=sequence, producer=producer
    )


def _adopt_to(
    schema: list[str], probe: str = ""
) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], list[bool]]:
    """An adopt callable that projects onto *schema*, plus a presence probe.

    The projection is the minimal shape of a real adoption. *probe* records, per
    call, whether that column was already on the frame handed to the hook -- which
    is how a test pins *when* the hook runs relative to the dedup backfill,
    without asserting on a pandas column index.
    """
    saw_probe: list[bool] = []

    def adopt(df: pd.DataFrame) -> pd.DataFrame:
        saw_probe.append(probe in df.columns)
        for column in schema:
            if column not in df.columns:
                df[column] = ""
        return df[schema]

    return adopt, saw_probe


class TestAdoptHook:
    def test_absent_by_default_so_the_other_families_are_untouched(
        self, tmp_path: Path
    ) -> None:
        idx = IndexCSV(tmp_path / "index.csv", WideRow, dedup_keys=["sequence"])
        assert idx.adopt is None
        idx.append([_wide_row(tmp_path, "a")])
        assert list(idx.read()["sequence"]) == ["a"]

    def test_brings_a_narrower_on_disk_schema_up_to_the_row_class(
        self, tmp_path: Path
    ) -> None:
        """The whole point: a legacy file gains the columns it never had."""
        csv_path = tmp_path / "index.csv"
        csv_path.write_text("group,sequence,abs_path\n,legacy,legacy.parquet\n")
        schema = [f.name for f in dataclasses.fields(WideRow)]
        adopt, _ = _adopt_to(schema)

        idx = IndexCSV(csv_path, WideRow, dedup_keys=["sequence"], adopt=adopt)
        idx.append([_wide_row(tmp_path, "fresh")])

        header, legacy, fresh = csv_path.read_text().splitlines()
        assert header.split(",") == schema
        # The legacy row keeps its identity and gains an honest empty producer.
        assert legacy.endswith("legacy.parquet,,,,,legacy,")
        assert fresh.endswith(",fresh,p")

    def test_runs_before_the_dedup_backfill(self, tmp_path: Path) -> None:
        """Adoption must see the raw frame, not one already stamped by dedup.

        ``sequence`` is the dedup key and is absent from this file, so if the
        backfill ran first it would stamp `""` into a column adoption would then
        have nothing honest to fill.
        """
        csv_path = tmp_path / "index.csv"
        csv_path.write_text("group,abs_path\n,legacy.parquet\n")
        adopt, saw_dedup_key = _adopt_to(
            [f.name for f in dataclasses.fields(WideRow)], probe="sequence"
        )

        idx = IndexCSV(csv_path, WideRow, dedup_keys=["sequence"], adopt=adopt)
        idx.append([_wide_row(tmp_path, "fresh")])

        assert saw_dedup_key == [False]

    def test_leaves_no_temp_file_and_writes_once(self, tmp_path: Path) -> None:
        """One atomic_write per append -- adoption and rows commit together.

        No longer an ``index_lock`` requirement (the lock is a sidecar no rename
        touches) but an atomicity one: a second write would publish the adopted
        schema without this append's rows, and a reader landing between the two
        would see it. A stray temp file is the visible symptom of a block that
        died between writes.
        """
        csv_path = tmp_path / "index.csv"
        csv_path.write_text("group,sequence,abs_path\n,legacy,legacy.parquet\n")
        adopt, _ = _adopt_to([f.name for f in dataclasses.fields(WideRow)])

        idx = IndexCSV(csv_path, WideRow, dedup_keys=["sequence"], adopt=adopt)
        idx.append([_wide_row(tmp_path, "fresh")])

        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []

    def test_a_schema_complete_adoption_unblocks_the_run_helpers(
        self, tmp_path: Path
    ) -> None:
        """list_runs/latest_run_id/mark_finished need run_id AND finished_at.

        This is why the contract is "every column", not "the ones you care
        about": a partial adoption still leaves these three raising KeyError.
        """
        csv_path = tmp_path / "index.csv"
        csv_path.write_text("group,sequence,abs_path\n,legacy,legacy.parquet\n")
        adopt, _ = _adopt_to([f.name for f in dataclasses.fields(WideRow)])

        idx = IndexCSV(csv_path, WideRow, dedup_keys=["sequence"], adopt=adopt)
        idx.append([_wide_row(tmp_path, "fresh")])

        assert len(idx.list_runs()) == 2
        assert idx.latest_run_id() == "r1"
        idx.mark_finished("r1")
        assert list(idx.read(run_id="r1")["sequence"]) == ["fresh"]


@dataclass(frozen=True, slots=True)
class PathlessRow(SchemaRowBase):
    """A row that names a sequence rather than a file -- the 4.4 shape."""

    group: str = ""
    sequence: str = ""
    composition: str = ""
    member_count: int = 0


class TestARowNeedNotNameAFile:
    """``IndexCSV`` over a row class with no ``abs_path``."""

    def test_ensure_append_and_read_work_without_a_path_column(
        self, tmp_path: Path
    ) -> None:
        idx: IndexCSV[PathlessRow] = IndexCSV(
            tmp_path / "sequences.csv", PathlessRow, dedup_keys=["group", "sequence"]
        )
        idx.append([PathlessRow(sequence="a", composition="abc", member_count=2)])
        idx.append([PathlessRow(sequence="a", composition="def", member_count=3)])
        idx.append([PathlessRow(sequence="b", composition="ghi", member_count=1)])

        frame = idx.read()
        assert dict(zip(frame["sequence"], frame["composition"])) == {
            "a": "def",
            "b": "ghi",
        }

    def test_a_path_question_is_refused_rather_than_key_errored(
        self, tmp_path: Path
    ) -> None:
        """The message names what was asked for, not a missing pandas column."""
        idx: IndexCSV[PathlessRow] = IndexCSV(tmp_path / "sequences.csv", PathlessRow)
        idx.append([PathlessRow(sequence="a")])

        with pytest.raises(TypeError, match="has no abs_path"):
            _ = idx.read(filter_ext=".parquet")
        with pytest.raises(TypeError, match="has no abs_path"):
            _ = idx.read(validate_paths=True)
        with pytest.raises(TypeError, match="has no abs_path"):
            _ = idx.prune_missing(Path)

    def test_prune_missing_cannot_delete_a_composition_baseline(
        self, tmp_path: Path
    ) -> None:
        """The reason the row has no path at all.

        Had it borrowed the sequence's directory to satisfy ``IndexRowBase``,
        a moved or unmounted directory would drop the stored drift baseline
        rule P3 says must be kept -- silently, and reported as "nothing was
        dropped".
        """
        idx: IndexCSV[PathlessRow] = IndexCSV(tmp_path / "sequences.csv", PathlessRow)
        idx.append([PathlessRow(sequence="a", composition="abc")])
        with pytest.raises(TypeError):
            _ = idx.prune_missing(lambda _: tmp_path / "gone")
        assert list(idx.read()["composition"]) == ["abc"]


class TestReplace:
    def test_replace_writes_exactly_the_rows_given(self, tmp_path: Path) -> None:
        """A projection, not an accumulation: what is gone must leave."""
        idx: IndexCSV[PathlessRow] = IndexCSV(
            tmp_path / "sequences.csv", PathlessRow, dedup_keys=["group", "sequence"]
        )
        idx.append([PathlessRow(sequence="a"), PathlessRow(sequence="b")])

        idx.replace([PathlessRow(sequence="b", composition="new")])

        frame = idx.read()
        assert list(frame["sequence"]) == ["b"]
        assert list(frame["composition"]) == ["new"]

    def test_replace_with_nothing_leaves_a_headered_empty_file(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "sequences.csv"
        idx: IndexCSV[PathlessRow] = IndexCSV(path, PathlessRow)
        idx.append([PathlessRow(sequence="a")])

        idx.replace([])

        assert idx.read().empty
        assert list(pd.read_csv(path, nrows=0).columns) == list(idx.schema)

    def test_replace_projects_onto_the_schema(self, tmp_path: Path) -> None:
        """Column order is fixed and an off-schema key never widens the file."""
        path = tmp_path / "sequences.csv"
        idx: IndexCSV[PathlessRow] = IndexCSV(path, PathlessRow)
        idx.replace([PathlessRow(sequence="a", member_count=3)])
        assert list(pd.read_csv(path, nrows=0).columns) == list(idx.schema)

    def test_replace_leaves_no_temp_orphan(self, tmp_path: Path) -> None:
        idx: IndexCSV[PathlessRow] = IndexCSV(tmp_path / "sequences.csv", PathlessRow)
        idx.replace([PathlessRow(sequence="a")])
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


class TestDropRuns:
    """The index half of deleting a run, and the reason it is not a bare rewrite."""

    def test_it_drops_only_the_named_runs(self, tmp_path: Path) -> None:
        idx = feature_index(tmp_path / "index.csv")
        idx.append([_feature_row(tmp_path, run_id="0.1-aaaaaaaaaa", sequence="a")])
        idx.append([_feature_row(tmp_path, run_id="0.1-bbbbbbbbbb", sequence="b")])
        idx.append([_feature_row(tmp_path, run_id="0.1-cccccccccc", sequence="c")])

        dropped = idx.drop_runs({"0.1-bbbbbbbbbb"})

        assert list(dropped["sequence"]) == ["b"]
        assert sorted(idx.read()["sequence"]) == ["a", "c"]

    def test_an_unknown_run_drops_nothing_and_rewrites_nothing(
        self, tmp_path: Path
    ) -> None:
        """A caller may name runs it is unsure about; the file must not move."""
        idx = feature_index(tmp_path / "index.csv")
        idx.append([_feature_row(tmp_path, sequence="a")])
        before = (tmp_path / "index.csv").read_bytes()

        dropped = idx.drop_runs({"0.1-nosuchrun"})

        assert dropped.empty
        assert (tmp_path / "index.csv").read_bytes() == before

    def test_an_empty_request_touches_nothing(self, tmp_path: Path) -> None:
        idx = feature_index(tmp_path / "index.csv")
        idx.append([_feature_row(tmp_path, sequence="a")])
        before = (tmp_path / "index.csv").read_bytes()

        assert idx.drop_runs(set()).empty
        assert (tmp_path / "index.csv").read_bytes() == before

    def test_a_dry_run_reports_without_writing(self, tmp_path: Path) -> None:
        idx = feature_index(tmp_path / "index.csv")
        idx.append([_feature_row(tmp_path, run_id="0.1-aaaaaaaaaa", sequence="a")])
        before = (tmp_path / "index.csv").read_bytes()

        dropped = idx.drop_runs({"0.1-aaaaaaaaaa"}, dry_run=True)

        assert list(dropped["sequence"]) == ["a"]
        assert (tmp_path / "index.csv").read_bytes() == before

    def test_a_surviving_row_keeps_its_string_cells(self, tmp_path: Path) -> None:
        """The corruption a bare ``pd.read_csv`` round trip caused.

        ``0.10`` re-parses as ``0.1`` and a leading-zero digest loses its zero,
        while the ``run_id`` that embeds both keeps the original spelling -- an
        index contradicting its own identifiers, which is the failure the dtype
        map exists to prevent. Dropping a *sibling* row is what forces the
        rewrite, so the surviving row is collateral.
        """
        idx = feature_index(tmp_path / "index.csv")
        idx.append(
            [
                _feature_row(
                    tmp_path,
                    version="0.10",
                    params_hash="0123456789",
                    run_id="0.10-0123456789",
                    sequence="keeper",
                )
            ]
        )
        idx.append([_feature_row(tmp_path, run_id="0.1-doomed0000", sequence="goner")])

        _ = idx.drop_runs({"0.1-doomed0000"})

        text = (tmp_path / "index.csv").read_text()
        assert ",0.10-0123456789," in text
        assert ",0.10," in text and ",0123456789," in text


class TestUpdateCells:
    """Rewriting named cells of one run's rows, as one locked read-modify-write."""

    def _two_runs(self, tmp_path: Path) -> IndexCSV[SampleRow]:
        idx = IndexCSV(tmp_path / "index.csv", SampleRow)
        idx.append(
            [
                _sample_row(tmp_path, run_id="r1", name="a", value=1),
                _sample_row(tmp_path, run_id="r2", name="b", value=2),
            ]
        )
        return idx

    def test_it_rewrites_the_cells_of_one_run_only(self, tmp_path: Path) -> None:
        idx = self._two_runs(tmp_path)
        before = idx.read().set_index("run_id")

        matched = idx.update_cells("r1", {"status": "finished"})

        after = idx.read().set_index("run_id")
        assert matched == 1
        assert after.loc["r1", "status"] == "finished"
        assert after.loc["r1", "name"] == "a", "a cell not named is kept"
        assert after.loc["r2"].to_dict() == before.loc["r2"].to_dict()

    def test_the_rows_of_other_runs_stay_byte_identical(self, tmp_path: Path) -> None:
        idx = self._two_runs(tmp_path)
        lines_before = idx.path.read_text().splitlines()

        _ = idx.update_cells("r1", {"status": "finished"})

        lines_after = idx.path.read_text().splitlines()
        assert lines_after[0] == lines_before[0]
        assert lines_after[2] == lines_before[2]
        assert lines_after[1] != lines_before[1]

    def test_an_unknown_column_is_refused_and_nothing_is_written(
        self, tmp_path: Path
    ) -> None:
        idx = self._two_runs(tmp_path)
        before = idx.path.read_bytes()

        with pytest.raises(ValueError, match="no_such_column"):
            _ = idx.update_cells("r1", {"no_such_column": "x"})

        assert idx.path.read_bytes() == before

    def test_an_index_without_run_ids_is_refused_by_name(self, tmp_path: Path) -> None:
        idx: IndexCSV[PathlessRow] = IndexCSV(tmp_path / "sequences.csv", PathlessRow)
        idx.append([PathlessRow(sequence="a")])

        with pytest.raises(TypeError, match="not a run index row type"):
            _ = idx.update_cells("r1", {"composition": "x"})

    def test_an_unknown_run_or_an_absent_index_matches_nothing(
        self, tmp_path: Path
    ) -> None:
        idx = self._two_runs(tmp_path)
        before = idx.path.read_bytes()

        assert idx.update_cells("nosuchrun", {"status": "x"}) == 0
        assert idx.path.read_bytes() == before

        absent = IndexCSV(tmp_path / "absent.csv", SampleRow)
        assert absent.update_cells("r1", {"status": "x"}) == 0
        assert not absent.path.exists(), "no lock, no empty index"

    def test_it_writes_under_the_index_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        idx = self._two_runs(tmp_path)
        held: list[bool] = []
        real_write = index_csv_module.atomic_write

        def probing_write(path: Path, writer: Callable[[Path], None]) -> None:
            held.append(index_locked_out(idx.path))
            real_write(path, writer)

        monkeypatch.setattr(index_csv_module, "atomic_write", probing_write)

        _ = idx.update_cells("r1", {"status": "finished"})

        assert held == [True], "another process was locked out during the rewrite"

    def test_a_cell_keeps_the_spelling_of_its_string_siblings(
        self, tmp_path: Path
    ) -> None:
        idx = IndexCSV(tmp_path / "index.csv", SampleRow)
        idx.append(
            [
                _sample_row(tmp_path, run_id="r1", name="0123"),
                _sample_row(tmp_path, run_id="r2", name="0.10"),
            ]
        )

        _ = idx.update_cells("r1", {"status": "x"})

        text = idx.path.read_text()
        assert ",0123," in text and ",0.10," in text


# --- the file of a finished row ---------------------------------------------

FINISHED_RUN = "select-frames.0.1-0123456789"
MISSING_FILE_ADVICE = "`mosaic reindex --apply` drops a row whose file is gone."


def _finished_record(abs_path: str) -> dict[str, str]:
    return {
        "run_id": FINISHED_RUN,
        "abs_path": abs_path,
        "group": "G",
        "sequence": "S1",
    }


def test_a_finished_row_file_resolves_against_the_dataset(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    written = ds.base_dir / "runs" / "G__S1" / "selection.parquet"
    written.parent.mkdir(parents=True)
    written.touch()

    found = finished_row_file(
        ds, _finished_record("runs/G__S1/selection.parquet"), run_kind="select-frames"
    )

    assert found == written


def test_a_named_file_is_found_in_the_directory_of_its_row(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    written = ds.base_dir / "runs" / FINISHED_RUN / "summary.json"
    written.parent.mkdir(parents=True)
    written.touch()

    found = finished_row_file(
        ds,
        _finished_record(f"runs/{FINISHED_RUN}"),
        run_kind="score-predictions",
        name="summary.json",
    )

    assert found == written


def test_a_missing_file_names_the_run_the_key_and_the_path(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    gone = ds.base_dir / "runs" / "G__S1" / "selection.parquet"

    with pytest.raises(FileNotFoundError) as raised:
        _ = finished_row_file(
            ds,
            _finished_record("runs/G__S1/selection.parquet"),
            run_kind="select-frames",
            key_columns=("group", "sequence"),
        )

    assert str(raised.value) == (
        f"select-frames run {FINISHED_RUN!r} is recorded as finished for "
        f"(group='G', sequence='S1'), and {gone} is not on disk. "
        f"{MISSING_FILE_ADVICE}"
    )


def test_a_missing_file_of_an_unkeyed_row_names_the_run_and_the_path(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    gone = ds.base_dir / "runs" / FINISHED_RUN / "summary.json"
    gone.parent.mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as raised:
        _ = finished_row_file(
            ds,
            _finished_record(f"runs/{FINISHED_RUN}"),
            run_kind="score-predictions",
            name="summary.json",
        )

    assert str(raised.value) == (
        f"score-predictions run {FINISHED_RUN!r} is recorded as finished, and "
        f"{gone} is not on disk. {MISSING_FILE_ADVICE}"
    )


def test_a_directory_at_the_path_of_a_row_is_not_its_file(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    (ds.base_dir / "runs" / "G__S1").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=FINISHED_RUN):
        _ = finished_row_file(
            ds, _finished_record("runs/G__S1"), run_kind="select-frames"
        )


# --- the finished rows of a run ---------------------------------------------


def _run_record(run_id: str, sequence: str, *, finished: bool) -> dict[str, str]:
    return {
        "run_id": run_id,
        "abs_path": f"runs/G__{sequence}/selection.parquet",
        "group": "G",
        "sequence": sequence,
        "finished_at": "2026-10-25T00:00:00+00:00" if finished else "",
    }


def test_the_finished_rows_of_a_run_are_keyed_by_their_key_columns() -> None:
    records = [
        _run_record(FINISHED_RUN, "S1", finished=True),
        _run_record(FINISHED_RUN, "S2", finished=False),
        _run_record("select-frames.0.1-ffffffffff", "S3", finished=True),
    ]

    found = finished_rows(records, FINISHED_RUN, ("group", "sequence"))

    assert found == {("G", "S1"): records[0]}


def test_a_run_keyed_by_no_column_has_its_finished_row_under_the_empty_key() -> None:
    records = [
        _run_record("score-predictions.0.1-ffffffffff", "S0", finished=True),
        _run_record(FINISHED_RUN, "S1", finished=True),
    ]

    assert finished_rows(records, FINISHED_RUN, ()) == {(): records[1]}


def test_a_row_file_exists_only_where_its_path_resolves_to_a_file(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    written = ds.base_dir / "runs" / "G__S1" / "selection.parquet"
    written.parent.mkdir(parents=True)
    written.touch()
    (ds.base_dir / "runs" / "G__S2").mkdir()

    assert row_file_exists(ds, _finished_record("runs/G__S1/selection.parquet"))
    assert not row_file_exists(ds, _finished_record("runs/G__S2"))
    assert not row_file_exists(ds, _finished_record("runs/G__S3/selection.parquet"))
    assert not row_file_exists(ds, _finished_record(""))
    assert not row_file_exists(ds, None)
