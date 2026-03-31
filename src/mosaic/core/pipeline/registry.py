"""SQLite-backed feature and training registry.

Replaces per-feature CSV indices with a single ``features/.mosaic.db``
database per dataset. WAL mode enables concurrent readers with a single
writer -- sufficient for workstation use with multiple users.

The schema is intentionally simple (plain TEXT/INTEGER columns) so that
external tools (mosaic-api, DB browsers) can read it without importing
mosaic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ._utils import now_iso

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS feature_runs (
    feature      TEXT    NOT NULL,
    run_id       TEXT    NOT NULL,
    version      TEXT    NOT NULL,
    params_hash  TEXT    NOT NULL,
    params_json  TEXT,
    inputs_json  TEXT,
    started_at   TEXT    NOT NULL,
    finished_at  TEXT    DEFAULT '',
    PRIMARY KEY (feature, run_id)
);

CREATE TABLE IF NOT EXISTS feature_entries (
    feature   TEXT    NOT NULL,
    run_id    TEXT    NOT NULL,
    group_    TEXT    NOT NULL,
    sequence  TEXT    NOT NULL,
    abs_path  TEXT    NOT NULL,
    n_rows    INTEGER DEFAULT 0,
    FOREIGN KEY (feature, run_id) REFERENCES feature_runs (feature, run_id),
    PRIMARY KEY (feature, run_id, group_, sequence)
);

CREATE TABLE IF NOT EXISTS dependencies (
    feature           TEXT NOT NULL,
    run_id            TEXT NOT NULL,
    upstream_feature  TEXT NOT NULL,
    upstream_run_id   TEXT NOT NULL,
    FOREIGN KEY (feature, run_id)
        REFERENCES feature_runs (feature, run_id),
    FOREIGN KEY (upstream_feature, upstream_run_id)
        REFERENCES feature_runs (feature, run_id),
    PRIMARY KEY (feature, run_id, upstream_feature)
);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id        TEXT PRIMARY KEY,
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    config_json   TEXT NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL DEFAULT 'pending',
    priority      INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    started_at    TEXT DEFAULT '',
    finished_at   TEXT DEFAULT '',
    run_id        TEXT DEFAULT '',
    error         TEXT DEFAULT '',
    worker_pid    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_progress (
    job_id      TEXT    NOT NULL,
    step_type   TEXT    NOT NULL,
    step_index  INTEGER NOT NULL,
    step_total  INTEGER DEFAULT 0,
    metric_json TEXT    DEFAULT '{}',
    message     TEXT    DEFAULT '',
    timestamp   TEXT    NOT NULL,
    PRIMARY KEY (job_id, step_type, step_index)
);
"""


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------


class FeatureRegistry:
    """Queryable SQLite registry for feature runs and entries.

    Parameters
    ----------
    db_path : Path
        Location of the ``.mosaic.db`` file (created if absent).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FeatureRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes -------------------------------------------------------------

    def record_run_start(
        self,
        feature: str,
        run_id: str,
        version: str,
        params_hash: str,
        params_json: str | None = None,
        inputs_json: str | None = None,
    ) -> None:
        """Record that a feature run has started."""
        self._conn.execute(
            """\
            INSERT OR IGNORE INTO feature_runs
                (feature, run_id, version, params_hash, params_json,
                 inputs_json, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (feature, run_id, version, params_hash, params_json, inputs_json, now_iso()),
        )
        self._conn.commit()

    def record_entry(
        self,
        feature: str,
        run_id: str,
        group: str,
        sequence: str,
        abs_path: str | Path,
        n_rows: int = 0,
    ) -> None:
        """Record a computed entry (one parquet file)."""
        self._conn.execute(
            """\
            INSERT OR REPLACE INTO feature_entries
                (feature, run_id, group_, sequence, abs_path, n_rows)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (feature, run_id, group, sequence, str(abs_path), n_rows),
        )
        self._conn.commit()

    def record_entries(
        self,
        rows: list[tuple[str, str, str, str, str | Path, int]],
    ) -> None:
        """Batch-record multiple entries in one transaction."""
        self._conn.executemany(
            """\
            INSERT OR REPLACE INTO feature_entries
                (feature, run_id, group_, sequence, abs_path, n_rows)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(f, r, g, s, str(p), n) for f, r, g, s, p, n in rows],
        )
        self._conn.commit()

    def mark_finished(self, feature: str, run_id: str) -> None:
        """Set ``finished_at`` on a run."""
        self._conn.execute(
            """\
            UPDATE feature_runs SET finished_at = ?
            WHERE feature = ? AND run_id = ? AND finished_at = ''
            """,
            (now_iso(), feature, run_id),
        )
        self._conn.commit()

    def record_dependency(
        self,
        feature: str,
        run_id: str,
        upstream_feature: str,
        upstream_run_id: str,
    ) -> None:
        """Record that *(feature, run_id)* consumed *(upstream_feature, upstream_run_id)*."""
        self._conn.execute(
            """\
            INSERT OR IGNORE INTO dependencies
                (feature, run_id, upstream_feature, upstream_run_id)
            VALUES (?, ?, ?, ?)
            """,
            (feature, run_id, upstream_feature, upstream_run_id),
        )
        self._conn.commit()

    # -- reads --------------------------------------------------------------

    def latest_run_id(self, feature: str) -> str:
        """Return most recent run_id for *feature* (prefers finished)."""
        row = self._conn.execute(
            """\
            SELECT run_id FROM feature_runs
            WHERE feature = ?
            ORDER BY
                CASE WHEN finished_at != '' THEN 0 ELSE 1 END,
                started_at DESC
            LIMIT 1
            """,
            (feature,),
        ).fetchone()
        if row is None:
            msg = f"No runs found for feature '{feature}'"
            raise ValueError(msg)
        return str(row[0])

    def list_runs(self, feature: str) -> pd.DataFrame:
        """Return all runs for *feature* as a DataFrame."""
        return pd.read_sql_query(
            """\
            SELECT * FROM feature_runs
            WHERE feature = ?
            ORDER BY
                CASE WHEN finished_at != '' THEN 0 ELSE 1 END,
                started_at DESC
            """,
            self._conn,
            params=(feature,),
        )

    def list_features(self) -> list[str]:
        """Return the names of all features that have at least one run."""
        rows = self._conn.execute(
            "SELECT DISTINCT feature FROM feature_runs ORDER BY feature"
        ).fetchall()
        return [r[0] for r in rows]

    def list_entries(
        self,
        feature: str,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """Return entries for *feature* (optionally filtered by run_id)."""
        if run_id is None:
            run_id = self.latest_run_id(feature)
        return pd.read_sql_query(
            """\
            SELECT * FROM feature_entries
            WHERE feature = ? AND run_id = ?
            ORDER BY group_, sequence
            """,
            self._conn,
            params=(feature, run_id),
        )

    def entry_count(self, feature: str, run_id: str) -> int:
        """Return the number of entries for a specific run."""
        row = self._conn.execute(
            """\
            SELECT COUNT(*) FROM feature_entries
            WHERE feature = ? AND run_id = ?
            """,
            (feature, run_id),
        ).fetchone()
        return int(row[0]) if row else 0

    def has_run(self, feature: str, run_id: str) -> bool:
        """Check whether a run exists in the registry."""
        row = self._conn.execute(
            """\
            SELECT 1 FROM feature_runs
            WHERE feature = ? AND run_id = ?
            LIMIT 1
            """,
            (feature, run_id),
        ).fetchone()
        return row is not None

    def run_is_finished(self, feature: str, run_id: str) -> bool:
        """Check whether a run has been marked finished."""
        row = self._conn.execute(
            """\
            SELECT finished_at FROM feature_runs
            WHERE feature = ? AND run_id = ?
            """,
            (feature, run_id),
        ).fetchone()
        return row is not None and row[0] != ""

    def pending_entries(
        self,
        feature: str,
        run_id: str,
        all_entries: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Return ``(group, sequence)`` pairs missing from a run.

        *all_entries* is the full set that should exist (e.g. from tracks).
        """
        existing = self._conn.execute(
            """\
            SELECT group_, sequence FROM feature_entries
            WHERE feature = ? AND run_id = ?
            """,
            (feature, run_id),
        ).fetchall()
        existing_set = {(r[0], r[1]) for r in existing}
        return sorted(all_entries - existing_set)

    def get_dependencies(
        self, feature: str, run_id: str
    ) -> list[tuple[str, str]]:
        """Return upstream ``(feature, run_id)`` pairs for a run."""
        rows = self._conn.execute(
            """\
            SELECT upstream_feature, upstream_run_id FROM dependencies
            WHERE feature = ? AND run_id = ?
            """,
            (feature, run_id),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def lineage(self, feature: str, run_id: str) -> dict[str, object]:
        """Return full dependency tree rooted at *(feature, run_id)*.

        Returns a nested dict: ``{"feature": ..., "run_id": ..., "upstream": [...]}``.
        """
        upstream = self.get_dependencies(feature, run_id)
        return {
            "feature": feature,
            "run_id": run_id,
            "upstream": [
                self.lineage(uf, ur) for uf, ur in upstream
            ],
        }

    # -- migration ----------------------------------------------------------

    def migrate_from_csv(self, features_root: Path) -> int:
        """Import existing ``index.csv`` files into the registry.

        Returns the number of entries imported. Safe to call multiple times --
        uses INSERT OR IGNORE so duplicates are skipped.
        """
        count = 0
        for csv_path in sorted(features_root.glob("*/index.csv")):
            feature_name = csv_path.parent.name
            try:
                df = pd.read_csv(csv_path, keep_default_na=False)
            except Exception:
                continue
            if df.empty:
                continue

            # Group by run_id to create feature_runs entries
            if "run_id" not in df.columns:
                continue

            for run_id, grp in df.groupby("run_id"):
                run_id_str = str(run_id)
                first = grp.iloc[0]
                version = str(first.get("version", ""))
                params_hash = str(first.get("params_hash", ""))
                started_at = str(first.get("started_at", now_iso()))
                finished_at = str(first.get("finished_at", ""))

                self._conn.execute(
                    """\
                    INSERT OR IGNORE INTO feature_runs
                        (feature, run_id, version, params_hash,
                         started_at, finished_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (feature_name, run_id_str, version, params_hash,
                     started_at, finished_at),
                )

                for _, row in grp.iterrows():
                    group = str(row.get("group", ""))
                    sequence = str(row.get("sequence", ""))
                    abs_path = str(row.get("abs_path", ""))
                    n_rows = int(row.get("n_rows", 0))
                    self._conn.execute(
                        """\
                        INSERT OR IGNORE INTO feature_entries
                            (feature, run_id, group_, sequence, abs_path, n_rows)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (feature_name, run_id_str, group, sequence, abs_path, n_rows),
                    )
                    count += 1

        self._conn.commit()
        return count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def open_registry(features_root: Path, *, migrate_csv: bool = True) -> FeatureRegistry:
    """Open (or create) the feature registry for a dataset.

    Parameters
    ----------
    features_root : Path
        The ``features/`` directory of the dataset.
    migrate_csv : bool
        If True and the DB is newly created, import existing ``index.csv`` files.
    """
    db_path = features_root / ".mosaic.db"
    is_new = not db_path.exists()
    reg = FeatureRegistry(db_path)
    if is_new and migrate_csv:
        n = reg.migrate_from_csv(features_root)
        if n > 0:
            print(f"[registry] migrated {n} entries from CSV indices into {db_path}")
    return reg
