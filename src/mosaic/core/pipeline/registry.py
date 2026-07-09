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
import sys
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

CREATE TABLE IF NOT EXISTS runs (
    execution_id   TEXT    NOT NULL,
    kind           TEXT    NOT NULL,
    target         TEXT    NOT NULL,
    run_id         TEXT    DEFAULT '',
    status         TEXT    NOT NULL,
    owner          TEXT    DEFAULT '',
    host           TEXT    DEFAULT '',
    pid            INTEGER DEFAULT 0,
    created_at     TEXT    NOT NULL,
    started_at     TEXT    DEFAULT '',
    heartbeat_at   TEXT    DEFAULT '',
    finished_at    TEXT    DEFAULT '',
    error_json     TEXT    DEFAULT '',
    progress_done  INTEGER DEFAULT 0,
    progress_total INTEGER DEFAULT 0,
    PRIMARY KEY (execution_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs (run_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_target ON runs (kind, target);
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

    # -- runs attempt ledger ------------------------------------------------
    #
    # The ``runs`` table records one row per execution *attempt*, keyed by a
    # ULID ``execution_id``. This is deliberately distinct from ``feature_runs``
    # (the content-addressed *result* ledger, keyed by ``run_id``): a retry or a
    # cache hit is a new attempt (new ``execution_id``) but may share -- or, for
    # a not-yet-computed / trex run, lack -- a ``run_id``. Status/error/heartbeat
    # live here so ``feature_runs`` stays a pure content ledger with no failure
    # history to overwrite.

    def record_attempt(
        self,
        execution_id: str,
        kind: str,
        target: str,
        *,
        owner: str = "",
        host: str = "",
        pid: int = 0,
        run_id: str = "",
        status: str = "running",
        progress_total: int = 0,
    ) -> None:
        """Insert (or claim) a run attempt row and mark it started.

        Upserts on ``execution_id`` so an attempt pre-created elsewhere (e.g. a
        ``queued`` row inserted by the API before it spawns the subprocess) is
        flipped to ``running`` when the library actually begins work.
        ``created_at`` is preserved across the conflict; ``started_at`` is set
        each time the attempt (re)enters this method.
        """
        now = now_iso()
        self._conn.execute(
            """\
            INSERT INTO runs
                (execution_id, kind, target, run_id, status, owner, host, pid,
                 created_at, started_at, heartbeat_at, progress_total)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(execution_id) DO UPDATE SET
                status = excluded.status,
                started_at = excluded.started_at,
                heartbeat_at = excluded.heartbeat_at,
                host = excluded.host,
                pid = excluded.pid,
                run_id = CASE WHEN excluded.run_id != '' THEN excluded.run_id
                              ELSE runs.run_id END,
                progress_total = excluded.progress_total
            """,
            (
                execution_id, kind, target, run_id, status, owner, host, pid,
                now, now, now, progress_total,
            ),
        )
        self._conn.commit()

    def set_attempt_run_id(self, execution_id: str, run_id: str) -> None:
        """Backfill the content-addressed ``run_id`` once the job computes it."""
        self._conn.execute(
            "UPDATE runs SET run_id = ? WHERE execution_id = ?",
            (run_id, execution_id),
        )
        self._conn.commit()

    def heartbeat_attempt(
        self,
        execution_id: str,
        *,
        progress_done: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        """Refresh ``heartbeat_at`` (and optionally progress counters).

        A running attempt whose heartbeat goes stale is how a supervisor
        (Layer-2 runs-sweeper) detects a dead worker and reclaims the row.
        """
        self._conn.execute(
            """\
            UPDATE runs SET
                heartbeat_at = ?,
                progress_done = COALESCE(?, progress_done),
                progress_total = COALESCE(?, progress_total)
            WHERE execution_id = ?
            """,
            (now_iso(), progress_done, progress_total, execution_id),
        )
        self._conn.commit()

    def finish_attempt(
        self,
        execution_id: str,
        status: str,
        *,
        error_json: str = "",
    ) -> None:
        """Record a terminal state (``finished``/``failed``/``cancelled``)."""
        now = now_iso()
        self._conn.execute(
            """\
            UPDATE runs SET
                status = ?, finished_at = ?, heartbeat_at = ?, error_json = ?
            WHERE execution_id = ?
            """,
            (status, now, now, error_json, execution_id),
        )
        self._conn.commit()

    def get_attempt(self, execution_id: str) -> dict[str, object] | None:
        """Return the attempt row as a dict, or ``None`` if absent."""
        cur = self._conn.execute(
            "SELECT * FROM runs WHERE execution_id = ?", (execution_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def list_attempts(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        target: str | None = None,
    ) -> pd.DataFrame:
        """Return run attempts (newest first), optionally filtered."""
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if target is not None:
            clauses.append("target = ?")
            params.append(target)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return pd.read_sql_query(
            f"SELECT * FROM runs {where} ORDER BY execution_id DESC",
            self._conn,
            params=params,
        )

    def stale_running_attempts(self, older_than_iso: str) -> list[dict[str, object]]:
        """Return ``running`` attempts whose heartbeat predates *older_than_iso*."""
        cur = self._conn.execute(
            """\
            SELECT * FROM runs
            WHERE status = 'running'
              AND heartbeat_at != ''
              AND heartbeat_at < ?
            ORDER BY heartbeat_at
            """,
            (older_than_iso,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

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
            # stderr, not stdout: a `mosaic run --json` invocation on a legacy
            # CSV-only dataset must keep stdout a single clean JSON object.
            print(
                f"[registry] migrated {n} entries from CSV indices into {db_path}",
                file=sys.stderr,
            )
    return reg


# ---------------------------------------------------------------------------
# Standalone attempt readers (for external tools -- no mosaic import required)
# ---------------------------------------------------------------------------
#
# These mirror ``progress.read_progress``: they open a short-lived WAL
# connection, query the ``runs`` table, and close. They let mosaic-api's
# runs-sweeper / status endpoints read attempt state (and reclaim stale ones)
# without constructing a ``FeatureRegistry`` or importing the rest of mosaic.


def _query_runs(db_path: Path, sql: str, params: tuple[object, ...]) -> list[dict[str, object]]:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def read_run(db_path: Path, execution_id: str) -> dict[str, object] | None:
    """Read one run attempt by ``execution_id`` (or ``None`` if absent)."""
    rows = _query_runs(
        db_path, "SELECT * FROM runs WHERE execution_id = ?", (execution_id,)
    )
    return rows[0] if rows else None


def read_runs(
    db_path: Path,
    *,
    kind: str | None = None,
    status: str | None = None,
) -> list[dict[str, object]]:
    """Read run attempts (newest first), optionally filtered by kind/status."""
    clauses: list[str] = []
    params: list[object] = []
    if kind is not None:
        clauses.append("kind = ?")
        params.append(kind)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return _query_runs(
        db_path, f"SELECT * FROM runs {where} ORDER BY execution_id DESC", tuple(params)
    )


def read_runs_by_run_id(db_path: Path, run_id: str) -> list[dict[str, object]]:
    """Read all attempts that produced (or targeted) a given content ``run_id``."""
    return _query_runs(
        db_path,
        "SELECT * FROM runs WHERE run_id = ? ORDER BY execution_id DESC",
        (run_id,),
    )
