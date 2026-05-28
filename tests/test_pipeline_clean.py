"""Regression tests for Pipeline.clean().

The shared-storage case: multiple FeatureSteps can point at the same
on-disk directory (same feature class + same input chain, different
params). clean() must keep the union of their expected run_ids — not
delete sibling steps' current directories.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from mosaic.core.pipeline import pipeline as pipeline_mod
from mosaic.core.pipeline.pipeline import CallbackStep, FeatureStep, Pipeline


@pytest.fixture
def fake_dataset():
    """Minimal placeholder — clean() only passes it through to helpers we patch."""
    return object()


def _make_run_dir(root, storage, rid):
    """Create a fake run directory with one parquet file."""
    d = root / storage / rid
    d.mkdir(parents=True)
    (d / "g1__s1.parquet").write_bytes(b"x")
    return d


def _patch_clean_helpers(monkeypatch, storage_root, runs_on_disk):
    """Wire feature_run_root / list_feature_runs / feature_index_path to tmp_path."""
    def fake_feature_run_root(_dataset, storage, rid):
        return storage_root / storage / rid

    def fake_list_feature_runs(_dataset, storage):
        rids = runs_on_disk.get(storage, [])
        return pd.DataFrame({"run_id": rids})

    def fake_feature_index_path(_dataset, storage):
        return storage_root / storage / "index.csv"

    monkeypatch.setattr(pipeline_mod, "feature_run_root", fake_feature_run_root)
    monkeypatch.setattr(pipeline_mod, "list_feature_runs", fake_list_feature_runs)
    monkeypatch.setattr(pipeline_mod, "feature_index_path", fake_feature_index_path)


def _resolved_step(name, storage, rid):
    """Build a fake resolved-step dict that clean() consumes."""
    return {
        "step": FeatureStep(name=name, feature_cls=type("F", (), {"name": storage})),
        "storage_name": storage,
        "expected_run_id": rid,
        "cached": True,
        "stale": False,
        "mock_result": None,
        "feature_short": storage,
    }


class TestCleanSharedStorage:
    """Two steps sharing one storage path must both be kept."""

    def test_dry_run_keeps_both_sibling_runs(
        self, tmp_path, monkeypatch, fake_dataset
    ):
        storage = "ffgroups__from__smooth__from__tracks"
        rid_a, rid_b = "0.1-aaaaaaaaaa", "0.1-bbbbbbbbbb"

        _make_run_dir(tmp_path, storage, rid_a)
        _make_run_dir(tmp_path, storage, rid_b)

        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [rid_a, rid_b]}
        )

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [
                _resolved_step("ff20", storage, rid_a),
                _resolved_step("ff10", storage, rid_b),
            ],
        )

        df = pipe.clean(fake_dataset, dry_run=True)

        statuses = dict(zip(df["run_id"], df["status"]))
        assert statuses[rid_a] == "current"
        assert statuses[rid_b] == "current"
        # Both step names should appear in the row label
        labels = set(df["step"].unique())
        assert labels == {"ff20, ff10"}
        # Files still on disk
        assert (tmp_path / storage / rid_a).exists()
        assert (tmp_path / storage / rid_b).exists()

    def test_actual_delete_only_removes_orphans(
        self, tmp_path, monkeypatch, fake_dataset
    ):
        storage = "ffgroups__from__smooth__from__tracks"
        rid_a = "0.1-aaaaaaaaaa"
        rid_b = "0.1-bbbbbbbbbb"
        rid_orphan = "0.1-cccccccccc"

        _make_run_dir(tmp_path, storage, rid_a)
        _make_run_dir(tmp_path, storage, rid_b)
        _make_run_dir(tmp_path, storage, rid_orphan)

        # Write an index csv with rows for all three rids
        idx_path = tmp_path / storage / "index.csv"
        pd.DataFrame({
            "run_id": [rid_a, rid_b, rid_orphan, "0.1-deadbeef00"],
            "group": ["g1"] * 4,
            "sequence": ["s1"] * 4,
        }).to_csv(idx_path, index=False)

        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [rid_a, rid_b, rid_orphan]}
        )

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [
                _resolved_step("ff20", storage, rid_a),
                _resolved_step("ff10", storage, rid_b),
            ],
        )

        df = pipe.clean(fake_dataset, dry_run=False)

        statuses = dict(zip(df["run_id"], df["status"]))
        assert statuses[rid_a] == "current"
        assert statuses[rid_b] == "current"
        assert statuses[rid_orphan] == "removed"

        # Only the orphan directory was removed
        assert (tmp_path / storage / rid_a).exists()
        assert (tmp_path / storage / rid_b).exists()
        assert not (tmp_path / storage / rid_orphan).exists()

        # Index now keeps rid_a and rid_b. The unrelated "0.1-deadbeef00"
        # row is also kept (clean() only touches rids it saw on disk).
        idx_after = pd.read_csv(idx_path)
        rid_set = set(idx_after["run_id"])
        assert rid_a in rid_set
        assert rid_b in rid_set
        assert rid_orphan not in rid_set
        assert "0.1-deadbeef00" in rid_set

    def test_single_step_unchanged_behavior(
        self, tmp_path, monkeypatch, fake_dataset
    ):
        """Single-step case should still treat non-matching rids as orphans."""
        storage = "speed-angvel__from__smooth__from__tracks"
        keep_rid = "0.1-1111111111"
        orphan_rid = "0.1-2222222222"

        _make_run_dir(tmp_path, storage, keep_rid)
        _make_run_dir(tmp_path, storage, orphan_rid)

        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [keep_rid, orphan_rid]}
        )

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [_resolved_step("speed", storage, keep_rid)],
        )

        df = pipe.clean(fake_dataset, dry_run=True)
        statuses = dict(zip(df["run_id"], df["status"]))
        assert statuses[keep_rid] == "current"
        assert statuses[orphan_rid] == "would remove"

    def test_callback_steps_skipped(
        self, tmp_path, monkeypatch, fake_dataset
    ):
        """CallbackSteps in resolved list should not contribute to clean()."""
        storage = "speed-angvel__from__smooth__from__tracks"
        keep_rid = "0.1-1111111111"
        _make_run_dir(tmp_path, storage, keep_rid)
        _patch_clean_helpers(monkeypatch, tmp_path, {storage: [keep_rid]})

        pipe = Pipeline()
        cb_info = {
            "step": CallbackStep(name="my_cb", fn=lambda _d, _r: None),
            "storage_name": None,
            "expected_run_id": None,
            "cached": None,
            "stale": False,
            "mock_result": None,
        }
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [
                cb_info,
                _resolved_step("speed", storage, keep_rid),
            ],
        )

        df = pipe.clean(fake_dataset, dry_run=True)
        # Only the feature step contributes a row
        assert len(df) == 1
        assert df.iloc[0]["run_id"] == keep_rid
        assert df.iloc[0]["status"] == "current"


class TestCleanSafeguards:
    """Defensive checks that catch silent data loss from a broken keep-set.

    The pre-fix bug (and any future regression that bypasses the
    storage-grouping logic) would delete a sibling step's "current" run.
    These tests verify the safeguards raise instead of silently destroying.
    """

    def test_global_keeper_conflict_raises(
        self, tmp_path, monkeypatch, fake_dataset
    ):
        """Safeguard #1: simulate the pre-fix bug via a broken keep-set."""
        storage = "ffgroups__from__smooth__from__tracks"
        rid_a, rid_b = "0.1-aaaaaaaaaa", "0.1-bbbbbbbbbb"

        _make_run_dir(tmp_path, storage, rid_a)
        _make_run_dir(tmp_path, storage, rid_b)

        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [rid_a, rid_b]}
        )

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [
                _resolved_step("ff_a", storage, rid_a),
                _resolved_step("ff_b", storage, rid_b),
            ],
        )
        # Simulate the pre-fix bug: keep-set drops rid_b. global_keepers
        # is built directly from `resolved` so still has both → mismatch.
        monkeypatch.setattr(
            pipe,
            "_build_clean_keep_sets",
            lambda _resolved: ({storage: {rid_a}}, {storage: ["ff_a", "ff_b"]}),
        )

        with pytest.raises(RuntimeError, match="expect it as current"):
            pipe.clean(fake_dataset, dry_run=False)

        # Both directories still on disk — the raise happened before deletion.
        assert (tmp_path / storage / rid_a).exists()
        assert (tmp_path / storage / rid_b).exists()

    def test_post_clean_existence_check_raises(
        self, tmp_path, monkeypatch, fake_dataset
    ):
        """Safeguard #3: if rmtree nukes a keeper, post-check catches it."""
        storage = "ffgroups__from__smooth__from__tracks"
        rid_a, rid_b = "0.1-aaaaaaaaaa", "0.1-bbbbbbbbbb"

        _make_run_dir(tmp_path, storage, rid_a)
        _make_run_dir(tmp_path, storage, rid_b)

        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [rid_a, rid_b]}
        )

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [
                _resolved_step("ff_a", storage, rid_a),
                _resolved_step("ff_b", storage, rid_b),
            ],
        )

        # Sabotage: rmtree nukes EVERY run directory under storage, not
        # just the one it was asked to remove. Both keepers vanish.
        storage_dir = tmp_path / storage
        real_rmtree = shutil.rmtree

        def evil_rmtree(_path, *args, **kwargs):
            for rd in list(storage_dir.iterdir()):
                if rd.is_dir():
                    real_rmtree(rd)

        # No orphans exist, so clean() won't actually call rmtree on its own.
        # Force a call by adding a fake orphan rid to the on-disk listing.
        rid_orphan = "0.1-cccccccccc"
        _make_run_dir(tmp_path, storage, rid_orphan)
        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [rid_a, rid_b, rid_orphan]}
        )
        monkeypatch.setattr(shutil, "rmtree", evil_rmtree)

        with pytest.raises(RuntimeError, match="should have been kept"):
            pipe.clean(fake_dataset, dry_run=False)

    def test_shared_storage_notice_printed(
        self, tmp_path, monkeypatch, capsys, fake_dataset
    ):
        """Safeguard #2: shared-storage steps get a visible notice line."""
        storage = "ffgroups__from__smooth__from__tracks"
        rid_a, rid_b = "0.1-aaaaaaaaaa", "0.1-bbbbbbbbbb"

        _make_run_dir(tmp_path, storage, rid_a)
        _make_run_dir(tmp_path, storage, rid_b)
        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [rid_a, rid_b]}
        )

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [
                _resolved_step("ff_close", storage, rid_a),
                _resolved_step("ff_loose", storage, rid_b),
            ],
        )

        pipe.clean(fake_dataset, dry_run=True)
        out = capsys.readouterr().out
        assert "shared by 2 steps" in out
        assert "ff_close" in out
        assert "ff_loose" in out
        assert storage in out

    def test_no_notice_when_storage_not_shared(
        self, tmp_path, monkeypatch, capsys, fake_dataset
    ):
        """Single-step storages should not produce the shared-by notice."""
        storage = "speed-angvel__from__smooth__from__tracks"
        keep_rid = "0.1-1111111111"
        _make_run_dir(tmp_path, storage, keep_rid)
        _patch_clean_helpers(monkeypatch, tmp_path, {storage: [keep_rid]})

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [_resolved_step("speed", storage, keep_rid)],
        )

        pipe.clean(fake_dataset, dry_run=True)
        out = capsys.readouterr().out
        assert "shared by" not in out

    def test_keeper_never_existed_does_not_raise(
        self, tmp_path, monkeypatch, fake_dataset
    ):
        """A keeper whose directory was never created should not false-alarm.

        Step not yet computed → its expected_run_id has no directory →
        safeguard #3 must not flag it as "destroyed".
        """
        storage = "speed-angvel__from__smooth__from__tracks"
        rid_orphan = "0.1-orphan0000"
        rid_keeper = "0.1-keeper0000"  # directory never created

        _make_run_dir(tmp_path, storage, rid_orphan)
        _patch_clean_helpers(
            monkeypatch, tmp_path, {storage: [rid_orphan]}
        )

        pipe = Pipeline()
        monkeypatch.setattr(
            pipe,
            "_resolve_step_cache",
            lambda _ds: [_resolved_step("speed", storage, rid_keeper)],
        )

        df = pipe.clean(fake_dataset, dry_run=False)
        # Orphan removed cleanly; keeper's missing directory is fine.
        assert "removed" in df["status"].values
        assert not (tmp_path / storage / rid_orphan).exists()
