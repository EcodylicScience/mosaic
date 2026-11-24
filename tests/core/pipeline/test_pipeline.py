"""Regression test: a multi-step Pipeline records a run-log for EVERY step.

``Pipeline.run`` no longer shares a DB handle across steps -- each step's
``run_feature`` opens its own append-only JSONL run-log under
``<dataset_root>/.mosaic/runs/``. This test locks in that every step leaves an
attempt record (not just the first), and that pipeline caching is decided purely
by the parquet outputs on disk (``index.csv`` + files), never by a status flag.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.feature_library import (
    ExtractTemplates,
    SpeedAngvel,
    TrajectorySmooth,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import CallbackStep, FeatureStep, Pipeline
from mosaic.core.pipeline.index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.run_log import read_runs, run_log_dir
from mosaic.core.pipeline.types import InputStream
from mosaic.core.scope import Scope
from tests.helpers import (
    add_track_sequences,
    add_tracks_variant,
    convert_tracks_from_raw,
    converted_source,
    copy_of,
    make_dataset,
    reconvert_entry,
    templates_pipeline,
)


def _dataset_with_tracks(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), n_rows=12)
    return ds


def test_pipeline_records_every_step(tmp_path: Path) -> None:
    ds = _dataset_with_tracks(tmp_path)

    # Two independent tracks-sourced steps (same feature, different params -> two
    # distinct run_ids). Each step runs under its own JSONL run-log.
    pipe = Pipeline()
    pipe.add(FeatureStep("speed_a", SpeedAngvel, {"step_size": 1}))
    pipe.add(FeatureStep("speed_b", SpeedAngvel, {"step_size": 2}))
    pipe.run(ds)

    runs = read_runs(run_log_dir(ds.base_dir), kind="feature")

    # BOTH steps must have left an attempt record -- not just the first.
    assert len(runs) == 2, f"expected 2 attempt logs, got {[r['run_id'] for r in runs]}"
    assert {str(r["status"]) for r in runs} == {"finished"}
    assert len({str(r["run_id"]) for r in runs}) == 2


# ---------------------------------------------------------------------------
# Completeness-aware caching: a partial run must NOT read as cached.
# ---------------------------------------------------------------------------


def _resolve(pipe: Pipeline, ds: Dataset, name: str) -> dict[str, object]:
    return next(r for r in pipe._resolve_step_cache(ds) if r["step"].name == name)


def _write_dummy_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": [0], "X": [0.0], "Y": [0.0]}).to_parquet(path)


def test_partial_run_reads_not_cached_and_resumes(tmp_path: Path) -> None:
    """A run dir with only some target sequences is 'partial', not 'cached'."""
    ds = _dataset_with_tracks(tmp_path)  # target = {(g, s1), (g, s2)}
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))

    info = _resolve(pipe, ds, "speed")
    run_root = feature_run_root(ds, info["storage_name"], info["expected_run_id"])
    _write_dummy_parquet(run_root / "g__s1.parquet")  # 1 of 2 -> partial

    row = pipe.status(ds)
    row = row[row["step"] == "speed"].iloc[0]
    assert row["cached"] == "partial"
    assert row["n_seq"] == "1/2"

    # The resolve reports it not-cached (bool), so run() executes/resumes it.
    assert _resolve(pipe, ds, "speed")["cached"] is False
    pipe.run(ds)
    assert len(list(run_root.glob("*.parquet"))) == 2


def test_scope_relative_finished_is_not_trusted(tmp_path: Path) -> None:
    """finished_at set over a truncated scope must NOT make a step read cached.

    Reproduces the ffgm_main failure: a downstream step whose partial upstream
    truncated its manifest honestly marks itself finished for that smaller scope.
    The completeness check is disk-based, so an ``index.csv`` that claims the run
    is finished over 1 of 2 sequences is still overruled by the missing parquet.
    """
    ds = _dataset_with_tracks(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))

    info = _resolve(pipe, ds, "speed")
    storage, rid = info["storage_name"], info["expected_run_id"]
    run_root = feature_run_root(ds, storage, rid)
    entry = run_root / "g__s1.parquet"
    _write_dummy_parquet(entry)

    # Seed index.csv to claim the run is FINISHED, but over only 1 of 2 sequences.
    idx = feature_index(feature_index_path(ds, storage))
    idx.append(
        [
            FeatureIndexRow(
                run_id=rid,
                feature=storage,
                version="0.1",
                group="g",
                sequence="s1",
                abs_path=Path(ds.relative_to_root(entry)),
                n_rows=1,
                params_hash="hash",
            )
        ]
    )
    idx.mark_finished(rid)
    finished_df = idx.read(run_id=rid)
    assert (finished_df["finished_at"] != "").all()  # index says done...

    # ...but only 1 of 2 target sequences exists on disk, so it is NOT cached.
    assert _resolve(pipe, ds, "speed")["cached"] is False
    pipe.run(ds)
    assert len(list(run_root.glob("*.parquet"))) == 2


def test_complete_run_stays_cached_without_run_logs(tmp_path: Path) -> None:
    """A fully complete run reads cached; deleting the run-logs doesn't change it."""
    import shutil

    ds = _dataset_with_tracks(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))
    pipe.run(ds)

    assert _resolve(pipe, ds, "speed")["cached"] is True

    # The cache check is disk-based (parquet + index.csv), never the status store.
    # Blowing away the JSONL run-logs must not force a recompute.
    run_dir = run_log_dir(ds.base_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    assert _resolve(pipe, ds, "speed")["cached"] is True


def test_a_copy_reads_the_sources_complete_run_as_cached(tmp_path: Path) -> None:
    source = converted_source(tmp_path / "source")
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))
    pipe.run(source)
    copy = copy_of(source, tmp_path / "copy")

    linked = _resolve(pipe, source, "speed")
    resolved = _resolve(pipe, copy, "speed")

    assert resolved["cached"] is True
    assert resolved["expected_run_id"] == linked["expected_run_id"]


def test_a_copy_holding_one_entry_of_the_run_reads_cached_with_the_sources_rest(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {"step_size": 1}))
    pipe.run(source)
    copy = copy_of(source, tmp_path / "copy")
    info = _resolve(pipe, source, "speed")
    storage, run_id = str(info["storage_name"]), str(info["expected_run_id"])
    linked = feature_run_root(source, storage, run_id)
    own = feature_run_root(copy, storage, run_id)
    own.mkdir(parents=True)
    first = sorted(linked.glob("*.parquet"))[0]
    _ = (own / first.name).write_bytes(first.read_bytes())

    resolved = _resolve(pipe, copy, "speed")

    assert resolved["cached"] is True
    assert resolved["n_present"] == resolved["n_target"] == 2


class _ScopedSpeed(SpeedAngvel):
    """A scope_dependent variant to exercise the run_id parity fix."""

    name = "scoped-speed"
    scope_dependent = True
    consumed_roots: tuple[str, ...] = ()


def test_scope_dependent_run_id_matches_on_disk(tmp_path: Path) -> None:
    """For scope_dependent features, status()'s predicted run_id must include the
    scope term so it matches the on-disk run_id run_feature wrote."""
    ds = _dataset_with_tracks(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("scoped", _ScopedSpeed, {"step_size": 1}))
    pipe.run(ds)

    info = _resolve(pipe, ds, "scoped")
    assert info["cached"] is True  # predicted run_id (with _scope_entries) matched
    run_root = feature_run_root(ds, info["storage_name"], info["expected_run_id"])
    assert run_root.exists() and any(run_root.glob("*.parquet"))


def test_a_step_still_spelling_the_retired_scope_kwargs_is_refused(
    tmp_path: Path,
) -> None:
    """The refusal has to precede the identity, which precedes the cache check.

    ``run_feature`` raises on a ``groups`` keyword. A step whose full-dataset
    run is already on disk never reaches that call: the prediction reads the
    run root as complete and reports a cache hit.
    """
    ds = _dataset_with_tracks(tmp_path)
    done = Pipeline()
    done.add(FeatureStep("scoped", _ScopedSpeed, {"step_size": 1}))
    done.run(ds)

    stale = Pipeline()
    stale.add(
        FeatureStep(
            "scoped", _ScopedSpeed, {"step_size": 1}, run_kwargs={"groups": ["g"]}
        )
    )
    with pytest.raises(TypeError, match="groups"):
        stale.run(ds)


# ---------------------------------------------------------------------------
# A drifted step is recomputed.
# ---------------------------------------------------------------------------

DRIFT_VARIANT = "convert-trex.0.2-1111111111"


def _converted(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path)
    convert_tracks_from_raw(ds, DRIFT_VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _reconvert_seq_a(ds: Dataset) -> None:
    convert_tracks_from_raw(ds, DRIFT_VARIANT, "seq_a", n=9, std_format="mosaic_v1")


def _smooth_then_speed() -> Pipeline:
    pipe = Pipeline()
    pipe.add(FeatureStep("smooth", TrajectorySmooth, {}))
    pipe.add(FeatureStep("speed", SpeedAngvel, {}, ["smooth"]))
    return pipe


def test_pipeline_run_recomputes_a_drifted_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _converted(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {}))
    pipe.run(ds)
    _reconvert_seq_a(ds)
    _ = capsys.readouterr()

    pipe.run(ds)
    captured = capsys.readouterr()

    assert "(cached)" not in captured.out
    assert "cannot serve cached output for (,seq_a)" in captured.err
    assert "cannot serve cached output for (,seq_b)" not in captured.err


def test_pipeline_run_rechecks_a_step_below_a_recomputed_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _converted(tmp_path)
    pipe = _smooth_then_speed()
    pipe.run(ds)
    _reconvert_seq_a(ds)
    _ = capsys.readouterr()

    pipe.run(ds)
    captured = capsys.readouterr()

    assert "(cached)" not in captured.out
    assert captured.err.count("cannot serve cached output for (,seq_a)") == 2
    _ = capsys.readouterr()
    pipe.run(ds)
    assert capsys.readouterr().out.count("(cached)") == 2


def test_pipeline_run_serves_its_steps_after_an_entry_is_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _converted(tmp_path)
    pipe = _smooth_then_speed()
    pipe.run(ds)
    _ = ds.drop_entries(scope=Scope(entries=[("", "seq_b")]))
    _ = capsys.readouterr()

    pipe.run(ds)

    assert capsys.readouterr().out.count("(cached)") == 2


def test_pipeline_load_still_loads_a_drifted_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _converted(tmp_path)
    pipe = Pipeline()
    pipe.add(FeatureStep("speed", SpeedAngvel, {}))
    pipe.run(ds)
    _reconvert_seq_a(ds)
    _ = capsys.readouterr()

    results = pipe.load(ds)

    assert "speed" in results
    assert "Drift: source has moved under speed (1 sequence(s))" in (
        capsys.readouterr().out
    )
    status = pipe.status(ds)
    assert list(status["drift"]) == [1]


TEMPLATES_VARIANT = "convert-trex.0.2-1111111111"


def _fitted_templates(tmp_path: Path) -> Dataset:
    """A dataset whose templates were fitted over ``seq_a`` and ``seq_b``."""
    ds = make_dataset(tmp_path)
    add_tracks_variant(ds, TEMPLATES_VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    templates_pipeline().run(ds)
    return ds


def _drifted_templates(tmp_path: Path) -> Dataset:
    """A dataset whose templates were fitted before ``seq_a`` changed.

    The speed step is computed again over the new table, so the speed rows the
    fit read have moved, and the fit is due again.
    """
    ds = _fitted_templates(tmp_path)
    reconvert_entry(ds, ("", "seq_a"), composition="converted-again")
    speed = Pipeline()
    speed.add(FeatureStep("speed", SpeedAngvel, {}))
    speed.run(ds)
    return ds


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """One item for each fit of ``extract-templates`` in the test."""
    fitted: list[None] = []
    original = ExtractTemplates.fit

    def counting(self: ExtractTemplates, inputs: InputStream) -> None:
        fitted.append(None)
        original(self, inputs)

    monkeypatch.setattr(ExtractTemplates, "fit", counting)
    return fitted


def test_pipeline_status_names_the_refit_of_a_drifted_fit(tmp_path: Path) -> None:
    ds = _drifted_templates(tmp_path)

    status = templates_pipeline().status(ds)

    assert list(status["step"]) == ["speed", "templates"]
    assert list(status["refit"]) == ["", "1 moved (seq_a)"]
    assert list(status["drift"]) == ["", 2]


def test_pipeline_status_names_no_refit_for_a_fit_that_did_not_drift(
    tmp_path: Path,
) -> None:
    ds = _fitted_templates(tmp_path)

    status = templates_pipeline().status(ds)

    assert list(status["refit"]) == ["", ""]


def test_pipeline_load_prints_the_refit_of_a_drifted_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _drifted_templates(tmp_path)
    _ = capsys.readouterr()

    results = templates_pipeline().load(ds)

    assert set(results) == {"speed", "templates"}
    assert "  Refit: templates (1 moved (seq_a))" in capsys.readouterr().out


def test_pipeline_load_prints_no_refit_for_a_fit_that_did_not_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _fitted_templates(tmp_path)
    _ = capsys.readouterr()

    _ = templates_pipeline().load(ds)

    assert "Refit" not in capsys.readouterr().out


def test_pipeline_run_serves_a_fit_it_is_not_asked_to_fit_again(
    tmp_path: Path, fits: list[None], capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _fitted_templates(tmp_path)
    assert len(fits) == 1
    _ = capsys.readouterr()

    templates_pipeline().run(ds)

    assert len(fits) == 1
    assert capsys.readouterr().out.count("(cached)") == 2


def test_pipeline_run_fits_a_named_step_again_and_serves_the_rest(
    tmp_path: Path, fits: list[None], capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _fitted_templates(tmp_path)
    _ = capsys.readouterr()

    results = templates_pipeline().run(ds, refit=("templates",))

    assert len(fits) == 2
    out = capsys.readouterr().out
    assert out.count("(cached)") == 1
    assert "[speed]" in out
    assert "(cached)" in next(line for line in out.splitlines() if "[speed]" in line)
    assert results["templates"].run_id is not None


def test_pipeline_run_refuses_a_refit_that_names_no_feature_step(
    tmp_path: Path, fits: list[None]
) -> None:
    ds = _fitted_templates(tmp_path)
    pipe = templates_pipeline()
    pipe.add(CallbackStep("note", lambda _ds, _results: None, ["templates"]))

    with pytest.raises(ValueError, match="refit.*'nope'.*Available"):
        _ = pipe.run(ds, refit=("templates", "nope"))
    with pytest.raises(ValueError, match="refit.*'note'"):
        _ = pipe.run(ds, refit=("note",))

    assert len(fits) == 1


def test_a_step_holding_refit_in_its_run_kwargs_fits_on_every_run(
    tmp_path: Path, fits: list[None]
) -> None:
    ds = _fitted_templates(tmp_path)
    pipe = templates_pipeline({"refit": True})

    _ = pipe.run(ds)
    assert len(fits) == 2

    _ = pipe.run(ds)
    assert len(fits) == 3
