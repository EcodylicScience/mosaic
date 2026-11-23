"""The state a fit saved, as an apply through ``model`` reads it.

The chain is the built-in clustering: ``speed-angvel`` over the tracks,
``extract-templates`` over it, ``global-kmeans`` fitted on those templates, and
a second ``global-kmeans`` step that applies the fit through ``model``. The fit
is scope-free and seeded: fitting the same templates again saves the same
state.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline._utils import new_execution_id
from mosaic.core.pipeline.fit_scope import (
    FIT_SCOPE_NAME,
    FitStateUnreadable,
    list_run_root,
    read_fit_state_digest,
    saved_state_digest,
)
from mosaic.core.pipeline.index import (
    feature_index_path,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.run_log import read_run, run_log_dir
from mosaic.core.pipeline.types import Result
from tests.helpers import (
    FITTED_AGAIN,
    KMeansChain,
    add_tracks_variant,
    drifted,
    kmeans_apply,
    kmeans_apply_feature,
    kmeans_fit,
    kmeans_fit_and_apply,
    make_dataset,
    model_cells,
    move_tracks,
    refit_kmeans,
    refusals,
    result_run_root,
    run_step,
    tree_snapshot,
)

VARIANT: Final = "convert-trex.0.2-1111111111"
SEQUENCES: Final = ("seq_a", "seq_b")
KMEANS_NAME: Final = "global-kmeans"

KMEANS_STATE: Final = frozenset(
    {
        "model.joblib",
        "cluster_centers.npz",
        "artifact_labels.npz",
        "cluster_sizes.parquet",
    }
)
"""The files ``global-kmeans``'s ``save_state`` writes."""


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two track tables of one variant, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def test_a_fit_records_the_digest_of_the_state_it_saved(tracked: Dataset) -> None:
    """The digest covers the k-means state files, and no entry output."""
    chain = kmeans_fit_and_apply(tracked, workers=2, mode="thread")

    root = result_run_root(tracked, chain.fit)
    listing = list_run_root(root)
    saved = {path: stamp for path, stamp in listing.items() if path in KMEANS_STATE}
    assert set(saved) == KMEANS_STATE
    assert (root / FIT_SCOPE_NAME).is_file()
    recorded = read_fit_state_digest(root)
    assert recorded
    assert recorded == saved_state_digest(root, {}, saved)


# --- an apply compares the fit's digest ------------------------------------

WORKERS: Final = 2
MODE: Final = "thread"

NOTHING_RECORDED: Final = "nothing records which fit of its model it applied"


def _blank_model_cells(ds: Dataset, result: Result[str]) -> None:
    """Blank the ``consumed_model_state`` cell of every row of *result*."""
    path = feature_index_path(ds, result.feature)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows.loc[rows["run_id"] == result.run_id, "consumed_model_state"] = ""
    rows.to_csv(path, index=False)


def _outputs(ds: Dataset, result: Result[str]) -> dict[str, tuple[int, int, int]]:
    """The inode, size and ``mtime_ns`` of each entry output of *result*."""
    return {
        path: stamp
        for path, stamp in list_run_root(result_run_root(ds, result)).items()
        if path.endswith(".parquet")
    }


def _rewrite_fit_scope(root: Path, edit: Callable[[dict[str, object]], str]) -> None:
    """Replace the ``fit_scope.json`` of *root* by *edit* of its record."""
    path = root / FIT_SCOPE_NAME
    record: dict[str, object] = json.loads(path.read_text())
    _ = path.write_text(edit(record))


def _apply_capturing(
    ds: Dataset,
    chain: KMeansChain,
    capsys: pytest.CaptureFixture[str],
    *,
    mode: str,
) -> str:
    """Apply the chain's fit again, and what that run printed to stderr."""
    _ = capsys.readouterr()
    applied = kmeans_apply(ds, chain, workers=WORKERS, mode=mode)
    assert applied.run_id == chain.apply.run_id
    return capsys.readouterr().err


def test_an_apply_records_the_digest_of_its_fit_and_serves_after(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = read_fit_state_digest(result_run_root(tracked, chain.fit))
    assert digest
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert refusals(err, KMEANS_NAME) == {}
    assert _outputs(tracked, chain.apply) == outputs
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}


def test_an_apply_recomputes_after_its_fit_is_written_again(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    move_tracks(tracked, VARIANT, SEQUENCES)
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    # The apply first recomputes for its moved input, which is compared before
    # the model, and records the digest of the fit not yet written again.
    _ = _apply_capturing(tracked, chain, capsys, mode=MODE)
    digest = refit_kmeans(tracked, chain, workers=WORKERS, mode=MODE)
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert refusals(err, KMEANS_NAME) == {"seq_a": FITTED_AGAIN, "seq_b": FITTED_AGAIN}
    recomputed = _outputs(tracked, chain.apply)
    assert all(recomputed[path] != stamp for path, stamp in outputs.items())
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert (
        refusals(_apply_capturing(tracked, chain, capsys, mode=MODE), KMEANS_NAME) == {}
    )
    assert _outputs(tracked, chain.apply) == recomputed


def test_an_apply_row_without_a_digest_recomputes_once(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = read_fit_state_digest(result_run_root(tracked, chain.fit))
    _blank_model_cells(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert refusals(err, KMEANS_NAME) == {
        "seq_a": NOTHING_RECORDED,
        "seq_b": NOTHING_RECORDED,
    }
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert (
        refusals(_apply_capturing(tracked, chain, capsys, mode=MODE), KMEANS_NAME) == {}
    )


def test_a_fit_without_a_digest_is_served_with_one_warning(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)

    def without_digest(record: dict[str, object]) -> str:
        del record["state_digest"]
        return json.dumps(record)

    _rewrite_fit_scope(result_run_root(tracked, chain.fit), without_digest)
    _blank_model_cells(tracked, chain.apply)
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert refusals(err, KMEANS_NAME) == {}
    assert _outputs(tracked, chain.apply) == outputs
    assert model_cells(tracked, chain.apply) == {"seq_a": "", "seq_b": ""}
    warning = (
        f"[feature:global-kmeans] 2 entries apply run {chain.fit.run_id} of "
        f"{chain.fit.feature}, which records no fit-state digest, so a fit "
        "written again under that run id goes undetected: remove the run and fit "
        "it again to record one"
    )
    assert err.splitlines().count(warning) == 1


def _unparsable(_record: dict[str, object]) -> str:
    """A ``fit_scope.json`` that is not JSON."""
    return "{not json"


def _numeric_digest(record: dict[str, object]) -> str:
    """*record* with a ``state_digest`` that is not a string."""
    record["state_digest"] = 3
    return json.dumps(record)


@pytest.mark.parametrize("edit", [_unparsable, _numeric_digest])
def test_an_unreadable_fit_digest_fails_the_apply_before_any_entry(
    tracked: Dataset, edit: Callable[[dict[str, object]], str]
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    fit = runs.fit
    assert fit.run_id is not None
    _rewrite_fit_scope(result_run_root(tracked, fit), edit)
    features_root = tracked.get_root("features")
    before = tree_snapshot(features_root)
    feature = kmeans_apply_feature(runs)
    execution_id = new_execution_id()

    with pytest.raises(FitStateUnreadable) as caught:
        _ = run_feature(
            tracked,
            feature,
            parallel_workers=WORKERS,
            parallel_mode=MODE,
            execution_id=execution_id,
        )

    assert caught.value.error_detail() == {"feature": fit.feature, "run_id": fit.run_id}
    assert tree_snapshot(features_root) == before
    logged = read_run(run_log_dir(tracked.base_dir), execution_id)
    assert logged is not None
    assert logged["status"] == "failed"


def test_a_run_without_a_model_records_blank_cells_and_reads_no_fit(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[Path] = []

    def recording(run_root: Path) -> str:
        read.append(run_root)
        return read_fit_state_digest(run_root)

    monkeypatch.setattr(run_module, "read_fit_state_digest", recording)

    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)

    assert read == []
    for result in (runs.speed, runs.templates, runs.fit):
        cells = model_cells(tracked, result)
        assert cells
        assert set(cells.values()) == {""}
    _ = kmeans_apply(tracked, runs, workers=WORKERS, mode=MODE)
    assert read == [result_run_root(tracked, runs.fit)]


def test_a_plan_names_every_apply_entry_after_the_refit_and_none_after_the_apply(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    move_tracks(tracked, VARIANT, SEQUENCES)
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    _ = _apply_capturing(tracked, chain, capsys, mode=MODE)
    assert drifted(tracked, chain.apply) == ()
    _ = refit_kmeans(tracked, chain, workers=WORKERS, mode=MODE)

    assert drifted(tracked, chain.apply) == (("", "seq_a"), ("", "seq_b"))

    _ = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert drifted(tracked, chain.apply) == ()
