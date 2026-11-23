"""The state a fit saved, as an apply through ``model`` reads it.

The chain is the built-in clustering: ``speed-angvel`` over the tracks,
``extract-templates`` over it, ``global-kmeans`` fitted on those templates, and
a second ``global-kmeans`` step that applies the fit through ``model``. The fit
is scope-free and seeded: fitting the same templates again saves the same
state.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
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
from mosaic.core.pipeline.graph.resolve import build_feature
from mosaic.core.pipeline.index import (
    drifted_entries,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.run_log import read_run, run_log_dir
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.types import Result
from tests.helpers import (
    add_tracks_table,
    add_tracks_variant,
    make_dataset,
    tree_snapshot,
)

VARIANT: Final = "convert-trex.0.2-1111111111"

KMEANS_STATE: Final = frozenset(
    {
        "model.joblib",
        "cluster_centers.npz",
        "artifact_labels.npz",
        "cluster_sizes.parquet",
    }
)
"""The files ``global-kmeans``'s ``save_state`` writes."""

KMEANS: Final[dict[str, object]] = {"k": 1, "n_init": 1}
"""The clustering params of the fit and of the apply."""


@dataclass(frozen=True, slots=True)
class KMeansChain:
    """The runs of one pass over the chain.

    Attributes:
        speed: The ``speed-angvel`` run over the tracks.
        templates: The ``extract-templates`` run over ``speed``.
        fit: The ``global-kmeans`` run fitted on ``templates``.
        apply: The ``global-kmeans`` run applying ``fit`` through ``model``.
    """

    speed: Result[str]
    templates: Result[str]
    fit: Result[str]
    apply: Result[str]


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two track tables of one variant, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _reference(result: Result[str]) -> dict[str, object]:
    """The params reference that names *result*'s run."""
    return {"feature": result.feature, "run_id": result.run_id}


def _run(
    ds: Dataset,
    slug: str,
    inputs: object,
    params: dict[str, object],
    *,
    workers: int,
    mode: str,
) -> Result[str]:
    """Run *slug* over *ds* with *workers* workers in *mode*."""
    feature = build_feature(slug, inputs, params)
    return run_feature(
        ds, feature, parallel_workers=workers, parallel_mode=mode, track=False
    )


def kmeans_fit(
    ds: Dataset, *, workers: int, mode: str
) -> tuple[Result[str], Result[str], Result[str]]:
    """Run ``speed-angvel``, its templates and the fit fitted on them."""
    speed = _run(ds, "speed-angvel", None, {}, workers=workers, mode=mode)
    templates = _run(
        ds,
        "extract-templates",
        [_reference(speed)],
        {"n_templates": 20},
        workers=workers,
        mode=mode,
    )
    fit = _run(
        ds,
        "global-kmeans",
        [_reference(speed)],
        {"templates": _reference(templates), **KMEANS},
        workers=workers,
        mode=mode,
    )
    return speed, templates, fit


def kmeans_apply(
    ds: Dataset, speed: Result[str], fit: Result[str], *, workers: int, mode: str
) -> Result[str]:
    """Apply *fit* to the entries of *speed* through ``model``."""
    return _run(
        ds,
        "global-kmeans",
        [_reference(speed)],
        {"model": _reference(fit), **KMEANS},
        workers=workers,
        mode=mode,
    )


def kmeans_fit_and_apply(ds: Dataset, *, workers: int, mode: str) -> KMeansChain:
    """Run the chain over *ds*, every step with *workers* workers in *mode*.

    A step whose run already has its outputs serves them. A second call after
    an upstream change recomputes only the entries that change affects.
    """
    speed, templates, fit = kmeans_fit(ds, workers=workers, mode=mode)
    applied = kmeans_apply(ds, speed, fit, workers=workers, mode=mode)
    return KMeansChain(speed=speed, templates=templates, fit=fit, apply=applied)


def _root(ds: Dataset, result: Result[str]) -> Path:
    """The run root of *result*."""
    assert result.run_id is not None
    return feature_run_root(ds, result.feature, result.run_id)


def test_a_fit_records_the_digest_of_the_state_it_saved(tracked: Dataset) -> None:
    """The digest covers the k-means state files, and no entry output."""
    chain = kmeans_fit_and_apply(tracked, workers=2, mode="thread")

    root = _root(tracked, chain.fit)
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

FITTED_AGAIN: Final = "the model it applies was fitted again"
NOTHING_RECORDED: Final = "nothing records which fit of its model it applied"

_REFUSAL: Final = re.compile(
    r"^\[feature:global-kmeans\] cannot serve cached output for "
    r"\((?P<group>[^,]*),(?P<sequence>[^)]*)\): (?P<reason>.*); recomputing\.$",
    re.MULTILINE,
)


def _refusals(err: str) -> dict[str, str]:
    """The reason the run printed for each sequence it did not serve."""
    return {match["sequence"]: match["reason"] for match in _REFUSAL.finditer(err)}


def _rows(ds: Dataset, result: Result[str]) -> pd.DataFrame:
    """The index rows of *result*'s run."""
    path = feature_index_path(ds, result.feature)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    return rows[rows["run_id"] == result.run_id]


def _model_cells(ds: Dataset, result: Result[str]) -> dict[str, str]:
    """The ``consumed_model_state`` cell of each row of *result*."""
    rows = _rows(ds, result)
    return dict(
        zip(
            rows["sequence"].astype(str),
            rows["consumed_model_state"].astype(str),
            strict=True,
        )
    )


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
        for path, stamp in list_run_root(_root(ds, result)).items()
        if path.endswith(".parquet")
    }


def _rewrite_fit_scope(root: Path, edit: Callable[[dict[str, object]], str]) -> None:
    """Replace the ``fit_scope.json`` of *root* by *edit* of its record."""
    path = root / FIT_SCOPE_NAME
    record: dict[str, object] = json.loads(path.read_text())
    _ = path.write_text(edit(record))


def _move_tracks(ds: Dataset) -> None:
    """Write each track table again with other values and a new composition.

    A conversion from changed sources writes the tables so. ``speed-angvel``
    then recomputes every entry through its tracks channel.
    """
    tracks = tracks_variant_root(ds.get_root("tracks"), VARIANT)
    for sequence in ("seq_a", "seq_b"):
        table = pd.read_parquet(tracks / f"{sequence}.parquet")
        moved = [
            column
            for column in table.columns
            if column in {"X", "Y"} or column.startswith(("X#", "Y#", "pose"))
        ]
        table[moved] = table[moved] * 2.0
        add_tracks_table(
            ds, VARIANT, table, std_format="mosaic_v1", consumed_composition="moved"
        )


def _refit(ds: Dataset, chain: KMeansChain, *, workers: int, mode: str) -> str:
    """Fit the chain again under its run ids after :func:`_move_tracks`.

    The templates and the fit are removed and computed again over the
    recomputed ``speed-angvel`` outputs, which give other templates.

    Returns:
        The digest the fit records after it was written again.
    """
    before = read_fit_state_digest(_root(ds, chain.fit))
    shutil.rmtree(_root(ds, chain.templates))
    shutil.rmtree(_root(ds, chain.fit))
    speed, templates, fit = kmeans_fit(ds, workers=workers, mode=mode)
    assert [result.run_id for result in (speed, templates, fit)] == [
        chain.speed.run_id,
        chain.templates.run_id,
        chain.fit.run_id,
    ]
    after = read_fit_state_digest(_root(ds, fit))
    assert after
    assert after != before
    return after


def _apply_capturing(
    ds: Dataset,
    chain: KMeansChain,
    capsys: pytest.CaptureFixture[str],
    *,
    mode: str,
) -> str:
    """Apply the chain's fit again, and what that run printed to stderr."""
    _ = capsys.readouterr()
    applied = kmeans_apply(ds, chain.speed, chain.fit, workers=WORKERS, mode=mode)
    assert applied.run_id == chain.apply.run_id
    return capsys.readouterr().err


def test_an_apply_records_the_digest_of_its_fit_and_serves_after(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = read_fit_state_digest(_root(tracked, chain.fit))
    assert digest
    assert _model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert _refusals(err) == {}
    assert _outputs(tracked, chain.apply) == outputs
    assert _model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}


def test_an_apply_recomputes_after_its_fit_is_written_again(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    _move_tracks(tracked)
    speed = _run(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    # The apply first recomputes for its moved input, which is compared before
    # the model, and records the digest of the fit not yet written again.
    _ = _apply_capturing(tracked, chain, capsys, mode=MODE)
    digest = _refit(tracked, chain, workers=WORKERS, mode=MODE)
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert _refusals(err) == {"seq_a": FITTED_AGAIN, "seq_b": FITTED_AGAIN}
    recomputed = _outputs(tracked, chain.apply)
    assert all(recomputed[path] != stamp for path, stamp in outputs.items())
    assert _model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert _refusals(_apply_capturing(tracked, chain, capsys, mode=MODE)) == {}
    assert _outputs(tracked, chain.apply) == recomputed


def test_an_apply_row_without_a_digest_recomputes_once(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = read_fit_state_digest(_root(tracked, chain.fit))
    _blank_model_cells(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert _refusals(err) == {"seq_a": NOTHING_RECORDED, "seq_b": NOTHING_RECORDED}
    assert _model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert _refusals(_apply_capturing(tracked, chain, capsys, mode=MODE)) == {}


def test_a_fit_without_a_digest_is_served_with_one_warning(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)

    def without_digest(record: dict[str, object]) -> str:
        del record["state_digest"]
        return json.dumps(record)

    _rewrite_fit_scope(_root(tracked, chain.fit), without_digest)
    _blank_model_cells(tracked, chain.apply)
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert _refusals(err) == {}
    assert _outputs(tracked, chain.apply) == outputs
    assert _model_cells(tracked, chain.apply) == {"seq_a": "", "seq_b": ""}
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
    speed, _templates, fit = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    assert fit.run_id is not None
    _rewrite_fit_scope(_root(tracked, fit), edit)
    features_root = tracked.get_root("features")
    before = tree_snapshot(features_root)
    feature = build_feature(
        "global-kmeans", [_reference(speed)], {"model": _reference(fit), **KMEANS}
    )
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

    speed, templates, fit = kmeans_fit(tracked, workers=WORKERS, mode=MODE)

    assert read == []
    for result in (speed, templates, fit):
        cells = _model_cells(tracked, result)
        assert cells
        assert set(cells.values()) == {""}
    _ = kmeans_apply(tracked, speed, fit, workers=WORKERS, mode=MODE)
    assert read == [_root(tracked, fit)]


def _drifted(ds: Dataset, result: Result[str]) -> tuple[tuple[str, str], ...]:
    """The entries ``drifted_entries`` reports for *result*'s run."""
    assert result.run_id is not None
    params = read_run_params(_root(ds, result)).params
    return drifted_entries(ds, result.feature, result.run_id, params)


def test_a_plan_names_every_apply_entry_after_the_refit_and_none_after_the_apply(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    _move_tracks(tracked)
    speed = _run(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    _ = _apply_capturing(tracked, chain, capsys, mode=MODE)
    assert _drifted(tracked, chain.apply) == ()
    _ = _refit(tracked, chain, workers=WORKERS, mode=MODE)

    assert _drifted(tracked, chain.apply) == (("", "seq_a"), ("", "seq_b"))

    _ = _apply_capturing(tracked, chain, capsys, mode=MODE)

    assert _drifted(tracked, chain.apply) == ()
