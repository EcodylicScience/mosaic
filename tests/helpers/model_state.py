"""The fit-state digest an apply records, and the runs that change it.

The k-means chain is the built-in clustering: ``speed-angvel`` over the tracks,
``extract-templates`` over it, ``global-kmeans`` fitted on those templates, and
a second ``global-kmeans`` step that applies the fit through ``model``. The fit
is scope-free and seeded: fitting the same templates again saves the same
state, and fitting other templates saves another.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.fit_scope import read_fit_state_digest
from mosaic.core.pipeline.index import drifted_entries, feature_index_path
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.types import Feature, Result
from tests.helpers.results import model_reference, result_run_root
from tests.helpers.tracks import add_tracks_table

__all__ = [
    "FITTED_AGAIN",
    "KMEANS",
    "KMeansChain",
    "KMeansFit",
    "drifted",
    "index_cells",
    "kmeans_apply",
    "kmeans_apply_feature",
    "kmeans_fit",
    "kmeans_fit_and_apply",
    "model_cells",
    "move_tracks",
    "reconvert_entry",
    "refit_kmeans",
    "refusals",
    "run_step",
]

FITTED_AGAIN: Final = "a fit it reads was fitted again"
"""The reason a run gives for an entry a fit of which was written again."""

KMEANS: Final[dict[str, object]] = {"k": 1, "n_init": 1}
"""The clustering params of the ``global-kmeans`` fit and of its apply."""

_REFUSAL: Final = re.compile(
    r"^\[feature:(?P<name>[^\]]*)\] cannot serve cached output for "
    r"\((?P<group>[^,]*),(?P<sequence>[^)]*)\): (?P<reason>.*); recomputing\.$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class KMeansFit:
    """The runs that fit ``global-kmeans``.

    Attributes:
        speed: The ``speed-angvel`` run over the tracks.
        templates: The ``extract-templates`` run over ``speed``.
        fit: The ``global-kmeans`` run fitted on ``templates``.
    """

    speed: Result[str]
    templates: Result[str]
    fit: Result[str]

    def run_ids(self) -> tuple[str | None, str | None, str | None]:
        """The run ids of ``speed``, ``templates`` and ``fit``."""
        return (self.speed.run_id, self.templates.run_id, self.fit.run_id)


@dataclass(frozen=True, slots=True)
class KMeansChain(KMeansFit):
    """The runs of one pass over the k-means chain.

    Attributes:
        apply: The ``global-kmeans`` run applying ``fit`` through ``model``.
    """

    apply: Result[str]


def run_step(
    ds: Dataset,
    slug: str,
    inputs: object,
    params: dict[str, object],
    *,
    workers: int = 2,
    mode: str = "thread",
) -> Result[str]:
    """Run *slug* over *ds* with *workers* workers in *mode*."""
    from mosaic.core.pipeline.graph.resolve import build_feature

    feature = build_feature(slug, inputs, params)
    return run_feature(
        ds, feature, parallel_workers=workers, parallel_mode=mode, track=False
    )


def kmeans_fit(ds: Dataset, *, workers: int = 2, mode: str = "thread") -> KMeansFit:
    """Run ``speed-angvel``, its templates and the fit fitted on them."""
    speed = run_step(ds, "speed-angvel", None, {}, workers=workers, mode=mode)
    templates = run_step(
        ds,
        "extract-templates",
        [model_reference(speed)],
        {"n_templates": 20},
        workers=workers,
        mode=mode,
    )
    fit = run_step(
        ds,
        "global-kmeans",
        [model_reference(speed)],
        {"templates": model_reference(templates), **KMEANS},
        workers=workers,
        mode=mode,
    )
    return KMeansFit(speed=speed, templates=templates, fit=fit)


def kmeans_apply_feature(runs: KMeansFit) -> Feature:
    """The ``global-kmeans`` step that applies the fit of *runs*."""
    from mosaic.core.pipeline.graph.resolve import build_feature

    return build_feature(
        "global-kmeans",
        [model_reference(runs.speed)],
        {"model": model_reference(runs.fit), **KMEANS},
    )


def kmeans_apply(
    ds: Dataset, runs: KMeansFit, *, workers: int = 2, mode: str = "thread"
) -> Result[str]:
    """Apply the fit of *runs* to the entries of its ``speed-angvel`` run."""
    return run_feature(
        ds,
        kmeans_apply_feature(runs),
        parallel_workers=workers,
        parallel_mode=mode,
        track=False,
    )


def kmeans_fit_and_apply(
    ds: Dataset, *, workers: int = 2, mode: str = "thread"
) -> KMeansChain:
    """Run the k-means chain over *ds* with *workers* workers in *mode*.

    A step whose run already has its outputs serves them. A second call after
    an upstream change recomputes only the entries that change affects.
    """
    runs = kmeans_fit(ds, workers=workers, mode=mode)
    applied = kmeans_apply(ds, runs, workers=workers, mode=mode)
    return KMeansChain(
        speed=runs.speed, templates=runs.templates, fit=runs.fit, apply=applied
    )


def move_tracks(ds: Dataset, variant: str, sequences: Iterable[str]) -> None:
    """Write each track table of *sequences* again with other values.

    Each table of the tracks variant *variant* gets its positions doubled
    and a new composition, as a conversion from changed sources writes it.
    ``speed-angvel`` then recomputes every entry through its tracks channel.
    """
    tracks = tracks_variant_root(ds.get_root("tracks"), variant)
    for sequence in sequences:
        table = _scaled(pd.read_parquet(tracks / f"{sequence}.parquet"), 2.0)
        add_tracks_table(
            ds, variant, table, std_format="mosaic_v1", consumed_composition="moved"
        )


def _scaled(table: pd.DataFrame, factor: float) -> pd.DataFrame:
    """*table* with each of its position columns multiplied by *factor*."""
    positions = [
        column
        for column in table.columns
        if column in {"X", "Y"} or column.startswith(("X#", "Y#", "pose"))
    ]
    table[positions] = table[positions] * factor
    return table


def reconvert_entry(
    ds: Dataset, entry: Entry, *, composition: str, scale: float = 1.0
) -> None:
    """Write each tracks table of *entry* again, recording *composition*.

    Stands for a conversion of the entry from changed sources: the table is
    written back under its variant, its positions multiplied by *scale*, and
    its row records the new composition, which a run reading the entry
    compares.
    """
    group, sequence = entry
    rows = read_tracks_index(ds)
    selected = rows[
        (rows["group"].astype(str) == group)
        & (rows["sequence"].astype(str) == sequence)
    ]
    assert not selected.empty
    for _, row in selected.iterrows():
        table = _scaled(pd.read_parquet(ds.resolve_path(str(row["abs_path"]))), scale)
        add_tracks_table(
            ds,
            str(row["run_id"]),
            table,
            std_format=str(row["std_format"]),
            consumed_composition=composition,
        )


def refit_kmeans(
    ds: Dataset, runs: KMeansFit, *, workers: int = 2, mode: str = "thread"
) -> str:
    """Fit the k-means chain again under its run ids after :func:`move_tracks`.

    The templates and the fit are removed and computed again over the
    recomputed ``speed-angvel`` outputs, which give other templates.

    Returns:
        The digest the fit records after it was written again.
    """
    before = read_fit_state_digest(result_run_root(ds, runs.fit))
    shutil.rmtree(result_run_root(ds, runs.templates))
    shutil.rmtree(result_run_root(ds, runs.fit))
    again = kmeans_fit(ds, workers=workers, mode=mode)
    assert again.run_ids() == runs.run_ids()
    after = read_fit_state_digest(result_run_root(ds, again.fit))
    assert after
    assert after != before
    return after


def refusals(err: str, feature: str) -> dict[str, str]:
    """The reason a run of *feature* printed for each sequence not served.

    Args:
        err: What the run printed to stderr.
        feature: The feature's name as the run prints it, such as
            ``global-kmeans`` or ``def:<name>``.
    """
    return {
        match["sequence"]: match["reason"]
        for match in _REFUSAL.finditer(err)
        if match["name"] == feature
    }


def index_cells(ds: Dataset, result: Result[str], column: str) -> dict[str, str]:
    """The *column* cell of each own row of *result*'s run, by sequence."""
    path = feature_index_path(ds, result.feature)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = rows[rows["run_id"] == result.run_id]
    return dict(
        zip(rows["sequence"].astype(str), rows[column].astype(str), strict=True)
    )


def model_cells(ds: Dataset, result: Result[str]) -> dict[str, str]:
    """The ``consumed_model_state`` cell of each own row of *result*'s run."""
    return index_cells(ds, result, "consumed_model_state")


def drifted(ds: Dataset, result: Result[str]) -> tuple[tuple[str, str], ...]:
    """The entries ``drifted_entries`` reports for *result*'s run."""
    assert result.run_id is not None
    params = read_run_params(result_run_root(ds, result)).params
    return drifted_entries(ds, result.feature, result.run_id, params)
