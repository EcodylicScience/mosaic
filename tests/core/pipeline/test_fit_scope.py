"""Item 5.3: what a run's state was fitted over, as distinct from applied to.

The distinction only bites for a **params-level fitter** -- scope-free, so every
apply scope resolves to one ``run_id`` and one run root. A ``scope_dependent``
feature gets a different identifier for a different scope and therefore cannot
reuse another scope's state at all, which is why item 5.3 calls itself narrower
than it looks.

A params-level fitter trains on a pinned artifact, so its record is derived from
that artifact's own record, never from the scope it was invoked over.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.fit_scope import (
    FIT_SCOPE_NAME,
    FitScope,
    fit_and_apply_scopes,
    read_fit_scope,
    write_fit_scope,
)
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    Feature,
    Inputs,
    InputStream,
    TemplatesRef,
)
from mosaic.core.params import Params
from mosaic.core.scope import Scope

SOURCE_DIR = "fit-scope-source__from__tracks"
FIT_DIR = "fit-scope-fit__from__tracks"
PROBE_DIR = "fit-scope-probe__from__tracks"
CONSUMER_DIR = "fit-scope-consumer__from__tracks"


class _P(Params):
    pass


class _StatefulStub:
    """The fit/save/load half every stub here shares."""

    parallelizable = False
    accepts_overlap = False
    emits: EmitsLevel = "as-input"
    consumed_roots: tuple[str, ...] = ()
    fits: int

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return (run_root / "state.txt").exists()

    def fit(self, inputs: InputStream) -> None:
        self.fits += 1

    def save_state(self, run_root: Path) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        _ = (run_root / "state.txt").write_text("fitted")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class _TemplateSource(_StatefulStub):
    """A scope-dependent sampler: it fits from its stream and writes templates."""

    name = "fit-scope-source"
    version = "0.1"
    scope_dependent = True

    def __init__(self) -> None:
        self.inputs = Inputs(("tracks",))
        self.params = _P()
        self.fits = 0

    def save_state(self, run_root: Path) -> None:
        super().save_state(run_root)
        pd.DataFrame({"x": [0.0, 1.0]}).to_parquet(run_root / "templates.parquet")


class _ArtifactParams(Params):
    templates: TemplatesRef = TemplatesRef()


class _ArtifactFit(_StatefulStub):
    """A params-level fitter: scope-free, trained on a pinned templates artifact."""

    name = "fit-scope-fit"
    version = "0.1"
    scope_dependent = False

    def __init__(self, source: str, run_id: str) -> None:
        self.inputs = Inputs(("tracks",))
        self.params = _ArtifactParams(
            templates=TemplatesRef(feature=source, run_id=run_id)
        )
        self.fits = 0


class _ArtifactConsumer(_StatefulStub):
    """A scope-dependent feature that applies an artifact another run fitted.

    The shape of ``evaluate-labels`` with a ``label_map``: it fits from its own
    stream, and reads a pinned artifact it did not fit.
    """

    name = "fit-scope-consumer"
    version = "0.1"
    scope_dependent = True

    def __init__(self, source: str, run_id: str) -> None:
        self.inputs = Inputs(("tracks",))
        self.params = _ArtifactParams(
            templates=TemplatesRef(feature=source, run_id=run_id)
        )
        self.fits = 0


class _ScopeFreeFit(_StatefulStub):
    """Fits from its stream but is scope-free, and pins no artifact.

    The shape of a fitter that reads its training data from somewhere no record
    describes (feral reads a directory of videos), so its fit scope is unknown.
    """

    name = "fit-scope-probe"
    version = "0.1"
    scope_dependent = False

    def __init__(self) -> None:
        self.inputs = Inputs(("tracks",))
        self.params = _P()
        self.fits = 0


def _scope(sequences: list[str] | None) -> Scope | None:
    return Scope(sequences=sequences) if sequences is not None else None


def _run(ds: Dataset, feature: Feature, sequences: list[str] | None) -> str:
    """Run *feature* over *sequences*; its run id."""
    run_id = run_feature(ds, feature, scope=_scope(sequences)).run_id
    assert run_id is not None
    return run_id


def _templates(ds: Dataset, sequences: list[str]) -> str:
    """Run the template source over *sequences*; its run id."""
    return _run(ds, _TemplateSource(), sequences)


def test_a_fit_records_the_scope_it_was_fitted_over(
    scenario_dataset: Dataset,
) -> None:
    run_id = _run(scenario_dataset, _TemplateSource(), ["seq_a"])

    fitted = read_fit_scope(feature_run_root(scenario_dataset, SOURCE_DIR, run_id))
    assert fitted is not None
    assert fitted.entries == (("", "seq_a"),)
    assert fitted.scope_dependent is True
    assert fitted.source == "stream"
    assert fitted.identity_scheme


def test_a_scope_free_fit_takes_its_scope_from_the_pinned_artifact(
    scenario_dataset: Dataset,
) -> None:
    """Invoked over two sequences, trained on templates drawn from one."""
    source_run = _templates(scenario_dataset, ["seq_a"])
    run_id = _run(
        scenario_dataset, _ArtifactFit(SOURCE_DIR, source_run), ["seq_a", "seq_b"]
    )

    fitted = read_fit_scope(feature_run_root(scenario_dataset, FIT_DIR, run_id))
    assert fitted is not None
    assert fitted.entries == (("", "seq_a"),), "the fit scope followed the invocation"
    assert fitted.scope_dependent is False
    assert fitted.source == "artifacts"
    assert [(ref.field, ref.feature, ref.run_id, ref.known) for ref in fitted.via] == [
        ("templates", SOURCE_DIR, source_run, True)
    ]


def test_a_scope_dependent_fit_keeps_its_stream_scope_and_names_the_artifact_it_read(
    scenario_dataset: Dataset,
) -> None:
    """Scored on one sequence, with a map fitted on another: the record says both."""
    source_run = _templates(scenario_dataset, ["seq_a"])

    run_id = _run(
        scenario_dataset, _ArtifactConsumer(SOURCE_DIR, source_run), ["seq_b"]
    )

    fitted = read_fit_scope(feature_run_root(scenario_dataset, CONSUMER_DIR, run_id))
    assert fitted is not None
    assert fitted.entries == (("", "seq_b"),), "the entries stay what the fit read"
    assert fitted.scope_dependent is True
    assert fitted.source == "stream"
    assert [(ref.field, ref.feature, ref.run_id, ref.known) for ref in fitted.via] == [
        ("templates", SOURCE_DIR, source_run, True)
    ]
    upstream = read_fit_scope(
        feature_run_root(scenario_dataset, SOURCE_DIR, source_run)
    )
    assert upstream is not None
    assert upstream.entries == (("", "seq_a"),)


def test_a_scope_dependent_fit_marks_an_artifact_with_no_record_as_unknown(
    scenario_dataset: Dataset,
) -> None:
    source_run = _templates(scenario_dataset, ["seq_a"])
    (
        feature_run_root(scenario_dataset, SOURCE_DIR, source_run) / FIT_SCOPE_NAME
    ).unlink()

    run_id = _run(
        scenario_dataset, _ArtifactConsumer(SOURCE_DIR, source_run), ["seq_b"]
    )

    fitted = read_fit_scope(feature_run_root(scenario_dataset, CONSUMER_DIR, run_id))
    assert fitted is not None
    assert fitted.entries == (("", "seq_b"),), "its own scope is known all the same"
    assert [ref.known for ref in fitted.via] == [False]


def test_the_fit_scope_carries_the_upstream_labels_and_tracks_variants(
    scenario_dataset: Dataset,
) -> None:
    source_run = _templates(scenario_dataset, ["seq_a"])
    source_root = feature_run_root(scenario_dataset, SOURCE_DIR, source_run)
    upstream = read_fit_scope(source_root)
    assert upstream is not None
    write_fit_scope(
        source_root,
        FitScope(
            scope_dependent=True,
            entries=upstream.entries,
            tracks_variants=("tracks-variant",),
            labels_variants=("labels-variant",),
            fitted_at=upstream.fitted_at,
            identity_scheme=upstream.identity_scheme,
        ),
    )

    run_id = _run(scenario_dataset, _ArtifactFit(SOURCE_DIR, source_run), None)

    fitted = read_fit_scope(feature_run_root(scenario_dataset, FIT_DIR, run_id))
    assert fitted is not None
    assert fitted.tracks_variants == ("tracks-variant",)
    assert fitted.labels_variants == ("labels-variant",)


def test_an_upstream_without_a_record_makes_the_fit_scope_unknown(
    scenario_dataset: Dataset,
) -> None:
    """Unknown upstream, unknown downstream -- never an empty or guessed set."""
    source_run = _templates(scenario_dataset, ["seq_a"])
    (
        feature_run_root(scenario_dataset, SOURCE_DIR, source_run) / FIT_SCOPE_NAME
    ).unlink()

    run_id = _run(scenario_dataset, _ArtifactFit(SOURCE_DIR, source_run), None)

    fitted = read_fit_scope(feature_run_root(scenario_dataset, FIT_DIR, run_id))
    assert fitted is not None
    assert fitted.source == "artifacts"
    assert fitted.entries is None
    assert fitted.tracks_variants is None
    assert fitted.labels_variants is None
    assert [ref.known for ref in fitted.via] == [False]


def test_a_scope_free_fit_with_no_artifact_source_is_unknown(
    scenario_dataset: Dataset,
) -> None:
    run_id = _run(scenario_dataset, _ScopeFreeFit(), ["seq_a"])

    fitted = read_fit_scope(feature_run_root(scenario_dataset, PROBE_DIR, run_id))
    assert fitted is not None
    assert fitted.source == "unknown"
    assert fitted.entries is None
    assert fitted.via == ()
    fit, apply = fit_and_apply_scopes(scenario_dataset, PROBE_DIR, run_id)
    assert fit is None
    assert apply == frozenset({("", "seq_a")})


def test_a_wider_apply_leaves_the_fit_scope_narrow(
    scenario_dataset: Dataset,
) -> None:
    """The finding this item exists for, pinned in both directions.

    ``params.json`` is written unconditionally at the top of every run, so its
    ``_scope.entries`` widens to whatever ran last. The fit record is written
    only when a fit actually ran, so it stays what it says it is. Asserting only
    the second half would pass against a record that simply never updated.
    """
    source_run = _templates(scenario_dataset, ["seq_a"])
    feature = _ArtifactFit(SOURCE_DIR, source_run)
    first = _run(scenario_dataset, feature, ["seq_a"])
    assert feature.fits == 1

    second = _run(scenario_dataset, feature, ["seq_a", "seq_b"])
    assert second == first, "a scope-free feature must not re-identify"
    assert feature.fits == 1, "the state was reloaded, so no second fit ran"

    run_root = feature_run_root(scenario_dataset, FIT_DIR, second)
    fitted = read_fit_scope(run_root)
    assert fitted is not None
    assert fitted.entries == (("", "seq_a"),), "the fit scope followed the apply scope"

    saved = json.loads((run_root / "params.json").read_text())
    widened = {tuple(entry) for entry in saved["_scope"]["entries"]}
    assert widened == {("", "seq_a"), ("", "seq_b")}, (
        "params.json is expected to widen; that is why it cannot be the fit record"
    )


def test_the_pairing_is_derived_in_one_place(scenario_dataset: Dataset) -> None:
    """Fit and apply come back together, and a row is in both or only in apply."""
    source_run = _templates(scenario_dataset, ["seq_a"])
    feature = _ArtifactFit(SOURCE_DIR, source_run)
    first = _run(scenario_dataset, feature, ["seq_a"])
    _ = _run(scenario_dataset, feature, ["seq_a", "seq_b"])

    fit, apply = fit_and_apply_scopes(scenario_dataset, FIT_DIR, first)
    assert fit == frozenset({("", "seq_a")})
    assert apply == frozenset({("", "seq_a"), ("", "seq_b")})
    assert fit is not None and fit < apply, "seq_b was applied but never fitted"


def test_a_run_that_predates_the_record_reads_as_unknown(tmp_path: Path) -> None:
    """``None`` is unknown, never "fitted over nothing".

    Reading an absent record as an empty fit scope would report every run written
    before item 5.3 as having trained on nothing -- a confident wrong answer where
    an honest unknown was available.
    """
    assert read_fit_scope(tmp_path) is None


def _write_record(run_root: Path, payload: dict[str, object]) -> None:
    _ = (run_root / FIT_SCOPE_NAME).write_text(json.dumps(payload))


def test_a_legacy_scope_free_record_reads_as_unknown_entries(tmp_path: Path) -> None:
    """A record without ``source`` held the invocation scope, not the fit's."""
    _write_record(
        tmp_path,
        {
            "scope_dependent": False,
            "entries": [["", "seq_a"]],
            "tracks_variants": ["t"],
            "labels_variants": [],
            "fitted_at": "2026-01-01T00:00:00Z",
            "identity_scheme": "x",
        },
    )

    fitted = read_fit_scope(tmp_path)

    assert fitted is not None
    assert fitted.source == "legacy"
    assert fitted.entries is None
    assert fitted.tracks_variants is None
    assert fitted.labels_variants is None


def test_a_legacy_scope_dependent_record_keeps_its_entries(tmp_path: Path) -> None:
    _write_record(
        tmp_path,
        {
            "scope_dependent": True,
            "entries": [["", "seq_a"]],
            "tracks_variants": ["t"],
            "labels_variants": [],
            "fitted_at": "2026-01-01T00:00:00Z",
            "identity_scheme": "x",
        },
    )

    fitted = read_fit_scope(tmp_path)

    assert fitted is not None
    assert fitted.source == "legacy"
    assert fitted.entries == (("", "seq_a"),)
    assert fitted.tracks_variants == ("t",)


def test_a_null_entries_record_reads_as_unknown(tmp_path: Path) -> None:
    """JSON ``null`` is unknown; reading it through the list path would give ``()``."""
    _write_record(
        tmp_path,
        {
            "scope_dependent": False,
            "entries": None,
            "tracks_variants": None,
            "labels_variants": None,
            "fitted_at": "2026-01-01T00:00:00Z",
            "identity_scheme": "x",
            "source": "unknown",
            "via": [],
        },
    )

    fitted = read_fit_scope(tmp_path)

    assert fitted is not None
    assert fitted.source == "unknown"
    assert fitted.entries is None
    assert fitted.tracks_variants is None


def test_a_cache_hit_does_not_rewrite_the_record(scenario_dataset: Dataset) -> None:
    """It is written in the fit branch, so a reload must leave it alone."""
    source_run = _templates(scenario_dataset, ["seq_a"])
    feature = _ArtifactFit(SOURCE_DIR, source_run)
    run_id = _run(scenario_dataset, feature, ["seq_a"])
    run_root = feature_run_root(scenario_dataset, FIT_DIR, run_id)
    before = (run_root / FIT_SCOPE_NAME).stat().st_mtime_ns

    _ = _run(scenario_dataset, feature, ["seq_a"])

    assert (run_root / FIT_SCOPE_NAME).stat().st_mtime_ns == before
