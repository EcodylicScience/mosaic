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

import hashlib
import json
import os
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.file_digest import MODEL_DIGEST_HEX, file_digest
from mosaic.core.pipeline.fit_scope import (
    FIT_SCOPE_NAME,
    FitEntryInputs,
    FitScope,
    FitSourceRef,
    FitStateUnreadable,
    fit_and_apply_scopes,
    fit_references,
    list_run_root,
    read_fit_scope,
    read_fit_state_digest,
    reported_fit_entries,
    read_run_fit_scope,
    records_fit_inputs,
    saved_state_digest,
    write_fit_scope,
)
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.graph import build_feature
from mosaic.core.pipeline.index import (
    DriftReads,
    EntryCells,
    FitReference,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.markers import (
    new_inflight,
    refresh_inflight,
    write_inflight,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    DefinitionModel,
    DependencyLookup,
    Feature,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    NNResult,
    Result,
    TemplatesRef,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.params import Params
from mosaic.core.scope import Scope
from tests.helpers import kpms_params

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
    describes, so its fit scope is unknown.
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


PEER_DIGEST = "0" * MODEL_DIGEST_HEX
"""The digest a peer's fit records while a run loads the state before it."""


class _RefittedWhileLoading(_TemplateSource):
    """Loads its state while a peer fits the run again and records that."""

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        loaded = super().load_state(run_root, artifact_paths, dependency_lookups)
        record = run_root / FIT_SCOPE_NAME
        if loaded and record.is_file():
            stored: dict[str, object] = json.loads(record.read_text())
            stored["state_digest"] = PEER_DIGEST
            _ = record.write_text(json.dumps(stored))
        return loaded


def _model_state_cells(ds: Dataset, run_id: str) -> dict[str, str]:
    """The ``consumed_model_state`` of each row of the source run *run_id*."""
    rows = pd.read_csv(
        feature_index_path(ds, SOURCE_DIR), dtype=str, keep_default_na=False
    )
    rows = rows[rows["run_id"] == run_id]
    return dict(zip(rows["sequence"], rows["consumed_model_state"], strict=True))


def test_rows_record_the_digest_of_the_state_loaded_not_a_peers_refit(
    scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run reads its fit record before ``load_state``, not after it."""
    run_id = _run(scenario_dataset, _TemplateSource(), ["seq_a"])
    loaded = read_fit_state_digest(
        feature_run_root(scenario_dataset, SOURCE_DIR, run_id)
    )
    assert loaded
    assert _model_state_cells(scenario_dataset, run_id) == {"seq_a": loaded}
    _ = capsys.readouterr()

    again = _run(scenario_dataset, _RefittedWhileLoading(), ["seq_a"])

    assert again == run_id
    assert "cannot serve cached output" not in capsys.readouterr().err
    assert _model_state_cells(scenario_dataset, run_id) == {"seq_a": loaded}


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


# --- the digest of the state a fit saved ------------------------------------


def _empty_digest() -> str:
    """The digest of a save that wrote no file."""
    return hashlib.blake2b(b"[]", digest_size=MODEL_DIGEST_HEX // 2).hexdigest()


def _fitted_run_root(tmp_path: Path) -> Path:
    """A run root holding what a run writes before ``save_state``."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    _ = (run_root / "params.json").write_text("{}")
    _ = (run_root / ".identity_scheme").write_text("scheme")
    _ = (run_root / "g__s1.parquet").write_bytes(b"entry output")
    return run_root


def _save_state(run_root: Path) -> None:
    """A save that writes a state file and a nested state file."""
    _ = (run_root / "state.bin").write_bytes(b"state")
    (run_root / "model").mkdir()
    _ = (run_root / "model" / "weights.bin").write_bytes(b"weights")


def _expected(run_root: Path, paths: list[str]) -> str:
    """The digest of *paths*, as the record names it."""
    pairs = [[path, file_digest(run_root / path)] for path in sorted(paths)]
    payload = json.dumps(pairs).encode()
    return hashlib.blake2b(payload, digest_size=MODEL_DIGEST_HEX // 2).hexdigest()


def test_the_digest_covers_exactly_the_files_the_save_wrote(tmp_path: Path) -> None:
    run_root = _fitted_run_root(tmp_path)
    before = list_run_root(run_root)

    _save_state(run_root)

    after = list_run_root(run_root)
    assert sorted(after) == ["g__s1.parquet", "model/weights.bin", "state.bin"]
    digest = saved_state_digest(run_root, before, after)
    assert digest == _expected(run_root, ["model/weights.bin", "state.bin"])
    assert len(digest) == MODEL_DIGEST_HEX
    assert int(digest, 16) >= 0


def test_the_digest_covers_a_file_the_save_rewrote(tmp_path: Path) -> None:
    """A file present before the save counts when its size or mtime changed."""
    run_root = _fitted_run_root(tmp_path)
    _ = (run_root / "resized.bin").write_bytes(b"old")
    _ = (run_root / "touched.bin").write_bytes(b"old")
    before = list_run_root(run_root)

    _ = (run_root / "resized.bin").write_bytes(b"longer")
    touched = run_root / "touched.bin"
    _ = touched.write_bytes(b"new")
    stamp = touched.stat().st_mtime_ns + 1_000_000_000
    os.utime(touched, ns=(stamp, stamp))

    after = list_run_root(run_root)
    assert saved_state_digest(run_root, before, after) == _expected(
        run_root, ["resized.bin", "touched.bin"]
    )


def test_the_digest_does_not_depend_on_listing_order(tmp_path: Path) -> None:
    run_root = _fitted_run_root(tmp_path)
    before = list_run_root(run_root)
    _save_state(run_root)
    after = list_run_root(run_root)

    reordered = dict(reversed(list(after.items())))

    assert list(reordered) != list(after)
    assert saved_state_digest(run_root, before, reordered) == saved_state_digest(
        run_root, before, after
    )


def test_a_save_that_writes_nothing_gives_the_digest_of_no_files(
    tmp_path: Path,
) -> None:
    run_root = _fitted_run_root(tmp_path)
    listing = list_run_root(run_root)

    assert saved_state_digest(run_root, listing, listing) == _empty_digest()


def _scope_with_digest(digest: str) -> FitScope:
    return FitScope(
        scope_dependent=True,
        entries=(("", "seq_a"),),
        tracks_variants=(),
        labels_variants=(),
        fitted_at="2026-01-01T00:00:00Z",
        identity_scheme="x",
        state_digest=digest,
    )


def test_the_record_holds_the_state_digest(scenario_dataset: Dataset) -> None:
    run_root = feature_run_root(scenario_dataset, FIT_DIR, "0.1-abc")
    run_root.mkdir(parents=True)

    write_fit_scope(run_root, _scope_with_digest("0123456789abcdef"))

    stored = json.loads((run_root / FIT_SCOPE_NAME).read_text())
    assert stored["state_digest"] == "0123456789abcdef"
    fitted = read_fit_scope(run_root)
    assert fitted is not None
    assert fitted.state_digest == "0123456789abcdef"
    linked = read_run_fit_scope(scenario_dataset, FIT_DIR, "0.1-abc")
    assert linked is not None
    assert linked.state_digest == "0123456789abcdef"
    assert read_fit_state_digest(run_root) == "0123456789abcdef"


def test_a_record_without_the_digest_reads_as_blank(tmp_path: Path) -> None:
    _write_record(
        tmp_path,
        {
            "scope_dependent": True,
            "entries": [["", "seq_a"]],
            "tracks_variants": [],
            "labels_variants": [],
            "fitted_at": "2026-01-01T00:00:00Z",
            "identity_scheme": "x",
            "source": "stream",
            "via": [],
        },
    )

    fitted = read_fit_scope(tmp_path)

    assert fitted is not None
    assert fitted.state_digest == ""
    assert read_fit_state_digest(tmp_path) == ""


def test_a_run_without_a_record_has_a_blank_digest(tmp_path: Path) -> None:
    assert read_fit_state_digest(tmp_path) == ""


@pytest.mark.parametrize(
    "text",
    ["[1, 2]", "not json", json.dumps({"state_digest": 7})],
    ids=["not-an-object", "not-json", "not-a-string"],
)
def test_an_unreadable_record_refuses_the_digest_read(
    tmp_path: Path, text: str
) -> None:
    """``read_fit_scope`` reads the same file as unknown without raising."""
    run_root = tmp_path / "fit-scope-fit__from__tracks" / "0.1-abc"
    run_root.mkdir(parents=True)
    _ = (run_root / FIT_SCOPE_NAME).write_text(text)

    with pytest.raises(FitStateUnreadable) as raised:
        _ = read_fit_state_digest(run_root)

    assert raised.value.error_kind == "conflict"
    assert raised.value.error_detail() == {
        "feature": "fit-scope-fit__from__tracks",
        "run_id": "0.1-abc",
    }
    assert "run 0.1-abc of fit-scope-fit__from__tracks" in str(raised.value)
    assert read_fit_scope(run_root) is None
    with pytest.raises(FitStateUnreadable):
        _ = read_fit_scope(run_root, strict=True)


def test_a_strict_read_of_a_readable_record_or_none_is_the_plain_read(
    tmp_path: Path,
) -> None:
    assert read_fit_scope(tmp_path, strict=True) is None
    _write_record(tmp_path, {"scope_dependent": True, "state_digest": "d1"})

    strict = read_fit_scope(tmp_path, strict=True)

    assert strict is not None
    assert strict == read_fit_scope(tmp_path)
    assert strict.state_digest == "d1"


class TestDriftReadsFitRecord:
    """``DriftReads.fit_record`` reads a run's ``fit_scope.json`` once."""

    RUN = "0.1-abc"

    def _root(self, ds: Dataset) -> Path:
        root = feature_run_root(ds, FIT_DIR, self.RUN)
        root.mkdir(parents=True)
        return root

    def test_a_record_is_read_once(
        self, scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mosaic.core.pipeline import fit_scope as fit_scope_module

        _write_record(self._root(scenario_dataset), {"state_digest": "d1"})
        reads: list[Path] = []

        def counted(run_root: Path, *, strict: bool = False) -> FitScope | None:
            reads.append(run_root)
            return read_fit_scope(run_root, strict=strict)

        monkeypatch.setattr(fit_scope_module, "read_fit_scope", counted)
        shared = DriftReads(scenario_dataset)

        first = shared.fit_record(FIT_DIR, self.RUN)
        second = shared.fit_record(FIT_DIR, self.RUN)

        assert first is not None
        assert first.state_digest == "d1"
        assert second is first
        assert len(reads) == 1

    def test_a_run_without_a_record_reads_none(self, scenario_dataset: Dataset) -> None:
        assert DriftReads(scenario_dataset).fit_record(FIT_DIR, self.RUN) is None

    def test_an_unparsable_record_reads_none_and_strictly_raises(
        self, scenario_dataset: Dataset
    ) -> None:
        root = self._root(scenario_dataset)
        _ = (root / FIT_SCOPE_NAME).write_text("{not json")

        assert DriftReads(scenario_dataset).fit_record(FIT_DIR, self.RUN) is None
        with pytest.raises(FitStateUnreadable) as raised:
            _ = DriftReads(scenario_dataset, strict=True).fit_record(FIT_DIR, self.RUN)
        assert raised.value.error_detail() == {"feature": FIT_DIR, "run_id": self.RUN}


def test_a_fit_records_the_digest_of_the_state_it_saved(
    scenario_dataset: Dataset,
) -> None:
    run_id = _templates(scenario_dataset, ["seq_a"])
    run_root = feature_run_root(scenario_dataset, SOURCE_DIR, run_id)

    recorded = read_fit_state_digest(run_root)

    assert recorded == _expected(run_root, ["state.txt", "templates.parquet"])


def _claimed_run_root(tmp_path: Path) -> Path:
    """A run root as a fit leaves it before ``save_state``: under its claim."""
    run_root = _fitted_run_root(tmp_path)
    marker = new_inflight(
        execution_id="fit", host="h", pid=1, phase=None, idle_seconds=60.0
    )
    write_inflight(run_root, marker)
    return run_root


def test_a_claim_refreshed_during_the_save_is_left_out(tmp_path: Path) -> None:
    """The claim timer rewrites the marker by rename while the fit saves."""
    run_root = _claimed_run_root(tmp_path)
    before = list_run_root(run_root)

    _save_state(run_root)
    marker = new_inflight(
        execution_id="fit", host="h", pid=1, phase=None, idle_seconds=60.0
    )
    assert refresh_inflight(run_root, marker, 3600.0) is not None

    after = list_run_root(run_root)
    assert saved_state_digest(run_root, before, after) == _expected(
        run_root, ["model/weights.bin", "state.bin"]
    )


def test_run_bookkeeping_a_peer_writes_during_the_save_is_left_out(
    tmp_path: Path,
) -> None:
    """A second invocation of the run rewrites ``params.json`` and the scheme marker.

    A write still in flight leaves its temporary file, a claim's or any other.
    """
    run_root = _claimed_run_root(tmp_path)
    before = list_run_root(run_root)

    _save_state(run_root)
    _ = (run_root / "params.json").write_text(json.dumps({"peer": True}))
    write_identity_scheme(run_root, "peer")
    _ = (run_root / FIT_SCOPE_NAME).write_text("{}")
    _ = (run_root / "..mosaic-inflight-abc123.tmp").write_text("{")
    _ = (run_root / ".params-abc123.tmp").write_text("{")
    _ = (run_root / ".g__s2-abc123.tmp").write_bytes(b"partial")

    after = list_run_root(run_root)
    assert sorted(after) == ["g__s1.parquet", "model/weights.bin", "state.bin"]
    assert saved_state_digest(run_root, before, after) == _expected(
        run_root, ["model/weights.bin", "state.bin"]
    )


_UPSTREAM = [{"feature": "speed-angvel__from__tracks", "run_id": "0.1-up"}]


def _params_of(slug: str, params: dict[str, object]) -> Params:
    return build_feature(slug, _UPSTREAM, params).params


def test_a_pinned_templates_reference_names_the_templates_file() -> None:
    params = _params_of(
        "global-kmeans",
        {"templates": {"feature": "extract-templates", "run_id": "0.1-tpl"}},
    )

    assert fit_references(params) == (
        FitReference(
            field="templates",
            feature="extract-templates",
            run_id="0.1-tpl",
            pattern="templates.parquet",
        ),
    )


def test_a_pinned_model_artifact_names_its_file() -> None:
    params = _params_of("global-kmeans", {"model": {"run_id": "0.1-fit"}})

    assert fit_references(params) == (
        FitReference(
            field="model",
            feature="global-kmeans",
            run_id="0.1-fit",
            pattern="model.joblib",
        ),
    )


def test_the_references_are_sorted_by_field_and_skip_an_unpinned_model() -> None:
    """``init_from`` and ``templates``; the default ``model`` names no run."""
    params = _params_of(
        "sequence-classifier",
        {
            "default_class": 0,
            "labels": {},
            "templates": {"feature": "extract-labeled-templates", "run_id": "0.1-t"},
            "init_from": {"run_id": "0.1-i"},
        },
    )

    assert fit_references(params) == (
        FitReference(
            field="init_from",
            feature="sequence-classifier",
            run_id="0.1-i",
            pattern="sequence_classifier_model.joblib",
        ),
        FitReference(
            field="templates",
            feature="extract-labeled-templates",
            run_id="0.1-t",
            pattern="templates.parquet",
        ),
    )


def test_a_label_map_is_a_fit_reference() -> None:
    params = _params_of(
        "evaluate-labels", {"label_map": {"feature": "fit-label-map", "run_id": "l"}}
    )

    assert [ref.field for ref in fit_references(params)] == ["label_map"]


def test_a_thresholds_artifact_is_a_fit_reference_and_a_float_is_not() -> None:
    tuned = _params_of(
        "smooth-probabilities",
        {"default_class": 0, "method": "none", "decision_threshold": {"run_id": "t"}},
    )
    fixed = _params_of(
        "smooth-probabilities", {"default_class": 0, "decision_threshold": 0.5}
    )

    assert [ref.field for ref in fit_references(tuned)] == ["decision_threshold"]
    assert fit_references(fixed) == ()


class _DefinitionParams(Params):
    model: DefinitionModel | None = None
    upstream: Result[str] | None = None
    pair_filter: NNResult | None = None
    labels: GroundTruthLabelsSource = GroundTruthLabelsSource()
    templates: TemplatesRef | None = None


def test_a_pinned_definition_model_has_no_pattern() -> None:
    params = _DefinitionParams(
        model=DefinitionModel(feature="def:mine", run_id="0.1-fit"),
        upstream=Result(feature="speed-angvel__from__tracks", run_id="0.1-up"),
        pair_filter=NNResult(run_id="0.1-nn"),
    )

    assert fit_references(params) == (
        FitReference(field="model", feature="def:mine", run_id="0.1-fit", pattern=None),
    )


def test_unpinned_references_and_other_results_are_not_fit_references() -> None:
    params = _DefinitionParams(
        model=DefinitionModel(feature="def:mine"),
        upstream=Result(feature="speed-angvel__from__tracks", run_id="0.1-up"),
        pair_filter=NNResult(run_id="0.1-nn"),
        templates=TemplatesRef(feature="extract-templates"),
    )

    assert fit_references(params) == ()


# --- the fit's inputs, its attempt and the states it read -------------------


def _fitted(
    inputs: tuple[FitEntryInputs, ...] | None,
    via: tuple[FitSourceRef, ...] = (),
    execution_id: str = "",
) -> FitScope:
    return FitScope(
        scope_dependent=True,
        entries=(("g1", "s1"), ("g2", "s2")),
        tracks_variants=(),
        labels_variants=(),
        fitted_at="2026-01-01T00:00:00Z",
        identity_scheme="x",
        via=via,
        state_digest="d1",
        inputs=inputs,
        execution_id=execution_id,
    )


_INPUTS = (
    FitEntryInputs(
        group="g1",
        sequence="s1",
        cells=EntryCells(roots="", tracks="b2c4e1", inputs=""),
    ),
    FitEntryInputs(
        group="g1",
        sequence="s2",
        cells=EntryCells(roots="r1", tracks="c3d5f2", inputs="i1"),
    ),
    FitEntryInputs(
        group="g2",
        sequence="s1",
        cells=EntryCells(roots="", tracks="", inputs="i2"),
    ),
)
_VIA = (
    FitSourceRef(
        field="model", feature="up", run_id="0.1-aa", known=True, state_digest="9af07c"
    ),
    FitSourceRef(
        field="templates", feature="tp", run_id="0.2-bb", known=False, state_digest=""
    ),
)


def test_the_fit_inputs_the_attempt_and_the_states_read_round_trip(
    tmp_path: Path,
) -> None:
    write_fit_scope(tmp_path, _fitted(_INPUTS, _VIA, "01JD8M3Q6W2F5Y0RKX7T4N9BCA"))

    fitted = read_fit_scope(tmp_path)

    assert fitted is not None
    assert fitted.inputs == _INPUTS
    assert fitted.execution_id == "01JD8M3Q6W2F5Y0RKX7T4N9BCA"
    assert fitted.via == _VIA
    assert [ref.state_digest for ref in fitted.via] == ["9af07c", ""]
    stored = json.loads((tmp_path / FIT_SCOPE_NAME).read_text())
    assert stored["fit_inputs"]["entries"][0] == {
        "group": "g1",
        "sequence": "s1",
        "roots": "",
        "tracks": "b2c4e1",
        "inputs": "",
    }
    assert stored["via"][0]["state_digest"] == "9af07c"


def test_an_empty_set_of_fit_inputs_is_a_record_and_none_is_not(
    tmp_path: Path,
) -> None:
    write_fit_scope(tmp_path, _fitted(()))
    recorded = read_fit_scope(tmp_path)
    assert recorded is not None
    assert recorded.inputs == ()

    write_fit_scope(tmp_path, _fitted(None))

    unrecorded = read_fit_scope(tmp_path)
    assert unrecorded is not None
    assert unrecorded.inputs is None
    assert "fit_inputs" not in json.loads((tmp_path / FIT_SCOPE_NAME).read_text())


def test_the_files_bytes_do_not_depend_on_the_order_given(tmp_path: Path) -> None:
    ordered = tmp_path / "ordered"
    shuffled = tmp_path / "shuffled"
    ordered.mkdir()
    shuffled.mkdir()

    write_fit_scope(ordered, _fitted(_INPUTS, _VIA))
    write_fit_scope(shuffled, _fitted(_INPUTS[::-1], _VIA[::-1]))

    assert (shuffled / FIT_SCOPE_NAME).read_bytes() == (
        ordered / FIT_SCOPE_NAME
    ).read_bytes()
    fitted = read_fit_scope(shuffled)
    assert fitted is not None
    assert fitted.inputs == _INPUTS
    assert fitted.via == _VIA


def test_a_record_without_the_new_keys_reads_them_as_unrecorded(
    tmp_path: Path,
) -> None:
    _write_record(
        tmp_path,
        {
            "scope_dependent": True,
            "entries": [["", "seq_a"]],
            "tracks_variants": [],
            "labels_variants": [],
            "fitted_at": "2026-01-01T00:00:00Z",
            "identity_scheme": "x",
            "source": "stream",
            "via": [
                {"field": "model", "feature": "up", "run_id": "0.1-aa", "known": True}
            ],
            "state_digest": "d1",
        },
    )

    fitted = read_fit_scope(tmp_path)

    assert fitted is not None
    assert fitted.inputs is None
    assert fitted.execution_id == ""
    assert [ref.state_digest for ref in fitted.via] == [""]


@pytest.mark.parametrize(
    "fit_inputs",
    [[], "entries", None, {"entries": {"g1": "s1"}}, {"rows": []}],
    ids=["list", "string", "null", "entries-not-a-list", "no-entries"],
)
def test_a_fit_inputs_that_is_not_an_object_with_entries_is_unreadable(
    tmp_path: Path, fit_inputs: object
) -> None:
    run_root = tmp_path / "fit-scope-fit__from__tracks" / "0.1-abc"
    run_root.mkdir(parents=True)
    _write_record(
        run_root,
        {"scope_dependent": True, "state_digest": "d1", "fit_inputs": fit_inputs},
    )

    assert read_fit_scope(run_root) is None
    with pytest.raises(FitStateUnreadable) as raised:
        _ = read_fit_scope(run_root, strict=True)

    assert "the fit this run reads" in str(raised.value)
    assert "run 0.1-abc of fit-scope-fit__from__tracks" in str(raised.value)
    with pytest.raises(FitStateUnreadable):
        _ = read_fit_state_digest(run_root)


def test_an_entry_of_the_fit_inputs_that_is_not_an_object_is_unreadable_strictly(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "fit-scope-fit__from__tracks" / "0.1-abc"
    run_root.mkdir(parents=True)
    entry = {"group": "g1", "sequence": "s1", "roots": "", "tracks": "t", "inputs": ""}
    _write_record(
        run_root,
        {
            "scope_dependent": True,
            "state_digest": "d1",
            "fit_inputs": {"entries": [entry, "g2:s2"]},
        },
    )

    lenient = read_fit_scope(run_root)
    assert lenient is not None
    assert [(item.group, item.sequence) for item in lenient.inputs or ()] == [
        ("g1", "s1")
    ]
    with pytest.raises(FitStateUnreadable) as raised:
        _ = read_fit_scope(run_root, strict=True)

    assert "an entry of its fit_inputs is not an object" in str(raised.value)
    assert "run 0.1-abc of fit-scope-fit__from__tracks" in str(raised.value)
    with pytest.raises(FitStateUnreadable):
        _ = read_fit_state_digest(run_root)


def test_a_fit_inputs_object_with_an_entries_list_is_readable_strictly(
    tmp_path: Path,
) -> None:
    _write_record(
        tmp_path,
        {
            "scope_dependent": True,
            "state_digest": "d1",
            "fit_inputs": {
                "entries": [
                    {
                        "group": "g1",
                        "sequence": "s1",
                        "roots": "",
                        "tracks": "b2c4e1",
                        "inputs": "",
                    }
                ]
            },
        },
    )

    fitted = read_fit_scope(tmp_path, strict=True)

    assert fitted is not None
    assert fitted.inputs == _INPUTS[:1]


_APPLIED_RUN: dict[str, object] = {"feature": "up", "run_id": "0.1-aa"}
_APPLIED_INPUT: list[object] = [_APPLIED_RUN]
_TEMPLATES_REF: dict[str, object] = {
    "feature": "extract-templates",
    "run_id": "0.1-aa",
    "pattern": "templates.parquet",
    "load": {"kind": "parquet"},
}


@pytest.mark.parametrize(
    ("slug", "params", "records"),
    [
        ("extract-templates", {"n_templates": 5}, True),
        (
            "sequence-classifier",
            {"templates": _TEMPLATES_REF, "labels": {}, "default_class": 0},
            True,
        ),
        ("evaluate-labels", {}, True),
        ("kpms", kpms_params(), True),
        ("speed-angvel", {}, False),
        ("global-kmeans", {"model": _APPLIED_RUN}, False),
        ("sequence-classifier", {"model": _APPLIED_RUN, "default_class": 0}, False),
        ("kpms", kpms_params(model=_APPLIED_RUN), False),
    ],
    ids=[
        "extract-templates",
        "sequence-classifier-fitting",
        "evaluate-labels",
        "kpms-fitting",
        "speed-angvel",
        "global-kmeans-given-a-model",
        "sequence-classifier-given-a-model",
        "kpms-given-a-model",
    ],
)
def test_a_run_records_fit_inputs_when_it_fits_or_depends_on_its_scope(
    slug: str, params: dict[str, object], records: bool
) -> None:
    feature = build_feature(slug, _APPLIED_INPUT, params)

    assert records_fit_inputs(feature) is records


class _ReportsFitEntries(_ScopeFreeFit):
    """A fit pinned to its training set that reports the entries it streams."""

    fit_set_pinned_by: ClassVar[str] = "templates"

    def __init__(self, entries: tuple[Entry, ...] | None) -> None:
        super().__init__()
        self.fitted_entries = entries


class _ReportsNothing(_ScopeFreeFit):
    """A fit pinned to its training set that never sets ``fitted_entries``."""

    fit_set_pinned_by: ClassVar[str] = "templates"


class _ReportsWithoutPinning(_ScopeFreeFit):
    """A fit that sets ``fitted_entries`` and declares no pinned fit set."""

    def __init__(self) -> None:
        super().__init__()
        self.fitted_entries: tuple[Entry, ...] = (("g1", "s1"),)


def test_the_entries_a_fit_reports_come_back_sorted() -> None:
    feature = _ReportsFitEntries((("g2", "s1"), ("g1", "s2"), ("g1", "s1")))

    assert reported_fit_entries(feature) == (
        ("g1", "s1"),
        ("g1", "s2"),
        ("g2", "s1"),
    )


def test_a_fit_that_reports_an_empty_set_reports_an_empty_tuple() -> None:
    assert reported_fit_entries(_ReportsFitEntries(())) == ()


@pytest.mark.parametrize(
    "feature",
    [
        _ReportsFitEntries(None),
        _ReportsNothing(),
        _ReportsWithoutPinning(),
        _ScopeFreeFit(),
    ],
    ids=["reports-none", "sets-nothing", "declares-no-pin", "has-no-attribute"],
)
def test_a_fit_that_reports_no_entries_reports_none(feature: Feature) -> None:
    assert reported_fit_entries(feature) is None
