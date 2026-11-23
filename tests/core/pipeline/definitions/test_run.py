"""Running a loaded definition: its storage, its identity and its workers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope, new_execution_id
from mosaic.core.pipeline.definitions.define import (
    define_feature,
    load_feature_definition,
)
from mosaic.core.pipeline.definitions.loader import definition_ref_of
from mosaic.core.pipeline.definitions.store import (
    DefinitionModelMissing,
    DefinitionModelRefused,
)
from mosaic.core.pipeline.definitions.scope import (
    DefinitionScope,
    current_scope,
    definition_scope,
)
from mosaic.core.pipeline.definitions.sidecar import DefinitionRef
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.fit_scope import (
    list_run_root,
    read_fit_state_digest,
    saved_state_digest,
)
from mosaic.core.pipeline.graph.resolve import build_feature
from mosaic.core.pipeline.index import feature_index_path, feature_run_root
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.run import (
    compute_run_id,
    resolve_feature_identity,
    run_feature,
)
from mosaic.core.pipeline.resolve import Resolution
from mosaic.core.pipeline.run_log import read_run, run_log_dir
from mosaic.core.pipeline.types import Feature, Result
from mosaic.core.scope import Scope
from tests.helpers import (
    FITTED_AGAIN,
    APPLYING_NAME,
    APPLYING_STORAGE,
    APPLYING_SECOND_SOURCE,
    APPLYING_SOURCE,
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    MISREADING_NAME,
    MISREADING_SOURCE,
    PARALLEL_APPLYING_NAME,
    PARALLEL_APPLYING_SOURCE,
    PlainFeature,
    add_track_sequences,
    built_by_class,
    defined_dataset,
    fit_over,
    fit_result,
    make_dataset,
    model_cells,
    model_reference,
    refusals,
    result_run_root,
    run_fresh,
    track_table,
    tree_snapshot,
    write_run_params,
)

WORKERS: Final = 2


def _written(ds: Dataset, result: Result[str], name: str) -> Path:
    """The run root of *result*, after checking it is the definition's storage."""
    assert result.feature == f"def.{name}__from__tracks"
    root = result_run_root(ds, result)
    assert sorted(path.name for path in root.glob("*.parquet")) == [
        "g__s1.parquet",
        "g__s2.parquet",
    ]
    return root


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_a_loaded_definition_runs_in_workers(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both entries are written under the definition's storage with its identity."""
    canary = tmp_path / "canary.jsonl"
    ds = defined_dataset(tmp_path, {CANARY_NAME: CANARY_SOURCE})
    digest = hashlib.sha256(CANARY_SOURCE.encode()).hexdigest()
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    feature = load_feature_definition(ds, f"def:{CANARY_NAME}")()

    result = run_feature(
        ds, feature, parallel_workers=WORKERS, parallel_mode=mode, track=False
    )

    root = _written(ds, result, CANARY_NAME)
    expected, _ = compute_run_id(feature, None, None, ResolvedScope())
    assert result.run_id == expected
    document = json.loads((root / "params.json").read_text())
    assert document["_definition"] == {"name": CANARY_NAME, "digest": digest}
    read = read_run_params(root)
    assert read.params is not None
    assert read.params.definition == DefinitionRef(CANARY_NAME, digest)
    importers = {json.loads(line)["pid"] for line in canary.read_text().splitlines()}
    if mode == "process":
        assert importers - {os.getpid()}
    else:
        assert importers == {os.getpid()}


def test_a_builtin_run_records_no_definition(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "plain")
    add_track_sequences(ds, ("g", "s1"))

    result = run_feature(ds, _RecordingFeature(), track=False)

    root = result_run_root(ds, result)
    assert "_definition" not in json.loads((root / "params.json").read_text())
    read = read_run_params(root)
    assert read.params is not None
    assert read.params.definition is None


class _RecordingFeature(PlainFeature):
    """A feature that notes the definition scope its ``apply`` runs under."""

    name = "scope-recording"
    accepts_overlap = False
    emits: EmitsLevel = "as-input"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[DefinitionScope | None] = []

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.seen.append(current_scope())
        return df


class _RecordingDefinition(_RecordingFeature):
    """The same, marked as the loaded class of a definition."""

    name = "def:scope-recording"
    __mosaic_definition__ = DefinitionRef("scope-recording", "d" * 64)


@pytest.fixture
def two_datasets(tmp_path: Path) -> tuple[Dataset, Dataset]:
    own = make_dataset(tmp_path / "own")
    add_track_sequences(own, ("g", "s1"))
    other = make_dataset(tmp_path / "other")
    return own, other


@pytest.mark.parametrize("feature_type", [_RecordingFeature, _RecordingDefinition])
@pytest.mark.parametrize("entered", ["no-dataset", "another-dataset"])
def test_a_run_inside_a_disabled_scope_keeps_it_disabled(
    feature_type: type[_RecordingFeature],
    entered: str,
    two_datasets: tuple[Dataset, Dataset],
) -> None:
    """A disabled scope is not replaced, filled or widened by a run."""
    own, other = two_datasets
    feature = feature_type()

    with definition_scope(
        None if entered == "no-dataset" else other, load=False, allowed=False
    ) as outer:
        _ = run_feature(own, feature, track=False)

    assert feature.seen == [outer]
    assert outer.allowed is False
    assert outer.load is False


@pytest.mark.parametrize("feature_type", [_RecordingFeature, _RecordingDefinition])
def test_a_run_enters_a_scope_of_its_dataset_that_loads_nothing(
    feature_type: type[_RecordingFeature],
    two_datasets: tuple[Dataset, Dataset],
) -> None:
    """A run receives its feature built and leaves every definition unloaded."""
    own, _ = two_datasets
    feature = feature_type()

    _ = run_feature(own, feature, track=False)

    (seen,) = feature.seen
    assert seen is not None
    assert (seen.base_dir, seen.load, seen.allowed) == (own.base_dir, False, True)
    assert current_scope() is None


# --- a state that pickles an object of the definition's own module -----------------

_POINT: Final = pd.DataFrame({"X": [3.0], "Y": [4.0]})

_FRESH_PROGRAM: Final = """
import sys
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import open_dataset
from mosaic.core.pipeline.definitions.define import load_feature_definition

manifest, slug, run_root = sys.argv[1:4]
ds = open_dataset(manifest, ensure_roots=False)
feature = load_feature_definition(ds, slug)()
assert feature.load_state(Path(run_root), {}, {})
out = feature.apply(pd.DataFrame({"X": [3.0], "Y": [4.0]}))
mean = type(feature.mean)
print(mean.__module__, mean.__name__, out["centered"].iloc[0])
"""


@pytest.mark.slow
def test_a_fresh_process_unpickles_state_of_the_definitions_own_class(
    tmp_path: Path,
) -> None:
    """``load_state`` finds the class a fit pickled, by the module's stable name."""
    ds = defined_dataset(tmp_path, {GLOBAL_NAME: GLOBAL_SOURCE})
    digest = hashlib.sha256(GLOBAL_SOURCE.encode()).hexdigest()
    slug = f"def:{GLOBAL_NAME}"
    feature = load_feature_definition(ds, slug)()

    result = run_feature(ds, feature, track=False)

    root = result_run_root(ds, result)
    module_name = f"mosaic_definitions.{GLOBAL_NAME.replace('-', '_')}__{digest}"
    assert module_name.encode() in (root / "mean.pkl").read_bytes()
    fitted = float(feature.apply(_POINT)["centered"].iloc[0])

    reported = run_fresh(
        _FRESH_PROGRAM, args=[str(ds.manifest_path), slug, str(root)]
    ).split()

    assert reported[:2] == [module_name, "Mean"]
    assert float(reported[2]) == fitted


# --- a predicted identity of a fitted definition given its model --------------


def test_an_applying_instance_is_predicted_without_its_scope(tmp_path: Path) -> None:
    """A class-built instance with a model hashes no entries, as its run does."""
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    model = {"feature": APPLYING_STORAGE, "run_id": "1-abc"}
    cls = load_feature_definition(ds, f"def:{APPLYING_NAME}")
    feature = built_by_class(cls, {"model": model})
    assert getattr(feature, "scope_dependent") is True

    run_ids = {
        resolve_feature_identity(ds, feature, scope=Scope(entries=entries))[0]
        for entries in ([("g", "s1")], [("g", "s1"), ("g", "s2")], [("g", "s2")])
    }

    assert len(run_ids) == 1
    assert getattr(feature, "scope_dependent") is False


# --- an apply of a fit, read from the fit's run root ----------------------------

_FITTED_ENTRY: Final = ("g", "s1")


def _distances(group: str, sequence: str, n_ids: int = 1) -> pd.Series:
    """The distance from the origin of each row of an entry's track table."""
    table = track_table(group, sequence, n_ids=n_ids)
    return pd.Series(np.hypot(table["X"], table["Y"]))


def _three_entry_dataset(tmp_path: Path, name: str, source: str) -> Dataset:
    """A dataset defining *source* whose third entry holds two individuals.

    The second individual is offset from the origin. A mean over every entry
    therefore differs from the mean over ``g/s1``.
    """
    ds = defined_dataset(tmp_path, {name: source})
    add_track_sequences(ds, ("g", "s3"), n_ids=2)
    return ds


def _fit(ds: Dataset, name: str) -> Result[str]:
    """A fit run of the definition *name* over ``g/s1`` alone."""
    return fit_result(ds, f"def:{name}", _FITTED_ENTRY)


def _canary_lines(canary: Path) -> list[dict[str, int]]:
    """The records the canary file holds, one per fit or apply."""
    if not canary.exists():
        return []
    return [json.loads(line) for line in canary.read_text().splitlines()]


def _fits_recorded(canary: Path) -> int:
    """How many fits the canary file records."""
    return sum("pid" in line for line in _canary_lines(canary))


def _applying_pids(canary: Path) -> set[int]:
    """The processes the canary file records an ``apply`` in."""
    return {line["applied"] for line in _canary_lines(canary) if "applied" in line}


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_an_apply_reads_the_state_of_the_fit_its_model_names(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every entry is centered on the mean over the fit's one entry.

    A process worker loads the state from the fit's root that it is handed.
    """
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    name = PARALLEL_APPLYING_NAME
    ds = _three_entry_dataset(tmp_path, name, PARALLEL_APPLYING_SOURCE)
    fit = _fit(ds, name)
    with definition_scope(ds, load=True):
        feature = build_feature(f"def:{name}", None, {"model": model_reference(fit)})

    applied = run_feature(
        ds, feature, parallel_workers=WORKERS, parallel_mode=mode, track=False
    )

    fitted_mean = float(_distances(*_FITTED_ENTRY).mean())
    every_mean = float(
        pd.concat(
            [_distances("g", "s1"), _distances("g", "s2"), _distances("g", "s3", 2)]
        ).mean()
    )
    assert not np.isclose(fitted_mean, every_mean)
    root = result_run_root(ds, applied)
    for group, sequence, n_ids in (("g", "s1", 1), ("g", "s2", 1), ("g", "s3", 2)):
        written = pd.read_parquet(root / f"{group}__{sequence}.parquet")
        np.testing.assert_allclose(
            written["centered"].to_numpy(),
            (_distances(group, sequence, n_ids) - fitted_mean).to_numpy(),
        )
    assert not (root / "mean.pkl").exists()
    assert not (root / "fit_scope.json").exists()
    assert _fits_recorded(canary) == 1
    appliers = _applying_pids(canary)
    if mode == "process":
        assert appliers - {os.getpid()}
    else:
        assert appliers == {os.getpid()}
    document = json.loads((root / "params.json").read_text())
    assert {
        "where": "params.model",
        "feature": fit.feature,
        "run_id": fit.run_id,
    } in document["_resolved"]


def test_a_definitions_fit_records_the_digest_of_its_saved_state(
    tmp_path: Path,
) -> None:
    """The digest covers the one file the definition's ``save_state`` writes."""
    name = PARALLEL_APPLYING_NAME
    ds = _three_entry_dataset(tmp_path, name, PARALLEL_APPLYING_SOURCE)

    root = result_run_root(ds, _fit(ds, name))

    listing = list_run_root(root)
    saved = {path: stamp for path, stamp in listing.items() if path == "mean.pkl"}
    assert saved
    recorded = read_fit_state_digest(root)
    assert recorded
    assert recorded == saved_state_digest(root, {}, saved)


def test_an_instance_built_by_its_class_with_a_model_applies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run narrows the instance itself, as ``build_feature`` does."""
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    ds = _three_entry_dataset(tmp_path, APPLYING_NAME, APPLYING_SOURCE)
    fit = _fit(ds, APPLYING_NAME)
    slug = f"def:{APPLYING_NAME}"
    cls = load_feature_definition(ds, slug)

    by_class = run_feature(
        ds, built_by_class(cls, {"model": model_reference(fit)}), track=False
    )

    assert _fits_recorded(canary) == 1
    root = result_run_root(ds, by_class)
    assert not (root / "mean.pkl").exists()
    with definition_scope(ds, load=True):
        built = build_feature(slug, None, {"model": model_reference(fit)})
    assert run_feature(ds, built, track=False).run_id == by_class.run_id


def test_an_apply_whose_load_reads_no_state_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A load that ignores its model is refused before any fit."""
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    ds = _three_entry_dataset(tmp_path, MISREADING_NAME, MISREADING_SOURCE)
    fit = _fit(ds, MISREADING_NAME)
    assert fit.run_id is not None
    with definition_scope(ds, load=True):
        feature = build_feature(
            f"def:{MISREADING_NAME}", None, {"model": model_reference(fit)}
        )
    execution_id = new_execution_id()

    with pytest.raises(DefinitionModelRefused) as caught:
        _ = run_feature(ds, feature, execution_id=execution_id)

    assert caught.value.reason == "not_loaded"
    assert caught.value.error_detail() == {
        "name": MISREADING_NAME,
        "feature": fit.feature,
        "run_id": fit.run_id,
        "reason": "not_loaded",
    }
    assert str(caught.value) == (
        f"def:{MISREADING_NAME}'s load_state returned False for the model of run "
        f"{fit.run_id}. A step given a model applies it and fits nothing: load the "
        "state from artifact_paths['model']"
    )
    assert _fits_recorded(canary) == 1
    logged = read_run(run_log_dir(ds.base_dir), execution_id)
    assert logged is not None
    assert logged["status"] == "failed"


def _pins_nothing(_ds: Dataset, _feature: Feature) -> tuple[Resolution, ...]:
    """A reference pass that leaves every reference as written."""
    return ()


def _checks_nothing(_ds: Dataset, _feature: Feature, _storage: str) -> None:
    """A model check that lets every reference through."""


def test_a_definition_model_naming_no_run_is_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that skips the checks and pinning reaches the dependencies."""
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    model = {"feature": APPLYING_STORAGE}
    with definition_scope(ds, load=True):
        feature = build_feature(f"def:{APPLYING_NAME}", None, {"model": model})
    monkeypatch.setattr(run_module, "_refuse_unusable_model", _checks_nothing)
    monkeypatch.setattr(run_module, "resolve_references", _pins_nothing)

    with pytest.raises(ValueError, match="Field 'model' names no run"):
        _ = run_feature(ds, feature, track=False)


# --- an unpinned model, refused with its candidate fits ---------------------------


def _unpinned_apply(ds: Dataset, slug: str) -> Feature:
    """An instance of *slug* whose ``model`` names the storage and no run."""
    with definition_scope(ds, load=True):
        return build_feature(slug, None, {"model": {"feature": APPLYING_STORAGE}})


def _refused_unpinned(ds: Dataset, feature: Feature) -> DefinitionModelMissing:
    """What ``run_feature`` raises for *feature*, after checking it wrote nothing."""
    features_root = ds.get_root("features")
    before = tree_snapshot(features_root)

    with pytest.raises(DefinitionModelMissing) as caught:
        _ = run_feature(ds, feature, track=False)

    assert tree_snapshot(features_root) == before
    assert caught.value.reason == "unpinned"
    return caught.value


def _unpinned_text(candidates: str) -> str:
    """The text of an unpinned ``model`` of the applying definition."""
    return (
        f"def:{APPLYING_NAME}'s 'model' names no run of {APPLYING_STORAGE}. "
        "An apply reads one fit, named by its run id: give run_id, one of "
        f"{candidates}, or reference the fit step as "
        '{"step": ...}'
    )


def test_an_unpinned_model_names_the_fits_of_its_version(tmp_path: Path) -> None:
    """Sibling fits are listed, and another version's fit and an apply are not."""
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    first = load_feature_definition(ds, f"def:{APPLYING_NAME}")
    reference = definition_ref_of(first)
    assert reference is not None
    slug = f"def:{APPLYING_NAME}@{reference.digest}"
    second = define_feature(ds, APPLYING_NAME, APPLYING_SECOND_SOURCE)
    fits = sorted(fit_over(ds, slug, entry) for entry in (("g", "s1"), ("g", "s2")))
    _ = fit_over(ds, f"def:{APPLYING_NAME}@{second.digest}", ("g", "s1"))
    expected = _unpinned_text(", ".join(fits))

    before_apply = _refused_unpinned(ds, _unpinned_apply(ds, slug))
    with definition_scope(ds, load=True):
        pinned = build_feature(
            slug, None, {"model": {"feature": APPLYING_STORAGE, "run_id": fits[0]}}
        )
    _ = run_feature(ds, pinned, track=False)
    after_apply = _refused_unpinned(ds, _unpinned_apply(ds, slug))

    for refused in (before_apply, after_apply):
        assert str(refused) == expected
        assert refused.error_detail() == {
            "name": APPLYING_NAME,
            "feature": APPLYING_STORAGE,
            "run_id": None,
            "reason": "unpinned",
            "candidates": fits,
        }


def test_an_unpinned_model_without_a_fit_says_to_run_one(tmp_path: Path) -> None:
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})

    refused = _refused_unpinned(ds, _unpinned_apply(ds, f"def:{APPLYING_NAME}"))

    assert str(refused) == (
        f"def:{APPLYING_NAME}'s 'model' names no run of {APPLYING_STORAGE}. "
        "An apply reads one fit, named by its run id, and no fit of this version "
        "exists: run the fit first"
    )
    assert refused.candidates == ()


# --- a pinned model that names no usable fit --------------------------------------


def _refused_apply[E: Exception](
    ds: Dataset, feature: Feature, canary: Path, error: type[E]
) -> E:
    """What ``run_feature`` raises for *feature*, after checking nothing ran.

    The apply fits nothing and leaves the features root as it was: no run root
    and no row in the storage's index.
    """
    features_root = ds.get_root("features")
    before = tree_snapshot(features_root)
    index_before = feature_index_path(ds, APPLYING_STORAGE).read_text()
    fits_before = _fits_recorded(canary)

    with pytest.raises(error) as caught:
        _ = run_feature(ds, feature, track=False)

    assert tree_snapshot(features_root) == before
    assert feature_index_path(ds, APPLYING_STORAGE).read_text() == index_before
    assert _fits_recorded(canary) == fits_before
    return caught.value


def _pinned_apply(ds: Dataset, slug: str, feature: str, run_id: str) -> Feature:
    """An instance of *slug* whose ``model`` names the run *run_id* of *feature*."""
    with definition_scope(ds, load=True):
        return build_feature(
            slug, None, {"model": {"feature": feature, "run_id": run_id}}
        )


def _detail(
    run_id: str, reason: str, feature: str = APPLYING_STORAGE
) -> dict[str, object]:
    """The ``error_detail`` of a refused model of the applying definition."""
    return {
        "name": APPLYING_NAME,
        "feature": feature,
        "run_id": run_id,
        "reason": reason,
    }


@pytest.fixture
def canary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The file the applying definition records each fit in."""
    path = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(path))
    return path


@pytest.fixture
def fitted(tmp_path: Path, canary: Path) -> tuple[Dataset, str]:
    """A dataset defining the applying definition, and a fit's run id."""
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    run_id = fit_over(ds, f"def:{APPLYING_NAME}", _FITTED_ENTRY)
    assert _fits_recorded(canary) == 1
    return ds, run_id


def test_a_model_naming_no_run_root_is_missing(
    fitted: tuple[Dataset, str], canary: Path
) -> None:
    ds, fit_run_id = fitted
    absent = f"{fit_run_id[:-4]}0000"
    feature = _pinned_apply(ds, f"def:{APPLYING_NAME}", APPLYING_STORAGE, absent)

    refused = _refused_apply(ds, feature, canary, DefinitionModelMissing)

    assert refused.reason == "missing"
    assert refused.error_detail() == _detail(absent, "missing")
    assert str(refused) == (
        f"def:{APPLYING_NAME}'s 'model' names run {absent} of {APPLYING_STORAGE}, "
        "which does not exist. A step given a model applies it and fits nothing, so "
        "run the fit first"
    )


def test_a_model_naming_a_fit_without_its_fit_scope_is_unfitted(
    fitted: tuple[Dataset, str], canary: Path
) -> None:
    ds, fit_run_id = fitted
    (feature_run_root(ds, APPLYING_STORAGE, fit_run_id) / "fit_scope.json").unlink()
    feature = _pinned_apply(ds, f"def:{APPLYING_NAME}", APPLYING_STORAGE, fit_run_id)

    refused = _refused_apply(ds, feature, canary, DefinitionModelMissing)

    assert refused.reason == "unfitted"
    assert refused.error_detail() == _detail(fit_run_id, "unfitted")
    assert str(refused) == (
        f"run {fit_run_id} of {APPLYING_STORAGE} has not saved its fit. Let the fit "
        "finish, or run it again"
    )


def test_a_model_naming_an_apply_run_is_refused(
    fitted: tuple[Dataset, str], canary: Path
) -> None:
    ds, fit_run_id = fitted
    slug = f"def:{APPLYING_NAME}"
    applied = run_feature(
        ds, _pinned_apply(ds, slug, APPLYING_STORAGE, fit_run_id), track=False
    )
    assert applied.run_id is not None
    feature = _pinned_apply(ds, slug, APPLYING_STORAGE, applied.run_id)

    refused = _refused_apply(ds, feature, canary, DefinitionModelRefused)

    assert refused.reason == "is_apply"
    assert refused.error_detail() == _detail(applied.run_id, "is_apply")
    assert str(refused) == (
        f"def:{APPLYING_NAME}'s 'model' names run {applied.run_id}, which applies "
        "a model itself and holds no fit. Name the step that fits"
    )


@pytest.mark.parametrize(
    "named",
    [f"def.{GLOBAL_NAME}__from__tracks", f"def.{APPLYING_NAME}__from__nn"],
    ids=["another-definition", "other-inputs"],
)
def test_a_model_naming_another_storage_is_refused(
    named: str, fitted: tuple[Dataset, str], canary: Path
) -> None:
    ds, fit_run_id = fitted
    feature = _pinned_apply(ds, f"def:{APPLYING_NAME}", named, fit_run_id)

    refused = _refused_apply(ds, feature, canary, DefinitionModelRefused)

    assert refused.reason == "other_feature"
    assert refused.error_detail() == _detail(fit_run_id, "other_feature", named)
    assert str(refused) == (
        f"def:{APPLYING_NAME}'s 'model' names {named}, and this step stores under "
        f"{APPLYING_STORAGE}. A model is applied by the definition that fitted it, "
        f"over the same inputs: name a step of def:{APPLYING_NAME} that reads them"
    )


def test_a_model_naming_a_run_recorded_for_no_definition_is_refused(
    fitted: tuple[Dataset, str], canary: Path
) -> None:
    """A run of the storage whose ``params.json`` names no definition fits none."""
    ds, fit_run_id = fitted
    run_root = feature_run_root(ds, APPLYING_STORAGE, fit_run_id)
    document: dict[str, object] = json.loads((run_root / "params.json").read_text())
    del document["_definition"]
    write_run_params(run_root, document)
    feature = _pinned_apply(ds, f"def:{APPLYING_NAME}", APPLYING_STORAGE, fit_run_id)

    refused = _refused_apply(ds, feature, canary, DefinitionModelRefused)

    assert refused.reason == "other_feature"
    assert refused.error_detail() == _detail(fit_run_id, "other_feature")


def test_a_model_naming_a_fit_of_another_version_is_refused(
    tmp_path: Path, canary: Path
) -> None:
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    first = definition_ref_of(load_feature_definition(ds, f"def:{APPLYING_NAME}"))
    assert first is not None
    fit_run_id = fit_over(ds, f"def:{APPLYING_NAME}@{first.digest}", _FITTED_ENTRY)
    second = define_feature(ds, APPLYING_NAME, APPLYING_SECOND_SOURCE)
    feature = _pinned_apply(
        ds, f"def:{APPLYING_NAME}@{second.digest}", APPLYING_STORAGE, fit_run_id
    )

    refused = _refused_apply(ds, feature, canary, DefinitionModelRefused)

    assert refused.reason == "other_version"
    assert refused.error_detail() == _detail(fit_run_id, "other_version")
    fit_digest = first.digest[:12]
    assert str(refused) == (
        f"def:{APPLYING_NAME}'s 'model' names run {fit_run_id}, fitted by version "
        f"{fit_digest} of def:{APPLYING_NAME}, and this step runs version "
        f"{second.digest[:12]}. An apply reads a fit of its own version: fit again "
        f"with this version, or run this step as def:{APPLYING_NAME}@{fit_digest}"
    )


# --- an apply after its fit is written again ---------------------------------------


def _apply_refusals(
    ds: Dataset,
    name: str,
    fit: Result[str],
    capsys: pytest.CaptureFixture[str],
    *,
    mode: str = "thread",
) -> tuple[Result[str], dict[str, str]]:
    """Apply *fit* to every entry with the definition *name*.

    Returns:
        The apply's result, and the reason printed for each entry not served.
    """
    with definition_scope(ds, load=True):
        feature = build_feature(f"def:{name}", None, {"model": model_reference(fit)})
    _ = capsys.readouterr()
    applied = run_feature(
        ds, feature, parallel_workers=WORKERS, parallel_mode=mode, track=False
    )
    return applied, refusals(capsys.readouterr().err, f"def:{name}")


def _fit_again(ds: Dataset, name: str, fit: Result[str]) -> None:
    """Remove *fit*'s run root and fit again under its run id."""
    shutil.rmtree(result_run_root(ds, fit))
    assert _fit(ds, name).run_id == fit.run_id


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_an_apply_recomputes_after_its_fit_saves_other_state(
    mode: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers in either mode apply the state, and the rows in the parent record it."""
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))
    name = PARALLEL_APPLYING_NAME
    ds = _three_entry_dataset(tmp_path, name, PARALLEL_APPLYING_SOURCE)
    fit = _fit(ds, name)
    applied, refusals = _apply_refusals(ds, name, fit, capsys, mode=mode)
    assert refusals == {}
    before = read_fit_state_digest(result_run_root(ds, fit))
    assert set(model_cells(ds, applied).values()) == {before}
    moved = track_table(*_FITTED_ENTRY)
    moved[["X", "Y"]] = moved[["X", "Y"]] * 2.0
    moved.to_parquet(ds.get_root("tracks") / "g__s1.parquet")
    _fit_again(ds, name, fit)
    after = read_fit_state_digest(result_run_root(ds, fit))
    assert after != before
    canary.unlink()

    applied, refusals = _apply_refusals(ds, name, fit, capsys, mode=mode)

    assert refusals == {"s1": FITTED_AGAIN, "s2": FITTED_AGAIN, "s3": FITTED_AGAIN}
    assert set(model_cells(ds, applied).values()) == {after}
    appliers = _applying_pids(canary)
    if mode == "process":
        assert appliers - {os.getpid()}
    else:
        assert appliers == {os.getpid()}
    assert _apply_refusals(ds, name, fit, capsys, mode=mode)[1] == {}


def test_an_apply_is_served_after_its_fit_saves_the_same_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _three_entry_dataset(tmp_path, APPLYING_NAME, APPLYING_SOURCE)
    fit = _fit(ds, APPLYING_NAME)
    assert _apply_refusals(ds, APPLYING_NAME, fit, capsys)[1] == {}
    digest = read_fit_state_digest(result_run_root(ds, fit))

    _fit_again(ds, APPLYING_NAME, fit)

    assert read_fit_state_digest(result_run_root(ds, fit)) == digest
    assert _apply_refusals(ds, APPLYING_NAME, fit, capsys)[1] == {}
