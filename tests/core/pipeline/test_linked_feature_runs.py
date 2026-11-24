"""The feature lookups read a dataset's own runs and those of the datasets it links.

Each test builds a source holding one scope-free feature run over two entries
and a copy that links it, so the copy holds no run of its own until a test
writes one.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import Final, NamedTuple

import joblib
import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.visualization_library.data_loading import load_tracks_and_labels
from mosaic.core.analysis import load_feature_sequence_index, list_feature_runs
from mosaic.core.dataset import Dataset, LinkUnreachableError, open_dataset
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.store import (
    DefinitionModelMissing,
    DefinitionModelRefused,
)
from mosaic.core.evaluation import EvaluationReport, tracks_variant_read
from mosaic.core.pipeline.fit_scope import fit_and_apply_scopes
from mosaic.core.pipeline.graph import (
    Recipe,
    StepRefused,
    plan_pipeline,
    run_pipeline,
)
from mosaic.core.pipeline.graph.pins import refuse_unusable_pins
from mosaic.core.pipeline.graph.score import run_group_scores
from mosaic.core.pipeline.artifact_kinds import LINKED_KINDS, LinkedKind
from mosaic.core.pipeline.fit_scope import (
    FitScope,
    FitStateUnreadable,
    LinkedFitDrifted,
    read_fit_scope,
    read_fit_state_digest,
    read_run_fit_scope,
    write_fit_scope,
)
from mosaic.core.pipeline.index import (
    DriftReads,
    FeatureIndexRow,
    feature_index,
    feature_index_path,
    feature_rows,
    feature_run_file,
    feature_run_root,
    feature_run_roots,
    latest_feature_run_root,
    readable_feature_storages,
    recorded_consumption,
    recorded_tracks_reads,
)
from mosaic.core.pipeline.inventory.model import FeatureRunRef
from mosaic.core.pipeline.inventory.scan import inventory as scan_inventory
from mosaic.core.pipeline.inventory.params import (
    TracksVariantWalk,
    linked_run_params,
    read_linked_run_report,
    run_tracks_variants,
    upstream_runs,
)
from mosaic.core.pipeline.loading import (
    build_nn_lookup,
    get_feature_run_root,
    load_joblib_artifact,
)
from mosaic.core.pipeline.manifest import unrun_feature_input, upstream_tracks_variants
from mosaic.core.pipeline.preview import preview_artifact
from mosaic.core.pipeline.provenance import reached_by
from mosaic.core.pipeline.resolve import resolve_references
from mosaic.core.pipeline.identity_scheme import MARKER_NAME
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.inventory.scan import run_coverage
from mosaic.core.pipeline.run import producer_version, run_feature
from mosaic.core.pipeline.types import DependencyLookup, Feature, InputStream
from mosaic.core.scope import Scope
from mosaic.core.pipeline.run_matrix import load_run_matrix
from mosaic.core.pipeline.track_space import upstream_track_spaces
from mosaic.core.pipeline.track_universe import (
    consumed_track_runs,
    current_run_id,
    track_universe,
)
from mosaic.core.pipeline.types import ArtifactSpec, JoblibLoadSpec
from mosaic.core.pipeline.verify import verify_plan
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.types.results import NNResult, Result, ResultColumn
from mosaic.runlog import RunLogSnapshot, read_run, run_log_path

from tests.helpers import (
    fit_result,
    APPLYING_NAME,
    APPLYING_STORAGE,
    APPLYING_SOURCE,
    CONVERTED_ENTRIES,
    FITTED_AGAIN,
    KMeansFit,
    SOURCE_ENTRIES,
    CenteringFeature,
    ParallelFeature,
    add_tracks_table,
    add_tracks_variant,
    add_feature_run,
    applying_instance,
    assert_tree_unchanged,
    calms21_dataset,
    converted_source,
    copy_of,
    defined_dataset,
    definitions_forgotten,
    distances_from_origin,
    drifted,
    feature_edge,
    forget_imported_definitions,
    imported_definitions,
    kmeans_apply_feature,
    kmeans_fit,
    link_forms,
    make_dataset,
    model_cells,
    move_tracks,
    plan_refused,
    refit_kmeans,
    refusals,
    result_run_root,
    run_step,
    scored_classifier_recipe,
    take_link_offline,
    track_table,
    tracks_edge,
    tree_snapshot,
    write_run_params,
)

STORAGE = "demo"
RUN = "0.1-0123456789"
LATER_RUN = "0.1-9876543210"


@pytest.fixture
def source(tmp_path: Path) -> Dataset:
    dataset = make_dataset(tmp_path / "source", name="source")
    _ = add_feature_run(dataset, STORAGE, RUN)
    return dataset


@pytest.fixture
def copy(source: Dataset, tmp_path: Path) -> Dataset:
    return copy_of(source, tmp_path / "copy")


@link_forms
def test_the_rows_of_a_linked_run_name_the_sources_files(
    source: Dataset, tmp_path: Path, relative: bool
) -> None:
    copy = copy_of(source, tmp_path / "copy", relative=relative)

    rows = feature_rows(copy, STORAGE)

    assert rows["sequence"].tolist() == ["s1", "s2"]
    assert rows["link_id"].tolist() == [f"source-{source.uuid}"] * 2
    expected = feature_run_root(source, STORAGE, RUN)
    assert [copy.resolve_path(cell) for cell in rows["abs_path"]] == [
        expected / "s1.parquet",
        expected / "s2.parquet",
    ]


def test_a_run_id_narrows_the_rows_to_that_run(source: Dataset, copy: Dataset) -> None:
    _ = add_feature_run(source, STORAGE, LATER_RUN, SOURCE_ENTRIES[:1])

    assert feature_rows(copy, STORAGE, run_id=LATER_RUN)["sequence"].tolist() == ["s1"]
    assert set(feature_rows(copy, STORAGE)["run_id"]) == {RUN, LATER_RUN}
    assert feature_rows(copy, STORAGE, run_id="absent").empty


def test_an_own_row_shadows_the_linked_row_of_its_entry(
    source: Dataset, copy: Dataset
) -> None:
    _ = add_feature_run(copy, STORAGE, RUN, SOURCE_ENTRIES[:1])

    rows = feature_rows(copy, STORAGE, run_id=RUN)

    assert dict(zip(rows["sequence"], rows["link_id"], strict=True)) == {
        "s1": "",
        "s2": f"source-{source.uuid}",
    }


def test_a_dataset_without_links_reads_its_own_rows_with_an_empty_link_id(
    source: Dataset,
) -> None:
    rows = feature_rows(source, STORAGE)

    assert rows["link_id"].tolist() == ["", ""]


def test_a_storage_nobody_holds_has_no_rows(copy: Dataset) -> None:
    rows = feature_rows(copy, "absent")

    assert rows.empty
    assert "run_id" in rows.columns


def test_the_run_roots_follow_the_lookup_order(source: Dataset, copy: Dataset) -> None:
    linked_root = feature_run_root(source, STORAGE, RUN)
    assert feature_run_roots(copy, STORAGE, RUN) == (linked_root,)

    own_root = add_feature_run(copy, STORAGE, RUN, SOURCE_ENTRIES[:1])

    assert feature_run_roots(copy, STORAGE, RUN) == (own_root, linked_root)
    assert feature_run_roots(copy, STORAGE, "absent") == ()


def test_a_run_file_comes_from_the_first_root_that_holds_it(
    source: Dataset, copy: Dataset
) -> None:
    own_root = add_feature_run(copy, STORAGE, RUN, SOURCE_ENTRIES[:1])
    linked_root = feature_run_root(source, STORAGE, RUN)
    _ = (linked_root / "report.json").write_text("{}", encoding="utf-8")

    assert feature_run_file(copy, STORAGE, RUN, "params.json") == (
        own_root / "params.json"
    )
    assert feature_run_file(copy, STORAGE, RUN, "report.json") == (
        linked_root / "report.json"
    )
    assert feature_run_file(copy, STORAGE, RUN, "absent.json") is None
    assert feature_run_file(copy, STORAGE, "absent", "params.json") is None


def test_a_storage_both_datasets_hold_is_listed_once(
    source: Dataset, copy: Dataset
) -> None:
    assert readable_feature_storages(copy) == [STORAGE]

    _ = add_feature_run(copy, STORAGE, LATER_RUN)
    _ = add_feature_run(copy, "other", RUN)

    assert readable_feature_storages(copy) == [STORAGE, "other"]
    assert readable_feature_storages(source) == [STORAGE]


def test_the_latest_run_is_the_first_links_until_the_copy_has_its_own(
    source: Dataset, copy: Dataset
) -> None:
    assert latest_feature_run_root(copy, STORAGE) == (
        RUN,
        feature_run_root(source, STORAGE, RUN),
    )

    own_root = add_feature_run(copy, STORAGE, "0.1-aaaaaaaaaa", SOURCE_ENTRIES[:1])
    _ = add_feature_run(source, STORAGE, LATER_RUN)

    assert latest_feature_run_root(copy, STORAGE) == ("0.1-aaaaaaaaaa", own_root)


def test_a_dataset_that_records_no_run_still_raises_as_before(copy: Dataset) -> None:
    with pytest.raises(FileNotFoundError):
        latest_feature_run_root(copy, "absent")


def test_the_run_listing_reads_the_sources_rows(source: Dataset, copy: Dataset) -> None:
    runs = list_feature_runs(copy, STORAGE)

    assert runs["run_id"].tolist() == [RUN]
    assert runs["n_entries"].tolist() == [2]


def test_the_listing_of_a_storage_nobody_holds_is_refused(copy: Dataset) -> None:
    with pytest.raises(FileNotFoundError, match="Feature index missing"):
        list_feature_runs(copy, "absent")


def test_a_storage_directory_without_an_index_is_refused_without_links(
    source: Dataset,
) -> None:
    (feature_run_root(source, "unindexed", RUN)).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Feature index missing"):
        list_feature_runs(source, "unindexed")
    with pytest.raises(FileNotFoundError, match="Feature index missing"):
        load_feature_sequence_index(source, "unindexed", None)


def test_an_unfinished_run_lists_no_finish_time(source: Dataset) -> None:
    index = feature_index(feature_index_path(source, STORAGE))
    index.append(
        [
            FeatureIndexRow(
                run_id=LATER_RUN,
                feature=STORAGE,
                version="0.1",
                group="g",
                sequence="s9",
                abs_path=Path("s9.parquet"),
                params_hash="hash",
            )
        ]
    )

    runs = list_feature_runs(source, STORAGE)

    finished = dict(zip(runs["run_id"], runs["finished_at"].tolist(), strict=True))
    assert pd.isna(finished[LATER_RUN])
    assert not pd.isna(finished[RUN])


def test_the_sequence_index_reads_the_sources_rows(
    source: Dataset, copy: Dataset
) -> None:
    mapping, resolved, run_root = load_feature_sequence_index(copy, STORAGE, None)

    linked_root = feature_run_root(source, STORAGE, RUN)
    assert resolved == RUN
    assert run_root == linked_root
    assert {name: bundle.path for name, bundle in mapping.items()} == {
        "s1": linked_root / "s1.parquet",
        "s2": linked_root / "s2.parquet",
    }


UPSTREAM = "up"
DOWNSTREAM = "down"
UPSTREAM_RUN = "0.1-1111111111"
DOWNSTREAM_RUN = "0.1-2222222222"


def _source_chain(source: Dataset, variant: str) -> None:
    """Record a run of ``up`` that read *variant* and a run of ``down`` that read it."""
    up_root = add_feature_run(source, UPSTREAM, UPSTREAM_RUN)
    write_run_params(
        up_root, {"_inputs": ["tracks"], "_resolved": [tracks_edge(variant)]}
    )
    down_root = add_feature_run(source, DOWNSTREAM, DOWNSTREAM_RUN)
    write_run_params(
        down_root,
        {
            "_inputs": [{"feature": UPSTREAM, "run_id": UPSTREAM_RUN}],
            "_resolved": [feature_edge(0, UPSTREAM, UPSTREAM_RUN)],
        },
    )


def test_the_run_root_of_a_linked_run_is_the_sources(
    source: Dataset, copy: Dataset
) -> None:
    assert get_feature_run_root(copy, STORAGE, RUN) == (
        RUN,
        feature_run_root(source, STORAGE, RUN),
    )


@pytest.mark.parametrize("pinned", [True, False], ids=["pinned", "latest"])
def test_an_artifact_of_a_run_the_copy_extends_is_read_from_the_source(
    source: Dataset, copy: Dataset, pinned: bool
) -> None:
    joblib.dump({"value": 1}, feature_run_root(source, STORAGE, RUN) / "model.joblib")
    _ = add_feature_run(copy, STORAGE, RUN, SOURCE_ENTRIES[:1])
    artifact = ArtifactSpec(
        feature=STORAGE, run_id=RUN if pinned else None, load=JoblibLoadSpec()
    )

    assert load_joblib_artifact(copy, artifact) == {"value": 1}


def test_a_run_nobody_holds_has_the_own_run_root(copy: Dataset) -> None:
    assert get_feature_run_root(copy, STORAGE, "absent") == (
        "absent",
        feature_run_root(copy, STORAGE, "absent"),
    )


def test_the_producer_version_of_a_linked_definition_run_is_the_recorded_one(
    source: Dataset, copy: Dataset
) -> None:
    root = feature_run_root(source, STORAGE, RUN)
    write_run_params(root, {"_definition": {"name": "mine", "digest": "ab" * 32}})
    result = Result(feature=STORAGE, run_id=RUN)

    assert producer_version(copy, result, "def:mine") == f"def:mine@{'ab' * 32}"


def test_the_runs_a_linked_run_read_are_listed(source: Dataset, copy: Dataset) -> None:
    root = feature_run_root(source, STORAGE, RUN)
    write_run_params(root, {"_resolved": [tracks_edge("v1")]})

    assert consumed_track_runs(copy, STORAGE, RUN) == frozenset({"v1"})
    assert consumed_track_runs(copy, STORAGE, "absent") == frozenset()


@pytest.fixture
def converted(tmp_path: Path) -> Dataset:
    return converted_source(tmp_path / "converted")


def test_the_track_spaces_behind_a_linked_run_are_read(
    converted: Dataset, tmp_path: Path
) -> None:
    variant = str(read_tracks_index(converted)["run_id"].iloc[0])
    root = add_feature_run(converted, STORAGE, RUN, CONVERTED_ENTRIES)
    write_run_params(root, {"_inputs": ["tracks"], "_resolved": [tracks_edge(variant)]})
    for _group, sequence in CONVERTED_ENTRIES:
        pd.DataFrame({"X": [1.0], "Y": [2.0]}).to_parquet(root / f"{sequence}.parquet")
    copy = copy_of(converted, tmp_path / "copy")
    entries = frozenset(CONVERTED_ENTRIES)

    spaces = upstream_track_spaces(copy, STORAGE, RUN, entries, {})

    assert spaces
    assert spaces == upstream_track_spaces(converted, STORAGE, RUN, entries, {})


def test_a_walk_with_the_linked_reader_follows_a_run_through_the_source(
    source: Dataset, copy: Dataset
) -> None:
    _source_chain(source, "v1")

    assert TracksVariantWalk(copy)(DOWNSTREAM, DOWNSTREAM_RUN) is None
    walk = TracksVariantWalk(copy, read_params=linked_run_params(copy))
    assert walk(DOWNSTREAM, DOWNSTREAM_RUN) == frozenset({"v1"})
    assert run_tracks_variants(copy, DOWNSTREAM, DOWNSTREAM_RUN) is None
    assert run_tracks_variants(
        copy, DOWNSTREAM, DOWNSTREAM_RUN, read_params=linked_run_params(copy)
    ) == frozenset({"v1"})


def test_the_upstream_runs_are_read_through_the_reader(
    source: Dataset, copy: Dataset
) -> None:
    _source_chain(source, "v1")

    walked = upstream_runs(
        copy, [(DOWNSTREAM, DOWNSTREAM_RUN)], read_params=linked_run_params(copy)
    )

    assert [(run, params is not None) for run, params in walked] == [
        ((DOWNSTREAM, DOWNSTREAM_RUN), True),
        ((UPSTREAM, UPSTREAM_RUN), True),
    ]


def test_a_run_no_dataset_holds_has_no_params(source: Dataset, copy: Dataset) -> None:
    read = linked_run_params(copy)

    assert read((STORAGE, RUN)) is not None
    assert read((STORAGE, "absent")) is None
    assert read(("absent", RUN)) is None


def test_the_variants_a_linked_run_read_are_known_to_the_manifest(
    source: Dataset, copy: Dataset
) -> None:
    _source_chain(source, "v1")

    assert upstream_tracks_variants(copy, DOWNSTREAM, DOWNSTREAM_RUN) == frozenset(
        {"v1"}
    )


def test_the_tracks_variant_a_prediction_read_is_the_sources(
    source: Dataset, copy: Dataset
) -> None:
    _source_chain(source, "v1")
    prediction = ResultColumn(feature=DOWNSTREAM, run_id=DOWNSTREAM_RUN, column="c")

    assert tracks_variant_read(copy, prediction) == "v1"


def test_the_report_of_a_linked_run_is_read(source: Dataset, copy: Dataset) -> None:
    root = feature_run_root(source, STORAGE, RUN)
    _ = (root / "metrics.json").write_text('{"f1": 1}', encoding="utf-8")

    assert read_linked_run_report(copy, STORAGE, RUN, "metrics.json") == {"f1": 1}
    assert read_linked_run_report(copy, STORAGE, RUN, "absent.json") is None
    assert read_linked_run_report(copy, STORAGE, "absent", "metrics.json") is None


def test_the_fit_scope_of_a_linked_run_is_read(source: Dataset, copy: Dataset) -> None:
    root = feature_run_root(source, STORAGE, RUN)
    write_fit_scope(
        root,
        FitScope(
            scope_dependent=True,
            entries=(("g", "s1"),),
            tracks_variants=(),
            labels_variants=(),
            fitted_at="",
            identity_scheme="",
        ),
    )

    scope = read_run_fit_scope(copy, STORAGE, RUN)

    assert scope is not None
    assert scope.entries == (("g", "s1"),)
    assert read_run_fit_scope(copy, STORAGE, "absent") is None


TABLE = pd.DataFrame(
    {
        "frame": [0, 1],
        "id": [0, 0],
        "X": [1.0, 2.0],
        "Y": [3.0, 4.0],
        "nn_id": [1.0, 1.0],
        "label": [0, 1],
        "v": [0.5, 0.25],
    }
)
"""The per-entry table that :func:`_write_tables` gives each output."""


def _write_tables(
    root: Path, entries: Collection[tuple[str, str]] = SOURCE_ENTRIES
) -> None:
    """Write :data:`TABLE` as the output of each of *entries* under *root*."""
    for _group, sequence in entries:
        TABLE.to_parquet(root / f"{sequence}.parquet")


def _record_consumption(dataset: Dataset, storage: str, roots: str) -> None:
    """Give each row of *storage*'s index the consumption cells of a feature."""
    path = feature_index_path(dataset, storage)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows["consumed_roots"] = roots
    rows["consumed_composition"] = "recorded"
    rows["consumed_tracks_composition"] = "tracks"
    rows.to_csv(path, index=False)


def test_the_neighbor_lookup_reads_the_sources_table(
    source: Dataset, copy: Dataset
) -> None:
    _write_tables(feature_run_root(source, STORAGE, RUN))

    for run_id in (RUN, None):
        lookup = build_nn_lookup(
            copy, "g", "s1", NNResult(feature=STORAGE, run_id=run_id)
        )

        assert len(lookup) == 2


def test_the_tracks_and_labels_of_a_copy_read_the_sources_run(
    converted: Dataset, tmp_path: Path
) -> None:
    root = add_feature_run(converted, STORAGE, RUN, CONVERTED_ENTRIES)
    _write_tables(root, CONVERTED_ENTRIES)
    copy = copy_of(converted, tmp_path / "copy")

    for run_id in (RUN, None):
        tracks, labels = load_tracks_and_labels(copy, "g", "a", {STORAGE: run_id})

        assert not tracks.empty
        assert list(labels["raw"]) == [STORAGE]
        assert len(labels["raw"][STORAGE]) == len(TABLE)
    with pytest.raises(FileNotFoundError, match="Missing feature index"):
        load_tracks_and_labels(copy, "g", "a", {"absent": RUN})


def test_the_fit_and_apply_scopes_count_the_sources_rows(
    source: Dataset, copy: Dataset
) -> None:
    fit, applied = fit_and_apply_scopes(copy, STORAGE, RUN)

    assert fit is None
    assert applied == frozenset(SOURCE_ENTRIES)
    assert fit_and_apply_scopes(copy, "absent", RUN) == (None, frozenset())


def test_a_run_matrix_reads_the_sources_run(source: Dataset, copy: Dataset) -> None:
    _write_tables(feature_run_root(source, STORAGE, RUN))

    matrix = load_run_matrix(copy, Result(feature=STORAGE, run_id=RUN), columns="v")
    unpinned = load_run_matrix(copy, Result(feature=STORAGE, run_id=None), columns="v")

    assert matrix.values.shape == (4, 1)
    assert matrix.finished
    assert unpinned.values.shape == (4, 1)
    with pytest.raises(FileNotFoundError, match="no runs in this dataset"):
        load_run_matrix(copy, Result(feature="absent", run_id=RUN))


def test_a_table_preview_reads_the_sources_file(source: Dataset, copy: Dataset) -> None:
    _write_tables(feature_run_root(source, STORAGE, RUN))

    preview = preview_artifact(copy, FeatureRunRef(STORAGE, RUN), ("g", "s1"))

    assert preview.path == str(feature_run_root(source, STORAGE, RUN) / "s1.parquet")
    assert preview.n_rows == len(TABLE)


def test_the_track_universe_lists_the_sources_track_shaped_runs(
    source: Dataset, copy: Dataset
) -> None:
    _write_tables(feature_run_root(source, STORAGE, RUN))

    assert [(item.storage, item.run_id) for item in track_universe(copy)] == [
        (STORAGE, RUN)
    ]


def test_an_unpinned_run_is_the_leaf_of_the_sources_chain(
    source: Dataset, copy: Dataset
) -> None:
    later = add_feature_run(source, STORAGE, LATER_RUN)
    write_run_params(later, {"_resolved": [feature_edge(0, STORAGE, RUN)]})
    assert current_run_id(copy, STORAGE) == LATER_RUN

    write_run_params(
        feature_run_root(source, STORAGE, RUN),
        {"_resolved": [feature_edge(0, STORAGE, LATER_RUN)]},
    )
    write_run_params(later, {})

    assert current_run_id(copy, STORAGE) == RUN
    with pytest.raises(FileNotFoundError):
        current_run_id(copy, "absent")


def test_the_variants_of_an_unpinned_linked_run_are_known(
    source: Dataset, copy: Dataset
) -> None:
    _source_chain(source, "v1")

    assert upstream_tracks_variants(copy, DOWNSTREAM, None) == frozenset({"v1"})


def test_an_unrun_input_names_the_storages_of_the_sources(copy: Dataset) -> None:
    assert f"Stored runs of '{STORAGE}': {STORAGE}." in str(
        unrun_feature_input(copy, STORAGE)
    )


def test_the_recorded_cells_of_a_linked_run_are_read(
    source: Dataset, copy: Dataset
) -> None:
    _record_consumption(source, STORAGE, "media_raw")
    entries = set(SOURCE_ENTRIES)

    assert set(recorded_tracks_reads(copy, STORAGE, RUN)) == entries
    assert recorded_consumption(copy, STORAGE, RUN) == {
        entry: (("media_raw",), "recorded") for entry in entries
    }
    assert recorded_consumption(copy, STORAGE, "absent") == {}
    shared = DriftReads(copy)
    assert set(shared.tracks_reads(STORAGE, RUN)) == entries
    assert shared.tracks_reads(STORAGE, "absent") == {}


def test_a_merged_row_of_a_linked_run_keeps_its_model_state_cell(
    source: Dataset, copy: Dataset
) -> None:
    path = feature_index_path(source, STORAGE)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows["consumed_model_state"] = "states/" + rows["sequence"]
    rows.to_csv(path, index=False)

    merged = feature_rows(copy, STORAGE, run_id=RUN)

    assert dict(
        zip(merged["sequence"], merged["consumed_model_state"], strict=True)
    ) == {sequence: f"states/{sequence}" for _group, sequence in SOURCE_ENTRIES}
    assert all(Path(cell).is_absolute() for cell in merged["abs_path"])
    reads = recorded_tracks_reads(copy, STORAGE, RUN)
    assert {entry: read.model_state for entry, read in reads.items()} == {
        entry: f"states/{entry[1]}" for entry in SOURCE_ENTRIES
    }


def test_a_change_reaches_the_sources_feature_rows(
    source: Dataset, copy: Dataset
) -> None:
    _record_consumption(source, STORAGE, "media_raw")

    reached = reached_by(copy, [("g", "s1")], "media_raw")

    features = reached[reached["kind"] == "features"]
    assert features[["name", "run_id", "sequence"]].values.tolist() == [
        [STORAGE, RUN, "s1"]
    ]
    assert features["link_id"].tolist() == [f"source-{source.uuid}"]


class Study(NamedTuple):
    """A dataset a classifier recipe ran over, and a copy that links it."""

    source: Dataset
    copy: Dataset


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> Study:
    """The recipe's source and its copy, built once for the tests that only read."""
    base = tmp_path_factory.mktemp("study")
    files = {"train": {"a": {"s1": 90, "s2": 90, "s3": 90}}}
    ds = calms21_dataset(base / "source", files, seed=11, n_classes=2)
    _ = run_pipeline(ds, scored_classifier_recipe())
    source = open_dataset(ds.manifest_path, ensure_roots=False)
    copy = copy_of(source, base / "copy")
    _ = copy.convert_all_labels(source_format="calms21_npy")
    return Study(source, copy)


def test_a_pin_of_a_run_only_the_source_holds_is_not_missing(study: Study) -> None:
    recipe = scored_classifier_recipe()
    plan = plan_pipeline(study.source, recipe)
    inventory = scan_inventory(study.source, kinds=("feature",))
    run_id = str(plan.step("labeled").run_id)

    refuse_unusable_pins(
        study.copy, recipe, plan, {"labeled": run_id}, inventory=inventory
    )

    with pytest.raises(StepRefused) as refused:
        refuse_unusable_pins(
            study.copy, recipe, plan, {"labeled": "absent"}, inventory=inventory
        )
    assert refused.value.reason == "pin_run_missing"


def test_the_group_scores_of_a_linked_evaluation_are_read(study: Study) -> None:
    planned = plan_pipeline(study.source, scored_classifier_recipe()).step("score")
    run_id = str(planned.run_id)
    report = EvaluationReport.model_validate(
        read_linked_run_report(study.copy, planned.storage_name, run_id, "metrics.json")
    )

    scores = run_group_scores(study.copy, planned.storage_name, run_id, report)

    assert list(scores) == ["train"]


def test_a_copy_verifies_the_sources_fit_and_scores(study: Study) -> None:
    plan = plan_pipeline(study.source, scored_classifier_recipe())

    report = verify_plan(study.copy, plan)

    assert report.ok
    statuses = {verdict.step: verdict.status for verdict in report.steps}
    assert [statuses[step] for step in ("labeled", "clf", "score")] == ["ok"] * 3


VARIANT = "convert-trex.0.2-3333333333"
"""The tracks variant that the source of a feature run on a copy holds."""


def _tracked_source(base: Path) -> Dataset:
    """A saved dataset at *base* with tracks of ``s1`` and ``s2`` in :data:`VARIANT`."""
    dataset = make_dataset(base, name="source")
    add_tracks_variant(dataset, VARIANT, "s1", "s2", std_format="mosaic_v1")
    return dataset


def _attempt(dataset: Dataset, execution_id: str) -> RunLogSnapshot:
    """The run-log snapshot of the attempt *execution_id* on *dataset*."""
    path = run_log_path(dataset.base_dir, execution_id)
    snapshot = read_run(path.parent, execution_id)
    assert snapshot is not None
    return snapshot


def _link_rows(copy: Dataset, result: Result) -> dict[str, str]:
    """Each sequence of *result*'s merged rows, with the link its row comes from."""
    rows = feature_rows(copy, result.feature, run_id=str(result.run_id))
    return dict(zip(rows["sequence"], rows["link_id"], strict=True))


def test_a_run_on_a_copy_serves_the_sources_entries_without_an_own_row(
    tmp_path: Path,
) -> None:
    source = _tracked_source(tmp_path / "source")
    computed = run_feature(source, ParallelFeature())
    copy = copy_of(source, tmp_path / "copy")
    link = f"source-{source.uuid}"

    result = run_feature(copy, ParallelFeature(), execution_id="TESTCOPYSERVED")

    assert result.run_id == computed.run_id
    assert result.cache_hit
    assert result.entries_written == 2
    attempt = _attempt(copy, "TESTCOPYSERVED")
    assert attempt["entries_written"] == 2
    assert (attempt["progress_done"], attempt["progress_total"]) == (2, 2)
    own_root = feature_run_root(copy, result.feature, str(result.run_id))
    assert sorted(path.name for path in own_root.iterdir()) == [
        MARKER_NAME,
        "params.json",
    ]
    own_index = feature_index_path(copy, result.feature)
    assert not own_index.exists() or feature_index(own_index).read().empty
    assert _link_rows(copy, result) == {"s1": link, "s2": link}


def test_an_entry_the_copy_adds_is_its_own_row_beside_the_sources(
    tmp_path: Path,
) -> None:
    source = _tracked_source(tmp_path / "source")
    _ = run_feature(source, ParallelFeature())
    copy = copy_of(source, tmp_path / "copy")
    _ = run_feature(copy, ParallelFeature())
    add_tracks_variant(copy, VARIANT, "s3", std_format="mosaic_v1")

    result = run_feature(copy, ParallelFeature(), execution_id="TESTCOPYADDS")

    assert result.entries_written == 3
    attempt = _attempt(copy, "TESTCOPYADDS")
    assert attempt["entries_written"] == 3
    assert (attempt["progress_done"], attempt["progress_total"]) == (3, 3)
    link = f"source-{source.uuid}"
    assert _link_rows(copy, result) == {"s1": link, "s2": link, "s3": ""}
    own_root = feature_run_root(copy, result.feature, str(result.run_id))
    assert sorted(path.name for path in own_root.glob("*.parquet")) == ["s3.parquet"]
    target = frozenset({("", "s1"), ("", "s2"), ("", "s3")})
    coverage = run_coverage(copy, result.feature, str(result.run_id), target)
    assert coverage.is_satisfied
    own_rows = feature_index(feature_index_path(copy, result.feature)).read()
    assert own_rows["finished_at"].ne("").all()


@pytest.fixture
def fit_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The fits of :class:`CenteringFeature` made in this process, one item each."""
    fits: list[int] = []
    fit = CenteringFeature.fit

    def spied(self: CenteringFeature, inputs: InputStream) -> None:
        fits.append(1)
        fit(self, inputs)

    monkeypatch.setattr(CenteringFeature, "fit", spied)
    return fits


def _output(root: Path, sequence: str) -> pd.DataFrame:
    """The output table of *sequence* in the run root *root*."""
    return pd.read_parquet(root / f"{sequence}.parquet")


def test_a_run_on_a_copy_uses_the_sources_fit_and_serves_its_entries(
    tmp_path: Path, fit_spy: list[int]
) -> None:
    source = _tracked_source(tmp_path / "source")
    computed = run_feature(source, CenteringFeature())
    copy = copy_of(source, tmp_path / "copy")
    assert fit_spy == [1]

    result = run_feature(copy, CenteringFeature())

    assert result.run_id == computed.run_id
    assert fit_spy == [1]
    own_root = feature_run_root(copy, result.feature, str(result.run_id))
    assert not (own_root / CenteringFeature.state_file).exists()
    assert not (own_root / "fit_scope.json").exists()
    assert list(own_root.glob("*.parquet")) == []
    link = f"source-{source.uuid}"
    assert _link_rows(copy, result) == {"s1": link, "s2": link}


def test_an_entry_the_copy_converted_again_is_computed_into_its_own_tree(
    tmp_path: Path, fit_spy: list[int]
) -> None:
    source = _tracked_source(tmp_path / "source")
    computed = run_feature(source, CenteringFeature())
    copy = copy_of(source, tmp_path / "copy")
    table = track_table("", "s1", n_rows=40, n_ids=2)
    add_tracks_table(
        copy, VARIANT, table, consumed_composition="changed", std_format="mosaic_v1"
    )

    result = run_feature(copy, CenteringFeature())

    assert fit_spy == [1]
    assert _link_rows(copy, result) == {"s1": "", "s2": f"source-{source.uuid}"}
    run_id = str(result.run_id)
    own_root = feature_run_root(copy, result.feature, run_id)
    assert sorted(path.name for path in own_root.glob("*.parquet")) == ["s1.parquet"]
    source_root = feature_run_root(source, computed.feature, run_id)
    pd.testing.assert_series_equal(
        _output(own_root, "s1")["X"], _output(source_root, "s1")["X"]
    )


class _ScopedCentering(CenteringFeature):
    """A :class:`CenteringFeature` over its scope, recording its entries in its fit."""

    name = "prov-centering-scoped"
    scope_dependent = True


def test_a_drifted_fit_the_source_holds_is_refused_in_the_copy(
    tmp_path: Path, fit_spy: list[int]
) -> None:
    source = _tracked_source(tmp_path / "source")
    computed = run_feature(source, _ScopedCentering())
    copy = copy_of(source, tmp_path / "copy")
    table = track_table("", "s1", n_rows=40, n_ids=2)
    add_tracks_table(
        copy, VARIANT, table, consumed_composition="changed", std_format="mosaic_v1"
    )
    source_before = tree_snapshot(source.base_dir)
    copy_features = copy.get_root("features")
    copy_before = tree_snapshot(copy_features) if copy_features.exists() else None
    link = f"source-{source.uuid}"

    with pytest.raises(LinkedFitDrifted) as refused:
        _ = run_feature(copy, _ScopedCentering(), track=False)

    assert refused.value.reason == "linked_fit_drifted"
    assert refused.value.detail == {
        "feature": _ScopedCentering.name,
        "run_id": computed.run_id,
        "link_id": link,
        "moved": 1,
        "dropped": 0,
        "refitted": [],
    }
    assert f"was fitted in link {link}" in str(refused.value)
    assert "(1 moved (s1))" in str(refused.value)
    assert fit_spy == [1]
    assert_tree_unchanged(source.base_dir, source_before)
    if copy_before is None:
        assert not copy_features.exists()
    else:
        assert_tree_unchanged(copy_features, copy_before)


def test_a_refit_in_the_copy_fits_into_its_own_run_root_and_leaves_the_source(
    tmp_path: Path, fit_spy: list[int]
) -> None:
    source = _tracked_source(tmp_path / "source")
    computed = run_feature(source, _ScopedCentering())
    copy = copy_of(source, tmp_path / "copy")
    table = track_table("", "s1", n_rows=40, n_ids=2)
    add_tracks_table(
        copy, VARIANT, table, consumed_composition="changed", std_format="mosaic_v1"
    )
    source_before = tree_snapshot(source.base_dir)

    result = run_feature(copy, _ScopedCentering(), track=False, refit=True)

    assert result.run_id == computed.run_id
    assert fit_spy == [1, 1]
    own_root = feature_run_root(copy, result.feature, str(result.run_id))
    assert (own_root / CenteringFeature.state_file).is_file()
    record = read_fit_scope(own_root, strict=True)
    assert record is not None
    assert record.execution_id == result.execution_id
    assert _link_rows(copy, result) == {"s1": "", "s2": ""}
    assert_tree_unchanged(source.base_dir, source_before)


def test_an_unreadable_record_under_the_link_does_not_stop_a_refit_in_the_copy(
    tmp_path: Path, fit_spy: list[int]
) -> None:
    source = _tracked_source(tmp_path / "source")
    computed = run_feature(source, _ScopedCentering())
    run_id = str(computed.run_id)
    source_root = feature_run_root(source, computed.feature, run_id)
    _ = (source_root / "fit_scope.json").write_text("{not json", encoding="utf-8")
    copy = copy_of(source, tmp_path / "copy")

    with pytest.raises(FitStateUnreadable):
        _ = run_feature(copy, _ScopedCentering(), track=False)

    result = run_feature(copy, _ScopedCentering(), track=False, refit=True)

    assert result.run_id == computed.run_id
    assert fit_spy == [1, 1]
    record = read_fit_scope(feature_run_root(copy, result.feature, run_id), strict=True)
    assert record is not None
    assert record.execution_id == result.execution_id


def _copy_adding(source: Dataset, directory: Path, *sequences: str) -> Dataset:
    """A copy of *source* at *directory* whose own tracks add *sequences*."""
    copy = copy_of(source, directory)
    add_tracks_variant(copy, VARIANT, *sequences, std_format="mosaic_v1")
    return copy


def test_each_process_worker_of_a_copy_loads_the_sources_fit(tmp_path: Path) -> None:
    source = _tracked_source(tmp_path / "source")
    _ = run_feature(source, CenteringFeature())
    in_processes = _copy_adding(source, tmp_path / "processes", "s3", "s4")
    in_threads = _copy_adding(source, tmp_path / "threads", "s3", "s4")

    by_process = run_feature(
        in_processes,
        CenteringFeature(),
        parallel_workers=2,
        parallel_mode="process",
        execution_id="TESTCOPYPROCESSES",
        measure_worker_base=True,
    )
    by_thread = run_feature(
        in_threads, CenteringFeature(), parallel_workers=2, parallel_mode="thread"
    )

    assert by_process.failed_entries == ()
    assert by_process.entries_written == 4
    log = run_log_path(in_processes.base_dir, "TESTCOPYPROCESSES")
    resources = [
        record
        for record in map(json.loads, log.read_text().splitlines())
        if record["ev"] == "resources"
    ]
    assert resources[-1]["worker_base_rss_bytes"] > 0
    run_id = str(by_process.run_id)
    process_root = feature_run_root(in_processes, by_process.feature, run_id)
    thread_root = feature_run_root(in_threads, by_thread.feature, run_id)
    for sequence in ("s3", "s4"):
        pd.testing.assert_frame_equal(
            _output(process_root, sequence), _output(thread_root, sequence)
        )


def test_the_run_level_files_of_a_copy_are_the_sources_beside_its_params(
    tmp_path: Path,
) -> None:
    source = _tracked_source(tmp_path / "source")
    computed = run_feature(source, CenteringFeature())
    run_id = str(computed.run_id)
    source_root = feature_run_root(source, computed.feature, run_id)
    _ = (source_root / "report.json").write_text("{}", encoding="utf-8")
    copy = _copy_adding(source, tmp_path / "copy", "s3")

    result = run_feature(copy, CenteringFeature(), scope=Scope(entries=[("", "s3")]))

    assert str(result.run_id) == run_id
    storage = result.feature
    own_root = feature_run_root(copy, storage, run_id)
    for name in ("report.json", CenteringFeature.state_file, "fit_scope.json"):
        assert feature_run_file(copy, storage, run_id, name) == source_root / name
    assert feature_run_file(copy, storage, run_id, "params.json") == (
        own_root / "params.json"
    )
    params = read_run_params(own_root).params
    assert params is not None
    assert params.asked_scope() == {("", "s1"), ("", "s2"), ("", "s3")}


# --- an unpinned definition model on a copy --------------------------------------


def _definition_fit(dataset: Dataset) -> Result[str]:
    """A fit of the applying definition over ``g/s1`` in *dataset*."""
    return fit_result(dataset, f"def:{APPLYING_NAME}", ("g", "s1"))


def _definition_copy(tmp_path: Path) -> tuple[Dataset, Dataset, str]:
    """A source holding a fit of the applying definition, a copy defining it too.

    Returns:
        The source, the copy and the fit's run id.
    """
    source = defined_dataset(tmp_path / "source", {APPLYING_NAME: APPLYING_SOURCE})
    fit = _definition_fit(source)
    assert fit.run_id is not None
    copy = copy_of(source, tmp_path / "copy")
    _ = define_feature(copy, APPLYING_NAME, APPLYING_SOURCE)
    return source, copy, fit.run_id


def test_an_unpinned_model_on_a_copy_names_the_sources_fit_by_its_link(
    tmp_path: Path,
) -> None:
    with definitions_forgotten():
        source, copy, run_id = _definition_copy(tmp_path)

        with pytest.raises(DefinitionModelMissing) as caught:
            _ = resolve_references(
                copy, applying_instance(copy, {"feature": APPLYING_STORAGE})
            )

        assert caught.value.reason == "unpinned"
        assert f"one of {run_id} (link source-{source.uuid}), or" in str(caught.value)
        assert caught.value.error_detail()["candidates"] == [run_id]


def test_an_unpinned_model_with_the_source_unreachable_is_not_missing(
    tmp_path: Path,
) -> None:
    with definitions_forgotten():
        _source, copy, _run_id = _definition_copy(tmp_path)
        offline = take_link_offline(copy)

        with pytest.raises(LinkUnreachableError) as caught:
            _ = resolve_references(
                offline, applying_instance(offline, {"feature": APPLYING_STORAGE})
            )

        assert caught.value.reason == "link_unreachable"


# --- a pinned definition model on a copy -----------------------------------------


def _pinned_apply(dataset: Dataset, run_id: str) -> Feature:
    """An instance of the applying definition whose ``model`` names *run_id*."""
    return applying_instance(dataset, {"feature": APPLYING_STORAGE, "run_id": run_id})


def test_a_pinned_model_on_a_copy_applies_the_sources_fit(tmp_path: Path) -> None:
    with definitions_forgotten():
        source, copy, run_id = _definition_copy(tmp_path)
        before = tree_snapshot(source.get_root("features"))

        applied = run_feature(copy, _pinned_apply(copy, run_id), track=False)

        assert applied.run_id is not None
        fitted_mean = float(distances_from_origin("g", "s1").mean())
        root = feature_run_root(copy, APPLYING_STORAGE, applied.run_id)
        for _group, sequence in SOURCE_ENTRIES:
            written = pd.read_parquet(root / f"g__{sequence}.parquet")
            np.testing.assert_allclose(
                written["centered"].to_numpy(),
                (distances_from_origin("g", sequence) - fitted_mean).to_numpy(),
            )
        assert not (root / "fit_scope.json").exists()
        assert tree_snapshot(source.get_root("features")) == before


def test_a_pinned_model_with_the_source_unreachable(tmp_path: Path) -> None:
    """Without the source, a fit the copy lacks is unreachable, an own apply refused."""
    with definitions_forgotten():
        source, copy, run_id = _definition_copy(tmp_path)
        applied = run_feature(copy, _pinned_apply(copy, run_id), track=False)
        assert applied.run_id is not None
        source_features = source.get_root("features")
        before = tree_snapshot(source_features)
        offline = take_link_offline(copy)

        with pytest.raises(LinkUnreachableError) as unreachable:
            _ = run_feature(offline, _pinned_apply(offline, run_id), track=False)
        with pytest.raises(DefinitionModelRefused) as refused:
            _ = run_feature(
                offline, _pinned_apply(offline, applied.run_id), track=False
            )

        assert unreachable.value.reason == "link_unreachable"
        assert refused.value.reason == "is_apply"
        assert tree_snapshot(source_features) == before


def _literal_apply_recipe(run_id: str) -> Recipe:
    """A recipe of one apply step whose ``model`` names *run_id* literally."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "apply",
                    "type": "feature",
                    "feature": f"def:{APPLYING_NAME}",
                    "inputs": ["tracks"],
                    "params": {
                        "model": {"feature": APPLYING_STORAGE, "run_id": run_id}
                    },
                }
            ],
        }
    )


def test_a_copy_plans_a_literal_model_naming_the_sources_fit(tmp_path: Path) -> None:
    with definitions_forgotten():
        _source, copy, run_id = _definition_copy(tmp_path)
        forget_imported_definitions()

        plan = plan_pipeline(copy, _literal_apply_recipe(run_id))

        assert plan.problems == ()
        assert not imported_definitions()


@pytest.mark.parametrize("applied", [False, True], ids=["unrun", "applied"])
def test_a_plan_of_a_literal_model_with_the_source_unreachable(
    tmp_path: Path, applied: bool
) -> None:
    """One problem at ``link_unreachable``, whether or not the step is complete.

    An apply the copy has not run is refused for the link as an incomplete step
    is. One it has run is complete, and only the model's fit is unreachable.
    """
    with definitions_forgotten():
        _source, copy, run_id = _definition_copy(tmp_path)
        if applied:
            _ = run_feature(copy, _pinned_apply(copy, run_id), track=False)
        offline = take_link_offline(copy)

        plan = plan_refused(offline, _literal_apply_recipe(run_id))

        assert not imported_definitions()
        assert plan.step("apply").is_complete is applied
        (problem,) = plan.problems
        assert (problem.step, problem.where) == ("apply", "link_unreachable")


# --- a fit the source holds, applied on a copy through ``model`` ---

UNRECORDED_TRACKS: Final = "nothing records what it was built from"
"""The reason a run gives for an entry whose row records no tracks composition.

The tracks are added without a composition and moved with one. An apply row
written before the move records none, and its input records one after.
"""

_WITHOUT_FEATURES: Final[tuple[LinkedKind, ...]] = tuple(
    kind for kind in LINKED_KINDS if kind != "features"
)


class Applied(NamedTuple):
    """One run of an apply, with what its ``load_state`` was handed.

    Attributes:
        result: The apply's result.
        models: The ``model`` path of each ``load_state`` call that had one.
        refusals: The reason printed for each sequence not served.
    """

    result: Result[str]
    models: frozenset[Path]
    refusals: dict[str, str]


def _applied(
    dataset: Dataset,
    feature: Feature,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> Applied:
    """Run the apply *feature* over *dataset* in two threads."""
    models: set[Path] = set()
    load_state = feature.load_state

    def spied(
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        model = artifact_paths.get("model")
        if model is not None:
            models.add(model)
        return load_state(run_root, artifact_paths, dependency_lookups)

    monkeypatch.setattr(feature, "load_state", spied)
    _ = capsys.readouterr()
    result = run_feature(
        dataset, feature, parallel_workers=2, parallel_mode="thread", track=False
    )
    err = capsys.readouterr().err
    return Applied(result, frozenset(models), refusals(err, feature.name))


def _relinked(copy: Dataset, provides: tuple[LinkedKind, ...]) -> Dataset:
    """*copy* opened again with its one link declared to provide *provides*."""
    (link,) = copy.manifest.links
    _ = copy.remove_link(link.id, save=False)
    _ = copy.add_link(link.model_copy(update={"provides": provides}))
    return Dataset(manifest_path=copy.manifest_path).load()


def _move_definition_tracks(dataset: Dataset) -> None:
    """Write the track table of ``g/s1`` again with its positions doubled.

    The table is written without a composition. The applying definition
    therefore sees the change only through the state a fit over it saves.
    """
    moved = track_table("g", "s1")
    moved[["X", "Y"]] = moved[["X", "Y"]] * 2.0
    moved.to_parquet(dataset.get_root("tracks") / "g__s1.parquet")


def test_a_copy_applies_a_definition_fit_the_source_holds_and_follows_its_refit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with definitions_forgotten():
        source, copy, run_id = _definition_copy(tmp_path)
        fit_root = feature_run_root(source, APPLYING_STORAGE, run_id)
        fitted = read_fit_state_digest(fit_root)
        assert fitted
        before = tree_snapshot(source.base_dir)

        first = _applied(copy, _pinned_apply(copy, run_id), capsys, monkeypatch)

        assert first.models == {fit_root}
        assert model_cells(copy, first.result) == {"s1": fitted, "s2": fitted}
        assert drifted(copy, first.result) == ()
        assert tree_snapshot(source.base_dir) == before

        _move_definition_tracks(source)
        shutil.rmtree(fit_root)
        assert _definition_fit(source).run_id == run_id
        refitted = read_fit_state_digest(fit_root)
        assert refitted not in {"", fitted}
        before = tree_snapshot(source.base_dir)

        assert drifted(copy, first.result) == SOURCE_ENTRIES
        again = _applied(copy, _pinned_apply(copy, run_id), capsys, monkeypatch)

        assert again.result.run_id == first.result.run_id
        assert again.refusals == {"s1": FITTED_AGAIN, "s2": FITTED_AGAIN}
        assert model_cells(copy, again.result) == {"s1": refitted, "s2": refitted}
        assert drifted(copy, again.result) == ()
        assert tree_snapshot(source.base_dir) == before


def test_a_copy_serves_the_sources_definition_apply_until_its_own_fit_shadows_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A linked apply row is compared with the fit the copy resolves."""
    with definitions_forgotten():
        source, copy, run_id = _definition_copy(tmp_path)
        _ = run_feature(source, _pinned_apply(source, run_id), track=False)
        fitted = read_fit_state_digest(
            feature_run_root(source, APPLYING_STORAGE, run_id)
        )
        link = f"source-{source.uuid}"
        before = tree_snapshot(source.base_dir)

        served = _applied(copy, _pinned_apply(copy, run_id), capsys, monkeypatch)

        assert served.refusals == {}
        assert _link_rows(copy, served.result) == {"s1": link, "s2": link}
        assert list(result_run_root(copy, served.result).glob("*.parquet")) == []
        assert tree_snapshot(source.base_dir) == before

        _move_definition_tracks(source)
        before = tree_snapshot(source.base_dir)
        unlinked = _relinked(copy, _WITHOUT_FEATURES)
        assert _definition_fit(unlinked).run_id == run_id
        own_root = feature_run_root(copy, APPLYING_STORAGE, run_id)
        own = read_fit_state_digest(own_root)
        assert own not in {"", fitted}
        relinked = _relinked(unlinked, LINKED_KINDS)

        shadowed = _applied(
            relinked, _pinned_apply(relinked, run_id), capsys, monkeypatch
        )

        assert shadowed.models == {own_root}
        assert shadowed.refusals == {"s1": FITTED_AGAIN, "s2": FITTED_AGAIN}
        assert model_cells(relinked, shadowed.result) == {"s1": own, "s2": own}
        assert _link_rows(relinked, shadowed.result) == {"s1": "", "s2": ""}
        assert tree_snapshot(source.base_dir) == before


def _rerun_speed(dataset: Dataset, runs: KMeansFit) -> None:
    """Run ``speed-angvel`` again under its run id."""
    assert run_step(dataset, "speed-angvel", None, {}).run_id == runs.speed.run_id


def test_a_copy_applies_a_kmeans_fit_the_source_holds_and_follows_its_refit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tracked_source(tmp_path / "source")
    runs = kmeans_fit(source)
    copy = copy_of(source, tmp_path / "copy")
    fit_root = result_run_root(source, runs.fit)
    fitted = read_fit_state_digest(fit_root)
    assert fitted
    before = tree_snapshot(source.base_dir)

    first = _applied(copy, kmeans_apply_feature(runs), capsys, monkeypatch)

    assert {model.parent for model in first.models} == {fit_root}
    assert model_cells(copy, first.result) == {"s1": fitted, "s2": fitted}
    assert drifted(copy, first.result) == ()
    assert tree_snapshot(source.base_dir) == before

    move_tracks(source, VARIANT, ("s1", "s2"))
    _rerun_speed(source, runs)
    # The copy's apply first recomputes for its moved input, compared before
    # the model, and records the digest of the fit before its refit.
    moved = _applied(copy, kmeans_apply_feature(runs), capsys, monkeypatch)
    assert moved.refusals == {"s1": UNRECORDED_TRACKS, "s2": UNRECORDED_TRACKS}
    refitted = refit_kmeans(source, runs)
    assert refitted not in {"", fitted}
    before = tree_snapshot(source.base_dir)

    assert drifted(copy, first.result) == (("", "s1"), ("", "s2"))
    again = _applied(copy, kmeans_apply_feature(runs), capsys, monkeypatch)

    assert again.result.run_id == first.result.run_id
    assert again.refusals == {"s1": FITTED_AGAIN, "s2": FITTED_AGAIN}
    assert model_cells(copy, again.result) == {"s1": refitted, "s2": refitted}
    assert drifted(copy, again.result) == ()
    assert tree_snapshot(source.base_dir) == before


def test_a_copy_serves_the_sources_kmeans_apply_until_its_own_fit_shadows_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source's apply records the fit made before its tracks moved.

    The fit the copy makes over the moved tracks saves another state under
    the same run id.
    """
    source = _tracked_source(tmp_path / "source")
    runs = kmeans_fit(source)
    _ = run_feature(source, kmeans_apply_feature(runs), track=False)
    move_tracks(source, VARIANT, ("s1", "s2"))
    _rerun_speed(source, runs)
    applied = run_feature(source, kmeans_apply_feature(runs), track=False)
    fitted = read_fit_state_digest(result_run_root(source, runs.fit))
    assert model_cells(source, applied) == {"s1": fitted, "s2": fitted}
    copy = copy_of(source, tmp_path / "copy")
    link = f"source-{source.uuid}"
    before = tree_snapshot(source.base_dir)

    served = _applied(copy, kmeans_apply_feature(runs), capsys, monkeypatch)

    assert served.refusals == {}
    assert _link_rows(copy, served.result) == {"s1": link, "s2": link}

    unlinked = _relinked(copy, _WITHOUT_FEATURES)
    assert kmeans_fit(unlinked).run_ids() == runs.run_ids()
    own_root = result_run_root(copy, runs.fit)
    own = read_fit_state_digest(own_root)
    assert own not in {"", fitted}
    relinked = _relinked(unlinked, LINKED_KINDS)

    shadowed = _applied(relinked, kmeans_apply_feature(runs), capsys, monkeypatch)

    assert {model.parent for model in shadowed.models} == {own_root}
    assert shadowed.refusals == {"s1": FITTED_AGAIN, "s2": FITTED_AGAIN}
    assert model_cells(relinked, shadowed.result) == {"s1": own, "s2": own}
    assert _link_rows(relinked, shadowed.result) == {"s1": "", "s2": ""}
    assert tree_snapshot(source.base_dir) == before
