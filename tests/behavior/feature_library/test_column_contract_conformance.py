"""Every feature declares its columns, and writes what its declaration says.

Pipeline validation decides from a feature's ``requires`` and ``provides``. A
declaration that the code contradicts gives a wrong answer before anything runs.
Each feature is therefore run, and its output compared with the prediction
that validation makes from the same declarations.

The chain each feature runs in is built from the declarations alone. The feature
reads the tracks, or :data:`MEASUREMENTS` when it reads only other features. Its
input may leave a requirement unguaranteed, and the chain then gains the
provider that ``input_candidates`` names for it. A step named in its params,
such as the templates that a clustering fits on, joins the chain too. The chain
runs in memory, one small tracks table through the real merge, and a global
feature fits on its merged input before it applies. Validation's
``resolved_columns`` over the same recipe gives the columns each output must and
may contain.

Every registered feature declares both halves. A feature registered outside
mosaic may declare neither, and validation then reports its steps as unchecked.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.column_contract import (
    MissingInputColumns,
    check_input_columns,
    contract_fields,
    declared_provides,
    declared_requires,
    needs_of,
    param_values,
)
from mosaic.core.pipeline.graph import (
    TRACKS_INPUT,
    Declaration,
    FeatureStepSpec,
    Recipe,
    ResolvedStep,
    build_feature,
    declaration_catalog,
    input_candidates,
    params_step_refs,
    resolve_step_spec,
    resolved_columns,
    topological_order,
)
from mosaic.core.pipeline.loading import CROSS_JOIN_FEATURES, load_entry_data
from mosaic.core.pipeline.tracks_index import write_tracks_row
from mosaic.core.pipeline.types import (
    ArtifactSpec,
    Feature,
    InputStream,
    LoadSpec,
    ParquetLoadSpec,
)
from tests.helpers import (
    assert_writes_declared_columns,
    make_dataset,
    require_ffmpeg,
    write_h264_mp4,
)

PARAMS: dict[str, dict[str, JsonValue]] = {
    "approach-avoidance": {"orientation_gate_cos": None},
    "arhmm": {"n_states": 2, "n_iter": 5},
    "egocentric-crop": {"transform_keypoints": True},
    "evaluate-labels": {"mode": "clusters", "prediction_column": "cluster"},
    "extract-labeled-templates": {"labels": {"kind": "behavior"}, "n_total": 40},
    "fit-label-map": {"prediction_column": "cluster"},
    "extract-templates": {"n_templates": 50},
    "frame-aggregate": {"column": "X"},
    "global-kmeans": {"templates": {"step": "extract-templates"}, "k": 2, "n_init": 1},
    "global-scaler": {"templates": {"step": "extract-templates"}},
    "global-tsne": {
        "templates": {"step": "extract-templates"},
        "perplexity": 5,
        "n_jobs": 1,
        "fit": {"exaggeration_iters": 10, "iters": 10},
        "mapping": {"k": 15, "iters": 5},
    },
    "global-ward": {"templates": {"step": "extract-templates"}, "n_clusters": 2},
    "id-tag-columns": {"fields": ["focal"], "field_renames": {"focal": "Focal_fish"}},
    "lightning-action": {
        "templates": {"step": "extract-labeled-templates"},
        "default_class": 0,
        "num_epochs": 1,
        "sequence_length": 10,
    },
    "local-order-metrics": {"radius": 10.0},
    "pair-egocentric": {"pose": {"pose_n": 3}, "neck_idx": 0, "tail_base_idx": 1},
    "pair-facing": {"pose_head_index": 0, "pose_abdomen_index": 1},
    "pair-interaction-filter": {
        "max_dist": 1000.0,
        "require_facing": False,
        "min_run_frames": 5,
        "morphological_structure_size": 0,
    },
    "pair-posedistance-pca": {"pose": {"pose_n": 3}},
    "scale-to-cm": {"cm_per_pixel": 0.1},
    "xgboost": {
        "templates": {"step": "extract-labeled-templates"},
        "default_class": 0,
        "n_estimators": 5,
    },
}
"""Params a feature needs, beside its defaults, to write any column it declares.

A global model's ``templates`` names the step that writes its templates, and
the sizes keep each fit small.

The keypoint features default to layouts of more than the three keypoints of
:func:`_tracks`. ``pair-interaction-filter`` writes only the frames of an
interaction, and the swimmers of :func:`_tracks` do not face each other.
``approach-avoidance`` reads ``ANGLE`` while ``orientation_gate_cos`` is set,
and its declaration does not require it.

``scale-to-cm`` reads its scale from the media index when ``cm_per_pixel`` is
unset, and the media index of :func:`_entry_dataset` records none.

``egocentric-crop`` writes its crop keypoints, its one conditional output, under
``transform_keypoints``. ``interaction-crop-pipeline`` writes a color crop
through the AV1 writer with NVENC requested, and the NVENC probe raises on an
ffmpeg build without ``av1_nvenc``. OpenCV writes a gray crop.
"""

EXTRA_INPUTS: dict[str, tuple[str, ...]] = {
    "interaction-crop-pipeline": ("pair-interaction-filter",),
    "nn-delta-bins": ("id-tag-columns",),
}
"""Inputs a feature needs that its declared providers do not choose.

``nn-delta-bins`` requires a category column. ``id-tag-columns`` writes one only
under the params in :data:`PARAMS`, which a provider is not read under.
``interaction-crop-pipeline`` crops the segments of ``pair-interaction-filter``,
their one writer.
"""

SOURCES: dict[str, tuple[str, ...]] = {
    "attention-target": ("pair-facing",),
    "pair-wavelet": ("pair-posedistance-pca",),
}
"""Inputs a feature reads in place of the tracks or :data:`MEASUREMENTS`.

A feature reading pair rows cannot also read the tracks, whose rows are about
individuals.
"""

MEASUREMENTS = "speed-angvel"
"""What a feature that does not read the tracks reads in their place.

Its output contains the identity columns and three numeric measurements, the
payload that a global feature samples or fits on.
"""

_CROPS = "it reads the crop images that egocentric-crop writes and fits a network"

UNRUN: dict[str, str] = {
    "feral": "it reads crop videos and fine-tunes a video transformer",
    "global-identity-dinov2-temporal": _CROPS,
    "global-identity-embedding": _CROPS,
    "global-identity-model": _CROPS,
    "kpms": "it fits keypoint-MoSeq in a separate server, as the slow kpms tests do",
    "lightning-action": (
        "its extra requires nvidia-dali-cuda110, which installs on no runner "
        "without CUDA"
    ),
    "movement-filter-interpolate": "no CI job installs the movement extra",
    "movement-smooth": "no CI job installs the movement extra",
    "vame": (
        "it trains VAME in a separate environment, and the slow "
        "test_vame_integration.py compares its output with its declaration"
    ),
}
"""Declared features whose chain runs in no CI job, and why.

Their declarations are still checked against their params. Each declares a
prerequisite beyond the base install: a feature that the base install runs
cannot be listed. A feature outside the list runs, and one whose prerequisites
are unmet fails rather than skipping, because a skip here is a case that runs
nowhere.
"""

GROUP = "g"
SEQUENCE = "s"
FRAMES = 60


def _slug(cls: type) -> str:
    return str(getattr(cls, "name", cls.__name__))


FEATURE_CLASSES = sorted(FEATURES.values(), key=_slug)
SLUGS = [_slug(cls) for cls in FEATURE_CLASSES]
CATALOG = declaration_catalog()


def _declaration(slug: str) -> Declaration:
    declared = CATALOG.get(slug)
    assert declared is not None, slug
    return declared


# --- the declarations -------------------------------------------------------------


def test_every_unrun_feature_is_registered() -> None:
    assert not set(UNRUN) - set(SLUGS), sorted(set(UNRUN) - set(SLUGS))


def test_only_a_feature_with_prerequisites_goes_unrun() -> None:
    """The base install runs every other feature, and its chain must run here."""
    plain = sorted(
        slug
        for slug in UNRUN
        if not getattr(FEATURES[_class_name(slug)], "prerequisites", ())
    )
    assert not plain, (
        f"{plain} declare no prerequisite beyond the base install, and their "
        "chains run here. Remove them from UNRUN."
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=_slug)
def test_a_feature_declares_its_columns(cls: type) -> None:
    """Validation checks a step of a feature that declares one half only in part."""
    halves = {"requires": declared_requires(cls), "provides": declared_provides(cls)}
    missing = [half for half, declared in halves.items() if declared is None]
    assert not missing, (
        f"{_slug(cls)} does not declare {' or '.join(missing)}. Declare requires "
        "and provides on the class, as speed-angvel does."
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=_slug)
def test_every_params_field_a_declaration_reads_is_a_params_field(cls: type) -> None:
    """A misspelled field resolves as unknown, which leaves its requirement unchecked."""
    params: object = getattr(cls, "Params", None)
    assert isinstance(params, type) and issubclass(params, BaseModel), _slug(cls)
    read = contract_fields(declared_requires(cls), declared_provides(cls))

    unknown = sorted(read - set(params.model_fields))
    assert not unknown, f"{_slug(cls)}'s Params lacks {unknown}, which it declares"


# --- the fixture --------------------------------------------------------------------


def _tracks(n_frames: int = FRAMES, n_ids: int = 3) -> pd.DataFrame:
    """A ``mosaic_v1`` table: three swimmers close together, with three keypoints.

    No ``ANGLE``, which ``mosaic_v1`` forbids: a feature reading one runs
    ``heading`` first, as a real chain does.
    """
    rows: list[dict[str, object]] = []
    noise = np.random.default_rng(0)
    for frame in range(n_frames):
        for identity in range(n_ids):
            phase = frame / 8.0 + identity
            x = 50.0 + 4.0 * identity + 5.0 * math.cos(phase) + noise.normal(0, 0.1)
            y = 50.0 + 5.0 * math.sin(phase) + noise.normal(0, 0.1)
            heading = phase + math.pi / 2
            row: dict[str, object] = {
                "frame": frame,
                "time": frame / 30.0,
                "id": identity,
                "group": GROUP,
                "sequence": SEQUENCE,
                "X": x,
                "Y": y,
            }
            for keypoint, offset in enumerate((2.0, -2.0, 0.0)):
                row[f"poseX{keypoint}"] = x + offset * math.cos(heading)
                row[f"poseY{keypoint}"] = y + offset * math.sin(heading)
            rows.append(row)
    return pd.DataFrame(rows)


def _labels(directory: Path) -> Path:
    """One ``id_tags`` labels file: individual 0 is the focal."""
    path = directory / "labels.npz"
    np.savez(path, ids=np.array([0, 1, 2]), focal=np.array([1, 0, 0]))
    return path


def _behavior_labels(directory: Path) -> Path:
    """One dense behavior labels file: classes 0 and 1 in turn, ten frames each."""
    path = directory / "behavior.npz"
    np.savez(path, labels=(np.arange(FRAMES) // 10) % 2)
    return path


LABELS: dict[str, Callable[[Path], Path]] = {
    "evaluate-labels": _behavior_labels,
    "extract-labeled-templates": _behavior_labels,
    "fit-label-map": _behavior_labels,
}
"""The labels file that a feature reads, where it is not the ``id_tags`` one."""


# --- the chain ------------------------------------------------------------------------


def _recipe(inputs: dict[str, list[str]]) -> Recipe:
    """A recipe with one step per feature, named by its slug."""
    return Recipe.model_validate(
        {
            "steps": [
                {
                    "id": name,
                    "type": "feature",
                    "feature": name,
                    "inputs": [
                        item if item == TRACKS_INPUT else {"step": item}
                        for item in reads
                    ],
                    "params": PARAMS.get(name, {}),
                }
                for name, reads in inputs.items()
            ]
        }
    )


def _sources(name: str) -> list[str]:
    """The inputs that *name* starts from.

    They are its :data:`SOURCES`, else the tracks, else :data:`MEASUREMENTS` for
    a feature that reads only other features.
    """
    if name in SOURCES:
        return list(SOURCES[name])
    return [
        TRACKS_INPUT if _declaration(name).consumes.accepts_tracks else MEASUREMENTS
    ]


def _referenced(name: str) -> list[str]:
    """The steps that *name*'s params in :data:`PARAMS` refer to."""
    return [ref.step for ref in params_step_refs(PARAMS.get(name, {})).values()]


def _chain(target: str) -> Recipe:
    """The smallest chain the declarations say *target* needs, as a recipe."""
    inputs: dict[str, list[str]] = {}

    def add(name: str) -> None:
        if name in inputs:
            return
        sources = _sources(name)
        for upstream in (*sources, *_referenced(name)):
            if upstream != TRACKS_INPUT:
                add(upstream)
        inputs[name] = sources
        for extra in EXTRA_INPUTS.get(name, ()):
            add(extra)
            inputs[name].append(extra)
        while True:
            answer = input_candidates(_recipe(inputs), name)
            provider = next(
                (
                    provider.feature
                    for still in answer.still_needs
                    for provider in still.providers
                    if not provider.when_param
                    and not provider.when_present
                    and provider.feature != TRACKS_INPUT
                    and provider.feature not in inputs[name]
                ),
                None,
            )
            if provider is None:
                return
            add(provider)
            inputs[name].append(provider)

    add(target)
    return _recipe(inputs)


def _merged(frames: list[pd.DataFrame], directory: Path, feature: str) -> pd.DataFrame:
    """*frames* merged the way ``run_feature`` merges the inputs of *feature*."""
    directory.mkdir(parents=True)
    specs: list[tuple[Path, LoadSpec]] = []
    for position, frame in enumerate(frames):
        path = directory / f"{position}.parquet"
        frame.to_parquet(path)
        specs.append((path, ParquetLoadSpec()))
    merged = load_entry_data(specs, cross_join=feature in CROSS_JOIN_FEATURES)
    assert merged is not None
    return merged


@runtime_checkable
class _ReadsTheDataset(Protocol):
    """A feature that opens the entry's video, or its tracks, through the dataset."""

    def bind_dataset(self, ds: Dataset) -> None: ...


def _entry_dataset(directory: Path, tracks: pd.DataFrame) -> Dataset:
    """Return a dataset containing *tracks* and a video of the entry.

    The tracks row is written first, because ``index_media`` reads the tracks
    index for the group and sequence of a video.
    """
    require_ffmpeg()
    dataset = make_dataset(directory)
    table = dataset.get_root("tracks") / f"{GROUP}__{SEQUENCE}.parquet"
    tracks.to_parquet(table)
    write_tracks_row(
        dataset,
        run_id="convert-fixture.0.1-aaaaaaaaaa",
        group=GROUP,
        sequence=SEQUENCE,
        out_path=table,
        producer="convert-fixture",
        std_format="mosaic_v1",
        n_rows=len(tracks),
    )
    media = dataset.get_root("media_raw")
    write_h264_mp4(
        media / f"{SEQUENCE}.mp4", frames=int(tracks["frame"].max()) + 1, size=(96, 64)
    )
    _ = dataset.index_media([media])
    return dataset


def _built(recipe: Recipe, step: FeatureStepSpec) -> Feature:
    """The feature that *step* runs, its references substituted as a plan does.

    Each step's output is stored under its id, which is its feature's slug.
    """
    resolved = {
        other.id: ResolvedStep(step_id=other.id, storage_name=other.id)
        for other in recipe.steps
    }
    spec = resolve_step_spec(recipe, step.id, resolved)
    return build_feature(step.feature, list(spec.inputs), dict(spec.params))


def _state(directory: Path, step_id: str) -> Path:
    """Where the step *step_id* keeps what its fit wrote."""
    return directory / step_id / "state"


def _artifacts(
    feature: Feature, step: FeatureStepSpec, directory: Path
) -> dict[str, Path]:
    """Each artifact that *step*'s params refer to, in its writer's state."""
    found: dict[str, Path] = {}
    for name, ref in params_step_refs(step.params).items():
        spec: object = getattr(feature.params, name)
        assert isinstance(spec, ArtifactSpec), name
        found[name] = _state(directory, ref.step) / spec.pattern
    return found


def _stream(merged: pd.DataFrame) -> InputStream:
    """The one entry of the fixture, as a fit reads its entries."""
    return InputStream(
        lambda: iter([(make_entry_key(GROUP, SEQUENCE), merged)]), n_entries=1
    )


def _run(
    recipe: Recipe, directory: Path
) -> dict[str, tuple[Feature, pd.DataFrame, pd.DataFrame]]:
    """Every step's feature, merged input and output, run in topological order.

    A feature that reads the dataset is bound to one containing the same tracks
    and a video of the entry. A step without a saved state fits on its merged
    input, as ``run_feature`` fits on every entry before it applies.
    """
    tracks = _tracks()
    dataset: Dataset | None = None
    ran: dict[str, tuple[Feature, pd.DataFrame, pd.DataFrame]] = {}
    for step in topological_order(recipe):
        assert isinstance(step, FeatureStepSpec)
        frames = [
            tracks if isinstance(item, str) else ran[item.step][2]
            for item in step.inputs
        ]
        merged = _merged(frames, directory / step.id, step.feature)
        feature = _built(recipe, step)
        if isinstance(feature, _ReadsTheDataset):
            if dataset is None:
                dataset = _entry_dataset(directory / "dataset", tracks)
            feature.bind_dataset(dataset)
        labels = LABELS.get(step.feature, _labels)(directory)
        lookups = {"labels": {(GROUP, SEQUENCE): labels}}
        state = _state(directory, step.id)
        if not feature.load_state(state, _artifacts(feature, step, directory), lookups):
            feature.fit(_stream(merged))
            feature.save_state(state)
        check_input_columns(
            feature, merged.columns, numeric=merged.select_dtypes("number").columns
        )
        ran[step.id] = (feature, merged, feature.apply(merged))
    return ran


# --- the contract -----------------------------------------------------------------------


def _skip_unless_runnable(slug: str) -> None:
    """Skip a feature listed in :data:`UNRUN`, naming the reason.

    Any other feature runs. One whose prerequisites are unmet fails, naming
    them.
    """
    if slug in UNRUN:
        pytest.skip(f"{slug} is not run here: {UNRUN[slug]}")
    unavailable = _declaration(slug).unavailable
    assert unavailable is None, (
        f"{slug} cannot run here: {unavailable}. Install its prerequisites in the "
        "job that runs this module, or list it in UNRUN with the reason no job can."
    )


@pytest.mark.parametrize("slug", SLUGS)
def test_a_feature_writes_the_columns_it_declares(slug: str, tmp_path: Path) -> None:
    _skip_unless_runnable(slug)
    recipe = _chain(slug)
    predicted = resolved_columns(recipe)[slug]
    _, merged, output = _run(recipe, tmp_path)[slug]
    provides = declared_provides(FEATURES[_class_name(slug)])
    assert provides is not None

    passed: set[str] = (
        {str(column) for column in merged.columns} if provides.passthrough else set()
    )
    assert_writes_declared_columns(slug, predicted, output, passed)


@pytest.mark.parametrize("slug", SLUGS)
def test_removing_a_required_column_raises_the_uniform_error(
    slug: str, tmp_path: Path
) -> None:
    _skip_unless_runnable(slug)
    feature, merged, _ = _run(_chain(slug), tmp_path)[slug]
    requires = declared_requires(type(feature))
    assert requires is not None
    values = param_values(feature.params, contract_fields(requires, None))

    for need in needs_of(requires, values):
        meeting = [column for column in merged.columns if need.met_by({str(column)})]
        with pytest.raises(MissingInputColumns) as raised:
            check_input_columns(feature, merged.drop(columns=meeting).columns)
        assert need in raised.value.missing, (slug, need.label)
        assert need.label in str(raised.value)


def _class_name(slug: str) -> str:
    """The registry key of the feature named *slug*."""
    return next(key for key, cls in FEATURES.items() if _slug(cls) == slug)
