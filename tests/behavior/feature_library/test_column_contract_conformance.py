"""Every feature that declares its columns writes what its declaration says.

Pipeline validation decides from a feature's ``requires`` and ``provides``. A
declaration that the code contradicts gives a wrong answer before anything runs.
Each declared feature is therefore run, and its output compared with the
prediction that validation makes from the same declarations.

The chain each feature runs in is built from the declarations alone. The feature
reads the tracks, and each requirement the tracks do not guarantee is met by
adding the provider that ``input_candidates`` names for it. The chain then runs
in memory, one small tracks table through the real merge. Validation's
``resolved_columns`` over the same recipe gives the columns each output must and
may contain.

Undeclared features are listed in :data:`UNDECLARED`. The list only shrinks: a
feature that gains a declaration must leave it, and a new feature must declare
its columns.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from mosaic.behavior.feature_library.registry import FEATURES
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
    FeatureStepSpec,
    Recipe,
    build_feature,
    input_candidates,
    resolved_columns,
    topological_order,
)
from mosaic.core.pipeline.loading import load_entry_data
from mosaic.core.pipeline.types import Feature, LoadSpec, ParquetLoadSpec

UNDECLARED: frozenset[str] = frozenset(
    {
        "approach-avoidance",
        "arhmm",
        "attention-target",
        "body-scale",
        "collective-motion-metrics",
        "egocentric-crop",
        "extract-labeled-templates",
        "extract-templates",
        "feral",
        "global-identity-dinov2-temporal",
        "global-identity-embedding",
        "global-identity-model",
        "global-kmeans",
        "global-scaler",
        "global-tsne",
        "global-ward",
        "interaction-crop-pipeline",
        "kpms",
        "lightning-action",
        "local-order-metrics",
        "movement-filter-interpolate",
        "movement-smooth",
        "orientation-rel",
        "overlay",
        "pair-egocentric",
        "pair-facing",
        "pair-interaction-filter",
        "pair-posedistance-pca",
        "pair-position",
        "pair-wavelet",
        "scale-to-cm",
        "temporal-stack",
        "track-subsample",
        "xgboost",
    }
)
"""The registered features without a column contract. Shrink only."""

PARAMS: dict[str, dict[str, JsonValue]] = {
    "frame-aggregate": {"column": "X"},
    "id-tag-columns": {"fields": ["focal"], "field_renames": {"focal": "Focal_fish"}},
}
"""Params a feature needs to write anything, beside its defaults."""

EXTRA_INPUTS: dict[str, tuple[str, ...]] = {"nn-delta-bins": ("id-tag-columns",)}
"""Inputs a feature needs that its declared providers do not choose.

``nn-delta-bins`` requires a category column. ``id-tag-columns`` writes one only
under the params in :data:`PARAMS`, which a provider is not read under.
"""

GROUP = "g"
SEQUENCE = "s"


def _slug(cls: type) -> str:
    return str(getattr(cls, "name", cls.__name__))


def _declares(cls: type) -> bool:
    return declared_requires(cls) is not None and declared_provides(cls) is not None


FEATURE_CLASSES = sorted(FEATURES.values(), key=_slug)
DECLARED = [_slug(cls) for cls in FEATURE_CLASSES if _declares(cls)]


# --- the list ---------------------------------------------------------------------


def test_the_undeclared_list_names_exactly_the_undeclared_features() -> None:
    undeclared = {_slug(cls) for cls in FEATURE_CLASSES if not _declares(cls)}

    declared_now = sorted(UNDECLARED - undeclared)
    assert not declared_now, (
        f"{declared_now} declare their columns now, or are no longer registered. "
        "Remove them from UNDECLARED."
    )
    new = sorted(undeclared - UNDECLARED)
    assert not new, (
        f"{new} do not declare their columns. Declare requires and provides on "
        "the class, as speed-angvel does."
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=_slug)
def test_a_feature_declares_both_halves_or_neither(cls: type) -> None:
    """One half alone reads as a contract and checks only part of one."""
    assert (declared_requires(cls) is None) == (declared_provides(cls) is None), (
        f"{_slug(cls)} declares only one of requires and provides"
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


def _tracks(n_frames: int = 60, n_ids: int = 3) -> pd.DataFrame:
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


def _chain(target: str) -> Recipe:
    """The smallest chain the declarations say *target* needs, as a recipe."""
    inputs: dict[str, list[str]] = {}

    def add(name: str) -> None:
        if name in inputs:
            return
        inputs[name] = [TRACKS_INPUT]
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


def _merged(frames: list[pd.DataFrame], directory: Path) -> pd.DataFrame:
    """*frames* merged the way ``run_feature`` merges a step's inputs."""
    directory.mkdir(parents=True)
    specs: list[tuple[Path, LoadSpec]] = []
    for position, frame in enumerate(frames):
        path = directory / f"{position}.parquet"
        frame.to_parquet(path)
        specs.append((path, ParquetLoadSpec()))
    merged = load_entry_data(specs)
    assert merged is not None
    return merged


def _built(step: FeatureStepSpec) -> Feature:
    payload: list[object] = [
        item if isinstance(item, str) else {"feature": item.step}
        for item in step.inputs
    ]
    return build_feature(step.feature, payload, dict(step.params))


def _run(
    recipe: Recipe, directory: Path
) -> dict[str, tuple[Feature, pd.DataFrame, pd.DataFrame]]:
    """Every step's feature, merged input and output, run in topological order."""
    tracks = _tracks()
    labels = {"labels": {(GROUP, SEQUENCE): _labels(directory)}}
    ran: dict[str, tuple[Feature, pd.DataFrame, pd.DataFrame]] = {}
    for step in topological_order(recipe):
        assert isinstance(step, FeatureStepSpec)
        frames = [
            tracks if isinstance(item, str) else ran[item.step][2]
            for item in step.inputs
        ]
        merged = _merged(frames, directory / step.id)
        feature = _built(step)
        assert feature.load_state(directory, {}, labels)
        check_input_columns(feature, merged.columns)
        ran[step.id] = (feature, merged, feature.apply(merged))
    return ran


# --- the contract -----------------------------------------------------------------------


@pytest.mark.parametrize("slug", DECLARED)
def test_a_declared_feature_writes_the_columns_it_declares(
    slug: str, tmp_path: Path
) -> None:
    recipe = _chain(slug)
    predicted = resolved_columns(recipe)[slug]
    _, merged, output = _run(recipe, tmp_path)[slug]
    provides = declared_provides(FEATURES[_class_name(slug)])
    assert provides is not None

    written = {str(column) for column in output.columns}
    assert not output.empty, f"{slug} wrote an empty frame"
    missing = predicted.guaranteed - written
    assert not missing, f"{slug} declares {sorted(missing)} and did not write them"
    assert not predicted.unbounded, f"{slug}'s declared output is unbounded"
    passed: set[str] = (
        {str(column) for column in merged.columns} if provides.passthrough else set()
    )
    undeclared = sorted(
        column for column in written - passed if not predicted.admits(column)
    )
    assert not undeclared, f"{slug} wrote {undeclared}, which it does not declare"


@pytest.mark.parametrize("slug", DECLARED)
def test_removing_a_required_column_raises_the_uniform_error(
    slug: str, tmp_path: Path
) -> None:
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
