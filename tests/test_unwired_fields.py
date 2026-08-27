"""Records the parameter fields that declare ``unwired=`` and the wiring each lacks.

Every field below is published into its params schema, and a client drawing a
form from that schema renders a control for it. A field whose ``Declared(...)``
passes ``unwired="<why>"`` publishes ``x-mosaic-unwired`` beside it, which is how
that client knows to leave the control out. Whether to delete such a field is
the maintainer's call, and until that call is made each field gets a ``strict``
xfail test naming the wiring it lacks. The day someone supplies that
wiring the test xpasses and the suite fails until the marker is deleted.

Every test below asserts across each place a reader could plausibly sit, never at
one call site. A test pinned to one call site keeps xfailing after the field is
wired elsewhere, and the record beside the field then states something untrue.

The scans come from ``tests.helpers.source_scan``. A read counts only off the
names that hold the object in the scanned region, since reading by name alone
lets any ``np.load`` retire a labels source's ``load`` field. Adding a binding to
a region therefore means adding its name here.

Each scanned region also gets an unmarked control asserting a field that region
does read. A ``strict`` xfail reports every failure as an xfail, and a control
written inside a record could never fire. ``functions_named`` raises on a renamed
function. A scan narrowed any other way -- a rebound local, an ast node shape the
walker no longer matches -- leaves every record xfailing against a set that lost
the field, and the control fails instead.

``test_every_unwired_field_is_recorded_by_a_test`` ties the two sets together, so
an eleventh field cannot land with no record. Three of the ten are recorded
beside their subjects rather than here:

- ``LabelConvertParams.strict_schema`` --
  ``tests/test_labels_index.py::test_the_label_conversion_reads_strict_schema``
- ``TrainLitposeParams.device`` --
  ``tests/test_train_litpose.py::test_the_device_reaches_the_trainer``
- ``PointInferParams.dor`` --
  ``tests/test_ultralytics_wire_contract.py::test_the_point_inference_request_carries_dor``
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import mosaic
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library import extract_labeled_templates, feral_feature
from mosaic.behavior.feature_library import track_subsample as track_subsample_module
from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures
from mosaic.core.pipeline import _loaders
from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.types import labels as labels_types
from mosaic.core.pipeline.types.data_config import COLUMNS, PoseConfig
from tests.helpers import (
    functions_named,
    module_tree,
    names_read,
    runs_in_an_external_environment,
    source_tree,
)

# --- GroundTruthLabelsSource: source, load, pattern -------------------------

_LABELS_SOURCE_CLASSES = frozenset({"LabelsSource", "GroundTruthLabelsSource"})

_LABELS_RESOLVERS = (
    "_resolve_dependencies",
    "_build_labels_lookup",
    "resolve_labels_variants",
    "load_values",
)

# Every name a labels source is bound to across the scanned region, and only
# those. ``value`` is the ``match`` subject in ``_resolve_dependencies`` and the
# isinstance-guarded params-field value in ``resolve_labels_variants``; ``s``,
# ``ls`` and ``source`` are the comprehension and loop variables in
# ``load_values``; ``self.params.labels`` is the params field the consuming
# feature reaches it through. ``self`` is there because ``types/labels.py``
# declares the class, and a validator or a computed field on it reads its own
# fields that way. A plausible-looking name that binds something else belongs
# nowhere here -- ``spec`` names an ``ArtifactSpec`` two cases up, and listing it
# would let a pure refactor retire the pattern marker.
_LABELS_SOURCE_OWNERS = frozenset(
    {"self", "value", "source", "s", "ls", "self.params.labels"}
)


def _labels_resolution_reads() -> set[str]:
    """Returns the field names every place meeting a labels source reads.

    Three regions. The four functions in ``core/pipeline/run.py`` that resolve
    one -- the dependency match, the two resolvers it calls, and the value
    loader. The module declaring the class, which is why ``self`` is an owner:
    a validator, a computed field or a ``Field(discriminator=...)`` on the class
    reads a field there without any resolver seeing it. And
    ``extract_labeled_templates``, the feature that reads the resolved file,
    where a ``LoadSpec`` or a pattern would be honored.
    """
    trees = functions_named(run_module, _LABELS_RESOLVERS)
    trees.append(module_tree(labels_types))
    trees.append(module_tree(extract_labeled_templates))
    return names_read(
        trees,
        owners=_LABELS_SOURCE_OWNERS,
        destructured_classes=_LABELS_SOURCE_CLASSES,
    )


def test_the_labels_scan_reports_a_field_the_resolution_reads() -> None:
    """Pins the scan the three records below are measured against.

    ``kind`` names the ``labels/<kind>`` directory ``_build_labels_lookup``
    resolves against. It is read whether or not ``source``, ``load`` and
    ``pattern`` ever are.
    """
    assert "kind" in _labels_resolution_reads()


@pytest.mark.xfail(
    strict=True,
    reason="GroundTruthLabelsSource.source is declared but never read",
)
def test_the_labels_resolution_reads_the_source_tag() -> None:
    """The record for a tag that decides nothing.

    ``source`` is fixed to ``"labels"``. Three of the four resolvers tell a
    ``GroundTruthLabelsSource`` apart from a ``ResultColumn`` by ``isinstance``
    and the fourth by a ``match`` class pattern. ``LabelsSourceSpec`` is a bare
    union, with no ``Field(discriminator=...)`` applied to it.
    """
    assert "source" in _labels_resolution_reads()


@pytest.mark.xfail(
    strict=True,
    reason="GroundTruthLabelsSource.load is declared but never read",
)
def test_the_labels_resolution_reads_the_load_spec() -> None:
    """The record for a load specification every reader ignores.

    ``load`` defaults to ``NpzLoadSpec(key="labels")`` and validates as a full
    ``LoadSpec``, down to the archive key and the transpose flag. Both readers of
    a resolved label file call ``load_labels_for_feature_frames``, which takes a
    path and detects the format itself. ``load_from_spec`` never receives this
    value.
    """
    assert "load" in _labels_resolution_reads()


@pytest.mark.xfail(
    strict=True,
    reason="GroundTruthLabelsSource.pattern is declared but never read",
)
def test_the_labels_resolution_reads_the_file_pattern() -> None:
    """The record for a glob the resolver never expands.

    ``pattern`` is documented as a glob over ``labels/<kind>/``.
    ``_build_labels_lookup`` resolves the label file from the ``abs_path``
    recorded in ``labels/<kind>/index.csv`` and never globs that directory.
    """
    assert "pattern" in _labels_resolution_reads()


# --- ParquetLoadSpec.frame_column ------------------------------------------


def _parquet_load_reads() -> set[str]:
    """Returns the field names the loader module reads off a ``ParquetLoadSpec``.

    The whole module is scanned rather than the one ``case``, since a reader
    could equally sit in a validator on the model or in a helper the case calls
    -- a helper's own parameter is picked up from its annotation.
    """
    return names_read(
        [module_tree(_loaders)],
        owners={"self", "spec"},
        destructured_classes={"ParquetLoadSpec"},
    )


def test_the_parquet_scan_reports_a_field_the_loader_reads() -> None:
    """Pins the scan the record below is measured against.

    ``columns`` is one of the four fields the ``ParquetLoadSpec`` case
    destructures. It is read whether or not ``frame_column`` ever is.
    """
    assert "columns" in _parquet_load_reads()


@pytest.mark.xfail(
    strict=True,
    reason="ParquetLoadSpec.frame_column is declared but never read",
)
def test_the_parquet_load_reads_the_frame_column() -> None:
    """The record for a column name the parquet loader ignores.

    ``frame_column`` names the column meant to be extracted as frame indices,
    and is published wherever a ``LoadSpec`` is. ``load_from_spec``'s
    ``ParquetLoadSpec`` case destructures ``columns``, ``drop_columns``,
    ``numeric_only`` and ``transpose``, then returns the frame with its index
    untouched.
    """
    assert "frame_column" in _parquet_load_reads()


# --- TrackSubsample.Params.drop_nan ----------------------------------------


def _subsampling_reads() -> set[str]:
    """Returns the field names ``track_subsample`` reads off its ``Params``."""
    return names_read(
        [module_tree(track_subsample_module)],
        owners={"self", "p", "params", "self.params"},
        destructured_classes={"Params"},
    )


def test_the_subsampling_scan_reports_a_field_apply_reads() -> None:
    """Pins the scan the record below is measured against.

    ``method`` selects the branch ``apply()`` takes. It is read whether or not
    ``drop_nan`` ever is.
    """
    assert "method" in _subsampling_reads()


@pytest.mark.xfail(
    strict=True,
    reason="TrackSubsample.Params.drop_nan is declared but never read",
)
def test_the_subsampling_reads_drop_nan() -> None:
    """The record for a flag ``apply()`` ignores.

    ``drop_nan`` defaults to ``True``. ``apply()`` emits whatever rows the
    chosen method selects: k-means excludes the rows it cannot canonicalize from
    the clustering whichever way the flag is set, and the uniform and clip
    methods take a stride over the input untouched.
    """
    assert "drop_nan" in _subsampling_reads()


# --- FeralTrainingConfig.wandb_project --------------------------------------


def _reaches_the_wandb_package(tree: ast.AST) -> bool:
    """Whether *tree* imports or names the ``wandb`` package."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "wandb" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "wandb":
                return True
        elif isinstance(node, ast.Name):
            if node.id == "wandb":
                return True
    return False


@pytest.mark.xfail(
    strict=True,
    reason="FeralTrainingConfig.wandb_project is declared but never starts a run",
)
def test_the_feral_training_starts_a_weights_and_biases_run() -> None:
    """The record for a project name the module never sends.

    ``fit`` writes ``wandb_project`` into the flat config dict it saves as
    ``config.json``. Setting it changes that file and the run identifier, and
    nothing else.

    Asserted against the ``wandb`` package being reached rather than against a
    read of the attribute, because the attribute is read -- into that config
    dict. What is absent is a run. This one module is the whole boundary: the
    training loop is written here, on ``torch``'s ``DataLoader``, ``AdamW`` and
    a cosine schedule, and FERAL supplies the dataset, the model and the
    checkpoint writer alone. ``cfg`` reaches FERAL once, as opaque checkpoint
    metadata, so an ``import wandb`` here is the only path to a run.
    """
    assert _reaches_the_wandb_package(module_tree(feral_feature))


# --- PairEgocentricFeatures.Params.center_mode ------------------------------

# Two individuals, four static frames. Keypoint 0 of each sits at y = 0, so the
# separation between the keypoint-0 centers is 10; the keypoint means sit at
# y = 3 and y = 1, so the separation between them is sqrt(10^2 + 2^2).
_KEYPOINTS_A: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 1.0), (0.0, 8.0))
_KEYPOINTS_B: tuple[tuple[float, float], ...] = ((10.0, 0.0), (10.0, 1.0), (10.0, 2.0))
_KEYPOINT_0_SEPARATION = 10.0
_KEYPOINT_MEAN_SEPARATION = float(np.sqrt(104.0))
_FRAMES = 4
_POSE = PoseConfig()
_TOLERANCE = 1e-4


def _pair_tracks() -> pd.DataFrame:
    """Builds a two-individual tracks frame whose keypoint 0 and mean differ."""
    rows: list[dict[str, object]] = []
    for frame in range(_FRAMES):
        for identity, keypoints in ((1, _KEYPOINTS_A), (2, _KEYPOINTS_B)):
            row: dict[str, object] = {
                COLUMNS.frame_col: frame,
                COLUMNS.id_col: identity,
                COLUMNS.seq_col: "s",
            }
            for index, (x, y) in enumerate(keypoints):
                row[f"{_POSE.x_prefix}{index}"] = x
                row[f"{_POSE.y_prefix}{index}"] = y
            rows.append(row)
    return pd.DataFrame(rows)


def _pair_separations(value: int | str) -> list[float]:
    """Returns the ``AB_dist`` the feature emits for ``center_mode=value``.

    Empty when the field refuses the value, which is the state this records. The
    branch selecting one keypoint requires an ``int``, and ``center_mode`` is
    typed ``str``.
    """
    params: dict[str, object] = {
        "center_mode": value,
        "neck_idx": 1,
        "tail_base_idx": 0,
        "pose": {"pose_n": len(_KEYPOINTS_A)},
    }
    try:
        feature = PairEgocentricFeatures(params=params)
    except ValidationError:
        return []
    out = feature.apply(_pair_tracks())
    return [float(distance) for distance in np.asarray(out["AB_dist"], dtype=float)]


def test_the_pair_fixture_separates_the_keypoint_mean_from_keypoint_zero() -> None:
    """Pins the fixture the record below is measured against.

    Unmarked, and separate, because a ``strict`` xfail reports every failure as
    an xfail. A drift check written inside the record -- prefixes that stop
    matching, an ``AB_dist`` defined against something other than the two
    centers, a mean that collapses onto keypoint 0 -- could never fail there,
    and the record would keep xfailing while measuring nothing. Here it fails.

    ``"mean"`` is the baseline because it averages under every wiring, so this
    stays true the day ``center_mode`` is wired and the record retires.
    """
    baseline = _pair_separations("mean")
    assert baseline, "the feature emitted no pair; the fixture no longer aligns"
    assert all(
        math.isclose(d, _KEYPOINT_MEAN_SEPARATION, abs_tol=_TOLERANCE) for d in baseline
    ), f"AB_dist is no longer the separation between the keypoint means: {baseline}"
    assert not math.isclose(
        _KEYPOINT_MEAN_SEPARATION, _KEYPOINT_0_SEPARATION, abs_tol=_TOLERANCE
    )


@pytest.mark.xfail(
    strict=True,
    reason="PairEgocentric.Params.center_mode is declared but never selects a keypoint",
)
def test_a_params_center_mode_selects_one_keypoint() -> None:
    """The record for a mode with one behavior under every accepted value.

    ``_center_from_points`` branches on ``isinstance(mode, (int, np.integer))``
    and returns one keypoint's coordinates, falling through to the mean of all
    keypoints otherwise. ``center_mode`` is typed ``str``. Pydantic therefore
    never produces an integer, and the per-keypoint branch is unreachable from
    params.

    Asserted behaviorally rather than by scanning for a read, because the field
    is read -- ``_build_ego_block_for_joined`` passes it straight to the helper.
    What is absent is a value reaching the branch, so the assertion offers the
    field both spellings a future author would supply, takes whichever
    validates, and measures the separation the feature emits. Widening the type
    to admit an integer retires the marker, and so does making the guard accept
    a numeric string. That the fixture still measures what it claims is pinned
    by the test above, which is unmarked so that it can fail.
    """
    candidates = [d for value in (0, "0") for d in _pair_separations(value)]
    assert candidates, "the field refused every candidate and emitted nothing"
    assert any(
        math.isclose(d, _KEYPOINT_0_SEPARATION, abs_tol=_TOLERANCE) for d in candidates
    )


# --- the registry: every unwired field is recorded --------------------------

_SRC = Path(mosaic.__file__ or "").resolve().parent

_HERE = "tests/test_unwired_fields.py"

# Every ``unwired=`` declaration under ``src/mosaic/``, keyed by its module path
# and field name, mapped to the test that records it.
_RECORDED_BY: dict[tuple[str, str], str] = {
    ("core/label_converter.py", "strict_schema"): (
        "tests/test_labels_index.py::test_the_label_conversion_reads_strict_schema"
    ),
    ("core/pipeline/_loaders.py", "frame_column"): (
        f"{_HERE}::test_the_parquet_load_reads_the_frame_column"
    ),
    ("core/pipeline/types/labels.py", "source"): (
        f"{_HERE}::test_the_labels_resolution_reads_the_source_tag"
    ),
    ("core/pipeline/types/labels.py", "load"): (
        f"{_HERE}::test_the_labels_resolution_reads_the_load_spec"
    ),
    ("core/pipeline/types/labels.py", "pattern"): (
        f"{_HERE}::test_the_labels_resolution_reads_the_file_pattern"
    ),
    ("behavior/feature_library/feral_feature.py", "wandb_project"): (
        f"{_HERE}::test_the_feral_training_starts_a_weights_and_biases_run"
    ),
    ("behavior/feature_library/pair_egocentric.py", "center_mode"): (
        f"{_HERE}::test_a_params_center_mode_selects_one_keypoint"
    ),
    ("behavior/feature_library/track_subsample.py", "drop_nan"): (
        f"{_HERE}::test_the_subsampling_reads_drop_nan"
    ),
    ("tracking/ops/train_litpose.py", "device"): (
        "tests/test_train_litpose.py::test_the_device_reaches_the_trainer"
    ),
    ("tracking/ops/infer.py", "dor"): (
        "tests/test_ultralytics_wire_contract.py"
        "::test_the_point_inference_request_carries_dor"
    ),
}


def _declares_unwired(annotation: ast.expr) -> bool:
    """Whether *annotation* declares a ``Declared(...)`` passing ``unwired=``."""
    for node in ast.walk(annotation):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else ""
        if isinstance(function, ast.Attribute):
            name = function.attr
        if name != "Declared":
            continue
        if any(keyword.arg == "unwired" for keyword in node.keywords):
            return True
    return False


def _unwired_declarations() -> set[tuple[str, str]]:
    """Every ``unwired=`` field under the package, as (module path, field name).

    The external-environment trees are skipped. A program under one of them
    takes no import from ``mosaic`` at all, so it cannot reach ``Declared``, and
    each is where the user builds a virtualenv holding thousands of files.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(_SRC.rglob("*.py")):
        if runs_in_an_external_environment(path, _SRC):
            continue
        tree = source_tree(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue
            target = node.target
            if isinstance(target, ast.Name) and _declares_unwired(node.annotation):
                found.add((path.relative_to(_SRC).as_posix(), target.id))
    return found


def _functions_defined_in(test_file: str) -> set[str]:
    """The test function names *test_file* defines, relative to the repository."""
    path = Path(__file__).resolve().parent.parent / test_file
    tree = source_tree(path)
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_every_unwired_field_is_recorded_by_a_test() -> None:
    """No ``unwired=`` field exists without a test naming the wiring it lacks.

    ``core/params.py`` promises that record, and prose does not fail. This does,
    in both directions. A new declaration with no entry means a field publishing
    ``x-mosaic-unwired`` that nothing will notice being wired. An entry with no
    declaration means a field that was wired and retired, leaving a row here
    pointing at a test that no longer records anything.

    The registry also names each recording test, and the test has to exist. A
    renamed one would leave the row resolving to nothing.
    """
    assert _unwired_declarations() == set(_RECORDED_BY)

    for location in sorted(set(_RECORDED_BY.values())):
        test_file, _, test_name = location.partition("::")
        assert test_name in _functions_defined_in(test_file), location
