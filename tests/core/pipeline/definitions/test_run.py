"""Running a loaded definition: its storage, its identity and its workers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.definitions.define import (
    load_feature_definition,
)
from mosaic.core.pipeline.definitions.scope import (
    DefinitionScope,
    current_scope,
    definition_scope,
)
from mosaic.core.pipeline.definitions.sidecar import DefinitionRef
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory.params import read_run_params
from mosaic.core.pipeline.run import compute_run_id, run_feature
from mosaic.core.pipeline.types import Result
from tests.helpers import (
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    PlainFeature,
    add_track_sequences,
    defined_dataset,
    make_dataset,
    run_fresh,
)

WORKERS: Final = 2


def _root(ds: Dataset, result: Result[str]) -> Path:
    """The run root of *result*."""
    assert result.run_id is not None
    return feature_run_root(ds, result.feature, result.run_id)


def _written(ds: Dataset, result: Result[str], name: str) -> Path:
    """The run root of *result*, after checking it is the definition's storage."""
    assert result.feature == f"def.{name}__from__tracks"
    root = _root(ds, result)
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

    root = _root(ds, result)
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

    root = _root(ds, result)
    module_name = f"mosaic_definitions.{GLOBAL_NAME.replace('-', '_')}__{digest}"
    assert module_name.encode() in (root / "mean.pkl").read_bytes()
    fitted = float(feature.apply(_POINT)["centered"].iloc[0])

    reported = run_fresh(
        _FRESH_PROGRAM, args=[str(ds.manifest_path), slug, str(root)]
    ).split()

    assert reported[:2] == [module_name, "Mean"]
    assert float(reported[2]) == fitted
