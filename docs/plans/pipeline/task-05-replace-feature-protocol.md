# Task 5: Replace Feature protocol and add `from_path()` to ArtifactSpec

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the old `Feature` protocol (12+ methods) with the new one (4 methods + 4 attributes). Add `from_path()` to `ArtifactSpec` and all 12 subclasses for type-safe artifact loading without Dataset dependency.

**Phase:** C (Protocol Transition -- clean break, all Phase C tasks land together)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Tasks 1-4

---

## Files

- Modify: `src/mosaic/core/pipeline/types.py`
- Modify: `tests/test_inputs_subclass.py`
- Modify: `src/mosaic/behavior/feature_library/global_tsne.py` (4 ArtifactSpec subclasses)
- Modify: `src/mosaic/behavior/feature_library/global_ward.py` (2 ArtifactSpec subclasses)
- Modify: `src/mosaic/behavior/feature_library/ward_assign.py` (1 ArtifactSpec subclass)
- Modify: `src/mosaic/behavior/feature_library/global_kmeans.py` (4 ArtifactSpec subclasses)
- Modify: `src/mosaic/behavior/feature_library/spec.py` (FeatureLabelsSource -- 1 ArtifactSpec subclass)

---

## Terminology note

> Task 4 renamed `KeyData` -> `EntryData`, `load_key_data` -> `load_entry_data`,
> and `key` -> `entry_key` throughout the codebase. The code below reflects that.

## Step 1: Write failing test for Feature protocol

```python
# Add to tests/test_inputs_subclass.py

from collections.abc import Iterator
from mosaic.core.pipeline.loading import EntryData


class _TestFeature:
    """Minimal feature for protocol testing."""
    name = "new-test"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self):
        self.inputs = self.Inputs(("tracks",))
        self._params = self.Params()

    @property
    def params(self):
        return self._params

    def load_state(self, run_root, artifact_paths):
        return True  # stateless

    def fit(self, inputs):
        pass

    def save_state(self, run_root):
        pass

    def apply(self, entry_key, entry_data):
        import pandas as pd
        return pd.DataFrame({"col": [1.0] * entry_data.features.shape[0]})


def test_feature_protocol():
    from mosaic.core.pipeline.types import Feature
    f: Feature = _TestFeature()
    assert f.scope_dependent is False
    assert f.load_state(Path("/tmp"), {}) is True
```

## Step 2: Run test to verify it fails

Expected: FAIL (`Feature` protocol lacks `scope_dependent`, `load_state`, etc.)

## Step 3: Implement

Delete the old `Feature` protocol from `types.py` and replace it with:

```python
class Feature(Protocol):
    """Feature protocol -- 4 attributes, 4 methods."""

    name: str
    version: str
    parallelizable: bool
    scope_dependent: bool

    @property
    def inputs(self) -> InputsLike: ...
    @property
    def params(self) -> Params: ...

    def load_state(
        self, run_root: Path, artifact_paths: dict[str, Path]
    ) -> bool: ...
    def fit(self, inputs: Iterator[tuple[str, EntryData]]) -> None: ...
    def save_state(self, run_root: Path) -> None: ...
    def apply(self, entry_key: str, entry_data: EntryData) -> pd.DataFrame: ...
```

Note: uses `Iterator` from `collections.abc` and `EntryData` from `loading`. Add appropriate TYPE_CHECKING import for `EntryData` to avoid circular imports.

Add `from_path()` to `ArtifactSpec`:

```python
class ArtifactSpec(Result[str], Generic[L]):
    load: L
    pattern: str = ""

    # ... existing _derive_pattern, from_result ...

    def from_path(self, path: Path) -> object:
        """Load artifact from a resolved path. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__qualname__}.from_path() not implemented"
        )

    @model_validator(mode="after")
    def _validate_from_path(self) -> Self:
        """Concrete specs (feature != '') must implement from_path."""
        if self.feature != "" and "from_path" not in type(self).__dict__:
            msg = f"{type(self).__qualname__} must implement from_path(path: Path)"
            raise TypeError(msg)
        return self
```

## Step 4: Run tests

Run: `cd mosaic && uv run pytest tests/test_inputs_subclass.py -v`
Expected: All pass

## Step 5: Add `from_path()` to all 12 ArtifactSpec subclasses

Each subclass gets a concrete `from_path()` that delegates to the appropriate loader. Examples:

In `global_tsne.py`:
```python
class TemplatesArtifact(ArtifactSpec[NpzLoadSpec]):
    feature: str = "global-tsne"
    pattern: str = "global_templates_features.npz"
    load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="templates"))

    def from_path(self, path: Path) -> np.ndarray:
        from mosaic.core.pipeline.loading import _load_array_from_spec
        arr, _ = _load_array_from_spec(path, self.load)
        if arr is None:
            msg = f"Failed to load templates from {path}"
            raise FileNotFoundError(msg)
        return arr
```

For joblib artifacts:
```python
class EmbeddingArtifact(ArtifactSpec[JoblibLoadSpec]):
    feature: str = "global-tsne"
    pattern: str = "global_opentsne_embedding.joblib"
    load: JoblibLoadSpec = Field(default_factory=lambda: JoblibLoadSpec(key=None))

    def from_path(self, path: Path) -> object:
        import joblib
        obj = joblib.load(path)
        return obj if self.load.key is None else obj[self.load.key]
```

All 12 subclasses across: `global_tsne.py` (4), `global_ward.py` (2), `ward_assign.py` (1), `global_kmeans.py` (4), `spec.py:FeatureLabelsSource` (1).

## Step 6: Run full test suite

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: 221 + new tests passed. Existing ArtifactSpec subclasses now validate from_path() at instantiation time.

## Step 7: Commit

```
feat: replace Feature protocol and add from_path() on ArtifactSpec subclasses
```
