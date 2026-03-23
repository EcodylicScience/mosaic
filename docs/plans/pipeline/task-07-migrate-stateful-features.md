# Task 7: Migrate stateful and global features to new protocol

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert features that have `needs_fit() = True` (plus three stateless stragglers missed by Task 6) to the new protocol. This includes per-sequence-with-fit features and global features that currently bypass the transform phase.

**Phase:** C (Protocol Transition -- clean break, all Phase C tasks land together)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Tasks 5-6

---

## Protocol reference (implemented in Task 5-6)

```python
class Feature(Protocol):
    name: str
    version: str
    parallelizable: bool
    scope_dependent: bool

    def load_state(self, run_root: Path, artifact_paths: dict[str, Path]) -> bool: ...
    def fit(self, inputs: Iterator[tuple[str, pd.DataFrame]]) -> None: ...
    def save_state(self, run_root: Path) -> None: ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

- `load_state()` returns True to skip fitting (cached model exists), False to proceed to fit().
- `fit()` receives `Iterator[tuple[str, pd.DataFrame]]` -- pairs of (entry_key, DataFrame).
- `apply()` receives a single DataFrame per sequence, returns a DataFrame.
- Features that need numpy arrays extract them from DataFrames (see extraction pattern below).
- `artifact_paths` maps Params field names to resolved filesystem paths (Task 9).

### Column name conventions

All migrated features import `COLUMNS as C` from `.spec` and use `C.*` for
standard column access. Never hardcode standard column names as string literals.

```python
from .spec import COLUMNS as C

# Standard columns via C.*:
C.frame_col   # "frame"
C.time_col    # "time"
C.id_col      # "id"
C.seq_col     # "sequence"
C.group_col   # "group"
C.x_col       # "X"
C.y_col       # "Y"
```

Pair identity columns `"id1"` and `"id2"` are NOT in `Columns` -- they are
pair-specific output columns from the loading infrastructure. Use them as
string literals (they are feature-specific, not configurable).

### Column validation pattern

Replace all manual column checks with `ensure_columns` from helpers:

```python
from .helpers import ensure_columns

# Before (manual -- do not use):
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing cols: {missing}")

# After:
ensure_columns(df, [C.id_col, C.seq_col, order_col] + pose_cols)
```

`resolve_order_col(df)` already validates that a temporal ordering column
exists, so there is no need to separately check for `C.frame_col` or
`C.time_col` after calling it.

### Numeric feature extraction pattern

Global features that operate on numeric matrices extract them from DataFrames:

```python
from .spec import COLUMNS as C

_META_COLS = frozenset({
    C.frame_col, C.time_col, C.id_col,
    "id1", "id2", "id_a", "id_b", "id_A", "id_B",
    C.seq_col, C.group_col, "entity_level", "perspective", "fps",
})

def _extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df.select_dtypes(include="number")
    drop = [c for c in _META_COLS if c in numeric.columns]
    if drop:
        numeric = numeric.drop(columns=drop)
    return numeric.to_numpy(dtype=np.float32)
```

This replaces the old `kd.features` access. The old `kd.frames`, `kd.id1`, `kd.id2`
accesses become direct column access on the DataFrame: `df[C.frame_col]`,
`df["id1"]`, etc.

---

## IMPORTANT: Sequential Migration

Each complex stateful feature below must be migrated one at a time. Before implementing each: extract pseudocode of its algorithmic flow, map it to the current code, then build the clean implementation from that reference. No logic lost.

Do not attempt to migrate multiple complex features in parallel or in a single batch.

---

## Stateless stragglers (3) -- missed by Task 6

These features have `needs_fit() = False` and are stateless. They follow the same
simple migration pattern as Task 6:

- `pair_wavelet.py` (PairWavelet)
- `ffgroups.py` (FFGroups)
- `ffgroups_metrics.py` (FFGroupsMetrics)

### Migration pattern (same as Task 6)

```python
scope_dependent = False
parallelizable = True  # already True for all 3

def load_state(self, run_root, artifact_paths):
    return True  # stateless, always skip fit

def fit(self, inputs):
    pass  # never called

def save_state(self, run_root):
    pass  # nothing to save

def apply(self, df):
    # Body of current transform(df) -- no changes to the logic
    ...
```

Remove: `bind_dataset`, `set_scope`, `needs_fit`, `supports_partial_fit`,
`finalize_fit`, `save_model`, `load_model`, `_ds`, `_scope`, old `fit`, old `transform`.

Replace manual column checks with `ensure_columns`:
- FFGroupsMetrics (lines 114-116): `ensure_columns(df, required)`.
- PairWavelet (line 170): `ensure_columns(df, ["perspective"])`.

Update `test_ffgroups.py`: change `transform()` calls to `apply()`.

---

## Per-sequence-with-fit features (2)

- `pairposedistancepca.py` (PairPoseDistancePCA) -- IncrementalPCA with partial_fit
- `global_ward.py` (GlobalWardClustering) -- scipy linkage on artifact matrix

### PairPoseDistancePCA

```python
scope_dependent = True
parallelizable = True

def load_state(self, run_root, artifact_paths):
    self._ipca = IncrementalPCA(
        n_components=self.params.n_components,
        batch_size=self.params.batch_size,
    )
    self._fitted = False
    self._tri_i = None
    self._tri_j = None
    self._feat_len = None
    path = run_root / "model.joblib"
    if path.exists():
        bundle = joblib.load(path)
        self._ipca = bundle["ipca"]
        self._tri_i = bundle.get("tri_i")
        self._tri_j = bundle.get("tri_j")
        self._feat_len = bundle.get("feat_len")
        self.params = self.Params.model_validate(bundle.get("params", {}))
        self._fitted = True
        return True
    return False

def fit(self, inputs):
    for entry_key, df in inputs:
        for batch, _, _ in self._feature_batches(df, for_fit=True):
            if batch.size == 0:
                continue
            self._ipca.partial_fit(batch)
            self._fitted = True

def save_state(self, run_root):
    if not self._fitted:
        return
    run_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "ipca": self._ipca,
            "params": self.params.model_dump(),
            "tri_i": self._tri_i,
            "tri_j": self._tri_j,
            "feat_len": self._feat_len,
        },
        run_root / "model.joblib",
    )

def apply(self, df):
    # Body of current transform(df) -- no signature change needed,
    # _feature_batches already operates on a DataFrame.
    if not self._fitted:
        msg = "pair-posedistance-pca: not fitted yet"
        raise RuntimeError(msg)
    # ... existing transform body (builds PC scores from batches) ...
```

Remove: `bind_dataset`, `set_scope`, `needs_fit`, `supports_partial_fit`,
`finalize_fit`, `_ds`, `_scope`, old `fit(X_iter)`, old `transform`, old
`save_model`, old `load_model`.

Replace `self._clean_one_animal()` with `clean_animal_track()` from helpers
(already extracted in Task 6, identical logic).

Replace manual column check in `_prep_pairs()` (lines 218-220) with
`ensure_columns(df, [C.id_col, C.seq_col, order_col] + pose_cols)`.

Clean up deprecated typing imports: `Any`, `Dict`, `List`, `Optional`, `Tuple`.

### GlobalWardClustering

GlobalWard has two modes: artifact-only (templates from Params) and stacked-features
(per-sequence features from Result inputs vstacked). Both produce a linkage matrix.

The feature has `output_type = "global"` -- its apply() returns a 1-row marker
DataFrame (same as current transform). No per-sequence outputs.

```python
scope_dependent = True
parallelizable = False

def load_state(self, run_root, artifact_paths):
    self._Z = None
    self._X_shape = None
    # Load templates artifact for artifact-only mode
    if "templates" in artifact_paths:
        self._artifact_matrix = self.params.templates.from_path(
            artifact_paths["templates"]
        )
    else:
        self._artifact_matrix = None
    # Check for cached model
    path = run_root / "model.joblib"
    if path.exists():
        bundle = joblib.load(path)
        self._Z = bundle.get("linkage_matrix")
        ns = bundle.get("n_samples")
        nf = bundle.get("n_features")
        self._X_shape = (ns, nf) if ns is not None and nf is not None else None
        return True
    return False

def fit(self, inputs):
    feature_inputs = self.inputs.feature_inputs
    if feature_inputs:
        # Stacked-features mode: consume iterator, extract numeric, vstack
        blocks = []
        for entry_key, df in inputs:
            X = _extract_feature_matrix(df)
            if X.size > 0:
                blocks.append(X.astype(np.float64))
        if not blocks:
            msg = "[global-ward] Result inputs produced no usable matrices."
            raise RuntimeError(msg)
        X = np.vstack(blocks)
    else:
        # Artifact-only mode: templates loaded in load_state
        if self._artifact_matrix is None:
            msg = "[global-ward] No templates artifact and no feature inputs."
            raise RuntimeError(msg)
        X = self._artifact_matrix.astype(np.float64, copy=False)

    if X.ndim != 2 or X.shape[0] < 2:
        msg = f"[global-ward] Need 2D matrix with >=2 samples; got shape={X.shape}"
        raise ValueError(msg)

    self._Z = _sch_linkage(X, method=self.params.method)
    self._X_shape = tuple(X.shape)

def save_state(self, run_root):
    # Existing save_model logic (joblib + npz backup)
    ...

def apply(self, df):
    # 1-row marker (existing transform logic)
    ns, nf = self._X_shape or (0, 0)
    return pd.DataFrame([{
        "linkage_method": self.params.method,
        "n_samples": int(ns),
        "n_features": int(nf),
        "model_file": "model.joblib",
    }])
```

Remove: `bind_dataset`, `set_scope`, `needs_fit`, `supports_partial_fit`,
`finalize_fit`, `_ds`, `_scope`, `_marker_written`, old `fit`, old `transform`,
old `save_model`, old `load_model`, `_load_artifact_matrix` (replaced by
`load_state` + artifact_paths).

---

## Global features (4) -- skip_transform_phase features

- `global_tsne.py` (GlobalTSNE)
- `global_kmeans.py` (GlobalKMeansClustering)
- `ward_assign.py` (WardAssignClustering)
- `kpms_apply.py` (KpmsApply)

These currently bypass the transform phase entirely and write outputs during
`fit()`/`save_model()`. Under the new protocol, per-sequence output moves to
`apply()`, and artifact loading moves to `load_state()`.

For each: remove `skip_transform_phase`, `set_run_root`, `get_additional_index_rows`,
`_persist_mapped_coords`, `_write_sequence_outputs`, `_append_index_row`,
`_additional_index_rows`, `bind_dataset`, `set_scope`, `_ds`, `_scope`.
Add `load_state`, `fit(iterator)`, `save_state`, `apply(df)`.

---

## GlobalTSNE migration

GlobalTSNE uses the openTSNE advanced API (`prepare_partial` + `optimize`) for
per-sequence mapping -- NOT simple `embedding.transform()`. The advanced API is
required because the embedding is built with custom affinity (FAISS kNN) and PCA
initialization. The `prepare_partial` path respects these settings; `.transform()`
does not.

The current `fit()` does two full passes over the data:
- **Pass 1:** Sample for scaler fitting and farthest-first template selection.
- **Pass 2:** Map every sequence through the embedding (`_map_sequences_streaming`).

Under the new protocol, Pass 2 moves out of `fit()` and into per-sequence
`apply()` calls. Pass 1 stays in `fit()` because it needs a sampling pass across
all sequences before the embedding can be trained.

`parallelizable = False` because `prepare_partial` creates temporary embedding
objects that reference the main embedding's affinity structure, and the aggressive
`gc.collect()` between chunks prevents openTSNE memory buildup. Concurrent
`apply()` calls would defeat this.

```python
scope_dependent = True
parallelizable = False  # prepare_partial shares embedding state, needs sequential GC

def load_state(self, run_root, artifact_paths):
    self._embedding = None
    self._scaler = None
    self._templates = None
    self._template_indices = None

    # Load upstream artifacts from artifact_paths (replaces
    # _prepare_reuse_artifacts, _load_embedding, _load_scaler,
    # _load_templates which all used self._ds).
    if "reuse_embedding" in artifact_paths and self.params.reuse_embedding is not None:
        self._embedding = self.params.reuse_embedding.from_path(
            artifact_paths["reuse_embedding"]
        )
    if "scaler" in artifact_paths and self.params.scaler is not None:
        self._scaler = self.params.scaler.from_path(
            artifact_paths["scaler"]
        )
    if "artifact" in artifact_paths and self.params.artifact is not None:
        self._templates = self.params.artifact.from_path(
            artifact_paths["artifact"]
        )

    # Check for own cached state
    path = run_root / "global_opentsne_embedding.joblib"
    if path.exists():
        bundle = joblib.load(path)
        self._embedding = bundle["embedding"]
        self._scaler = bundle.get("scaler", self._scaler)
        return True

    # map_existing_inputs: reuse artifacts were loaded above, no fitting
    # needed. Return False so save_state() persists the bundle under this
    # run_id (first run creates the cache; second run hits the check above).
    if self.params.map_existing_inputs:
        if self._embedding is None:
            msg = "map_existing_inputs=True requires params.reuse_embedding"
            raise ValueError(msg)
        if self._scaler is None:
            msg = "Reusable embedding missing scaler; provide params.scaler"
            raise ValueError(msg)
        return False

    return False

def fit(self, inputs):
    """Pass 1 only: sample for scaler + templates, then fit embedding.

    The per-sequence mapping (Pass 2) now happens in apply().

    When map_existing_inputs=True, this is a no-op -- the reuse
    embedding and scaler were already loaded in load_state() from
    artifact_paths. save_state() will persist them under this run_id.
    """
    if self.params.map_existing_inputs:
        return

    # --- Pass 1: stream through inputs, sampling for scaler and templates ---
    r_scaler = self.params.r_scaler
    total_templates = self.params.total_templates
    scaler_samples = []
    template_samples = []
    n_keys = 0

    for entry_key, df in inputs:
        n_keys += 1
        X = _extract_feature_matrix(df)
        if X.shape[0] == 0:
            continue
        quota = max(self.params.pre_quota_per_key, total_templates // max(1, n_keys))
        samples_per_key = max(1000, r_scaler // max(1, n_keys))

        take_scaler = min(X.shape[0], samples_per_key)
        idx_s = self._rng.choice(X.shape[0], size=take_scaler, replace=False)
        scaler_samples.append(X[idx_s].copy())

        take_templ = min(X.shape[0], quota * 3)
        idx_t = self._rng.choice(X.shape[0], size=take_templ, replace=False)
        template_samples.append(X[idx_t].copy())

    if not scaler_samples:
        msg = "No combined feature frames after alignment."
        raise RuntimeError(msg)

    # Fit scaler
    combined = np.vstack(scaler_samples)
    del scaler_samples
    if combined.shape[0] > r_scaler:
        idx = self._rng.choice(combined.shape[0], size=r_scaler, replace=False)
        combined = combined[idx]
    self._scaler = StandardScaler().fit(combined)
    del combined

    # Farthest-first template selection
    scaled_pre = np.vstack([self._scaler.transform(s) for s in template_samples])
    del template_samples
    sel = [int(self._rng.integers(0, scaled_pre.shape[0]))]
    d2 = np.sum((scaled_pre - scaled_pre[sel[0]]) ** 2, axis=1)
    while len(sel) < min(total_templates, scaled_pre.shape[0]):
        i = int(np.argmax(d2))
        sel.append(i)
        d2 = np.minimum(d2, np.sum((scaled_pre - scaled_pre[i]) ** 2, axis=1))
    templates = scaled_pre[np.array(sel)]
    self._templates = templates
    self._template_indices = np.array(sel)
    del scaled_pre, d2

    # Fit openTSNE on templates (affinity, init, embedding -- unchanged)
    self._embedding = self._fit_embedding(templates)

def save_state(self, run_root):
    run_root.mkdir(parents=True, exist_ok=True)
    # Save templates npz
    if self._templates is not None:
        np.savez_compressed(
            run_root / "global_templates_features.npz",
            templates=self._templates,
        )
    # Save template coords npz
    if self._embedding is not None:
        Y_templates = np.asarray(self._embedding)
        np.savez_compressed(
            run_root / "global_tsne_templates.npz",
            Y=Y_templates,
            sel=self._template_indices or np.array([], int),
        )
    # Save embedding + scaler bundle
    joblib.dump(
        {"embedding": self._embedding, "scaler": self._scaler,
         "params": self.params.model_dump()},
        run_root / "global_opentsne_embedding.joblib",
    )
    # NOTE: per-sequence parquets are NOT written here.
    # They are produced by apply() and written by the pipeline.

def apply(self, df):
    """Map one sequence through the embedding using prepare_partial + optimize.

    Uses chunked streaming (params.map_chunk) to control memory. Each chunk
    gets its own prepare_partial call with aggressive GC between chunks to
    prevent openTSNE memory buildup.
    """
    import pyarrow as pa

    embedding = self._embedding
    scaler = self._scaler
    chunk_size = self.params.map_chunk
    partial_k = self.params.partial_k
    perplexity = self.params.perplexity
    partial_iters = self.params.partial_iters
    partial_lr = self.params.partial_lr

    X = _extract_feature_matrix(df)
    scaled = scaler.transform(X).astype(np.float32, copy=False)
    del X

    # Chunked mapping -- same logic as current _map_sequences_streaming
    Y_seq = np.empty((scaled.shape[0], 2), dtype=np.float32)
    for j in range(0, scaled.shape[0], chunk_size):
        chunk = scaled[j : j + chunk_size]
        part = embedding.prepare_partial(
            chunk, initialization="median", k=partial_k, perplexity=perplexity
        )
        part.optimize(
            n_iter=partial_iters, learning_rate=partial_lr, exaggeration=2.0,
            momentum=0.0, inplace=True, verbose=False,
        )
        coords = np.asarray(part).astype(np.float32, copy=False).copy()
        Y_seq[j : j + coords.shape[0], :] = coords
        # Aggressively free openTSNE internals
        if hasattr(part, "affinities"):
            del part.affinities
        del part, coords
        if (j // chunk_size) % 5 == 4:
            gc.collect()

    del scaled
    gc.collect()
    pa.default_memory_pool().release_unused()

    # Build output DataFrame from the input's identity columns
    data = {
        "tsne_x": Y_seq[:, 0],
        "tsne_y": Y_seq[:, 1],
    }
    if C.frame_col in df.columns:
        data[C.frame_col] = df[C.frame_col].to_numpy()
    else:
        data[C.frame_col] = np.arange(Y_seq.shape[0], dtype=np.int64)
    if "id1" in df.columns:
        data["id1"] = df["id1"].values
    if "id2" in df.columns:
        data["id2"] = df["id2"].values
    return pd.DataFrame(data)
```

The `_fit_embedding(templates)` helper extracts the existing affinity + init +
TSNEEmbedding construction and two-phase optimize (lines 353-386 of the current
`fit()`). No logic changes -- just factored out of the monolithic `fit()`.

Remove: `bind_dataset`, `set_scope`, `set_run_root`, `get_additional_index_rows`,
`needs_fit`, `supports_partial_fit`, `finalize_fit`, `_ds`, `_scope`, `_run_root`,
`_additional_index_rows`, `_mapped_coords`, `_keys`,
`_persist_mapped_coords`, `_append_index_row`, `_discover_existing_coord_rows`,
`_map_sequences_streaming`, `_prepare_reuse_artifacts`, `_load_embedding`,
`_load_scaler`, `_load_templates`,
old `fit`, old `transform`, old `save_model`, old `load_model`.

---

## GlobalKMeans migration

GlobalKMeans fits KMeans on an artifact matrix (templates), then optionally assigns
cluster labels to per-sequence features. The current `transform()` already
implements per-DataFrame cluster assignment and moves directly to `apply()`.

The artifact loading (templates, scaler) moves from fit() to load_state().
The per-sequence assignment (currently done in fit() via StreamingFeatureHelper)
moves to apply() -- the pipeline calls apply() per sequence.

```python
scope_dependent = True
parallelizable = False  # fitting is global

def load_state(self, run_root, artifact_paths):
    self._kmeans = None
    self._fit_dim = None
    self._fit_columns = None
    self._artifact_labels = None
    self._artifact_matrix = None
    # Load templates artifact
    if "templates" in artifact_paths:
        self._artifact_matrix = self.params.templates.from_path(
            artifact_paths["templates"]
        )
    # Load scaler if specified
    self._scaler = None
    if "scaler" in artifact_paths and self.params.scaler is not None:
        self._scaler = self.params.scaler.from_path(artifact_paths["scaler"])
    # Check for cached model
    path = run_root / "model.joblib"
    if path.exists():
        bundle = joblib.load(path)
        self._kmeans = bundle["kmeans"]
        self._fit_dim = int(bundle.get("fit_dim") or 0)
        self._fit_columns = bundle.get("fit_columns")
        return True
    return False

def fit(self, inputs):
    """Fit KMeans on templates (loaded in load_state).

    Does NOT iterate the input data -- the templates artifact is
    the training data, not the per-sequence features.
    """
    if self._artifact_matrix is None:
        msg = "[global-kmeans] No templates artifact loaded."
        raise RuntimeError(msg)
    X = self._artifact_matrix
    self._fit_dim = X.shape[1]

    if X.shape[0] < self.params.k:
        msg = f"Not enough samples: n={X.shape[0]} < k={self.params.k}"
        raise ValueError(msg)

    KMeansCls = _get_kmeans_class(self.params.device)
    self._kmeans = KMeansCls(
        n_clusters=self.params.k,
        n_init=self.params.n_init,
        random_state=self.params.random_state,
        max_iter=self.params.max_iter,
    ).fit(X)

    if self.params.label_artifact_points:
        self._artifact_labels = self._kmeans.predict(X)

def save_state(self, run_root):
    run_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kmeans": self._kmeans,
            "fit_dim": int(self._fit_dim or 0),
            "fit_columns": self._fit_columns,
            "version": self.version,
            "params": self.params.model_dump(),
        },
        run_root / "model.joblib",
    )
    centers = np.asarray(self._kmeans.cluster_centers_, dtype=np.float32)
    np.savez_compressed(run_root / "cluster_centers.npz", centers=centers)
    if self._artifact_labels is not None:
        np.savez_compressed(
            run_root / "artifact_labels.npz", labels=self._artifact_labels
        )
        uniq, cnt = np.unique(self._artifact_labels, return_counts=True)
        pd.DataFrame(
            {"cluster": uniq.astype(int), "count": cnt.astype(int)}
        ).to_parquet(run_root / "cluster_sizes.parquet", index=False)
    # NOTE: per-sequence parquets are NOT written here.
    # They are produced by apply() and written by the pipeline.

def apply(self, df):
    """Assign cluster labels to a per-sequence DataFrame.

    Body is the current transform() method -- already operates on
    a DataFrame with named columns. No logic changes.
    """
    if self._kmeans is None or self._fit_dim is None:
        msg = "GlobalKMeansClustering not fitted yet."
        raise RuntimeError(msg)

    if self._fit_columns:
        ensure_columns(df, self._fit_columns)
        A = df[self._fit_columns].to_numpy(dtype=np.float32, copy=False)
    else:
        A = _extract_feature_matrix(df)
        if A.shape[1] != self._fit_dim:
            msg = (
                f"Feature dim mismatch: fitted on {self._fit_dim} columns, "
                f"got {A.shape[1]}"
            )
            raise ValueError(msg)

    mask = np.isfinite(A).all(axis=1)
    labels = np.full(A.shape[0], -1, dtype=np.int32)
    if mask.any():
        labels[mask] = self._kmeans.predict(A[mask])

    return pd.DataFrame({
        C.frame_col: df[C.frame_col].astype(int, errors="ignore")
        if C.frame_col in df.columns
        else np.arange(len(df), dtype=int),
        "cluster": labels,
    })
```

Remove: `bind_dataset`, `set_scope`, `get_additional_index_rows`, `needs_fit`,
`supports_partial_fit`, `finalize_fit`, `skip_transform_phase`, `_ds`, `_scope`,
`_additional_index_rows`, `_assign_labels`, `_assign_frames`, `_assign_id1`,
`_assign_id2`, `_assign_entity_level`, `_fit_artifact_info`,
`_load_npz_matrix`, `_load_parquet_matrix`, `_load_artifact_matrix`,
old `fit`, old `transform`, old `save_model`, old `load_model`, old `partial_fit`.

---

## WardAssign migration

WardAssign loads a Ward linkage matrix and templates artifact, cuts the linkage
to derive cluster centroids, fits a NearestNeighbors model on centroids, then
assigns each per-sequence feature vector to the nearest centroid.

The centroid computation stays in fit(). The per-sequence NN assignment (currently
done in the fit() loop via StreamingFeatureHelper) moves to apply().

```python
scope_dependent = True
parallelizable = True

def load_state(self, run_root, artifact_paths):
    self._Z = None
    self._templates = None
    self._cluster_ids = None
    self._assign_nn = None
    self._scaler = None
    # Load Ward linkage from artifact_paths
    if "ward" in artifact_paths:
        bundle = joblib.load(artifact_paths["ward"])
        self._Z = bundle.get("linkage_matrix")
    # Load templates from artifact_paths
    if "templates" in artifact_paths:
        self._templates = self.params.templates.from_path(
            artifact_paths["templates"]
        )
    # Load scaler if specified
    if "scaler" in artifact_paths and self.params.scaler is not None:
        self._scaler = self.params.scaler.from_path(artifact_paths["scaler"])
    # Check for cached NN model
    path = run_root / "model.joblib"
    if path.exists():
        bundle = joblib.load(path)
        self._assign_nn = bundle.get("assign_nn")
        self._cluster_ids = bundle.get("cluster_ids")
        self._scaler = bundle.get("scaler", self._scaler)
        return True
    return False

def fit(self, inputs):
    """Compute cluster centroids and fit NearestNeighbors.

    Does NOT iterate input data -- the centroid computation uses
    the templates and linkage loaded in load_state().
    """
    if self._Z is None:
        msg = "[ward-assign] Ward linkage not loaded."
        raise RuntimeError(msg)
    if self._templates is None:
        msg = "[ward-assign] Templates not loaded."
        raise RuntimeError(msg)

    n_clusters = self.params.n_clusters
    labels_templates = fcluster(self._Z, n_clusters, criterion="maxclust")
    unique_labels = np.unique(labels_templates)
    centroids = []
    for cluster_id in unique_labels:
        mask = labels_templates == cluster_id
        if mask.any():
            centroids.append(self._templates[mask].mean(axis=0))
    centroids = np.vstack(centroids)
    self._cluster_ids = unique_labels.astype(int)
    self._assign_nn = NearestNeighbors(n_neighbors=1).fit(centroids)

    # Free templates and linkage -- no longer needed
    del self._templates, self._Z
    self._templates = None
    self._Z = None

def save_state(self, run_root):
    run_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "assign_nn": self._assign_nn,
            "cluster_ids": self._cluster_ids,
            "scaler": self._scaler,
            "params": self.params.model_dump(),
        },
        run_root / "model.joblib",
    )

def apply(self, df):
    """Assign cluster labels to a per-sequence DataFrame."""
    X = _extract_feature_matrix(df)

    if self._scaler is not None:
        X = self._scaler.transform(X)

    idxs = self._assign_nn.kneighbors(X, return_distance=False)
    labels = self._cluster_ids[idxs.ravel()]

    data = {
        C.frame_col: df[C.frame_col].to_numpy(dtype=np.int64)
        if C.frame_col in df.columns
        else np.arange(len(df), dtype=np.int64),
        "cluster": labels.astype(np.int32),
    }
    if "id1" in df.columns:
        data["id1"] = df["id1"].values
    if "id2" in df.columns:
        data["id2"] = df["id2"].values
    return pd.DataFrame(data)
```

Remove: `bind_dataset`, `set_scope`, `set_run_root`, `get_additional_index_rows`,
`needs_fit`, `supports_partial_fit`, `finalize_fit`, `skip_transform_phase`,
`skip_existing_outputs`, `_ds`, `_scope`, `_run_root`, `_additional_index_rows`,
`_processed_sequences`, `_write_sequence_outputs`,
old `fit`, old `transform`, old `save_model`, old `load_model`, old `partial_fit`.

---

## KpmsApply migration

KpmsApply runs keypoint-MoSeq inference via subprocess. The subprocess processes
ALL tracks in a single batch, so the work cannot be decomposed into per-sequence
apply() calls in the usual sense. Instead:

- `fit()` iterates the input data, serializes tracks to disk, runs the subprocess,
  and parses results into a lookup dict `{entry_key: pd.DataFrame}`.
- `apply()` returns the pre-computed DataFrame for the given sequence.

The track serialization currently uses `_collect_and_serialize_tracks(self._ds, ...)`
which accesses the dataset directly. Under the new protocol, fit() receives
tracks via the iterator and must serialize them itself.

The fitted model path (from kpms-fit) moves to `load_state()` via `artifact_paths`.

```python
scope_dependent = True
parallelizable = False  # subprocess batch, sequential apply

def load_state(self, run_root, artifact_paths):
    self._run_root = run_root
    self._model_dir = None
    self._results = {}
    # Locate fitted kpms model from artifact_paths
    if "kpms_fit" in artifact_paths:
        self._model_dir = artifact_paths["kpms_fit"]
    # Check for cached results
    processed_path = run_root / "_kpms_output" / "processed_recordings.json"
    if processed_path.exists():
        self._results = self._load_cached_results(run_root / "_kpms_output")
        return bool(self._results)
    return False

def fit(self, inputs):
    if self._model_dir is None:
        msg = "[kpms-apply] kpms-fit model path not resolved."
        raise RuntimeError(msg)

    # 1. Serialize tracks from iterator to disk
    data_dir = self._run_root / "_kpms_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for entry_key, df in inputs:
        self._serialize_one_sequence(entry_key, df, data_dir)

    # 2. Write apply config JSON
    config_path = self._run_root / "kpms_apply_config.json"
    config_path.write_text(json.dumps(
        {"num_iters_apply": self.params.num_iters_apply}, indent=2
    ))

    # 3. Run kpms_runner.py apply in subprocess
    output_dir = self._run_root / "_kpms_output"
    _run_kpms_subprocess(self.params.kpms_python, "apply", [
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--model-dir", str(self._model_dir),
        "--config", str(config_path),
        *(["--batch-size", str(self.params.apply_batch_size)]
          if self.params.apply_batch_size else []),
    ], label="kpms-apply")

    # 4. Parse results into lookup dict
    self._results = self._load_cached_results(output_dir)

def save_state(self, run_root):
    run_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "params": self.params.model_dump(),
            "version": self.version,
        },
        run_root / "model.joblib",
    )

def apply(self, df):
    """Return pre-computed syllable labels for this sequence."""
    # Derive entry_key from the DataFrame's group/sequence columns
    group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
    sequence = str(df[C.seq_col].iloc[0]) if C.seq_col in df.columns else ""
    entry_key = make_entry_key(group, sequence)
    if entry_key in self._results:
        return self._results[entry_key]
    return pd.DataFrame(columns=[C.frame_col, "syllable"])
```

The `_serialize_one_sequence()` helper replaces `_collect_and_serialize_tracks()`.
It extracts pose columns from the DataFrame (using `self.params.pose_prefix_x`,
`pose_prefix_y`, `pose_confidence_prefix`) and writes per-recording npz files
to `data_dir`, matching the format expected by `kpms_runner.py`.

The `_load_cached_results()` helper reads `processed_recordings.json` and
`syllables__{name}.npz` files, building the `{entry_key: pd.DataFrame}` lookup.
This is a refactor of the current `_collect_results()`.

Remove: `bind_dataset`, `set_scope`, `set_run_root`, `get_additional_index_rows`,
`needs_fit`, `supports_partial_fit`, `finalize_fit`, `skip_transform_phase`,
`_ds`, `_scope`, `_additional_index_rows`,
old `fit`, old `transform`, old `save_model`, old `load_model`, old `partial_fit`,
`_collect_results` (replaced by `_load_cached_results`).

---

## Step 1: Migrate stateless stragglers (PairWavelet, FFGroups, FFGroupsMetrics)

Same pattern as Task 6. Update `test_ffgroups.py` to call `apply()` instead of
`transform()`.

## Step 2: Run tests

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: 95 passed, 1 pre-existing failure (test_index_csv)

## Step 3: Migrate PairPoseDistancePCA

Replace `_clean_one_animal` with `helpers.clean_animal_track`.
Clean up deprecated typing imports.

## Step 4: Run tests

## Step 5: Migrate GlobalWardClustering

## Step 6: Migrate GlobalKMeans

## Step 7: Migrate WardAssign

## Step 8: Migrate GlobalTSNE

Most complex. Factor out `_fit_embedding()` from the monolithic fit().
Preserve chunked mapping + GC pattern in apply().

## Step 9: Migrate KpmsApply

Reimplement track serialization to work from DataFrames (replace
`_collect_and_serialize_tracks` which used self._ds).

## Step 10: Run tests

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: All pass (same baseline)

## Step 11: Commit

```
feat: migrate stateful and global features to new protocol
```
