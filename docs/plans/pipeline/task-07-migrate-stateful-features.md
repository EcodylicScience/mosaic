# Task 7: Migrate per-sequence-with-fit and global features

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert features that have `needs_fit() = True` to the new protocol. This includes both per-sequence-with-fit features and global features that currently bypass the transform phase.

**Phase:** C (Protocol Transition -- clean break, all Phase C tasks land together)

**Parent plan:** `docs/plans/2026-03-12-pipeline-unification-implementation.md`

**Depends on:** Tasks 5-6

---

## Terminology note

> Task 4 renamed `KeyData` -> `EntryData`, `load_key_data` -> `load_entry_data`,
> and `key` -> `entry_key` throughout the codebase. The code below reflects that.

## IMPORTANT: Sequential Migration

Each complex stateful feature below must be migrated one at a time. Before implementing each: extract pseudocode of its algorithmic flow, map it to the current code, then build the clean implementation from that reference. No logic lost.

Do not attempt to migrate multiple complex features in parallel or in a single batch.

---

## Per-sequence-with-fit features (5)

- `pairposedistancepca.py` (PairPoseDistancePCA)
- `pair_wavelet.py` (PairWavelet)
- `global_ward.py` (GlobalWardClustering) -- has transform, not skip_transform_phase
- `ffgroups.py` (FFGroups)
- `ffgroups_metrics.py` (FFGroupsMetrics)

## Migration pattern

1. `scope_dependent = True`
2. `load_state()`: check for cached model (e.g., `run_root / "model.joblib"`), load artifacts via `artifact_paths`, return True if model exists, False otherwise
3. `fit(inputs: Iterator[tuple[str, EntryData]])`: receives the full input iterator, accumulate or batch as needed
4. `save_state(run_root)`: save model to `run_root / "model.joblib"`
5. `apply(entry_key, entry_data)`: replaces `transform()`, per-sequence computation using fitted model

## Example: PairPoseDistancePCA

```python
scope_dependent = True
parallelizable = True

def load_state(self, run_root, artifact_paths):
    self._pca = None
    path = run_root / "model.joblib"
    if path.exists():
        self._pca = joblib.load(path)
        return True
    return False

def fit(self, inputs):
    from sklearn.decomposition import IncrementalPCA
    self._pca = IncrementalPCA(n_components=self.params.n_components)
    for entry_key, entry_data in inputs:
        for batch in self._make_batches(entry_data):
            self._pca.partial_fit(batch)

def save_state(self, run_root):
    joblib.dump(self._pca, run_root / "model.joblib")

def apply(self, entry_key, entry_data):
    return self._transform_one(entry_data)
```

---

## Global features (4) -- skip_transform_phase features

- `global_tsne.py` (GlobalTSNE)
- `global_kmeans.py` (GlobalKMeansClustering)
- `ward_assign.py` (WardAssignClustering)
- `kpms_apply.py` (KpmsApply)

These currently bypass the transform phase entirely and write outputs during `fit()`/`save_model()`. Under the new protocol, they get `apply()` like every other feature:

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

    # Always load upstream artifacts from artifact_paths (replaces
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
    T_target = self.params.total_templates
    scaler_samples = []
    template_samples = []
    n_keys = 0

    for entry_key, entry_data in inputs:
        n_keys += 1
        X = entry_data.features
        if X.shape[0] == 0:
            continue
        quota = max(self.params.pre_quota_per_key, T_target // max(1, n_keys))
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
    Xsamp = np.vstack(scaler_samples)
    del scaler_samples
    if Xsamp.shape[0] > r_scaler:
        idx = self._rng.choice(Xsamp.shape[0], size=r_scaler, replace=False)
        Xsamp = Xsamp[idx]
    self._scaler = StandardScaler().fit(Xsamp)
    del Xsamp

    # Farthest-first template selection
    X_pre = np.vstack([self._scaler.transform(s) for s in template_samples])
    del template_samples
    sel = [int(self._rng.integers(0, X_pre.shape[0]))]
    d2 = np.sum((X_pre - X_pre[sel[0]]) ** 2, axis=1)
    while len(sel) < min(T_target, X_pre.shape[0]):
        i = int(np.argmax(d2))
        sel.append(i)
        d2 = np.minimum(d2, np.sum((X_pre - X_pre[i]) ** 2, axis=1))
    templates = X_pre[np.array(sel)]
    self._templates = templates
    self._template_indices = np.array(sel)
    del X_pre, d2

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

def apply(self, entry_key, entry_data):
    """Map one sequence through the embedding using prepare_partial + optimize.

    Uses chunked streaming (params.map_chunk) to control memory. Each chunk
    gets its own prepare_partial call with aggressive GC between chunks to
    prevent openTSNE memory buildup.
    """
    import pyarrow as pa

    embedding = self._embedding
    scaler = self._scaler
    CHUNK = self.params.map_chunk
    Kp = self.params.partial_k
    Pp = self.params.perplexity
    It = self.params.partial_iters
    Lr = self.params.partial_lr

    X = entry_data.features
    Xs = scaler.transform(X).astype(np.float32, copy=False)
    del X

    # Chunked mapping -- same logic as current _map_sequences_streaming
    Y_seq = np.empty((Xs.shape[0], 2), dtype=np.float32)
    for j in range(0, Xs.shape[0], CHUNK):
        X_chunk = Xs[j : j + CHUNK]
        part = embedding.prepare_partial(
            X_chunk, initialization="median", k=Kp, perplexity=Pp
        )
        part.optimize(
            n_iter=It, learning_rate=Lr, exaggeration=2.0,
            momentum=0.0, inplace=True, verbose=False,
        )
        coords = np.asarray(part).astype(np.float32, copy=False).copy()
        Y_seq[j : j + coords.shape[0], :] = coords
        # Aggressively free openTSNE internals
        if hasattr(part, "affinities"):
            del part.affinities
        del part, coords
        if (j // CHUNK) % 5 == 4:
            gc.collect()

    del Xs
    gc.collect()
    pa.default_memory_pool().release_unused()

    # Build output DataFrame
    data = {
        "tsne_x": Y_seq[:, 0],
        "tsne_y": Y_seq[:, 1],
        "frame": entry_data.frames,
    }
    if entry_data.id1 is not None:
        data["id1"] = pd.array(entry_data.id1, dtype="Int64")
    if entry_data.id2 is not None:
        data["id2"] = pd.array(entry_data.id2, dtype="Int64")
    return pd.DataFrame(data)
```

The `_fit_embedding(templates)` helper extracts the existing affinity + init +
TSNEEmbedding construction and two-phase optimize (lines 348-381 of the current
`fit()`). No logic changes -- just factored out of the monolithic `fit()`.

---

## WardAssign migration

```python
scope_dependent = True
parallelizable = True

def load_state(self, run_root, artifact_paths):
    self._model = None
    if "model" in artifact_paths:
        self._model = self.params.model.from_path(artifact_paths["model"])
    path = run_root / "assignments.joblib"
    if path.exists():
        self._assignments = joblib.load(path)
        return True
    return False

def fit(self, inputs):
    # Compute assignments from ward model
    self._assignments = self._compute_assignments(self._model)

def save_state(self, run_root):
    joblib.dump(self._assignments, run_root / "assignments.joblib")

def apply(self, entry_key, entry_data):
    return self._assign_sequence(entry_key, entry_data)
```

---

## Step 1: Migrate all per-sequence-with-fit features

## Step 2: Run tests

## Step 3: Migrate all global features

For each: remove `skip_transform_phase`, `set_run_root`, `get_additional_index_rows`, `_persist_mapped_coords`, `_write_sequence_outputs`, `_append_index_row`. Add `load_state`, `fit(iterator)`, `save_state`, `apply`.

## Step 4: Run tests

Run: `cd mosaic && uv run pytest tests/ -x -q`
Expected: All pass

## Step 5: Commit

```
feat: migrate per-sequence-with-fit and global features to new protocol
```
