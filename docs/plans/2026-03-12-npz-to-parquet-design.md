# Migrate per-sequence npz to standard parquet output

## Goal

Replace the legacy per-sequence `.npz` outputs in GlobalTSNE, WardAssign, and
GlobalKMeans with standard per-frame parquet files. Drop the `ArtifactSpec`
subclasses for those outputs. Drop the `seq=` filename convention entirely.
Update all consumers to use the standard pipeline conventions.

## Background

Three global features write per-sequence results as `.npz` files using the
legacy `global_*_seq={safe_seq}.npz` naming convention. These are per-frame
outputs (one row per frame) that should follow the same parquet conventions as
every other per-frame feature. WardAssign and GlobalKMeans already write dual
format (npz + parquet); GlobalTSNE writes npz only.

## Per-feature changes

### GlobalTSNE

Currently writes `global_tsne_coords_seq={safe_seq}.npz` containing key `Y`
(N x 2 float32 array of t-SNE coordinates). Replace with standard parquet
output: columns `tsne_x`, `tsne_y`, `frame`, `sequence`, `group`. Drop
`SeqCoordsArtifact`.

Global model artifacts stay unchanged:
- `global_templates_features.npz`
- `global_tsne_templates.npz`
- `global_opentsne_embedding.joblib`

### WardAssign

Already writes per-sequence parquet with columns `frame`, `cluster`, `id1`,
`id2`, `entity_level`, `sequence`, `group`. Drop the duplicate npz write
(`global_ward_labels_seq={safe_seq}.npz`) and `SeqLabelsArtifact`.

### GlobalKMeans

Same as WardAssign -- already writes parquet. Drop the npz write
(`global_kmeans_labels_seq={safe_seq}.npz`) and `SeqLabelsArtifact`.

## Drop seq= naming convention

Remove the `seq=` regex from `_extract_key` in both `helpers.py` and
`viz_global_colored.py`. The method simplifies to returning `path.stem`
directly. Any remaining callers that depend on the regex are updated as part
of this migration.

## ResultColumn: generic column reference

Replace `FeatureLabelsSource(ArtifactSpec[NpzLoadSpec])` with a `Result`-based
reference:

```python
class ResultColumn(Result[str]):
    """Reference to a column in a feature's parquet output."""
    column: str  # required, no default
```

No default on `column` -- works for discrete labels (`cluster` from
ward-assign) or continuous values (`speed`, `tsne_x`) for colormaps. The
labels union becomes:

```python
LabelsSourceSpec = ResultColumn | GroundTruthLabelsSource
```

This follows the same pattern as `NNResult` for pair filtering: a `Result`
subclass used as a params field, where the feature reads the referenced
output on its own terms.

## VizGlobalColored

Keep `Inputs` empty (no pipeline inputs), same pattern as GlobalWard. Instead,
expose `x` and `y` as `ResultColumn` params -- fully customizable scatter plot
axes. For a t-SNE plot, the user sets `x=ResultColumn(feature="global-tsne",
column="tsne_x")` and `y=ResultColumn(feature="global-tsne", column="tsne_y")`.
But any feature column works: speed vs approach distance, PCA components, etc.

Labels configured in params via `ResultColumn` (reads any column from any
feature's parquet) or `GroundTruthLabelsSource` (unchanged). Drop the
`coords` ArtifactSpec param and `coord_key_regex`/`label_key_regex` params.

### Data alignment

When `x` and `y` reference the same feature+run_id, load once and extract both
columns. When they reference different features, load separately and align via
`pd.merge` on `frame` (+ `sequence`/`group` for identity). This uses the same
approach as `yield_input_data` in the standard pipeline path
(`iteration.py:489`), which merges on shared key columns.

Note: `StreamingFeatureHelper.load_key_data()` does NOT perform frame-level
alignment -- it min-trims arrays and horizontally concatenates, assuming rows
are in the same order. This is a latent bug for global features that consume
multi-input Results: there is no guard against mixing frame-scoped and
unscoped runs (which produce different row counts per sequence), nor against
partial/scoped runs where not all sequences/groups were computed. These
mismatches lead to silent data misalignment. Fixing this in
`StreamingFeatureHelper` is a separate concern from this migration.

## analysis.py

Update `_augment_with_saved_sequences()` and `_load_cluster_labels()` to read
standard parquet outputs instead of `*_labels_seq=*.npz` files.

## Unchanged

- Global model artifacts (templates, embeddings, cluster centers, joblib files)
- `GroundTruthLabelsSource` mechanism (reads from `labels/` directory)
- `NNResult` pattern for pair filtering
