"""Build the MABe22 example notebooks.

Run once to scaffold `notebooks/mabe22-beetle-ant.ipynb` and
`notebooks/mabe22-mouse-triplets.ipynb`. After the first run the two
notebooks are independent — edit them directly, re-run for outputs, etc.
This script exists so the initial scaffolding stays consistent between
the two examples.

Usage:
    python scripts/build_example_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"


def md(*lines: str) -> dict:
    src = "\n".join(lines)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True) or [""],
    }


def code(src: str) -> dict:
    body = dedent(src).strip("\n") + "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "source": body.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


NB_METADATA = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.5",
    },
}


def save_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": NB_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))
    print(f"wrote {path.relative_to(REPO)}  ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------

def imports_cell() -> dict:
    return code("""
        import json
        import os
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from mosaic.behavior.feature_library import (
            ExtractLabeledTemplates,
            ExtractTemplates,
            GlobalKMeansClustering,
            GlobalScaler,
            GlobalTSNE,
            GlobalWardClustering,
            GroundTruthLabelsSource,
            Inputs,
            ResultColumn,
            TemporalStackingFeature,
            XgboostFeature,
        )
        from mosaic.core.analysis import compute_cluster_label_agreement
        from mosaic.core.dataset import Dataset, new_dataset_manifest
        from mosaic.core.pipeline import load_values
    """)


def setup_paths_cell(default_example_name: str, subset_source: str, subset_name: str) -> dict:
    return code(f"""
        # Raw MABe22 .npy dump. Override with:  export MABE22_DATA_ROOT=/path/to/mabe22
        DATA_ROOT = Path(os.environ.get("MABE22_DATA_ROOT", "/home/jdavidson/mabe22"))

        # Where this example's dataset (manifest, tracks, features, labels) will live.
        EXAMPLE_ROOT = Path(
            os.environ.get("MOSAIC_EXAMPLE_ROOT", "./example_datasets/{default_example_name}")
        ).resolve()
        EXAMPLE_ROOT.mkdir(parents=True, exist_ok=True)

        # Subset source file and the smaller file we'll write into tracks_raw/.
        # Subsetting keeps runtime quick; edit N_SEQUENCES_SUBSET below to change.
        SUBSET_SOURCE = DATA_ROOT / "{subset_source}"
        TRACKS_RAW = EXAMPLE_ROOT / "tracks_raw"
        TRACKS_RAW.mkdir(parents=True, exist_ok=True)
        SUBSET_DST = TRACKS_RAW / "{subset_name}"

        print("DATA_ROOT:   ", DATA_ROOT)
        print("EXAMPLE_ROOT:", EXAMPLE_ROOT)
        print("source file: ", SUBSET_SOURCE, "exists:", SUBSET_SOURCE.exists())
    """)


def subset_cell(n_sequences: int) -> dict:
    return code(f"""
        # Build a small subset of the raw MABe22 file — fewer sequences = faster demo.
        # Runs once; re-run with SUBSET_DST.unlink() first if you want to rebuild.
        N_SEQUENCES_SUBSET = {n_sequences}

        if not SUBSET_DST.exists():
            raw = np.load(SUBSET_SOURCE, allow_pickle=True).item()
            seq_ids = list(raw["sequences"].keys())[:N_SEQUENCES_SUBSET]
            subset = {{
                "vocabulary": raw.get("vocabulary", []),
                "sequences": {{sid: raw["sequences"][sid] for sid in seq_ids}},
            }}
            np.save(SUBSET_DST, subset, allow_pickle=True)
            print(f"wrote {{SUBSET_DST}} with {{len(seq_ids)}} sequences")
        else:
            print(f"subset already exists: {{SUBSET_DST}}")
            raw = np.load(SUBSET_DST, allow_pickle=True).item()
            print(f"  {{len(raw['sequences'])}} sequences, vocabulary={{raw.get('vocabulary', [])}}")
    """)


def manifest_and_convert_cell(dataset_name: str) -> dict:
    return code(f"""
        # Create dataset manifest (idempotent — skips if dataset.yaml already exists).
        manifest_path = EXAMPLE_ROOT / "dataset.yaml"
        if not manifest_path.exists():
            manifest_path = new_dataset_manifest(
                name="{dataset_name}",
                base_dir=EXAMPLE_ROOT,
                version="0.1.0",
            )

        dataset = Dataset(manifest_path=manifest_path).load()

        # Index raw tracks — MABe22 stores many sequences in one .npy file.
        dataset.index_tracks_raw(
            search_dirs=[TRACKS_RAW],
            patterns="*.npy",
            src_format="mabe22_npy",
            multi_sequences_per_file=True,
            group_from="filename",
            recursive=False,
        )

        # Convert raw .npy -> standardized per-sequence parquet.
        dataset.convert_all_tracks(overwrite=False)

        # Convert MABe22 annotation tracks -> dense per-frame behavior NPZ labels.
        dataset.convert_all_labels(
            kind="behavior",
            source_format="mabe22_npy",
            overwrite=False,
        )

        print(dataset)
    """)


# ---------------------------------------------------------------------------
# Global embedding + clustering cells (shared)
# ---------------------------------------------------------------------------

def global_embedding_cells(input_wave_names: list[str]) -> list[dict]:
    """Cells for ExtractTemplates -> GlobalScaler -> TSNE -> KMeans -> Ward.

    input_wave_names: list of local variable names pointing to wavelet Result
    objects to combine into the downstream pipeline.
    """
    wave_tuple = ", ".join(input_wave_names)
    if len(input_wave_names) == 1:
        wave_tuple += ","  # single-element tuple

    cells = [
        md("## Global embedding and clustering"),
        code(f"""
            # 1. Fit global scaler on 2000 random templates from the wavelet features.
            templates = ExtractTemplates(
                Inputs(({wave_tuple})),
                params={{"n_templates": 2000}},
            )
            templates_result = dataset.run_feature(templates)

            scaler = GlobalScaler(
                Inputs(({wave_tuple})),
                params={{
                    "templates": ExtractTemplates.TemplatesArtifact().from_result(templates_result),
                }},
            )
            scaler_result = dataset.run_feature(scaler)

            # 2. Re-extract 2000 farthest-first templates on the scaled features,
            #    then fit t-SNE for visualization.
            scaled_templates = ExtractTemplates(
                Inputs((scaler_result,)),
                params={{"n_templates": 2000, "strategy": "farthest_first"}},
            )
            scaled_templates_result = dataset.run_feature(scaled_templates)

            tsne = GlobalTSNE(
                Inputs((scaled_templates_result,)),
                params={{
                    "perplexity": 50,
                    "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
                }},
            )
            tsne_result = dataset.run_feature(tsne)
        """),
        code("""
            # KMeans at two granularities.
            kmeans_results = []
            for k in [50, 100]:
                kmeans = GlobalKMeansClustering(
                    Inputs((scaled_templates_result,)),
                    params={
                        "k": int(k),
                        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
                        "label_artifact_points": True,
                    },
                )
                k_result = dataset.run_feature(kmeans)
                kmeans_results.append({"k": int(k), "run": k_result})
        """),
        code("""
            # Ward hierarchical clustering for comparison.
            ward_results = []
            for cut in [25, 50, 100]:
                ward = GlobalWardClustering(
                    Inputs((scaled_templates_result,)),
                    params={
                        "templates": ExtractTemplates.TemplatesArtifact().from_result(scaled_templates_result),
                        "method": "ward",
                        "n_clusters": cut,
                    },
                )
                ward_result = dataset.run_feature(ward)
                ward_results.append(ward_result)
        """),
    ]
    return cells


def cluster_agreement_cells() -> list[dict]:
    return [
        md("## Cluster/label agreement (ground truth vs clusters)"),
        code("""
            kmeans_feature = k_result.feature
            run_id = k_result.run_id

            agr = compute_cluster_label_agreement(
                dataset,
                cluster_feature=kmeans_feature,
                cluster_run_id=run_id,
                label_kind="behavior",
                sequences=None,
            )
            agr["metrics"]
        """),
    ]


def tsne_viz_cells() -> list[dict]:
    return [
        md("## Visualizations (ground truth + clusters on t-SNE)"),
        code("""
            df = load_values(
                dataset,
                [
                    ResultColumn(column="tsne_x").from_result(tsne_result),
                    ResultColumn(column="tsne_y").from_result(tsne_result),
                    ResultColumn(column="cluster").from_result(k_result),
                    GroundTruthLabelsSource(),
                ],
            )
            print(df.columns.tolist())
            df.head()
        """),
        code("""
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            sns.scatterplot(
                data=df, x="tsne_x", y="tsne_y", hue="labels-behavior",
                palette=sns.color_palette(n_colors=df["labels-behavior"].nunique()),
                lw=0, s=5, alpha=0.2, legend=None, ax=axes[0],
            )
            axes[0].set_aspect("equal")
            axes[0].set_title("ground truth labels")

            sns.scatterplot(
                data=df, x="tsne_x", y="tsne_y", hue="cluster",
                palette=sns.color_palette(n_colors=df["cluster"].nunique()),
                lw=0, s=5, alpha=0.2, legend=None, ax=axes[1],
            )
            axes[1].set_aspect("equal")
            axes[1].set_title("KMeans clusters")

            plt.tight_layout()
            plt.show()
        """),
    ]


def xgboost_cells(default_class_comment: str) -> list[dict]:
    return [
        md(
            "## Supervised classification (XGBoost)",
            "",
            "Temporal-stacked scaled features → train/test split by sequence → "
            "per-class template subsampling → XGBoost multiclass + per-sequence "
            "inference. Writes `reports.json` and `summary.csv` next to the run.",
        ),
        code("""
            ts_stack = TemporalStackingFeature(
                Inputs((scaler_result,)),
                params={
                    "half": 2,
                    "skip": 1,
                    "use_temporal_stack": True,
                    "sigma_stack": 2,
                    "add_pool": False,
                    "pool_stats": ("mean",),
                    "fps": 30.0,
                },
            )
            ts_stack_result = dataset.run_feature(
                ts_stack, parallel_workers=4, parallel_mode="thread"
            )
        """),
        code(f"""
            labeled_templates = ExtractLabeledTemplates(
                Inputs((ts_stack_result,)),
                params={{
                    "labels": GroundTruthLabelsSource(),
                    "n_per_class": 500,
                    "test_fraction": 0.2,
                }},
            )
            labeled_templates_result = dataset.run_feature(labeled_templates)

            xgb = XgboostFeature(
                Inputs((ts_stack_result,)),
                params={{
                    "templates": ExtractLabeledTemplates.LabeledTemplatesArtifact().from_result(labeled_templates_result),
                    "strategy": "multiclass",
                    "default_class": 0,  {default_class_comment}
                    "n_estimators": 10,
                    "max_depth": 3,
                }},
            )
            xgb_result = dataset.run_feature(xgb)
        """),
        code("""
            from mosaic.core.pipeline.run import feature_run_root

            xgb_run_root = feature_run_root(dataset, xgb_result.feature, xgb_result.run_id)
            summary_path = xgb_run_root / "summary.csv"
            reports_path = xgb_run_root / "reports.json"

            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                display(summary_df)

            if reports_path.exists():
                with open(reports_path) as f:
                    reports = json.load(f)
                print(json.dumps(reports, indent=2))
        """),
        md("### Visualize predictions on t-SNE"),
        code("""
            df_pred = load_values(
                dataset,
                [
                    ResultColumn(column="predicted_label").from_result(xgb_result),
                    ResultColumn(column="split").from_result(labeled_templates_result),
                    ResultColumn(column="tsne_x").from_result(tsne_result),
                    ResultColumn(column="tsne_y").from_result(tsne_result),
                    GroundTruthLabelsSource(),
                ],
            )
            df_test = df_pred[df_pred["split"] == "test"]
            print(f"{len(df_pred)} total, "
                  f"{(df_pred['split']=='train').sum()} train, "
                  f"{len(df_test)} test")
        """),
        code("""
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, col, title in [
                (axes[0], "labels-behavior", "Ground truth (test only)"),
                (axes[1], "predicted_label", "XGBoost predictions (test only)"),
            ]:
                sns.scatterplot(
                    data=df_test, x="tsne_x", y="tsne_y", hue=col,
                    palette=sns.color_palette(n_colors=int(df_test[col].nunique())),
                    lw=0, s=3, alpha=0.15, legend=None, ax=ax,
                )
                ax.set_aspect("equal")
                ax.set_title(title)
            plt.tight_layout()
            plt.show()
        """),
        code("""
            from sklearn.metrics import ConfusionMatrixDisplay

            y_true = df_test["labels-behavior"].values
            y_pred = df_test["predicted_label"].values

            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, normalize="true", cmap="Blues", ax=ax
            )
            ax.set_title("XGBoost confusion matrix (test only, row-normalized)")
            plt.tight_layout()
            plt.show()
        """),
    ]


# ---------------------------------------------------------------------------
# Notebook 1: beetle-ant (centroid-only)
# ---------------------------------------------------------------------------

def build_beetle_notebook() -> list[dict]:
    cells: list[dict] = []

    cells.append(md(
        "# MABe22 beetle-ant end-to-end template",
        "",
        "Lightweight example of the mosaic pipeline on **centroid-only** data "
        "(no pose keypoints). Uses the MABe22 beetle-ant subset: 2 animals, "
        "flat `(T, 4)` centroid layout, with behavior annotations.",
        "",
        "Pipeline: MABe22 `.npy` → tracks parquet + labels NPZ → `PairPositionFeatures` "
        "→ `PairWavelet` → global t-SNE / KMeans / Ward → XGBoost supervised classifier.",
        "",
        "**Source paper**: Sun et al. 2023, *[MABe22](https://arxiv.org/abs/2207.10553)*. "
        "Dataset DOI: [10.22002/rdsa8-rde65](https://doi.org/10.22002/rdsa8-rde65).",
    ))

    cells.append(imports_cell())
    cells.append(md("## Setup paths"))
    cells.append(setup_paths_cell(
        default_example_name="mabe22-beetle-ant",
        subset_source="beetle_user_train.npy",
        subset_name="beetle_subset.npy",
    ))

    cells.append(md(
        "## Build a small subset of the raw MABe22 file",
        "",
        "The full `beetle_user_train.npy` is ~50 MB and holds many sequences. "
        "We grab the first few to keep the demo fast.",
    ))
    cells.append(subset_cell(n_sequences=20))

    cells.append(md("## Create dataset + convert tracks and labels"))
    cells.append(manifest_and_convert_cell(dataset_name="mabe22-beetle-ant"))

    cells.append(md(
        "## Feature pipeline — centroid-only path",
        "",
        "Beetles have only per-frame `(x, y)` (no pose keypoints), so we use "
        "**`PairPositionFeatures`** — a drop-in replacement for `PairEgocentricFeatures` "
        "that works from `(x, y, angle)` alone. ANGLE comes from velocity direction "
        "(computed by the MABe22 track converter).",
    ))
    cells.append(code("""
        from mosaic.behavior.feature_library import PairPositionFeatures, PairWavelet

        feat_pos = PairPositionFeatures()
        pos_result = dataset.run_feature(feat_pos, parallel_workers=4, parallel_mode="process")
        print(pos_result)
    """))

    cells.append(code("""
        wavelet_params = {
            "f_min": 0.2,
            "f_max": 5.0,
            "n_freq": 25,
            "wavelet": "cmor1.5-1.0",
            "log_floor": -3.0,
            "sampling": {"fps_default": 30.0},
        }

        feat_wave = PairWavelet(Inputs((pos_result,)), params=wavelet_params)
        wave_result = dataset.run_feature(feat_wave, parallel_workers=4, parallel_mode="process")
        print(wave_result)
    """))

    cells.extend(global_embedding_cells(input_wave_names=["wave_result"]))
    cells.extend(cluster_agreement_cells())
    cells.extend(tsne_viz_cells())
    cells.extend(xgboost_cells(default_class_comment="# 0 = background"))

    return cells


# ---------------------------------------------------------------------------
# Notebook 2: mouse triplets (pose, N=3)
# ---------------------------------------------------------------------------

def build_mouse_notebook() -> list[dict]:
    cells: list[dict] = []

    cells.append(md(
        "# MABe22 mouse-triplets end-to-end template",
        "",
        "Pose-tracked example using MABe22 **mouse triplets** — 3 mice per video, "
        "12 keypoints each, with behavior annotations (e.g. *chases*). Demonstrates "
        "the N>2 animals case: 3 unique pairs → 6 pair/perspective rows per frame.",
        "",
        "Pipeline: MABe22 `.npy` → tracks parquet + labels NPZ → "
        "`PairPoseDistancePCA` + `PairEgocentricFeatures` → `PairWavelet` (on both) "
        "→ global t-SNE / KMeans / Ward → XGBoost supervised classifier.",
        "",
        "**Source paper**: Sun et al. 2023, *[MABe22](https://arxiv.org/abs/2207.10553)*. "
        "Dataset DOI: [10.22002/rdsa8-rde65](https://doi.org/10.22002/rdsa8-rde65).",
        "",
        "### How this differs from the CalMS21 template",
        "",
        "- **N=3 animals** instead of 2 — `PairPoseDistancePCA` / `PairEgocentricFeatures` "
        "automatically produce features for all 3 unique pairs.",
        "- **12 keypoints** instead of 7 — pass `pose_n=12` and update neck/tail indices "
        "to match the MABe22 mouse skeleton.",
        "- **Labels are sequence-level** (all animals share the same annotation track "
        "at each frame) — the dense NPZ format handles this; pair features each get the "
        "per-frame label via frame lookup.",
    ))

    cells.append(imports_cell())
    cells.append(md("## Setup paths"))
    cells.append(setup_paths_cell(
        default_example_name="mabe22-mouse-triplets",
        subset_source="mouse_user_train.npy",
        subset_name="mouse_subset.npy",
    ))

    cells.append(md(
        "## Build a small subset of the raw MABe22 file",
        "",
        "The full `mouse_user_train.npy` is ~420 MB. We subset to a small number "
        "of sequences to keep the demo tractable — increase for a fuller run.",
    ))
    cells.append(subset_cell(n_sequences=10))

    cells.append(md("## Create dataset + convert tracks and labels"))
    cells.append(manifest_and_convert_cell(dataset_name="mabe22-mouse-triplets"))

    cells.append(md(
        "## Feature pipeline — pose path",
        "",
        "Same pose pipeline as CalMS21, parameterized for the MABe22 mouse skeleton "
        "(12 keypoints). Check the MABe22 paper for the exact keypoint order — the "
        "`neck_idx` / `tail_base_idx` here are reasonable defaults; adjust if your "
        "skeleton definition differs.",
    ))
    cells.append(code("""
        from mosaic.behavior.feature_library import (
            PairEgocentricFeatures,
            PairPoseDistancePCA,
            PairWavelet,
        )

        # MABe22 mouse skeleton: 12 keypoints per animal.
        # Indices here are placeholders — consult the paper's keypoint list
        # and adjust if your neck/tail indices differ.
        POSE_N = 12
        NECK_IDX = 3
        TAIL_BASE_IDX = 9

        feat_pose = PairPoseDistancePCA(
            params={"n_components": 6, "pose": {"pose_n": POSE_N}},
        )
        pose_result = dataset.run_feature(
            feat_pose, parallel_workers=4, parallel_mode="process",
        )
        print(pose_result)
    """))

    cells.append(code("""
        feat_ego = PairEgocentricFeatures(
            params={
                "neck_idx": NECK_IDX,
                "tail_base_idx": TAIL_BASE_IDX,
                "pose": {"pose_n": POSE_N},
            },
        )
        ego_result = dataset.run_feature(
            feat_ego, parallel_workers=4, parallel_mode="process",
        )
        print(ego_result)
    """))

    cells.append(code("""
        wavelet_params = {
            "f_min": 0.2,
            "f_max": 5.0,
            "n_freq": 25,
            "wavelet": "cmor1.5-1.0",
            "log_floor": -3.0,
            "sampling": {"fps_default": 30.0},
        }

        feat_wave_social = PairWavelet(Inputs((pose_result,)), params=wavelet_params)
        social_wave_result = dataset.run_feature(
            feat_wave_social, parallel_workers=4, parallel_mode="process",
        )

        feat_wave_ego = PairWavelet(Inputs((ego_result,)), params=wavelet_params)
        ego_wave_result = dataset.run_feature(
            feat_wave_ego, parallel_workers=4, parallel_mode="process",
        )
    """))

    cells.extend(global_embedding_cells(
        input_wave_names=["social_wave_result", "ego_wave_result"]
    ))
    cells.extend(cluster_agreement_cells())
    cells.extend(tsne_viz_cells())
    cells.extend(xgboost_cells(default_class_comment="# 0 = background"))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    save_notebook(NOTEBOOKS / "mabe22-beetle-ant.ipynb", build_beetle_notebook())
    save_notebook(NOTEBOOKS / "mabe22-mouse-triplets.ipynb", build_mouse_notebook())


if __name__ == "__main__":
    main()
