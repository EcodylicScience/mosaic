"""
VizGlobalColored feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import ClassVar

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from mosaic.core.helpers import (
    detect_label_format,
    expand_labels_to_dense,
    load_labels_for_feature_frames,
    make_entry_key,
    to_safe_name,
)
from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.loading import get_feature_run_root
from mosaic.core.pipeline.types import (
    GroundTruthLabelsSource,
    InputRequire,
    Inputs,
    NpzLoadSpec,
    Params,
    ResultColumn,
)

from ..feature_library.registry import register_feature


@register_feature
class VizGlobalColored:
    """
    Generic scatter plot visualization for any global embedding or feature columns.

    Uses ResultColumn params for fully customizable x and y axes. For example,
    t-SNE coordinates, PCA components, speed vs approach distance, etc.
    Labels can be from any feature's parquet output (via ResultColumn) or
    ground truth labels (via GroundTruthLabelsSource).
    """

    name = "viz-global-colored"
    version = "0.1"
    parallelizable = False

    class Inputs(Inputs[ResultColumn]):
        _require: ClassVar[InputRequire] = "empty"

    class Params(Params):
        x: ResultColumn
        y: ResultColumn
        labels: ResultColumn | GroundTruthLabelsSource | None = None
        label_missing_value: int = -1
        label_order: list[int] | None = None
        label_name_map: dict[int, str] | None = None
        plot_max: int = 300_000
        palette: str | list[tuple[float, float, float]] = "tab20"
        title: str = "Global embedding colored scatter"
        point_size: float = 2.0
        point_alpha: float = 0.35
        debug_save_arrays: bool = False

    def __init__(
        self,
        inputs: VizGlobalColored.Inputs = Inputs(()),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self._ds = None
        self._figs: list[tuple[str, Figure]] = []
        self._marker_written = False
        self._summary: dict[str, object] = {}
        self._scope: Scope = Scope()
        self._debug_arrays: dict[str, object] | None = None

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope(self, scope: Scope) -> None:
        self._scope = scope

    def needs_fit(self):
        return True

    def supports_partial_fit(self):
        return False

    def partial_fit(self, df):
        raise NotImplementedError

    def finalize_fit(self):
        pass

    def _resolve_run_root(self, rc: ResultColumn) -> tuple[str, Path]:
        """Resolve (run_id, run_root) for a ResultColumn reference."""
        return get_feature_run_root(self._ds, rc.feature, rc.run_id)

    def _load_result_columns(
        self,
        *result_columns: ResultColumn,
        allowed_keys: set[str] | None,
    ) -> dict[str, pd.DataFrame]:
        """Load columns from a feature's per-sequence parquet output files.

        All arguments must reference the same feature and run_id.
        Uses the feature index CSV to discover per-sequence files,
        avoiding global artifacts (e.g. cluster_sizes.parquet).

        Returns a dict mapping entry_key -> DataFrame with the requested
        columns plus alignment columns (frame, sequence, group).
        """
        from mosaic.core.pipeline.index import feature_index_path

        if self._ds is None:
            raise RuntimeError("dataset not bound; call bind_dataset() first")

        first = result_columns[0]
        for rc in result_columns[1:]:
            if rc.feature != first.feature or rc.run_id != first.run_id:
                raise ValueError(
                    "all result columns must reference the same feature/run_id"
                )
        resolved_run_id, _ = self._resolve_run_root(first)

        idx_path = feature_index_path(self._ds, first.feature)
        if not idx_path.exists():
            msg = f"feature index not found: {idx_path}"
            raise FileNotFoundError(msg)
        idx_df = pd.read_csv(idx_path)
        idx_df = idx_df[idx_df["run_id"].astype(str) == resolved_run_id]

        columns = [rc.column for rc in result_columns]
        load_columns = list(columns) + ["frame", "sequence", "group"]
        result: dict[str, pd.DataFrame] = {}
        for _, row in idx_df.iterrows():
            key = make_entry_key(
                str(row.get("group", "") or ""), str(row.get("sequence", "") or "")
            )
            if allowed_keys and key not in allowed_keys:
                continue
            abs_path = str(row.get("abs_path", ""))
            if not abs_path:
                continue
            path = self._ds.resolve_path(abs_path)
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=load_columns)
            if not all(c in df.columns for c in columns):
                continue
            result[key] = df
        return result

    def _load_label_file(
        self,
        path: Path,
        load_spec: NpzLoadSpec,
        n_frames: int | None = None,
        feature_frames: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Load labels from file, auto-detecting and handling both dense and sparse formats.

        For individual_pair_v1 (sparse) format, expands to dense per-frame array
        OR aligns to feature_frames if provided.

        Parameters
        ----------
        path : Path
            Path to the label file (NPZ)
        load_spec : NpzLoadSpec
            Load specification
        n_frames : int, optional
            Number of frames for dense expansion. If None, uses max(frames) + 1.
        feature_frames : np.ndarray, optional
            If provided, return labels aligned to these specific frame indices.

        Returns
        -------
        np.ndarray
            1D array of labels.
        """
        # If feature_frames provided, use the alignment helper
        if feature_frames is not None:
            return load_labels_for_feature_frames(
                path, feature_frames, default_label=0, deduplicate_symmetric=True
            )

        # For NPZ files, detect format and handle appropriately
        with np.load(path, allow_pickle=True) as npz:
            fmt = detect_label_format(npz)

            if fmt == "individual_pair_v1":
                frames = np.asarray(npz["frames"], dtype=np.int64).ravel()
                labels = np.asarray(npz["labels"], dtype=np.int64).ravel()
                individual_ids = np.asarray(npz["individual_ids"])
                if individual_ids.ndim == 1:
                    individual_ids = individual_ids.reshape(-1, 2)

                # For pair behaviors stored symmetrically, deduplicate.
                # Keep only events where id1 <= id2 to avoid double-counting.
                mask = individual_ids[:, 0] <= individual_ids[:, 1]
                frames = frames[mask]
                labels = labels[mask]

                return expand_labels_to_dense(
                    frames,
                    labels,
                    n_frames=n_frames,
                    default_label=0,
                )

            else:
                # Dense format or unknown - use standard key-based loading
                key = load_spec.key
                if key not in npz.files:
                    raise KeyError(f"Key '{key}' not found in {path.name}")
                return np.asarray(npz[key]).ravel()

    def _load_labels_from_index(
        self, spec: GroundTruthLabelsSource
    ) -> list[tuple[str, Path]]:
        """
        Load (sequence_safe, path) pairs from labels/<kind>/index.csv.
        Supports optional glob-style filtering via spec.pattern on filename.
        """
        if self._ds is None:
            raise RuntimeError(
                "VizGlobalColored requires dataset binding to load labels."
            )
        kind = spec.kind
        if not kind:
            raise ValueError("labels spec with source='labels' requires 'kind'.")
        labels_root = self._ds.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"Labels index not found: {idx_path}")
        df = pd.read_csv(idx_path)
        pattern = spec.pattern
        entries: list[tuple[str, Path]] = []
        for _, row in df.iterrows():
            abs_raw = row.get("abs_path", "")
            if not isinstance(abs_raw, str) or not abs_raw:
                continue
            pth = self._ds.resolve_path(abs_raw)
            if pattern and not fnmatch.fnmatch(pth.name, pattern):
                continue
            key = make_entry_key(
                str(row.get("group", "")), str(row.get("sequence", ""))
            )
            if not key:
                key = to_safe_name(pth.stem)
            entries.append((key, pth))
        return entries

    def _prepare_color_map(self, labels):
        unique_vals = list(pd.unique(pd.Series(labels, dtype="object")))
        missing = self.params.label_missing_value
        order = self.params.label_order
        if order:
            order_list = [o for o in order if o in unique_vals]
            unique_vals = order_list + [u for u in unique_vals if u not in order_list]

        palette = sns.color_palette(
            self.params.palette,
            max(1, len([u for u in unique_vals if u != missing])),
        )
        color_map = {}
        idx = 0
        for val in unique_vals:
            if val == missing:
                continue
            color_map[val] = palette[idx % len(palette)]
            idx += 1
        if missing in unique_vals:
            color_map[missing] = (0.7, 0.7, 0.7)
        return color_map

    def _label_display(self, val):
        name_map = self.params.label_name_map or {}
        if isinstance(name_map, dict):
            if val in name_map:
                return str(name_map[val])
            try:
                ival = int(val)
                if ival in name_map:
                    return str(name_map[ival])
            except Exception:
                pass
        return str(val)

    def fit(self, X_iter):
        allowed_keys = self._scope.entry_keys or None

        x_rc = self.params.x
        y_rc = self.params.y

        # Load x and y columns
        same_source = x_rc.feature == y_rc.feature and x_rc.run_id == y_rc.run_id
        if same_source:
            xy_frames: dict[str, pd.DataFrame] = self._load_result_columns(
                x_rc, y_rc, allowed_keys=allowed_keys
            )
        else:
            x_data = self._load_result_columns(x_rc, allowed_keys=allowed_keys)
            y_data = self._load_result_columns(y_rc, allowed_keys=allowed_keys)
            xy_frames = {}
            for key in set(x_data) & set(y_data):
                xdf = x_data[key]
                ydf = y_data[key]
                shared_columns = [
                    c
                    for c in ("frame", "sequence", "group")
                    if c in xdf.columns and c in ydf.columns
                ]
                if shared_columns:
                    y_keep = [y_rc.column] + [
                        c for c in shared_columns if c != y_rc.column
                    ]
                    merged = xdf.merge(ydf[y_keep], on=shared_columns, how="inner")
                elif len(xdf) == len(ydf):
                    merged = pd.DataFrame(
                        {
                            x_rc.column: xdf[x_rc.column].values,
                            y_rc.column: ydf[y_rc.column].values,
                        }
                    )
                else:
                    msg = f"cannot align x and y for key={key}: no shared columns and lengths differ ({len(xdf)} vs {len(ydf)})"
                    raise RuntimeError(msg)
                xy_frames[key] = merged

        if not xy_frames:
            raise RuntimeError("[viz-global-colored] No coordinate data loaded.")

        # Build coordinate arrays
        Y_list, key_list, n_list = [], [], []
        for key in sorted(xy_frames):
            df = xy_frames[key]
            x_vals = df[x_rc.column].to_numpy(dtype=np.float32)
            y_vals = df[y_rc.column].to_numpy(dtype=np.float32)
            Y_list.append(np.column_stack([x_vals, y_vals]))
            key_list.append(key)
            n_list.append(len(df))

        # Load labels
        labels_spec = self.params.labels
        missing_value = self.params.label_missing_value
        if labels_spec is not None:
            lab_map: dict[str, np.ndarray] = {}

            if isinstance(labels_spec, GroundTruthLabelsSource):
                entries = self._load_labels_from_index(labels_spec)
                if allowed_keys:
                    entries = [(s, p) for s, p in entries if s in allowed_keys]
                for label_key, path in entries:
                    lab_map[label_key] = self._load_label_file(path, labels_spec.load)
            elif isinstance(labels_spec, ResultColumn):
                label_data = self._load_result_columns(
                    labels_spec, allowed_keys=allowed_keys
                )
                for label_key, ldf in label_data.items():
                    # Align labels to coord frames if possible
                    coord_df = xy_frames.get(label_key)
                    if coord_df is not None and len(coord_df) == len(ldf):
                        # Same row count: assume aligned by construction
                        # (both produced by StreamingFeatureHelper in same order)
                        vals = ldf[labels_spec.column].to_numpy()
                    elif coord_df is not None:
                        # Different lengths: try merging on shared columns
                        shared_columns = [
                            c
                            for c in ("frame", "sequence", "group", "id1", "id2", "id")
                            if c in coord_df.columns and c in ldf.columns
                        ]
                        if shared_columns:
                            label_keep = [labels_spec.column] + [
                                c for c in shared_columns if c != labels_spec.column
                            ]
                            merged = coord_df[shared_columns].merge(
                                ldf[label_keep],
                                on=shared_columns,
                                how="left",
                            )
                            vals = (
                                merged[labels_spec.column]
                                .fillna(missing_value)
                                .to_numpy()
                            )
                        else:
                            vals = ldf[labels_spec.column].to_numpy()
                    else:
                        vals = ldf[labels_spec.column].to_numpy()
                    lab_map[label_key] = vals

            L_parts = []
            for idx, (key, n) in enumerate(zip(key_list, n_list)):
                arr = lab_map.get(key)
                if arr is None:
                    arr = lab_map.get(to_safe_name(str(key)))
                if arr is None:
                    print(
                        f"[viz-global-colored] WARN: missing labels for key={key}; assigning {missing_value}",
                        file=sys.stderr,
                    )
                    arr = np.full(n, missing_value, dtype=int)
                else:
                    arr = np.asarray(arr).ravel()
                    if arr.shape[0] < n and n % arr.shape[0] == 0:
                        # Coord stream duplicated per-id/perspective;
                        # repeat labels to match.
                        factor = n // arr.shape[0]
                        print(
                            f"[viz-global-colored] INFO: labels shorter than coords for key={key} "
                            f"({arr.shape[0]} vs {n}); repeating labels x{factor}",
                            file=sys.stderr,
                        )
                        arr = np.tile(arr, factor)
                    elif arr.shape[0] != n:
                        nmin = min(arr.shape[0], n)
                        if nmin <= 0:
                            continue
                        if nmin < n:
                            print(
                                f"[viz-global-colored] INFO: trimming coords for key={key} "
                                f"from {n} to {arr.shape[0]} (labels length)",
                                file=sys.stderr,
                            )
                            Y_list[idx] = Y_list[idx][:nmin]
                            n_list[idx] = nmin
                        arr = arr[:nmin]
                L_parts.append(arr)
            if not L_parts:
                raise RuntimeError("No labels aligned with coordinates.")
            Y_all = np.vstack(Y_list)
            L_all = np.concatenate(L_parts)
        else:
            L_parts = [np.zeros(n, dtype=int) for n in n_list]
            Y_all = np.vstack(Y_list)
            L_all = np.concatenate(L_parts)

        if self.params.debug_save_arrays:
            self._debug_arrays = {
                "Y_all": Y_all.copy(),
                "L_all": L_all.copy(),
                "key_list": list(key_list),
                "n_list": list(n_list),
            }

        plot_max = self.params.plot_max
        if Y_all.shape[0] > plot_max:
            rng = np.random.default_rng(42)
            sel = rng.choice(Y_all.shape[0], size=plot_max, replace=False)
            Y_plot = Y_all[sel]
            L_plot = L_all[sel]
        else:
            Y_plot = Y_all
            L_plot = L_all

        color_map = self._prepare_color_map(
            L_all if labels_spec is not None else L_plot
        )
        colors = np.array([color_map.get(val, (0.7, 0.7, 0.7)) for val in L_plot])

        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.scatter(
            Y_plot[:, 0],
            Y_plot[:, 1],
            c=colors,
            s=self.params.point_size,
            alpha=self.params.point_alpha,
            linewidths=0,
        )
        ax.set_title(self.params.title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")

        unique_vals = pd.unique(pd.Series(L_plot, dtype="object"))
        if len(unique_vals) <= 15:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=color_map.get(val, (0.7, 0.7, 0.7)),
                    markeredgecolor="none",
                    label=self._label_display(val),
                )
                for val in unique_vals
            ]
            if handles:
                ax.legend(handles=handles, title="label", loc="best", fontsize=8)

        out_name = "global_colored.png"
        self._figs = [(out_name, fig)]
        self._summary = {
            "points": int(Y_all.shape[0]),
            "plotted": int(Y_plot.shape[0]),
            "labels_present": labels_spec is not None,
        }

    def transform(self, df):
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        return pd.DataFrame(
            [
                {
                    "outputs": ",".join(fname for fname, _ in self._figs),
                    "labels_present": self.params.labels is not None,
                }
            ]
        )

    def save_model(self, path: Path):
        run_root = path.parent
        for fname, fig in self._figs:
            fig.savefig(run_root / fname, dpi=150, bbox_inches="tight")
        if self.params.debug_save_arrays and self._debug_arrays:
            try:
                np.savez_compressed(
                    run_root / "debug_viz_arrays.npz", **self._debug_arrays
                )
            except Exception as exc:
                print(
                    f"[viz-global-colored] WARN: failed to save debug arrays: {exc}",
                    file=sys.stderr,
                )
        joblib.dump(
            {
                "params": self.params.model_dump(),
                "summary": self._summary,
                "files": [fname for fname, _ in self._figs],
            },
            run_root / "viz.joblib",
        )

    def load_model(self, path: Path):
        bundle = joblib.load(path)
        saved = bundle.get("params", {})
        if isinstance(saved, dict):
            self.params = self.Params.model_validate(saved)
        self._summary = bundle.get("summary", {})
