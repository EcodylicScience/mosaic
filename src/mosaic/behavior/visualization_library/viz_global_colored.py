"""
VizGlobalColored feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

import joblib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mosaic.behavior.feature_library.helpers import _build_path_sequence_map
from mosaic.behavior.feature_library.spec import (
    ArtifactSpec,
    FeatureLabelsSource,
    GroundTruthLabelsSource,
    InputRequire,
    Inputs,
    JoblibLoadSpec,
    LoadSpec,
    NpzLoadSpec,
    OutputType,
    Params,
    ParquetLoadSpec,
    Result,
)
from mosaic.behavior.feature_library.spec import register_feature
from mosaic.core.helpers import (
    detect_label_format,
    expand_labels_to_dense,
    load_labels_for_feature_frames,
    to_safe_name,
)
from mosaic.core.pipeline._utils import Scope


@register_feature
class VizGlobalColored:
    """
    Generic wrapper to visualize any global embedding (t-SNE, PCA, UMAP, etc.)
    colored by arbitrary labels.

    Params (defaults target the global t-SNE + k-means workflow):
      coords: {feature, run_id, pattern, load:{kind:"npz", key:"Y"}}
      labels: optional spec matching coords (same structure as coords) OR
              {"source": "labels", "kind": "<label_kind>", "load": {...}}
              to load from labels/<kind>/index.csv (e.g., CalMS21 ground truth)
      coord_key_regex: regex applied to coord filenames (default extracts the sequence name)
      label_key_regex: regex for label filenames (defaults to coord regex)
      label_missing_value: sentinel for missing labels (default -1 -> gray)
      label_order: optional list of label values to fix the color ordering
      label_name_map: optional dict mapping label ids -> display names
      plot_max: max points to scatter
      palette: seaborn palette name or list
      title: plot title
      point_size / point_alpha: scatter style tweaks
      debug_save_arrays: if True, save Y_all/L_all (aligned, pre-subsample) as debug_viz_arrays.npz
    Output:
      - PNG in run_root plus a single marker parquet row for indexing.
    """

    name = "viz-global-colored"
    version = "0.1"
    output_type: OutputType = "viz"
    parallelizable = False

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "empty"

    class Params(Params):
        coords: ArtifactSpec
        labels: FeatureLabelsSource | GroundTruthLabelsSource | None = None
        coord_key_regex: str = r"seq=(.+?)(?:_persp=.*)?$"
        label_key_regex: str | None = None
        label_missing_value: int = -1
        label_order: list[int] | None = None
        label_name_map: dict[int, str] | None = None
        plot_max: int = 300_000
        palette: str | list[tuple[float, float, float]] = "tab20"
        title: str = "Global embedding colored scatter"
        point_size: float = 2.0
        point_alpha: float = 0.35
        debug_save_arrays: bool = False
        frame_source_feature: str | None = None

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
        self._summary: dict = {}
        self._seq_path_cache: dict[tuple[str, str], dict[Path, str]] = {}
        self._scope: Scope = Scope()
        self._debug_arrays: dict | None = None

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

    def _load_artifacts_glob(self, spec: ArtifactSpec):
        ds = self._ds
        idx = ds.get_root("features") / spec.feature / "index.csv"
        df = pd.read_csv(idx)
        run_id = spec.run_id
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(
                    by=["finished_at" if len(cand) else "started_at"],
                    ascending=False,
                    kind="stable",
                )
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            run_id = str(df.iloc[0]["run_id"])
        run_root = ds.get_root("features") / spec.feature / run_id
        files = sorted(run_root.glob(spec.pattern))
        return run_id, run_root, files

    def _load_one(self, path: Path, load_spec: LoadSpec) -> np.ndarray:
        if isinstance(load_spec, NpzLoadSpec):
            npz = np.load(path, allow_pickle=True)
            A = np.asarray(npz[load_spec.key])
            if load_spec.transpose:
                A = A.T
            return A
        if isinstance(load_spec, ParquetLoadSpec):
            df = pd.read_parquet(path)
            if load_spec.drop_columns:
                df = df.drop(
                    columns=[c for c in load_spec.drop_columns if c in df.columns],
                    errors="ignore",
                )
            if load_spec.columns:
                use = [c for c in load_spec.columns if c in df.columns]
                if not use:
                    return np.empty((0, len(load_spec.columns)), dtype=np.float32)
                df = df[use]
            elif load_spec.numeric_only:
                df = df.select_dtypes(include=[np.number])
                # Drop metadata columns that are numeric but not features
                for mc in ("frame", "time", "id1", "id2"):
                    if mc in df.columns:
                        df = df.drop(columns=[mc])
            else:
                # coerce everything numeric-style to avoid object columns
                df = df.apply(pd.to_numeric, errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if load_spec.transpose:
                arr = arr.T
            return arr
        if isinstance(load_spec, JoblibLoadSpec):
            obj = joblib.load(path)
            return obj if load_spec.key is None else obj[load_spec.key]
        raise ValueError(f"Unsupported load spec type: {type(load_spec)}")

    def _load_label_file(
        self,
        path: Path,
        load_spec: LoadSpec,
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
        load_spec : LoadSpec
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
        if not isinstance(load_spec, NpzLoadSpec):
            # Non-NPZ files use standard loading
            return np.asarray(self._load_one(path, load_spec)).ravel()

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

    def _load_labels_from_index(self, spec: GroundTruthLabelsSource) -> list[tuple[str, Path]]:
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
            seq_safe = to_safe_name(str(row.get("sequence", "")))
            if not seq_safe:
                seq_safe = to_safe_name(pth.stem)
            entries.append((seq_safe, pth))
        return entries

    def _feature_seq_map(
        self, feature_name: str, run_id: str | None
    ) -> dict[Path, str]:
        key = (feature_name, str(run_id))
        if key in self._seq_path_cache:
            return self._seq_path_cache[key]
        mapping = _build_path_sequence_map(self._ds, feature_name, run_id)
        self._seq_path_cache[key] = mapping
        return mapping

    def _extract_key(self, path: Path, regex: str | None, seq_map: dict[Path, str]):
        stem = path.stem
        m = re.search(regex, stem) if regex else None
        if not m:
            safe_seq = seq_map.get(path.resolve())
            if not safe_seq:
                safe_seq = to_safe_name(stem)
            return str(safe_seq)
        if m.lastindex is None:
            return m.group(0)
        if m.lastindex == 1:
            return m.group(1)
        return tuple(m.groups())

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

    def _load_frame_source_map(self, allowed_any: set) -> dict[str, np.ndarray]:
        """
        Load frame indices from the frame_source_feature for each sequence.

        Returns a dict mapping sequence key -> frame array.
        """
        frame_source = self.params.frame_source_feature
        if not frame_source or not self._ds:
            return {}

        # Try to find frame source feature files
        features_root = self._ds.get_root("features")
        idx_path = features_root / frame_source / "index.csv"
        if not idx_path.exists():
            return {}

        try:
            df = pd.read_csv(idx_path)
            # Get the most recent run
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                df = base.sort_values(
                    by=["finished_at" if len(cand) else "started_at"],
                    ascending=False,
                    kind="stable",
                )
            run_id = str(df.iloc[0]["run_id"])
            run_root = features_root / frame_source / run_id

            frame_map: dict[str, np.ndarray] = {}
            for _, row in df[df["run_id"].astype(str) == run_id].iterrows():
                abs_path = row.get("abs_path", "")
                if not abs_path:
                    continue
                pth = self._ds.resolve_path(abs_path)
                if not pth.exists():
                    continue
                # Get sequence key
                seq_safe = to_safe_name(str(row.get("sequence", "")))
                if not seq_safe:
                    seq_safe = to_safe_name(
                        pth.stem.split("__")[-1] if "__" in pth.stem else pth.stem
                    )

                if allowed_any and seq_safe not in allowed_any:
                    continue

                # Load frame column from parquet
                if pth.suffix.lower() == ".parquet":
                    try:
                        feat_df = pd.read_parquet(pth)
                        if "frame" in feat_df.columns:
                            frame_map[seq_safe] = feat_df["frame"].to_numpy(
                                dtype=np.int64
                            )
                    except Exception:
                        pass
            return frame_map
        except Exception as e:
            print(
                f"[viz-global-colored] WARN: failed to load frame source '{frame_source}': {e}",
                file=sys.stderr,
            )
            return {}

    def fit(self, X_iter):
        coord_spec = self.params.coords
        coord_run_id, _, coord_files = self._load_artifacts_glob(coord_spec)
        if not coord_files:
            raise FileNotFoundError("[viz-global-colored] No coordinate files found.")

        allowed_any = self._scope.entry_keys

        key_regex = self.params.coord_key_regex
        coord_seq_map = self._feature_seq_map(coord_spec.feature, coord_run_id)
        Y_list, key_list, n_list = [], [], []
        for f in coord_files:
            key = self._extract_key(f, key_regex, coord_seq_map)
            key_safe = to_safe_name(key)
            if allowed_any and key not in allowed_any and key_safe not in allowed_any:
                continue
            arr = self._load_one(f, coord_spec.load)
            if arr.ndim != 2:
                arr = np.atleast_2d(arr)
            Y_list.append(arr)
            key_list.append(key)
            n_list.append(arr.shape[0])
        if not Y_list:
            raise RuntimeError("No coordinate arrays loaded.")

        # Load frame indices from frame_source_feature if specified
        frame_source_map = self._load_frame_source_map(allowed_any)

        labels_spec = self.params.labels
        missing_value = self.params.label_missing_value
        if labels_spec is not None:
            label_regex = key_regex if self.params.label_key_regex is None else self.params.label_key_regex
            label_load_spec = labels_spec.load
            lab_map: dict[Any, np.ndarray] = {}

            # Build label entries to load
            label_entries: list[tuple[str, Path]] = []
            if isinstance(labels_spec, GroundTruthLabelsSource):
                entries = self._load_labels_from_index(labels_spec)
                if allowed_any:
                    entries = [(s, p) for s, p in entries if s in allowed_any]
                label_entries = entries
            else:
                label_run_id, _, label_files = self._load_artifacts_glob(labels_spec)
                label_seq_map = self._feature_seq_map(
                    labels_spec.feature, label_run_id
                )
                for lf in label_files:
                    lab_key = self._extract_key(lf, label_regex, label_seq_map)
                    label_entries.append((lab_key, lf))

            # Load labels with frame alignment if available
            seq_map = {pth.resolve(): safe_seq for safe_seq, pth in label_entries}
            for safe_seq, pth in label_entries:
                lab_key = self._extract_key(pth, label_regex, seq_map)
                # Get feature frames for this sequence if available
                feature_frames = frame_source_map.get(safe_seq)
                if feature_frames is None:
                    feature_frames = frame_source_map.get(to_safe_name(safe_seq))
                lab_map[lab_key] = self._load_label_file(
                    pth, label_load_spec, feature_frames=feature_frames
                )
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
                    if arr.shape[0] < n:
                        if n % arr.shape[0] == 0:
                            # common case: coord stream duplicated per-id/perspective; repeat labels accordingly
                            factor = n // arr.shape[0]
                            if factor == 2:
                                print(
                                    f"[viz-global-colored] INFO: labels shorter than coords for key={key} "
                                    f"({arr.shape[0]} vs {n}); repeating labels x{factor} "
                                    "(likely per-id/perspective duplication in inputs).",
                                    file=sys.stderr,
                                )
                            arr = np.tile(arr, factor)
                        else:
                            # lengths differ but not a clean multiple; trim coords to label length
                            print(
                                f"[viz-global-colored] INFO: trimming coords for key={key} "
                                f"from {n} to {arr.shape[0]} (labels length) due to mismatch.",
                                file=sys.stderr,
                            )
                            Y_list[idx] = Y_list[idx][: arr.shape[0]]
                            n_list[idx] = arr.shape[0]
                            n = arr.shape[0]
                    if arr.shape[0] != n:
                        nmin = min(arr.shape[0], n)
                        if nmin <= 0:
                            continue
                        if nmin < n:
                            # trim coords to match available labels
                            print(
                                f"[viz-global-colored] INFO: trimming coords for key={key} "
                                f"from {n} to {nmin} to match labels {arr.shape[0]}",
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

        color_map = self._prepare_color_map(L_all if labels_spec is not None else L_plot)
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
