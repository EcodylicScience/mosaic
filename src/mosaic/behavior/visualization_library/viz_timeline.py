"""
TimelinePlot feature.

Visualizes per-frame behavior/cluster labels as horizontal colored-bar
timelines.  Works with any feature that produces per-frame labels
(kpms-apply syllables, global-kmeans clusters, ground-truth labels, etc.).

Unlike VizGlobalColored which needs separate coord + label specs, TimelinePlot
takes a single feature reference and auto-detects what to plot.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import re
import sys

import numpy as np
import pandas as pd
import joblib

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from mosaic.core.dataset import register_feature
from mosaic.behavior.feature_library.helpers import (
    _normalize_identity_columns,
)
from mosaic.core.helpers import (
    to_safe_name,
    detect_label_format,
    expand_labels_to_dense,
)

# Priority list for auto-detecting the label column in a DataFrame
_LABEL_COL_PRIORITY = [
    "syllable", "cluster", "label_id", "label",
    "prediction", "behavior", "state",
]
# Metadata columns that are never labels
_SKIP_COLS = frozenset({
    "frame", "time", "sequence", "group",
    "id", "id1", "id2", "id_a", "id_b", "id_A", "id_B",
    "entity_level", "perspective", "fps",
})


@register_feature
class TimelinePlot:
    """
    Visualize per-frame labels as horizontal colored-bar timelines.

    Params
    ------
    source : dict
        Feature reference: ``{"feature": "kpms-apply", "run_id": None, "pattern": "*.parquet"}``
        Or ground-truth labels: ``{"source": "labels", "kind": "CalMS21"}``
    label_column : str or None
        Column containing the labels.  Auto-detected if None.
    label_columns : list[str] or None
        Combine multiple binary (0/1) columns into one composite label.
        The label value is the column name of the first active column,
        or ``0`` when none are active.  Overrides ``label_column``.
        Use with ``skip_labels=[0]`` to hide inactive frames.
    skip_labels : list or None
        Label values to *not* draw (rendered as white/blank space).
        Example: ``[0]`` to hide "no event" frames.
    symmetric_pairs : bool
        If True, treat (A,B) == (B,A) for pair-level data.
    palette : str
        Seaborn palette name for label colors (default ``"tab20"``).
    pair_palette : str
        Palette for asymmetric pair role shading (default ``"Paired"``).
    figsize_width : float
        Width of each figure in inches.
    row_height : float
        Height per timeline row in inches.
    min_fig_height / max_fig_height : float
        Clamp figure height.
    show_legend : bool
        Whether to add a legend.  When there are many labels the legend is
        placed outside the plot area.
    title_template : str
        Format string for plot title; ``{sequence}`` is replaced.
    dpi : int
        Output resolution.
    per_sequence : bool
        One PNG per sequence (True) or a single combined PNG (False).
    missing_label_value : int
        Sentinel for unlabeled frames (rendered gray).
    label_name_map : dict or None
        Optional ``{label_id: display_name}`` for the legend.

    Output
    ------
    PNG file(s) in the run folder plus a single marker parquet row for indexing.
    """

    name = "viz-timeline"
    version = "0.1"
    output_type = "viz"

    # -----------------------------------------------------------------
    # Init / dataset hooks
    # -----------------------------------------------------------------

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        defaults: dict = {
            "source": {
                "feature": None,
                "run_id": None,
                "pattern": "*.parquet",
            },
            "label_column": None,
            "label_columns": None,
            "skip_labels": None,
            "symmetric_pairs": True,
            "palette": "tab20",
            "pair_palette": "Paired",
            "figsize_width": 16,
            "row_height": 0.4,
            "min_fig_height": 2.0,
            "max_fig_height": 20.0,
            "show_legend": True,
            "title_template": "{sequence}",
            "dpi": 150,
            "per_sequence": True,
            "missing_label_value": -1,
            "label_name_map": None,
        }
        self.params = dict(defaults)
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(self.params.get(k), dict):
                    merged = dict(self.params[k])
                    merged.update(v)
                    self.params[k] = merged
                else:
                    self.params[k] = v

        self._ds = None
        self._figs: list[tuple[str, plt.Figure]] = []
        self._marker_written = False
        self._summary: dict = {}
        self._scope_constraints: Optional[dict] = None

    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_constraints(self, scope: Optional[dict]) -> None:
        self._scope_constraints = scope or {}

    def needs_fit(self):
        return True

    def supports_partial_fit(self):
        return False

    def loads_own_data(self):
        return True

    def partial_fit(self, X):
        raise NotImplementedError

    def finalize_fit(self):
        pass

    # -----------------------------------------------------------------
    # Artifact loading helpers (same pattern as VizGlobalColored)
    # -----------------------------------------------------------------

    def _load_artifacts_glob(self, spec: dict):
        """Resolve run_id and glob for files in a feature's run folder."""
        ds = self._ds
        idx_path = ds.get_root("features") / spec["feature"] / "index.csv"
        df = pd.read_csv(idx_path)
        run_id = spec.get("run_id")
        if run_id is None:
            if "finished_at" in df.columns:
                cand = df[df["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df
                sort_col = "finished_at" if len(cand) else "started_at"
                df = base.sort_values(by=[sort_col], ascending=False, kind="stable")
            else:
                df = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            run_id = str(df.iloc[0]["run_id"])
        run_root = ds.get_root("features") / spec["feature"] / run_id
        files = sorted(run_root.glob(spec["pattern"]))
        return run_id, run_root, files

    def _load_labels_from_index(self, spec: dict) -> list[tuple[str, Path]]:
        """Load (sequence_safe, path) pairs from labels/<kind>/index.csv."""
        if self._ds is None:
            raise RuntimeError("[viz-timeline] Dataset not bound.")
        kind = spec.get("kind")
        if not kind:
            raise ValueError("[viz-timeline] labels source requires 'kind'.")
        idx_path = self._ds.get_root("labels") / kind / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"[viz-timeline] Labels index not found: {idx_path}")
        df = pd.read_csv(idx_path)
        import fnmatch
        pattern = spec.get("pattern")
        entries: list[tuple[str, Path]] = []
        for _, row in df.iterrows():
            abs_raw = row.get("abs_path", "")
            if not isinstance(abs_raw, str) or not abs_raw:
                continue
            pth = self._ds.resolve_path(abs_raw)
            if pattern and not fnmatch.fnmatch(pth.name, pattern):
                continue
            seq_safe = str(row.get("sequence_safe") or row.get("sequence") or "").strip()
            if not seq_safe:
                seq_safe = to_safe_name(pth.stem)
            entries.append((seq_safe, pth))
        return entries

    # -----------------------------------------------------------------
    # Scope filtering
    # -----------------------------------------------------------------

    def _allowed_set(self) -> set[str]:
        scope = self._scope_constraints or {}
        allowed = set()
        for s in (scope.get("safe_sequences") or []):
            allowed.add(s)
        for s in (scope.get("sequences") or []):
            allowed.add(s)
            allowed.add(to_safe_name(s))
        return allowed

    def _is_filtered_out(self, seq_key: str, allowed: set[str]) -> bool:
        if not allowed:
            return False
        return seq_key not in allowed and to_safe_name(seq_key) not in allowed

    # -----------------------------------------------------------------
    # Auto-detection helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _auto_detect_label_col(df: pd.DataFrame) -> Optional[str]:
        for col in _LABEL_COL_PRIORITY:
            if col in df.columns:
                return col
        for col in df.columns:
            if col not in _SKIP_COLS:
                return col
        return None

    @staticmethod
    def _sequence_key_from_df(df: pd.DataFrame, path: Path) -> str:
        """Derive the sequence grouping key from the DataFrame columns.

        Reads the ``group`` and ``sequence`` columns directly from the data,
        which is more reliable than parsing filenames (avoids mismatches when
        per-individual files like ``__id{N}.parquet`` exist).
        """
        group = ""
        seq = ""
        if "group" in df.columns:
            g = df["group"].dropna()
            if len(g):
                group = str(g.iloc[0]).strip()
        if "sequence" in df.columns:
            s = df["sequence"].dropna()
            if len(s):
                seq = str(s.iloc[0]).strip()
        if seq:
            return f"{group}__{seq}" if group else seq
        # Fallback to filename stem (strip __id{N})
        stem = re.sub(r"__id\d+$", "", path.stem)
        return stem

    @staticmethod
    def _combine_label_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
        """Combine multiple binary columns into a single categorical label.

        For each row, the label is the name of the first column (in order)
        whose value == 1.  If no column is active, the label is 0.
        """
        result = pd.Series(0, index=df.index, dtype=object)
        # Process in reverse so the first column in the list wins ties
        for col in reversed(columns):
            if col in df.columns:
                mask = df[col].astype(int) == 1
                result[mask] = col
        return result

    @staticmethod
    def _normalize_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Normalize a raw parquet to ``[frame, label, id1?, id2?]``."""
        out = pd.DataFrame()
        out["frame"] = df["frame"].astype(np.int64) if "frame" in df.columns else np.arange(len(df), dtype=np.int64)
        out["label"] = df[label_col].values

        id1, id2, entity_level = _normalize_identity_columns(df)
        if id1 is not None:
            out["id1"] = id1.values
        if id2 is not None and entity_level == "pair":
            out["id2"] = id2.values
        out["entity_level"] = entity_level
        return out

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------

    def _load_from_feature(self, source_spec: dict) -> dict[str, pd.DataFrame]:
        run_id, run_root, files = self._load_artifacts_glob(source_spec)
        if not files:
            raise FileNotFoundError("[viz-timeline] No files found for source feature.")

        label_col_override = self.params.get("label_column")
        allowed = self._allowed_set()

        seq_data: dict[str, pd.DataFrame] = {}
        for f in files:
            try:
                df = pd.read_parquet(f)
            except Exception as exc:
                print(f"[viz-timeline] WARN: failed to read {f}: {exc}", file=sys.stderr)
                continue
            if df.empty:
                continue

            # Derive sequence key from the data itself (most reliable)
            seq_key = self._sequence_key_from_df(df, f)
            if self._is_filtered_out(seq_key, allowed):
                continue

            label_columns = self.params.get("label_columns")
            if label_columns:
                df["_combined_label"] = self._combine_label_columns(df, label_columns)
                label_col = "_combined_label"
            else:
                label_col = label_col_override or self._auto_detect_label_col(df)
            if label_col is None or label_col not in df.columns:
                continue
            norm = self._normalize_df(df, label_col)
            if seq_key in seq_data:
                seq_data[seq_key] = pd.concat([seq_data[seq_key], norm], ignore_index=True)
            else:
                seq_data[seq_key] = norm

        return seq_data

    def _load_from_labels(self, source_spec: dict) -> dict[str, pd.DataFrame]:
        entries = self._load_labels_from_index(source_spec)
        allowed = self._allowed_set()
        if allowed:
            entries = [(s, p) for s, p in entries if not self._is_filtered_out(s, allowed)]

        seq_data: dict[str, pd.DataFrame] = {}
        for seq_safe, pth in entries:
            try:
                with np.load(pth, allow_pickle=True) as npz:
                    fmt = detect_label_format(npz)
                    label_names_arr = npz.get("label_names")
                    label_ids_arr = npz.get("label_ids")

                    if fmt == "individual_pair_v1":
                        frames = np.asarray(npz["frames"], dtype=np.int64).ravel()
                        labels = np.asarray(npz["labels"], dtype=np.int64).ravel()
                        individual_ids = np.asarray(npz["individual_ids"])
                        if individual_ids.ndim == 1:
                            individual_ids = individual_ids.reshape(-1, 2)
                        out = pd.DataFrame({
                            "frame": frames,
                            "label": labels,
                            "id1": individual_ids[:, 0],
                            "id2": individual_ids[:, 1],
                        })
                        has_pair = (individual_ids[:, 1] != -1).any()
                        out["entity_level"] = "pair" if has_pair else "individual"
                    else:
                        labels = np.asarray(npz.get("labels", npz.get("label"))).ravel()
                        out = pd.DataFrame({
                            "frame": np.arange(len(labels), dtype=np.int64),
                            "label": labels,
                            "entity_level": "global",
                        })

                    # Store label_name_map from NPZ if not user-provided
                    if self.params.get("label_name_map") is None and label_names_arr is not None and label_ids_arr is not None:
                        try:
                            name_map = {int(lid): str(lname) for lid, lname in zip(label_ids_arr, label_names_arr)}
                            self.params["label_name_map"] = name_map
                        except Exception:
                            pass

                    seq_data[seq_safe] = out
            except Exception as exc:
                print(f"[viz-timeline] WARN: failed to load {pth}: {exc}", file=sys.stderr)

        return seq_data

    # -----------------------------------------------------------------
    # Row grouping
    # -----------------------------------------------------------------

    @staticmethod
    def _build_row_groups(
        df: pd.DataFrame,
        entity_level: str,
        symmetric: bool,
    ) -> list[tuple[str, pd.DataFrame]]:
        """Group a normalized DataFrame into labeled timeline rows."""
        rows: list[tuple[str, pd.DataFrame]] = []

        if entity_level == "global" or "id1" not in df.columns:
            rows.append(("all", df))

        elif entity_level == "individual":
            for id_val, sub in df.groupby("id1", sort=True):
                try:
                    id_label = str(int(id_val))
                except (ValueError, TypeError):
                    id_label = str(id_val)
                rows.append((f"id {id_label}", sub))

        elif entity_level == "pair":
            if symmetric:
                df = df.copy()
                id1 = pd.to_numeric(df["id1"], errors="coerce")
                id2 = pd.to_numeric(df["id2"], errors="coerce")
                lo = np.minimum(id1, id2)
                hi = np.maximum(id1, id2)
                df["_pk"] = list(zip(lo.astype("Int64"), hi.astype("Int64")))
                for pk, sub in df.groupby("_pk", sort=True):
                    rows.append((f"pair ({pk[0]}, {pk[1]})", sub.drop(columns=["_pk"])))
            else:
                for (a, b), sub in df.groupby(["id1", "id2"], sort=True):
                    try:
                        a_s, b_s = str(int(a)), str(int(b))
                    except (ValueError, TypeError):
                        a_s, b_s = str(a), str(b)
                    rows.append((f"{a_s} \u2192 {b_s}", sub))

        return rows if rows else [("all", df)]

    # -----------------------------------------------------------------
    # Run-length encoding for efficient bar rendering
    # -----------------------------------------------------------------

    @staticmethod
    def _label_runs_to_bars(
        frames: np.ndarray,
        labels: np.ndarray,
    ) -> list[tuple[int, int, Any]]:
        """Convert sorted (frames, labels) into ``(start, width, label)`` segments."""
        if len(frames) == 0:
            return []
        bars: list[tuple[int, int, Any]] = []
        run_start = int(frames[0])
        run_label = labels[0]
        prev = int(frames[0])
        for i in range(1, len(frames)):
            f = int(frames[i])
            if labels[i] != run_label or f > prev + 1:
                bars.append((run_start, prev - run_start + 1, run_label))
                run_start = f
                run_label = labels[i]
            prev = f
        bars.append((run_start, prev - run_start + 1, run_label))
        return bars

    # -----------------------------------------------------------------
    # Color mapping
    # -----------------------------------------------------------------

    def _build_color_map(self, label_series: pd.Series) -> dict:
        unique_vals = sorted(pd.unique(label_series), key=lambda x: (isinstance(x, str), x))
        missing = self.params.get("missing_label_value", -1)
        skip_labels = set(self.params.get("skip_labels") or [])
        palette_name = self.params.get("palette", "tab20")
        excluded = {missing} | skip_labels
        n_active = len([v for v in unique_vals if v not in excluded])
        palette = sns.color_palette(palette_name, max(1, n_active))
        cmap: dict = {}
        idx = 0
        for val in unique_vals:
            if val in skip_labels:
                continue
            elif val == missing:
                cmap[val] = (0.7, 0.7, 0.7)
            else:
                cmap[val] = palette[idx % len(palette)]
                idx += 1
        return cmap

    def _build_asymmetric_color_map(self, label_series: pd.Series) -> tuple[dict, dict]:
        """Build two color maps for asymmetric pairs (id1-role / id2-role).

        Uses the ``"Paired"`` palette which gives consecutive light/dark pairs.
        Returns ``(cmap_focal, cmap_other)`` where focal uses the darker shade.
        """
        unique_vals = sorted(pd.unique(label_series), key=lambda x: (isinstance(x, str), x))
        missing = self.params.get("missing_label_value", -1)
        pair_palette_name = self.params.get("pair_palette", "Paired")
        n_non_missing = len([v for v in unique_vals if v != missing])
        # Paired palette has pairs at (2i, 2i+1): lighter, darker
        palette = sns.color_palette(pair_palette_name, max(2, n_non_missing * 2))
        cmap_focal: dict = {}
        cmap_other: dict = {}
        idx = 0
        for val in unique_vals:
            if val == missing:
                cmap_focal[val] = (0.7, 0.7, 0.7)
                cmap_other[val] = (0.7, 0.7, 0.7)
            else:
                pi = (idx * 2) % len(palette)
                cmap_other[val] = palette[pi]      # lighter shade
                cmap_focal[val] = palette[pi + 1] if pi + 1 < len(palette) else palette[pi]  # darker shade
                idx += 1
        return cmap_focal, cmap_other

    def _label_display(self, val) -> str:
        name_map = self.params.get("label_name_map") or {}
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

    # -----------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------

    def _render_timeline(self, df: pd.DataFrame, seq_key: str) -> plt.Figure:
        """Render a single-sequence timeline figure.

        Uses ``imshow`` with a pre-computed RGBA image instead of per-bar
        ``barh()`` calls, which is orders of magnitude faster for features
        with many labels or long sequences (e.g. 25 clusters x 86k frames).
        """
        skip_set = set(self.params.get("skip_labels") or [])
        entity_level = df["entity_level"].mode().iloc[0] if "entity_level" in df.columns else "global"
        symmetric = self.params.get("symmetric_pairs", True)
        is_asymmetric_pair = entity_level == "pair" and not symmetric

        rows = self._build_row_groups(df, entity_level, symmetric)
        n_rows = len(rows)
        fig_w = float(self.params.get("figsize_width", 16))
        fig_h = float(np.clip(
            n_rows * float(self.params["row_height"]) + 1.5,
            float(self.params["min_fig_height"]),
            float(self.params["max_fig_height"]),
        ))

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

        # Build color map(s)
        if is_asymmetric_pair:
            cmap_focal, cmap_other = self._build_asymmetric_color_map(df["label"])
        else:
            cmap = self._build_color_map(df["label"])

        # Build RGBA image: white background, paint per (row, label)
        max_frame = int(df["frame"].max()) + 1
        img = np.ones((n_rows, max_frame, 4), dtype=np.float32)  # white RGBA

        for i, (row_label, row_df) in enumerate(rows):
            frames = row_df["frame"].to_numpy().astype(np.intp)
            labels = row_df["label"].to_numpy()

            if is_asymmetric_pair:
                use_cmap = cmap_focal if i % 2 == 0 else cmap_other
            else:
                use_cmap = cmap

            for label_val, color in use_cmap.items():
                if label_val in skip_set:
                    continue
                mask = labels == label_val
                f_idx = frames[mask]
                if len(f_idx) == 0:
                    continue
                img[i, f_idx, 0] = color[0]
                img[i, f_idx, 1] = color[1]
                img[i, f_idx, 2] = color[2]
                # alpha already 1.0

        ax.imshow(img, aspect="auto", interpolation="nearest",
                  extent=[0, max_frame, n_rows - 0.5, -0.5])

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([label for label, _ in rows], fontsize=8)
        ax.set_xlabel("Frame")
        title = self.params.get("title_template", "{sequence}").format(sequence=seq_key)
        ax.set_title(title)
        ax.set_xlim(left=0)
        sns.despine(ax=ax, left=True)

        # Legend
        if is_asymmetric_pair:
            self._add_paired_legend(ax, cmap_focal, cmap_other)
        else:
            self._add_legend(ax, cmap)

        fig.tight_layout()
        return fig

    def _add_legend(self, ax, color_map: dict):
        if not self.params.get("show_legend", True):
            return
        items = [(v, c) for v, c in color_map.items() if v != self.params.get("missing_label_value", -1)]
        if not items:
            return
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                       markerfacecolor=c, markeredgecolor="none",
                       label=self._label_display(v))
            for v, c in items
        ]
        n = len(handles)
        if n <= 15:
            ax.legend(handles=handles, title="label", loc="upper right",
                      fontsize=7)
        else:
            # Many labels: place legend outside the axes
            ncol = max(1, (n + 29) // 30)  # ~30 items per column
            ax.legend(handles=handles, title="label",
                      loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      fontsize=6, ncol=ncol, borderaxespad=0)

    def _add_paired_legend(self, ax, cmap_focal: dict, cmap_other: dict):
        if not self.params.get("show_legend", True):
            return
        missing = self.params.get("missing_label_value", -1)
        items = [v for v in cmap_focal if v != missing]
        if not items:
            return
        handles = []
        for v in items:
            disp = self._label_display(v)
            handles.append(
                plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                           markerfacecolor=cmap_focal[v], markeredgecolor="none",
                           label=f"{disp} (focal)")
            )
            handles.append(
                plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                           markerfacecolor=cmap_other[v], markeredgecolor="none",
                           label=f"{disp} (other)")
            )
        n = len(handles)
        if n <= 15:
            ax.legend(handles=handles, title="label", loc="upper right",
                      fontsize=7)
        else:
            ncol = max(1, (n + 29) // 30)
            ax.legend(handles=handles, title="label",
                      loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      fontsize=6, ncol=ncol, borderaxespad=0)

    # -----------------------------------------------------------------
    # fit / transform / save_model
    # -----------------------------------------------------------------

    def fit(self, X_iter: Iterable):
        source = self.params["source"]
        label_source = str(source.get("source", "feature")).lower()

        if label_source == "labels":
            seq_data = self._load_from_labels(source)
        else:
            seq_data = self._load_from_feature(source)

        if not seq_data:
            raise RuntimeError("[viz-timeline] No data loaded.")

        self._figs = []
        if self.params.get("per_sequence", True):
            for seq_key in sorted(seq_data):
                df = seq_data[seq_key]
                fig = self._render_timeline(df, seq_key)
                safe = to_safe_name(seq_key)
                self._figs.append((f"timeline_{safe}.png", fig))
                plt.show()
        else:
            # Combined: render all sequences stacked
            combined_parts = []
            for seq_key in sorted(seq_data):
                df = seq_data[seq_key].copy()
                df["_seq"] = seq_key
                combined_parts.append(df)
            combined = pd.concat(combined_parts, ignore_index=True)
            fig = self._render_timeline(combined, "all_sequences")
            self._figs.append(("timeline_all.png", fig))
            plt.show()

        self._summary = {
            "sequences": len(seq_data),
            "figures": len(self._figs),
        }

    def transform(self, X):
        if self._marker_written:
            return pd.DataFrame(index=[])
        self._marker_written = True
        return pd.DataFrame([{
            "outputs": ",".join(fname for fname, _ in self._figs),
            "source_feature": self.params["source"].get("feature", ""),
        }])

    def save_model(self, path: Path):
        run_root = path.parent
        dpi = int(self.params.get("dpi", 150))
        for fname, fig in self._figs:
            fig.savefig(run_root / fname, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        joblib.dump(
            {"params": self.params, "summary": self._summary,
             "files": [fname for fname, _ in self._figs]},
            run_root / "viz.joblib",
        )

    def load_model(self, path: Path):
        bundle = joblib.load(path)
        self.params.update(bundle.get("params", {}))
        self._summary = bundle.get("summary", {})
