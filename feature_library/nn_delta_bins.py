from pathlib import Path
from typing import Optional, Dict, Any, Iterable, Sequence
import numpy as np
import pandas as pd

from behavior.dataset import register_feature


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _binned_mean_fast(xvals: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(xvals) == 0:
        return np.full(len(edges) - 1, np.nan)
    bins = np.digitize(xvals, edges) - 1
    mask = (bins >= 0) & (bins < len(edges) - 1) & np.isfinite(values)
    if not mask.any():
        return np.full(len(edges) - 1, np.nan)
    bins = bins[mask]
    values = values[mask]
    counts = np.bincount(bins, minlength=len(edges) - 1)
    sums = np.bincount(bins, weights=values, minlength=len(edges) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return means


def _mean_binned_force(dfsel: pd.DataFrame, edges: np.ndarray, max_for_avg: float, antisymm: bool):
    """
    Compute 1D binned means for turn (dangle) and speed (dspeed) responses.
    Mirrors the original get1Dhists behavior with symmetry handling.
    """
    if dfsel.empty:
        n_bins = len(edges) - 1
        return (np.full(n_bins, np.nan), np.full(n_bins, np.nan))

    # Turn (front/back symmetry)
    dfturn = dfsel[(dfsel["neighbor_x"] >= 0) & (dfsel["neighbor_x"] <= max_for_avg)]
    if antisymm:
        y_data_turn = np.concatenate([dfturn["neighbor_y"].to_numpy(), -dfturn["neighbor_y"].to_numpy()])
        dangle_values_turn = np.concatenate([dfturn["dangle"].to_numpy(), -dfturn["dangle"].to_numpy()])
    else:
        y_data_turn = dfturn["neighbor_y"].to_numpy()
        dangle_values_turn = dfturn["dangle"].to_numpy()

    # Speed (left/right symmetry already captured)
    dfspeed = dfsel[np.abs(dfsel["neighbor_y"]) <= max_for_avg / 2]
    x_data_speed = dfspeed["neighbor_x"].to_numpy()
    dspeed_values_speed = dfspeed["dspeed"].to_numpy()

    turnforces = _binned_mean_fast(y_data_turn, dangle_values_turn, edges)
    speedforces = _binned_mean_fast(x_data_speed, dspeed_values_speed, edges)
    return turnforces, speedforces


def _maybe_make_category(df: pd.DataFrame, spec: dict) -> pd.Series:
    """
    Optional helper to derive a category column from a source column and a quantile threshold.
    spec:
      - source_col: column name in df
      - new_col: name to create
      - quantile: float in (0,1)
      - op: '>' or '<=' (default '>')
    Returns the created series (or an empty Series if skipped).
    """
    source_col = spec.get("source_col")
    new_col = spec.get("new_col")
    if not source_col or not new_col or source_col not in df.columns:
        return pd.Series(dtype=float)
    q = float(spec.get("quantile", 0.75))
    op = str(spec.get("op", ">")).strip()
    thresh = df[source_col].quantile(q)
    if op == "<=":
        series = df[source_col] <= thresh
    else:
        series = df[source_col] > thresh
    return series.rename(new_col)


@register_feature
class NearestNeighborDeltaBins:
    """
    Bin nearest-neighbor response fields (dangle, dspeed) over neighbor position.

    Inputs: expect outputs from nn-delta-response (neighbor_x/neighbor_y in ego frame,
    dangle, dspeed, group_size, and focal/neighbor category columns).

    Outputs: tidy DataFrame with mean turn/speed per bin for focal role and neighbor role:
      columns: [group, sequence, exp, trial, role, category, group_size, metric, bin_idx, value]
    """

    name = "nn-delta-bins"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        nbins=45,
        binmax=14.0,
        max_for_avg=5.0,
        antisymm=True,
        focal_category_col="Focal_fish",
        neighbor_category_col="neighbor_focal",
        group_size_col="group_size",
        exp_col="Exp",
        trial_col="Trial",
        # optional derived categories
        category_specs=[],  # list of {source_col,new_col,quantile,op}
        # optional filter to exclude focal rows from neighbor-role stats
        nonfocal_flag_col="Focal_fish",
        nonfocal_flag_value=False,
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False
        self._ds = None
        self._scope_filter: Optional[dict] = None

    # ----------------------- Dataset hooks -----------------------
    def bind_dataset(self, ds):
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def finalize_fit(self) -> None:
        return

    def save_model(self, path: Path) -> None:
        return

    def load_model(self, path: Path) -> None:
        return

    # ----------------------- Core logic --------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        required = ["neighbor_x", "neighbor_y", "dangle", "dspeed"]
        # helper to coalesce suffix-split columns (e.g., focal_fish_x, focal_fish_y)
        def _coalesce_column(base: str):
            if base in df.columns:
                return
            cand = [c for c in df.columns if c.startswith(base + "_")]
            if not cand:
                return
            series = None
            for c in cand:
                if series is None:
                    series = df[c]
                else:
                    series = series.combine_first(df[c])
            if series is not None:
                df[base] = series

        # Coalesce focal/neighbor/group_size if needed before checks
        focal_col = p.get("focal_category_col")
        neighbor_col = p.get("neighbor_category_col")
        group_size_col = p.get("group_size_col")
        if focal_col:
            _coalesce_column(focal_col)
        if neighbor_col:
            _coalesce_column(neighbor_col)
        if group_size_col:
            _coalesce_column(group_size_col)

        # Drop rows missing required numeric inputs
        df = df.dropna(subset=required)
        if df.empty:
            return pd.DataFrame()

        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        # Optionally derive category columns from specs
        category_specs: Sequence[dict] = p.get("category_specs") or []
        for spec in category_specs:
            series = _maybe_make_category(df, spec)
            if not series.empty:
                df[series.name] = series

        nbins = int(p["nbins"])
        binmax = float(p["binmax"])
        max_for_avg = float(p["max_for_avg"])
        antisymm = bool(p["antisymm"])
        edges = np.linspace(-binmax, binmax, nbins)

        exp_col = p.get("exp_col") or "group"
        trial_col = p.get("trial_col") or "sequence"

        # Build identifiers
        exp_val = df[exp_col] if exp_col in df.columns else ""
        trial_val = df[trial_col] if trial_col in df.columns else ""

        # Ensure group_size exists
        if group_size_col not in df.columns:
            group_sizes = ["all"]
        else:
            group_sizes = sorted([g for g in df[group_size_col].dropna().unique().tolist() if g != ""])
            group_sizes = ["all"] + group_sizes

        results = []

        def _clean_cat(val):
            if pd.isna(val):
                return None
            if isinstance(val, (bool, np.bool_)):
                return int(val)
            if isinstance(val, (int, np.integer)):
                return int(val)
            if isinstance(val, (float, np.floating)) and val in (0.0, 1.0):
                return int(val)
            return val

        def _clean_neighbor(val):
            if val == "all":
                return "all"
            if pd.isna(val):
                return "all"
            if isinstance(val, (bool, np.bool_)):
                return int(val)
            if isinstance(val, (int, np.integer)):
                return int(val)
            if isinstance(val, (float, np.floating)) and val in (0.0, 1.0):
                return int(val)
            return val

        def _clean_group_size(val):
            if pd.isna(val):
                return "all"
            if isinstance(val, (float, np.floating)) and val.is_integer():
                return int(val)
            return val

        def _hist2d_sum_count(xvals, yvals, weights, x_edges: np.ndarray, y_edges: np.ndarray):
            xvals = np.asarray(xvals, dtype=float)
            yvals = np.asarray(yvals, dtype=float)
            weights = np.asarray(weights, dtype=float)
            w_sum, _, _ = np.histogram2d(xvals, yvals, bins=[x_edges, y_edges], weights=weights)
            count, _, _ = np.histogram2d(xvals, yvals, bins=[x_edges, y_edges])
            return w_sum, count

        def _append_rows(subdf: pd.DataFrame, role: str, center_cat, neighbor_cat):
            turnforces, speedforces = _mean_binned_force(subdf, edges, max_for_avg, antisymm)
            cat_center = _clean_cat(center_cat)
            cat_neighbor = _clean_neighbor(neighbor_cat)
            for metric, forces in (("turn", turnforces), ("speed", speedforces)):
                for bin_idx, val in enumerate(forces):
                    results.append({
                        "dim": "1d",
                        exp_col: exp_val.iloc[0] if isinstance(exp_val, pd.Series) and not exp_val.empty else "",
                        trial_col: trial_val.iloc[0] if isinstance(trial_val, pd.Series) and not trial_val.empty else "",
                        "center_category": cat_center,
                        "neighbor_category": cat_neighbor,
                        group_size_col if group_size_col else "group_size": _clean_group_size(current_group_size),
                        "metric": metric,
                        "bin_idx": bin_idx,
                        "value": val,
                        "bin_x": np.nan,
                        "bin_y": np.nan,
                        "sum_value": np.nan,
                        "count": np.nan,
                    })
            # 2D bins (sum and count, no normalization)
            x_edges = np.linspace(-binmax, binmax, nbins)
            y_edges = np.linspace(-binmax, binmax, nbins + 1)
            x_base = np.asarray(subdf["neighbor_x"], dtype=float)
            y_base = np.asarray(subdf["neighbor_y"], dtype=float)
            dangle = np.asarray(subdf["dangle"], dtype=float)
            dspeed = np.asarray(subdf["dspeed"], dtype=float)
            base_mask = np.isfinite(x_base) & np.isfinite(y_base)
            x_base = x_base[base_mask]
            y_base = y_base[base_mask]
            dangle = dangle[base_mask]
            dspeed = dspeed[base_mask]

            # turn (antisymmetric option)
            if antisymm:
                x_turn = np.concatenate([x_base, x_base])
                y_turn = np.concatenate([y_base, -y_base])
                w_turn = np.concatenate([dangle, -dangle])
            else:
                x_turn, y_turn, w_turn = x_base, y_base, dangle
            turn_sum, turn_count = _hist2d_sum_count(x_turn, y_turn, w_turn, x_edges, y_edges)

            # speed (symmetric duplication)
            x_speed = np.concatenate([x_base, x_base])
            y_speed = np.concatenate([y_base, -y_base])
            w_speed = np.concatenate([dspeed, dspeed])
            speed_sum, speed_count = _hist2d_sum_count(x_speed, y_speed, w_speed, x_edges, y_edges)

            for metric, sum_arr, cnt_arr in (("turn", turn_sum, turn_count), ("speed", speed_sum, speed_count)):
                for ix in range(sum_arr.shape[0]):
                    for iy in range(sum_arr.shape[1]):
                        results.append({
                            "dim": "2d",
                            exp_col: exp_val.iloc[0] if isinstance(exp_val, pd.Series) and not exp_val.empty else "",
                            trial_col: trial_val.iloc[0] if isinstance(trial_val, pd.Series) and not trial_val.empty else "",
                            "center_category": cat_center,
                            "neighbor_category": cat_neighbor,
                            group_size_col if group_size_col else "group_size": _clean_group_size(current_group_size),
                            "metric": metric,
                            "bin_idx": np.nan,
                            "value": np.nan,
                            "bin_x": ix,
                            "bin_y": iy,
                            "sum_value": sum_arr[ix, iy],
                            "count": cnt_arr[ix, iy],
                        })

        # Focal role
        if focal_col and focal_col in df.columns:
            grouped_center = df.groupby([focal_col, group_size_col], dropna=False) if group_size_col in df.columns else df.groupby(focal_col, dropna=False)
            for keys, df_group in grouped_center:
                if group_size_col in df.columns:
                    center_cat, current_group_size = keys
                else:
                    center_cat, current_group_size = keys, "all"
                if current_group_size != "all" and current_group_size not in group_sizes:
                    continue
                # all neighbors (None to keep dtype numeric/nullable)
                _append_rows(df_group, "focal", center_cat, None)
                # per-neighbor category
                if neighbor_col in df_group.columns:
                    for neighbor_cat, df_neigh in df_group.groupby(neighbor_col, dropna=False):
                        _append_rows(df_neigh, "focal", center_cat, neighbor_cat)

        # Neighbor-role binning was originally used to represent "nonfocal response" separately.
        # With unified center/neighbor categories (no explicit role column), this is redundant
        # when focal_col is present (it duplicates the center=0 rows). Only run it if we lack
        # a focal_col; otherwise skip to avoid duplicate entries.
        if not (focal_col and focal_col in df.columns):
            nf_flag_col = p.get("nonfocal_flag_col")
            nf_flag_val = p.get("nonfocal_flag_value")
            neighbor_df = df
            if nf_flag_col in df.columns:
                neighbor_df = df[df[nf_flag_col] == nf_flag_val]

            if neighbor_col in neighbor_df.columns:
                if focal_col and focal_col in neighbor_df.columns:
                    grouped_n = neighbor_df.groupby([focal_col, neighbor_col, group_size_col], dropna=False) if group_size_col in neighbor_df.columns else neighbor_df.groupby([focal_col, neighbor_col], dropna=False)
                    for keys, df_group in grouped_n:
                        if group_size_col in neighbor_df.columns:
                            center_cat, neighbor_cat, current_group_size = keys
                        else:
                            center_cat, neighbor_cat = keys
                            current_group_size = "all"
                        if current_group_size != "all" and current_group_size not in group_sizes:
                            continue
                        _append_rows(df_group, "neighbor", center_cat, neighbor_cat)
                else:
                    grouped_n = neighbor_df.groupby([neighbor_col, group_size_col], dropna=False) if group_size_col in neighbor_df.columns else neighbor_df.groupby(neighbor_col, dropna=False)
                    for keys, df_group in grouped_n:
                        if group_size_col in neighbor_df.columns:
                            neighbor_cat, current_group_size = keys
                        else:
                            neighbor_cat, current_group_size = keys, "all"
                        if current_group_size != "all" and current_group_size not in group_sizes:
                            continue
                        _append_rows(df_group, "neighbor", None, neighbor_cat)

        if not results:
            return pd.DataFrame()

        out_df = pd.DataFrame(results)
        if "neighbor_category" in out_df.columns:
            out_df["neighbor_category"] = (
                out_df["neighbor_category"]
                .fillna("all")
                .astype("string")  # keep homogeneous type for parquet (mix of ints/strings otherwise fails)
            )
        # Normalize category dtypes to nullable ints where possible
        if "center_category" in out_df.columns:
            out_df["center_category"] = pd.to_numeric(out_df["center_category"], errors="coerce").astype("Int64")
        # Add sequence/group if present
        for meta_col in ("group", "sequence"):
            if meta_col in df.columns and meta_col not in out_df.columns:
                out_df[meta_col] = df[meta_col].iloc[0] if not df[meta_col].empty else ""
        return out_df
