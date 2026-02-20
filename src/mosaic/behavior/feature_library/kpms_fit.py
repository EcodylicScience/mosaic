"""
Keypoint-MoSeq global model fitting feature.

Fits an AR-HMM (autoregressive hidden Markov model) to keypoint data by
invoking keypoint-moseq in a subprocess. The kpms package does NOT need
to be installed in the mosaic environment — only in a separate Python
environment whose interpreter path is passed via the ``kpms_python`` param.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from mosaic.core.dataset import register_feature, _dataset_base_dir
from mosaic.core.helpers import to_safe_name
from .helpers import _merge_params, _build_index_row


# ---------------------------------------------------------------------------
# Path to the runner script shipped alongside this module
# ---------------------------------------------------------------------------

_RUNNER_SCRIPT = Path(__file__).parent / "external" / "kpms_runner.py"


# ---------------------------------------------------------------------------
# Data conversion: mosaic tracks → keypoint-moseq format on disk
# ---------------------------------------------------------------------------

def _tracks_df_to_kpms_arrays(
    df: pd.DataFrame,
    pose_prefix_x: str = "poseX",
    pose_prefix_y: str = "poseY",
    pose_confidence_prefix: str = "poseP",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a single-sequence track DataFrame to keypoint-moseq arrays.

    Returns
    -------
    coords : ndarray, shape (T, K, 2)
    confidences : ndarray, shape (T, K)
    """
    x_cols = sorted(
        [c for c in df.columns if c.startswith(pose_prefix_x)],
        key=lambda c: int(c[len(pose_prefix_x):]),
    )
    y_cols = [f"{pose_prefix_y}{c[len(pose_prefix_x):]}" for c in x_cols]
    K = len(x_cols)
    T = len(df)

    if K == 0:
        raise ValueError(
            f"No keypoint columns found with prefix '{pose_prefix_x}'. "
            f"Available columns: {list(df.columns)}"
        )

    coords = np.empty((T, K, 2), dtype=np.float32)
    coords[:, :, 0] = df[x_cols].to_numpy(dtype=np.float32)
    coords[:, :, 1] = df[y_cols].to_numpy(dtype=np.float32)

    conf_cols = [f"{pose_confidence_prefix}{c[len(pose_prefix_x):]}" for c in x_cols]
    if all(c in df.columns for c in conf_cols):
        confidences = df[conf_cols].to_numpy(dtype=np.float32)
    else:
        confidences = np.ones((T, K), dtype=np.float32)

    # Replace Inf/NaN coords with NaN and zero out their confidences.
    # kpms.format_data will interpolate NaN coordinates and down-weight
    # them using the near-zero confidence values.
    bad_mask = ~np.isfinite(coords)  # (T, K, 2)
    bad_any = bad_mask.any(axis=2)   # (T, K) — True if either x or y is bad
    n_bad = bad_any.sum()
    if n_bad > 0:
        coords[bad_mask] = np.nan
        confidences[bad_any] = 0.0
        pct = 100 * n_bad / (T * K)
        print(
            f"[kpms-fit] Replaced {n_bad} non-finite keypoint observations ({pct:.1f}%) with NaN",
            file=sys.stderr,
        )

    return coords, confidences


def _collect_and_serialize_tracks(
    ds,
    data_dir: Path,
    pose_prefix_x: str,
    pose_prefix_y: str,
    pose_confidence_prefix: str,
    id_col: str = "id",
    bodypart_names: Optional[list[str]] = None,
    groups: Optional[list[str]] = None,
    sequences: Optional[list[str]] = None,
    fit_sample_sequences: Optional[int] = None,
    downsample_rate: Optional[int] = None,
) -> list[str]:
    """Iterate all track parquets, convert, and write to data_dir as npz+json.

    Writes:
      data_dir/coordinates.npz  — one key per recording
      data_dir/confidences.npz  — one key per recording
      data_dir/metadata.json    — {bodyparts, recording_keys}

    Parameters
    ----------
    fit_sample_sequences : int or None
        If set, randomly subsample this many recordings (deterministic seed).
    downsample_rate : int or None
        If set, keep every Nth frame to reduce memory.

    Returns
    -------
    bodypart_names : list[str]
    """
    tracks_root = ds.get_root("tracks")
    idx_path = tracks_root / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"tracks/index.csv not found at {idx_path}")

    df_idx = pd.read_csv(idx_path)
    df_idx["group"] = df_idx["group"].fillna("").astype(str)
    df_idx["sequence"] = df_idx["sequence"].fillna("").astype(str)

    # Apply group/sequence filters
    if groups is not None:
        df_idx = df_idx[df_idx["group"].isin(groups)]
    if sequences is not None:
        df_idx = df_idx[df_idx["sequence"].isin(sequences)]

    coordinates: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    n_keypoints = 0
    total_frames = 0

    for _, row in df_idx.iterrows():
        abs_path_str = str(row.get("abs_path", ""))
        if not abs_path_str.endswith(".parquet"):
            continue
        p = ds.resolve_path(abs_path_str)
        if not p.exists():
            print(f"[kpms-fit] WARN: missing track parquet: {p}", file=sys.stderr)
            continue

        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[kpms-fit] WARN: failed to read {p}: {e}", file=sys.stderr)
            continue

        if len(df) <= 1:
            continue

        g = str(row.get("group", ""))
        s = str(row.get("sequence", ""))
        # Use safe names to avoid '/' in keys (breaks hdf5 inside kpms)
        g_safe = to_safe_name(g) if g else ""
        s_safe = to_safe_name(s)

        if id_col in df.columns:
            for ind_id, sub in df.groupby(id_col, sort=False):
                sub = sub.sort_values("frame" if "frame" in sub.columns else sub.columns[0])
                sub = sub.reset_index(drop=True)
                key = f"{g_safe}__{s_safe}__id{ind_id}" if g_safe else f"{s_safe}__id{ind_id}"
                try:
                    coords, conf = _tracks_df_to_kpms_arrays(
                        sub, pose_prefix_x, pose_prefix_y, pose_confidence_prefix
                    )
                except ValueError:
                    continue
                # Temporal downsampling
                if downsample_rate and int(downsample_rate) > 1:
                    rate = int(downsample_rate)
                    coords = coords[::rate]
                    conf = conf[::rate]
                coordinates[key] = coords
                confidences[key] = conf
                n_keypoints = coords.shape[1]
                total_frames += coords.shape[0]
        else:
            key = f"{g_safe}__{s_safe}" if g_safe else s_safe
            try:
                coords, conf = _tracks_df_to_kpms_arrays(
                    df, pose_prefix_x, pose_prefix_y, pose_confidence_prefix
                )
            except ValueError:
                continue
            # Temporal downsampling
            if downsample_rate and int(downsample_rate) > 1:
                rate = int(downsample_rate)
                coords = coords[::rate]
                conf = conf[::rate]
            coordinates[key] = coords
            confidences[key] = conf
            n_keypoints = coords.shape[1]
            total_frames += coords.shape[0]

    if not coordinates:
        raise RuntimeError("[kpms-fit] No valid track data found.")

    # Subsample recordings if requested
    if fit_sample_sequences is not None and len(coordinates) > int(fit_sample_sequences):
        n_sample = int(fit_sample_sequences)
        all_keys = sorted(coordinates.keys())
        rng = np.random.RandomState(42)
        sampled_keys = set(rng.choice(all_keys, size=n_sample, replace=False))
        dropped = len(coordinates) - n_sample
        coordinates = {k: v for k, v in coordinates.items() if k in sampled_keys}
        confidences = {k: v for k, v in confidences.items() if k in sampled_keys}
        total_frames = sum(v.shape[0] for v in coordinates.values())
        print(
            f"[kpms-fit] Subsampled {n_sample} of {n_sample + dropped} recordings for fitting.",
            file=sys.stderr,
        )

    # Auto-generate bodypart names if needed
    if bodypart_names is None:
        bodypart_names = [f"kp{i}" for i in range(n_keypoints)]

    # Write to disk
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / "coordinates.npz", **coordinates)
    np.savez(data_dir / "confidences.npz", **confidences)
    with open(data_dir / "metadata.json", "w") as f:
        json.dump({
            "bodyparts": bodypart_names,
            "recording_keys": list(coordinates.keys()),
        }, f, indent=2)

    ds_msg = f" (downsampled {downsample_rate}x)" if downsample_rate and int(downsample_rate) > 1 else ""
    print(
        f"[kpms-fit] Serialized {len(coordinates)} recordings, "
        f"{n_keypoints} keypoints, {total_frames:,} total frames{ds_msg}.",
        file=sys.stderr,
    )
    return bodypart_names


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------

def _run_kpms_subprocess(
    kpms_python: str,
    command: str,
    args: list[str],
    label: str = "kpms",
) -> None:
    """Invoke the kpms_runner.py script in a subprocess.

    Streams stderr line-by-line so tqdm progress bars and diagnostic
    messages are visible in real-time rather than buffered until exit.
    """
    cmd = [kpms_python, str(_RUNNER_SCRIPT), command] + args

    print(f"[{label}] Running: {' '.join(cmd)}", file=sys.stderr)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
        env=env,
    )

    # Stream stderr in real-time (progress bars, diagnostics)
    stderr_lines = []
    for line in proc.stderr:
        stripped = line.rstrip()
        stderr_lines.append(stripped)
        print(f"  {stripped}", file=sys.stderr, flush=True)

    proc.wait()
    stdout = proc.stdout.read() if proc.stdout else ""

    if proc.returncode != 0:
        raise RuntimeError(
            f"[{label}] subprocess failed (exit {proc.returncode}).\n"
            f"stderr:\n" + "\n".join(stderr_lines) + "\n"
            f"stdout:\n{stdout}"
        )


# ---------------------------------------------------------------------------
# Feature class
# ---------------------------------------------------------------------------

@register_feature
class KpmsFit:
    """
    Global feature that fits a keypoint-MoSeq AR-HMM model via subprocess.

    keypoint-moseq runs in a separate Python environment to avoid dependency
    conflicts. Pass the interpreter path via ``kpms_python``.

    Parameters
    ----------
    kpms_python : str
        Path to Python interpreter with keypoint-moseq installed.
        Example: "~/miniforge3/envs/kpms/bin/python"
    pose_prefix_x, pose_prefix_y : str
        Column prefixes for keypoint x/y coordinates (default: "poseX", "poseY").
    pose_confidence_prefix : str
        Column prefix for confidence scores (default: "poseP").
    bodypart_names : list[str] or None
        Names for each keypoint. Auto-generated as ["kp0", ...] if None.
    use_bodyparts : list[str] or None
        Subset of bodyparts to use for modeling. None = all.
    anterior_bodyparts, posterior_bodyparts : list[str] or None
        Bodyparts used for rotational alignment.
    fps : int
        Frame rate of the input data.
    num_iters_ar : int
        Gibbs sampling iterations for AR-HMM fitting (default: 50).
    num_iters_full : int
        Iterations for full model fitting (default: 500). Set to 0 to skip.
    kappa_ar, kappa_full : float or None
        Kappa hyperparameter controlling syllable duration.
    latent_dim : int
        Number of PCA dimensions for pose trajectory (default: 10).
    outlier_scale_factor : float
        Stringency of outlier detection (default: 6.0).
    remove_outliers : bool
        Whether to run outlier removal (default: True).
    resume : bool
        Auto-resume from checkpoint if a previous run was interrupted (default: True).
    fit_sample_sequences : int or None
        Randomly sample this many recordings for fitting (default: None = use all).
        Recommended for large datasets: fit on a subset, then apply_model to everything.
    downsample_rate : int or None
        Temporal downsampling factor (default: None). E.g., 2 keeps every 2nd frame.
        Reduces memory proportionally.
    """

    name = "kpms-fit"
    version = "0.1"
    parallelizable = False
    output_type = "global"
    skip_transform_phase = True

    _defaults = dict(
        # Subprocess config
        kpms_python=None,  # REQUIRED

        # Column detection
        pose_prefix_x="poseX",
        pose_prefix_y="poseY",
        pose_confidence_prefix="poseP",

        # Keypoint-moseq config
        bodypart_names=None,
        use_bodyparts=None,
        anterior_bodyparts=None,
        posterior_bodyparts=None,
        fps=30,

        # Fitting params
        num_iters_ar=50,
        num_iters_full=500,
        kappa_ar=None,
        kappa_full=None,
        latent_dim=10,
        location_aware=False,
        outlier_scale_factor=6.0,
        remove_outliers=True,

        # Memory management
        mixed_map_iters=None,           # Split into N serial batches (reduces VRAM ~N×)
        parallel_message_passing=None,  # None=auto (GPU→True, CPU→False). False saves 4-6× VRAM

        # Scaling
        resume=True,                    # Auto-resume from checkpoint if interrupted
        fit_sample_sequences=None,      # Subsample N recordings for fitting (None=all)
        downsample_rate=None,           # Temporal downsampling factor (None=no downsampling)
        save_every_n_iters=25,          # Checkpoint frequency

        # ID handling
        id_col="id",
        seq_col="sequence",
        group_col="group",
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self._ds = None
        self._run_root: Optional[Path] = None
        self._additional_index_rows: list[dict] = []
        self._scope_filter_dict: Optional[dict] = None
        self._scope_constraints: Optional[dict] = None

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        self._ds = ds

    def set_run_root(self, run_root: Path) -> None:
        self._run_root = Path(run_root)

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter_dict = scope

    def set_scope_constraints(self, constraints: Optional[dict]) -> None:
        self._scope_constraints = constraints

    def get_additional_index_rows(self) -> list[dict]:
        return list(self._additional_index_rows)

    # ----------------------- Feature protocol --------------------

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        return True

    def partial_fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=[])

    # ----------------------- Fit ---------------------------------

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("[kpms-fit] Dataset not bound.")

        p = self.params
        kpms_python = p.get("kpms_python")
        if not kpms_python:
            raise ValueError(
                "[kpms-fit] 'kpms_python' param is required. "
                "Set it to the path of a Python interpreter with keypoint-moseq installed. "
                "Example: '~/miniforge3/envs/kpms/bin/python'"
            )
        kpms_python = str(Path(kpms_python).expanduser())

        self._additional_index_rows = []

        if self._run_root is None:
            raise RuntimeError("[kpms-fit] run_root not set.")

        # 1. Serialize track data to a temp directory
        data_dir = self._run_root / "_kpms_data"

        # Extract group/sequence scope from constraints set by run_feature
        scope_groups = None
        scope_sequences = None
        if self._scope_constraints:
            scope_groups = self._scope_constraints.get("groups")
            scope_sequences = self._scope_constraints.get("sequences")

        print("[kpms-fit] Collecting and serializing track data...", file=sys.stderr)
        bodypart_names = _collect_and_serialize_tracks(
            self._ds,
            data_dir,
            p["pose_prefix_x"],
            p["pose_prefix_y"],
            p["pose_confidence_prefix"],
            p["id_col"],
            p.get("bodypart_names"),
            groups=scope_groups,
            sequences=scope_sequences,
            fit_sample_sequences=p.get("fit_sample_sequences"),
            downsample_rate=p.get("downsample_rate"),
        )

        # 2. Write kpms config JSON
        kpms_config = {
            "bodyparts": bodypart_names,
            "use_bodyparts": p.get("use_bodyparts") or bodypart_names,
            "anterior_bodyparts": p.get("anterior_bodyparts") or [],
            "posterior_bodyparts": p.get("posterior_bodyparts") or [],
            "fps": p["fps"],
            "latent_dim": p["latent_dim"],
            "num_iters_ar": p["num_iters_ar"],
            "num_iters_full": p["num_iters_full"],
            "kappa_ar": p.get("kappa_ar"),
            "kappa_full": p.get("kappa_full"),
            "location_aware": p.get("location_aware", False),
            "outlier_scale_factor": p.get("outlier_scale_factor", 6.0),
            "remove_outliers": p.get("remove_outliers", True),
            "mixed_map_iters": p.get("mixed_map_iters"),
            "parallel_message_passing": p.get("parallel_message_passing"),
            "save_every_n_iters": p.get("save_every_n_iters", 25),
        }
        config_path = self._run_root / "kpms_config.json"
        config_path.write_text(json.dumps(kpms_config, indent=2, default=str))

        # 3. Run kpms_runner.py fit in subprocess
        output_dir = self._run_root / "_kpms_output"
        fit_args = [
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--config", str(config_path),
        ]
        if p.get("resume", True):
            fit_args.append("--resume")

        _run_kpms_subprocess(
            kpms_python,
            "fit",
            fit_args,
            label="kpms-fit",
        )

        # 4. Verify model was saved
        model_file = output_dir / "kpms_model.joblib"
        if not model_file.exists():
            raise RuntimeError(
                f"[kpms-fit] kpms_runner did not produce model at {model_file}"
            )

        print("[kpms-fit] Model fitting complete.", file=sys.stderr)

    # ----------------------- Save / Load -------------------------

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        self._additional_index_rows = []

        # The actual kpms model was saved by the subprocess into _kpms_output/.
        # Copy it to the standard model.joblib location for discoverability.
        kpms_model_src = run_root / "_kpms_output" / "kpms_model.joblib"
        if kpms_model_src.exists():
            shutil.copy2(kpms_model_src, path)
        else:
            # Write a placeholder so the feature index knows fit completed
            import joblib
            joblib.dump({"params": self.params, "version": self.version}, path)

        # Write marker parquet for index registration
        marker_seq = "__global__"
        safe_marker_seq = to_safe_name(marker_seq)
        marker_path = run_root / f"{safe_marker_seq}.parquet"
        marker_df = pd.DataFrame({"run_marker": [True]})
        marker_df.to_parquet(marker_path, index=False)
        self._additional_index_rows.append(
            _build_index_row(safe_marker_seq, "", marker_seq, marker_path, 1,
                             dataset_root=_dataset_base_dir(self._ds) if self._ds else None)
        )

    def load_model(self, path: Path) -> None:
        pass  # Model loading happens in kpms-apply via subprocess
