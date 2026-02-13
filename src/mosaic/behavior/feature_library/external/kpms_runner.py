"""
Standalone runner for keypoint-moseq operations.

This script is called via subprocess by the mosaic kpms-fit / kpms-apply features.
It communicates entirely through files on disk (npz, json, joblib).

keypoint-moseq must NOT be installed in the mosaic environment (numpy conflicts).
Instead, create a dedicated environment and pass its interpreter path via the
``kpms_python`` parameter in the mosaic feature config.

Environment setup (CPU — macOS / Linux)::

    conda create -n kpms python=3.10 -y
    conda activate kpms
    pip install "numpy<2" joblib
    pip install keypoint-moseq

GPU acceleration (Linux with CUDA)::

    conda create -n kpms python=3.10 -y
    conda activate kpms
    pip install "numpy<2" joblib
    pip install keypoint-moseq
    pip install --upgrade "jax[cuda12]"

Use ``cuda11`` instead of ``cuda12`` if on an older CUDA toolkit.

The two-step install matters: keypoint-moseq pins numpy<=1.26.4 but pip
may resolve a numpy 2.x first, which breaks bokeh (a transitive dependency).
Installing ``numpy<2`` first avoids this.

After installation, verify::

    python -c "import keypoint_moseq; import numpy; print(numpy.__version__)"
    python -c "import jax; print(jax.devices())"  # [GpuDevice(...)] or [CpuDevice(...)]

See: https://github.com/dattalab/keypoint-moseq

Usage (invoked automatically by mosaic, but can also be run manually)::

    /path/to/kpms/python kpms_runner.py fit   --data-dir /path --output-dir /path --config config.json
    /path/to/kpms/python kpms_runner.py apply --data-dir /path --output-dir /path --model-dir /path --config config.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_data(data_dir: Path) -> tuple[dict, dict, list[str]]:
    """Load coordinate and confidence arrays from data_dir.

    Expects:
      data_dir/coordinates.npz  — one key per recording, each (T, K, 2)
      data_dir/confidences.npz  — one key per recording, each (T, K)
      data_dir/metadata.json    — {"bodyparts": [...], "recording_keys": [...]}
    """
    coords_npz = np.load(data_dir / "coordinates.npz", allow_pickle=False)
    conf_npz = np.load(data_dir / "confidences.npz", allow_pickle=False)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    recording_keys = meta["recording_keys"]
    bodyparts = meta["bodyparts"]

    coordinates = {k: coords_npz[k] for k in recording_keys}
    confidences = {k: conf_npz[k] for k in recording_keys}

    return coordinates, confidences, bodyparts


def _load_config(config_path: Path) -> dict:
    """Load the kpms configuration JSON."""
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FIT command
# ---------------------------------------------------------------------------

def cmd_fit(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(Path(args.config))

    print("[kpms_runner:fit] Loading data...", file=sys.stderr)
    coordinates, confidences, bodyparts = _load_data(data_dir)
    print(f"[kpms_runner:fit] {len(coordinates)} recordings, {len(bodyparts)} bodyparts", file=sys.stderr)

    # Build kpms config
    use_bodyparts = config.get("use_bodyparts", bodyparts)
    anterior = config.get("anterior_bodyparts", [])
    posterior = config.get("posterior_bodyparts", [])

    # Validate: anterior/posterior bodyparts must be in use_bodyparts
    bad_ant = [b for b in anterior if b not in use_bodyparts]
    bad_post = [b for b in posterior if b not in use_bodyparts]
    if bad_ant:
        print(f"[kpms_runner:fit] WARN: anterior_bodyparts {bad_ant} not in use_bodyparts, removing them", file=sys.stderr)
        anterior = [b for b in anterior if b in use_bodyparts]
    if bad_post:
        print(f"[kpms_runner:fit] WARN: posterior_bodyparts {bad_post} not in use_bodyparts, removing them", file=sys.stderr)
        posterior = [b for b in posterior if b in use_bodyparts]

    # Compute integer indices into use_bodyparts (required by jax-moseq)
    anterior_idxs = [use_bodyparts.index(bp) for bp in anterior] if anterior else []
    posterior_idxs = [use_bodyparts.index(bp) for bp in posterior] if posterior else []
    print(f"[kpms_runner:fit] anterior_idxs={anterior_idxs}, posterior_idxs={posterior_idxs}", file=sys.stderr)

    latent_dim = config.get("latent_dim", 10)

    kpms_config = {
        "bodyparts": bodyparts,
        "use_bodyparts": use_bodyparts,
        "anterior_bodyparts": anterior,
        "posterior_bodyparts": posterior,
        "anterior_idxs": anterior_idxs,
        "posterior_idxs": posterior_idxs,
        "fps": config.get("fps", 30),
        "latent_dim": latent_dim,

        # Default hyperparameters (required by jax-moseq init_model)
        "trans_hypparams": {"alpha": 5.7, "gamma": 1000.0, "kappa": 1e6, "num_states": 100},
        "ar_hypparams": {"K_0_scale": 10.0, "S_0_scale": 0.01, "latent_dim": latent_dim, "nlags": 3},
        "obs_hypparams": {"nu_s": 5, "nu_sigma": 1e5, "sigmasq_0": 0.1, "sigmasq_C": 0.1},
        "cen_hypparams": {"sigmasq_loc": 0.5},
        "error_estimator": {"intercept": 0.25, "slope": -0.5},
    }

    # Log recording shapes for diagnostics
    for k, v in coordinates.items():
        print(f"[kpms_runner:fit]   {k}: coords={v.shape}, conf={confidences[k].shape}", file=sys.stderr)

    # Outlier removal
    if config.get("remove_outliers", True):
        print("[kpms_runner:fit] Removing outliers...", file=sys.stderr)
        try:
            coordinates, confidences = kpms.outlier_removal(
                coordinates,
                confidences,
                str(output_dir),
                overwrite=True,
                outlier_scale_factor=config.get("outlier_scale_factor", 6.0),
                bodyparts=bodyparts,
                use_bodyparts=kpms_config["use_bodyparts"],
            )
        except Exception as e:
            print(f"[kpms_runner:fit] WARN: outlier removal failed: {e}", file=sys.stderr)

    # Format data
    print("[kpms_runner:fit] Formatting data...", file=sys.stderr)
    data, metadata = kpms.format_data(coordinates, confidences, **kpms_config)

    # Convert data arrays to 64-bit (format_data returns 32-bit,
    # but init_model requires 64-bit when jax_enable_x64 is True)
    import jax.numpy as jnp
    for k in list(data.keys()):
        if hasattr(data[k], 'dtype') and data[k].dtype != jnp.float64:
            data[k] = jnp.array(data[k], dtype=jnp.float64) if 'mask' not in k else jnp.array(data[k])

    # Log formatted data shapes
    for k, v in data.items():
        shape = getattr(v, 'shape', '?')
        dtype = getattr(v, 'dtype', '?')
        print(f"[kpms_runner:fit]   data['{k}']: {shape} {dtype}", file=sys.stderr)

    # Fit PCA
    print("[kpms_runner:fit] Fitting PCA...", file=sys.stderr)
    pca = kpms.fit_pca(**data, **kpms_config)

    # Initialize model
    print("[kpms_runner:fit] Initializing model...", file=sys.stderr)
    model = kpms.init_model(data, pca=pca, **kpms_config)

    # Optionally set kappa for AR-HMM
    kappa_ar = config.get("kappa_ar")
    if kappa_ar is not None:
        model = kpms.update_hypparams(model, kappa=kappa_ar)

    # Fit AR-HMM
    num_ar = int(config.get("num_iters_ar", 50))
    if num_ar > 0:
        print(f"[kpms_runner:fit] Fitting AR-HMM ({num_ar} iters)...", file=sys.stderr)
        result = kpms.fit_model(
            model, data, metadata,
            project_dir=str(output_dir),
            ar_only=True,
            num_iters=num_ar,
            save_every_n_iters=max(25, num_ar),
        )
        model = result[0] if isinstance(result, tuple) else result

    # Fit full model
    num_full = int(config.get("num_iters_full", 500))
    if num_full > 0:
        kappa_full = config.get("kappa_full")
        if kappa_full is not None:
            model = kpms.update_hypparams(model, kappa=kappa_full)

        print(f"[kpms_runner:fit] Fitting full model ({num_full} iters)...", file=sys.stderr)
        result = kpms.fit_model(
            model, data, metadata,
            project_dir=str(output_dir),
            ar_only=False,
            start_iter=num_ar,
            num_iters=num_ar + num_full,
            save_every_n_iters=max(25, num_full),
        )
        model = result[0] if isinstance(result, tuple) else result

    # Save outputs using joblib (cross-environment compatible)
    import joblib
    joblib.dump({
        "model": model,
        "pca": pca,
        "metadata": metadata,
        "kpms_config": kpms_config,
        "bodyparts": bodyparts,
    }, output_dir / "kpms_model.joblib")

    print("[kpms_runner:fit] Done. Model saved.", file=sys.stderr)


# ---------------------------------------------------------------------------
# APPLY command
# ---------------------------------------------------------------------------

def cmd_apply(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms
    import joblib

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(Path(args.config))

    # Load fitted model
    print("[kpms_runner:apply] Loading model...", file=sys.stderr)
    bundle = joblib.load(model_dir / "kpms_model.joblib")
    model = bundle["model"]
    kpms_config = bundle["kpms_config"]
    bodyparts = bundle["bodyparts"]

    # Load new data
    print("[kpms_runner:apply] Loading data...", file=sys.stderr)
    coordinates, confidences, _ = _load_data(data_dir)
    print(f"[kpms_runner:apply] {len(coordinates)} recordings", file=sys.stderr)

    # Format data
    data, metadata = kpms.format_data(coordinates, confidences, **kpms_config)

    # Convert data arrays to 64-bit (same as in cmd_fit)
    import jax.numpy as jnp
    for k in list(data.keys()):
        if hasattr(data[k], 'dtype') and data[k].dtype != jnp.float64:
            data[k] = jnp.array(data[k], dtype=jnp.float64) if 'mask' not in k else jnp.array(data[k])

    # Apply model
    num_iters = int(config.get("num_iters_apply", 500))
    model_name = "applied"
    (output_dir / model_name).mkdir(parents=True, exist_ok=True)
    print(f"[kpms_runner:apply] Applying model ({num_iters} iters)...", file=sys.stderr)
    results = kpms.apply_model(
        model, data, metadata,
        project_dir=str(output_dir),
        model_name=model_name,
        num_iters=num_iters,
        **kpms_config,
    )

    # Save per-recording results as individual npz files
    for recording_name, rec_data in results.items():
        syllables = rec_data.get("syllable", rec_data.get("syllables"))
        if syllables is None:
            print(f"[kpms_runner:apply] WARN: no syllables for {recording_name}", file=sys.stderr)
            continue

        syllables = np.asarray(syllables, dtype=np.int32).ravel()
        out_path = output_dir / f"syllables__{recording_name}.npz"
        np.savez_compressed(out_path, syllables=syllables, recording=recording_name)

    # Save list of processed recordings
    processed = list(results.keys())
    with open(output_dir / "processed_recordings.json", "w") as f:
        json.dump(processed, f)

    print(f"[kpms_runner:apply] Done. {len(processed)} recordings processed.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Keypoint-MoSeq runner for mosaic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fit
    p_fit = subparsers.add_parser("fit", help="Fit AR-HMM model")
    p_fit.add_argument("--data-dir", required=True, help="Directory with coordinates.npz, confidences.npz, metadata.json")
    p_fit.add_argument("--output-dir", required=True, help="Directory to save model outputs")
    p_fit.add_argument("--config", required=True, help="Path to config JSON")

    # apply
    p_apply = subparsers.add_parser("apply", help="Apply fitted model")
    p_apply.add_argument("--data-dir", required=True, help="Directory with coordinates.npz, confidences.npz, metadata.json")
    p_apply.add_argument("--output-dir", required=True, help="Directory to save syllable outputs")
    p_apply.add_argument("--model-dir", required=True, help="Directory containing kpms_model.joblib")
    p_apply.add_argument("--config", required=True, help="Path to config JSON")

    args = parser.parse_args()

    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "apply":
        cmd_apply(args)


if __name__ == "__main__":
    main()
