"""Persistent subprocess server for keypoint-moseq operations.

Runs in the external .venv (keypoint-moseq environment). Imports JAX and
keypoint-moseq once at startup, then serves commands over a Unix domain socket.

Commands: add_track, fit, load_model, apply, save_model, shutdown.

Wire protocol: newline-delimited JSON. Arrays are base64-encoded in the JSON
with dtype and shape metadata.

Usage::

    .venv/bin/python kpms_server.py /tmp/kpms.sock
"""

# pyright: reportAny=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import ctypes
import logging
import os
import signal
import socket
import sys
import tempfile
from typing import SupportsIndex, TypedDict

import jax
import jax.numpy as jnp
import joblib
import keypoint_moseq as kpms
import numpy as np
from jax_moseq.utils import set_mixed_map_iters
from numpy.typing import ArrayLike
from pydantic import BaseModel

from kpms_protocol import (
    AddTrackRequest,
    ApplyRequest,
    ApplyResponse,
    ErrorResponse,
    FitRequest,
    LoadModelRequest,
    OkResponse,
    ReadyResponse,
    Request,
    RequestAdapter,
    SaveModelRequest,
    ShutdownRequest,
    decode_array,
    encode_array,
    receive_message,
    send_message,
)

# Monkeypatch np.bincount for numpy >= 2.4 compatibility.
# numpy 2.4 no longer accepts minlength=None (must be int).
_original_bincount = np.bincount


def _patched_bincount(
    x: ArrayLike,
    weights: ArrayLike | None = None,
    minlength: SupportsIndex | None = None,
):
    if minlength is None:
        minlength = 0
    return _original_bincount(x, weights=weights, minlength=minlength)


np.bincount = _patched_bincount


jax.config.update("jax_enable_x64", True)

log = logging.getLogger("kpms_server")


class KpmsConfig(TypedDict):
    bodyparts: list[str]
    use_bodyparts: list[str]
    anterior_bodyparts: list[str]
    posterior_bodyparts: list[str]
    anterior_idxs: list[int]
    posterior_idxs: list[int]
    fps: int
    latent_dim: int
    trans_hypparams: dict[str, float | int]
    ar_hypparams: dict[str, float | int]
    obs_hypparams: dict[str, float | int]
    cen_hypparams: dict[str, float]
    error_estimator: dict[str, float]


# --- Orphan protection ---


def prctl_set_pdeathsig() -> None:
    """Ask the kernel to send SIGTERM when the parent process dies."""
    PR_SET_PDEATHSIG = 1
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if result != 0:
            log.warning("prctl failed")
    except OSError:
        log.warning("prctl not available")


def recv_request(conn: socket.socket) -> Request:
    """Read a newline-terminated JSON request."""
    return RequestAdapter.validate_json(receive_message(conn))


# --- Server state ---


class KpmsServer:
    def __init__(self) -> None:
        self.coordinates: dict[str, np.ndarray] = {}
        self.confidences: dict[str, np.ndarray] = {}
        self.model: object | None = None
        self.pca: object | None = None
        self.metadata: object | None = None
        self.kpms_config: KpmsConfig | None = None
        self.bodyparts: list[str] | None = None

    def handle_add_track(self, request: AddTrackRequest) -> None:
        self.coordinates[request.key] = decode_array(request.coords)
        self.confidences[request.key] = decode_array(request.conf)

    def handle_fit(self, request: FitRequest) -> None:

        config = request.config
        bodyparts = config.bodyparts
        use_bodyparts = config.use_bodyparts or bodyparts
        anterior = list(config.anterior_bodyparts)
        posterior = list(config.posterior_bodyparts)

        # Validate anterior/posterior bodyparts
        bad_ant = [b for b in anterior if b not in use_bodyparts]
        bad_post = [b for b in posterior if b not in use_bodyparts]
        if bad_ant:
            log.warning("anterior_bodyparts %s not in use_bodyparts, removing", bad_ant)
            anterior = [b for b in anterior if b in use_bodyparts]
        if bad_post:
            log.warning(
                "posterior_bodyparts %s not in use_bodyparts, removing", bad_post
            )
            posterior = [b for b in posterior if b in use_bodyparts]

        anterior_idxs = [use_bodyparts.index(bp) for bp in anterior] if anterior else []
        posterior_idxs = (
            [use_bodyparts.index(bp) for bp in posterior] if posterior else []
        )
        latent_dim = config.latent_dim

        kpms_config: KpmsConfig = {
            "bodyparts": bodyparts,
            "use_bodyparts": use_bodyparts,
            "anterior_bodyparts": anterior,
            "posterior_bodyparts": posterior,
            "anterior_idxs": anterior_idxs,
            "posterior_idxs": posterior_idxs,
            "fps": config.fps,
            "latent_dim": latent_dim,
            "trans_hypparams": {
                "alpha": 5.7,
                "gamma": 1000.0,
                "kappa": 1e6,
                "num_states": 100,
            },
            "ar_hypparams": {
                "K_0_scale": 10.0,
                "S_0_scale": 0.01,
                "latent_dim": latent_dim,
                "nlags": 3,
            },
            "obs_hypparams": {
                "nu_s": 5,
                "nu_sigma": 1e5,
                "sigmasq_0": 0.1,
                "sigmasq_C": 0.1,
            },
            "cen_hypparams": {
                "sigmasq_loc": 0.5 if config.location_aware else 1e6,
            },
            "error_estimator": {"intercept": 0.25, "slope": -0.5},
        }

        coordinates = self.coordinates
        confidences = self.confidences

        log.info("fit: %d recordings, %d bodyparts", len(coordinates), len(bodyparts))

        # Outlier removal
        checkpoint_dir = config.checkpoint_dir or tempfile.mkdtemp(prefix="kpms_")
        if config.remove_outliers:
            log.info("Removing outliers...")
            coordinates, confidences = kpms.outlier_removal(
                coordinates,
                confidences,
                checkpoint_dir,
                overwrite=True,
                outlier_scale_factor=config.outlier_scale_factor,
                bodyparts=bodyparts,
                use_bodyparts=use_bodyparts,
            )

        # Format data
        log.info("Formatting data...")
        data, metadata = kpms.format_data(coordinates, confidences, **kpms_config)

        # Convert to float64
        for k in list(data.keys()):
            if hasattr(data[k], "dtype") and data[k].dtype != jnp.float64:
                data[k] = (
                    jnp.array(data[k], dtype=jnp.float64)
                    if "mask" not in k
                    else jnp.array(data[k])
                )

        # NaN fixup in Y
        Y_np = np.array(data["Y"])
        n_nan = int(np.isnan(Y_np).sum())
        if n_nan > 0:
            Y_np = np.nan_to_num(Y_np, nan=0.0)
            data["Y"] = jnp.array(Y_np, dtype=data["Y"].dtype)
            log.info("fixed %d NaN in Y -> 0 (conf/mask untouched)", n_nan)

        # PCA
        log.info("Fitting PCA...")
        pca = kpms.fit_pca(**data, **kpms_config)

        # Init model
        log.info("Initializing model...")
        model = kpms.init_model(data, pca=pca, **kpms_config)

        # Mixed map iters
        if config.mixed_map_iters is not None and config.mixed_map_iters > 1:
            set_mixed_map_iters(config.mixed_map_iters)
            log.info("set_mixed_map_iters(%d)", config.mixed_map_iters)

        parallel_mp = config.parallel_message_passing
        save_every = config.save_every_n_iters

        # Kappa for AR-HMM
        if config.kappa_ar is not None:
            model = kpms.update_hypparams(model, kappa=config.kappa_ar)

        # Fit AR-HMM
        num_ar = config.num_iters_ar
        if num_ar > 0:
            ar_start_iter = 0
            if config.resume:
                ckpt = _try_load_checkpoint(checkpoint_dir, "ar_hmm")
                if ckpt is not None:
                    model, ar_start_iter = ckpt
                    log.info("Resuming AR-HMM from iter %d", ar_start_iter)

            if ar_start_iter < num_ar:
                log.info(
                    "Fitting AR-HMM (%d iters, start=%d)...", num_ar, ar_start_iter
                )
                result = kpms.fit_model(
                    model,
                    data,
                    metadata,
                    project_dir=checkpoint_dir,
                    model_name="ar_hmm",
                    ar_only=True,
                    start_iter=ar_start_iter,
                    num_iters=num_ar,
                    save_every_n_iters=save_every,
                    parallel_message_passing=parallel_mp,
                    plot_every_n_iters=0,
                )
                model = result[0]

        # Fit full model
        num_full = config.num_iters_full
        if num_full > 0:
            if config.kappa_full is not None:
                model = kpms.update_hypparams(model, kappa=config.kappa_full)

            full_start_iter = num_ar
            total_full_iters = num_ar + num_full

            if config.resume:
                ckpt = _try_load_checkpoint(checkpoint_dir, "full_model")
                if ckpt is not None:
                    model, full_start_iter = ckpt
                    log.info("Resuming full model from iter %d", full_start_iter)

            if full_start_iter < total_full_iters:
                log.info(
                    "Fitting full model (%d iters, start=%d)...",
                    num_full,
                    full_start_iter,
                )
                result = kpms.fit_model(
                    model,
                    data,
                    metadata,
                    project_dir=checkpoint_dir,
                    model_name="full_model",
                    ar_only=False,
                    start_iter=full_start_iter,
                    num_iters=total_full_iters,
                    save_every_n_iters=save_every,
                    parallel_message_passing=parallel_mp,
                    plot_every_n_iters=0,
                )
                model = result[0]

        # Store results
        self.model = model
        self.pca = pca
        self.metadata = metadata
        self.kpms_config = kpms_config
        self.bodyparts = bodyparts

        # Free accumulated tracks
        self.coordinates = {}
        self.confidences = {}

    def handle_load_model(self, request: LoadModelRequest) -> None:
        log.info("Loading model from %s...", request.path)
        bundle = joblib.load(request.path)
        self.model = bundle["model"]
        self.pca = bundle["pca"]
        self.metadata = bundle["metadata"]
        self.kpms_config = bundle["kpms_config"]
        self.bodyparts = bundle["bodyparts"]

    def handle_apply(self, request: ApplyRequest) -> ApplyResponse:
        assert self.kpms_config is not None, (
            "no model loaded (call fit or load_model first)"
        )
        kpms_config = self.kpms_config

        coords = decode_array(request.coords)
        conf = decode_array(request.conf)

        coords_dict = {request.key: coords}
        conf_dict = {request.key: conf}

        data, metadata = kpms.format_data(coords_dict, conf_dict, **kpms_config)

        # Convert to float64
        for k in list(data.keys()):
            if hasattr(data[k], "dtype") and data[k].dtype != jnp.float64:
                data[k] = (
                    jnp.array(data[k], dtype=jnp.float64)
                    if "mask" not in k
                    else jnp.array(data[k])
                )

        apply_dir = tempfile.mkdtemp(prefix="kpms_apply_")
        model_name = "applied"
        os.makedirs(os.path.join(apply_dir, model_name), exist_ok=True)

        results, _ = kpms.apply_model(
            self.model,
            data,
            metadata,
            project_dir=apply_dir,
            model_name=model_name,
            num_iters=request.num_iters,
            return_model=True,
            **kpms_config,
        )

        rec_data = results[request.key]
        syllables = rec_data.get("syllable", rec_data.get("syllables"))
        syllables = np.asarray(syllables, dtype=np.int32).ravel()

        return ApplyResponse(syllables=encode_array(syllables))

    def handle_save_model(self, request: SaveModelRequest) -> None:
        log.info("Saving model to %s...", request.path)
        _ = joblib.dump(
            {
                "model": self.model,
                "pca": self.pca,
                "metadata": self.metadata,
                "kpms_config": self.kpms_config,
                "bodyparts": self.bodyparts,
            },
            request.path,
        )


# --- Checkpoint helper ---


def _try_load_checkpoint(
    project_dir: str, model_name: str
) -> tuple[object, int] | None:
    """Load a kpms checkpoint if it exists. Returns (model, iteration) or None."""
    checkpoint_path = os.path.join(project_dir, model_name, "checkpoint.h5")
    if not os.path.exists(checkpoint_path):
        return None
    model, _, _, iteration = kpms.load_checkpoint(
        project_dir=project_dir,
        model_name=model_name,
    )
    return model, iteration


# --- Dispatch loop ---


def serve(server: KpmsServer, conn: socket.socket) -> None:
    """Read commands from conn and dispatch to server handlers."""
    while True:
        try:
            request = recv_request(conn)
        except ConnectionError:
            log.info("Client disconnected.")
            break

        log.info("command: %s", request.command)

        if isinstance(request, ShutdownRequest):
            send_message(conn, OkResponse())
            break

        try:
            response: BaseModel | None = None
            if isinstance(request, AddTrackRequest):
                server.handle_add_track(request)
            elif isinstance(request, FitRequest):
                server.handle_fit(request)
            elif isinstance(request, LoadModelRequest):
                server.handle_load_model(request)
            elif isinstance(request, ApplyRequest):
                response = server.handle_apply(request)
            else:
                server.handle_save_model(request)
            send_message(conn, response or OkResponse())
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            log.error("ERROR in %s: %s\n%s", request.command, exc, tb)
            try:
                send_message(conn, ErrorResponse(message=str(exc), traceback=tb))
            except Exception:
                log.error("failed to send error response")
                break


# --- Entry point ---


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s"
    )
    prctl_set_pdeathsig()

    if len(sys.argv) != 2:
        log.error("Usage: %s <socket_path>", sys.argv[0])
        sys.exit(1)

    socket_path = sys.argv[1]

    server = KpmsServer()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(socket_path)
        sock.listen(1)
        conn, _ = sock.accept()
        try:
            send_message(conn, ReadyResponse())
            serve(server, conn)
        finally:
            conn.close()
    finally:
        sock.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


if __name__ == "__main__":
    main()
