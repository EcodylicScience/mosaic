"""The keypoint-MoSeq runner, which fits and applies models in the tool's environment.

mosaic's ``kpms`` feature starts this program in the environment built under this
directory. The program serves the commands that ``kpms_protocol`` declares through
``runner_wire``, reading requests on standard input and writing replies on
standard output. Arrays cross as ``.npz`` files that a payload names by path.

The module scope imports the standard library, pydantic and the two siblings
only. JAX, keypoint-moseq and joblib are imported by :func:`load_libraries` at the
first command, which then patches ``np.bincount`` and enables 64-bit JAX. The
program therefore imports, and answers ``Ready``, in an environment without them.

Usage::

    .venv/bin/python kpms_server.py

This is the only file in mosaic that imports keypoint-moseq, and it runs in the
external environment rather than mosaic's own. keypoint-MoSeq is licensed by the
Harvard University Office of Technology Development for non-commercial research
and academic use only, and commercial use is expressly prohibited -- including
fee-for-service work, core facilities serving for-profit third parties, and
industry-sponsored projects granting commercial rights to the sponsor. Read the
terms at https://github.com/dattalab/keypoint-moseq/blob/main/LICENSE.md before
running this, by any route. mosaic's own entry point refuses to spawn this
server until ``MOSAIC_KPMS_LICENSE_ACCEPTED=1`` says the terms are met; running
the script directly, as above, bypasses nothing but that reminder.
"""

# JAX, jax_moseq and keypoint-moseq are installed in the keypoint-MoSeq
# environment only. The rules disabled below report the consequences:
# - reportMissingImports: their imports do not resolve in mosaic's environment;
# - reportUnknownMemberType, reportUnknownVariableType and
#   reportUnknownArgumentType: every value read from them has an unknown type,
#   and the command handlers and their helpers compute with those values.
# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import functools
import importlib
import logging
import os
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, SupportsIndex, TypedDict, TypeVar

from kpms_protocol import (
    AddTrack,
    Apply,
    Fit,
    FitConfig,
    LoadModel,
    Payload,
    SaveModel,
)
from runner_wire import (
    Dispatch,
    Emit,
    claim_stdout,
    installed_version,
    log_to_stderr,
    serve,
    serving,
)

if TYPE_CHECKING:
    import numpy as np

TOOL: Final = "keypoint-moseq"
"""The tool name that ``Ready`` reports. It is also the tool's distribution name."""

log = logging.getLogger("kpms_server")

_LIBRARIES: Final = (
    "jax",
    "jax.numpy",
    "joblib",
    "keypoint_moseq",
    "numpy",
    "jax_moseq.utils",
)
"""The modules that :func:`load_libraries` imports, in order."""

_P = TypeVar("_P", bound=Payload)


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


# --- Libraries ---


@functools.cache
def load_libraries() -> None:
    """Import JAX, keypoint-moseq and joblib, and configure them, once.

    numpy 2.4 refuses ``minlength=None`` in ``np.bincount``, which keypoint-moseq
    passes. The patch below reads it as 0. JAX then computes in 64 bits.
    """
    for name in _LIBRARIES:
        _ = importlib.import_module(name)
    import jax
    import numpy as np
    import numpy.typing as npt

    original_bincount = np.bincount

    def bincount(
        x: npt.NDArray[np.integer],
        weights: npt.NDArray[np.floating] | None = None,
        minlength: SupportsIndex | None = None,
    ) -> npt.NDArray[np.intp] | npt.NDArray[np.float64]:
        length = 0 if minlength is None else minlength
        if weights is None:
            return original_bincount(x, minlength=length)
        return original_bincount(x, weights=weights, minlength=length)

    np.bincount = bincount
    jax.config.update("jax_enable_x64", True)


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
        # Message-passing mode chosen at fit time; reused for apply so the
        # (memory-heavy) parallel scan can't reappear on the apply pass.
        self.parallel_mp: bool | None = None

    def handle_add_track(self, request: AddTrack) -> None:
        import numpy as np

        with np.load(request.track) as track:
            self.coordinates[request.key] = track["coords"]
            self.confidences[request.key] = track["conf"]

    def handle_fit(self, request: Fit) -> None:
        config = request.config
        if config.checkpoint_dir is not None:
            self._fit(config, config.checkpoint_dir)
            return
        with tempfile.TemporaryDirectory(prefix="kpms_") as checkpoint_dir:
            self._fit(config, checkpoint_dir)

    def _fit(self, config: FitConfig, checkpoint_dir: str) -> None:
        import jax.numpy as jnp
        import keypoint_moseq as kpms
        import numpy as np
        from jax_moseq.utils import set_mixed_map_iters

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
        _as_float64(data)

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
        self.parallel_mp = parallel_mp
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

    def handle_load_model(self, request: LoadModel) -> None:
        import joblib

        log.info("Loading model from %s...", request.path)
        bundle = joblib.load(request.path)
        self.model = bundle["model"]
        self.pca = bundle["pca"]
        self.metadata = bundle["metadata"]
        self.kpms_config = bundle["kpms_config"]
        self.bodyparts = bundle["bodyparts"]
        self.parallel_mp = bundle.get("parallel_mp")

    def handle_apply(self, request: Apply) -> None:
        import keypoint_moseq as kpms
        import numpy as np

        assert self.kpms_config is not None, (
            "no model loaded (call fit or load_model first)"
        )
        kpms_config = self.kpms_config

        with np.load(request.track) as track:
            coords_dict = {request.key: track["coords"]}
            conf_dict = {request.key: track["conf"]}

        data, metadata = kpms.format_data(coords_dict, conf_dict, **kpms_config)
        _as_float64(data)

        model_name = "applied"
        with tempfile.TemporaryDirectory(prefix="kpms_apply_") as apply_dir:
            os.makedirs(os.path.join(apply_dir, model_name), exist_ok=True)

            results, _ = kpms.apply_model(
                self.model,
                data,
                metadata,
                project_dir=apply_dir,
                model_name=model_name,
                num_iters=request.num_iters,
                return_model=True,
                parallel_message_passing=self.parallel_mp,
                **kpms_config,
            )

        rec_data = results[request.key]
        syllables = rec_data.get("syllable", rec_data.get("syllables"))
        syllables = np.asarray(syllables, dtype=np.int32).ravel()

        np.savez(request.syllables, syllables=syllables)

    def handle_save_model(self, request: SaveModel) -> None:
        import joblib

        log.info("Saving model to %s...", request.path)
        _ = joblib.dump(
            {
                "model": self.model,
                "pca": self.pca,
                "metadata": self.metadata,
                "kpms_config": self.kpms_config,
                "bodyparts": self.bodyparts,
                "parallel_mp": self.parallel_mp,
            },
            request.path,
        )


def _as_float64(data: dict[str, object]) -> None:
    """Convert each array of *data* to float64 in place, except a mask's dtype."""
    import jax.numpy as jnp

    for k in list(data.keys()):
        value = data[k]
        dtype = getattr(value, "dtype", None)
        if dtype is not None and dtype != jnp.float64:
            data[k] = (
                jnp.array(value, dtype=jnp.float64)
                if "mask" not in k
                else jnp.array(value)
            )


# --- Checkpoint helper ---


def _try_load_checkpoint(
    project_dir: str, model_name: str
) -> tuple[object, int] | None:
    """Load a kpms checkpoint if it exists. Returns (model, iteration) or None."""
    import keypoint_moseq as kpms

    checkpoint_path = os.path.join(project_dir, model_name, "checkpoint.h5")
    if not os.path.exists(checkpoint_path):
        return None
    model, _, _, iteration = kpms.load_checkpoint(
        project_dir=project_dir,
        model_name=model_name,
    )
    return model, iteration


# --- Dispatch ---


def _loading(handle: Callable[[_P], None]) -> Callable[[_P, Emit], None]:
    """Return *handle* as a handler that calls :func:`load_libraries` first."""

    def loaded(request: _P, emit: Emit) -> None:
        load_libraries()
        handle(request)

    return loaded


def dispatch(server: KpmsServer) -> Dispatch:
    """The handler of each command, run on *server*."""
    return {
        AddTrack.command: serving(AddTrack, _loading(server.handle_add_track)),
        Fit.command: serving(Fit, _loading(server.handle_fit)),
        LoadModel.command: serving(LoadModel, _loading(server.handle_load_model)),
        Apply.command: serving(Apply, _loading(server.handle_apply)),
        SaveModel.command: serving(SaveModel, _loading(server.handle_save_model)),
    }


# --- Entry point ---


def main() -> None:
    protocol = claim_stdout()
    log_to_stderr()
    serve(dispatch(KpmsServer()), TOOL, installed_version(TOOL), protocol=protocol)


if __name__ == "__main__":
    main()
