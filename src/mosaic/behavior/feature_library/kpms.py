"""Unified keypoint-MoSeq feature.

Fits an AR-HMM model and applies it to extract per-frame syllable labels,
using a persistent subprocess server to avoid repeated JAX startup costs.
The kpms package does NOT need to be installed in the mosaic environment --
only in a separate .venv whose interpreter path is passed via ``kpms_python``.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, TYPE_CHECKING, Self, TypedDict, final

import numpy as np
from pydantic import Field, model_validator

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Params,
    PoseConfig,
    TrackInput,
)

from .registry import register_feature

if TYPE_CHECKING:
    import pandas as pd

from .external.kpms_protocol import (
    AddTrackRequest,
    ApplyRequest,
    ApplyResponse,
    FitConfig,
    FitRequest,
    LoadModelRequest,
    ReadyResponse,
    SaveModelRequest,
    ShutdownRequest,
    check_latent_dim,
    decode_array,
    encode_array,
    receive_message,
    send_message,
)

log = logging.getLogger(__name__)

_KPMS_SERVER_SCRIPT = Path(__file__).parent / "external" / "kpms_server.py"
_EXTERNAL_VENV_PYTHON = Path(__file__).parent / "external" / ".venv" / "bin" / "python"


# --- Data conversion ---


def _tracks_df_to_kpms_arrays(
    df: pd.DataFrame,
    pose_prefix_x: str = "poseX",
    pose_prefix_y: str = "poseY",
    pose_confidence_prefix: str = "poseP",
) -> tuple[
    np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    np.ndarray[tuple[int, int], np.dtype[np.float32]],
]:
    """Convert a single-sequence track DataFrame to keypoint-moseq arrays.

    Returns
    -------
    coords : ndarray, shape (T, K, 2)
    confidences : ndarray, shape (T, K)
    """
    x_cols = sorted(
        [c for c in df.columns if c.startswith(pose_prefix_x)],
        key=lambda c: int(c[len(pose_prefix_x) :]),
    )
    y_cols = [f"{pose_prefix_y}{c[len(pose_prefix_x) :]}" for c in x_cols]
    num_keypoints = len(x_cols)
    num_frames = len(df)

    if num_keypoints == 0:
        msg = (
            f"No keypoint columns found with prefix '{pose_prefix_x}'. "
            f"Available columns: {list(df.columns)}"
        )
        raise ValueError(msg)

    coords: np.ndarray[tuple[int, int, int], np.dtype[np.float32]] = np.empty(
        (num_frames, num_keypoints, 2), dtype=np.float32
    )
    x_vals: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.asarray(
        df[x_cols], dtype=np.float32
    )
    y_vals: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.asarray(
        df[y_cols], dtype=np.float32
    )
    coords[:, :, 0] = x_vals
    coords[:, :, 1] = y_vals

    conf_cols = [f"{pose_confidence_prefix}{c[len(pose_prefix_x) :]}" for c in x_cols]
    confidences: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    if all(c in df.columns for c in conf_cols):
        confidences = np.asarray(df[conf_cols], dtype=np.float32)
    else:
        confidences = np.ones((num_frames, num_keypoints), dtype=np.float32)

    # Replace Inf/NaN coords with NaN and zero out their confidences.
    bad_mask = ~np.isfinite(coords)
    bad_any = bad_mask.any(axis=2)
    n_bad = int(bad_any.sum())
    if n_bad > 0:
        coords[bad_mask] = np.nan
        confidences[bad_any] = 0.0
        pct = 100 * n_bad / (num_frames * num_keypoints)
        print(
            f"[kpms] Replaced {n_bad} non-finite keypoint observations ({pct:.1f}%) with NaN",
            file=sys.stderr,
        )

    return coords, confidences


# --- Model artifact ---


class KpmsModelBundle(TypedDict):
    model: object
    pca: object
    metadata: object
    kpms_config: dict[str, object]
    bodyparts: list[str]


class KpmsModelArtifact(JoblibArtifact[KpmsModelBundle]):
    feature: str = "kpms"
    pattern: str = "kpms_model.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


# --- Feature class ---


@final
@register_feature
class KpmsFeature:
    """Unified keypoint-MoSeq feature: fit + apply via persistent subprocess.

    Params:
        model: Pre-fitted KpmsModelArtifact to load (skip fit). Default:
            None (fit from scratch).
        kpms_python: Path to a Python interpreter with keypoint-moseq
            installed. None uses the bundled external .venv. Default: None.
        pose: Pose keypoint configuration (indices, column prefixes).
            Default: PoseConfig().
        anterior_bodyparts: List of bodypart names forming the anterior
            reference (required, min 1 element).
        posterior_bodyparts: List of bodypart names forming the posterior
            reference (required, min 1 element).
        fps: Frames per second of the input data. Default: 30.
        num_iters_ar: Number of AR-only fitting iterations. Default: 50.
        num_iters_full: Number of full model fitting iterations.
            Default: 500.
        kappa_ar: AR transition concentration parameter. None lets
            keypoint-moseq choose. Default: None.
        kappa_full: Full-model transition concentration parameter. None
            lets keypoint-moseq choose. Default: None.
        latent_dim: Dimensionality of the latent pose space. Must satisfy
            latent_dim < 2 * num_keypoints. Default: 10.
        location_aware: If True, include centroid location in the model.
            Default: False.
        outlier_scale_factor: Scale factor for outlier detection.
            Default: 6.0.
        remove_outliers: If True, remove detected outlier frames before
            fitting. Default: True.
        mixed_map_iters: Number of mixed MAP iterations. None uses the
            keypoint-moseq default. Default: None.
        parallel_message_passing: Enable parallel message passing. None
            uses the keypoint-moseq default. Default: None.
        resume: If True, resume fitting from a previously saved
            checkpoint. Default: True.
        downsample_rate: Temporal downsampling factor applied before
            fitting. None disables downsampling. Default: None.
        save_every_n_iters: Save a checkpoint every N iterations during
            fit. Default: 25.
        num_iters_apply: Number of iterations when applying the model to
            new data. Default: 500.
    """

    name = "kpms"
    version = "0.1"
    parallelizable = False
    scope_dependent = True

    KpmsModelBundle = KpmsModelBundle
    KpmsModelArtifact = KpmsModelArtifact

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        model: KpmsModelArtifact | None = None
        kpms_python: str | None = None
        pose: PoseConfig = Field(default_factory=PoseConfig)
        anterior_bodyparts: list[str] = Field(min_length=1)
        posterior_bodyparts: list[str] = Field(min_length=1)
        fps: int = 30
        num_iters_ar: int = Field(default=50, ge=1)
        num_iters_full: int = Field(default=500, ge=1)
        kappa_ar: float | None = None
        kappa_full: float | None = None
        latent_dim: int = Field(default=10, ge=1)
        location_aware: bool = False
        outlier_scale_factor: float = Field(default=6.0, gt=0)
        remove_outliers: bool = True
        mixed_map_iters: int | None = None
        parallel_message_passing: bool | None = None
        resume: bool = True
        downsample_rate: int | None = Field(default=None, ge=1)
        save_every_n_iters: int = Field(default=25, ge=1)
        num_iters_apply: int = Field(default=500, ge=1)

        @model_validator(mode="after")
        def _check_latent_dim(self) -> Self:
            if self.pose.pose_indices is not None:
                num_keypoints = len(self.pose.pose_indices)
            else:
                num_keypoints = self.pose.pose_n
            check_latent_dim(self.latent_dim, num_keypoints)
            return self

    def __init__(
        self,
        inputs: KpmsFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._proc: subprocess.Popen[bytes] | None = None
        self._conn: socket.socket | None = None
        self._stderr_file: IO[str] | None = None
        self._bodypart_names: list[str] | None = None
        self._server_log: str = ""
        self._socket_path: str | None = None
        self._tmpdir: str | None = None

    # --- Server lifecycle ---

    def _start_server(self) -> None:
        if self._conn is not None:
            return

        kpms_python = self.params.kpms_python
        if kpms_python is not None:
            resolved = Path(kpms_python).expanduser()
        else:
            resolved = _EXTERNAL_VENV_PYTHON
        if not resolved.exists():
            msg = f"[kpms] Python interpreter not found: {resolved}"
            raise FileNotFoundError(msg)

        self._tmpdir = tempfile.mkdtemp(prefix="kpms_")
        socket_path = str(Path(self._tmpdir) / "kpms.sock")
        self._socket_path = socket_path

        self._stderr_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="kpms_log_", suffix=".log", delete=False
        )
        self._proc = subprocess.Popen(
            [str(resolved), str(_KPMS_SERVER_SCRIPT), socket_path],
            stderr=self._stderr_file,
        )
        self._server_log = ""

        # Poll until socket file appears
        import time

        deadline = time.monotonic() + 120
        while not Path(socket_path).exists():
            if self._proc.poll() is not None:
                self._read_server_log()
                msg = (
                    f"[kpms] Server exited with code {self._proc.returncode} "
                    f"during startup.\n{self._server_log}"
                )
                raise RuntimeError(msg)
            if time.monotonic() > deadline:
                self._proc.kill()
                msg = "[kpms] Server did not create socket within 120s"
                raise RuntimeError(msg)
            time.sleep(0.1)

        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(socket_path)
        self._conn = conn

        line = receive_message(conn)
        ReadyResponse.model_validate_json(line)
        log.info("[kpms] Server ready.")

    def _send(
        self,
        request: AddTrackRequest
        | FitRequest
        | LoadModelRequest
        | ApplyRequest
        | SaveModelRequest
        | ShutdownRequest,
    ) -> None:
        if self._conn is None:
            msg = "[kpms] Server not started"
            raise RuntimeError(msg)
        send_message(self._conn, request)

    def _receive_json(self) -> dict[str, object]:
        if self._conn is None:
            msg = "[kpms] Server not started"
            raise RuntimeError(msg)
        import json

        line = receive_message(self._conn)
        data: dict[str, object] = json.loads(line)
        if data.get("status") == "error":
            tb = data.get("traceback", "")
            detail = data.get("message", "(no message)")
            msg = f"[kpms] Server error: {detail}"
            if tb:
                msg += f"\n\nServer traceback:\n{tb}"
            raise RuntimeError(msg)
        return data

    def _receive_ok(self) -> None:
        self._receive_json()

    def _receive_apply(self) -> ApplyResponse:
        data = self._receive_json()
        return ApplyResponse.model_validate(data)

    def _read_server_log(self) -> None:
        if self._stderr_file is not None:
            self._stderr_file.flush()
            log_path = Path(self._stderr_file.name)
            if log_path.exists():
                self._server_log = log_path.read_text()

    def _shutdown_server(self) -> None:
        if self._conn is None:
            return
        try:
            self._send(ShutdownRequest(command="shutdown"))
            self._receive_ok()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None

        if self._proc is not None:
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

        self._read_server_log()

        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            try:
                Path(self._stderr_file.name).unlink(missing_ok=True)
            except Exception:
                pass
            self._stderr_file = None

        if self._socket_path is not None:
            Path(self._socket_path).unlink(missing_ok=True)
            self._socket_path = None
        if self._tmpdir is not None:
            import shutil

            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    # --- Track sending ---

    def _send_track(self, key: str, df: pd.DataFrame) -> None:
        pose = self.params.pose
        coords, conf = _tracks_df_to_kpms_arrays(
            df, pose.x_prefix, pose.y_prefix, pose.confidence_prefix
        )
        if self.params.downsample_rate is not None and self.params.downsample_rate > 1:
            rate = self.params.downsample_rate
            coords = coords[::rate]
            conf = conf[::rate]
        self._send(
            AddTrackRequest(
                command="add_track",
                key=key,
                coords=encode_array(coords),  # pyright: ignore[reportArgumentType] # cross-env numpy
                conf=encode_array(conf),  # pyright: ignore[reportArgumentType] # cross-env numpy
            )
        )
        self._receive_ok()

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        # Branch 1: cached model in run_root
        cached_path = run_root / "kpms_model.joblib"
        if cached_path.exists():
            self._start_server()
            self._send(LoadModelRequest(command="load_model", path=str(cached_path)))
            self._receive_ok()
            return True

        # Branch 2: pre-fitted model from artifact_paths
        if self.params.model is not None and "model" in artifact_paths:
            self._start_server()
            self._send(
                LoadModelRequest(
                    command="load_model", path=str(artifact_paths["model"])
                )
            )
            self._receive_ok()
            return True

        return False

    def fit(self, inputs: InputStream) -> None:
        self._start_server()

        pose = self.params.pose
        bodypart_names: list[str] | None = pose.keypoint_names

        for entry_key, df in inputs():
            if "id" in df.columns:
                id_col: pd.Series[int] = df["id"]
                for ind_id, sub in df.groupby(id_col, sort=False):  # pyright: ignore[reportUnknownMemberType]
                    sub = sub.sort_values("frame").reset_index(drop=True)
                    key = f"{entry_key}__id{ind_id}"
                    self._send_track(key, sub)
            else:
                self._send_track(entry_key, df)

            # Auto-detect bodypart names from first track if not configured
            if bodypart_names is None:
                x_cols = [c for c in df.columns if c.startswith(pose.x_prefix)]
                bodypart_names = [f"kp{i}" for i in range(len(x_cols))]

        if bodypart_names is None:
            msg = "[kpms] No tracks found -- nothing to fit."
            raise RuntimeError(msg)

        self._bodypart_names = bodypart_names

        # Derive use_bodyparts from pose_indices
        use_bodyparts: list[str] | None = None
        if pose.pose_indices is not None:
            use_bodyparts = [bodypart_names[i] for i in pose.pose_indices]

        # Adjust FPS for downsampling
        params = self.params
        effective_fps = params.fps
        if params.downsample_rate is not None and params.downsample_rate > 1:
            effective_fps = params.fps // params.downsample_rate

        config = FitConfig(
            bodyparts=bodypart_names,
            use_bodyparts=use_bodyparts,
            anterior_bodyparts=params.anterior_bodyparts,
            posterior_bodyparts=params.posterior_bodyparts,
            fps=effective_fps,
            latent_dim=params.latent_dim,
            remove_outliers=params.remove_outliers,
            outlier_scale_factor=params.outlier_scale_factor,
            mixed_map_iters=params.mixed_map_iters,
            parallel_message_passing=params.parallel_message_passing,
            save_every_n_iters=params.save_every_n_iters,
            resume=params.resume,
            kappa_ar=params.kappa_ar,
            kappa_full=params.kappa_full,
            num_iters_ar=params.num_iters_ar,
            num_iters_full=params.num_iters_full,
            location_aware=params.location_aware,
        )
        self._send(FitRequest(command="fit", config=config))
        self._receive_ok()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        group = str(df["group"].iloc[0]) if "group" in df.columns else ""
        sequence = str(df["sequence"].iloc[0]) if "sequence" in df.columns else ""
        key = make_entry_key(group, sequence)

        results: list[pd.DataFrame] = []

        if "id" in df.columns:
            id_col: pd.Series[int] = df["id"]
            for ind_id, sub in df.groupby(id_col, sort=False):  # pyright: ignore[reportUnknownMemberType]
                sub = sub.sort_values("frame").reset_index(drop=True)
                rec_key = f"{key}__id{ind_id}"
                syllables = self._apply_one(rec_key, sub)
                result = pd.DataFrame(
                    {
                        "frame": sub["frame"].values,
                        "syllable": syllables,
                    }
                )
                result["id"] = ind_id
                results.append(result)
        else:
            syllables = self._apply_one(key, df)
            results.append(
                pd.DataFrame(
                    {
                        "frame": df["frame"].values,
                        "syllable": syllables,
                    }
                )
            )

        return pd.concat(results, ignore_index=True)

    def _apply_one(
        self, key: str, df: pd.DataFrame
    ) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        pose = self.params.pose
        coords, conf = _tracks_df_to_kpms_arrays(
            df, pose.x_prefix, pose.y_prefix, pose.confidence_prefix
        )
        request = ApplyRequest(
            command="apply",
            key=key,
            num_iters=self.params.num_iters_apply,
            coords=encode_array(coords),  # pyright: ignore[reportArgumentType] # cross-env numpy
            conf=encode_array(conf),  # pyright: ignore[reportArgumentType] # cross-env numpy
        )
        self._send(request)
        response = self._receive_apply()
        return decode_array(response.syllables)  # pyright: ignore[reportReturnType] # cross-env numpy

    def save_state(self, run_root: Path) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        model_path = run_root / "kpms_model.joblib"
        self._send(SaveModelRequest(command="save_model", path=str(model_path)))
        self._receive_ok()
        self._shutdown_server()

        if self._server_log:
            (run_root / "kpms_server.log").write_text(self._server_log)

    def __del__(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                self._shutdown_server()
            except Exception:
                if proc.poll() is None:
                    proc.kill()
