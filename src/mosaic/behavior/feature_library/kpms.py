"""Unified keypoint-MoSeq feature.

Fits an AR-HMM model and applies it to extract per-frame syllable labels. A
runner program in a separate environment, located through :data:`KPMS_ENV`,
fits and applies the model. One runner serves a feature from its first request
until :meth:`KpmsFeature.close`, which avoids repeated JAX startup costs. The
mosaic environment does not need the kpms package.

The method is keypoint-MoSeq: Weinreb et al. (2024), Nature Methods
21(7):1329-1339, doi:10.1038/s41592-024-02318-2, which builds on the AR-HMM
syllables of MoSeq: Wiltschko et al. (2015), Neuron 88(6):1121-1135,
doi:10.1016/j.neuron.2015.11.031. See NOTICE.

keypoint-MoSeq is licensed for non-commercial research and academic use only,
and mosaic does not bundle it. :data:`LICENSE_NOTICE` and ``external/README.md``
state the terms. This module does not import it. The licensed code runs only in
the runner's interpreter, which a feature starts only after
:func:`check_license_accepted`.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    Self,
    TypedDict,
    final,
)

import numpy as np
from pydantic import Field, model_validator

from mosaic.core.helpers import make_entry_key
from mosaic.core.individuals import iter_individuals
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.column_contract import (
    Column,
    Provides,
    Requirement,
    Template,
    kept,
)
from mosaic.core.pipeline.job import CancelCheck, CancelToken, cancel_check
from mosaic.optional_dependency import Prerequisite
from mosaic.core.pipeline.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
)
from mosaic.core.pose_columns import KeypointArrays, keypoint_arrays, keypoint_suffixes
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    POSE_CONFIG_DESCRIPTION,
    PoseConfig,
    TrackInputs,
)
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)

from .external.kpms_protocol import (
    AddTrack,
    Apply,
    Fit,
    FitConfig,
    LoadModel,
    Payload,
    SaveModel,
    check_latent_dim,
)
from .helpers import entry_identity, label_individuals
from .registry import register_feature
from .runner_client import RunnerInterpreter, RunnerProcess

if TYPE_CHECKING:
    import pandas as pd
    from pandas._typing import Scalar

log = logging.getLogger(__name__)

KPMS_SERVER_SCRIPT: Final = Path(__file__).parent / "external" / "kpms_server.py"
"""The runner program, which runs in the keypoint-MoSeq environment."""

KPMS_MODEL_FILE: Final = "kpms_model.joblib"
"""The model that :meth:`KpmsFeature.save_state` writes to its run root."""

KPMS_SERVER_LOG: Final = "kpms_server.log"
"""The runners' standard error in the run root.

:meth:`KpmsFeature.save_state` appends the log of the fit, and
:meth:`KpmsFeature.close` the rest.
"""

_Unsaved = Literal["tracks", "fit"]
"""The state of a runner that is not recorded in a file."""

_UNSAVED_NAMES: Final[dict[_Unsaved, str]] = {
    "tracks": "the tracks sent for the fit",
    "fit": "the fitted model",
}
"""Each state's name in the refusal of a runner started in place of an exited one."""

KPMS_CONDA_ENV_VAR = "MOSAIC_KPMS_CONDA_ENV"
KPMS_PYTHON_ENV = "MOSAIC_KPMS_PYTHON"
KPMS_LICENSE_ENV = "MOSAIC_KPMS_LICENSE_ACCEPTED"

_ACCEPT_LICENSE = (
    "Read the terms at "
    "https://github.com/dattalab/keypoint-moseq/blob/main/LICENSE.md, then set "
    f"{KPMS_LICENSE_ENV}=1 to confirm your use is permitted."
)

LICENSE_NOTICE = (
    "[kpms] keypoint-MoSeq is licensed by the Harvard University Office of "
    "Technology Development for non-commercial research and academic use only, "
    "and commercial use is expressly prohibited.  Harvard's definition of "
    "commercial use reaches fee-for-service arrangements, core facilities "
    "providing research services to (or with) for-profit third parties for a "
    "fee, and industry-sponsored or collaborative research projects granting "
    "commercial rights to the sponsor or collaborator.\n" + _ACCEPT_LICENSE
)

_USE_RESTRICTION = (
    "keypoint-MoSeq is licensed for non-commercial research and academic use "
    "only. Commercial use is prohibited, including fee-for-service analysis and "
    "research services to for-profit third parties."
)
"""The limit that the catalog lists beside ``kpms``, accepted or not."""

_LICENSE_UNACCEPTED = (
    "keypoint-MoSeq is licensed for non-commercial research and academic use "
    "only, and this installation has not accepted its license. " + _ACCEPT_LICENSE
)

_NOT_FOUND_HELP = (
    "keypoint-moseq is not installed with mosaic and is never bundled in a "
    "mosaic distribution. It runs in a separate environment. Build it with "
    "the command in mosaic/behavior/feature_library/external/README.md, "
    f"point {KPMS_PYTHON_ENV} at an interpreter that already has it, or name a "
    f"conda environment with {KPMS_CONDA_ENV_VAR}."
)


class KpmsNotFoundError(ToolNotFoundError):
    """Raised when no Python interpreter with keypoint-moseq can be located."""

    default_message = _NOT_FOUND_HELP


class KpmsExitError(ToolExitError):
    """Raised when the keypoint-MoSeq runner exits during a command.

    Also raised when the runner writes a line that breaks the runner protocol.
    """

    tool_name = "The keypoint-MoSeq runner"


# keypoint-moseq does not install a console script. Rung 5 reads the interpreter
# under external/.venv.
KPMS_ENV: Final = ToolEnv(
    tool="keypoint-moseq",
    conda_env_var=KPMS_CONDA_ENV_VAR,
    bin_var=KPMS_PYTHON_ENV,
    bin_mode="direct",
    not_found=KpmsNotFoundError,
    repo_venv=Path(__file__).parent / "external",
)


def check_license_accepted() -> None:
    """Raise unless the keypoint-MoSeq license has been explicitly accepted.

    Acceptance is ``MOSAIC_KPMS_LICENSE_ACCEPTED=1``. Surrounding whitespace is
    ignored: a value carried in from a ``.env`` file or a YAML ``env:`` block
    picks up a trailing space easily, and refusing on an invisible character is
    the worst failure mode for a check whose whole job is to be explicable.

    Exactly ``1`` is accepted and nothing else. Widening to ``true`` or ``yes``
    would widen the ways to accept *accidentally* -- a job matrix that sets
    every flag to ``true`` would assert an institution's entitlement on its
    behalf.

    Unlike ``_int_env`` in :mod:`mosaic.media_probe_config`, a malformed value
    gets no error of its own. Unset, blank, ``0`` and ``maybe`` are one answer,
    the fallback state is the safe one, and every non-accepting value already
    produces a single message naming the variable and the value it wants; a
    second failure mode with an identical remedy would only add noise.

    Raises:
        RuntimeError: with :data:`LICENSE_NOTICE` when acceptance is absent.
    """
    if license_accepted():
        return
    raise RuntimeError(LICENSE_NOTICE)


def license_accepted() -> bool:
    """Whether ``MOSAIC_KPMS_LICENSE_ACCEPTED`` is ``1``, whitespace aside."""
    return os.environ.get(KPMS_LICENSE_ENV, "").strip() == "1"


@final
@dataclass(frozen=True, slots=True)
class KpmsLicense:
    """A prerequisite of :class:`KpmsFeature`: the accepted keypoint-MoSeq license.

    Read from the environment, as :func:`check_license_accepted` reads it before
    the feature starts its runner. It is also a
    :class:`~mosaic.optional_dependency.RestrictedUse`, and the catalog lists
    its license note beside ``kpms``.
    """

    def missing(self) -> str | None:
        """How to accept the license, or ``None`` once it is accepted."""
        return None if license_accepted() else _LICENSE_UNACCEPTED

    def license_note(self) -> str:
        """The uses that keypoint-MoSeq's license forbids."""
        return _USE_RESTRICTION


# --- Data conversion ---


def recording_key(entry_key: str, individual: Scalar | None) -> str:
    """Name one individual of an entry as a keypoint-MoSeq recording."""
    if individual is None:
        return entry_key
    return f"{entry_key}__id{individual}"


def kpms_keypoint_arrays(rows: pd.DataFrame, pose: PoseConfig) -> KeypointArrays:
    """Read every keypoint in *rows*, as keypoint-MoSeq takes them.

    Every suffix with an x column under ``pose.x_prefix`` is read, in keypoint
    order, whatever ``pose_n`` says. keypoint-MoSeq narrows to ``pose_indices``
    itself, as positions among these keypoints, through ``use_bodyparts``.

    Args:
        rows: One individual's rows.
        pose: The column prefixes.

    Returns:
        The arrays, one row per row of *rows*.

    Raises:
        ValueError: When *rows* lacks an x column under ``pose.x_prefix``.
    """
    suffixes = keypoint_suffixes(rows.columns, pose.x_prefix)
    if not suffixes:
        msg = (
            f"No keypoint columns found with prefix '{pose.x_prefix}'. "
            f"Available columns: {list(rows.columns)}"
        )
        raise ValueError(msg)
    arrays = keypoint_arrays(
        rows,
        x_prefix=pose.x_prefix,
        y_prefix=pose.y_prefix,
        confidence_prefix=pose.confidence_prefix,
        suffixes=suffixes,
    )
    n_missing = int(np.isnan(arrays.position).any(axis=2).sum())
    if n_missing > 0:
        percent = 100 * n_missing / arrays.confidence.size
        print(
            f"[kpms] Replaced {n_missing} non-finite keypoint observations "
            f"({percent:.1f}%) with NaN",
            file=sys.stderr,
        )
    return arrays


_MODEL_DESCRIPTION = "A pre-fitted model to load. Unset, the feature fits a new model."

_KPMS_PYTHON_DESCRIPTION = (
    "Path to a Python interpreter with keypoint-moseq installed. Unset, "
    "resolution reads MOSAIC_KPMS_CONDA_ENV, then MOSAIC_KPMS_PYTHON, then "
    "the in-repository environment under feature_library/external/.venv."
)

_ANTERIOR_BODYPARTS_DESCRIPTION = (
    "Bodypart names marking the front of the body, used to initialize "
    "heading for egocentric alignment."
)

_POSTERIOR_BODYPARTS_DESCRIPTION = (
    "Bodypart names marking the back of the body, used to initialize "
    "heading for egocentric alignment."
)

_FPS_DESCRIPTION = "The frame rate of the input tracks."

_NUM_ITERS_AR_DESCRIPTION = (
    "How long the AR-only fitting stage runs before the full-model stage begins."
)

_NUM_ITERS_FULL_DESCRIPTION = (
    "How long the full-model fitting stage runs after the AR-only stage."
)

_KAPPA_AR_DESCRIPTION = (
    "The transition stickiness hyperparameter for the AR-only fitting "
    "stage. Unset, keypoint-moseq chooses its own value."
)

_KAPPA_FULL_DESCRIPTION = (
    "The transition stickiness hyperparameter for the full-model fitting "
    "stage. Unset, keypoint-moseq chooses its own value."
)

_LATENT_DIM_DESCRIPTION = (
    "The dimensionality of the latent pose space fit by PCA. A value "
    "exceeding (number of keypoints - 1) * 2 is refused."
)

_LOCATION_AWARE_DESCRIPTION = (
    "Constrain the centroid-movement prior, setting sigmasq_loc to 0.5 "
    "rather than 1e6. The fit then treats frame-to-frame centroid "
    "movement as informative."
)

_OUTLIER_SCALE_FACTOR_DESCRIPTION = (
    "The multiplier setting the outlier distance threshold when removing "
    "keypoint outliers before fitting."
)

_REMOVE_OUTLIERS_DESCRIPTION = (
    "Interpolate keypoints identified as outliers by their distance from "
    "the per-frame medoid, before fitting."
)

_MIXED_MAP_ITERS_DESCRIPTION = (
    "How many passes jax_moseq's mixed-map dispatch performs. A value of "
    "1 or below is ignored."
)

_PARALLEL_MESSAGE_PASSING_DESCRIPTION = (
    "Use keypoint-moseq's associative-scan Kalman sampling, faster to "
    "run and slower to compile. Unset, keypoint-moseq enables it when "
    "the JAX backend is not the CPU."
)

_RESUME_DESCRIPTION = (
    "Continue fitting from a previously saved checkpoint, falling back to "
    "a fresh fit when none exists."
)

_DOWNSAMPLE_RATE_DESCRIPTION = (
    "Keep every Nth frame of the tracks before fitting. Unset, no "
    "downsampling is applied. Not applied when the fitted model is later "
    "used to label new tracks."
)

_SAVE_EVERY_N_ITERS_DESCRIPTION = "How often keypoint-moseq writes a fit checkpoint."

_NUM_ITERS_APPLY_DESCRIPTION = (
    "How long resampling runs when applying a fitted model to new tracks."
)

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
    """Fits keypoint-MoSeq to keypoint tracks and applies it, in a runner process.

    !!! warning "Non-commercial license"

        keypoint-MoSeq is licensed by the Harvard University Office of
        Technology Development for **non-commercial research and academic use
        only**, and commercial use is expressly prohibited. Harvard's
        definition reaches fee-for-service arrangements, core facilities
        providing research services to (or with) for-profit third parties for a
        fee, and industry-sponsored or collaborative projects granting
        commercial rights to the sponsor. This is not a copyleft license, so no
        paid exception cures it.

        mosaic does not install, bundle, or import keypoint-moseq. It runs in a
        separate environment you build yourself, as a program that mosaic
        drives over its standard input and output, and will not start until you
        set ``MOSAIC_KPMS_LICENSE_ACCEPTED=1`` to confirm your use is
        permitted. ``external/README.md`` in this package is the setup, and
        ``NOTICE`` records the terms.

    The runner starts at the feature's first request and serves it until
    :meth:`close`. A fitted model therefore stays in the runner through
    :meth:`save_state` for :meth:`apply`. :meth:`load_state` does not start a
    runner. When it finds a model to apply, it raises unless the license is
    accepted and the interpreter is located. A runner that exits is started
    again by the next request, and loads the model of :meth:`load_state` or
    :meth:`save_state`. A runner that exits after a fit begins and before
    :meth:`save_state` loses tracks or a model that no file records. The next
    request then raises. A fired ``cancel_token`` kills the runner during its
    command, which then raises :class:`~mosaic.core.pipeline.job.Cancelled`.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.kpms.KpmsFeature.Params`.
    """

    category = "global"
    costly_fit = True  # a superseded-run cleanup keeps its runs
    # keypoint-MoSeq fits in JAX, and is built for a GPU; on CPU a fit takes days.
    resource_class = "gpu"
    name = "kpms"
    version = "0.1"
    parallelizable = False
    scope_dependent = True
    accepts_overlap = (
        False  # decodes a whole sequence at once, so context is not a window
    )
    consumed_roots: tuple[str, ...] = ()
    prerequisites: tuple[Prerequisite, ...] = (
        KpmsLicense(),
        RunnerInterpreter(KPMS_ENV),
    )
    emits: EmitsLevel = "individual"
    # Apply decodes each individual's keypoints in frame order. Its code names
    # the frame and id columns literally, and so does this declaration.
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column("frame"),
        Template("{pose.x_prefix}", family=True),
        Template("{pose.y_prefix}", family=True),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(Column("frame"), Column("syllable")), conditional=kept("id")
    )
    KpmsModelBundle = KpmsModelBundle
    KpmsModelArtifact = KpmsModelArtifact

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        model: Annotated[KpmsModelArtifact | None, Declared(_MODEL_DESCRIPTION)] = None
        # Where the interpreter lives, not what it computes: the same model on
        # two machines resolves through two different paths. Folding that into
        # the identity would mint a second run_id for byte-identical output and
        # recompute a full fit for nothing. HASH_EXCLUDE: dropped from the
        # run_id hash, still recorded in params.json.
        kpms_python: Annotated[
            str | None, HASH_EXCLUDE, Declared(_KPMS_PYTHON_DESCRIPTION)
        ] = None
        pose: Annotated[PoseConfig, Declared(POSE_CONFIG_DESCRIPTION)] = Field(
            default_factory=PoseConfig
        )
        anterior_bodyparts: Annotated[
            list[str], Declared(_ANTERIOR_BODYPARTS_DESCRIPTION)
        ] = Field(min_length=1)
        posterior_bodyparts: Annotated[
            list[str], Declared(_POSTERIOR_BODYPARTS_DESCRIPTION)
        ] = Field(min_length=1)
        fps: Annotated[int, Declared(_FPS_DESCRIPTION, unit="fps")] = 30
        num_iters_ar: Annotated[
            int, Declared(_NUM_ITERS_AR_DESCRIPTION, unit="iterations")
        ] = Field(default=50, ge=1)
        num_iters_full: Annotated[
            int, Declared(_NUM_ITERS_FULL_DESCRIPTION, unit="iterations")
        ] = Field(default=500, ge=1)
        kappa_ar: Annotated[float | None, Declared(_KAPPA_AR_DESCRIPTION)] = None
        kappa_full: Annotated[float | None, Declared(_KAPPA_FULL_DESCRIPTION)] = None
        latent_dim: Annotated[int, Declared(_LATENT_DIM_DESCRIPTION)] = Field(
            default=10, ge=1
        )
        location_aware: Annotated[bool, Declared(_LOCATION_AWARE_DESCRIPTION)] = False
        outlier_scale_factor: Annotated[
            float, Declared(_OUTLIER_SCALE_FACTOR_DESCRIPTION)
        ] = Field(default=6.0, gt=0)
        remove_outliers: Annotated[bool, Declared(_REMOVE_OUTLIERS_DESCRIPTION)] = True
        mixed_map_iters: Annotated[
            int | None, Declared(_MIXED_MAP_ITERS_DESCRIPTION, unit="iterations")
        ] = None
        parallel_message_passing: Annotated[
            bool | None, Declared(_PARALLEL_MESSAGE_PASSING_DESCRIPTION)
        ] = None
        resume: Annotated[bool, Declared(_RESUME_DESCRIPTION)] = True
        downsample_rate: Annotated[
            int | None, Declared(_DOWNSAMPLE_RATE_DESCRIPTION)
        ] = Field(default=None, ge=1)
        save_every_n_iters: Annotated[
            int, Declared(_SAVE_EVERY_N_ITERS_DESCRIPTION, unit="iterations")
        ] = Field(default=25, ge=1)
        num_iters_apply: Annotated[
            int, Declared(_NUM_ITERS_APPLY_DESCRIPTION, unit="iterations")
        ] = Field(default=500, ge=1)

        @model_validator(mode="after")
        def _check_latent_dim(self) -> Self:
            check_latent_dim(self.latent_dim, len(self.pose.selected_indices()))
            return self

    def __init__(
        self,
        inputs: KpmsFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.cancel_token: CancelToken | None = None
        self._runner = RunnerProcess(
            KPMS_ENV.placed(bin_path=self.params.kpms_python),
            KPMS_SERVER_SCRIPT,
            failure=KpmsExitError,
            version=None,
            on_start=self._prepare,
        )
        # The model file that a started runner loads.
        self._model: Path | None = None
        # The tracks or the fit in the runner that no file records. A runner
        # started in its place lacks them, and its start raises.
        self._unsaved: _Unsaved | None = None

    # --- Runner ---

    def _request(self, payload: Payload) -> None:
        """Run *payload*'s command in the runner, after checking the license.

        Every request checks the license before it can start a runner. The
        refusal therefore covers every caller (CLI, notebooks, mosaic-api), and
        it does not depend on whether an environment is present. ``__init__``
        does not check it, because constructing a KpmsFeature is not use of
        keypoint-MoSeq. ``mosaic reconcile`` constructs every feature that it
        re-addresses, and ``reconcile_features._build`` catches only
        ``ValidationError``, ``ValueError`` and ``TypeError``. Its sweep
        therefore passes a dataset that ran kpms without the license.
        """
        check_license_accepted()
        _ = self._runner.request(
            payload.command, payload, cancel=cancel_check(self.cancel_token)
        )

    def _prepare(self, runner: RunnerProcess, cancel: CancelCheck | None) -> None:
        """Load the model into a runner that has started.

        Raises:
            RuntimeError: The exited runner had tracks or a fit that no file
                records.
        """
        if self._unsaved is not None:
            msg = (
                "[kpms] The keypoint-MoSeq runner exited with "
                f"{_UNSAVED_NAMES[self._unsaved]}, which no file records. Fit again."
            )
            raise RuntimeError(msg)
        if self._model is not None:
            load = LoadModel(path=str(self._model))
            _ = runner.request(load.command, load, cancel=cancel)

    @contextmanager
    def _exchange_file(self, name: str) -> Generator[Path]:
        """Yield a path for *name* in the exchange directory, and remove it on exit."""
        path = self._runner.exchange_dir() / name
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        """Stop the runner, append its log to the run root, and remove its files.

        The log in the run root receives the runner's standard error that
        :meth:`save_state` did not append, and the exchange directory is removed.
        A second call does nothing. A later request starts a new runner.
        """
        _ = self._runner.close()

    # --- Track sending ---

    def _send_track(self, key: str, df: pd.DataFrame) -> None:
        arrays = kpms_keypoint_arrays(df, self.params.pose)
        coords, conf = arrays.position, arrays.confidence
        if self.params.downsample_rate is not None and self.params.downsample_rate > 1:
            rate = self.params.downsample_rate
            coords = coords[::rate]
            conf = conf[::rate]
        with self._exchange_file("track.npz") as track:
            np.savez(track, coords=coords, conf=conf)
            self._request(AddTrack(key=key, track=str(track)))
        self._unsaved = "tracks"

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        # A runner of an earlier run contains that run's state. It is stopped,
        # and the next request starts one that loads the model found here.
        self.close()
        self._runner.keep_log(run_root / KPMS_SERVER_LOG)
        self._model = None
        self._unsaved = None

        # The cached model of this run root, else a pre-fitted model.
        cached_path = run_root / KPMS_MODEL_FILE
        if cached_path.exists():
            model = cached_path
        elif self.params.model is not None and "model" in artifact_paths:
            model = artifact_paths["model"]
        else:
            return False
        # A run that applies a model is refused here, once, before any entry
        # is read. The two checks do not start a process. The license comes
        # first, as at a request.
        check_license_accepted()
        _ = self._runner.locate()
        self._model = model
        return True

    def fit(self, inputs: InputStream) -> None:
        pose = self.params.pose
        bodypart_names: list[str] | None = pose.keypoint_names
        self._model = None
        self._unsaved = None

        for entry_key, df in inputs():
            for individual, rows in iter_individuals(df):
                self._send_track(recording_key(entry_key, individual), rows)

            # Name the keypoints of the first track when the pose names none.
            if bodypart_names is None:
                bodypart_names = pose.names(
                    len(keypoint_suffixes(df.columns, pose.x_prefix))
                )

        if bodypart_names is None:
            msg = "[kpms] No tracks found -- nothing to fit."
            raise RuntimeError(msg)

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
        self._request(Fit(config=config))
        self._unsaved = "fit"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        key = make_entry_key(*entry_identity(df))
        return label_individuals(
            df,
            lambda individual, rows: self._apply_one(
                recording_key(key, individual), rows
            ),
        )

    def _apply_one(
        self, key: str, df: pd.DataFrame
    ) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        arrays = kpms_keypoint_arrays(df, self.params.pose)
        with (
            self._exchange_file("track.npz") as track,
            self._exchange_file("syllables.npz") as syllables,
        ):
            np.savez(track, coords=arrays.position, conf=arrays.confidence)
            self._request(
                Apply(
                    key=key,
                    num_iters=self.params.num_iters_apply,
                    track=str(track),
                    syllables=str(syllables),
                )
            )
            with np.load(syllables) as written:
                labels = written["syllables"]
        return np.asarray(labels, dtype=np.int32).reshape(-1)

    def save_state(self, run_root: Path) -> None:
        """Write the model and the runner's log to *run_root*, and keep the runner.

        The runner writes the model to a temporary file in *run_root*, which
        then replaces the model's path. :meth:`load_state` takes a file at that
        path as the run's model, and a runner killed during the write leaves
        none there.
        """
        model_path = run_root / KPMS_MODEL_FILE
        atomic_write(
            model_path, lambda target: self._request(SaveModel(path=str(target)))
        )
        self._model = model_path
        self._unsaved = None
        self._runner.keep_log(run_root / KPMS_SERVER_LOG)
