"""HiDRA mouse social-behavior classifier.

HiDRA (https://github.com/neuroethology/HiDRA) is the Talmo Lab's pretrained
classifier of mouse social behavior from pose. A self-supervised trunk feeds 82
per-lab heads over 37 actions, and five ensemble configs average their outputs. Each
head is one lab's definition of one action. This feature runs the published heads
zero-shot on two to four mice per arena and writes, per frame and ordered pair, the
probability that ``id1`` performs each chosen action toward ``id2``.

The ``hidra-predict`` console script runs in its own environment, located through
:data:`HIDRA_ENV`. Its interface is files. This module writes an entry's pose in
HiDRA's long format to a temporary directory, runs the script, and reads back the
per-frame probabilities. HiDRA uses ``/dev/shm`` for scratch, so it runs on Linux.

The weights come from the Hugging Face repository ``Neuroethology/HiDRA``, one
directory per revision under ``models_root``. Probabilities differ between
revisions, so the revision is part of the run identity. The feature never downloads
weights. A missing revision is refused with the command that fetches it. See NOTICE.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self, final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mosaic.core.entry import make_entry_key
from mosaic.core.individuals import id_arenas, individual_pairs
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    Keypoints,
    Prefix,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.job import CancelToken, cancel_check
from mosaic.core.pipeline.resample_tracks import native_frame_rate
from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.pipeline.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    locate_tool,
    missing_output_error,
    subprocess_env,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    POSE_CONFIG_DESCRIPTION,
    DependencyLookup,
    EmitsLevel,
    InputStream,
    PoseConfig,
    TrackInputs,
    resolve_order_col,
)
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.optional_dependency import Prerequisite
from mosaic.user_paths import user_path

from .registry import register_feature

logger = logging.getLogger(__name__)

HIDRA_CONDA_ENV_VAR: Final = "MOSAIC_HIDRA_CONDA_ENV"
HIDRA_BIN_VAR: Final = "MOSAIC_HIDRA_BIN"
HIDRA_MODELS_ROOT_VAR: Final = "MOSAIC_HIDRA_MODELS_ROOT"
_PREDICT_SCRIPT: Final = "hidra-predict"

DEFAULT_WEIGHTS_REVISION: Final = "b5b56d316117de5900832de2dea6f2b7fea6f75f"
"""The Hugging Face revision of the weights released in October 2026."""

_INSTALL_LINE: Final = (
    "pip install 'hidra[torch] @ git+https://github.com/neuroethology/HiDRA'"
)

_NOT_FOUND_HELP: Final = (
    "HiDRA is not installed with mosaic. Install it in its own Linux environment "
    f"with {_INSTALL_LINE}, then point {HIDRA_BIN_VAR} at that environment's "
    f"hidra-predict, or name a conda environment with {HIDRA_CONDA_ENV_VAR}."
)


class HidraNotFoundError(ToolNotFoundError):
    """Raised when no ``hidra-predict`` (or ``conda``) can be located."""

    default_message = _NOT_FOUND_HELP


class HidraExitError(ToolExitError):
    """Raised when ``hidra-predict`` exits with a non-zero return code."""

    tool_name = "hidra-predict"


class HidraInputError(ValueError):
    """Raised when an entry cannot be sent to HiDRA as it is."""


class HidraWeightsMissing(FileNotFoundError):
    """Raised when the weights of the chosen revision are not on disk."""


HIDRA_ENV: Final = ToolEnv(
    tool="HiDRA",
    conda_env_var=HIDRA_CONDA_ENV_VAR,
    bin_var=HIDRA_BIN_VAR,
    bin_mode="direct",
    not_found=HidraNotFoundError,
)

type HidraBodypart = Literal[
    "tail_base", "ear_right", "ear_left", "nose", "neck", "body_center", "tail_tip"
]
"""The bodyparts HiDRA reads. Each config reads the first 4 to 7, in this order."""

type HidraConfig = Literal[
    "11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"
]
"""The five ensemble configs: the rate each samples at and its bodypart count."""

ALL_CONFIGS: Final[tuple[HidraConfig, ...]] = (
    "11fps_4bp",
    "15fps_5bp",
    "19fps_6bp",
    "23fps_7bp",
    "27fps_6bp",
)

PUBLISHED_HEADS: Final[dict[str, tuple[str, ...]]] = {
    "AdaptableSnail": (
        "approach",
        "attack",
        "avoid",
        "chase",
        "chaseattack",
        "rear",
        "submit",
    ),
    "BoisterousParrot": ("shepherd",),
    "CautiousGiraffe": (
        "chase",
        "escape",
        "reciprocalsniff",
        "sniffall",
        "sniffbody",
        "sniffgenital",
    ),
    "DeliriousFly": ("attack", "dominance", "sniff"),
    "ElegantMink": ("allogroom", "attemptmount", "intromit", "mount", "sniff"),
    "GroovyShrew": (
        "approach",
        "attemptmount",
        "climb",
        "defend",
        "dig",
        "escape",
        "rear",
        "rest",
        "run",
        "selfgroom",
        "sniffall",
        "sniffgenital",
    ),
    "InvincibleJellyfish": (
        "allogroom",
        "attack",
        "dig",
        "dominancegroom",
        "selfgroom",
        "sniffall",
        "sniffgenital",
    ),
    "JovialSwallow": ("attack", "sniff"),
    "LyricalHare": (
        "approach",
        "attack",
        "defend",
        "escape",
        "freeze",
        "rear",
        "sniff",
    ),
    "NiftyGoldfinch": (
        "approach",
        "attack",
        "biteobject",
        "chase",
        "climb",
        "defend",
        "dig",
        "escape",
        "exploreobject",
        "flinch",
        "follow",
        "rear",
        "selfgroom",
        "sniffall",
        "sniffface",
        "sniffgenital",
        "tussle",
    ),
    "PleasantMeerkat": ("follow",),
    "ReflectiveManatee": ("attack", "sniff"),
    "SparklingTapir": ("attack", "defend", "escape"),
    "TranquilPanther": (
        "intromit",
        "mount",
        "rear",
        "selfgroom",
        "sniffall",
        "sniffgenital",
    ),
    "UppityFerret": ("huddle", "reciprocalsniff", "sniffgenital"),
}
"""The 82 heads of the October 2026 weights, as ``hidra-predict --list-heads`` lists them."""

_MAX_MICE: Final = 4
_SELF_TARGET: Final = "self"
NO_CALL: Final = "none"
"""``predicted_label`` of a row on which no chosen head is called."""

_KEYPOINT_MAP_DESCRIPTION = (
    "Which HiDRA bodypart each keypoint is, keyed by keypoint name (pose.keypoint_names, "
    "or kp0, kp1, ... when those are unset). HiDRA reads tail_base, ear_right, "
    "ear_left, nose, neck, body_center and tail_tip, and treats a bodypart no keypoint "
    "maps to as unobserved. A keypoint left out is not sent. Two keypoints may not map "
    "to one bodypart."
)
_HEADS_DESCRIPTION = (
    "The HiDRA heads to report, each a lab and an action from the published heads. "
    "Each lab costs one pass of the five-config ensemble, whatever number of its "
    "actions are chosen."
)
_PIX_PER_CM_DESCRIPTION = (
    "Pixels per centimeter of the tracks. HiDRA computes in centimeters, and a wrong "
    "scale lowers every head's accuracy."
)
_MIN_CONFIDENCE_DESCRIPTION = (
    "A keypoint whose confidence column is below this value is sent as unobserved. "
    "0 sends every finite keypoint."
)
_CONFIGS_DESCRIPTION = (
    "The ensemble configs to run, or None for all five. A subset is faster and less "
    "accurate, and the bundled thresholds were calibrated on all five."
)
_THRESHOLD_DESCRIPTION = (
    "One decision threshold for every head, or None for HiDRA's per-head thresholds."
)
_WEIGHTS_REVISION_DESCRIPTION = (
    "The Hugging Face revision of the Neuroethology/HiDRA weights. Probabilities "
    "from two revisions are not comparable."
)
_MODELS_ROOT_DESCRIPTION = (
    "The directory holding one subdirectory of weights per revision, or None for "
    f"{HIDRA_MODELS_ROOT_VAR}, and then ~/.cache/mosaic/hidra."
)
_GPU_DESCRIPTION = "The CUDA device index to run on, or None for HiDRA's choice."
_HIDRA_CONDA_ENV_DESCRIPTION = (
    f"A conda environment with hidra-predict, or None to read {HIDRA_CONDA_ENV_VAR}."
)
_HIDRA_BIN_DESCRIPTION = (
    f"The path of hidra-predict, or None to read {HIDRA_BIN_VAR} and then $PATH."
)


class HidraHead(BaseModel):
    """One published HiDRA head: a lab's definition of one action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lab: str
    action: str

    @model_validator(mode="after")
    def _published(self) -> Self:
        actions = PUBLISHED_HEADS.get(self.lab)
        if actions is None:
            message = (
                f"[hidra] unknown lab {self.lab!r}. The published labs are "
                f"{sorted(PUBLISHED_HEADS)}."
            )
            raise ValueError(message)
        if self.action not in actions:
            message = (
                f"[hidra] lab {self.lab} has no {self.action!r} head. Its heads are "
                f"{list(actions)}."
            )
            raise ValueError(message)
        return self

    @property
    def suffix(self) -> str:
        """The column suffix ``<action>__<lab>``."""
        return f"{self.action}__{self.lab}"


@dataclass(frozen=True, slots=True)
class HidraPredictAvailable:
    """The prerequisite that ``hidra-predict`` can be located."""

    def missing(self) -> str | None:
        """The install help when ``hidra-predict`` cannot be located, else ``None``."""
        try:
            _ = locate_tool(HIDRA_ENV, executable=_PREDICT_SCRIPT)
        except ToolNotFoundError:
            return _NOT_FOUND_HELP
        return None


def weights_dir(models_root: str | None, revision: str) -> Path:
    """The directory that holds the weights of *revision*.

    The root is *models_root*, else ``MOSAIC_HIDRA_MODELS_ROOT``, else
    ``~/.cache/mosaic/hidra``.
    """
    chosen = models_root or os.environ.get(HIDRA_MODELS_ROOT_VAR)
    root = user_path(chosen) if chosen else Path.home() / ".cache" / "mosaic" / "hidra"
    return root / revision


def missing_weight_files(directory: Path, configs: Sequence[HidraConfig]) -> list[str]:
    """The weight files of *configs* that *directory* lacks, in either format."""
    missing: list[str] = []
    for config in configs:
        for part in ("unsupervised", "supervised_perlab_sniffall"):
            stem = f"{config}_{part}"
            if not any(
                (directory / f"{stem}{suffix}").is_file()
                for suffix in (".safetensors", ".pkl")
            ):
                missing.append(f"{stem}.safetensors")
    if not (directory / "thresholds.json").is_file():
        missing.append("thresholds.json")
    return missing


def run_hidra_predict(
    argv: list[str],
    env: dict[str, str],
    *,
    cancel_token: CancelToken | None = None,
) -> tuple[str, str]:
    """Run ``hidra-predict`` with *argv*, returning its stdout and stderr.

    Args:
        argv: The whole command, the located script first.
        env: The child's environment.
        cancel_token: Cancels the run when it fires.

    Raises:
        HidraExitError: The script exited with a non-zero status.
    """
    logger.info("Running: %s", command_summary(argv))
    stdout, stderr, returncode = run_supervised(
        argv, env=env, cancel_check=cancel_check(cancel_token)
    )
    if returncode != 0:
        raise HidraExitError(argv, returncode, stdout, stderr)
    return stdout, stderr


PROVENANCE_DIR: Final = "hidra_provenance"
"""The run-root directory recording which HiDRA install produced each entry."""

_PROBE: Final = """
import importlib.metadata, json, pathlib, subprocess
distribution = importlib.metadata.distribution("hidra")
direct = json.loads(distribution.read_text("direct_url.json") or "null") or {}
info = {"version": distribution.version, "direct_url": direct}
url = direct.get("url", "")
if url.startswith("file://"):
    try:
        info["git_commit"] = subprocess.run(
            ["git", "-C", url[len("file://"):], "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
    except OSError:
        pass
from hidra import cli
info["thresholds_csv"] = pathlib.Path(cli.THR_CSV).read_text()
print(json.dumps(info, sort_keys=True))
"""
"""Run by the interpreter beside ``hidra-predict``: its version, install source and
bundled thresholds. A git install records its commit in ``direct_url.json``, and an
editable one is asked for its checkout's commit."""


def read_hidra_provenance(predict_argv: list[str], env: dict[str, str]) -> str:
    """The installed HiDRA's version, source, thresholds and head table, as JSON text.

    The interpreter is the ``python`` beside the located ``hidra-predict``, the
    last element of *predict_argv*, also under ``conda run``. A part that cannot be
    read is recorded as an error instead of failing the run.

    Args:
        predict_argv: The argv prefix that launches ``hidra-predict``.
        env: The environment HiDRA runs in.
    """
    python = str(Path(predict_argv[-1]).parent / "python")
    install = _probe_text([*predict_argv[:-1], python, "-c", _PROBE], env)
    if not install.startswith("{"):
        install = json.dumps({"error": install})
    heads = _probe_text([*predict_argv, "--list-heads"], env)
    return f'{{"install": {install}, "heads": {json.dumps(heads)}}}\n'


def _probe_text(argv: list[str], env: dict[str, str]) -> str:
    """What *argv* prints, or ``error: ...`` when it cannot run or fails."""
    try:
        stdout, stderr, returncode = run_supervised(argv, env=env)
    except OSError as error:
        return f"error: {error}"
    if returncode != 0:
        return f"error: exit {returncode}: {stderr.strip()[-2000:]}"
    return stdout.strip()


def _per_frame_first(df: pd.DataFrame, order_col: str, column: str) -> pd.Series:
    """The first value of *column* at each value of *order_col*."""
    return df.groupby(order_col, sort=True)[column].first()


HIDRA_PROVIDES: Final = Provides(
    adds=(
        Column(C.frame_col),
        Column(C.seq_col),
        Column("id1"),
        Column("id2"),
        Column("perspective"),
        Prefix("prob_"),
        Prefix("call_"),
        Column("predicted_label"),
    ),
    conditional=kept(C.group_col),
)


@final
@register_feature
class HidraFeature:
    """HiDRA's pretrained mouse social-behavior heads, per frame and ordered pair.

    Writes one row per frame and ordered pair of individuals (``emits = "pair"``):

      - ``frame``, ``sequence``, ``id1``, ``id2``, ``perspective``, and ``group``
        when the input has it. Perspective 0 has ``id1`` before ``id2`` in sorted
        order, and perspective 1 swaps them.
      - ``prob_<action>__<lab>``: HiDRA's probability that ``id1`` performs the
        action toward ``id2``. A self-directed head (such as ``selfgroom``) has one
        value per animal, written on every row whose ``id1`` is that animal.
      - ``call_<action>__<lab>``: whether that probability reaches the head's
        threshold.
      - ``predicted_label``: the called head with the highest probability, as
        ``<action>__<lab>``, or ``none``.

    Each arena of an entry (the entry, without an ``arena`` column) is one HiDRA
    scene of two to four individuals, and an entry must have one frame rate. Field
    documentation is on
    :class:`~mosaic.behavior.feature_library.hidra_feature.HidraFeature.Params`.
    """

    category = "per-frame"
    name = "hidra"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    resource_class = "gpu"
    prerequisites: ClassVar[tuple[Prerequisite, ...]] = (HidraPredictAvailable(),)
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.seq_col),
        ORDER_COLUMN,
        Keypoints(),
    )
    provides: ClassVar[Provides] = HIDRA_PROVIDES

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        pose: Annotated[PoseConfig, Declared(POSE_CONFIG_DESCRIPTION)] = Field(
            default_factory=PoseConfig
        )
        keypoint_map: Annotated[
            dict[str, HidraBodypart], Declared(_KEYPOINT_MAP_DESCRIPTION)
        ] = Field(default_factory=dict)
        heads: Annotated[list[HidraHead], Declared(_HEADS_DESCRIPTION)] = Field(
            default_factory=list
        )
        pix_per_cm: Annotated[
            float | None, Declared(_PIX_PER_CM_DESCRIPTION, unit="px/cm")
        ] = Field(default=None, gt=0)
        min_confidence: Annotated[float, Declared(_MIN_CONFIDENCE_DESCRIPTION)] = 0.0
        configs: Annotated[list[HidraConfig] | None, Declared(_CONFIGS_DESCRIPTION)] = (
            None
        )
        threshold: Annotated[float | None, Declared(_THRESHOLD_DESCRIPTION)] = Field(
            default=None, ge=0, le=1
        )
        weights_revision: Annotated[str, Declared(_WEIGHTS_REVISION_DESCRIPTION)] = (
            DEFAULT_WEIGHTS_REVISION
        )
        models_root: Annotated[
            str | None, HASH_EXCLUDE, Declared(_MODELS_ROOT_DESCRIPTION)
        ] = None
        gpu: Annotated[int | None, HASH_EXCLUDE, Declared(_GPU_DESCRIPTION)] = None
        hidra_conda_env: Annotated[
            str | None, HASH_EXCLUDE, Declared(_HIDRA_CONDA_ENV_DESCRIPTION)
        ] = None
        hidra_bin: Annotated[
            str | None, HASH_EXCLUDE, Declared(_HIDRA_BIN_DESCRIPTION)
        ] = None

        @model_validator(mode="after")
        def _one_keypoint_per_bodypart(self) -> Self:
            bodyparts = list(self.keypoint_map.values())
            repeated = sorted({b for b in bodyparts if bodyparts.count(b) > 1})
            if repeated:
                message = (
                    f"[hidra] keypoint_map maps two keypoints to {repeated}. Map one "
                    "keypoint to each HiDRA bodypart."
                )
                raise ValueError(message)
            return self

    def __init__(
        self,
        inputs: HidraFeature.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.cancel_token: CancelToken | None = None
        self._run_root: Path | None = None
        self._provenance: str | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._run_root = run_root
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        if not p.heads:
            raise HidraInputError("[hidra] choose at least one head in heads.")
        if not p.keypoint_map:
            raise HidraInputError(
                "[hidra] keypoint_map is empty. Map keypoint names to HiDRA bodyparts."
            )
        if p.pix_per_cm is None:
            raise HidraInputError("[hidra] set pix_per_cm, the tracks' pixel scale.")
        configs = list(p.configs) if p.configs else list(ALL_CONFIGS)
        models = weights_dir(p.models_root, p.weights_revision)
        absent = missing_weight_files(models, configs)
        if absent:
            message = (
                f"[hidra] the weights of revision {p.weights_revision} are not in "
                f"{models} (missing {', '.join(absent)}). Fetch them with "
                f"HIDRA_MODELS_DIR={models} hidra-download-models --revision "
                f"{p.weights_revision}"
            )
            raise HidraWeightsMissing(message)

        order_col = resolve_order_col(df)
        scenes = _scenes(df)
        fps = native_frame_rate(df, source="[hidra]")
        sequence = str(df[C.seq_col].iloc[0])
        first_frame = int(df[order_col].min())

        with tempfile.TemporaryDirectory(prefix="mosaic-hidra-") as scratch:
            staged = Path(scratch) / "in"
            out = Path(scratch) / "out"
            staged.mkdir()
            files: list[str] = []
            for index, scene in enumerate(scenes):
                members = df[df[C.id_col].isin(scene)]
                long = self._long_format(members, order_col, scene)
                long["video_frame"] = long["video_frame"] - first_frame
                name = f"{_scene_stem(index)}.parquet"
                _ = write_parquet_atomic(long, staged / name)
                files.append(name)
            pd.DataFrame(
                {
                    "file": files,
                    "pix_per_cm": [p.pix_per_cm] * len(files),
                    "fps": [fps] * len(files),
                }
            ).to_csv(staged / "metadata.csv", index=False)
            env = subprocess_env({"HIDRA_MODELS_DIR": str(models)})
            predict = self._predict_argv()
            self._record_provenance(df, sequence, predict, env)
            stdout, stderr = run_hidra_predict(
                self._argv(predict, staged, out, configs),
                env,
                cancel_token=self.cancel_token,
            )
            predicted: list[pd.DataFrame] = []
            for index in range(len(scenes)):
                frames_path = out / f"{_scene_stem(index)}.frames.parquet"
                if not frames_path.is_file():
                    raise missing_output_error(
                        _PREDICT_SCRIPT, frames_path, stdout, stderr
                    )
                predicted.append(pd.read_parquet(frames_path))

        return self._pair_rows(df, order_col, scenes, first_frame, sequence, predicted)

    def _predict_argv(self) -> list[str]:
        """The argv prefix that launches ``hidra-predict`` where this run placed it."""
        p = self.params
        return locate_tool(
            HIDRA_ENV.placed(conda_env=p.hidra_conda_env, bin_path=p.hidra_bin),
            executable=_PREDICT_SCRIPT,
        ).argv

    def _record_provenance(
        self,
        df: pd.DataFrame,
        sequence: str,
        predict: list[str],
        env: dict[str, str],
    ) -> None:
        """Record which HiDRA install this entry is predicted by, under the run root.

        The install is read once per feature instance and stored by its digest in
        ``hidra_provenance/<digest>.json``, and ``hidra_provenance/entries/<entry>.txt``
        names the digest of each entry. An update of HiDRA between two invocations of
        one run therefore shows as two digests. Nothing is written when the feature
        runs outside ``run_feature``.
        """
        if self._run_root is None:
            return
        if self._provenance is None:
            self._provenance = read_hidra_provenance(predict, env)
        text = self._provenance
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        directory = self._run_root / PROVENANCE_DIR
        record = directory / f"{digest}.json"
        if not record.is_file():
            atomic_write(record, lambda path: path.write_text(text))
        group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
        entry = directory / "entries" / f"{make_entry_key(group, sequence)}.txt"
        atomic_write(entry, lambda path: path.write_text(f"{digest}\n"))

    def _argv(
        self,
        predict: list[str],
        staged: Path,
        out: Path,
        configs: list[HidraConfig],
    ) -> list[str]:
        p = self.params
        labs = sorted({head.lab for head in p.heads})
        actions = sorted({head.action for head in p.heads})
        argv = [
            *predict,
            str(staged),
            "--labs",
            ",".join(labs),
            "--actions",
            ",".join(actions),
            "--output",
            "probs",
            "--out",
            str(out),
        ]
        if p.configs:
            argv += ["--configs", ",".join(configs)]
        if p.threshold is not None:
            argv += ["--threshold", repr(p.threshold)]
        if p.gpu is not None:
            argv += ["--gpu", str(p.gpu)]
        return argv

    def _long_format(
        self, df: pd.DataFrame, order_col: str, ids: list[object]
    ) -> pd.DataFrame:
        """The entry's mapped keypoints as HiDRA's rows, unobserved points dropped."""
        p = self.params
        x_cols, y_cols = p.pose.keypoint_columns()
        indices = p.pose.selected_indices()
        names = p.pose.names(len(x_cols))
        unknown = sorted(set(p.keypoint_map) - set(names))
        if unknown:
            message = (
                f"[hidra] keypoint_map names {unknown}, which are not keypoints of "
                f"this pose; its keypoints are {names}."
            )
            raise HidraInputError(message)
        mouse_of = {identity: f"mouse{n + 1}" for n, identity in enumerate(ids)}
        mice = df[C.id_col].map(mouse_of).to_numpy()
        frames = df[order_col].to_numpy(dtype=np.int64)
        parts: list[pd.DataFrame] = []
        for position, name in enumerate(names):
            bodypart = p.keypoint_map.get(name)
            if bodypart is None:
                continue
            if x_cols[position] not in df.columns or y_cols[position] not in df.columns:
                message = (
                    f"[hidra] keypoint {name} has no columns {x_cols[position]} and "
                    f"{y_cols[position]} in this entry."
                )
                raise HidraInputError(message)
            x = df[x_cols[position]].to_numpy(dtype=np.float64)
            y = df[y_cols[position]].to_numpy(dtype=np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            confidence = f"{p.pose.confidence_prefix}{indices[position]}"
            if p.min_confidence > 0:
                if confidence not in df.columns:
                    message = (
                        f"[hidra] min_confidence is {p.min_confidence}, and keypoint "
                        f"{name} has no confidence column {confidence}. Set "
                        "min_confidence to 0 for tracks without confidences."
                    )
                    raise HidraInputError(message)
                keep &= df[confidence].to_numpy(dtype=np.float64) >= p.min_confidence
            parts.append(
                pd.DataFrame(
                    {
                        "video_frame": frames[keep],
                        "mouse_id": mice[keep],
                        "bodypart": bodypart,
                        "x": x[keep].astype(np.float32),
                        "y": y[keep].astype(np.float32),
                    }
                )
            )
        if not parts:
            raise HidraInputError("[hidra] keypoint_map maps no keypoint of this pose.")
        long = pd.concat(parts, ignore_index=True)
        if long.empty:
            message = (
                f"[hidra] individuals {ids} have no observed mapped keypoint in this "
                "entry."
            )
            raise HidraInputError(message)
        return long

    def _pair_rows(
        self,
        df: pd.DataFrame,
        order_col: str,
        scenes: list[list[object]],
        first_frame: int,
        sequence: str,
        predicted: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """One row per input frame and ordered pair, with each head's columns."""
        p = self.params
        frame_values = np.sort(df[order_col].dropna().unique()).astype(np.int64)
        offsets = frame_values - first_frame
        series: dict[_SeriesKey, tuple[np.ndarray, np.ndarray]] = {}
        for index, table in enumerate(predicted):
            for key, group in table.groupby(
                ["subject", "target", "lab", "action"], sort=False
            ):
                subject, target, lab, action = (str(part) for part in key)
                length = int(group["frame"].max()) + 1
                at = group["frame"].to_numpy(dtype=np.int64)
                prob = np.full(length, np.nan, dtype=np.float32)
                prob[at] = group["prob"].to_numpy(dtype=np.float32)
                call = np.zeros(length, dtype=bool)
                call[at] = group["call"].to_numpy().astype(bool)
                series[(index, subject, target, lab, action)] = (prob, call)

        def head_columns(
            index: int, subject: str, target: str, head: HidraHead
        ) -> tuple[np.ndarray, np.ndarray]:
            found = series.get((index, subject, target, head.lab, head.action))
            if found is None:
                found = series.get(
                    (index, subject, _SELF_TARGET, head.lab, head.action)
                )
            prob = np.full(len(offsets), np.nan, dtype=np.float32)
            call = np.zeros(len(offsets), dtype=bool)
            if found is not None:
                inside = offsets < len(found[0])
                prob[inside] = found[0][offsets[inside]]
                call[inside] = found[1][offsets[inside]]
            return prob, call

        arenas = id_arenas(df)
        blocks: list[pd.DataFrame] = []
        for index, scene in enumerate(scenes):
            mouse_of = {identity: f"mouse{n + 1}" for n, identity in enumerate(scene)}
            for id_a, id_b in individual_pairs(scene, arenas):
                for perspective, (focal, other) in enumerate(
                    ((id_a, id_b), (id_b, id_a))
                ):
                    block: dict[str, object] = {
                        C.frame_col: frame_values,
                        C.seq_col: sequence,
                        "id1": focal,
                        "id2": other,
                        "perspective": np.int64(perspective),
                    }
                    for head in p.heads:
                        prob, call = head_columns(
                            index, mouse_of[focal], mouse_of[other], head
                        )
                        block[f"prob_{head.suffix}"] = prob
                        block[f"call_{head.suffix}"] = call
                    blocks.append(pd.DataFrame(block))
        result = pd.concat(blocks, ignore_index=True)
        if C.group_col in df.columns:
            group = _per_frame_first(df, order_col, C.group_col)
            result[C.group_col] = result[C.frame_col].map(group).to_numpy()
        result["predicted_label"] = _predicted_label(result, p.heads)
        return result


type _SeriesKey = tuple[int, str, str, str, str]
"""A scene's index, then HiDRA's subject, target, lab and action."""


def _scene_stem(index: int) -> str:
    """The file stem of the *index*-th scene staged for ``hidra-predict``."""
    return f"scene{index}"


def _scenes(df: pd.DataFrame) -> list[list[object]]:
    """The individuals HiDRA sees together: each arena's, or the entry's.

    HiDRA reads one file as one scene, so individuals of two arenas are staged in two
    files. An arena of one individual has no pair and is left out, and an individual
    without an arena relates to none.

    Raises:
        HidraInputError: An arena (or the entry, without arenas) holds more than four
            individuals, or no arena holds two.
    """
    ids: list[object] = sorted(df[C.id_col].dropna().unique().tolist())
    arenas = id_arenas(df)
    if arenas is None:
        groups = [ids]
        where = "this entry holds"
    else:
        by_arena: dict[str, list[object]] = {}
        for identity in ids:
            arena = arenas.get(identity)
            if arena is not None:
                by_arena.setdefault(arena, []).append(identity)
        groups = [by_arena[arena] for arena in sorted(by_arena)]
        where = "its arenas hold"
    sizes = [len(group) for group in groups]
    if any(size > _MAX_MICE for size in sizes) or not any(size >= 2 for size in sizes):
        message = (
            f"[hidra] HiDRA relates 2 to {_MAX_MICE} individuals in a scene; {where} "
            f"{sizes or [0]}."
        )
        raise HidraInputError(message)
    return [group for group in groups if len(group) >= 2]


def _predicted_label(result: pd.DataFrame, heads: Sequence[HidraHead]) -> np.ndarray:
    """The called head with the highest probability on each row, else ``none``."""
    suffixes = [head.suffix for head in heads]
    probs = np.column_stack(
        [result[f"prob_{suffix}"].to_numpy(dtype=np.float64) for suffix in suffixes]
    )
    calls = np.column_stack(
        [result[f"call_{suffix}"].to_numpy(dtype=bool) for suffix in suffixes]
    )
    masked = np.where(calls, probs, -np.inf)
    best = np.argmax(masked, axis=1)
    labels = np.array(suffixes, dtype=object)[best]
    return np.where(calls.any(axis=1), labels, NO_CALL)
