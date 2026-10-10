"""The payloads of the keypoint-MoSeq runner's commands.

mosaic's ``kpms`` feature sends each payload in a request of ``runner_wire``, and
``kpms_server.py`` validates it with the same model. Arrays cross as ``.npz``
files that a payload names by path, and each payload names the arrays of its
files.

This module imports only the standard library and pydantic, which both
environments install, and keeps the syntax of Python 3.10. The environment built
here runs Python 3.11 or later, and an existing keypoint-MoSeq environment may run
3.10. The runner imports it by its bare name, and mosaic as
``mosaic.behavior.feature_library.external.kpms_protocol``.
"""

from __future__ import annotations

from typing import ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AddTrack",
    "Apply",
    "Command",
    "Fit",
    "FitConfig",
    "LoadModel",
    "Payload",
    "SaveModel",
    "check_latent_dim",
]

Command: TypeAlias = Literal["add_track", "fit", "load_model", "apply", "save_model"]
"""The commands that the runner serves, besides ``shutdown``."""


def check_latent_dim(latent_dim: int, num_keypoints: int) -> None:
    """Raise ValueError if latent_dim exceeds (num_keypoints - 1) * 2."""
    max_latent = (num_keypoints - 1) * 2
    if latent_dim > max_latent:
        msg = (
            f"latent_dim={latent_dim} exceeds maximum "
            f"(num_keypoints - 1) * 2 = {max_latent} "
            f"for {num_keypoints} keypoints"
        )
        raise ValueError(msg)


class Payload(BaseModel):
    """The payload of one command. Its ``command`` names the command."""

    model_config = ConfigDict(extra="forbid")

    command: ClassVar[Command]


class AddTrack(Payload):
    """Add one recording to the tracks that the next fit reads.

    Attributes:
        key: The recording's name.
        track: An ``.npz`` file containing ``coords``, the keypoints as float
            ``(frames, keypoints, 2)``, and ``conf``, their confidences as float
            ``(frames, keypoints)``.
    """

    command: ClassVar[Command] = "add_track"
    key: str
    track: str


class FitConfig(BaseModel):
    bodyparts: list[str]
    use_bodyparts: list[str] | None = None
    anterior_bodyparts: list[str] = Field(min_length=1)
    posterior_bodyparts: list[str] = Field(min_length=1)
    fps: int = 30
    latent_dim: int = 10
    remove_outliers: bool = True
    outlier_scale_factor: float = 6.0
    mixed_map_iters: int | None = None
    parallel_message_passing: bool | None = None
    save_every_n_iters: int = 25
    resume: bool = False
    kappa_ar: float | None = None
    kappa_full: float | None = None
    num_iters_ar: int = 50
    num_iters_full: int = 500
    location_aware: bool = False
    checkpoint_dir: str | None = None

    @model_validator(mode="after")
    def _check_latent_dim(self) -> FitConfig:
        effective = (
            self.use_bodyparts if self.use_bodyparts is not None else self.bodyparts
        )
        check_latent_dim(self.latent_dim, len(effective))
        return self


class Fit(Payload):
    """Fit a model to the tracks added since the last fit, and then drop them."""

    command: ClassVar[Command] = "fit"
    config: FitConfig


class LoadModel(Payload):
    """Load a model that a ``save_model`` wrote.

    Attributes:
        path: The model file.
    """

    command: ClassVar[Command] = "load_model"
    path: str


class Apply(Payload):
    """Label each frame of one recording with the loaded or fitted model.

    Attributes:
        key: The recording's name.
        num_iters: How many iterations the resampling runs.
        track: An ``.npz`` file containing ``coords`` and ``conf``, as in
            :class:`AddTrack`.
        syllables: The ``.npz`` file that the runner writes. It contains
            ``syllables``, one int32 label per frame of *track*.
    """

    command: ClassVar[Command] = "apply"
    key: str
    num_iters: int = 500
    track: str
    syllables: str


class SaveModel(Payload):
    """Write the fitted or loaded model.

    Attributes:
        path: The model file.
    """

    command: ClassVar[Command] = "save_model"
    path: str
