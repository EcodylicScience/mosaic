"""Shared protocol models and wire helpers for the kpms server/client.

Defines the request/response Pydantic models and the newline-delimited JSON
framing used over Unix domain sockets. Importable from both the main mosaic
environment (client) and the external .venv (server).

Dependencies: pydantic, numpy (available in both environments).
"""

from __future__ import annotations

import base64
import socket
from typing import Annotated, Literal, Self, TypeAlias

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter, model_validator

# --- Array encoding ---


class ArraySpec(BaseModel):
    dtype: str
    shape: tuple[int, ...]
    data: str


def decode_array(spec: ArraySpec) -> np.ndarray:
    raw = base64.b64decode(spec.data)
    return np.frombuffer(raw, dtype=np.dtype(spec.dtype)).reshape(spec.shape)


def encode_array(arr: np.ndarray) -> ArraySpec:
    return ArraySpec(
        dtype=str(arr.dtype),
        shape=tuple(arr.shape),
        data=base64.b64encode(arr.tobytes()).decode(),
    )


# --- Validation helpers ---


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


# --- Requests ---


class AddTrackRequest(BaseModel):
    command: Literal["add_track"]
    key: str
    coords: ArraySpec
    conf: ArraySpec


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
    def _check_latent_dim(self) -> Self:
        effective = (
            self.use_bodyparts if self.use_bodyparts is not None else self.bodyparts
        )
        check_latent_dim(self.latent_dim, len(effective))
        return self


class FitRequest(BaseModel):
    command: Literal["fit"]
    config: FitConfig


class LoadModelRequest(BaseModel):
    command: Literal["load_model"]
    path: str


class ApplyRequest(BaseModel):
    command: Literal["apply"]
    key: str
    num_iters: int = 500
    coords: ArraySpec
    conf: ArraySpec


class SaveModelRequest(BaseModel):
    command: Literal["save_model"]
    path: str


class ShutdownRequest(BaseModel):
    command: Literal["shutdown"]


Request: TypeAlias = Annotated[
    AddTrackRequest
    | FitRequest
    | LoadModelRequest
    | ApplyRequest
    | SaveModelRequest
    | ShutdownRequest,
    Field(discriminator="command"),
]

RequestAdapter: TypeAdapter[Request] = TypeAdapter(Request)


# --- Responses ---


class OkResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str
    traceback: str | None = None


class ApplyResponse(BaseModel):
    status: Literal["ok"] = "ok"
    syllables: ArraySpec


# --- Wire helpers ---


def send_message(conn: socket.socket, message: BaseModel) -> None:
    """Send a newline-terminated JSON message."""
    conn.sendall(message.model_dump_json().encode() + b"\n")


def receive_message(conn: socket.socket) -> bytes:
    """Read a single newline-terminated line from conn."""
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            raise ConnectionError("connection closed while reading")
        buf += chunk
    return buf.split(b"\n", 1)[0]
