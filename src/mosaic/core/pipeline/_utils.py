from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel

# --- Pipeline data types ---


@dataclass
class ChunkedPayload:
    """Feature output: pre-computed 2D array, written in row-chunks."""

    parquet_data: np.ndarray
    columns: list[str]
    sequence: str = ""
    group: str | None = None
    chunk_size: int = 10000


@dataclass
class StreamPayload:
    """Feature output: lazy chunk iterator, written as streaming parquet."""

    parquet_chunk_iter: Iterator[tuple[int, np.ndarray]]
    columns: list[str]
    sequence: str = ""
    group: str | None = None
    pair_ids: tuple[int, int] | None = None
    frame_indices: np.ndarray | None = None


@dataclass
class DataPayload:
    """Feature output: raw data array with explicit column names."""

    data: np.ndarray
    columns: list[str]
    sequence: str | None = None
    group: str | None = None
    frame_indices: np.ndarray | None = None
    pair_ids_per_row: np.ndarray | None = None


@dataclass
class ResolvedInput:
    """A single resolved input within an InputScope."""

    kind: str
    feature: str | None = None
    run_id: str | None = None
    path_map: dict[tuple[str, str], Path] = field(default_factory=dict)
    columns: list[str] | None = None


@dataclass
class InputScope:
    """Resolved scope for multi-input / inputset feature runs."""

    pairs: set[tuple[str, str]] = field(default_factory=set)
    safe_sequences: set[str] = field(default_factory=set)
    resolved_inputs: list[ResolvedInput] = field(default_factory=list)
    inputset: str | None = None
    meta: dict[str, object] = field(default_factory=dict)
    groups: list[str] | None = None
    sequences: list[str] | None = None


@dataclass
class FeatureMeta:
    """Output metadata for a single (group, sequence) within run_feature."""

    group: str
    sequence: str
    out_path: Path


# --- Utility functions ---


def json_ready(obj: object) -> object:
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return f"<{type(obj).__name__}>"
    return obj


def hash_params(d: object) -> str:
    d = json_ready(d)
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def build_scope_key(
    groups: Iterable[str] | None, sequences: Iterable[str] | None
) -> dict[str, list[str]] | None:
    """Normalize scope into a deterministic hashable dict, or None if unrestricted."""
    scope: dict[str, list[str]] = {}
    if groups is not None:
        scope["groups"] = sorted({str(g) for g in groups})
    if sequences is not None:
        scope["sequences"] = sorted({str(s) for s in sequences})
    return scope if scope else None


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def coerce_np(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
