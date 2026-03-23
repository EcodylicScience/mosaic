# dataset.py
from __future__ import annotations

import datetime
import fnmatch
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml  # pip install pyyaml

from .helpers import ensure_text_column, to_safe_name
from .pipeline._utils import coerce_np as _coerce_np, now_iso as _now_iso


def _probe_video_metadata(path: Path) -> dict[str, Any]:
    """
    Use ffprobe to collect width/height/fps/codec metadata.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return {}
        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        rate_raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        fps = _parse_ffprobe_rate(rate_raw)
        codec = stream.get("codec_name") or ""
        return {
            "width": width if width > 0 else "",
            "height": height if height > 0 else "",
            "fps": fps if fps else "",
            "codec": codec,
        }
    except Exception as exc:
        print(f"[index_media] ffprobe failed for {path}: {exc}", file=sys.stderr)
        return {}


def _parse_ffprobe_rate(rate: Optional[str]) -> Optional[float]:
    if not rate:
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            num = float(num)
            den = float(den)
            if den == 0:
                return None
            return num / den
        return float(rate)
    except Exception:
        return None


def _normalize_patterns(pats) -> tuple[str, ...]:
    if pats is None:
        return tuple()
    if isinstance(pats, str):
        return (pats,)
    try:
        return tuple(pats)
    except TypeError:
        return (str(pats),)


def _normalize_path_map(path_map: Mapping[str, str]) -> list[tuple[Path, Path]]:
    normalized: list[tuple[Path, Path]] = []
    for src, dst in path_map.items():
        if not src or not dst:
            continue
        normalized.append((Path(src).expanduser(), Path(dst).expanduser()))
    normalized = [pair for pair in normalized if pair[0] != pair[1]]
    normalized.sort(key=lambda pair: len(pair[0].as_posix()), reverse=True)
    return normalized


def _remap_single_path(
    path: Path, mapping: Sequence[tuple[Path, Path]]
) -> Optional[Path]:
    for src, dst in mapping:
        try:
            rel = path.relative_to(src)
            return dst / rel
        except ValueError:
            continue
    return None


from dataclasses import field
from typing import Any, Callable, Dict, Mapping, Tuple

# A tiny registry so you can plug converters: src_format -> callable
TrackConverter = Callable[[Path, dict], pd.DataFrame]
TRACK_CONVERTERS: dict[str, TrackConverter] = {}


def register_track_converter(src_format: str, fn: TrackConverter):
    TRACK_CONVERTERS[src_format] = fn


# Optional: per-format sequence enumerators (for multi-sequence files)
TrackSeqEnumerator = Callable[[Path], list[tuple[str, str]]]
TRACK_SEQ_ENUM: dict[str, TrackSeqEnumerator] = {}


def register_track_seq_enumerator(src_format: str, fn: TrackSeqEnumerator):
    TRACK_SEQ_ENUM[src_format] = fn


# ----------- Label converter registry -----------


class LabelConverter(Protocol):
    """Protocol for label converter plugins."""

    src_format: str  # e.g., "calms21_npy", "boris_csv"
    label_kind: str  # e.g., "behavior", "id_tags"
    label_format: str  # e.g., "calms21_behavior_v1"

    def convert(
        self,
        src_path: Path,
        raw_row: pd.Series,
        labels_root: Path,
        params: dict,
        overwrite: bool,
        existing_pairs: set[tuple[str, str]],
    ) -> list[dict]:
        """
        Convert a source file to label npz files.

        Returns: List of index row dicts for labels/index.csv
        """
        ...

    def get_metadata(self) -> dict:
        """Optional: return format-specific metadata for dataset.meta['labels'][kind]."""
        ...


# Registry: (src_format, label_kind) -> converter class
LABEL_CONVERTERS: dict[tuple[str, str], type] = {}


def register_label_converter(cls: type):
    """Decorator to register label converters."""
    key = (cls.src_format, cls.label_kind)
    LABEL_CONVERTERS[key] = cls
    return cls


# ----------- Track schema system (extracted to tracking/schema.py) -----------
from .schema import (
    TRACK_SCHEMAS,
    TrackSchema,
    ensure_track_schema,
)

# --- Standardized label metadata ---
BEHAVIOR_LABEL_MAP = {
    0: "attack",
    1: "investigation",
    2: "mount",
    3: "other_interaction",
}

LABEL_INDEX_COLUMNS = [
    "kind",
    "label_format",
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    "abs_path",
    "source_abs_path",
    "source_md5",
    "n_frames",
    "label_ids",
    "label_names",
]


def _md5(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


try:
    import yaml

    _YAML_OK = True
except Exception:
    _YAML_OK = False

INPUTSET_DIRNAME = "inputsets"


def _dataset_base_dir(ds) -> Path:
    """
    Resolve the directory that holds dataset-level config (sibling to dataset manifest).
    """
    base = getattr(ds, "manifest_path", None)
    if base is not None:
        base = Path(base)
        base = base.parent if base.is_file() else base
    else:
        base = Path(ds.get_root("features")).parent
    base.mkdir(parents=True, exist_ok=True)
    return base


def _inputset_dir(ds) -> Path:
    base = _dataset_base_dir(ds)
    path = base / INPUTSET_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _inputset_path(ds, name: str) -> Path:
    safe = to_safe_name(name)
    if not safe:
        raise ValueError("Inputset name must contain alphanumeric characters.")
    return _inputset_dir(ds) / f"{safe}.json"


def save_inputset(
    ds,
    name: str,
    inputs: list[dict],
    description: Optional[str] = None,
    overwrite: bool = False,
    filter_start_frame: Optional[int] = None,
    filter_end_frame: Optional[int] = None,
    filter_start_time: Optional[float] = None,
    filter_end_time: Optional[float] = None,
    pair_filter: Optional[dict] = None,
) -> Path:
    """
    Persist an inputset JSON under <dataset_root>/inputsets/<name>.json.

    Parameters
    ----------
    ds : Dataset
        The dataset instance
    name : str
        Name for the inputset
    inputs : list[dict]
        List of input specifications
    description : str, optional
        Human-readable description
    overwrite : bool, default False
        Whether to overwrite existing inputset
    filter_start_frame : int, optional
        Discard frames < this value when loading
    filter_end_frame : int, optional
        Discard frames >= this value when loading
    filter_start_time : float, optional
        Discard rows where time < this value (seconds)
    filter_end_time : float, optional
        Discard rows where time >= this value (seconds)
    pair_filter : dict, optional
        Pair-level filter applied when loading pair features.  Example for
        nearest-neighbor filtering::

            {"type": "nearest_neighbor",
             "feature": "nearest-neighbor",
             "run_id": None}
    """
    path = _inputset_path(ds, name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Inputset '{name}' already exists: {path}")
    payload = {
        "name": name,
        "description": description or "",
        "inputs": inputs or [],
    }
    # Add filter params if any are specified
    if filter_start_frame is not None:
        payload["filter_start_frame"] = filter_start_frame
    if filter_end_frame is not None:
        payload["filter_end_frame"] = filter_end_frame
    if filter_start_time is not None:
        payload["filter_start_time"] = filter_start_time
    if filter_end_time is not None:
        payload["filter_end_time"] = filter_end_time
    if pair_filter is not None:
        payload["pair_filter"] = pair_filter
    path.write_text(json.dumps(payload, indent=2))
    return path


def _fingerprint_inputs(inputs: list[dict]) -> str:
    serialized = json.dumps(inputs or [], sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _load_inputset(ds, name: str) -> tuple[list[dict], dict]:
    path = _inputset_path(ds, name)
    if not path.exists():
        raise FileNotFoundError(f"Inputset '{name}' not found at {path}")
    data = json.loads(path.read_text())
    inputs = data.get("inputs") or []
    fingerprint = _fingerprint_inputs(inputs)
    meta = {
        "inputset": name,
        "inputs_fingerprint": fingerprint,
        "inputs_source": "inputset",
        "inputset_path": str(path),
        "description": data.get("description", ""),
        # Time/frame filtering params
        "filter_start_frame": data.get("filter_start_frame"),
        "filter_end_frame": data.get("filter_end_frame"),
        "filter_start_time": data.get("filter_start_time"),
        "filter_end_time": data.get("filter_end_time"),
        # Pair-level filtering
        "pair_filter": data.get("pair_filter"),
    }
    return inputs, meta


def _resolve_inputs(
    ds,
    explicit_inputs: Optional[list[dict]],
    inputset_name: Optional[str],
    explicit_override: bool = False,
) -> tuple[list[dict], dict]:
    """
    Determine which inputs to use based on explicit params vs. named inputset.
    If inputset_name is provided, it overrides defaults unless explicit_inputs was
    explicitly supplied by the caller (explicit_override=True).
    """
    if inputset_name:
        inputs, meta = _load_inputset(ds, inputset_name)
        if explicit_inputs and explicit_override:
            inputs = explicit_inputs
            meta = {
                "inputset": inputset_name,
                "inputs_fingerprint": _fingerprint_inputs(inputs),
                "inputs_source": "explicit",
            }
    else:
        inputs = explicit_inputs or []
        meta = {
            "inputset": None,
            "inputs_fingerprint": _fingerprint_inputs(inputs),
            "inputs_source": "explicit" if explicit_inputs else "default",
        }

    if not inputs:
        raise ValueError(
            "No inputs resolved; provide params['inputs'] or params['inputset']."
        )
    return inputs, meta


############# DATASET

default_roots = {
    "media": "media",
    "features": "features",  # calculated features (input to models), e.g. wavelets, projections, embeddings
    "labels": "labels",  # GT annotations, .npy/.csv
    "models": "models",  # trained models, reports, plots
    "tracks": "tracks",
    "tracks_raw": "tracks_raw",
    "frames": "frames",  # extracted video frames (PNGs), can be very large
}


def new_dataset_manifest(
    name: str,
    base_dir: str | Path,
    roots: dict[str, str | Path] = default_roots,
    version: str = "0.1.0",
    index_format: str = "group/sequence",
    outfile: str | Path | None = None,
    # Continuous dataset support
    dataset_type: str = "discrete",  # "discrete" or "continuous"
    segment_duration: str | None = None,  # e.g., "1H", "30min", "1D"
    time_column: str | None = None,  # column name for timestamps, e.g., "timestamp"
) -> Path:
    """
    Create a minimal, extensible dataset manifest (YAML) with only a few required fields.
    - name: dataset name (e.g., "CALMS21")
    - base_dir: absolute or relative base directory for the dataset
    - roots: dict of subpaths you actually use NOW (e.g., {"media": "videos", "features": "features", "labels": "labels"})
    - index_format: how you think about addressing items ("group/sequence" is recommended)
    Returns the path to the created YAML.
    """
    base_dir = Path(base_dir).resolve()
    # Normalize roots -> relative paths (portable) when inside base_dir
    norm_roots = {}
    for k, v in roots.items():
        full = (base_dir / Path(v)).resolve()
        full.mkdir(parents=True, exist_ok=True)
        try:
            norm_roots[k] = str(full.relative_to(base_dir))
        except ValueError:
            norm_roots[k] = str(full)  # outside base_dir, keep absolute

    manifest = {
        "name": name,
        "version": version,
        "uuid": str(uuid.uuid4()),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "index_format": index_format,  # recommended: "group/sequence"
        "roots": norm_roots,  # required minimal roots you actually use now
        "dataset_type": dataset_type,  # "discrete" (default) or "continuous"
        # You can append optional fields later without placeholders
    }

    # Add continuous-specific fields if applicable
    if dataset_type == "continuous":
        if segment_duration:
            manifest["segment_duration"] = segment_duration
        if time_column:
            manifest["time_column"] = time_column

    header_comment = """# ==========================================================
# DATASET MANIFEST (extensible YAML)
# Minimal required fields above; append optional fields below
#
# DATASET TYPES:
#   dataset_type: "discrete"     # Default: distinct recordings (trials, sessions)
#   dataset_type: "continuous"   # Long continuous recordings (days/months)
#     segment_duration: "1H"     # Segment size for continuous (e.g., "1H", "30min", "1D")
#     time_column: "timestamp"   # Column name for time-based operations
#
# Common OPTIONAL fields you may add later:
#   fps_default: 30.0
#   resolution_default: [1920, 1080]
#   n_animals_default: 2
#   species: ""
#   groups:                      # [{id, notes, condition, date, ...}]
#   sequences:                   # [{id, group, media_path, pose_path, fps, n_frames, n_animals, ...}]
#   splits:                      # {task1_train: [...], task1_test: [...], ...}
#   labels_map:                  # {0: attack, 1: investigation, ...}
#   skeleton:                    # [[p1, p2], ...]
#   bodyparts:                   # ["snout","neck",...]
#   processing:                  # [{step, time, params_hash, code_commit, ...}]
#   pose_model:                  # {name, engine, checkpoint, config}
#   behavior_model:              # {name, checkpoint, config}
#   provenance:                  # {repo, commit, env}
#   quality:                     # {missing_rate, drift, ...}
#   modalities:                  # ["video","pose","audio",...]
#   cameras:                     # {cam0: {intrinsics:..., extrinsics:...}, ...}
#   notes: |
#     Free-form notes about the dataset.
# ==========================================================
"""

    text = header_comment + yaml.safe_dump(
        manifest, sort_keys=False, default_flow_style=False
    )

    if outfile is None:
        outfile = base_dir / "dataset.yaml"
    else:
        outfile = Path(outfile)

    outfile.write_text(text, encoding="utf-8")
    print(f"Wrote dataset manifest -> {outfile}")
    return outfile


# --------------------------
# Dataset manifest + manager
# --------------------------


@dataclass
class Dataset:
    manifest_path: Path
    name: str = "unnamed"
    version: str = "0.1"
    format: str = "yaml"
    roots: Dict[str, str] = field(
        default_factory=lambda: {
            "media": "",
            "tracks_raw": "",
            "tracks": "",
            "tracks_raw": "",
            "features": "",
            "labels": "",
            "models": "",
        }
    )
    meta: Dict[str, Any] = field(default_factory=dict)
    _path_map: list[tuple[Path, Path]] = field(
        default_factory=list, init=False, repr=False
    )

    # Continuous dataset support
    dataset_type: str = "discrete"  # "discrete" or "continuous"
    segment_duration: str | None = None  # e.g., "1H", "30min", "1D"
    time_column: str | None = None  # column name for timestamps

    @property
    def is_continuous(self) -> bool:
        """Check if this is a continuous recording dataset."""
        return self.dataset_type == "continuous"

    # ---- Instance load method ----
    def load(self, ensure_roots: bool = True) -> "Dataset":
        """Load dataset metadata from self.manifest_path."""
        mp = Path(self.manifest_path)

        if mp.is_dir():
            # allow passing a dataset directory instead of a file
            for cand in ("dataset.yaml", "dataset.yml", "dataset.json"):
                candp = mp / cand
                if candp.exists():
                    mp = candp
                    break
            else:
                raise FileNotFoundError(f"No manifest found in directory: {mp}")

        if not mp.exists():
            raise FileNotFoundError(mp)

        if mp.suffix.lower() in (".yaml", ".yml"):
            if not _YAML_OK:
                raise RuntimeError("pyyaml not installed but manifest is YAML.")
            data = yaml.safe_load(mp.read_text())
            fmt = "yaml"
        elif mp.suffix.lower() == ".json":
            data = json.loads(mp.read_text())
            fmt = "json"
        else:
            # fallback: try yaml then json
            if _YAML_OK:
                try:
                    data = yaml.safe_load(mp.read_text())
                    fmt = "yaml"
                except Exception:
                    data = json.loads(mp.read_text())
                    fmt = "json"
            else:
                data = json.loads(mp.read_text())
                fmt = "json"

        # overwrite instance fields
        self.name = data.get("name", self.name)
        self.version = str(data.get("version", self.version))
        self.format = data.get("format", fmt)
        self.roots = data.get("roots", self.roots)
        self.meta = data.get("meta", self.meta)

        # Continuous dataset fields
        self.dataset_type = data.get("dataset_type", "discrete")
        self.segment_duration = data.get("segment_duration", None)
        self.time_column = data.get("time_column", None)

        if ensure_roots:
            self._ensure_roots()
        return self

    def save(self) -> None:
        """Persist manifest."""
        self._ensure_roots()
        payload = {
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "roots": self.roots,
            "meta": self.meta,
            "dataset_type": self.dataset_type,
        }
        # Only include continuous-specific fields if set
        if self.segment_duration:
            payload["segment_duration"] = self.segment_duration
        if self.time_column:
            payload["time_column"] = self.time_column

        if self.format == "json":
            self.manifest_path.write_text(json.dumps(payload, indent=2))
        else:
            if not _YAML_OK:
                raise RuntimeError(
                    "pyyaml not installed; set format='json' or install pyyaml."
                )
            self.manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    # ---- Helpers ----
    def get_root(self, key: str) -> Path:
        if key not in self.roots or not self.roots[key]:
            raise KeyError(
                f"Root '{key}' is not set in manifest. "
                f"Available roots: {list(self.roots.keys())}"
            )
        p = Path(self.roots[key])
        if not p.is_absolute():
            return (_dataset_base_dir(self) / p).resolve()
        return p

    def set_root(self, key: str, path: str | Path) -> None:
        self.roots[key] = str(Path(path))
        self._ensure_roots()

    def _ensure_roots(self) -> None:
        for p in self.roots.values():
            if p:
                path = Path(p)
                if not path.is_absolute():
                    path = _dataset_base_dir(self) / path
                path.mkdir(parents=True, exist_ok=True)

    def ensure_roots(self) -> None:
        """Public wrapper so callers can trigger directory creation after mutations."""
        self._ensure_roots()

    def remap_roots(self, path_map: Mapping[str, str]) -> None:
        """
        Remap dataset roots by replacing the longest matching path prefixes using path_map.
        path_map entries are {source_prefix: dest_prefix}.
        """
        if not path_map:
            return
        normalized = _normalize_path_map(path_map)
        if not normalized:
            return
        updated: dict[str, str] = {}
        for key, raw_path in self.roots.items():
            if not raw_path:
                continue
            current = Path(raw_path).expanduser()
            new_value = _remap_single_path(current, normalized)
            if new_value is not None:
                updated[key] = str(new_value)
        self.roots.update(updated)
        self._path_map = list(normalized)

    def remap_path(self, path: str | Path) -> Path:
        p = Path(str(path).strip())
        if not self._path_map:
            return p
        new_value = _remap_single_path(p, self._path_map)
        return new_value if new_value is not None else p

    def resolve_path(self, stored_path: str | Path, anchor: Path | None = None) -> Path:
        """Resolve a stored path (absolute or relative) to an absolute path.

        Relative paths are resolved against *anchor* (default: dataset root).
        Absolute paths that exist are returned as-is; absolute paths that don't
        exist are tried through :meth:`remap_path`.
        """
        p = Path(str(stored_path).strip())
        if not p.is_absolute():
            base = anchor if anchor is not None else _dataset_base_dir(self)
            return (base / p).resolve()
        if p.exists():
            return p
        return self.remap_path(p)

    def _relative_to_root(self, abs_path: Path) -> str:
        """Convert an absolute path to relative-to-dataset-root for storage.

        Internal paths (inside dataset tree) become relative strings.
        External paths (outside dataset tree) remain absolute.
        """
        root = _dataset_base_dir(self)
        try:
            return str(abs_path.resolve().relative_to(root))
        except ValueError:
            return str(abs_path.resolve())

    def rewrite_index_paths(
        self, path_map: Mapping[str, str], dry_run: bool = False
    ) -> dict[str, int]:
        """
        Permanently rewrite abs_path in all index CSV files on disk.

        Args:
            path_map: {old_prefix: new_prefix} mapping
            dry_run: If True, report what would change without writing

        Returns:
            Dict of {index_path: num_paths_changed}
        """
        normalized = _normalize_path_map(path_map)
        if not normalized:
            return {}

        def rewrite_index(idx_path: Path) -> int:
            if not idx_path.exists():
                return 0
            df = pd.read_csv(idx_path)
            if "abs_path" not in df.columns:
                return 0
            changed = 0
            new_paths = []
            for p in df["abs_path"]:
                if pd.isna(p):
                    new_paths.append(p)
                    continue
                remapped = _remap_single_path(Path(p), normalized)
                if remapped is not None and str(remapped) != p:
                    new_paths.append(str(remapped))
                    changed += 1
                else:
                    new_paths.append(p)
            if changed > 0 and not dry_run:
                df["abs_path"] = new_paths
                df.to_csv(idx_path, index=False)
            return changed

        results: dict[str, int] = {}

        # All roots that may have index files
        root_keys = ["tracks", "tracks_raw", "labels", "media", "models", "inputsets"]
        for key in root_keys:
            root = self.roots.get(key)
            if not root:
                continue
            idx_path = Path(root) / "index.csv"
            count = rewrite_index(idx_path)
            if count > 0:
                results[str(idx_path)] = count

        # Features: has per-feature subdirectories with their own index.csv
        features_root = self.roots.get("features")
        if features_root:
            features_path = Path(features_root)
            # Root-level index
            root_idx = features_path / "index.csv"
            count = rewrite_index(root_idx)
            if count > 0:
                results[str(root_idx)] = count
            # Per-feature indexes
            for subdir in features_path.iterdir():
                if subdir.is_dir():
                    sub_idx = subdir / "index.csv"
                    count = rewrite_index(sub_idx)
                    if count > 0:
                        results[str(sub_idx)] = count

        # Labels: has per-kind subdirectories (e.g., id_tags) with their own index.csv
        labels_root = self.roots.get("labels")
        if labels_root:
            labels_path = Path(labels_root)
            for subdir in labels_path.iterdir():
                if subdir.is_dir():
                    sub_idx = subdir / "index.csv"
                    count = rewrite_index(sub_idx)
                    if count > 0:
                        results[str(sub_idx)] = count

        # Frames: has per-method subdirectories (uniform, kmeans) with their own index.csv
        frames_root = self.roots.get("frames")
        if frames_root:
            frames_path = Path(frames_root)
            if frames_path.exists():
                for subdir in frames_path.iterdir():
                    if subdir.is_dir():
                        sub_idx = subdir / "index.csv"
                        count = rewrite_index(sub_idx)
                        if count > 0:
                            results[str(sub_idx)] = count

        return results

    def make_portable(self, dry_run: bool = False) -> dict[str, int]:
        """Convert all internal absolute paths to relative (to dataset root).

        Only needed for datasets created before relative-path support.
        Idempotent — safe to call multiple times. Already-relative paths
        are left unchanged.

        Args:
            dry_run: If True, report what would change without writing.

        Returns:
            Dict of ``{file_path: num_paths_changed}``.
        """
        root = _dataset_base_dir(self)
        results: dict[str, int] = {}

        def _make_rel(abs_str: str) -> tuple[str, bool]:
            """Try to make *abs_str* relative to dataset root.
            Returns (new_str, changed)."""
            p = Path(abs_str)
            if not p.is_absolute():
                return abs_str, False  # already relative
            try:
                rel = str(p.resolve().relative_to(root))
                return rel, rel != abs_str
            except ValueError:
                return abs_str, False  # external — keep absolute

        # --- 8a. Roots in dataset.yaml ---
        roots_changed = 0
        new_roots = {}
        for k, v in self.roots.items():
            if not v:
                new_roots[k] = v
                continue
            new_val, changed = _make_rel(v)
            new_roots[k] = new_val
            if changed:
                roots_changed += 1
        if roots_changed > 0:
            if not dry_run:
                self.roots = new_roots
                self.save()
            results["dataset.yaml (roots)"] = roots_changed

        # --- 8b. Index CSVs: convert abs_path column ---
        def _convert_index(
            idx_path: Path, external_columns: set[str] | None = None
        ) -> int:
            """Convert abs_path (and optionally other path columns) to relative."""
            if not idx_path.exists():
                return 0
            df = pd.read_csv(idx_path)
            total_changed = 0
            for col in ["abs_path"]:
                if col not in df.columns:
                    continue
                new_vals = []
                for val in df[col]:
                    if pd.isna(val):
                        new_vals.append(val)
                        continue
                    new_val, changed = _make_rel(str(val))
                    new_vals.append(new_val)
                    if changed:
                        total_changed += 1
                if total_changed > 0 and not dry_run:
                    df[col] = new_vals
            if total_changed > 0 and not dry_run:
                df.to_csv(idx_path, index=False)
            return total_changed

        # Walk all roots that have index files
        for key in ["tracks", "tracks_raw", "media", "models", "inputsets"]:
            r = self.roots.get(key)
            if not r:
                continue
            rp = self.get_root(key)
            idx_path = rp / "index.csv"
            count = _convert_index(idx_path)
            if count > 0:
                results[str(idx_path)] = count

        # Labels: per-kind subdirectories
        labels_root = self.roots.get("labels")
        if labels_root:
            lp = self.get_root("labels")
            idx = lp / "index.csv"
            count = _convert_index(idx)
            if count > 0:
                results[str(idx)] = count
            if lp.exists():
                for subdir in lp.iterdir():
                    if subdir.is_dir():
                        sub_idx = subdir / "index.csv"
                        count = _convert_index(sub_idx)
                        if count > 0:
                            results[str(sub_idx)] = count

        # Features: per-feature subdirectories (possibly with run_id subdirs)
        features_root = self.roots.get("features")
        if features_root:
            fp = self.get_root("features")
            root_idx = fp / "index.csv"
            count = _convert_index(root_idx)
            if count > 0:
                results[str(root_idx)] = count
            if fp.exists():
                for subdir in fp.iterdir():
                    if subdir.is_dir():
                        sub_idx = subdir / "index.csv"
                        count = _convert_index(sub_idx)
                        if count > 0:
                            results[str(sub_idx)] = count

        # Frames: per-method subdirectories
        frames_root = self.roots.get("frames")
        if frames_root:
            frp = self.get_root("frames")
            if frp.exists():
                for subdir in frp.iterdir():
                    if subdir.is_dir():
                        sub_idx = subdir / "index.csv"
                        count = _convert_index(sub_idx)
                        if count > 0:
                            results[str(sub_idx)] = count

        # --- 8c. run_info.json files (frame extraction manifests) ---
        if frames_root:
            frp = self.get_root("frames")
            if frp.exists():
                for ri_path in frp.rglob("run_info.json"):
                    try:
                        data = json.loads(ri_path.read_text())
                    except Exception:
                        continue
                    changed = 0
                    # output_dir -> relative to dataset root
                    if "output_dir" in data:
                        new_val, did_change = _make_rel(data["output_dir"])
                        if did_change:
                            data["output_dir"] = new_val
                            changed += 1
                    # video_path -> relative to dataset root
                    if "video_path" in data:
                        new_val, did_change = _make_rel(data["video_path"])
                        if did_change:
                            data["video_path"] = new_val
                            changed += 1
                    # files[].path -> filename only (they're siblings of run_info.json)
                    for f in data.get("files", []):
                        if "path" in f:
                            p = Path(f["path"])
                            if p.is_absolute():
                                f["path"] = p.name
                                changed += 1
                    if changed > 0:
                        if not dry_run:
                            ri_path.write_text(json.dumps(data, indent=2, default=str))
                        results[str(ri_path)] = changed

        return results

    def list_groups(self) -> list[str]:
        """Return a sorted list of unique group names in tracks/index.csv."""
        idx_path = self.get_root("tracks") / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError("tracks/index.csv not found.")
        df = pd.read_csv(idx_path)
        return sorted(df["group"].fillna("").unique())

    def list_sequences(self, group: str | None = None) -> list[str]:
        """Return all sequences (optionally filtered by group) in tracks/index.csv."""
        idx_path = self.get_root("tracks") / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError("tracks/index.csv not found.")
        df = pd.read_csv(idx_path)
        df["group"] = df["group"].fillna("")
        if group is not None:
            df = df[df["group"] == group]
        return sorted(df["sequence"].fillna("").unique())

    def get_sequence_metadata(
        self,
        level_names: list[str] | None = None,
        separator: str = "__",
    ) -> pd.DataFrame:
        """
        Return a DataFrame with all sequences and optionally parsed hierarchy columns.

        This method provides a way to view the full dataset structure and filter
        by arbitrary hierarchy levels, supporting datasets with different organizational
        structures (2, 3, 4+ levels).

        Parameters
        ----------
        level_names : list[str], optional
            Names for hierarchy levels. If provided, parses the full path
            (group + sequence) into columns with these names.
            E.g., ["fish", "speed", "loop"] for a 3-level hierarchy.
        separator : str, default "__"
            The separator used in compound names.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - group, sequence: Original values from index
            - group_safe, sequence_safe: URL-encoded versions
            - abs_path: Path to the parquet file
            - Additional columns from index (n_rows, etc.)
            - If level_names provided: one column per level name

        Examples
        --------
        >>> # Basic usage - get all sequences
        >>> meta = ds.get_sequence_metadata()
        >>> meta[['group', 'sequence']].head()

        >>> # Parse into hierarchy levels
        >>> meta = ds.get_sequence_metadata(level_names=["fish", "speed", "loop"])
        >>> meta.groupby("speed")["sequence"].count()

        >>> # 4-level hierarchy for continuous recordings
        >>> meta = ds.get_sequence_metadata(
        ...     level_names=["experiment", "arena", "day", "hour"]
        ... )
        """
        from .helpers import parse_hierarchy

        idx_path = self.get_root("tracks") / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError("tracks/index.csv not found.")

        df = pd.read_csv(idx_path)
        df["group"] = df["group"].fillna("")
        df["sequence"] = df["sequence"].fillna("")

        if level_names:
            # Parse each row into hierarchy levels
            parsed_rows = []
            for _, row in df.iterrows():
                parsed = parse_hierarchy(
                    row["group"], row["sequence"], level_names, separator
                )
                parsed_rows.append(parsed)

            # Add parsed columns to DataFrame
            parsed_df = pd.DataFrame(parsed_rows)
            df = pd.concat([df, parsed_df], axis=1)

        return df

    def query_sequences(
        self,
        group_contains: str | None = None,
        group_startswith: str | None = None,
        group_endswith: str | None = None,
        sequence_contains: str | None = None,
        sequence_startswith: str | None = None,
        sequence_endswith: str | None = None,
    ) -> list[tuple[str, str]]:
        """
        Return (group, sequence) pairs matching the specified criteria.

        Provides flexible filtering for hierarchical datasets where group and/or
        sequence names encode multiple factors.

        Parameters
        ----------
        group_contains : str, optional
            Filter groups containing this substring
        group_startswith : str, optional
            Filter groups starting with this prefix
        group_endswith : str, optional
            Filter groups ending with this suffix
        sequence_contains : str, optional
            Filter sequences containing this substring
        sequence_startswith : str, optional
            Filter sequences starting with this prefix
        sequence_endswith : str, optional
            Filter sequences ending with this suffix

        Returns
        -------
        list[tuple[str, str]]
            List of (group, sequence) pairs matching all criteria

        Examples
        --------
        >>> # Get all sequences for fish_01
        >>> pairs = ds.query_sequences(group_startswith="fish_01")

        >>> # Get all speed_3 recordings across all fish
        >>> pairs = ds.query_sequences(sequence_startswith="speed_3")

        >>> # Get all loop_1 recordings at speed_3
        >>> pairs = ds.query_sequences(
        ...     sequence_contains="speed_3",
        ...     sequence_endswith="loop_1"
        ... )
        """
        idx_path = self.get_root("tracks") / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError("tracks/index.csv not found.")

        df = pd.read_csv(idx_path)
        df["group"] = df["group"].fillna("")
        df["sequence"] = df["sequence"].fillna("")

        mask = pd.Series([True] * len(df))

        if group_contains is not None:
            mask &= df["group"].str.contains(group_contains, na=False)
        if group_startswith is not None:
            mask &= df["group"].str.startswith(group_startswith, na=False)
        if group_endswith is not None:
            mask &= df["group"].str.endswith(group_endswith, na=False)
        if sequence_contains is not None:
            mask &= df["sequence"].str.contains(sequence_contains, na=False)
        if sequence_startswith is not None:
            mask &= df["sequence"].str.startswith(sequence_startswith, na=False)
        if sequence_endswith is not None:
            mask &= df["sequence"].str.endswith(sequence_endswith, na=False)

        filtered = df[mask]
        return list(zip(filtered["group"], filtered["sequence"]))

    # ----------------------------
    # Media indexing (no symlinks)
    # ----------------------------
    def index_media(
        self,
        search_dirs: Iterable[str | Path],
        extensions: Tuple[str, ...] = (".mp4", ".avi"),
        index_filename: str = "index.csv",
        recursive: bool = True,
        sequence_match_mode: str = "exact",
    ) -> Path:
        """
        Scan search_dirs for media files with given extensions and write an index CSV into media root.
        - No symlinks created; absolute paths recorded.
        - Columns: name, abs_path, size_bytes, mtime_iso, group, sequence, group_safe, sequence_safe, video_order

        Parameters
        ----------
        search_dirs : Iterable[str | Path]
            Directories to scan for media files.
        extensions : tuple of str
            File extensions to include.
        index_filename : str
            Output CSV filename within media root.
        recursive : bool
            Whether to search subdirectories.
        sequence_match_mode : str
            How to match video filenames to known sequences from tracks/index.csv.
            - "exact" (default): video stem must exactly match a sequence name.
            - "prefix": video stem is matched to the longest sequence name that
              is a prefix of the stem. This handles split recordings where files
              are named like ``session01_001.mp4``, ``session01_002.mp4`` mapping
              to sequence ``session01``.
        """
        if sequence_match_mode not in {"exact", "prefix"}:
            raise ValueError(
                f"sequence_match_mode must be 'exact' or 'prefix', got '{sequence_match_mode}'"
            )

        media_root = self.get_root("media")
        out_csv = media_root / index_filename
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        seq_key_map = self._build_media_sequence_keymap()

        rows = []
        for d in map(Path, search_dirs):
            if not d.exists():
                print(f"[WARN] search dir missing: {d}", file=sys.stderr)
                continue
            it = d.rglob("*") if recursive else d.glob("*")
            for p in it:
                if not p.is_file():
                    continue
                # Skip macOS resource forks (._* files)
                if p.name.startswith("._"):
                    continue
                if p.suffix.lower() in exts:
                    try:
                        st = p.stat()
                        meta = self._match_media_sequence(
                            seq_key_map,
                            p.stem,
                            mode=sequence_match_mode,
                        )
                        probe = _probe_video_metadata(p)
                        # When no track match, use video stem as sequence
                        # so each video is its own sequence (not all lumped
                        # together under an empty key).
                        fallback_seq = p.stem
                        fallback_safe = to_safe_name(p.stem)
                        rows.append(
                            {
                                "name": p.name,
                                "group": meta.get("group", "") if meta else "",
                                "sequence": meta.get("sequence", fallback_seq)
                                if meta
                                else fallback_seq,
                                "group_safe": meta.get("group_safe", "")
                                if meta
                                else "",
                                "sequence_safe": meta.get(
                                    "sequence_safe", fallback_safe
                                )
                                if meta
                                else fallback_safe,
                                "abs_path": str(p.resolve()),
                                "size_bytes": st.st_size,
                                "mtime_iso": _to_iso(st.st_mtime),
                                "width": probe.get("width", ""),
                                "height": probe.get("height", ""),
                                "fps": probe.get("fps", ""),
                                "codec": probe.get("codec", ""),
                            }
                        )
                    except OSError as e:
                        print(f"[WARN] skip {p}: {e}", file=sys.stderr)

        # De-duplicate by absolute path
        seen = set()
        dedup = []
        for r in rows:
            k = r["abs_path"]
            if k in seen:
                continue
            seen.add(k)
            dedup.append(r)

        # Assign video_order within each (group, sequence) by filename sort
        df_out = (
            pd.DataFrame(dedup)
            if dedup
            else pd.DataFrame(
                columns=[
                    "name",
                    "group",
                    "sequence",
                    "group_safe",
                    "sequence_safe",
                    "abs_path",
                    "size_bytes",
                    "mtime_iso",
                    "width",
                    "height",
                    "fps",
                    "codec",
                ]
            )
        )
        df_out["video_order"] = 0
        if not df_out.empty:
            for (g, s), sub in df_out.groupby(["group", "sequence"]):
                if len(sub) > 1:
                    sorted_idx = sub.sort_values("name").index
                    for rank, idx in enumerate(sorted_idx):
                        df_out.loc[idx, "video_order"] = rank

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "name",
            "group",
            "sequence",
            "group_safe",
            "sequence_safe",
            "abs_path",
            "size_bytes",
            "mtime_iso",
            "width",
            "height",
            "fps",
            "codec",
            "video_order",
        ]
        df_out[fieldnames].to_csv(out_csv, index=False)

        multi_count = 0
        if not df_out.empty:
            seq_counts = df_out.groupby(["group", "sequence"]).size()
            multi_count = int((seq_counts > 1).sum())
        print(
            f"[index_media] Wrote {len(df_out)} entries -> {out_csv}"
            + (f" ({multi_count} multi-video sequences)" if multi_count else "")
        )
        return out_csv

    def resolve_media_paths(
        self, group: str, sequence: str, index_filename: str = "index.csv"
    ) -> list[Path]:
        """
        Resolve all media file paths for a given (group, sequence), ordered.

        For multi-video sequences, returns paths sorted by ``video_order``.
        For single-video sequences, returns a list with one element.
        """
        media_root = self.get_root("media")
        idx_path = media_root / index_filename
        if not idx_path.exists():
            raise FileNotFoundError(f"Media index not found: {idx_path}")
        df = pd.read_csv(idx_path)
        if df.empty:
            raise FileNotFoundError("Media index is empty.")

        # Ensure video_order column (backward compat with old indexes)
        if "video_order" not in df.columns:
            df["video_order"] = 0

        def _resolve_matches(df_match):
            if df_match.empty:
                return None
            df_sorted = df_match.sort_values("video_order")
            return [
                self.resolve_path(row["abs_path"]) for _, row in df_sorted.iterrows()
            ]

        # Direct match by (group, sequence)
        if "group" in df.columns and "sequence" in df.columns:
            df_match = df[
                (df["group"].fillna("") == str(group))
                & (df["sequence"].fillna("") == str(sequence))
            ]
            paths = _resolve_matches(df_match)
            if paths:
                return paths

        # Safe-name fallback
        safe_group = to_safe_name(group) if group else ""
        safe_sequence = to_safe_name(sequence)
        if {"group_safe", "sequence_safe"}.issubset(df.columns):
            df_match = df[
                (df["group_safe"].fillna("") == safe_group)
                & (df["sequence_safe"].fillna("") == safe_sequence)
            ]
            paths = _resolve_matches(df_match)
            if paths:
                return paths

        # Fallback: by filename stem
        tail = Path(sequence).name
        stem = tail.lower()
        df["name_lower"] = df["name"].astype(str).str.lower()
        candidates = df[df["name_lower"].str.contains(stem, na=False)]
        if candidates.empty:
            raise FileNotFoundError(
                f"No media file found matching sequence '{sequence}'."
            )
        paths = _resolve_matches(candidates)
        if paths:
            return paths

        raise FileNotFoundError(f"No media file found matching sequence '{sequence}'.")

    def resolve_media_path(
        self, group: str, sequence: str, index_filename: str = "index.csv"
    ) -> Path:
        """
        Resolve a single media file path for a given (group, sequence).

        For multi-video sequences, raises ``RuntimeError`` with a message
        to use :meth:`resolve_media_paths` instead.
        """
        paths = self.resolve_media_paths(group, sequence, index_filename)
        if len(paths) > 1:
            raise RuntimeError(
                f"Sequence '{sequence}' has {len(paths)} video files. "
                f"Use resolve_media_paths() for multi-video sequences."
            )
        return paths[0]

    def _build_media_sequence_keymap(self) -> dict[str, list[dict]]:
        """
        Build a lookup of various sequence keys -> metadata for mapping media files to sequences.
        """
        idx_path = self.get_root("tracks") / "index.csv"
        if not idx_path.exists():
            return {}
        df = pd.read_csv(idx_path)
        keymap: dict[str, list[dict]] = {}
        for _, row in df.iterrows():
            group = str(row.get("group", "") or "")
            sequence = str(row.get("sequence", "") or "")
            if not sequence:
                continue
            group_safe = row.get("group_safe") or (to_safe_name(group) if group else "")
            sequence_safe = row.get("sequence_safe") or to_safe_name(sequence)
            tail = Path(sequence).name
            tail_safe = to_safe_name(tail) if tail else ""
            keys = {
                sequence,
                sequence.lower(),
                sequence_safe,
                sequence_safe.lower(),
                tail,
                tail.lower(),
                tail_safe,
                tail_safe.lower(),
            }
            meta = {
                "group": group,
                "sequence": sequence,
                "group_safe": group_safe,
                "sequence_safe": sequence_safe,
            }
            for key in keys:
                if not key:
                    continue
                keymap.setdefault(key, []).append(meta)
        return keymap

    @staticmethod
    def _match_media_sequence(
        seq_key_map: dict[str, list[dict]],
        stem: str,
        mode: str = "exact",
    ) -> Optional[dict]:
        if not seq_key_map or not stem:
            return None
        candidates = [
            stem,
            stem.lower(),
            to_safe_name(stem),
            to_safe_name(stem).lower(),
        ]

        # Exact match: try each candidate key directly
        for key in candidates:
            hits = seq_key_map.get(key)
            if not hits:
                continue
            # Return the first metadata dict (all hits for the same
            # key originate from the same track entry)
            return hits[0]

        if mode == "prefix":
            # Prefix match: find the longest known key that is a prefix
            # of any candidate form of the stem. Longest wins to avoid
            # ambiguity (e.g. "session01" vs "session01_special").
            stem_lc = stem.lower()
            stem_safe = to_safe_name(stem).lower()
            best_key: Optional[str] = None
            best_len = 0
            for key in seq_key_map:
                key_lc = key.lower()
                if len(key_lc) <= best_len:
                    continue
                if stem_lc.startswith(key_lc) or stem_safe.startswith(key_lc):
                    best_key = key
                    best_len = len(key_lc)
            if best_key is not None:
                return seq_key_map[best_key][0]

        return None

    def index_tracks_raw(
        self,
        search_dirs: Iterable[str | Path],
        patterns: Iterable[str] | str = ("*.npy", "*.h5", "*.csv"),
        src_format: str = "calms21_npy",
        index_filename: str = "index.csv",
        recursive: bool = True,
        multi_sequences_per_file: bool = False,
        group_from: Optional[str] = None,
        group_pattern: Optional[str] = None,
        exclude_patterns: Optional[Iterable[str]] = None,
        compute_md5: bool = False,
    ) -> Path:
        """
        Scan for original tracking files and write tracks_raw/index.csv
        Columns: group, sequence, abs_path, src_format, size_bytes, mtime_iso, md5

        Parameters
        ----------
        search_dirs : Iterable[str | Path]
            Directories to search for files
        patterns : Iterable[str] | str
            Glob patterns to match files
        src_format : str
            Source format identifier (e.g., "trex_npz", "calms21_npy")
        index_filename : str
            Name of output index file
        recursive : bool
            Whether to search recursively
        multi_sequences_per_file : bool
            If True (e.g., CalMS files), set 'group' from group_from and leave 'sequence' blank
        group_from : str | None
            For multi_sequences_per_file: 'filename' or 'parent'
        group_pattern : str | None
            Regex pattern to extract group from sequence name. Must have a capturing group.
            Examples:
                r'^(hex|OCI|OLE)_' -> extracts 'hex', 'OCI', or 'OLE' as group
                r'^([A-Za-z]+)_'   -> extracts letters before first underscore as group
            Applied AFTER sequence is determined (e.g., after stripping _fish0 suffix).
        exclude_patterns : Iterable[str] | None
            Glob patterns to exclude
        compute_md5 : bool
            If True, compute MD5 hash of each file (slow for large files). Default False.
        """
        out_csv = self.get_root("tracks_raw") / index_filename
        rows = []

        pat_list = _normalize_patterns(patterns)
        exc_list = _normalize_patterns(exclude_patterns)
        group_re = re.compile(group_pattern) if group_pattern else None

        for root in map(Path, search_dirs):
            for pat in pat_list:
                it = root.rglob(pat) if recursive else root.glob(pat)
                for p in it:
                    if not p.is_file():
                        continue
                    name = p.name
                    if exc_list and any(fnmatch.fnmatch(name, ex) for ex in exc_list):
                        continue
                    st = p.stat()
                    if multi_sequences_per_file:
                        # put file-level grouping into 'group', leave sequence blank
                        if group_from == "filename":
                            grp = p.stem
                        elif group_from == "parent":
                            grp = p.parent.name
                        else:
                            grp = ""
                        seq = ""
                    else:
                        if src_format == "trex_npz":
                            seq = _strip_trex_seq(p.stem)
                        else:
                            seq = p.stem  # 1 file ~= 1 sequence default

                        # Extract group from sequence using pattern
                        if group_re:
                            m = group_re.search(seq)
                            grp = m.group(1) if m else ""
                        else:
                            grp = ""

                    rows.append(
                        {
                            "group": grp,
                            "sequence": seq,
                            "abs_path": str(p.resolve()),
                            "src_format": src_format,
                            "size_bytes": st.st_size,
                            "mtime_iso": _to_iso(st.st_mtime),
                            "md5": _md5(p) if compute_md5 else "",
                        }
                    )

        df = pd.DataFrame(rows).drop_duplicates(subset=["abs_path"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[index_tracks_raw] {len(df)} -> {out_csv}")
        return out_csv

    # ----------------------------
    # Convert one original -> standard (T-Rex-like)
    # ----------------------------
    def convert_one_track(
        self, raw_row: pd.Series, params: Optional[dict] = None, overwrite: bool = False
    ) -> Path:
        """
        Convert a single raw track file (row from tracks_raw/index.csv) to standard trex_v1 parquet.
        Returns path to standardized file, updates tracks/index.csv.
        """
        params = params or {}
        std_fmt = self.meta.get("tracks", {}).get("standard_format", "trex_v1")
        src_format = str(raw_row["src_format"])
        src_path = self.resolve_path(raw_row["abs_path"])

        if src_format not in TRACK_CONVERTERS:
            raise KeyError(f"No converter registered for src_format='{src_format}'")

        # Where to place standardized file:
        # group/sequence.parquet if group present, else just sequence.parquet
        tracks_root = self.get_root("tracks")

        # If sequence missing/blank and we have an enumerator, expand this file into multiple per-sequence outputs
        raw_seq_val = raw_row.get("sequence", "")
        seq_value = "" if _is_empty_like(raw_seq_val) else str(raw_seq_val).strip()
        if (not seq_value) and (src_format in TRACK_SEQ_ENUM):
            # policy: 'infile' (default), 'filename', 'both'
            policy = str(params.get("group_from", "infile")).lower()
            if policy not in {"infile", "filename", "both"}:
                policy = "infile"

            raw_collection = (
                str(raw_row.get("group", "")) if raw_row is not None else ""
            )
            pairs = TRACK_SEQ_ENUM[src_format](src_path)
            if not pairs:
                raise ValueError(
                    f"No (group, sequence) pairs enumerated for {src_path}"
                )
            produced = []

            for g, s in pairs:
                # canonical (with '/')
                canon_seq = s
                # decide output group by policy
                canon_group_infile = g or ""
                out_group_canon = canon_group_infile
                if policy in {"filename", "both"} and raw_collection:
                    out_group_canon = raw_collection

                # safe names for path
                safe_seq = to_safe_name(canon_seq)
                safe_group = to_safe_name(out_group_canon) if out_group_canon else ""

                # output path
                tracks_root = self.get_root("tracks")
                stem = f"{safe_group + '__' if safe_group else ''}{safe_seq}"
                out_path = tracks_root / f"{stem}.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Respect overwrite flag when outputs already exist
                if out_path.exists() and not overwrite:
                    produced.append(out_path)
                    continue

                # hints to converter (keep canonical in-file keys; converter may preserve them in columns)
                params_with_hints = dict(params)
                params_with_hints["group"] = canon_group_infile
                params_with_hints["sequence"] = canon_seq

                df_std = TRACK_CONVERTERS[src_format](src_path, params_with_hints)

                # Overwrite group column in DataFrame to match policy
                if "group" in df_std.columns and out_group_canon != canon_group_infile:
                    df_std["group"] = out_group_canon

                # Ensure schema, then write
                _, _schema_report = ensure_track_schema(
                    df_std, std_fmt, strict=bool(params.get("strict_schema", False))
                )
                df_std.to_parquet(out_path, index=False)

                # Index row: group follows policy; keep file-level hint as 'collection'
                row = {
                    "group": out_group_canon,
                    "sequence": canon_seq,
                    "group_safe": safe_group,
                    "sequence_safe": safe_seq,
                    "collection": raw_collection,
                    "collection_safe": to_safe_name(raw_collection)
                    if raw_collection
                    else "",
                    "abs_path": self._relative_to_root(out_path),
                    "std_format": std_fmt,
                    "source_abs_path": str(src_path.resolve()),
                    "source_md5": raw_row.get("md5", ""),
                    "n_rows": int(len(df_std)),
                }
                self._write_tracks_index_row(row)
                produced.append(out_path)

            return self.get_root("tracks") / "index.csv"

        # Normal single-sequence path (default)
        safe_group = (
            to_safe_name(str(raw_row.get("group", "")) or "")
            if raw_row.get("group")
            else ""
        )
        safe_seq = to_safe_name(seq_value)
        rel_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
        out_path = tracks_root / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            return out_path

        # pass group/sequence hints to the converter
        params_with_hints = dict(params)
        params_with_hints.setdefault("group", str(raw_row.get("group", "")))
        params_with_hints.setdefault("sequence", str(raw_row.get("sequence", "")))
        df_std = TRACK_CONVERTERS[src_format](src_path, params_with_hints)

        # Validate/coerce against the declared standard format schema (if any)
        strict_schema = bool(params.get("strict_schema", False))
        _, _schema_report = ensure_track_schema(df_std, std_fmt, strict=strict_schema)

        df_std.to_parquet(out_path, index=False)

        # Update tracks/index.csv using the helper
        row = {
            "group": raw_row.get("group", ""),
            "sequence": raw_row["sequence"],
            "group_safe": to_safe_name(str(raw_row.get("group", "")))
            if raw_row.get("group")
            else "",
            "sequence_safe": to_safe_name(seq_value),
            "collection": str(raw_row.get("group", ""))
            if raw_row.get("group") is not None
            else "",
            "collection_safe": to_safe_name(str(raw_row.get("group", "")))
            if raw_row.get("group")
            else "",
            "abs_path": self._relative_to_root(out_path),
            "std_format": std_fmt,
            "source_abs_path": str(src_path.resolve()),
            "source_md5": raw_row.get("md5", ""),
            "n_rows": int(len(df_std)),
        }
        self._write_tracks_index_row(row)
        return out_path

    def _write_tracks_index_row(self, row: dict):
        """
        Helper to write/update a row in tracks/index.csv, removing any existing entry for the same (group, sequence).
        Ensures safe-name columns are present and filled.
        """
        # Ensure safe-name columns are present in row
        row = dict(row)
        row["group_safe"] = to_safe_name(row["group"]) if row.get("group") else ""
        row["sequence_safe"] = (
            to_safe_name(row["sequence"]) if row.get("sequence") else ""
        )
        if "collection_safe" not in row:
            row["collection_safe"] = (
                to_safe_name(row.get("collection", "")) if row.get("collection") else ""
            )
        tracks_root = self.get_root("tracks")
        idx_std = tracks_root / "index.csv"
        columns = [
            "group",
            "sequence",
            "group_safe",
            "sequence_safe",
            "collection",
            "collection_safe",
            "abs_path",
            "std_format",
            "source_abs_path",
            "source_md5",
            "n_rows",
        ]
        if idx_std.exists():
            df_idx = pd.read_csv(idx_std)
            # If missing safe columns, add and fill them
            for col, canon_col in [
                ("group_safe", "group"),
                ("sequence_safe", "sequence"),
            ]:
                if col not in df_idx.columns:
                    df_idx[col] = df_idx[canon_col].apply(
                        lambda v: to_safe_name(v) if pd.notnull(v) and str(v) else ""
                    )
            # Ensure collection/collection_safe columns exist and are filled appropriately
            if "collection" not in df_idx.columns:
                df_idx["collection"] = ""
            if "collection_safe" not in df_idx.columns:
                # derive from collection (which may be empty strings)
                df_idx["collection_safe"] = df_idx["collection"].apply(
                    lambda v: to_safe_name(v) if pd.notnull(v) and str(v) else ""
                )
            # Remove any existing entry with the same (group, sequence)
            df_idx = df_idx[
                ~(
                    (df_idx["group"].fillna("") == row["group"])
                    & (df_idx["sequence"] == row["sequence"])
                )
            ]
            df_idx = pd.concat(
                [df_idx, pd.DataFrame([{k: row.get(k, "") for k in columns}])],
                ignore_index=True,
            )
        else:
            # Ensure all columns present in correct order
            df_idx = pd.DataFrame([[row.get(k, "") for k in columns]], columns=columns)
        df_idx.to_csv(idx_std, index=False)

    def list_converters(self) -> Dict[str, TrackConverter]:
        """Return registered raw->standard track converters."""
        return dict(TRACK_CONVERTERS)

    def list_schemas(self) -> Dict[str, TrackSchema]:
        """Return registered track schemas."""
        return dict(TRACK_SCHEMAS)

    # ----------------------------
    # Bulk convert
    # ----------------------------
    def convert_all_tracks(
        self,
        params: Optional[dict] = None,
        overwrite: bool = False,
        merge_per_sequence: Optional[bool] = None,
        group_from: Optional[str] = None,
    ) -> None:
        """
        Convert all raw track files (from tracks_raw/index.csv) to standard T-Rex-like parquet files.

        By default, for src_format == 'trex_npz', files are merged per (group, sequence) into a single
        parquet file (one per unique (group, sequence)). For other formats, or if merge_per_sequence=False,
        each row is converted individually.

        Parameters
        ----------
        params : dict | None
            Extra parameters to pass to converters.
        overwrite : bool
            If True, overwrite existing output files.
        merge_per_sequence : bool | None
            If True, merge per (group, sequence) for formats that support it (currently trex_npz).
            If None, defaults to True if all rows are trex_npz, else False.
        group_from : {'infile','filename','both'} | None
            Controls which *group* ends up in the standardized output & index:
            - 'infile' (default): use the group from inside the source file (e.g., 'annotator-id_0').
            - 'filename'   : use the raw file-level group hint from tracks_raw/index.csv (e.g., 'calms21_task1_test').
            - 'both'  : set output group to the raw file-level group, and still record in-file group in the data
                        (converters should already keep in-file columns; we always keep raw file-level hint in
                        the 'collection' column).
            If None, defaults to 'infile'.
        """
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "tracks_raw/index.csv not found; run index_tracks_raw first."
            )
        try:
            df = pd.read_csv(raw_idx)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"tracks_raw/index.csv is empty or malformed: {raw_idx}\n"
                "This usually means index_tracks_raw() found no matching files.\n"
                "Check your search_dirs and patterns parameters."
            )

        # Decide merging default for trex
        if merge_per_sequence is None:
            merge_per_sequence = len(df) > 0 and (df["src_format"] == "trex_npz").all()

        # normalize group_from
        group_from = (group_from or "infile").lower()
        if group_from not in {"infile", "filename", "both"}:
            raise ValueError(
                f"group_from must be one of 'infile', 'filename', 'both'; got {group_from}"
            )

        if not merge_per_sequence:
            # Convert each row individually
            for _, row in df.iterrows():
                try:
                    call_params = dict(params) if params else {}
                    call_params["group_from"] = group_from
                    self.convert_one_track(row, params=call_params, overwrite=overwrite)
                except Exception as e:
                    print(f"[WARN] convert failed for {row.get('abs_path')}: {e}")
            return

        # Merge per (group, sequence, src_format)
        groupby_cols = ["group", "sequence", "src_format"]
        df = df.copy()
        for col in groupby_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = (
                df[col]
                .astype("string")
                .fillna("")
                .replace({"nan": "", "None": ""}, regex=False)
                .str.strip()
            )

        for keys, group_df in df.groupby(groupby_cols):
            group, sequence, src_format = keys

            # Non-mergeable formats -> fall back to individual conversion
            if src_format != "trex_npz":
                for _, row in group_df.iterrows():
                    try:
                        call_params = dict(params) if params else {}
                        call_params["group_from"] = group_from
                        self.convert_one_track(
                            row, params=call_params, overwrite=overwrite
                        )
                    except Exception as e:
                        print(f"[WARN] convert failed for {row.get('abs_path')}: {e}")
                continue

            # Merge TRex NPZ per (group, sequence)
            dfs = []
            first_row = group_df.iloc[0]
            for _, row in group_df.iterrows():
                src_path = self.resolve_path(row["abs_path"])
                hints = {
                    "group": group if group else "",
                    "sequence": sequence if sequence else "",
                }
                call_params = dict(params) if params else {}
                call_params.update(hints)
                call_params["group_from"] = group_from
                df_std = TRACK_CONVERTERS[src_format](src_path, call_params)
                dfs.append(df_std)

            # Align columns across IDs
            all_cols = sorted(set().union(*[set(d.columns) for d in dfs]))
            aligned = []
            for d in dfs:
                missing = [c for c in all_cols if c not in d.columns]
                if missing:
                    for mc in missing:
                        d[mc] = np.nan
                aligned.append(d[all_cols])
            merged_df = pd.concat(aligned, ignore_index=True)
            ensure_track_schema(merged_df, "trex_v1", strict=False)

            # Determine output group based on policy
            raw_group_hint = str(first_row.get("group", "")) or ""
            out_group = group  # default: infile (already what we grouped by)
            if group_from in {"filename", "both"} and raw_group_hint:
                out_group = raw_group_hint

            # Write output
            tracks_root = self.get_root("tracks")
            safe_group = to_safe_name(out_group) if out_group else ""
            safe_seq = to_safe_name(sequence)
            rel_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
            out_path = tracks_root / rel_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(out_path, index=False)

            # Index row (also keep raw hint as 'collection')
            row_out = {
                "group": out_group,
                "sequence": sequence,
                "group_safe": safe_group,
                "sequence_safe": safe_seq,
                "collection": raw_group_hint,
                "collection_safe": to_safe_name(raw_group_hint)
                if raw_group_hint
                else "",
                "abs_path": self._relative_to_root(out_path),
                "std_format": "trex_v1",
                "source_abs_path": str(first_row["abs_path"]),
                "source_md5": first_row.get("md5", ""),
                "n_rows": int(len(merged_df)),
            }
            self._write_tracks_index_row(row_out)

    # ----------------------------
    # Labels: conversion + indexing
    # ----------------------------
    def convert_all_labels(
        self,
        kind: str = "behavior",
        overwrite: bool = False,
        params: Optional[dict] = None,
        source_format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Convert labels from raw files using registered label converters.

        This method now uses a plugin architecture via the label_library.
        Converters are automatically registered for different source formats.

        Parameters
        ----------
        kind : str, default="behavior"
            Type of labels to convert (e.g., "behavior", "id_tags")
        overwrite : bool, default=False
            Whether to overwrite existing label files
        params : dict, optional
            Configuration parameters passed to converter
        source_format : str, optional
            Source format identifier (e.g., "calms21_npy", "boris_csv")
            Must match a registered converter's src_format
        **kwargs : additional keyword arguments
            Passed to converter (e.g., group_from, fps, etc.)

        Raises
        ------
        ValueError
            If no converter is registered for (source_format, kind) combination
        FileNotFoundError
            If tracks_raw/index.csv is missing

        Examples
        --------
        Convert CalMS21 labels:
        >>> dataset.convert_all_labels(
        ...     kind="behavior",
        ...     source_format="calms21_npy",
        ...     group_from="filename"
        ... )

        Convert Boris labels (once implemented):
        >>> dataset.convert_all_labels(
        ...     kind="behavior",
        ...     source_format="boris_csv",
        ...     fps=30.0
        ... )
        """
        params = params or {}
        kind = str(kind or "").lower()
        src_format = source_format or params.get("source_format", "calms21_npy")

        # Look up converter in registry
        converter_key = (src_format, kind)
        if converter_key not in LABEL_CONVERTERS:
            available = list(LABEL_CONVERTERS.keys())
            raise ValueError(
                f"No label converter registered for (src_format='{src_format}', kind='{kind}'). "
                f"Available converters: {available}\n"
                f"To add support for a new format, create a converter in label_library/ "
                f"and import it in label_library/__init__.py"
            )

        # Instantiate converter
        converter_cls = LABEL_CONVERTERS[converter_key]
        converter = converter_cls(params=params, **kwargs)

        # Load raw index
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "tracks_raw/index.csv not found; run index_tracks_raw first."
            )

        df_raw = pd.read_csv(raw_idx)
        if "src_format" not in df_raw.columns:
            raise ValueError("tracks_raw/index.csv missing 'src_format' column.")
        df_raw = df_raw[df_raw["src_format"].astype(str) == str(src_format)]
        if df_raw.empty:
            raise ValueError(
                f"No rows in tracks_raw/index.csv with src_format='{src_format}'."
            )

        # Setup output directory
        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)
        idx_path = labels_root / "index.csv"
        _ensure_labels_index(idx_path)

        # Load existing pairs
        existing_pairs: set[tuple[str, str]] = set()
        if idx_path.exists():
            df_idx = pd.read_csv(idx_path)
            if not df_idx.empty:
                grouped = df_idx.get("group", pd.Series(dtype=str)).fillna("")
                seqs = df_idx.get("sequence", pd.Series(dtype=str)).fillna("")
                existing_pairs = set(zip(grouped.astype(str), seqs.astype(str)))

        # Convert each raw file using the converter
        new_rows: list[dict] = []
        for _, raw_row in df_raw.iterrows():
            src_path = self.resolve_path(raw_row["abs_path"])
            created = converter.convert(
                src_path=src_path,
                raw_row=raw_row,
                labels_root=labels_root,
                params=params,
                overwrite=overwrite,
                existing_pairs=existing_pairs,
            )
            if created:
                new_rows.extend(created)

        # Update index and metadata
        if new_rows:
            _append_labels_index(idx_path, new_rows)

            # Update metadata with converter's metadata
            labels_meta = self.meta.setdefault("labels", {})
            labels_meta[kind] = {
                "index": str(idx_path.resolve()),
                "label_format": converter.label_format,
                "updated_at": _now_iso(),
            }

            # Add format-specific metadata if converter provides it
            if hasattr(converter, "get_metadata"):
                labels_meta[kind].update(converter.get_metadata())

            try:
                self.save()
            except Exception:
                pass

        print(
            f"[convert_all_labels] kind={kind} wrote {len(new_rows)} sequences using {src_format} converter (overwrite={overwrite})."
        )

    def convert_labels_custom(
        self,
        converter_fn: Callable,
        kind: str = "behavior",
        label_format: str = "individual_pair_v1",
        overwrite: bool = False,
        **kwargs,
    ) -> int:
        """
        Convert labels using a custom converter function.

        This method provides flexibility for one-off datasets with unique label
        structures that don't fit the standard converter pattern. The Dataset
        handles all index.csv bookkeeping while you provide the conversion logic.

        Parameters
        ----------
        converter_fn : callable
            A function that performs the actual label conversion. Must have signature:

                converter_fn(dataset, labels_root, existing_pairs, overwrite, **kwargs)
                    -> list[dict]

            Where:
            - dataset: This Dataset instance (for accessing paths, metadata, etc.)
            - labels_root: Path to output directory (e.g., dataset/labels/behavior/)
            - existing_pairs: set of (group, sequence) tuples already converted
            - overwrite: bool, whether to overwrite existing files
            - **kwargs: Any additional arguments passed to convert_labels_custom

            Returns:
            - list[dict]: Index rows for each converted sequence. Each dict should have:
                - 'kind': str, label kind (e.g., "behavior")
                - 'label_format': str, format name (e.g., "individual_pair_v1")
                - 'group': str, group name
                - 'sequence': str, sequence name
                - 'group_safe': str, filesystem-safe group name
                - 'sequence_safe': str, filesystem-safe sequence name
                - 'abs_path': str, absolute path to output NPZ file
                - 'n_frames': int, number of unique frames with labels
                - 'n_events': int, total number of label events
                - 'label_ids': str, comma-separated label IDs (e.g., "0,1,2")
                - 'label_names': str, comma-separated label names (e.g., "none,troph,other")
                - (optional) additional metadata columns

        kind : str, default="behavior"
            Type of labels being converted (e.g., "behavior", "id_tags")

        label_format : str, default="individual_pair_v1"
            Format name for metadata. Should match what's saved in NPZ files.

        overwrite : bool, default=False
            Whether to overwrite existing label files

        **kwargs
            Additional arguments passed to converter_fn

        Returns
        -------
        int
            Number of sequences converted

        Examples
        --------
        >>> def my_converter(dataset, labels_root, existing_pairs, overwrite, **kwargs):
        ...     '''Custom converter for my unique dataset.'''
        ...     boris_path = kwargs['boris_path']
        ...     metadata_path = kwargs['metadata_path']
        ...     fps = kwargs.get('fps', 50.0)
        ...
        ...     # ... your conversion logic here ...
        ...     # Save NPZ files to labels_root
        ...     # Return list of index row dicts
        ...
        ...     return index_rows
        >>>
        >>> n_converted = dataset.convert_labels_custom(
        ...     converter_fn=my_converter,
        ...     kind="behavior",
        ...     boris_path=Path("/path/to/boris.tsv"),
        ...     metadata_path=Path("/path/to/metadata.json"),
        ...     fps=50.0,
        ... )

        NPZ File Format (individual_pair_v1)
        ------------------------------------
        The converter should save NPZ files with these keys:
        - 'group': str, group name
        - 'sequence': str, sequence name
        - 'label_format': str, "individual_pair_v1"
        - 'frames': int32 array, shape (n_events,), frame indices
        - 'labels': int32 array, shape (n_events,), label IDs
        - 'individual_ids': int32 array, shape (n_events, 2), [id1, id2] per event
          - For individual behaviors: [subject_id, -1]
          - For pair behaviors: [id1, id2] (symmetric: store both directions)
          - For scene-level: [-1, -1]
        - 'label_ids': int32 array, all label IDs (e.g., [0, 1, 2])
        - 'label_names': object array, label names (e.g., ["none", "troph", "other"])
        - 'fps': float, frames per second
        - (optional) additional metadata

        See Also
        --------
        convert_all_labels : For standard converters registered in label_library
        load_labels : Load converted labels
        """
        kind = str(kind or "behavior").lower()

        # Setup output directory
        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)
        idx_path = labels_root / "index.csv"
        _ensure_labels_index(idx_path)

        # Load existing pairs to avoid duplicates
        existing_pairs: set[tuple[str, str]] = set()
        if idx_path.exists():
            df_idx = pd.read_csv(idx_path)
            if not df_idx.empty:
                grouped = df_idx.get("group", pd.Series(dtype=str)).fillna("")
                seqs = df_idx.get("sequence", pd.Series(dtype=str)).fillna("")
                existing_pairs = set(zip(grouped.astype(str), seqs.astype(str)))

        # Call the custom converter
        new_rows = converter_fn(
            dataset=self,
            labels_root=labels_root,
            existing_pairs=existing_pairs,
            overwrite=overwrite,
            **kwargs,
        )

        # Update index and metadata
        if new_rows:
            _append_labels_index(idx_path, new_rows)

            # Update dataset metadata
            labels_meta = self.meta.setdefault("labels", {})
            labels_meta[kind] = {
                "index": str(idx_path.resolve()),
                "label_format": label_format,
                "updated_at": _now_iso(),
            }

            try:
                self.save()
            except Exception:
                pass

        print(
            f"[convert_labels_custom] kind={kind} wrote {len(new_rows)} sequences (overwrite={overwrite})."
        )
        return len(new_rows)

    def save_id_labels(
        self,
        kind: str,
        group: str,
        sequence: str,
        per_id_labels: dict,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Persist per-(sequence, id) tags under labels/<kind>.

        per_id_labels: {id_value -> {"field": value, ...}}
        """
        if not per_id_labels:
            raise ValueError("per_id_labels must contain at least one entry.")
        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)
        idx_path = labels_root / "index.csv"
        _ensure_labels_index(idx_path)

        safe_group = to_safe_name(group) if group else ""
        safe_seq = to_safe_name(sequence)
        fname = f"{safe_group + '__' if safe_group else ''}{safe_seq}.npz"
        out_path = labels_root / fname
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"ID labels already exist for ({group},{sequence}); set overwrite=True to replace."
            )

        id_keys = sorted(per_id_labels.keys(), key=lambda v: str(v))
        ids_array = np.asarray(id_keys, dtype=object)
        field_names = sorted(
            {field for tags in per_id_labels.values() for field in (tags or {}).keys()}
        )

        payload: dict[str, np.ndarray] = {"ids": ids_array}
        for field in field_names:
            values = []
            for key in id_keys:
                tags = per_id_labels.get(key) or {}
                values.append(tags.get(field))
            payload[field] = np.asarray(values, dtype=object)

        if metadata:
            for meta_key, meta_val in metadata.items():
                payload[f"meta__{meta_key}"] = np.asarray([meta_val], dtype=object)

        np.savez_compressed(out_path, **payload)

        row = {
            "kind": kind,
            "label_format": "id_tags_v1",
            "group": group,
            "sequence": sequence,
            "group_safe": safe_group,
            "sequence_safe": safe_seq,
            "abs_path": self._relative_to_root(out_path),
            "source_abs_path": "",
            "source_md5": "",
            "n_frames": len(id_keys),
            "label_ids": ",".join(map(str, id_keys)),
            "label_names": ",".join(field_names),
        }
        _append_labels_index(idx_path, [row])
        return out_path

    def convert_id_tags_from_csv(
        self,
        csv_path: str | Path,
        csv_type: str = "focal",
        all_ids: Optional[list] = None,
        overwrite: bool = False,
        # Type-specific options:
        focal_id_column: str = "focal_id",
        id_column: str = "id",
        category_column: str = "category",
        field_columns: Optional[list[str]] = None,
    ) -> list[Path]:
        """
        Convert a CSV file to id_tags labels.

        This method supports different CSV formats for per-individual metadata:

        Supported csv_type values
        -------------------------
        "focal"
            One focal ID per sequence. CSV columns: group, sequence, focal_id.
            Creates boolean 'focal' field for all IDs (True for focal, False otherwise).
            Requires `all_ids` parameter to populate non-focal IDs.

        "category"
            Per-ID category labels. CSV columns: group, sequence, id, category.
            Creates 'category' field with the value from CSV.
            IDs not in CSV are skipped (or use all_ids to include them with None).

        "multi"
            Per-ID multiple fields. CSV columns: group, sequence, id, field1, field2...
            Creates one field per column specified in `field_columns`.

        Parameters
        ----------
        csv_path : str or Path
            Path to input CSV file
        csv_type : str
            One of "focal", "category", "multi"
        all_ids : list, optional
            List of all valid IDs. Required for csv_type="focal" to populate non-focal IDs.
            For other types, auto-detected from CSV if not provided.
        overwrite : bool
            Whether to overwrite existing id_tags files
        focal_id_column : str
            Column name for focal ID (csv_type="focal")
        id_column : str
            Column name for individual ID (csv_type="category" or "multi")
        category_column : str
            Column name for category value (csv_type="category")
        field_columns : list[str], optional
            List of column names to use as fields (csv_type="multi")

        Returns
        -------
        list[Path]
            Paths to created npz files

        Examples
        --------
        # Focal labels (one focal fish per sequence)
        >>> dataset.convert_id_tags_from_csv(
        ...     csv_path="focal_ids.csv",
        ...     csv_type="focal",
        ...     all_ids=list(range(8)),
        ...     overwrite=True,
        ... )

        # Category labels (e.g., strain per fish)
        >>> dataset.convert_id_tags_from_csv(
        ...     csv_path="strain_labels.csv",
        ...     csv_type="category",
        ...     category_column="strain",
        ... )

        # Multiple fields per individual
        >>> dataset.convert_id_tags_from_csv(
        ...     csv_path="fish_metadata.csv",
        ...     csv_type="multi",
        ...     field_columns=["strain", "treatment", "sex"],
        ... )
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate required columns
        if "group" not in df.columns or "sequence" not in df.columns:
            raise ValueError("CSV must have 'group' and 'sequence' columns")

        created: list[Path] = []

        if csv_type == "focal":
            # Focal type: one focal ID per sequence, boolean field for all IDs
            if all_ids is None:
                raise ValueError("all_ids is required for csv_type='focal'")
            if focal_id_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{focal_id_column}' column for csv_type='focal'"
                )

            for _, row in df.iterrows():
                group = str(row["group"]) if pd.notna(row["group"]) else ""
                seq = str(row["sequence"])
                focal_id = row[focal_id_column]

                # Convert focal_id to same type as all_ids elements for comparison
                if pd.notna(focal_id):
                    # Try to match type with all_ids
                    if all_ids and isinstance(all_ids[0], int):
                        focal_id = int(focal_id)

                per_id_labels = {
                    id_val: {"focal": (id_val == focal_id)} for id_val in all_ids
                }

                path = self.save_id_labels(
                    kind="id_tags",
                    group=group,
                    sequence=seq,
                    per_id_labels=per_id_labels,
                    overwrite=overwrite,
                )
                created.append(path)

        elif csv_type == "category":
            # Category type: per-ID category value
            if id_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{id_column}' column for csv_type='category'"
                )
            if category_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{category_column}' column for csv_type='category'"
                )

            # Group by (group, sequence)
            for (group, seq), group_df in df.groupby(["group", "sequence"]):
                group = str(group) if pd.notna(group) else ""
                seq = str(seq)

                per_id_labels = {}
                for _, row in group_df.iterrows():
                    id_val = row[id_column]
                    if isinstance(id_val, float) and id_val.is_integer():
                        id_val = int(id_val)
                    cat_val = row[category_column]
                    per_id_labels[id_val] = {category_column: cat_val}

                # Add missing IDs with None if all_ids provided
                if all_ids is not None:
                    for id_val in all_ids:
                        if id_val not in per_id_labels:
                            per_id_labels[id_val] = {category_column: None}

                path = self.save_id_labels(
                    kind="id_tags",
                    group=group,
                    sequence=seq,
                    per_id_labels=per_id_labels,
                    overwrite=overwrite,
                )
                created.append(path)

        elif csv_type == "multi":
            # Multi type: multiple fields per ID
            if id_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{id_column}' column for csv_type='multi'"
                )
            if field_columns is None:
                # Auto-detect: all columns except group, sequence, id
                field_columns = [
                    c for c in df.columns if c not in ["group", "sequence", id_column]
                ]
            if not field_columns:
                raise ValueError("No field columns found for csv_type='multi'")

            # Group by (group, sequence)
            for (group, seq), group_df in df.groupby(["group", "sequence"]):
                group = str(group) if pd.notna(group) else ""
                seq = str(seq)

                per_id_labels = {}
                for _, row in group_df.iterrows():
                    id_val = row[id_column]
                    if isinstance(id_val, float) and id_val.is_integer():
                        id_val = int(id_val)
                    per_id_labels[id_val] = {col: row[col] for col in field_columns}

                # Add missing IDs with None values if all_ids provided
                if all_ids is not None:
                    for id_val in all_ids:
                        if id_val not in per_id_labels:
                            per_id_labels[id_val] = {col: None for col in field_columns}

                path = self.save_id_labels(
                    kind="id_tags",
                    group=group,
                    sequence=seq,
                    per_id_labels=per_id_labels,
                    overwrite=overwrite,
                )
                created.append(path)

        else:
            raise ValueError(
                f"Unknown csv_type: '{csv_type}'. Must be 'focal', 'category', or 'multi'."
            )

        print(f"Created {len(created)} id_tags files from {csv_path.name}")
        return created

    def load_id_labels(
        self,
        kind: str = "id_tags",
        groups: Optional[Iterable[str]] = None,
        sequences: Optional[Iterable[str]] = None,
    ) -> dict[tuple[str, str], dict]:
        """
        Load per-id labels for the requested kind.
        Returns {(group, sequence): {"labels": {id: {field: value}}, "sequence_safe": str, "path": str, "metadata": dict}}
        """
        labels_root = self.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Labels index not found for kind='{kind}': {idx_path}"
            )
        df = pd.read_csv(idx_path)
        if groups is not None:
            groups = {str(g) for g in groups}
            df = df[df["group"].fillna("").astype(str).isin(groups)]
        if sequences is not None:
            sequences = {str(s) for s in sequences}
            df = df[df["sequence"].fillna("").astype(str).isin(sequences)]
        result: dict[tuple[str, str], dict] = {}
        for _, row in df.iterrows():
            group = str(row.get("group", "") or "")
            sequence = str(row.get("sequence", "") or "")
            safe_seq = row.get("sequence_safe") or to_safe_name(sequence)
            abs_path = str(row.get("abs_path", "")).strip()
            if not abs_path:
                continue
            path = self.resolve_path(abs_path)
            if not path.exists():
                continue
            with np.load(path, allow_pickle=True) as npz:
                ids = npz["ids"]
                meta = {}
                field_arrays: dict[str, np.ndarray] = {}
                for key in npz.files:
                    if key == "ids":
                        continue
                    if key.startswith("meta__"):
                        meta[key.split("meta__", 1)[1]] = _coerce_np(npz[key][0])
                        continue
                    field_arrays[key] = npz[key]
                per_id: dict[Any, dict[str, Any]] = {}
                for idx_id, raw_id in enumerate(ids):
                    id_value = _coerce_np(raw_id)
                    tags: dict[str, Any] = {}
                    for field, arr in field_arrays.items():
                        if arr.shape[0] == ids.shape[0]:
                            tags[field] = _coerce_np(arr[idx_id])
                        else:
                            tags[field] = _coerce_np(arr[0])
                    per_id[id_value] = tags
            result[(group, sequence)] = {
                "group": group,
                "sequence": sequence,
                "sequence_safe": safe_seq,
                "path": str(path),
                "labels": per_id,
                "metadata": meta,
            }
        return result

    def load_labels(self, group: str, sequence: str, kind: str = "behavior") -> dict:
        """
        Load behavior labels for a specific (group, sequence).

        Returns dict with keys:
        - frames: np.ndarray of frame indices
        - labels: np.ndarray of behavior IDs
        - individual_ids: np.ndarray of shape (n_events, 2) if individual_pair_v1 format
        - label_ids: np.ndarray of all possible label IDs
        - label_names: np.ndarray of label names
        - label_format: str indicating format version
        - group, sequence, sequence_key: metadata

        For backward compatibility with old dense formats, individual_ids may not be present.
        """
        labels_root = self.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Labels index not found for kind='{kind}': {idx_path}"
            )

        df = pd.read_csv(idx_path)
        df = df[(df["group"].fillna("") == group) & (df["sequence"] == sequence)]

        if len(df) == 0:
            raise ValueError(
                f"No labels found for group='{group}', sequence='{sequence}', kind='{kind}'"
            )

        if len(df) > 1:
            print(
                f"Warning: Multiple label entries found for ({group}, {sequence}). Using first."
            )

        row = df.iloc[0]
        abs_path = str(row.get("abs_path", "")).strip()
        if not abs_path:
            raise ValueError(f"No abs_path in index for ({group}, {sequence})")

        path = self.resolve_path(abs_path)
        if not path.exists():
            raise FileNotFoundError(f"Label file not found: {path}")

        with np.load(path, allow_pickle=True) as npz:
            data = {key: npz[key] for key in npz.files}

        return data

    def get_label_map(self, kind: str = "behavior") -> dict[int, str]:
        """
        Get the label map {id: name} for a label kind.

        Reads from the labels index.csv (first row).
        """
        idx_path = self.get_root("labels") / kind / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Labels index not found for kind='{kind}': {idx_path}"
            )

        df = pd.read_csv(idx_path, nrows=1)
        row = df.iloc[0]

        ids_str = str(row.get("label_ids", "")).strip()
        names_str = str(row.get("label_names", "")).strip()
        if not ids_str or not names_str:
            raise ValueError(f"No label_ids/label_names in index for kind='{kind}'")

        ids = [int(x) for x in ids_str.split(",")]
        names = names_str.split(",")
        return dict(zip(ids, names))

    def get_labels_for_individual(
        self,
        group: str,
        sequence: str,
        individual_id: int,
        kind: str = "behavior",
        frame_range: Optional[tuple[int, int]] = None,
    ) -> dict:
        """
        Get all label events for a specific individual.

        Parameters
        ----------
        group : str
            Group name
        sequence : str
            Sequence name
        individual_id : int
            Individual ID to filter by
        kind : str
            Label kind (default "behavior")
        frame_range : tuple[int, int], optional
            (start_frame, end_frame) to filter events

        Returns
        -------
        dict
            Dictionary with keys:
            - frames: np.ndarray of frame indices
            - labels: np.ndarray of behavior IDs
            - individual_ids: np.ndarray of shape (n_events, 2)
        """
        data = self.load_labels(group, sequence, kind)

        # Check format
        if "individual_ids" not in data:
            # Old format: backward compatibility
            # Return all frames assuming labels apply to this individual
            result = {
                "frames": data["frames"],
                "labels": data["labels"],
                "individual_ids": None,
            }
            if frame_range:
                start, end = frame_range
                mask = (data["frames"] >= start) & (data["frames"] <= end)
                result["frames"] = data["frames"][mask]
                result["labels"] = data["labels"][mask]
            return result

        # New format: filter by individual_id
        ids = data["individual_ids"]
        mask = (ids[:, 0] == individual_id) | (ids[:, 1] == individual_id)

        if frame_range:
            start, end = frame_range
            mask &= (data["frames"] >= start) & (data["frames"] <= end)

        return {
            "frames": data["frames"][mask],
            "labels": data["labels"][mask],
            "individual_ids": ids[mask],
        }

    def get_labels_at_frame(
        self,
        group: str,
        sequence: str,
        frame: int,
        kind: str = "behavior",
        individual_id: Optional[int] = None,
    ) -> dict:
        """
        Get all labels at a specific frame.

        Parameters
        ----------
        group : str
            Group name
        sequence : str
            Sequence name
        frame : int
            Frame index
        kind : str
            Label kind (default "behavior")
        individual_id : int, optional
            Filter by individual ID if provided

        Returns
        -------
        dict
            Dictionary with keys:
            - frames: np.ndarray of frame indices (should all equal frame)
            - labels: np.ndarray of behavior IDs
            - individual_ids: np.ndarray or None
        """
        data = self.load_labels(group, sequence, kind)

        mask = data["frames"] == frame

        if individual_id is not None and "individual_ids" in data:
            ids = data["individual_ids"]
            mask &= (ids[:, 0] == individual_id) | (ids[:, 1] == individual_id)

        result = {
            "frames": data["frames"][mask],
            "labels": data["labels"][mask],
        }

        if "individual_ids" in data:
            result["individual_ids"] = data["individual_ids"][mask]
        else:
            result["individual_ids"] = None

        return result

    # ----------------------------
    # Load tracks (by group/sequence)
    # ----------------------------
    def load_tracks(
        self,
        group: str,
        sequence: str,
        prefer: str = "standard",
        auto_convert: bool = True,
        convert_params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Load T-Rex-like standardized tracks if present; otherwise optionally auto-convert from raw.
        """
        # Try standardized index first
        idx_std = self.get_root("tracks") / "index.csv"
        if idx_std.exists():
            df_idx = pd.read_csv(idx_std)
            hit = df_idx[
                (df_idx["group"].fillna("") == group) & (df_idx["sequence"] == sequence)
            ]
            if len(hit) == 1:
                return pd.read_parquet(self.resolve_path(hit.iloc[0]["abs_path"]))

        if prefer != "standard":
            raise FileNotFoundError(
                f"No non-standard loader implemented for prefer='{prefer}'"
            )

        # Fallback: find in raw index and convert
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "tracks_raw/index.csv not found; run index_tracks_raw first."
            )
        df_raw = pd.read_csv(raw_idx)
        hit = df_raw[
            (df_raw["group"].fillna("") == group) & (df_raw["sequence"] == sequence)
        ]
        if len(hit) == 0:
            raise FileNotFoundError(
                f"No raw track for ({group}, {sequence}) found in tracks_raw/index.csv"
            )
        if not auto_convert:
            raise FileNotFoundError(
                f"Standardized track missing for ({group},{sequence}) and auto_convert=False"
            )

        std_path = self.convert_one_track(hit.iloc[0], params=convert_params or {})
        return pd.read_parquet(std_path)

    # --- Pipeline delegation methods ---

    def run_feature(self, feature, **kwargs):
        from .pipeline.run import run_feature

        return run_feature(self, feature, **kwargs)

    def extract_frames(self, n_frames, method="uniform", **kwargs):
        from .pipeline.frames import extract_frames

        return extract_frames(self, n_frames, method, **kwargs)

    def list_frame_runs(self, method=None):
        from .pipeline.frames import list_frame_runs

        return list_frame_runs(self, method)

    def get_frame_paths(self, method, run_id=None, group=None, sequence=None):
        from .pipeline.frames import get_frame_paths

        return get_frame_paths(self, method, run_id, group, sequence)

    def get_frame_manifests(self, method, run_id=None, group=None, sequence=None):
        from .pipeline.frames import get_frame_manifests

        return get_frame_manifests(self, method, run_id, group, sequence)

    def train_model(self, model, config=None, overwrite=False):
        from .pipeline.models import train_model

        return train_model(self, model, config, overwrite)


# --- Backward compat: track converter helpers moved to core/track_library ---

from mosaic.core.track_library.trex import _strip_trex_seq

def _is_empty_like(x: Optional[Any]) -> bool:
    """True for None/NaN/''/'nan'/'none' (case-insensitive)."""
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, str):
        s = x.strip().lower()
        return s in ("", "nan", "none")
    return False


def _to_iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _ensure_labels_index(idx_path: Path):
    if not idx_path.exists():
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "kind": pd.Series(dtype="string"),
                "label_format": pd.Series(dtype="string"),
                "group": pd.Series(dtype="string"),
                "sequence": pd.Series(dtype="string"),
                "group_safe": pd.Series(dtype="string"),
                "sequence_safe": pd.Series(dtype="string"),
                "abs_path": pd.Series(dtype="string"),
                "source_abs_path": pd.Series(dtype="string"),
                "source_md5": pd.Series(dtype="string"),
                "n_frames": pd.Series(dtype="Int64"),
                "label_ids": pd.Series(dtype="string"),
                "label_names": pd.Series(dtype="string"),
            }
        ).to_csv(idx_path, index=False)


def _append_labels_index(idx_path: Path, rows: list[dict]):
    if not idx_path.exists():
        _ensure_labels_index(idx_path)
    df = pd.read_csv(idx_path)
    for col in LABEL_INDEX_COLUMNS:
        fill = "" if col != "n_frames" else None
        df = ensure_text_column(df, col, "" if fill is None else fill)
    updated = df.copy()
    for r in rows:
        row = dict(r)
        row.setdefault("kind", "")
        row.setdefault("label_format", "")
        row.setdefault("group", "")
        row.setdefault("sequence", "")
        if "group_safe" not in row:
            row["group_safe"] = to_safe_name(row["group"]) if row["group"] else ""
        if "sequence_safe" not in row:
            row["sequence_safe"] = (
                to_safe_name(row["sequence"]) if row["sequence"] else ""
            )
        row.setdefault("abs_path", "")
        row.setdefault("source_abs_path", "")
        row.setdefault("source_md5", "")
        if "n_frames" not in row:
            row["n_frames"] = ""
        row.setdefault("label_ids", "")
        row.setdefault("label_names", "")
        mask = (updated["group"].fillna("") == row["group"]) & (
            updated["sequence"].fillna("") == row["sequence"]
        )
        updated = updated[~mask]
        updated = pd.concat([updated, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(idx_path, index=False)
