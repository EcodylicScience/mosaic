# dataset.py
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pandas as pd 
import numpy as np
import math  # used by _norm_hint
import uuid, datetime
import yaml  # pip install pyyaml
import importlib
import multiprocessing as mp

import csv, json, os, sys, re, subprocess, shlex, gc, textwrap
from urllib.parse import urlparse, parse_qs, urlencode
import fnmatch
from .helpers import to_safe_name, from_safe_name

from typing import Protocol, Iterable, Optional, Sequence
from dataclasses import dataclass
import json, joblib, hashlib, time


def _probe_video_metadata(path: Path) -> dict[str, Any]:
    """
    Use ffprobe to collect width/height/fps/codec metadata.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,codec_name",
        "-of", "json",
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


def _remote_only_file(cfg: dict) -> Path:
    local_root = Path(cfg["local_root"]).expanduser().resolve()
    return local_root / ".remote_only_patterns.json"


def _load_remote_only_patterns(cfg: dict) -> list[str]:
    try:
        path = _remote_only_file(cfg)
    except Exception:
        return []
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(data, list):
        return [str(p) for p in data if isinstance(p, str)]
    return []


def _save_remote_only_patterns(cfg: dict, patterns: Sequence[str]) -> None:
    try:
        path = _remote_only_file(cfg)
        path.write_text(json.dumps(list(dict.fromkeys(patterns)), indent=2))
    except Exception:
        pass


def _record_remote_only_pattern(cfg: dict, pattern: str) -> None:
    if not pattern:
        return
    pats = cfg.setdefault("remote_only_patterns", [])
    if pattern not in pats:
        pats.append(pattern)
    _save_remote_only_patterns(cfg, pats)


def _normalize_path_map(path_map: Mapping[str, str]) -> list[tuple[Path, Path]]:
    normalized: list[tuple[Path, Path]] = []
    for src, dst in path_map.items():
        if not src or not dst:
            continue
        normalized.append((Path(src).expanduser(), Path(dst).expanduser()))
    normalized = [pair for pair in normalized if pair[0] != pair[1]]
    normalized.sort(key=lambda pair: len(pair[0].as_posix()), reverse=True)
    return normalized


def _remap_single_path(path: Path, mapping: Sequence[tuple[Path, Path]]) -> Optional[Path]:
    for src, dst in mapping:
        try:
            rel = path.relative_to(src)
            return dst / rel
        except ValueError:
            continue
    return None


from dataclasses import dataclass, field
from typing import Iterable, Dict, Any, Tuple, Callable, Optional, Mapping, Sequence
import hashlib


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

# ----------- Track schema system -----------
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Iterable

@dataclass(frozen=True)
class TrackSchema:
    name: str
    required: Set[str]                    # exact column names that MUST exist
    required_prefixes: Set[str] = None    # any column that starts with these prefixes (at least one match each)
    recommended: Set[str] = None          # warn-only
    description: str = ""

TRACK_SCHEMAS: Dict[str, TrackSchema] = {}

def register_track_schema(schema: TrackSchema):
    TRACK_SCHEMAS[schema.name] = schema

def ensure_track_schema(df: pd.DataFrame, schema_name: str, strict: bool = False) -> tuple[pd.DataFrame, Dict[str, Iterable[str]]]:
    """
    Validate that df satisfies schema. Returns (df, report_dict).
    report_dict contains keys: missing_required, missing_prefixes, missing_recommended.
    If strict=True and required are missing, raises ValueError.
    """
    if schema_name not in TRACK_SCHEMAS:
        # no schema registered -> nothing to validate
        return df, {}

    sch = TRACK_SCHEMAS[schema_name]
    missing_required = sorted([c for c in (sch.required or set()) if c not in df.columns])
    missing_prefixes = []
    if sch.required_prefixes:
        for pref in sch.required_prefixes:
            if not any(col.startswith(pref) for col in df.columns):
                missing_prefixes.append(pref)
    missing_recommended = sorted([c for c in (sch.recommended or set()) if c not in df.columns])

    report = {
        "missing_required": missing_required,
        "missing_prefixes": missing_prefixes,
        "missing_recommended": missing_recommended,
    }
    if strict and (missing_required or missing_prefixes):
        raise ValueError(f"Schema '{schema_name}' validation failed: {report}")
    if missing_required or missing_prefixes or missing_recommended:
        print(f"[schema:{schema_name}] Validation report -> {report}")
    return df, report

# Default T-Rex-like schema (flexible): must have these core columns; poseX/poseY are prefix-validated
register_track_schema(TrackSchema(
    name="trex_v1",
    required={
        "frame", "time", "id", "group", "sequence",
    },
    required_prefixes={"poseX", "poseY"},
    recommended={
        "X#wcentroid", "Y#wcentroid", "SPEED", "ANGLE",
    },
    description="Minimal T-Rex-like per-frame, per-id tracks with centroid/pose columns."
))

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

def _md5(path: Path, chunk=1<<20) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
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


def save_inputset(ds, name: str, inputs: list[dict], description: Optional[str] = None,
                  overwrite: bool = False) -> Path:
    """
    Persist an inputset JSON under <dataset_root>/inputsets/<name>.json.
    """
    path = _inputset_path(ds, name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Inputset '{name}' already exists: {path}")
    payload = {
        "name": name,
        "description": description or "",
        "inputs": inputs or [],
    }
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
    }
    return inputs, meta


def _resolve_inputs(ds, explicit_inputs: Optional[list[dict]], inputset_name: Optional[str],
                    explicit_override: bool = False) -> tuple[list[dict], dict]:
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
        raise ValueError("No inputs resolved; provide params['inputs'] or params['inputset'].")
    return inputs, meta


############# DATASET

default_roots = {
    "media": "media",
    "features": "features",     # calculated features (input to models), e.g. wavelets, projections, embeddings
    "labels": "labels",         # GT annotations, .npy/.csv
    "models": "models",   # trained models, reports, plots
    "tracks": "tracks",
    "tracks_raw": "tracks_raw"
}

def new_dataset_manifest(
    name: str,
    base_dir: str | Path,
    roots: dict[str, str | Path] = default_roots,
    version: str = "0.1.0",
    index_format: str = "group/sequence",
    outfile: str | Path | None = None,
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
    # Normalize roots -> absolute paths rooted at base_dir
    norm_roots = {k: str((base_dir / Path(v)).resolve()) for k, v in roots.items()}
    # Make sure these directories exist
    for p in norm_roots.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "version": version,
        "uuid": str(uuid.uuid4()),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "index_format": index_format,   # recommended: "group/sequence"
        "roots": norm_roots,            # required minimal roots you actually use now
        # You can append optional fields later without placeholders
    }

    header_comment = """# ==========================================================
# DATASET MANIFEST (extensible YAML)
# Minimal required fields above; append optional fields below
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

    text = header_comment + yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)

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
    roots: Dict[str, str] = field(default_factory=lambda: {
        "media": "",
        "tracks_raw": "",
        "tracks": "",
        "tracks_raw": "",
        "features": "",
        "labels": "",
        "models": "",
    })
    meta: Dict[str, Any] = field(default_factory=dict)
    _path_map: list[tuple[Path, Path]] = field(default_factory=list, init=False, repr=False)

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
                    data = yaml.safe_load(mp.read_text()); fmt = "yaml"
                except Exception:
                    data = json.loads(mp.read_text()); fmt = "json"
            else:
                data = json.loads(mp.read_text()); fmt = "json"

        # overwrite instance fields
        self.name = data.get("name", self.name)
        self.version = str(data.get("version", self.version))
        self.format = data.get("format", fmt)
        self.roots = data.get("roots", self.roots)
        self.meta = data.get("meta", self.meta)
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
        }
        if self.format == "json":
            self.manifest_path.write_text(json.dumps(payload, indent=2))
        else:
            if not _YAML_OK:
                raise RuntimeError("pyyaml not installed; set format='json' or install pyyaml.")
            self.manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    # ---- Helpers ----
    def get_root(self, key: str) -> Path:
        if key not in self.roots or not self.roots[key]:
            print(key)
            print(self.roots)
            raise KeyError(f"Root '{key}' is not set in manifest.")
        return Path(self.roots[key])

    def set_root(self, key: str, path: str | Path) -> None:
        self.roots[key] = str(Path(path))
        self._ensure_roots()

    def _ensure_roots(self) -> None:
        for p in self.roots.values():
            if p:
                Path(p).mkdir(parents=True, exist_ok=True)

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

    # ----------------------------
    # Media indexing (no symlinks)
    # ----------------------------
    def index_media(self,
                    search_dirs: Iterable[str | Path],
                    extensions: Tuple[str, ...] = (".mp4", ".avi"),
                    index_filename: str = "index.csv",
                    recursive: bool = True) -> Path:
        """
        Scan search_dirs for media files with given extensions and write an index CSV into media root.
        - No symlinks created; absolute paths recorded.
        - Columns: name, abs_path, size_bytes, mtime_iso, group, sequence, group_safe, sequence_safe (when resolvable)
        """
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
                if p.suffix.lower() in exts:
                    try:
                        st = p.stat()
                        meta = self._match_media_sequence(seq_key_map, p.stem)
                        probe = _probe_video_metadata(p)
                        rows.append({
                            "name": p.name,
                            "group": meta.get("group", "") if meta else "",
                            "sequence": meta.get("sequence", "") if meta else "",
                            "group_safe": meta.get("group_safe", "") if meta else "",
                            "sequence_safe": meta.get("sequence_safe", "") if meta else "",
                            "abs_path": str(p.resolve()),
                            "size_bytes": st.st_size,
                            "mtime_iso": _to_iso(st.st_mtime),
                            "width": probe.get("width", ""),
                            "height": probe.get("height", ""),
                            "fps": probe.get("fps", ""),
                            "codec": probe.get("codec", ""),
                        })
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

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["name", "group", "sequence", "group_safe", "sequence_safe",
                      "abs_path", "size_bytes", "mtime_iso", "width", "height", "fps", "codec"]
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(dedup)

        print(f"[index_media] Wrote {len(dedup)} entries -> {out_csv}")
        return out_csv

    def resolve_media_path(self,
                           group: str,
                           sequence: str,
                           index_filename: str = "index.csv") -> Path:
        """
        Resolve the media file path for a given (group, sequence).
        """
        media_root = self.get_root("media")
        idx_path = media_root / index_filename
        if not idx_path.exists():
            raise FileNotFoundError(f"Media index not found: {idx_path}")
        df = pd.read_csv(idx_path)
        if df.empty:
            raise FileNotFoundError("Media index is empty.")

        def _match(df_subset):
            if df_subset.empty:
                return None
            if len(df_subset) > 1:
                return None
            row = df_subset.iloc[0]
            path = Path(row["abs_path"])
            return self.remap_path(path) if hasattr(self, "remap_path") else path

        # direct match
        if "group" in df.columns and "sequence" in df.columns:
            df_match = df[
                (df["group"].fillna("") == str(group)) &
                (df["sequence"].fillna("") == str(sequence))
            ]
            path = _match(df_match)
            if path:
                return path

        # safe-name match
        safe_group = to_safe_name(group) if group else ""
        safe_sequence = to_safe_name(sequence)
        if {"group_safe", "sequence_safe"}.issubset(df.columns):
            df_match = df[
                (df["group_safe"].fillna("") == safe_group) &
                (df["sequence_safe"].fillna("") == safe_sequence)
            ]
            path = _match(df_match)
            if path:
                return path

        # fallback: by filename stem
        tail = Path(sequence).name
        stem = tail.lower()
        df["name_lower"] = df["name"].astype(str).str.lower()
        candidates = df[df["name_lower"].str.contains(stem, na=False)]
        if candidates.empty:
            raise FileNotFoundError(f"No media file found matching sequence '{sequence}'.")
        unique_paths = candidates["abs_path"].unique()
        if len(unique_paths) > 1:
            raise RuntimeError(f"Multiple media files match sequence '{sequence}'; disambiguate manually.")
        path = Path(unique_paths[0])
        return self.remap_path(path) if hasattr(self, "remap_path") else path

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
    def _match_media_sequence(seq_key_map: dict[str, list[dict]], stem: str) -> Optional[dict]:
        if not seq_key_map or not stem:
            return None
        candidates = [
            stem,
            stem.lower(),
            to_safe_name(stem),
            to_safe_name(stem).lower(),
        ]
        for key in candidates:
            hits = seq_key_map.get(key)
            if not hits:
                continue
            if len(hits) == 1:
                return hits[0]
        return None
    
    def index_tracks_raw(self,
                         search_dirs: Iterable[str | Path],
                         patterns: Iterable[str] | str = ("*.npy", "*.h5", "*.csv"),
                         src_format: str = "calms21_npy",
                         index_filename: str = "index.csv",
                         recursive: bool = True,
                         multi_sequences_per_file: bool = False,
                         group_from: Optional[str] = None,
                         exclude_patterns: Optional[Iterable[str]] = None) -> Path:
        """
        Scan for original tracking files and write tracks_raw/index.csv
        Columns: group, sequence, abs_path, src_format, size_bytes, mtime_iso, md5
        If multi_sequences_per_file=True (e.g., CalMS files), set 'group' from group_from ('filename' or 'parent')
        and leave 'sequence' blank; conversion will expand to per-sequence later.
        """
        out_csv = self.get_root("tracks_raw") / index_filename
        rows = []

        pat_list = _normalize_patterns(patterns)
        exc_list = _normalize_patterns(exclude_patterns)

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
                        grp = ""
                        if src_format == "trex_npz":
                            seq = _strip_trex_seq(p.stem)
                        else:
                            seq = p.stem  # 1 file ~= 1 sequence default

                    rows.append({
                        "group": grp,
                        "sequence": seq,
                        "abs_path": str(p.resolve()),
                        "src_format": src_format,
                        "size_bytes": st.st_size,
                        "mtime_iso": _to_iso(st.st_mtime),
                        "md5": _md5(p),
                    })

        df = pd.DataFrame(rows).drop_duplicates(subset=["abs_path"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[index_tracks_raw] {len(df)} -> {out_csv}")
        return out_csv

    # ----------------------------
    # Convert one original -> standard (T-Rex-like)
    # ----------------------------
    def convert_one_track(self,
                          raw_row: pd.Series,
                          params: Optional[dict] = None,
                          overwrite: bool = False) -> Path:
        """
        Convert a single raw track file (row from tracks_raw/index.csv) to standard trex_v1 parquet.
        Returns path to standardized file, updates tracks/index.csv.
        """
        params = params or {}
        std_fmt = self.meta.get("tracks", {}).get("standard_format", "trex_v1")
        src_format = str(raw_row["src_format"])
        src_path = Path(raw_row["abs_path"])

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

            raw_collection = str(raw_row.get("group", "")) if raw_row is not None else ""
            pairs = TRACK_SEQ_ENUM[src_format](src_path)
            if not pairs:
                raise ValueError(f"No (group, sequence) pairs enumerated for {src_path}")
            produced = []

            for g, s in pairs:
                # canonical (with '/')
                canon_seq = s
                # decide output group by policy
                canon_group_infile = (g or "")
                out_group_canon = canon_group_infile
                if policy in {"filename", "both"} and raw_collection:
                    out_group_canon = raw_collection

                # safe names for path
                safe_seq  = to_safe_name(canon_seq)
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

                # Ensure schema, then write
                _, _schema_report = ensure_track_schema(df_std, std_fmt, strict=bool(params.get("strict_schema", False)))
                df_std.to_parquet(out_path, index=False)

                # Index row: group follows policy; keep file-level hint as 'collection'
                row = {
                    "group": out_group_canon,
                    "sequence": canon_seq,
                    "group_safe": safe_group,
                    "sequence_safe": safe_seq,
                    "collection": raw_collection,
                    "collection_safe": to_safe_name(raw_collection) if raw_collection else "",
                    "abs_path": str(out_path.resolve()),
                    "std_format": std_fmt,
                    "source_abs_path": str(src_path.resolve()),
                    "source_md5": raw_row.get("md5", ""),
                    "n_rows": int(len(df_std)),
                }
                self._write_tracks_index_row(row)
                produced.append(out_path)

            return self.get_root("tracks") / "index.csv"

        # Normal single-sequence path (default)
        safe_group = to_safe_name(str(raw_row.get('group', '')) or '') if raw_row.get('group') else ''
        safe_seq   = to_safe_name(seq_value)
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
            "group_safe": to_safe_name(str(raw_row.get("group", ""))) if raw_row.get("group") else "",
            "sequence_safe": to_safe_name(seq_value),
            "collection": str(raw_row.get("group", "")) if raw_row.get("group") is not None else "",
            "collection_safe": to_safe_name(str(raw_row.get("group", ""))) if raw_row.get("group") else "",
            "abs_path": str(out_path.resolve()),
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
        row["sequence_safe"] = to_safe_name(row["sequence"]) if row.get("sequence") else ""
        if "collection_safe" not in row:
            row["collection_safe"] = to_safe_name(row.get("collection", "")) if row.get("collection") else ""
        tracks_root = self.get_root("tracks")
        idx_std = tracks_root / "index.csv"
        columns = [
            "group", "sequence", "group_safe", "sequence_safe",
            "collection", "collection_safe",
            "abs_path", "std_format", "source_abs_path", "source_md5", "n_rows"
        ]
        if idx_std.exists():
            df_idx = pd.read_csv(idx_std)
            # If missing safe columns, add and fill them
            for col, canon_col in [("group_safe", "group"), ("sequence_safe", "sequence")]:
                if col not in df_idx.columns:
                    df_idx[col] = df_idx[canon_col].apply(lambda v: to_safe_name(v) if pd.notnull(v) and str(v) else "")
            # Ensure collection/collection_safe columns exist and are filled appropriately
            if "collection" not in df_idx.columns:
                df_idx["collection"] = ""
            if "collection_safe" not in df_idx.columns:
                # derive from collection (which may be empty strings)
                df_idx["collection_safe"] = df_idx["collection"].apply(lambda v: to_safe_name(v) if pd.notnull(v) and str(v) else "")
            # Remove any existing entry with the same (group, sequence)
            df_idx = df_idx[~((df_idx["group"].fillna("") == row["group"]) & (df_idx["sequence"] == row["sequence"]))]
            df_idx = pd.concat([df_idx, pd.DataFrame([{k: row.get(k, "") for k in columns}])], ignore_index=True)
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
    def convert_all_tracks(self,
                        params: Optional[dict] = None,
                        overwrite: bool = False,
                        merge_per_sequence: Optional[bool] = None,
                        group_from: Optional[str] = None) -> None:
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
            raise FileNotFoundError("tracks_raw/index.csv not found; run index_tracks_raw first.")
        df = pd.read_csv(raw_idx)

        # Decide merging default for trex
        if merge_per_sequence is None:
            merge_per_sequence = (len(df) > 0 and (df["src_format"] == "trex_npz").all())

        # normalize group_from
        group_from = (group_from or "infile").lower()
        if group_from not in {"infile", "filename", "both"}:
            raise ValueError(f"group_from must be one of 'infile', 'filename', 'both'; got {group_from}")

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
                        self.convert_one_track(row, params=call_params, overwrite=overwrite)
                    except Exception as e:
                        print(f"[WARN] convert failed for {row.get('abs_path')}: {e}")
                continue

            # Merge TRex NPZ per (group, sequence)
            dfs = []
            first_row = group_df.iloc[0]
            for _, row in group_df.iterrows():
                src_path = Path(row["abs_path"])
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
            safe_seq   = to_safe_name(sequence)
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
                "collection_safe": to_safe_name(raw_group_hint) if raw_group_hint else "",
                "abs_path": str(out_path.resolve()),
                "std_format": "trex_v1",
                "source_abs_path": str(first_row["abs_path"]),
                "source_md5": first_row.get("md5", ""),
                "n_rows": int(len(merged_df)),
            }
            self._write_tracks_index_row(row_out)

    # ----------------------------
    # Labels: conversion + indexing
    # ----------------------------
    def convert_all_labels(self,
                           kind: str = "behavior",
                           overwrite: bool = False,
                           params: Optional[dict] = None,
                           source_format: Optional[str] = None,
                           group_from: Optional[str] = None) -> None:
        """
        Convert behavioral labels from raw CalMS21 files into per-sequence npz bundles under labels/<kind>.

        Parameters
        ----------
        group_from : {'filename','infile','both'}
            Determines which group name is used for the output artifacts/index.
            - 'filename' (default): use the tracks_raw/index.csv group column (e.g., calms21_task1_train).
            - 'infile': keep the original nested group from inside the CalMS21 payload (e.g., annotator-id_0).
            - 'both': write outputs under the filename group but record the infile group in the npz metadata.
        """
        params = params or {}
        kind = str(kind or "").lower()
        if kind != "behavior":
            raise ValueError(f"Unsupported label kind '{kind}'. Only 'behavior' implemented.")

        src_format = source_format or params.get("source_format") or "calms21_npy"
        group_from = group_from or params.get("group_from") or "filename"
        group_from = str(group_from).lower()
        if group_from not in {"infile", "filename", "both"}:
            raise ValueError("group_from must be 'infile', 'filename', or 'both'")
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError("tracks_raw/index.csv not found; run index_tracks_raw first.")
        df_raw = pd.read_csv(raw_idx)
        if "src_format" not in df_raw.columns:
            raise ValueError("tracks_raw/index.csv missing 'src_format' column.")
        df_raw = df_raw[df_raw["src_format"].astype(str) == str(src_format)]
        if df_raw.empty:
            raise ValueError(f"No rows in tracks_raw/index.csv with src_format='{src_format}'.")

        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            _ensure_labels_index(idx_path)

        existing_pairs: set[tuple[str, str]] = set()
        if idx_path.exists():
            df_idx = pd.read_csv(idx_path)
            if not df_idx.empty:
                grouped = df_idx.get("group", pd.Series(dtype=str)).fillna("")
                seqs = df_idx.get("sequence", pd.Series(dtype=str)).fillna("")
                existing_pairs = set(zip(grouped.astype(str), seqs.astype(str)))

        new_rows: list[dict] = []
        total_sequences = 0
        for _, raw_row in df_raw.iterrows():
            created = self._convert_calms21_behavior_labels(
                raw_row,
                labels_root,
                overwrite=overwrite,
                existing_pairs=existing_pairs,
                group_from=group_from,
            )
            if created:
                new_rows.extend(created)
                total_sequences += len(created)

        if new_rows:
            _append_labels_index(idx_path, new_rows)
            labels_meta = self.meta.setdefault("labels", {})
            labels_meta["behavior"] = {
                "index": str(idx_path.resolve()),
                "label_format": "calms21_behavior_v1",
                "label_ids": list(BEHAVIOR_LABEL_MAP.keys()),
                "label_names": list(BEHAVIOR_LABEL_MAP.values()),
                "updated_at": _now_iso(),
            }
            try:
                self.save()
            except Exception:
                pass
        print(f"[convert_all_labels] kind={kind} wrote {len(new_rows)} sequences (overwrite={overwrite}).")

    def _convert_calms21_behavior_labels(self,
                                         raw_row: pd.Series,
                                         labels_root: Path,
                                         overwrite: bool,
                                         existing_pairs: set[tuple[str, str]],
                                         group_from: str = "filename") -> list[dict]:
        """
        Convert one CalMS21 npy/json row into per-sequence behavior label npz files.
        """
        src_path = Path(raw_row["abs_path"])
        nested = load_calms21(src_path)
        rows_out: list[dict] = []
        label_ids = np.array(list(BEHAVIOR_LABEL_MAP.keys()), dtype=int)
        label_names = np.array(list(BEHAVIOR_LABEL_MAP.values()), dtype=object)
        raw_group_hint = str(raw_row.get("group", "") or "")

        for group_name, seqs in nested.items():
            group_val_infile = str(group_name or "")
            if group_from == "filename" and raw_group_hint:
                group_val = raw_group_hint
            elif group_from == "both" and raw_group_hint:
                group_val = raw_group_hint
            else:
                group_val = group_val_infile
            for seq_key, seq_dict in seqs.items():
                if "annotations" not in seq_dict:
                    continue
                labels = np.asarray(seq_dict["annotations"])
                if labels.ndim > 1:
                    labels = labels[:, 0]
                labels = labels.astype(int, copy=False)
                frames = np.arange(labels.shape[0], dtype=np.int32)
                seq_val = str(seq_key)
                pair = (group_val, seq_val)
                safe_group = to_safe_name(group_val) if group_val else ""
                safe_seq = to_safe_name(seq_val)
                fname = f"{safe_group + '__' if safe_group else ''}{safe_seq}.npz"
                out_path = labels_root / fname

                if not overwrite and pair in existing_pairs and out_path.exists():
                    continue

                payload = {
                    "group": group_val,
                    "sequence": seq_val,
                    "sequence_key": seq_val,
                    "frames": frames,
                    "labels": labels,
                    "label_ids": label_ids,
                    "label_names": label_names,
                }
                if group_from in {"filename", "both"} and group_val_infile and group_val_infile != group_val:
                    payload["source_group"] = group_val_infile
                np.savez_compressed(out_path, **payload)
                existing_pairs.add(pair)
                rows_out.append({
                    "kind": "behavior",
                    "label_format": "calms21_behavior_v1",
                    "group": group_val,
                    "sequence": seq_val,
                    "group_safe": safe_group,
                    "sequence_safe": safe_seq,
                    "abs_path": str(out_path.resolve()),
                    "source_abs_path": str(src_path.resolve()),
                    "source_md5": raw_row.get("md5", ""),
                    "n_frames": int(labels.shape[0]),
                    "label_ids": ",".join(map(str, BEHAVIOR_LABEL_MAP.keys())),
                    "label_names": ",".join(BEHAVIOR_LABEL_MAP.values()),
                })
        return rows_out

    def save_id_labels(self,
                       kind: str,
                       group: str,
                       sequence: str,
                       per_id_labels: dict,
                       metadata: Optional[dict] = None,
                       overwrite: bool = False) -> Path:
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
            raise FileExistsError(f"ID labels already exist for ({group},{sequence}); set overwrite=True to replace.")

        id_keys = sorted(per_id_labels.keys(), key=lambda v: str(v))
        ids_array = np.asarray(id_keys, dtype=object)
        field_names = sorted({field for tags in per_id_labels.values() for field in (tags or {}).keys()})

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
            "abs_path": str(out_path.resolve()),
            "source_abs_path": "",
            "source_md5": "",
            "n_frames": len(id_keys),
            "label_ids": ",".join(map(str, id_keys)),
            "label_names": ",".join(field_names),
        }
        _append_labels_index(idx_path, [row])
        return out_path

    def load_id_labels(self,
                       kind: str = "id_tags",
                       groups: Optional[Iterable[str]] = None,
                       sequences: Optional[Iterable[str]] = None) -> dict[tuple[str, str], dict]:
        """
        Load per-id labels for the requested kind.
        Returns {(group, sequence): {"labels": {id: {field: value}}, "sequence_safe": str, "path": str, "metadata": dict}}
        """
        labels_root = self.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"Labels index not found for kind='{kind}': {idx_path}")
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
            path = self.remap_path(abs_path) if hasattr(self, "remap_path") else Path(abs_path)
            if not Path(path).exists():
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

    # ----------------------------
    # Load tracks (by group/sequence)
    # ----------------------------
    def load_tracks(self,
                    group: str,
                    sequence: str,
                    prefer: str = "standard",
                    auto_convert: bool = True,
                    convert_params: Optional[dict] = None) -> pd.DataFrame:
        """
        Load T-Rex-like standardized tracks if present; otherwise optionally auto-convert from raw.
        """
        # Try standardized index first
        idx_std = self.get_root("tracks") / "index.csv"
        if idx_std.exists():
            df_idx = pd.read_csv(idx_std)
            hit = df_idx[(df_idx["group"].fillna("") == group) & (df_idx["sequence"] == sequence)]
            if len(hit) == 1:
                return pd.read_parquet(Path(hit.iloc[0]["abs_path"]))

        if prefer != "standard":
            raise FileNotFoundError(f"No non-standard loader implemented for prefer='{prefer}'")

        # Fallback: find in raw index and convert
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError("tracks_raw/index.csv not found; run index_tracks_raw first.")
        df_raw = pd.read_csv(raw_idx)
        hit = df_raw[(df_raw["group"].fillna("") == group) & (df_raw["sequence"] == sequence)]
        if len(hit) == 0:
            raise FileNotFoundError(f"No raw track for ({group}, {sequence}) found in tracks_raw/index.csv")
        if not auto_convert:
            raise FileNotFoundError(f"Standardized track missing for ({group},{sequence}) and auto_convert=False")

        std_path = self.convert_one_track(hit.iloc[0], params=convert_params or {})
        return pd.read_parquet(std_path)

# ===================== CalMS21 -> T-Rex-like converter =====================

def load_calms21(path: Path | str):
    """
    Load a single CalMS21 file: either .npy (dict) or the original .json.
    Returns a nested dict: group -> seq_id -> dict(...)
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p, allow_pickle=True).item()
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported CalMS21 path (expect .npy or .json): {p}")

def angle_from_two_points(neck_xy: np.ndarray, tail_xy: np.ndarray) -> np.ndarray:
    """
    heading from tail -> neck, angle w.r.t +x (radians)
    neck_xy, tail_xy: (T,2)
    """
    v = neck_xy - tail_xy
    return np.arctan2(v[:, 1], v[:, 0])

def angle_from_pca(XY: np.ndarray) -> np.ndarray:
    """
    PCA-based heading (fallback). XY: (T, L, 2) landmarks for one animal.
    Uses first principal component per frame; sign is arbitrary.
    """
    T = XY.shape[0]
    ang = np.zeros(T, dtype=float)
    for t in range(T):
        pts = XY[t]  # (L,2)
        mu = pts.mean(axis=0)
        c = pts - mu
        cov = c.T @ c
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]  # 2-vector
        ang[t] = np.arctan2(v[1], v[0])
    return ang


############################################################################################################
#### CALMS 21 ####
############################################################################################################

def _calms21_seq_to_trex_df(one_seq_dict: dict,
                            groupname: str,
                            seq_id: str,
                            neck_idx: Optional[int] = None,
                            tail_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Convert a single sequence dict to T-Rex-like long DataFrame (rows = frames x animals).
    """
    # Pick features: either 'features' present or 'keypoints'
    use_features = ("features" in one_seq_dict)
    if use_features:
        # not used in output columns; could be stored elsewhere if needed
        _ = np.asarray(one_seq_dict["features"])  # (T, K)
    keypoints = np.asarray(one_seq_dict["keypoints"])    # (T, 2, 2, L)
    scores    = np.asarray(one_seq_dict.get("scores", None))       # (T, 2, L) or None
    ann       = np.asarray(one_seq_dict["annotations"]) if "annotations" in one_seq_dict else None
    meta      = one_seq_dict.get("metadata", {})
    fps       = float(meta.get("fps", meta.get("frame_rate", 30.0)))

    T = keypoints.shape[0]
    n_anim = keypoints.shape[1]
    n_lm   = keypoints.shape[3]

    rows = []
    for a in range(n_anim):
        # Extract XY for this animal: (T, L, 2)
        X = keypoints[:, a, 0, :]  # (T, L)
        Y = keypoints[:, a, 1, :]  # (T, L)
        XY = np.stack([X, Y], axis=-1)  # (T, L, 2)

        # Centroid over landmarks
        cx = X.mean(axis=1)  # (T,)
        cy = Y.mean(axis=1)

        # Vel/acc (finite diff)
        VX = np.gradient(cx) * fps
        VY = np.gradient(cy) * fps
        SPEED = np.hypot(VX, VY)
        AX = np.gradient(VX) * fps
        AY = np.gradient(VY) * fps

        # Heading angle
        if (neck_idx is not None) and (tail_idx is not None) and 0 <= neck_idx < n_lm and 0 <= tail_idx < n_lm:
            neck = XY[:, neck_idx, :]  # (T,2)
            tail = XY[:, tail_idx, :]
            ANGLE = angle_from_two_points(neck, tail)
        else:
            ANGLE = angle_from_pca(XY)

        # Build a per-frame DataFrame
        data = {
            "frame": np.arange(T, dtype=int),
            "time":  np.arange(T, dtype=float) / fps,
            "id":    np.full(T, a, dtype=int),
            "X#wcentroid": cx,
            "Y#wcentroid": cy,
            "VX": VX, "VY": VY,
            "SPEED": SPEED, "AX": AX, "AY": AY,
            "ANGLE": ANGLE,
            "group": np.full(T, groupname),
            "sequence": np.full(T, seq_id),
        }

        # Pose columns
        for k in range(n_lm):
            data[f"poseX{k}"] = X[:, k]
            data[f"poseY{k}"] = Y[:, k]

        # Optional: label per frame if present (flatten if multi-dim)
        if ann is not None:
            lbl = ann
            if lbl.ndim > 1:
                lbl = lbl[:, 0]
            data["label"] = lbl.astype(int, copy=False)

        # Optional: keypoint scores columns, if provided
        if scores is not None:
            S = np.asarray(scores)  # (T, 2, L)
            S_a = S[:, a, :]        # (T, L)
            for k in range(n_lm):
                data[f"poseP{k}"] = S_a[:, k]

        rows.append(pd.DataFrame(data))

    out = pd.concat(rows, ignore_index=True)
    # Add placeholders often present in T-Rex schema
    out["missing"] = False
    out["visual_identification_p"] = 1.0
    out["timestamp"] = out["time"]
    for col in ["X","Y","SPEED#pcentroid","SPEED#wcentroid","midline_x","midline_y",
                "midline_length","midline_segment_length","normalized_midline",
                "ANGULAR_V#centroid","ANGULAR_A#centroid","BORDER_DISTANCE#pcentroid",
                "MIDLINE_OFFSET","num_pixels","detection_p"]:
        if col not in out.columns:
            out[col] = np.nan
    return out

def calms21_to_trex_df(path: Path | str,
                       prefer_group: Optional[str] = None,
                       prefer_sequence: Optional[str] = None,
                       neck_idx: Optional[int] = None,
                       tail_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Load a CalMS21 .npy/.json and return a concatenated T-Rex-like DataFrame.
    Optionally filter to a specific (group, sequence).
    """
    nested = load_calms21(path)

    groups_present = set(nested.keys())
    seq_filter = None
    direct_group_match_only = True
    if prefer_group and prefer_group not in groups_present:
        # interpret dataset-level hint (e.g., calms21_task1_test)
        seq_filter = _calms21_make_seq_filter_from_hint(prefer_group)
        if seq_filter is not None:
            direct_group_match_only = False

    rows = []
    for groupname, group in nested.items():
        for seq_id, seq in group.items():
            # strict sequence filter (exact match) if requested
            if prefer_sequence and seq_id != prefer_sequence:
                continue
            # group filter: either exact top-level match, or sequence-path filter if hint provided
            if direct_group_match_only:
                if prefer_group and groupname != prefer_group:
                    continue
            else:
                if seq_filter and not seq_filter(groupname, seq_id):
                    continue

            # ensure arrays where needed
            seq = {k: (np.array(v) if isinstance(v, list) else v) for k, v in seq.items()}
            rows.append(_calms21_seq_to_trex_df(seq, groupname, seq_id,
                                                neck_idx=neck_idx, tail_idx=tail_idx))
    if not rows:
        if prefer_group or prefer_sequence:
            raise KeyError(f"Requested CalMS21 ({prefer_group}, {prefer_sequence}) not found in {path}")
        raise RuntimeError(f"No sequences found in CalMS21 file: {path}")
    return pd.concat(rows, ignore_index=True)


def _norm_hint(x: Optional[Any]) -> Optional[str]:
    # Treat None, NaN, "", "nan", "none" as no-hint
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none"):
            return None
        return s
    # Pandas NA/NaT, etc.
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return str(x)

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

def _calms21_converter(path: Path, params: dict) -> pd.DataFrame:
    prefer_group   = _norm_hint(params.get("group"))
    prefer_sequence= _norm_hint(params.get("sequence"))
    neck_idx = params.get("neck_idx", None)
    tail_idx = params.get("tail_idx", None)
    debug = bool(params.get("debug", False))

    # quick inspect
    nested = load_calms21(path)
    if debug:
        pairs = [(g, s) for g, grp in nested.items() for s in grp.keys()]
        print(f"[calms21] in-file pairs ({len(pairs)}): {pairs[:10]}{' ...' if len(pairs)>10 else ''}")
        print(f"[calms21] prefer_group={prefer_group} prefer_sequence={prefer_sequence}")

    # if explicit selection given, return only that
    if prefer_group or prefer_sequence:
        return calms21_to_trex_df(
            path,
            prefer_group=prefer_group,
            prefer_sequence=prefer_sequence,
            neck_idx=neck_idx,
            tail_idx=tail_idx,
        )

    # else single-pair inference
    pairs = [(g, s) for g, grp in nested.items() for s in grp.keys()]
    if len(pairs) == 1:
        g, s = pairs[0]
        return calms21_to_trex_df(
            path,
            prefer_group=g,
            prefer_sequence=s,
            neck_idx=neck_idx,
            tail_idx=tail_idx,
        )
    raise ValueError(
        f"Ambiguous CalMS21 file {path}; contains multiple sequences {pairs}. "
        f"Pass params with group/sequence to disambiguate."
    )

def _calms21_make_seq_filter_from_hint(hint: Optional[str]):
    """
    Return a predicate f(groupname, seq_id)->bool for dataset-level hints like
    'calms21_task1_train', 'calms21_task1_test', 'calms21_task2_train/test',
    'calms21_task3_train/test'. If not applicable, return None.
    """
    if not hint:
        return None
    h = hint.strip().lower()

    def pred_task_split(task_prefix: str, split: str):
        def _pred(_g, _s):
            # matches path patterns like taskX/.../<split>/...
            return _s.startswith(task_prefix) and (f"/{split}/" in _s)
        return _pred

    # task1
    if h.startswith("calms21_task1_"):
        split = "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        if split:
            return pred_task_split("task1/", split)

    # task2 (note: has an annotator level 'task2/annotator1/<split>/...')
    if h.startswith("calms21_task2_"):
        split = "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        if split:
            def _pred(_g, _s):
                return _s.startswith("task2/") and (f"/{split}/" in _s)
            return _pred

    # task3 (behavior level: 'task3/<behavior>/<split>/...')
    if h.startswith("calms21_task3_"):
        split = "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        if split:
            return pred_task_split("task3/", split)

    return None

# Register the converter for both .npy and .json sources (same structure)
register_track_converter("calms21_npy", _calms21_converter)
register_track_converter("calms21_json", _calms21_converter)

# Sequence enumerator for CalMS21 files
def _enumerate_calms21_sequences(path: Path) -> list[tuple[str, str]]:
    nested = load_calms21(path)
    pairs: list[tuple[str, str]] = []
    for g, grp in nested.items():
        for s in grp.keys():
            pairs.append((str(g), str(s)))
    return pairs

register_track_seq_enumerator("calms21_npy", _enumerate_calms21_sequences)
register_track_seq_enumerator("calms21_json", _enumerate_calms21_sequences)

def _to_iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


############################################################################################################
#### TRex ####
############################################################################################################

# --- TRex per-id NPZ support ---
_TREX_ID_SUFFIX = re.compile(r"_id(\d+)$")

def _strip_trex_seq(stem: str) -> str:
    """Return filename stem with trailing '_id<digits>' removed, if present."""
    m = _TREX_ID_SUFFIX.search(stem)
    if m:
        return stem[: m.start()]
    return stem


def _load_npz_to_df(filepath: Path) -> pd.DataFrame:
    """
    Flatten a TRex-like NPZ (per-id) into a DataFrame, robust to arrays with
    slightly different lengths in the same file. We pick the most common length
    across arrays as the target 'n', and truncate longer arrays to fit.
    """
    data = np.load(filepath, allow_pickle=True)
    keys = list(data.files)

    skip_keys = {} # including all for now.  Could make this a parameter to pass

    # Determine candidate lengths per key
    lens = []
    for k in keys:
        if k in skip_keys:
            continue
        v = data[k]
        if getattr(v, "ndim", 0) > 0:
            lens.append(int(v.shape[0]))
    if not lens:
        raise ValueError(f"No array-like keys with length found in NPZ: {filepath}")

    # Prefer 'time' length if present and 1D; else use the mode length
    if "time" in data.files and getattr(data["time"], "ndim", 0) == 1:
        n = int(data["time"].shape[0])
    else:
        # mode (most common) length among arrays
        vals, counts = np.unique(np.array(lens), return_counts=True)
        n = int(vals[np.argmax(counts)])

    cols: dict[str, np.ndarray] = {}

    for k in sorted(keys):
        if k in skip_keys:
            continue
        v = data[k]

        if np.ndim(v) == 0:
            # scalar -> broadcast
            cols[k] = np.repeat(v.item(), n)
            continue

        # 1D or ND: align first dimension to n by truncation (or pad if you prefer)
        if v.shape[0] < n:
            # If you want to pad instead of skip, do it here; for now, we take the shorter n
            # but to keep a consistent table width we’ll just truncate n down for this column.
            vi = v  # keep as-is; we will align by slicing [:vi.shape[0]] and then pad
            # simple pad with NaN to length n
            pad = np.full((n - vi.shape[0],), np.nan, dtype=float) if v.ndim == 1 else \
                  np.full((n - vi.shape[0],) + v.shape[1:], np.nan, dtype=float)
            vi = np.concatenate([vi, pad], axis=0)
        else:
            vi = v[:n]

        if vi.ndim == 1:
            cols[k] = vi
        else:
            flat = vi.reshape(n, -1)
            for i in range(flat.shape[1]):
                cols[f"{k}_{i}"] = flat[:, i]

    df = pd.DataFrame(cols)

    # id
    if "id" in data.files and np.ndim(data["id"]) > 0:
        try:
            id_val = int(np.array(data["id"]).ravel()[0])
        except Exception:
            # fallback for weird dtypes
            id_val = int(np.array(data["id"]).ravel()[0].astype(np.int64))
    else:
        # derive from filename suffix _id<digits>
        m = _TREX_ID_SUFFIX.search(filepath.stem)
        id_val = int(m.group(1)) if m else 0
    df["id"] = id_val

    # frame/time
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df), dtype=int)
    if "time" not in df.columns:
        if "frame_rate" in data.files:
            fr = float(np.array(data["frame_rate"]).ravel()[0])
            fr = fr if fr > 0 else 1.0
            df["time"] = df["frame"] / fr
        else:
            df["time"] = df["frame"].astype(float)

    return df


def _trex_npz_converter(path: Path, params: dict) -> pd.DataFrame:
    """
    Convert a per-id TRex NPZ into our standard T-Rex-like DataFrame.
    Ensures 'group' and 'sequence' columns; derives sequence from stem by
    stripping '_id\\d+' unless explicitly provided in params.
    """
    df = _load_npz_to_df(path)

    group = params.get("group") or ""
    sequence = params.get("sequence") or _strip_trex_seq(path.stem)

    df["group"] = group
    df["sequence"] = sequence

    # Validate (non-strict) against trex_v1 schema
    ensure_track_schema(df, "trex_v1", strict=False)
    return df

# register converter
register_track_converter("trex_npz", _trex_npz_converter)


# ========================================================================================================= #
# ===================== Feature framework =====================
# ========================================================================================================= #
class Feature(Protocol):
    """Interface for a feature/calculation applied over tracks."""
    name: str
    version: str
    params: dict

    # Fit/transform contract
    def needs_fit(self) -> bool: ...
    def supports_partial_fit(self) -> bool: ...
    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: ...
    def partial_fit(self, X: pd.DataFrame) -> None: ...
    def finalize_fit(self) -> None: ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    # Persistence of model state (if any)
    def save_model(self, path: Path) -> None: ...
    def load_model(self, path: Path) -> None: ...

# Simple registry
FEATURES: dict[str, type[Feature]] = {}

def register_feature(cls: type[Feature]):
    FEATURES[cls.__name__] = cls
    return cls

def _hash_params(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:10]

# ----------------------------
# Dataset helpers for tracks
# ----------------------------
def _yield_sequences(ds: "Dataset",
                     groups: Optional[Iterable[str]] = None,
                     sequences: Optional[Iterable[str]] = None,
                     allowed_pairs: Optional[set[tuple[str, str]]] = None):
    """
    Yield (group, sequence, df) for standardized tracks present in tracks/index.csv,
    filtered by groups and/or sequences if provided.
    """
    idx_path = ds.get_root("tracks") / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError("tracks/index.csv not found; run conversion first.")
    df_idx = pd.read_csv(idx_path)
    df_idx["group"] = df_idx["group"].fillna("")
    df_idx["sequence"] = df_idx["sequence"].fillna("")

    mask = pd.Series(True, index=df_idx.index)
    if groups is not None:
        groups = {g for g in groups}
        mask &= df_idx["group"].isin(groups)
    if sequences is not None:
        sequences = {s for s in sequences}
        mask &= df_idx["sequence"].isin(sequences)
    if allowed_pairs is not None:
        allowed_groups = {g for g, _ in allowed_pairs}
        mask &= df_idx["group"].isin(allowed_groups)

    for _, row in df_idx[mask].iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        if allowed_pairs is not None and (g, s) not in allowed_pairs:
            continue
        p = ds.remap_path(row["abs_path"]) if hasattr(ds, "remap_path") else Path(row["abs_path"])
        if not p.exists():
            print(f"[feature] missing parquet for ({g},{s}) -> {p}", file=sys.stderr)
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[feature] failed to read {p}: {e}", file=sys.stderr)
            continue
        yield g, s, df

# ----------------------------
# Yield feature outputs as frames (helper)
# ----------------------------
def _yield_feature_frames(ds: "Dataset",
                          feature_name: str,
                          run_id: Optional[str] = None,
                          groups: Optional[Iterable[str]] = None,
                          sequences: Optional[Iterable[str]] = None,
                          allowed_pairs: Optional[set[tuple[str, str]]] = None):
    """
    Yield (group, sequence, df) from a prior feature's saved outputs.
    If run_id is None, pick the most recent finished run_id for that feature (by finished_at).
    """
    idx_path = _feature_index_path(ds, feature_name)
    if not idx_path.exists():
        raise FileNotFoundError(f"No index for feature '{feature_name}'. Expected: {idx_path}")
    df_idx = pd.read_csv(idx_path)

    # If no run_id specified, choose the latest finished run
    if run_id is None:
        if "finished_at" in df_idx.columns:
            # choose the most recent non-empty finished_at; fall back to started_at
            cand = df_idx[df_idx["finished_at"].fillna("").astype(str) != ""]
            base = cand if len(cand) else df_idx
            # both timestamps are iso strings; sort descending
            base = base.sort_values(
                by=["finished_at" if len(cand) else "started_at"],
                ascending=False,
                kind="stable",
            )
            if base.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(base.iloc[0]["run_id"])
        else:
            base = df_idx.sort_values(by=["started_at"], ascending=False, kind="stable")
            if base.empty:
                raise ValueError(f"No runs found for feature '{feature_name}'.")
            run_id = str(base.iloc[0]["run_id"])

    df_sel = df_idx[df_idx["run_id"] == run_id].copy()
    if df_sel.empty:
        raise ValueError(f"No entries for feature '{feature_name}' with run_id '{run_id}'.")

    df_sel["group"] = df_sel["group"].fillna("")
    df_sel["sequence"] = df_sel["sequence"].fillna("")

    if groups is not None:
        groups = set(groups)
        df_sel = df_sel[df_sel["group"].isin(groups)]
    if sequences is not None:
        sequences = set(sequences)
        df_sel = df_sel[df_sel["sequence"].isin(sequences)]
    if allowed_pairs is not None:
        mask_pairs = []
        for _, row in df_sel.iterrows():
            pair = (str(row["group"]), str(row["sequence"]))
            mask_pairs.append(pair in allowed_pairs)
        if len(mask_pairs):
            df_sel = df_sel[pd.Series(mask_pairs, index=df_sel.index)]

    for _, row in df_sel.iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        p = ds.remap_path(row["abs_path"]) if hasattr(ds, "remap_path") else Path(row["abs_path"])
        if not p.exists():
            print(f"[feature-input] missing parquet for ({g},{s}) -> {p}", file=sys.stderr)
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[feature-input] failed to read {p}: {e}", file=sys.stderr)
            continue
        # Skip marker tiny tables (<= 1 row or < 2 numeric cols)
        if len(df) <= 1:
            continue
        if df.select_dtypes(include=[np.number]).shape[1] < 2:
            continue
        yield g, s, df

# ----------------------------
# Feature runner on Dataset
# ----------------------------
def _feature_run_root(ds: "Dataset", feature_name: str, run_id: str) -> Path:
    return ds.get_root("features") / feature_name / run_id

def _feature_index_path(ds: "Dataset", feature_name: str) -> Path:
    return ds.get_root("features") / feature_name / "index.csv"


def _resolve_inputset_scope(ds: "Dataset",
                            inputset_name: str,
                            groups: Optional[Iterable[str]] = None,
                            sequences: Optional[Iterable[str]] = None) -> dict:
    inputs, meta = _load_inputset(ds, inputset_name)
    if not inputs:
        raise ValueError(f"Inputset '{inputset_name}' has no inputs defined.")

    groups_set = set(groups) if groups is not None else None
    seq_set = set(sequences) if sequences is not None else None

    per_feature_pairs: list[set[tuple[str, str]]] = []
    pair_safe_map: dict[tuple[str, str], str] = {}
    resolved_inputs: list[dict] = []

    for spec in inputs:
        kind = str(spec.get("kind", "feature")).lower()
        if kind == "tracks":
            idx_path = ds.get_root("tracks") / "index.csv"
            if not idx_path.exists():
                raise FileNotFoundError("tracks/index.csv not found; run conversion first.")
            df_idx = pd.read_csv(idx_path)
            df_idx["group"] = df_idx["group"].fillna("")
            df_idx["sequence"] = df_idx["sequence"].fillna("")

            df_scope = df_idx
            if groups_set:
                df_scope = df_scope[df_scope["group"].isin(groups_set)]
            if seq_set:
                df_scope = df_scope[df_scope["sequence"].isin(seq_set)]

            pairs = set(zip(df_scope["group"], df_scope["sequence"]))
            if not pairs:
                print(f"[inputset:{inputset_name}] WARN: tracks spec has no data matching the requested scope.", file=sys.stderr)
            per_feature_pairs.append(pairs)
            resolved_inputs.append({"kind": "tracks", "columns": spec.get("columns")})

            for _, row in df_scope.iterrows():
                pair = (row["group"], row["sequence"])
                seq_safe = to_safe_name(pair[1])
                pair_safe_map.setdefault(pair, seq_safe)
            continue

        feat_name = spec.get("feature")
        if not feat_name:
            continue
        run_id = spec.get("run_id")
        if run_id is None:
            try:
                run_id, _ = _latest_feature_run_root(ds, feat_name)
            except Exception as exc:
                raise RuntimeError(f"Unable to resolve latest run for feature '{feat_name}': {exc}") from exc
        else:
            run_root = _feature_run_root(ds, feat_name, run_id)
            if not run_root.exists():
                raise FileNotFoundError(f"Feature '{feat_name}' run '{run_id}' not found at {run_root}")

        idx_path = _feature_index_path(ds, feat_name)
        if not idx_path.exists():
            raise FileNotFoundError(f"Feature '{feat_name}' has no index at {idx_path}")
        df = pd.read_csv(idx_path)
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            print(f"[inputset:{inputset_name}] WARN: feature '{feat_name}' run '{run_id}' has no indexed rows.", file=sys.stderr)
            per_feature_pairs.append(set())
            resolved_inputs.append({"feature": feat_name, "run_id": run_id, "kind": "feature"})
            continue
        df["group"] = df["group"].fillna("").astype(str)
        df["sequence"] = df["sequence"].fillna("").astype(str)
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].apply(lambda v: to_safe_name(v) if v else "")

        # drop marker/global rows
        df = df[df["sequence"].str.strip() != ""]
        df = df[df["sequence"] != "__global__"]

        avail_groups = set(df["group"])
        avail_sequences = set(df["sequence"])

        if groups_set:
            missing_groups = sorted(groups_set - avail_groups)
            if missing_groups:
                print(f"[inputset:{inputset_name}] WARN: feature '{feat_name}' run '{run_id}' missing requested groups {missing_groups}; continuing with overlap.", file=sys.stderr)
        if seq_set:
            missing_seq = sorted(seq_set - avail_sequences)
            if missing_seq:
                print(f"[inputset:{inputset_name}] WARN: feature '{feat_name}' run '{run_id}' missing requested sequences {missing_seq}; continuing with overlap.", file=sys.stderr)

        df_scope = df
        if groups_set:
            df_scope = df_scope[df_scope["group"].isin(groups_set)]
        if seq_set:
            df_scope = df_scope[df_scope["sequence"].isin(seq_set)]

        pairs = set(zip(df_scope["group"], df_scope["sequence"]))
        if not pairs:
            print(f"[inputset:{inputset_name}] WARN: feature '{feat_name}' run '{run_id}' has no data matching the requested scope.", file=sys.stderr)
        per_feature_pairs.append(pairs)
        resolved_inputs.append({"feature": feat_name, "run_id": run_id, "kind": "feature"})

        for _, row in df_scope.iterrows():
            pair = (row["group"], row["sequence"])
            seq_safe = row.get("sequence_safe")
            if not isinstance(seq_safe, str) or not seq_safe:
                seq_safe = to_safe_name(pair[1])
            pair_safe_map.setdefault(pair, seq_safe)

    if not per_feature_pairs:
        raise ValueError(f"Inputset '{inputset_name}' resolved no usable feature runs.")

    allowed_pairs = set.intersection(*per_feature_pairs) if per_feature_pairs else set()
    if not allowed_pairs:
        raise ValueError(f"Inputset '{inputset_name}' has no overlapping sequences for the requested scope.")

    safe_sequences = {pair_safe_map.get(pair, to_safe_name(pair[1])) for pair in allowed_pairs}

    return {
        "inputset": inputset_name,
        "meta": meta,
        "groups": sorted(groups_set) if groups_set else None,
        "sequences": sorted(seq_set) if seq_set else None,
        "pairs": allowed_pairs,
        "pair_safe_map": {pair: pair_safe_map.get(pair, to_safe_name(pair[1])) for pair in allowed_pairs},
        "safe_sequences": safe_sequences,
        "resolved_inputs": resolved_inputs,
    }


def _yield_inputset_frames(ds: "Dataset",
                           inputset_name: str,
                           groups: Optional[Iterable[str]] = None,
                           sequences: Optional[Iterable[str]] = None,
                           scope: Optional[dict] = None):
    scope = scope or _resolve_inputset_scope(ds, inputset_name, groups, sequences)
    allowed_pairs = scope.get("pairs") or set()
    if not allowed_pairs:
        return
    resolved_inputs = scope.get("resolved_inputs") or []
    track_specs = [spec for spec in resolved_inputs if str(spec.get("kind")) == "tracks"]
    feat_specs = [spec for spec in resolved_inputs if str(spec.get("kind", "feature")) == "feature"]

    # Helper to load a prior feature dataframe for a given pair/run_id
    def _load_feature_df(feat_name: str, run_id: str, group: str, sequence: str) -> Optional[pd.DataFrame]:
        idx_path = _feature_index_path(ds, feat_name)
        if not idx_path.exists():
            return None
        df_idx = pd.read_csv(idx_path)
        df_idx = df_idx[
            (df_idx["run_id"].astype(str) == str(run_id)) &
            (df_idx["group"].astype(str) == str(group)) &
            (df_idx["sequence"].astype(str) == str(sequence))
        ]
        if df_idx.empty:
            return None
        pth = ds.remap_path(df_idx.iloc[0]["abs_path"]) if hasattr(ds, "remap_path") else Path(df_idx.iloc[0]["abs_path"])
        if not pth.exists():
            return None
        try:
            return pd.read_parquet(pth)
        except Exception:
            return None

    for g, s, df_tracks in _yield_sequences(ds, groups, sequences, allowed_pairs=allowed_pairs):
        df = df_tracks
        # Apply track column filter if provided
        if track_specs:
            cols = track_specs[-1].get("columns")
            if cols:
                keep = [c for c in cols if c in df.columns]
                df = df[keep]
        # Merge in feature specs if any
        for spec in feat_specs:
            feat_name = spec.get("feature")
            run_id = spec.get("run_id")
            if not feat_name or run_id is None:
                continue
            df_feat = _load_feature_df(feat_name, run_id, g, s)
            if df_feat is None or df_feat.empty:
                continue
            # Merge on shared meta columns
            on_cols = [c for c in ("frame", "time", "id", "group", "sequence") if c in df.columns and c in df_feat.columns]
            if not on_cols:
                on_cols = [c for c in ("frame", "time") if c in df.columns and c in df_feat.columns]
            if not on_cols:
                # fallback to cartesian concat
                df = pd.concat([df.reset_index(drop=True), df_feat.reset_index(drop=True)], axis=1)
            else:
                df = df.merge(df_feat, how="left", on=on_cols)
        yield g, s, df

@dataclass
class FeatureRunInfo:
    feature: str
    version: str
    params_hash: str
    params: dict
    scope: dict
    started_at: str
    finished_at: Optional[str] = None

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _ensure_feature_index(idx_path: Path):
    if not idx_path.exists():
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "feature": pd.Series(dtype="string"),
            "version": pd.Series(dtype="string"),
            "run_id": pd.Series(dtype="string"),
            "group": pd.Series(dtype="string"),
            "group_safe": pd.Series(dtype="string"),
            "sequence": pd.Series(dtype="string"),
            "sequence_safe": pd.Series(dtype="string"),
            "abs_path": pd.Series(dtype="string"),
            "n_rows": pd.Series(dtype="Int64"),
            "params_hash": pd.Series(dtype="string"),
            "started_at": pd.Series(dtype="string"),
            "finished_at": pd.Series(dtype="string"),
        }).to_csv(idx_path, index=False)

def _ensure_text_column(df: pd.DataFrame, column: str, fill: str = "") -> pd.DataFrame:
    """
    Make sure df[column] exists with object/string dtype so string assignments won't raise warnings.
    """
    if column not in df.columns:
        df[column] = fill
    else:
        if df[column].dtype != object:
            df[column] = df[column].astype(object)
        if fill is not None:
            df.loc[df[column].isna(), column] = fill
    return df

def _append_feature_index(idx_path: Path, rows: list[dict]):
    if not idx_path.exists():
        _ensure_feature_index(idx_path)
    df = pd.read_csv(idx_path)
    for col in ["feature", "version", "run_id", "group", "sequence",
                "group_safe", "sequence_safe", "abs_path",
                "params_hash", "started_at", "finished_at"]:
        df = _ensure_text_column(df, col, "")
    # Ensure group_safe/sequence_safe and finished_at in new rows
    new_rows = []
    for r in rows:
        r = dict(r)
        if "group_safe" not in r:
            r["group_safe"] = to_safe_name(r.get("group", "")) if r.get("group") else ""
        if "sequence_safe" not in r:
            r["sequence_safe"] = to_safe_name(r.get("sequence", "")) if r.get("sequence") else ""
        if "finished_at" not in r:
            r["finished_at"] = ""
        new_rows.append(r)
    # If index missing group_safe/sequence_safe, add and fill them
    for col, canon_col in [("group_safe", "group"), ("sequence_safe", "sequence")]:
        if col not in df.columns:
            df[col] = df[canon_col].apply(lambda v: to_safe_name(v) if pd.notnull(v) and str(v) else "")
    if "finished_at" not in df.columns:
        df["finished_at"] = ""
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(idx_path, index=False)


def _process_transform_worker(payload):
    """
    Helper for process-based feature transforms.
    payload: (module, cls_name, params, df, extra_attrs)
    """
    module, cls_name, params, df, extra_attrs = payload
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    feat = cls(params)
    for name, val in (extra_attrs or {}).items():
        try:
            setattr(feat, name, val)
        except Exception:
            pass
    return feat.transform(df)


def _model_run_root(ds: "Dataset", model_name: str, run_id: str) -> Path:
    return ds.get_root("models") / model_name / run_id


def _model_index_path(ds: "Dataset", model_name: str) -> Path:
    return ds.get_root("models") / model_name / "index.csv"


def _ensure_model_index(idx_path: Path):
    if not idx_path.exists():
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "model": pd.Series(dtype="string"),
            "version": pd.Series(dtype="string"),
            "run_id": pd.Series(dtype="string"),
            "config_path": pd.Series(dtype="string"),
            "config_hash": pd.Series(dtype="string"),
            "metrics_path": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
            "notes": pd.Series(dtype="string"),
            "started_at": pd.Series(dtype="string"),
            "finished_at": pd.Series(dtype="string"),
        }).to_csv(idx_path, index=False)


def _append_model_index(idx_path: Path, rows: list[dict]):
    if not idx_path.exists():
        _ensure_model_index(idx_path)
    df = pd.read_csv(idx_path)
    for col in ["model", "version", "run_id", "config_path", "config_hash",
                "metrics_path", "status", "notes", "started_at", "finished_at"]:
        df = _ensure_text_column(df, col, "")
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(idx_path, index=False)


def _load_model_config(config: str | Path | dict | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def _write_model_config(path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(config), indent=2))


def _validate_remote_cfg(remote_cfg: dict) -> dict:
    required = ["local_root", "remote_root", "ssh_host"]
    missing = [k for k in required if k not in remote_cfg or not remote_cfg.get(k)]
    if missing:
        raise ValueError(f"Remote config missing required keys: {', '.join(missing)}")
    out = dict(remote_cfg)
    out["local_root"] = str(Path(remote_cfg["local_root"]).expanduser().resolve())
    out["remote_root"] = str(remote_cfg["remote_root"]).rstrip("/")
    out["ssh_host"] = str(remote_cfg["ssh_host"])
    existing_patterns = _load_remote_only_patterns(out)
    if existing_patterns:
        pats = list(dict.fromkeys(existing_patterns))
        cfg_list = out.get("remote_only_patterns") or []
        out["remote_only_patterns"] = list(dict.fromkeys(cfg_list + pats))
    return out


def _rsync_remote(direction: str,
                  remote_cfg: dict,
                  include: Optional[Sequence[str]] = None,
                  exclude: Optional[Sequence[str]] = None):
    cfg = _validate_remote_cfg(remote_cfg)
    remote_cfg["remote_only_patterns"] = cfg.get("remote_only_patterns", [])
    _refresh_remote_only_patterns(cfg)
    include = include or []
    exclude = exclude or []
    base_cmd = ["rsync", "-az"]
    progress_flag = remote_cfg.get("rsync_progress")
    if progress_flag:
        if isinstance(progress_flag, str):
            base_cmd.append(progress_flag)
        else:
            base_cmd.append("--progress")
    if remote_cfg.get("delete", True):
        base_cmd.append("--delete")
    for pat in include:
        base_cmd.extend(["--include", pat])
    for pat in exclude:
        base_cmd.extend(["--exclude", pat])
    remote_only_patterns = cfg.get("remote_only_patterns") or []
    if direction == "to":
        base_cmd.extend(["--exclude", ".remote_jobs/**"])
    if direction in {"from", "to"} and remote_only_patterns:
        for pat in remote_only_patterns:
            if pat:
                base_cmd.extend(["--exclude", f"{pat.rstrip('/')}/**"])
    extra = remote_cfg.get("rsync_opts") or []
    base_cmd.extend(extra)

    local_root = cfg["local_root"]
    remote_root = cfg["remote_root"]
    ssh_host = cfg["ssh_host"]
    if direction == "to":
        src = f"{local_root.rstrip(os.sep)}/"
        dest = f"{ssh_host}:{remote_root.rstrip('/')}/"
    elif direction == "from":
        src = f"{ssh_host}:{remote_root.rstrip('/')}/"
        dest = f"{local_root.rstrip(os.sep)}/"
    else:
        raise ValueError("direction must be 'to' or 'from'")
    cmd = base_cmd + [src, dest]
    print(f"[remote-sync:{direction}] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def sync_to_remote(remote_cfg: dict,
                   include: Optional[Sequence[str]] = None,
                   exclude: Optional[Sequence[str]] = None):
    """
    Rsync local_root -> remote_root based on remote_cfg (ssh_host, paths).
    """
    _rsync_remote("to", remote_cfg, include, exclude)


def sync_from_remote(remote_cfg: dict,
                     include: Optional[Sequence[str]] = None,
                     exclude: Optional[Sequence[str]] = None):
    """
    Rsync remote_root -> local_root (pull results back).
    """
    _rsync_remote("from", remote_cfg, include, exclude)


def _ensure_relative_to(path: Path, root: Path) -> Path:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path '{path}' must live under local_root '{root}'.") from exc
    return rel


def _import_requests_for_jupyter():
    try:
        import requests  # type: ignore
        return requests
    except ImportError as exc:
        raise RuntimeError("Jupyter-based remote execution requires the 'requests' package.") from exc


def _import_websocket_for_jupyter():
    try:
        from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException  # type: ignore
        return create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
    except ImportError as exc:
        raise RuntimeError("Jupyter-based remote execution requires the 'websocket-client' package.") from exc


def _parse_jupyter_endpoint(remote_cfg: dict) -> dict:
    raw_url = remote_cfg.get("jupyter_url")
    if not raw_url:
        raise ValueError("remote_cfg missing 'jupyter_url' but Jupyter execution was requested.")
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid jupyter_url: {raw_url}")
    query = parse_qs(parsed.query)
    token = remote_cfg.get("jupyter_token") or (query.get("token", [""])[0])
    base_path = parsed.path or ""
    base_path = base_path.rstrip("/")
    for suffix in ("/lab", "/tree"):
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break
    if base_path == "/":
        base_path = ""
    http_base = f"{parsed.scheme}://{parsed.netloc}{base_path}"
    http_base = http_base.rstrip("/")
    if not http_base:
        http_base = f"{parsed.scheme}://{parsed.netloc}"
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_base = f"{ws_scheme}://{parsed.netloc}{base_path}"
    ws_base = ws_base.rstrip("/")
    verify = bool(remote_cfg.get("jupyter_verify_ssl", True))
    headers = dict(remote_cfg.get("jupyter_headers") or {})
    return {
        "http_base": http_base,
        "ws_base": ws_base,
        "token": token,
        "verify": verify,
        "headers": headers,
    }


def _build_jupyter_message(msg_type: str, content: dict, session_id: str) -> dict:
    return {
        "header": {
            "msg_id": str(uuid.uuid4()),
            "username": "dataset-remote",
            "session": session_id,
            "msg_type": msg_type,
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": content,
        "channel": "shell",
    }


def _run_remote_python_via_jupyter(remote_cfg: dict, script: str) -> tuple[str, str, str]:
    endpoint = _parse_jupyter_endpoint(remote_cfg)
    requests = _import_requests_for_jupyter()
    create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException = _import_websocket_for_jupyter()

    params = {}
    if endpoint["token"]:
        params["token"] = endpoint["token"]
    verify = endpoint["verify"]
    headers = endpoint["headers"]
    http_timeout = float(remote_cfg.get("jupyter_http_timeout", 30))
    ws_timeout = float(remote_cfg.get("jupyter_ws_timeout", 120))

    session = requests.Session()
    kernel_id = None
    stdout = ""
    stderr = ""
    remote_cmd = f"jupyter:{endpoint['http_base']}"
    print(f"[remote-run] executing via Jupyter -> {endpoint['http_base']}")
    try:
        resp = session.post(
            f"{endpoint['http_base']}/api/kernels",
            params=params,
            headers=headers,
            json={},
            timeout=http_timeout,
            verify=verify,
        )
        resp.raise_for_status()
        kernel_id = resp.json().get("id")
        if not kernel_id:
            raise RuntimeError("Failed to create remote kernel via Jupyter API (missing id).")
        ws_url = f"{endpoint['ws_base']}/api/kernels/{kernel_id}/channels"
        if params:
            ws_url = f"{ws_url}?{urlencode(params)}"
        ws_headers = [f"{k}: {v}" for k, v in headers.items()]
        sslopt = None
        if ws_url.startswith("wss://") and not verify:
            import ssl  # lazy import
            sslopt = {"cert_reqs": ssl.CERT_NONE}
        ws = create_connection(ws_url, header=ws_headers, sslopt=sslopt, timeout=ws_timeout)
        try:
            session_id = str(uuid.uuid4())
            execute_msg = _build_jupyter_message(
                "execute_request",
                {
                    "code": script,
                    "silent": False,
                    "store_history": False,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True,
                },
                session_id=session_id,
            )
            ws.send(json.dumps(execute_msg))
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            error_payload: dict | None = None
            idle_seen = False
            reply_seen = False
            while True:
                try:
                    raw = ws.recv()
                except WebSocketTimeoutException as exc:
                    raise RuntimeError("Timed out waiting for output from remote Jupyter kernel.") from exc
                except WebSocketConnectionClosedException as exc:
                    raise RuntimeError("Remote Jupyter kernel connection closed unexpectedly.") from exc
                if raw is None:
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                channel = msg.get("channel")
                msg_type = msg.get("msg_type")
                content = msg.get("content", {})
                if channel == "iopub":
                    if msg_type == "stream":
                        text = content.get("text", "")
                        if content.get("name") == "stderr":
                            stderr_parts.append(text)
                        else:
                            stdout_parts.append(text)
                    elif msg_type == "error":
                        error_payload = content
                        tb = "\n".join(content.get("traceback") or [])
                        if tb:
                            stderr_parts.append(tb + "\n")
                    elif msg_type == "status" and content.get("execution_state") == "idle":
                        idle_seen = True
                    elif msg_type in ("execute_result", "display_data"):
                        data = content.get("data") or {}
                        text_data = data.get("text/plain")
                        if text_data:
                            stdout_parts.append(f"{text_data}\n")
                elif channel == "shell" and msg_type == "execute_reply":
                    reply_seen = True
                    if content.get("status") == "error" and not error_payload:
                        error_payload = content
                if idle_seen and reply_seen:
                    break
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            if error_payload:
                ename = error_payload.get("ename", "RemoteError")
                evalue = error_payload.get("evalue", "")
                trace = "\n".join(error_payload.get("traceback") or [])
                if trace and trace not in stderr:
                    stderr = f"{stderr}{trace}\n"
                raise RuntimeError(
                    "[remote-run:jupyter] command failed.\n"
                    f"Command: {remote_cmd}\n"
                    f"{ename}: {evalue}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
        finally:
            try:
                ws.close()
            except Exception:
                pass
    finally:
        if kernel_id:
            try:
                session.delete(
                    f"{endpoint['http_base']}/api/kernels/{kernel_id}",
                    params=params,
                    headers=headers,
                    timeout=http_timeout,
                    verify=verify,
                )
            except Exception:
                pass
        session.close()
    return stdout, stderr, remote_cmd


def _run_remote_python(remote_cfg: dict, script: str) -> tuple[str, str, str]:
    cfg = _validate_remote_cfg(remote_cfg)
    remote_root = cfg["remote_root"]
    python_cmd = remote_cfg.get("python_cmd", "python")
    ssh_host = cfg["ssh_host"]
    if remote_cfg.get("jupyter_url"):
        return _run_remote_python_via_jupyter(remote_cfg, script)
    remote_cmd = (
        f"cd {shlex.quote(remote_root)} && "
        f"{python_cmd} - <<'PY'\n{script}\nPY\n"
    )
    print(f"[remote-run] executing on {ssh_host}")
    ssh_command = ["ssh", ssh_host, remote_cmd]
    try:
        result = subprocess.run(
            ssh_command,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        cmd_str = " ".join(ssh_command)
        raise RuntimeError(
            "[remote-run] command failed.\n"
            f"Command: {cmd_str}\n"
            f"Exit code: {exc.returncode}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        ) from exc
    return result.stdout, result.stderr, remote_cmd


def _build_job_code(work_code: str, result_expr: str = "{}") -> str:
    template = """import json, traceback, datetime, gc
from datetime import timezone
import os, sys
meta_path = "__META_PATH__"


def _update(status, **extra):
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        meta = {}
    meta.update(extra)
    meta["status"] = status
    meta["updated_at"] = datetime.datetime.now(timezone.utc).isoformat()
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


_update("running", started_at=datetime.datetime.now(timezone.utc).isoformat())
try:
__WORK_CODE__
    _update("finished", finished_at=datetime.datetime.now(timezone.utc).isoformat(), **(__RESULT_EXPR__))
except Exception as exc:
    import traceback as _tb
    _update(
        "failed",
        finished_at=datetime.datetime.now(timezone.utc).isoformat(),
        error=str(exc),
        traceback=_tb.format_exc(),
    )
    raise
finally:
    gc.collect()
"""
    template = template.replace("__WORK_CODE__", textwrap.indent(work_code, "    "))
    template = template.replace("__RESULT_EXPR__", result_expr or "{}")
    return template


def _submit_remote_detached_job(remote_cfg: dict,
                                job_kind: str,
                                work_code: str,
                                result_expr: str = "{}",
                                meta_extra: Optional[dict] = None) -> str:
    cfg = _validate_remote_cfg(remote_cfg)
    remote_root = cfg["remote_root"]
    python_cmd = remote_cfg.get("python_cmd", "python3")
    job_code_template = _build_job_code(work_code, result_expr=result_expr)
    job_code_json = json.dumps(job_code_template)
    meta_literal = repr(meta_extra or {})
    remote_script = f"""
import os, json, datetime, subprocess, uuid
remote_root = r"{remote_root}"
job_root = os.path.join(remote_root, ".remote_jobs")
scripts_dir = os.path.join(job_root, "scripts")
logs_dir = os.path.join(job_root, "logs")
meta_dir = os.path.join(job_root, "meta")
for d in (job_root, scripts_dir, logs_dir, meta_dir):
    os.makedirs(d, exist_ok=True)
job_id = "{job_kind}-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
script_path = os.path.join(scripts_dir, job_id + ".py")
log_path = os.path.join(logs_dir, job_id + ".log")
meta_path = os.path.join(meta_dir, job_id + ".json")
meta = {meta_literal}
meta.update({{
    "job_id": job_id,
    "job_kind": "{job_kind}",
    "status": "queued",
    "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "log_path": log_path,
    "script_path": script_path,
}})
with open(meta_path, "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
job_code = {job_code_json}
job_code = job_code.replace("__META_PATH__", meta_path)
with open(script_path, "w", encoding="utf-8") as fh:
    fh.write(job_code)
log_handle = open(log_path, "w")
proc = subprocess.Popen(
    ["{python_cmd}", "-u", script_path],
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
log_handle.close()
meta["pid"] = proc.pid
with open(meta_path, "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
print("REMOTE_JOB_ID=" + job_id)
"""
    remote_cfg_ssh = dict(remote_cfg)
    remote_cfg_ssh.pop("jupyter_url", None)
    stdout, stderr, _ = _run_remote_python(remote_cfg_ssh, remote_script)
    job_id = None
    for line in stdout.splitlines():
        if line.startswith("REMOTE_JOB_ID="):
            job_id = line.split("=", 1)[1].strip()
            break
    if job_id is None:
        raise RuntimeError(f"Failed to submit remote {job_kind} job. STDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    return job_id


def _maybe_apply_remote_only_pattern(remote_cfg: dict, job_meta: dict) -> Optional[str]:
    if not job_meta or not job_meta.get("remote_only"):
        return None
    run_id = job_meta.get("run_id")
    storage = job_meta.get("storage_feature")
    if not run_id or not storage:
        return None
    pattern = f"features/{storage}/{run_id}"
    _record_remote_only_pattern(remote_cfg, pattern)
    return pattern

def _update_finished_times(idx_path: Path, run_id: str, finished_at: str):
    df = pd.read_csv(idx_path)
    df = _ensure_text_column(df, "run_id", "")
    df = _ensure_text_column(df, "finished_at", "")
    sel = (df["run_id"] == str(run_id)) & ((df["finished_at"].isna()) | (df["finished_at"] == ""))
    if sel.any():
        df.loc[sel, "finished_at"] = str(finished_at)
        df.to_csv(idx_path, index=False)

def _ensure_labels_index(idx_path: Path):
    if not idx_path.exists():
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
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
        }).to_csv(idx_path, index=False)

def _append_labels_index(idx_path: Path, rows: list[dict]):
    if not idx_path.exists():
        _ensure_labels_index(idx_path)
    df = pd.read_csv(idx_path)
    for col in LABEL_INDEX_COLUMNS:
        fill = "" if col != "n_frames" else None
        df = _ensure_text_column(df, col, "" if fill is None else fill)
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
            row["sequence_safe"] = to_safe_name(row["sequence"]) if row["sequence"] else ""
        row.setdefault("abs_path", "")
        row.setdefault("source_abs_path", "")
        row.setdefault("source_md5", "")
        if "n_frames" not in row:
            row["n_frames"] = ""
        row.setdefault("label_ids", "")
        row.setdefault("label_names", "")
        mask = (updated["group"].fillna("") == row["group"]) & (updated["sequence"].fillna("") == row["sequence"])
        updated = updated[~mask]
        updated = pd.concat([updated, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(idx_path, index=False)


def _coerce_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _list_feature_runs(ds: "Dataset", feature_name: str) -> pd.DataFrame:
    idx = _feature_index_path(ds, feature_name)
    if not idx.exists():
        raise FileNotFoundError(f"No index for feature '{feature_name}'. Expected: {idx}")
    df = pd.read_csv(idx)
    # prefer finished runs, newest first
    if "finished_at" in df.columns:
        cand = df[df["finished_at"].fillna("").astype(str) != ""]
        base = cand if len(cand) else df
        base = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                ascending=False, kind="stable")
    else:
        base = df.sort_values(by=["started_at"], ascending=False, kind="stable")
    return base

def _latest_feature_run_root(ds: "Dataset", feature_name: str) -> tuple[str, Path]:
    base = _list_feature_runs(ds, feature_name)
    if base.empty:
        raise ValueError(f"No runs found for feature '{feature_name}'.")
    run_id = str(base.iloc[0]["run_id"])
    return run_id, _feature_run_root(ds, feature_name, run_id)        

def run_feature(self,
                feature: Feature,
                groups: Optional[Iterable[str]] = None,
                sequences: Optional[Iterable[str]] = None,
                overwrite: bool = False,
                input_kind: str = "tracks",
                input_feature: Optional[str] = None,
                input_run_id: Optional[str] = None,
                parallel_workers: Optional[int] = None,
                parallel_mode: Optional[str] = "thread"):
    """
    Apply a Feature over a chosen scope (default: whole dataset).

    Parameters
    ----------
    feature : Feature
        The feature object implementing the Feature protocol.
    groups, sequences : optional iterables
        Scope filter (applies to whichever input source is used).
    overwrite : bool
        Overwrite existing outputs for this run_id.
    input_kind : {'tracks','feature'}
        Where to read inputs from. 'tracks' (default) streams standardized tracks.
        'feature' streams outputs of a previously run feature (use input_feature/run_id).
    input_feature : str | None
        Name of the prior feature to use as input when input_kind='feature'.
        If None with input_kind='feature', raises.
    input_run_id : str | None
        Specific run_id of the prior feature to use. If None, the latest finished run is used.
    parallel_workers : int | None
        When >1 and the feature declares itself parallelizable, run the transform phase in parallel
        across per-sequence chunks using this many threads. Defaults to sequential execution.
    parallel_mode : {'thread','process'}
        Execution backend when parallel_workers > 1. 'thread' (default) uses ThreadPoolExecutor;
        'process' uses ProcessPoolExecutor and requires picklable params/feature/inputs.

    Behavior
    --------
    - Fits globally if feature.needs_fit().
      If feature.supports_partial_fit(), streams tables and calls partial_fit(); otherwise collects in memory.
    - Transforms per (group,sequence) table and writes:
          features/<feature>/<run_id>/<group_safe__sequence_safe>.parquet
    - Saves model (if any) under:
          features/<feature>/<run_id>/model.joblib
    - Returns run_id.
    """
    # Determine on-disk storage name (may encode upstream)
    storage_feature_name = getattr(feature, "storage_feature_name", feature.name)
    use_input_suffix = getattr(feature, "storage_use_input_suffix", True)
    if input_kind in {"feature", "inputset"} and input_feature and use_input_suffix:
        storage_feature_name = f"{storage_feature_name}__from__{input_feature}"

    # Prepare run id & root
    feature_params = getattr(feature, "params", {})
    params_hash = _hash_params(feature_params)
    run_id = f"{feature.version}-{params_hash}"
    run_root = _feature_run_root(self, storage_feature_name, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Persist params for discoverability
    params_path = run_root / "params.json"
    try:
        params_path.write_text(json.dumps(_json_ready(feature_params), indent=2))
    except Exception as exc:
        print(f"[feature:{feature.name}] failed to save params.json: {exc}", file=sys.stderr)

    idx_path = _feature_index_path(self, storage_feature_name)
    _ensure_feature_index(idx_path)
    started = _now_iso()

    max_workers = int(parallel_workers) if parallel_workers and int(parallel_workers) > 1 else 1
    parallel_mode = (parallel_mode or "thread").lower()
    if parallel_mode not in {"thread", "process"}:
        parallel_mode = "thread"
    parallel_allowed = bool(getattr(feature, "parallelizable", False))
    if max_workers > 1 and not parallel_allowed:
        print(f"[feature:{feature.name}] parallel_workers requested but feature is not parallelizable; running sequentially.",
              file=sys.stderr)
        max_workers = 1

    # Helpers to enumerate pairs without loading full data (to skip upfront)
    def _list_track_pairs():
        idx_path = self.get_root("tracks") / "index.csv"
        if not idx_path.exists():
            return set()
        df_idx = pd.read_csv(idx_path)
        df_idx["group"] = df_idx["group"].fillna("").astype(str)
        df_idx["sequence"] = df_idx["sequence"].fillna("").astype(str)
        mask = pd.Series(True, index=df_idx.index)
        if groups is not None:
            mask &= df_idx["group"].isin({str(g) for g in groups})
        if sequences is not None:
            mask &= df_idx["sequence"].isin({str(s) for s in sequences})
        return set(zip(df_idx.loc[mask, "group"], df_idx.loc[mask, "sequence"]))

    def _resolve_feature_run_and_pairs(feat_name: str, run_id_opt: Optional[str]):
        idx_path = _feature_index_path(self, feat_name)
        if not idx_path.exists():
            raise FileNotFoundError(f"No index for feature '{feat_name}'. Expected: {idx_path}")
        df_idx = pd.read_csv(idx_path)
        df_idx["group"] = df_idx["group"].fillna("").astype(str)
        df_idx["sequence"] = df_idx["sequence"].fillna("").astype(str)
        resolved_run_id = run_id_opt
        if resolved_run_id is None:
            if "finished_at" in df_idx.columns:
                cand = df_idx[df_idx["finished_at"].fillna("").astype(str) != ""]
                base = cand if len(cand) else df_idx
                base = base.sort_values(by=["finished_at" if len(cand) else "started_at"],
                                        ascending=False, kind="stable")
            else:
                base = df_idx.sort_values(by=["started_at"], ascending=False, kind="stable")
            if base.empty:
                raise ValueError(f"No runs found for feature '{feat_name}'.")
            resolved_run_id = str(base.iloc[0]["run_id"])
        df_sel = df_idx[df_idx["run_id"].astype(str) == str(resolved_run_id)]
        if groups is not None:
            df_sel = df_sel[df_sel["group"].isin({str(g) for g in groups})]
        if sequences is not None:
            df_sel = df_sel[df_sel["sequence"].isin({str(s) for s in sequences})]
        pairs = set(zip(df_sel["group"], df_sel["sequence"]))
        return resolved_run_id, pairs

    def _out_path_for_pair(pair: tuple[str, str]) -> Path:
        g, s = pair
        g = "" if g is None else str(g)
        s = "" if s is None else str(s)
        safe_group = to_safe_name(g) if g else ""
        safe_seq = to_safe_name(s)
        out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
        return run_root / out_name

    # Pre-pass: resolve candidate pairs, skip existing outputs before loading data
    input_scope = None
    pairs_all: Optional[set[tuple[str, str]]] = None
    pairs_to_compute: Optional[set[tuple[str, str]]] = None
    preexisting_rows: list[dict] = []
    resolved_input_run_id = input_run_id

    if input_kind not in {"tracks", "feature", "inputset"}:
        raise ValueError("input_kind must be 'tracks', 'feature', or 'inputset'")

    if input_kind == "tracks":
        pairs_all = _list_track_pairs()
    elif input_kind == "feature":
        if not input_feature:
            raise ValueError("input_feature must be provided when input_kind='feature'")
        resolved_input_run_id, pairs_all = _resolve_feature_run_and_pairs(input_feature, input_run_id)
    elif input_kind == "inputset":
        if not input_feature:
            raise ValueError("input_feature (inputset name) must be provided when input_kind='inputset'")
        input_scope = _resolve_inputset_scope(self, input_feature, groups, sequences)
        pairs_all = input_scope.get("pairs") or set()

    if pairs_all is not None:
        pairs_to_compute = set()
        for pair in pairs_all:
            out_path = _out_path_for_pair(pair)
            if out_path.exists() and not overwrite:
                if getattr(feature, "skip_existing_outputs", False):
                    continue  # neither compute nor index append
                g, s = pair
                g = "" if g is None else str(g)
                s = "" if s is None else str(s)
                safe_group = to_safe_name(g) if g else ""
                safe_seq = to_safe_name(s)
                preexisting_rows.append({
                    "feature": storage_feature_name,
                    "version": feature.version,
                    "run_id": run_id,
                    "group": g,
                    "sequence": s,
                    "group_safe": safe_group,
                    "sequence_safe": safe_seq,
                    "abs_path": str(out_path.resolve()),
                    "n_rows": None,
                    "params_hash": params_hash,
                    "started_at": started,
                    "finished_at": "",
                })
            else:
                pairs_to_compute.add(pair)

    # Choose input iterator (filtered to pairs_to_compute when known)
    if input_kind == "feature":
        iter_inputs = lambda: _yield_feature_frames(self, input_feature, resolved_input_run_id, groups, sequences,
                                                    allowed_pairs=pairs_to_compute)
    elif input_kind == "inputset":
        scope_for_iter = input_scope or {}
        if pairs_to_compute is not None:
            scope_for_iter = dict(scope_for_iter)
            scope_for_iter["pairs"] = pairs_to_compute
        iter_inputs = lambda: _yield_inputset_frames(self, input_feature, groups, sequences, scope_for_iter)
        input_scope = scope_for_iter
    else:
        iter_inputs = lambda: _yield_sequences(self, groups, sequences, allowed_pairs=pairs_to_compute)

    # ===== FIT PHASE =====
    if hasattr(feature, "bind_dataset"):
        try:
            feature.bind_dataset(self)   # allow feature to read feature indexes & roots
        except Exception as e:
            print(f"[feature:{feature.name}] bind_dataset failed: {e}", file=sys.stderr)

    scope_constraints = {}
    if input_scope is not None:
        for key in ("groups", "sequences", "safe_sequences", "pairs", "pair_safe_map"):
            val = input_scope.get(key)
            if val:
                scope_constraints[key] = val
    if groups is not None:
        norm_groups = sorted({str(g) for g in groups})
        if norm_groups:
            scope_constraints["groups"] = norm_groups
            scope_constraints["safe_groups"] = sorted({to_safe_name(g) for g in norm_groups})
    if sequences is not None:
        norm_sequences = sorted({str(s) for s in sequences})
        if norm_sequences:
            scope_constraints["sequences"] = norm_sequences
            if not scope_constraints.get("safe_sequences"):
                scope_constraints["safe_sequences"] = sorted({to_safe_name(s) for s in norm_sequences})
    if scope_constraints:
        setattr(feature, "_scope_constraints", scope_constraints)
        if hasattr(feature, "set_scope_constraints"):
            try:
                feature.set_scope_constraints(scope_constraints)
            except Exception as e:
                print(f"[feature:{feature.name}] set_scope_constraints failed: {e}", file=sys.stderr)

    if input_scope is not None:
        setattr(feature, "_scope_filter", input_scope)
        if hasattr(feature, "set_scope_filter"):
            try:
                feature.set_scope_filter(input_scope)
            except Exception as e:
                print(f"[feature:{feature.name}] set_scope_filter failed: {e}", file=sys.stderr)
    if feature.needs_fit():
        if feature.supports_partial_fit():
            for _, _, df in iter_inputs():
                try:
                    feature.partial_fit(df)
                except Exception as e:
                    print(f"[feature:{feature.name}] partial_fit failed: {e}", file=sys.stderr)
            try:
                feature.finalize_fit()
            except Exception:
                pass
        else:
            all_dfs = []
            for _, _, df in iter_inputs():
                all_dfs.append(df)
            # Always call fit, even if no streamed inputs were found.
            # Many "global/artifact" features load their own matrices from disk.
            try:
                feature.fit(all_dfs)
            except TypeError:
                # Backward-compat: some features may define fit(self) with no args.
                try:
                    feature.fit()  # type: ignore
                except Exception as e:
                    print(f"[feature:{feature.name}] fit() failed: {e}", file=sys.stderr)

        # Save model state if any
        model_path = run_root / "model.joblib"
        try:
            feature.save_model(model_path)
        except NotImplementedError:
            print('No Feature Implemented')
            pass

    # ===== TRANSFORM PHASE =====
    out_rows = list(preexisting_rows) if preexisting_rows else []
    had_transform_inputs = False

    def _make_meta(group, sequence):
        group_str = "" if group is None else str(group)
        seq_str = "" if sequence is None else str(sequence)
        safe_group = to_safe_name(group_str) if group_str else ""
        safe_seq = to_safe_name(seq_str)
        out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
        return {
            "group": group_str,
            "sequence": seq_str,
            "safe_group": safe_group,
            "safe_seq": safe_seq,
            "out_path": run_root / out_name,
        }

    def _append_row(meta, n_rows, abs_path: Optional[str] = None):
        path_str = abs_path or str(meta["out_path"].resolve())
        out_rows.append({
            "feature": storage_feature_name,
            "version": feature.version,
            "run_id": run_id,
            "group": meta["group"],
            "sequence": meta["sequence"],
            "group_safe": meta["safe_group"],
            "sequence_safe": meta["safe_seq"],
            "abs_path": path_str,
            "n_rows": n_rows,
            "params_hash": params_hash,
            "started_at": started,
            "finished_at": "",
        })

    def _write_parquet_chunks(meta, payload: dict):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for chunked parquet writes; install pyarrow to continue.") from exc
        data = payload["parquet_data"]
        columns = payload["columns"]
        sequence = payload.get("sequence", "")
        group = payload.get("group")
        chunk_size = int(payload.get("chunk_size", 10000))
        chunk_size = max(1, chunk_size)
        n_rows = data.shape[0]
        schema_fields = [("frame", pa.int32())]
        schema_fields.extend([(name, pa.float32()) for name in columns])
        schema_fields.append(("sequence", pa.string()))
        if group:
            schema_fields.append(("group", pa.string()))
        schema = pa.schema(schema_fields)
        writer = pq.ParquetWriter(meta["out_path"], schema, compression="snappy")
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            arrays = {"frame": pa.array(np.arange(start, end, dtype=np.int32))}
            for idx, name in enumerate(columns):
                arrays[name] = pa.array(data[start:end, idx])
            arrays["sequence"] = pa.array([sequence] * (end - start))
            if group:
                arrays["group"] = pa.array([group] * (end - start))
            table = pa.Table.from_pydict(arrays, schema=schema)
            writer.write_table(table)
        writer.close()
        gc.collect()
        _append_row(meta, n_rows)

    def _write_parquet_stream(meta, payload: dict):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for chunked parquet writes; install pyarrow to continue.") from exc
        columns = payload["columns"]
        sequence = payload.get("sequence", "")
        group = payload.get("group")
        chunk_iter = payload["parquet_chunk_iter"]
        schema_fields = [("frame", pa.int32())]
        schema_fields.extend([(name, pa.float32()) for name in columns])
        schema_fields.append(("sequence", pa.string()))
        if group:
            schema_fields.append(("group", pa.string()))
        schema = pa.schema(schema_fields)
        writer = pq.ParquetWriter(meta["out_path"], schema, compression="snappy")
        total_rows = 0
        for start, chunk in chunk_iter:
            chunk_len = chunk.shape[0]
            arrays = {"frame": pa.array(np.arange(start, start + chunk_len, dtype=np.int32))}
            for idx, name in enumerate(columns):
                arrays[name] = pa.array(chunk[:, idx])
            arrays["sequence"] = pa.array([sequence] * chunk_len)
            if group:
                arrays["group"] = pa.array([group] * chunk_len)
            table = pa.Table.from_pydict(arrays, schema=schema)
            writer.write_table(table)
            total_rows = max(total_rows, start + chunk_len)
        writer.close()
        gc.collect()
        _append_row(meta, total_rows)

    def _write_output(meta, df_feat):
        out_path = meta["out_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(df_feat, dict) and "parquet_chunk_iter" in df_feat:
            _write_parquet_stream(meta, df_feat)
            return

        if isinstance(df_feat, dict) and "parquet_data" in df_feat:
            _write_parquet_chunks(meta, df_feat)
            return

        if isinstance(df_feat, dict) and "data" in df_feat and "columns" in df_feat:
            data = df_feat["data"]
            columns = df_feat["columns"]
            seq = df_feat.get("sequence")
            group = df_feat.get("group")
            df_out = pd.DataFrame(data, columns=columns)
            df_out.insert(0, "frame", np.arange(df_out.shape[0], dtype=np.int32))
            if seq is not None and "sequence" not in df_out.columns:
                df_out["sequence"] = seq
            if group is not None and "group" not in df_out.columns:
                df_out["group"] = group
        else:
            if df_feat is None:
                df_out = pd.DataFrame()
            elif isinstance(df_feat, pd.DataFrame):
                df_out = df_feat
            else:
                df_out = pd.DataFrame(df_feat)

        n_rows = int(len(df_out))
        df_out.to_parquet(out_path, index=False)
        del df_out
        gc.collect()
        _append_row(meta, n_rows)

    executor = None
    if max_workers > 1:
        if parallel_mode == "process":
            executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))
        else:
            executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    extra_attrs = {}
    for attr in ("_scope_filter", "_scope_constraints"):
        if hasattr(feature, attr):
            extra_attrs[attr] = getattr(feature, attr)

    for g, s, df in iter_inputs():
        had_transform_inputs = True
        meta = _make_meta(g, s)
        out_path = meta["out_path"]
        if out_path.exists() and not overwrite:
            if getattr(feature, "skip_existing_outputs", False):
                continue
            try:
                n_rows = int(pd.read_parquet(out_path).shape[0])
            except Exception:
                n_rows = None
            _append_row(meta, n_rows, abs_path=str(out_path.resolve()))
            continue

        if executor:
            if parallel_mode == "process":
                payload = (feature.__module__, feature.__class__.__name__, getattr(feature, "params", {}), df, extra_attrs)
                futures[executor.submit(_process_transform_worker, payload)] = meta
            else:
                futures[executor.submit(feature.transform, df)] = meta
        else:
            try:
                df_feat = feature.transform(df)
            except Exception as e:
                print(f"[feature:{feature.name}] transform failed for ({g},{s}): {e}", file=sys.stderr)
                continue
            _write_output(meta, df_feat)
            try:
                del df_feat
            except Exception:
                pass
            gc.collect()

    if executor:
        if futures:
            for future in as_completed(futures):
                meta = futures[future]
                try:
                    df_feat = future.result()
                except Exception as e:
                    print(f"[feature:{feature.name}] transform failed for ({meta['group']},{meta['sequence']}): {e}",
                          file=sys.stderr)
                    continue
                _write_output(meta, df_feat)
                try:
                    del df_feat
                except Exception:
                    pass
                gc.collect()
        executor.shutdown(wait=True)

    # Some global-only features (e.g., clustering over standalone artifacts) never
    # receive streamed tables. In that case we still want to record the run in the
    # index so downstream tooling can discover the run_id.
    if not out_rows and not had_transform_inputs:
        marker_seq = "__global__"
        safe_marker_seq = to_safe_name(marker_seq)
        marker_path = run_root / f"{safe_marker_seq}.parquet"
        marker_df = pd.DataFrame({"run_marker": [True]})
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_df.to_parquet(marker_path, index=False)
        out_rows.append({
            "feature": storage_feature_name, "version": feature.version, "run_id": run_id,
            "group": "", "sequence": marker_seq, "group_safe": "", "sequence_safe": safe_marker_seq,
            "abs_path": str(marker_path.resolve()),
            "n_rows": int(len(marker_df)), "params_hash": params_hash,
            "started_at": started, "finished_at": ""
        })

    _append_feature_index(idx_path, out_rows)
    _update_finished_times(idx_path, run_id, _now_iso())
    print(f"[feature:{storage_feature_name}] completed run_id={run_id} -> {run_root}")
    return run_id

# Attach to class
Dataset.run_feature = run_feature


def train_model(self,
                model,
                config: str | Path | dict | None = None,
                overwrite: bool = False) -> str:
    """
    Train a registered model using a JSON (or dict) configuration.

    Parameters
    ----------
    model : object
        Model/trainer instance implementing:
          - name (str)
          - version (str)
          - bind_dataset(self, ds) optional
          - configure(self, config: dict, run_root: Path) optional
          - train(self) -> dict | None
    config : str | Path | dict | None
        Path to a JSON config file or an in-memory dict of hyperparameters.
    overwrite : bool
        Reserved for future use (run_ids are hash-based, so reruns overwrite same folder).
    """
    storage_model_name = getattr(model, "storage_model_name", getattr(model, "name", None))
    if not storage_model_name:
        raise ValueError("Model must define 'name' or 'storage_model_name'.")
    config_dict = _load_model_config(config)
    config_hash = _hash_params(config_dict or {})
    run_id = f"{model.version}-{config_hash}"
    run_root = _model_run_root(self, storage_model_name, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    idx_path = _model_index_path(self, storage_model_name)
    _ensure_model_index(idx_path)

    config_path = run_root / "config.json"
    _write_model_config(config_path, config_dict)

    started = _now_iso()
    if hasattr(model, "bind_dataset"):
        try:
            model.bind_dataset(self)
        except Exception as exc:
            print(f"[model:{storage_model_name}] bind_dataset failed: {exc}", file=sys.stderr)

    if hasattr(model, "configure"):
        model.configure(config_dict, run_root)
    else:
        setattr(model, "config", config_dict)
        setattr(model, "run_root", run_root)

    metrics = None
    metrics_path = run_root / "metrics.json"
    status = "success"
    notes = ""
    try:
        metrics = model.train()
    except Exception as exc:
        status = "failed"
        notes = str(exc)
        finished = _now_iso()
        rows = [{
            "model": storage_model_name,
            "version": model.version,
            "run_id": run_id,
            "config_path": str(config_path),
            "config_hash": config_hash,
            "metrics_path": "",
            "status": status,
            "notes": notes[:500],
            "started_at": started,
            "finished_at": finished,
        }]
        _append_model_index(idx_path, rows)
        raise

    finished = _now_iso()
    if metrics:
        metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2))
    else:
        if metrics_path.exists():
            metrics_path.unlink()

    rows = [{
        "model": storage_model_name,
        "version": model.version,
        "run_id": run_id,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "metrics_path": str(metrics_path) if metrics and metrics_path.exists() else "",
        "status": status,
        "notes": notes[:500],
        "started_at": started,
        "finished_at": finished,
    }]
    _append_model_index(idx_path, rows)
    print(f"[model:{storage_model_name}] completed run_id={run_id} -> {run_root}")
    return run_id


Dataset.train_model = train_model


def train_model_remote(self,
                       model_class: str,
                       config: str | Path | dict | None,
                       remote_cfg: dict,
                       sync_before: bool = True,
                       sync_after: bool = True,
                       include: Optional[Sequence[str]] = None,
                       exclude: Optional[Sequence[str]] = None,
                       detached: bool = False) -> str | dict:
    """
    Sync the project to a remote server, run Dataset.train_model there, then pull results back.

    Parameters
    ----------
    model_class : str
        Dotted import path for the model class (e.g., "models_behavior.BehaviorXGBoostModel").
    config : str | Path | dict
        Config JSON path (relative to local_root) or an in-memory dict to serialize.
    remote_cfg : dict
        {
          "local_root": "/local/project",
          "remote_root": "/remote/project",
          "ssh_host": "user@host",
          "jupyter_url": "...token...",   # optional: execute via running Jupyter server
          "jupyter_token": "...",         # optional token override
          "jupyter_verify_ssl": True,     # optional SSL toggle for HTTPS/WSS
          "jupyter_headers": {...},       # optional extra HTTP headers
          "jupyter_http_timeout": 30,     # optional HTTP timeout seconds
          "jupyter_ws_timeout": 120,      # optional websocket timeout seconds
          "python_cmd": "python",         # optional (used for SSH mode)
          "rsync_opts": [...],            # optional rsync args
          "rsync_progress": True,         # optional: pass --progress (or custom str) to rsync
          "delete": True,                 # optional rsync delete flag
          "root_map": {"/local/data": "/remote/data"}  # optional root prefix remap
        }
    sync_before / sync_after : bool
        Whether to rsync before/after remote execution.
    include / exclude : Optional sequences of rsync patterns.
    """
    if detached and sync_after:
        raise ValueError("sync_after cannot be used when detached; wait for job completion to sync artifacts.")

    if detached and sync_after:
        raise ValueError("sync_after cannot be used when detached; wait for job completion to sync artifacts.")

    if sync_before:
        sync_to_remote(remote_cfg, include=include, exclude=exclude)

    local_root = Path(remote_cfg["local_root"]).expanduser().resolve()
    remote_root = Path(remote_cfg["remote_root"])
    manifest_path = Path(self.manifest_path).resolve()
    rel_manifest = _ensure_relative_to(manifest_path, local_root)
    manifest_remote = (remote_root / rel_manifest).as_posix()

    if isinstance(config, dict):
        config_hash = _hash_params(config)
        cfg_dir = Path(self.get_root("models")) / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        config_path_local = cfg_dir / f"{model_class.split('.')[-1]}_{config_hash}.json"
        config_path_local.write_text(json.dumps(_json_ready(config), indent=2))
    elif config is None:
        raise ValueError("Remote training requires a config file or dict.")
    else:
        config_path_local = Path(config).expanduser().resolve()

    rel_config = _ensure_relative_to(config_path_local, local_root)
    config_remote = (remote_root / rel_config).as_posix()

    root_map = remote_cfg.get("root_map") or {}
    root_map_json = json.dumps(root_map)
    remote_cwd = remote_root.as_posix()
    work_code = f"""
import os, sys, importlib, json
os.chdir(r"{remote_cwd}")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
src_dir = os.path.join(cwd, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from behavior.dataset import Dataset
manifest_path = r"{manifest_remote}"
config_path = r"{config_remote}"
root_map = json.loads({root_map_json!r})
module_path, class_name = "{model_class}".rsplit(".", 1)
ModelCls = getattr(importlib.import_module(module_path), class_name)
dataset = Dataset(manifest_path).load(ensure_roots=not bool(root_map))
if root_map:
    dataset.remap_roots(root_map)
    dataset.ensure_roots()
model = ModelCls()
run_id = dataset.train_model(model, config_path)
print("REMOTE_RUN_ID=" + run_id)
"""

    if detached:
        meta_extra = {
            "job_type": "train_model",
            "model_class": model_class,
            "config_remote": config_remote,
        }
        job_id = _submit_remote_detached_job(
            remote_cfg,
            job_kind="train_model",
            work_code=work_code,
            result_expr="{'run_id': run_id}",
            meta_extra=meta_extra,
        )
        return {"job_id": job_id}

    stdout, stderr, remote_cmd = _run_remote_python(remote_cfg, work_code)
    run_id = None
    for line in stdout.splitlines():
        if line.startswith("REMOTE_RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            break
    if run_id is None:
        raise RuntimeError(
            f"Remote training finished but run_id not captured.\n"
            f"Command: {remote_cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    print(stdout)
    if stderr.strip():
        print(stderr, file=sys.stderr)

    if sync_after:
        sync_from_remote(remote_cfg, include=include, exclude=exclude)
    return run_id


Dataset.train_model_remote = train_model_remote
Dataset.sync_to_remote = lambda self, remote_cfg, include=None, exclude=None: sync_to_remote(remote_cfg, include, exclude)
Dataset.sync_from_remote = lambda self, remote_cfg, include=None, exclude=None: sync_from_remote(remote_cfg, include, exclude)


def run_feature_remote(self,
                       feature_class: str,
                       params: Optional[dict],
                       remote_cfg: dict,
                       groups: Optional[Sequence[str]] = None,
                       sequences: Optional[Sequence[str]] = None,
                       overwrite: bool = False,
                       input_kind: str = "tracks",
                       input_feature: Optional[str] = None,
                       input_run_id: Optional[str] = None,
                       parallel_workers: Optional[int] = None,
                       sync_before: bool = True,
                       sync_after: bool = False,
                       include: Optional[Sequence[str]] = None,
                       exclude: Optional[Sequence[str]] = None,
                       remote_only: bool = True,
                       detached: bool = False) -> str | dict:
    """
    Run Dataset.run_feature on a remote machine via SSH or Jupyter kernel, leaving heavy artifacts remote.
    """
    if sync_before:
        sync_to_remote(remote_cfg, include=include, exclude=exclude)

    local_root = Path(remote_cfg["local_root"]).expanduser().resolve()
    remote_root = Path(remote_cfg["remote_root"])
    manifest_path = Path(self.manifest_path).resolve()
    rel_manifest = _ensure_relative_to(manifest_path, local_root)
    manifest_remote = (remote_root / rel_manifest).as_posix()

    module_path, class_name = feature_class.rsplit(".", 1)
    FeatureCls = getattr(importlib.import_module(module_path), class_name)
    feature_obj = FeatureCls(params or {})
    feature_params = getattr(feature_obj, "params", params or {})
    params_hash = _hash_params(feature_params)
    expected_run_id = f"{feature_obj.version}-{params_hash}"
    storage_feature_name = getattr(feature_obj, "storage_feature_name", feature_obj.name)
    use_input_suffix = getattr(feature_obj, "storage_use_input_suffix", True)
    if input_kind in {"feature", "inputset"} and input_feature and use_input_suffix:
        storage_feature_name = f"{storage_feature_name}__from__{input_feature}"

    if remote_only:
        pattern = f"features/{storage_feature_name}/{expected_run_id}"
        _record_remote_only_pattern(remote_cfg, pattern)

    params_json = json.dumps(params or {})
    groups_json = json.dumps(list(groups) if groups else None)
    sequences_json = json.dumps(list(sequences) if sequences else None)
    root_map = remote_cfg.get("root_map") or {}
    root_map_json = json.dumps(root_map)
    remote_cwd = remote_root.as_posix()
    overwrite_literal = "True" if overwrite else "False"
    input_feature_literal = repr(input_feature)
    input_run_id_literal = repr(input_run_id)
    parallel_literal = "None" if parallel_workers is None else str(int(parallel_workers))
    work_code = f"""
import os, sys, json, importlib
os.chdir(r"{remote_cwd}")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
src_dir = os.path.join(cwd, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from behavior.dataset import Dataset
manifest_path = r"{manifest_remote}"
root_map = json.loads({root_map_json!r})
module_path, class_name = "{feature_class}".rsplit(".", 1)
FeatureCls = getattr(importlib.import_module(module_path), class_name)
params = json.loads({params_json!r})
groups = json.loads({groups_json!r})
sequences = json.loads({sequences_json!r})
dataset = Dataset(manifest_path).load(ensure_roots=not bool(root_map))
if root_map:
    dataset.remap_roots(root_map)
    dataset.ensure_roots()
feature = FeatureCls(params)
run_id = dataset.run_feature(
    feature,
    groups=groups,
    sequences=sequences,
    overwrite={overwrite_literal},
    input_kind={input_kind!r},
    input_feature={input_feature_literal},
    input_run_id={input_run_id_literal},
    parallel_workers={parallel_literal}
)
print("REMOTE_FEATURE_RUN_ID=" + run_id)
print("REMOTE_STORAGE_FEATURE=" + {json.dumps(storage_feature_name)})
"""

    if detached:
        meta_extra = {
            "job_type": "run_feature",
            "feature_class": feature_class,
            "storage_feature": storage_feature_name,
            "remote_only": bool(remote_only),
        }
        job_id = _submit_remote_detached_job(
            remote_cfg,
            job_kind="run_feature",
            work_code=work_code,
            result_expr="{'run_id': run_id}",
            meta_extra=meta_extra,
        )
        return {"job_id": job_id}

    stdout, stderr, remote_cmd = _run_remote_python(remote_cfg, work_code)
    run_id = None
    storage_remote = storage_feature_name
    for line in stdout.splitlines():
        if line.startswith("REMOTE_FEATURE_RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
        elif line.startswith("REMOTE_STORAGE_FEATURE="):
            storage_remote = line.split("=", 1)[1].strip()
    if run_id is None:
        raise RuntimeError(
            f"Remote feature run finished but run_id not captured.\n"
            f"Command: {remote_cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if run_id != expected_run_id:
        raise RuntimeError(
            f"Remote feature run_id mismatch: expected {expected_run_id}, got {run_id}."
        )
    print(stdout)
    if stderr.strip():
        print(stderr, file=sys.stderr)

    if remote_only and not detached:
        path = f"features/{storage_remote}/{run_id}"
        _record_remote_only_pattern(remote_cfg, path)

    if sync_after and not remote_only:
        sync_from_remote(remote_cfg, include=include, exclude=exclude)
    return run_id


Dataset.run_feature_remote = run_feature_remote


def _fetch_remote_jobs(remote_cfg: dict, job_id: Optional[str] = None) -> list[dict] | dict | None:
    cfg = _validate_remote_cfg(remote_cfg)
    remote_root = cfg["remote_root"]
    if job_id:
        remote_script = f"""
import os, json
meta_path = os.path.join(r"{remote_root}", ".remote_jobs", "meta", "{job_id}.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as fh:
        print(json.dumps(json.load(fh)))
else:
    print("")
"""
        stdout, _, _ = _run_remote_python(remote_cfg, remote_script)
        payload = stdout.strip()
        if not payload:
            return None
        return json.loads(payload)

    remote_script = f"""
import os, json, glob
meta_dir = os.path.join(r"{remote_root}", ".remote_jobs", "meta")
jobs = []
if os.path.isdir(meta_dir):
    for path in sorted(glob.glob(os.path.join(meta_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        jobs.append(data)
print(json.dumps(jobs))
"""
    stdout, _, _ = _run_remote_python(remote_cfg, remote_script)
    payload = stdout.strip()
    if not payload:
        return []
    return json.loads(payload)


def _refresh_remote_only_patterns(remote_cfg: dict) -> None:
    jobs = _fetch_remote_jobs(remote_cfg) or []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if str(job.get("status", "")).lower() == "finished" and job.get("remote_only"):
            _maybe_apply_remote_only_pattern(remote_cfg, job)


def list_remote_jobs(self,
                     remote_cfg: dict,
                     job_kind: Optional[str] = None,
                     status: Optional[str] = None) -> list[dict]:
    jobs = _fetch_remote_jobs(remote_cfg) or []
    job_kind = job_kind.lower() if job_kind else None
    status = status.lower() if status else None
    filtered = []
    for job in jobs:
        jk = str(job.get("job_kind", "")).lower()
        st = str(job.get("status", "")).lower()
        if job_kind and jk != job_kind:
            continue
        if status and st != status:
            continue
        _maybe_apply_remote_only_pattern(remote_cfg, job if st == "finished" else {})
        filtered.append(job)
    return filtered


def remote_job_status(self, remote_cfg: dict, job_id: str) -> Optional[dict]:
    job = _fetch_remote_jobs(remote_cfg, job_id=job_id)
    if isinstance(job, dict):
        if job.get("status") == "finished":
            _maybe_apply_remote_only_pattern(remote_cfg, job)
        return job
    return None


Dataset.list_remote_jobs = list_remote_jobs
Dataset.remote_job_status = remote_job_status


def _stop_remote_job(remote_cfg: dict, job_id: str, sig: str = "TERM") -> Optional[dict]:
    job = _fetch_remote_jobs(remote_cfg, job_id=job_id)
    if not isinstance(job, dict):
        return None
    pid = job.get("pid")
    if not pid:
        return job
    cfg = _validate_remote_cfg(remote_cfg)
    remote_root = cfg["remote_root"]
    remote_cfg_ssh = dict(remote_cfg)
    remote_cfg_ssh.pop("jupyter_url", None)

    stop_script = f"""
import os, signal, json
meta_path = os.path.join(r"{remote_root}", ".remote_jobs", "meta", "{job_id}.json")
pid = {int(pid)}
try:
    os.kill(pid, signal.SIG{sig})
    status = "cancelled"
except ProcessLookupError:
    status = "finished"
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        meta = {{}}
    meta["status"] = status
    meta["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
"""
    _run_remote_python(remote_cfg_ssh, stop_script)
    return _fetch_remote_jobs(remote_cfg, job_id=job_id)


def remote_job_stop(self, remote_cfg: dict, job_id: str, sig: str = "TERM") -> Optional[dict]:
    return _stop_remote_job(remote_cfg, job_id, sig=sig.upper())


Dataset.remote_job_stop = remote_job_stop
