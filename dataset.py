# dataset.py
from __future__ import annotations
from pathlib import Path
import pandas as pd 
import numpy as np
import math  # used by _norm_hint
import uuid, datetime
import yaml  # pip install pyyaml

import csv, json, os, sys, re
import fnmatch
from helpers import to_safe_name, from_safe_name

from typing import Protocol, Iterable, Optional
from dataclasses import dataclass
import json, joblib, hashlib, time

def _normalize_patterns(pats) -> tuple[str, ...]:
    if pats is None:
        return tuple()
    if isinstance(pats, str):
        return (pats,)
    try:
        return tuple(pats)
    except TypeError:
        return (str(pats),)
from dataclasses import dataclass, field
from typing import Iterable, Dict, Any, Tuple, Callable, Optional
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

    # ---- Instance load method ----
    def load(self) -> "Dataset":
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
        - Columns: name, abs_path, size_bytes, mtime_iso
        """
        media_root = self.get_root("media")
        out_csv = media_root / index_filename
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

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
                        rows.append({
                            "name": p.name,
                            "abs_path": str(p.resolve()),
                            "size_bytes": st.st_size,
                            "mtime_iso": _to_iso(st.st_mtime),
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
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "abs_path", "size_bytes", "mtime_iso"])
            w.writeheader()
            w.writerows(dedup)

        print(f"[index_media] Wrote {len(dedup)} entries -> {out_csv}")
        return out_csv
    
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
                     sequences: Optional[Iterable[str]] = None):
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

    for _, row in df_idx[mask].iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        p = Path(row["abs_path"])
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
                          sequences: Optional[Iterable[str]] = None):
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

    for _, row in df_sel.iterrows():
        g, s = str(row["group"]), str(row["sequence"])
        p = Path(row["abs_path"])
        if not p.exists():
            print(f"[feature-input] missing parquet for ({g},{s}) -> {p}", file=sys.stderr)
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[feature-input] failed to read {p}: {e}", file=sys.stderr)
            continue
        yield g, s, df

# ----------------------------
# Feature runner on Dataset
# ----------------------------
def _feature_run_root(ds: "Dataset", feature_name: str, run_id: str) -> Path:
    return ds.get_root("features") / feature_name / run_id

def _feature_index_path(ds: "Dataset", feature_name: str) -> Path:
    return ds.get_root("features") / feature_name / "index.csv"

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

def _append_feature_index(idx_path: Path, rows: list[dict]):
    if not idx_path.exists():
        _ensure_feature_index(idx_path)
    df = pd.read_csv(idx_path)
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

def _update_finished_times(idx_path: Path, run_id: str, finished_at: str):
    df = pd.read_csv(idx_path)
    if "finished_at" not in df.columns:
        df["finished_at"] = ""
    sel = (df["run_id"] == run_id) & ((df["finished_at"].isna()) | (df["finished_at"] == ""))
    if sel.any():
        df.loc[sel, "finished_at"] = str(finished_at)
        df.to_csv(idx_path, index=False)

def run_feature(self,
                feature: Feature,
                groups: Optional[Iterable[str]] = None,
                sequences: Optional[Iterable[str]] = None,
                overwrite: bool = False,
                input_kind: str = "tracks",
                input_feature: Optional[str] = None,
                input_run_id: Optional[str] = None) -> str:
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
    # Prepare run id & root
    params_hash = _hash_params(getattr(feature, "params", {}))
    run_id = f"{feature.version}-{params_hash}"
    run_root = _feature_run_root(self, feature.name, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    idx_path = _feature_index_path(self, feature.name)
    _ensure_feature_index(idx_path)
    started = _now_iso()

    # Choose input iterator
    if input_kind not in {"tracks", "feature"}:
        raise ValueError("input_kind must be 'tracks' or 'feature'")
    if input_kind == "feature":
        if not input_feature:
            raise ValueError("input_feature must be provided when input_kind='feature'")
        iter_inputs = lambda: _yield_feature_frames(self, input_feature, input_run_id, groups, sequences)
    else:
        iter_inputs = lambda: _yield_sequences(self, groups, sequences)

    # ===== FIT PHASE =====
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
            if all_dfs:
                feature.fit(all_dfs)

        # Save model state if any
        model_path = run_root / "model.joblib"
        try:
            feature.save_model(model_path)
        except NotImplementedError:
            pass

    # ===== TRANSFORM PHASE =====
    out_rows = []
    for g, s, df in iter_inputs():
        safe_group = to_safe_name(g) if g else ""
        safe_seq = to_safe_name(s)
        out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
        out_path = run_root / out_name
        if out_path.exists() and not overwrite:
            # still record to index with current run_id
            try:
                n_rows = int(pd.read_parquet(out_path).shape[0])
            except Exception:
                n_rows = None
            out_rows.append({
                "feature": feature.name, "version": feature.version, "run_id": run_id,
                "group": g, "sequence": s, "group_safe": safe_group, "sequence_safe": safe_seq,
                "abs_path": str(out_path.resolve()),
                "n_rows": n_rows, "params_hash": params_hash,
                "started_at": started, "finished_at": ""
            })
            continue

        try:
            df_feat = feature.transform(df)
        except Exception as e:
            print(f"[feature:{feature.name}] transform failed for ({g},{s}): {e}", file=sys.stderr)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_parquet(out_path, index=False)
        out_rows.append({
            "feature": feature.name, "version": feature.version, "run_id": run_id,
            "group": g, "sequence": s, "group_safe": safe_group, "sequence_safe": safe_seq,
            "abs_path": str(out_path.resolve()),
            "n_rows": int(len(df_feat)), "params_hash": params_hash,
            "started_at": started, "finished_at": ""
        })

    _append_feature_index(idx_path, out_rows)
    _update_finished_times(idx_path, run_id, _now_iso())
    print(f"[feature:{feature.name}] completed run_id={run_id} -> {run_root}")
    return run_id

# Attach to class
Dataset.run_feature = run_feature