"""Track schema system for validating standardized track DataFrames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Dict, Iterable

import pandas as pd


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
