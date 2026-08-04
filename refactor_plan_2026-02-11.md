# Refactoring Plan: behavior → mosaic-aligned module structure

**Date**: 2026-02-11
**Goal**: Create `media` and `tracking` sub-packages within the existing `behavior` package as a stepping stone toward the eventual mosaic monorepo split. Custom pose model training will live under `tracking`.

---

## Current structure

```
src/behavior/
    __init__.py              # Exports Dataset, register_feature, helpers; imports sub-libraries
    dataset.py               # ~5K lines — Dataset class, registries, converters, media indexing, track loading
    helpers.py               # to_safe_name, from_safe_name, filter_time_range, label utilities
    models_behavior.py       # (legacy duplicate of model_library code)
    analysis.py              # Cluster/label agreement metrics
    visualization.py         # Track visualization
    feature_library/         # 20+ registered features (per-frame, global, viz, transform)
    label_library/           # 3 label converters (CalMS21, BORIS CSV, BORIS pickle)
    model_library/           # XGBoost behavior classifier + helpers
```

Everything funnels through `dataset.py` which contains media indexing, track schema, track converters, label converters, feature orchestration, and the Dataset class.

---

## Target structure (this PR)

```
src/behavior/
    __init__.py              # Updated: also imports media, tracking
    dataset.py               # Slimmed: delegates media/track ops to new sub-packages
    helpers.py               # Unchanged
    analysis.py              # Unchanged
    visualization.py         # Unchanged

    media/                   # NEW — video/media utilities
        __init__.py          # Imports Dataset for type hints; re-exports media functions
        indexing.py           # index_media(), resolve_media_path(), _probe_video_metadata(), _parse_ffprobe_rate()

    tracking/                # NEW — track schema, converters, and pose model training
        __init__.py          # Imports Dataset for type hints; re-exports tracking functions
        schema.py            # TrackSchema dataclass, TRACK_SCHEMAS registry, register_track_schema(), ensure_track_schema(), trex_v1 definition
        converters/          # Track format converters
            __init__.py
            trex.py          # T-Rex NPZ converter
            calms21.py       # CalMS21 NPY/JSON converter + helpers (angle_from_two_points, angle_from_pca, etc.)
        pose_training/       # NEW — custom pose model training (main deliverable)
            __init__.py
            (future: datasets.py, models/, train.py, evaluate.py, inference.py, export_tracks.py)

    feature_library/         # Unchanged
    label_library/           # Unchanged
    model_library/           # Unchanged
```

---

## Step-by-step plan

### Step 1: Create `tracking/schema.py`

Extract from `dataset.py` lines 198–256:
- `TrackSchema` dataclass
- `TRACK_SCHEMAS` dict
- `register_track_schema()` function
- `ensure_track_schema()` function
- `trex_v1` schema registration

In `dataset.py`, replace with:
```python
from .tracking.schema import TrackSchema, TRACK_SCHEMAS, register_track_schema, ensure_track_schema
```

This preserves backward compat — anything importing from `behavior.dataset` still works.

**Files modified**: `dataset.py`
**Files created**: `tracking/__init__.py`, `tracking/schema.py`

### Step 2: Create `tracking/converters/`

Extract from `dataset.py`:
- `_trex_npz_converter()` → `tracking/converters/trex.py`
- `load_calms21()`, `_calms21_seq_to_trex_df()`, `_strip_trex_seq()`, `angle_from_two_points()`, `angle_from_pca()`, CalMS21 JSON/NPY converters → `tracking/converters/calms21.py`
- `TRACK_CONVERTERS` dict, `register_track_converter()` → `tracking/converters/__init__.py`

In `dataset.py`, replace with imports from the new locations. `convert_one_track()`, `convert_all_tracks()`, `load_tracks()`, `index_tracks_raw()` stay in `dataset.py` for now (they're Dataset methods that orchestrate converters).

**Files modified**: `dataset.py`
**Files created**: `tracking/converters/__init__.py`, `tracking/converters/trex.py`, `tracking/converters/calms21.py`

### Step 3: Create `media/`

Extract from `dataset.py`:
- `_probe_video_metadata()` (line 23)
- `_parse_ffprobe_rate()` (line 58)
- Media-related helpers: `_normalize_patterns()`, `_build_media_sequence_keymap()`

The Dataset methods `index_media()` and `resolve_media_path()` stay on Dataset for now but call into `media.indexing` for the heavy lifting.

**Files modified**: `dataset.py`
**Files created**: `media/__init__.py`, `media/indexing.py`

### Step 4: Create `tracking/pose_training/` scaffold

Create the empty package structure for custom pose model training:
```
tracking/pose_training/
    __init__.py          # Package docstring, future public API
```

This is where the actual pose model training implementation will go next.

### Step 5: Update `__init__.py`

Add imports:
```python
from . import media
from . import tracking
```

Update `__all__` to include `media` and `tracking`.

### Step 6: Update `pyproject.toml`

Add optional dependency group for pose training:
```toml
[project.optional-dependencies]
wavelets = ["PyWavelets>=1.4"]
pose = ["torch>=2.0", "torchvision>=0.15", "albumentations>=1.3"]
```

---

## What stays in `dataset.py`

The Dataset class methods that *orchestrate* media/tracking operations stay put:
- `index_media()`, `resolve_media_path()` — call into `media.indexing`
- `index_tracks_raw()`, `convert_one_track()`, `convert_all_tracks()`, `load_tracks()` — call into `tracking.converters`
- All feature/model/label orchestration — unchanged

This is intentional: Dataset remains the user-facing API. The sub-packages are implementation modules. Moving the Dataset methods themselves is Phase 2 (the full mosaic split, later).

---

## What does NOT change

- `feature_library/` — no modifications
- `label_library/` — no modifications
- `model_library/` — no modifications
- `helpers.py` — no modifications
- `analysis.py` — no modifications
- `visualization.py` — no modifications
- All existing imports like `from behavior.dataset import FEATURES, register_feature` — still work

---

## Backward compatibility

All existing public imports continue to work via re-exports in `dataset.py`. The refactoring is purely additive — code that works today will work after this change.

---

## Verification

1. `pip install -e .` succeeds
2. `python -c "from behavior import Dataset; print('OK')"` works
3. `python -c "from behavior.tracking.schema import TrackSchema, TRACK_SCHEMAS; print(TRACK_SCHEMAS)"` shows trex_v1
4. `python -c "from behavior.tracking.converters import TRACK_CONVERTERS; print(list(TRACK_CONVERTERS.keys()))"` shows registered converters
5. `python -c "from behavior.media.indexing import probe_video_metadata; print('OK')"` works
6. `python -c "from behavior.dataset import TrackSchema, TRACK_SCHEMAS, ensure_track_schema; print('backward compat OK')"` — re-exports work
7. Run the calms21-template notebook end-to-end (if dataset available) to verify nothing broke
