# Label Library - Creating Custom Label Converters

This guide shows you how to create custom label converters for your specific
annotation formats.

A label converter now returns **data, not files**. Its `convert()` yields one
`LabelEntry` per sequence, and the `Dataset` writes the `.npz` files into
`labels/<kind>/<run_id>/` and records the typed index row. A converter never
computes an output path, writes an `.npz`, or handles overwrite.

## Quick Start

**5-Minute Custom Converter:**

1. Copy the template:
   ```bash
   cp custom_label_template.py my_converter.py
   ```

2. Edit 3 methods in `my_converter.py`:
   - `_load_source_file()` - Load your file
   - `_build_label_map()` - Define behavior names
   - `_extract_annotations()` - Extract events

3. Register it in `__init__.py`:
   ```python
   from mosaic.core.label_converter import register_label_converter
   from . import my_converter

   my_converter.MyConverter = register_label_converter(
       my_converter.MyConverter
   )
   ```

4. Index the raw files, then convert:
   ```python
   dataset.index_labels_raw(["/path/to/annotations"], patterns="*.csv",
                            src_format="my_custom_format")
   dataset.convert_all_labels(kind="behavior", source_format="my_custom_format")
   ```

## What is a Label Converter?

A label converter transforms annotation files into the standardized behavior
dataset format:

**Input:** Your annotation file (CSV, JSON, Excel, etc.)
```
video_001.mp4, 0.0, 5.2, grooming
video_001.mp4, 10.5, 15.0, eating
video_001.mp4, 20.0, 20.0, jump  # Point event
```

**Output:** One `LabelEntry` per sequence, carrying the `.npz` payload:
```python
LabelEntry(
    group="",
    sequence="video_001",
    payload={
        "frames": [0, 1, 2, ..., 599],     # Frame indices
        "labels": [0, 1, 1, ..., 0],       # Per-frame behavior IDs
        "label_names": ["none", "grooming", "eating", "jump"],
        "fps": 30.0,
    },
    n_frames=600,
    label_ids=(0, 1, 2, 3),
    label_names=("none", "grooming", "eating", "jump"),
)
```

The `Dataset` writes `payload` into `labels/behavior/<run_id>/__video_001.npz`
and records the typed index row for you.

## The Converter Contract

A converter subclasses `LabelConverter[MyParams]` from
`mosaic.core.label_converter`, declares its class variables, and implements
`convert()`:

```python
from collections.abc import Mapping
from pathlib import Path

from mosaic.core.label_converter import (
    LabelConvertParams, LabelConverter, LabelEntry,
)


class MyParams(LabelConvertParams):
    fps: float = 30.0            # Hashed: changes the labels.


class MyConverter(LabelConverter[MyParams]):
    src_format = "my_format"     # Matches labels_raw/index.csv src_format.
    label_kind = "behavior"      # The labels/<kind>/ subdirectory.
    label_format = "my_format_v1"  # Payload format a reader dispatches on.
    version = "0.1"              # Declared compatibility version.
    Params = MyParams

    def convert(self, src_path: Path, params: MyParams,
                raw_row: Mapping[str, object]) -> list[LabelEntry]:
        ...
        return [LabelEntry(group=..., sequence=..., payload=..., n_frames=...,
                           label_ids=..., label_names=...)]
```

### Required Class Attributes

```python
class MyConverter(LabelConverter[MyParams]):
    src_format = "my_format"       # Passed as source_format to convert_all_labels.
    label_kind = "behavior"        # Usually "behavior" or "id_tags".
    label_format = "my_format_v1"  # On-disk payload format.
    version = "0.1"                # Bump by hand when output semantics change.
    Params = MyParams              # The typed parameter model.
```

`(src_format, label_kind)` is the registry key -- one raw format can feed two
kinds. `version` is a **declared** compatibility version and a visible segment of
the label variant identity; bump it by hand when the output semantics change, so
a bit-identical conversion keeps its identity across an unrelated release.

### Typed Parameters

Parameters are a Pydantic model subclassing `LabelConvertParams`. There is no
`__init__` and no `_defaults` dict:

```python
from typing import Annotated
from mosaic.core.label_converter import LabelConvertParams
from mosaic.core.pipeline.types import HASH_EXCLUDE


class MyParams(LabelConvertParams):
    # group_from and strict_schema are ALREADY on LabelConvertParams -- do not
    # redeclare them.

    fps: float = 30.0                          # Hashed: changes the labels.
    background_label: str = "none"             # Hashed.

    verbose: Annotated[bool, HASH_EXCLUDE] = False  # Excluded: a throughput knob.
```

- **Plain fields are hashed.** Everything that changes the labels is part of the
  label variant identity, so two conversions differing in it mint two variants.
- **`HASH_EXCLUDE` fields are not.** A validation-only or throughput-only knob is
  tagged `Annotated[T, HASH_EXCLUDE]`, so retuning it never busts the cache. It
  still appears in `params.json` and reaches the converter.
- **`group_from` and `strict_schema` live on the base** and are already excluded
  from identity -- `group_from` is entry policy (which group string to assign),
  `strict_schema` is validation strictness. Do not redeclare them.

### `convert()` Returns `list[LabelEntry]`

`convert(self, src_path, params, raw_row)` returns one `LabelEntry` per
sequence. It never writes files. Each `LabelEntry` carries:

- `group`, `sequence` - the composite identity for this sequence.
- `payload` - the exact dict of arrays and scalars a reader loads from the
  `.npz` (the dict the old contract passed to `np.savez_compressed`).
- `n_frames` - the sequence length, recorded on the index row.
- `label_ids`, `label_names` - the kind's vocabulary, recorded on the index row
  so a listing need not open every file.

`raw_row` is the `labels_raw/index.csv` row (the group hint and the source
`md5`). It is never hashed.

## Required Methods to Implement

When you subclass `CustomLabelConverter` (from `custom_label_template.py`), the
base `convert()` handles the flow and you implement three methods.

#### 1. `_load_source_file(src_path)` - Load Your File

Load your annotation file in whatever format it is.

**Examples:**

```python
# CSV
def _load_source_file(self, src_path: Path):
    return pd.read_csv(src_path)

# JSON
def _load_source_file(self, src_path: Path):
    import json
    with open(src_path) as f:
        return json.load(f)

# Excel
def _load_source_file(self, src_path: Path):
    return pd.read_excel(src_path, sheet_name="Annotations")
```

#### 2. `_build_label_map(data, params)` - Define Behaviors

Map behavior IDs to behavior names.

**Examples:**

```python
# From a DataFrame column
def _build_label_map(self, data, params):
    behaviors = sorted(data["behavior"].unique())
    behaviors = [params.background_label] + behaviors
    return {i: name for i, name in enumerate(behaviors)}

# Hardcoded
def _build_label_map(self, data, params):
    return {0: "none", 1: "grooming", 2: "eating", 3: "resting"}
```

**Important:** MUST include `params.background_label` (usually at ID 0). Read
parameters as typed attributes (`params.background_label`), not dict lookups.

#### 3. `_extract_annotations(data, src_path, raw_row, params)` - Extract Events

This is the **KEY METHOD**. Extract your annotations into a standard structure.

**Return format:**
```python
[
    {
        "sequence_name": "video_001",           # REQUIRED: Unique name
        "annotations": [                        # REQUIRED: List of events
            (0.0, 5.2, "grooming"),            # (start, stop, behavior)
            (10.5, 15.0, "eating"),
            (20.0, 20.0, "jump"),              # Point event: start == stop
        ],
        "fps": 30.0,                           # OPTIONAL: Override default FPS
        "metadata": {"animal_id": "A12"},      # OPTIONAL: Extra metadata
    }
]
```

**Examples:**

```python
# Simple: Single sequence from DataFrame
def _extract_annotations(self, data, src_path, raw_row, params):
    annotations = [
        (row["start"], row["stop"], row["behavior"])
        for _, row in data.iterrows()
    ]
    return [{
        "sequence_name": "recording_001",
        "annotations": annotations,
        "fps": params.fps,
    }]

# Multiple videos
def _extract_annotations(self, data, src_path, raw_row, params):
    sequences = []
    for video_name, group_df in data.groupby("video"):
        annotations = [
            (row["start"], row["stop"], row["behavior"])
            for _, row in group_df.iterrows()
        ]
        sequences.append({
            "sequence_name": video_name,
            "annotations": annotations,
            "fps": group_df["fps"].iloc[0],
        })
    return sequences
```

## Common Use Cases

### Use Case 1: Simple CSV Annotations

**File format:**
```csv
time_start,time_end,behavior
0.5,3.2,grooming
5.0,8.5,eating
10.0,10.0,bite
```

**Converter** (uses only the base parameters, so no `Params` subclass is needed):
```python
class SimpleCSVConverter(CustomLabelConverter):
    src_format = "simple_csv"
    label_kind = "behavior"
    label_format = "simple_csv_v1"
    version = "0.1"

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        behaviors = ["none"] + sorted(data["behavior"].unique())
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self, data, src_path, raw_row, params):
        annotations = [
            (row["time_start"], row["time_end"], row["behavior"])
            for _, row in data.iterrows()
        ]
        return [{
            "sequence_name": src_path.stem,
            "annotations": annotations,
        }]
```

### Use Case 2: Video Snippets with Time Offsets

**Problem:** You annotated video snippets in BORIS, but need to add:
- Time offset (when the snippet starts in the full video)
- Animal ID (not recorded during annotation)

**Solution** (extra parameters go in a typed `Params` subclass):

```python
class SnippetBorisParams(CustomLabelParams):
    time_offset: float = 0.0   # Hashed: shifts every timestamp.
    animal_id: str = "unknown" # Hashed: names the sequence.


class SnippetBorisConverter(CustomLabelConverter):
    src_format = "snippet_boris"
    label_kind = "behavior"
    label_format = "snippet_boris_v1"
    version = "0.1"
    Params = SnippetBorisParams

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        behaviors = sorted(data["Behavior"].unique())
        return {0: "none", **{i + 1: b for i, b in enumerate(behaviors)}}

    def _extract_annotations(self, data, src_path, raw_row, params):
        offset = params.time_offset
        animal_id = params.animal_id

        annotations = []
        for _, row in data.iterrows():
            start = row["Start (s)"] + offset
            stop = row["Stop (s)"] + offset
            annotations.append((start, stop, row["Behavior"]))

        return [{
            "sequence_name": f"{src_path.stem}_{animal_id}",
            "annotations": annotations,
            "metadata": {"animal_id": animal_id, "time_offset": offset},
        }]
```

**Usage:**
```python
# Convert a snippet that starts at 2:30 in the full video, animal A12.
dataset.convert_all_labels(
    source_format="snippet_boris",
    time_offset=150.0,      # 2:30 = 150 seconds
    animal_id="mouse_A12",
    fps=30.0,
)
```

### Use Case 3: Multiple Animals per File

**File format:**
```csv
time,animal_id,behavior
0.5,mouse1,grooming
0.5,mouse2,eating
3.0,mouse1,resting
```

**Converter:**
```python
class MultiAnimalConverter(CustomLabelConverter):
    src_format = "multi_animal"
    label_kind = "behavior"
    label_format = "multi_animal_v1"
    version = "0.1"

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        behaviors = ["none"] + sorted(data["behavior"].unique())
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self, data, src_path, raw_row, params):
        sequences = []
        for animal_id, animal_df in data.groupby("animal_id"):
            annotations = [
                (row["time"], row["time"], row["behavior"])  # Point events
                for _, row in animal_df.iterrows()
            ]
            sequences.append({
                "sequence_name": f"{src_path.stem}_{animal_id}",
                "annotations": annotations,
                "metadata": {"animal_id": animal_id},
            })
        return sequences
```

### Use Case 4: JSON with Nested Structure

**File format:**
```json
{
    "session_id": "exp_001",
    "fps": 30,
    "animals": {
        "mouse1": {"behaviors": [{"start": 0.5, "end": 3.2, "type": "grooming"}]},
        "mouse2": {"behaviors": [{"start": 1.0, "end": 4.0, "type": "resting"}]}
    }
}
```

**Converter:**
```python
class NestedJSONConverter(CustomLabelConverter):
    src_format = "nested_json"
    label_kind = "behavior"
    label_format = "nested_json_v1"
    version = "0.1"

    def _load_source_file(self, src_path):
        import json
        with open(src_path) as f:
            return json.load(f)

    def _build_label_map(self, data, params):
        behaviors = set()
        for animal_data in data["animals"].values():
            for event in animal_data["behaviors"]:
                behaviors.add(event["type"])
        behaviors = ["none"] + sorted(behaviors)
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self, data, src_path, raw_row, params):
        sequences = []
        session_id = data["session_id"]
        fps = data.get("fps", params.fps)
        for animal_id, animal_data in data["animals"].items():
            annotations = [
                (event["start"], event["end"], event["type"])
                for event in animal_data["behaviors"]
            ]
            sequences.append({
                "sequence_name": f"{session_id}_{animal_id}",
                "annotations": annotations,
                "fps": fps,
                "metadata": {"session_id": session_id, "animal_id": animal_id},
            })
        return sequences
```

## Custom Parameters

Add custom parameters as typed fields on a `Params` subclass and read them as
attributes. Plain fields change the labels and are hashed; a knob that does not
change the labels is tagged `HASH_EXCLUDE`:

```python
from typing import Annotated
from mosaic.core.pipeline.types import HASH_EXCLUDE


class MyParams(CustomLabelParams):
    time_offset: float = 0.0            # Hashed.
    scale_factor: float = 1.0           # Hashed.
    ignore_behaviors: list[str] = []    # Hashed: changes which events survive.
    verbose: Annotated[bool, HASH_EXCLUDE] = False  # Excluded.


class MyConverter(CustomLabelConverter):
    src_format = "my_format"
    label_kind = "behavior"
    label_format = "my_format_v1"
    version = "0.1"
    Params = MyParams

    def _extract_annotations(self, data, src_path, raw_row, params):
        annotations = []
        for _, row in data.iterrows():
            behavior = row["behavior"]
            if behavior in params.ignore_behaviors:
                continue
            start = (row["start"] + params.time_offset) * params.scale_factor
            stop = (row["stop"] + params.time_offset) * params.scale_factor
            annotations.append((start, stop, behavior))
        return [{"sequence_name": src_path.stem, "annotations": annotations}]
```

**Usage:**
```python
dataset.convert_all_labels(
    source_format="my_format",
    time_offset=10.0,
    scale_factor=1.5,
    ignore_behaviors=["artifact", "unclear"],
)
```

## Loading Animal IDs from External File

**Scenario:** Video snippet filenames map to animal IDs in a separate file.

**Mapping file (animal_mapping.csv):**
```csv
video_file,animal_id,date
snippet_001.mp4,mouse_A12,2024-01-15
snippet_002.mp4,mouse_B03,2024-01-15
```

**Converter** (read the mapping inside `_extract_annotations`; there is no
`__init__` and no params passed to a constructor under the new contract):
```python
class MappedAnimalParams(CustomLabelParams):
    animal_mapping_file: str | None = None  # Hashed: selects the mapping.


class MappedAnimalConverter(CustomLabelConverter):
    src_format = "mapped_animal"
    label_kind = "behavior"
    label_format = "mapped_animal_v1"
    version = "0.1"
    Params = MappedAnimalParams

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        behaviors = ["none"] + sorted(data["Behavior"].unique())
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self, data, src_path, raw_row, params):
        animal_id, date = "unknown", None
        if params.animal_mapping_file:
            mapping = pd.read_csv(params.animal_mapping_file).set_index("video_file")
            if src_path.name in mapping.index:
                animal_id = mapping.loc[src_path.name, "animal_id"]
                date = mapping.loc[src_path.name, "date"]

        annotations = [
            (row["Start (s)"], row["Stop (s)"], row["Behavior"])
            for _, row in data.iterrows()
        ]
        metadata = {"animal_id": animal_id}
        if date:
            metadata["date"] = date

        return [{
            "sequence_name": f"{date}_{animal_id}_{src_path.name}",
            "annotations": annotations,
            "metadata": metadata,
        }]
```

**Usage:**
```python
dataset.convert_all_labels(
    source_format="mapped_animal",
    animal_mapping_file="/path/to/animal_mapping.csv",
)
```

## Registration and Usage

### 1. Register Your Converter

`register_label_converter` moved out of `dataset.py` (to break the
converter/dataset import cycle, as tracks did). Import it from
`mosaic.core.label_converter` and wire it in [\_\_init\_\_.py](__init__.py):

```python
# Add import (the registry now lives here, not in dataset.py)
from mosaic.core.label_converter import register_label_converter
from . import my_converter

# Add registration
my_converter.MyConverter = register_label_converter(
    my_converter.MyConverter
)

# Add to __all__
__all__ = [
    "calms21_behavior",
    "boris_aggregated_csv",
    "boris_pandas_pickle",
    "my_converter",  # Add this
]
```

### 2. Index Source Files

Raw label files are indexed into `labels_raw/index.csv` with `index_labels_raw`
(the label sibling of `index_tracks_raw` -- raw labels no longer share the
`tracks_raw` root):

```python
from mosaic.core import Dataset

dataset = Dataset("/path/to/dataset")

# Scan a directory for annotation files and record them in labels_raw.
dataset.index_labels_raw(
    ["/path/to/annotations"],       # Directories to scan.
    patterns="*.csv",
    src_format="my_custom_format",  # Must match the converter's src_format.
)
```

### 3. Convert Labels

`convert_all_labels` reads `labels_raw`, dispatches to your converter, and
writes the outputs:

```python
dataset.convert_all_labels(
    kind="behavior",
    source_format="my_custom_format",  # Must match the converter's src_format.
    # Any Params field can be passed as a keyword:
    time_offset=10.0,
    animal_id="mouse_A12",
)
```

### 4. Verify Output

The labels index is a typed index (`labels/<kind>/index.csv`) with a `run_id`
column, and the `.npz` files live in variant subdirectories
`labels/<kind>/<run_id>/`. Read it with `read_labels_index`, and resolve the
root-relative `abs_path` with `dataset.resolve_path`:

```python
import numpy as np
from mosaic.core.pipeline.labels_index import read_labels_index

# Check the typed index (one row per variant x group x sequence).
df = read_labels_index(dataset, "behavior")
print(df[["run_id", "group", "sequence", "n_frames", "label_names"]])

# Load a sequence's .npz.
row = df.iloc[0]
data = np.load(dataset.resolve_path(row["abs_path"]))
print(f"Frames: {len(data['frames'])}")
print(f"Behaviors: {data['label_names']}")
print(f"Labels shape: {data['labels'].shape}")
```

## Output Format

Each `LabelEntry.payload` is the `.npz` contents the `Dataset` writes.

### Required Fields

```python
{
    "group": str,                   # Group name
    "sequence": str,                # Sequence name
    "sequence_key": str,            # Same as sequence
    "frames": np.ndarray,           # Frame indices [0, 1, 2, ..., n-1]
    "labels": np.ndarray,           # Per-frame behavior IDs (integers)
    "label_ids": np.ndarray,        # Valid behavior IDs [0, 1, 2, ...]
    "label_names": np.ndarray,      # Behavior names ["none", "groom", ...]
    "fps": float,                   # Frames per second
}
```

### Optional Fields

Add custom metadata (the `CustomLabelConverter` base folds a sequence's
`metadata` dict into the payload under a `meta_` prefix):

```python
{
    # ... required fields above ...
    "meta_animal_id": str,          # Your custom fields
    "meta_time_offset": float,
}
```

The `LabelEntry.label_ids` and `LabelEntry.label_names` tuples are recorded on
the typed index row (as comma-joined columns), so a listing need not open every
`.npz`.

## Testing Your Converter

### Step 1: Syntax Check

```bash
python -m py_compile my_converter.py
```

### Step 2: Small Test File

Create a minimal annotation file and run the flow:

```python
from mosaic.core import Dataset

dataset = Dataset("/tmp/test_dataset")

# Index the test file into labels_raw.
dataset.index_labels_raw(
    ["/path/to"], patterns="test_annotation.csv",
    src_format="my_custom_format",
)

# Convert.
dataset.convert_all_labels(source_format="my_custom_format", fps=30.0)

# Verify.
from mosaic.core.pipeline.labels_index import read_labels_index
print(read_labels_index(dataset, "behavior"))
```

### Step 3: Check Output

```python
import numpy as np
from mosaic.core.pipeline.labels_index import read_labels_index

df = read_labels_index(dataset, "behavior")
data = np.load(dataset.resolve_path(df.iloc[0]["abs_path"]))

print("Keys:", list(data.keys()))
print("Frames:", data["frames"][:10])
print("Labels:", data["labels"][:10])
print("Label names:", data["label_names"])
print("FPS:", data["fps"])
```

## Troubleshooting

### Error: "No label converter registered"

Make sure you:
1. Imported your module in `__init__.py`
2. Called `register_label_converter()` on your class (imported from
   `mosaic.core.label_converter`)
3. Restarted Python (to reload the module)

### Error: "KeyError: 'behavior_name'"

Check that:
- Your label map includes all behaviors in the annotations
- Behavior names are spelled consistently
- `params.background_label` is in the label map

### Labels are all zeros

Check that:
- Your annotations have valid start/stop times
- Times are in seconds (not frames)
- FPS is correct
- Start times < stop times for state events

### Wrong number of frames

Check that:
- FPS is correct (30 fps = 30 frames per second)
- Times are in seconds
- The last annotation defines the video length

### A re-conversion did not overwrite

A conversion with different (hashed) params mints a **new** variant beside the
old one under `labels/<kind>/<run_id>/`, rather than overwriting in place. To
replace files within the *same* variant, pass `overwrite=True` to
`convert_all_labels`.

## Best Practices

1. **Keep it simple**: Only implement what you need
2. **Test early**: Test with a small file first
3. **Type your parameters**: Add typed fields with defaults; mark throughput-only
   knobs `HASH_EXCLUDE`
4. **Handle edge cases**: Empty files, missing columns, etc.
5. **Validate inputs**: Check that required columns exist
6. **Use a background label**: Always include a "none"/"other" category
7. **Consistent naming**: Use clear, descriptive sequence names
8. **Bump `version`**: When the output semantics change, so identity tracks it

## Examples in This Directory

- [custom_label_template.py](custom_label_template.py) - Full template with examples
- [label_converter_template.py](label_converter_template.py) - Nested/dense template
- [calms21_behavior.py](calms21_behavior.py) - CalMS21 format (the reference converter)
- [boris_aggregated_csv.py](boris_aggregated_csv.py) - BORIS CSV/TSV
- [boris_pandas_pickle.py](boris_pandas_pickle.py) - BORIS pickle

## Summary Workflow

```python
# 1. Create the converter (my_converter.py)
class MyConverter(CustomLabelConverter):
    src_format = "my_format"
    label_kind = "behavior"
    label_format = "my_format_v1"
    version = "0.1"

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        return {0: "none", 1: "behavior_a", 2: "behavior_b"}

    def _extract_annotations(self, data, src_path, raw_row, params):
        return [{
            "sequence_name": src_path.stem,
            "annotations": [(0.0, 5.0, "behavior_a")],
        }]

# 2. Register in __init__.py
from mosaic.core.label_converter import register_label_converter
from . import my_converter
my_converter.MyConverter = register_label_converter(my_converter.MyConverter)

# 3. Index, then convert
dataset.index_labels_raw(["/path/to/annotations"], patterns="*.csv",
                         src_format="my_format")
dataset.convert_all_labels(source_format="my_format")
```

That's it! You now have a custom label converter that fits your exact needs.
