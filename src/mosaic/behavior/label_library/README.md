# Label Library - Creating Custom Label Converters

This guide shows you how to create custom label converters for your specific annotation formats.

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
   from . import my_converter
   my_converter.MyConverter = register_label_converter(
       my_converter.MyConverter
   )
   ```

4. Use it:
   ```python
   dataset.convert_all_labels(source_format="my_custom_format")
   ```

## What is a Label Converter?

A label converter transforms annotation files into the standardized behavior dataset format:

**Input:** Your annotation file (CSV, JSON, Excel, etc.)
```
video_001.mp4, 0.0, 5.2, grooming
video_001.mp4, 10.5, 15.0, eating
video_001.mp4, 20.0, 20.0, jump  # Point event
```

**Output:** Per-frame labels in `.npz` files
```python
{
    "frames": [0, 1, 2, ..., 599],     # Frame indices
    "labels": [0, 1, 1, ..., 0],       # Per-frame behavior IDs
    "label_names": ["none", "grooming", "eating", "jump"],
    "fps": 30.0,
}
```

## Template Overview

The [custom_label_template.py](custom_label_template.py) provides a minimal, well-commented starting point.

### Required Class Attributes

```python
class MyConverter(CustomLabelConverter):
    # REQUIRED: Set these
    src_format = "my_format"        # Used in convert_all_labels(source_format="...")
    label_kind = "behavior"         # Usually "behavior"
    label_format = "my_format_v1"   # Version identifier
```

### Required Methods to Implement

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

# Multiple files (CSV + metadata)
def _load_source_file(self, src_path: Path):
    df = pd.read_csv(src_path)
    meta_path = src_path.with_suffix('.json')
    with open(meta_path) as f:
        metadata = json.load(f)
    return {"annotations": df, "metadata": metadata}
```

#### 2. `_build_label_map(data, params)` - Define Behaviors

Map behavior IDs to behavior names.

**Examples:**

```python
# From DataFrame column
def _build_label_map(self, data, params):
    behaviors = sorted(data["behavior"].unique())
    behaviors = [params["background_label"]] + behaviors
    return {i: name for i, name in enumerate(behaviors)}

# From metadata
def _build_label_map(self, data, params):
    behaviors = data["metadata"]["behavior_names"]
    return {i: name for i, name in enumerate(behaviors)}

# Hardcoded
def _build_label_map(self, data, params):
    return {
        0: "none",
        1: "grooming",
        2: "eating",
        3: "resting",
    }
```

**Important:** MUST include `background_label` (usually at ID 0)

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
        "fps": params["fps"],
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

# With time offset and animal ID injection
def _extract_annotations(self, data, src_path, raw_row, params):
    time_offset = params.get("time_offset", 0.0)
    animal_id = params.get("animal_id", "unknown")

    annotations = [
        (row["start"] + time_offset,
         row["stop"] + time_offset,
         row["behavior"])
        for _, row in data.iterrows()
    ]

    return [{
        "sequence_name": f"video_{animal_id}",
        "annotations": annotations,
        "metadata": {
            "animal_id": animal_id,
            "time_offset": time_offset,
        },
    }]
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

**Converter:**
```python
class SimpleCSVConverter(CustomLabelConverter):
    src_format = "simple_csv"
    label_kind = "behavior"
    label_format = "simple_csv_v1"

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

**Solution:**

```python
class SnippetBorisConverter(CustomLabelConverter):
    src_format = "snippet_boris"
    label_kind = "behavior"
    label_format = "snippet_boris_v1"

    _defaults = dict(
        group_from="filename",
        fps=30.0,
        background_label="none",
        time_offset=0.0,       # Add this to all times
        animal_id="unknown",   # Inject this ID
    )

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        behaviors = sorted(data["Behavior"].unique())
        return {0: "none", **{i+1: b for i, b in enumerate(behaviors)}}

    def _extract_annotations(self, data, src_path, raw_row, params):
        offset = params["time_offset"]
        animal_id = params["animal_id"]

        # Extract and adjust times
        annotations = []
        for _, row in data.iterrows():
            start = row["Start (s)"] + offset
            stop = row["Stop (s)"] + offset
            annotations.append((start, stop, row["Behavior"]))

        return [{
            "sequence_name": f"{src_path.stem}_{animal_id}",
            "annotations": annotations,
            "metadata": {
                "animal_id": animal_id,
                "time_offset": offset,
            },
        }]
```

**Usage:**
```python
# Convert snippet 1 (starts at 2:30 in full video, animal A12)
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
5.0,mouse2,grooming
```

**Converter:**
```python
class MultiAnimalConverter(CustomLabelConverter):
    src_format = "multi_animal"
    label_kind = "behavior"
    label_format = "multi_animal_v1"

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        behaviors = ["none"] + sorted(data["behavior"].unique())
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self, data, src_path, raw_row, params):
        sequences = []

        # Create one sequence per animal
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
        "mouse1": {
            "behaviors": [
                {"start": 0.5, "end": 3.2, "type": "grooming"},
                {"start": 5.0, "end": 8.0, "type": "eating"}
            ]
        },
        "mouse2": {
            "behaviors": [
                {"start": 1.0, "end": 4.0, "type": "resting"}
            ]
        }
    }
}
```

**Converter:**
```python
class NestedJSONConverter(CustomLabelConverter):
    src_format = "nested_json"
    label_kind = "behavior"
    label_format = "nested_json_v1"

    def _load_source_file(self, src_path):
        import json
        with open(src_path) as f:
            return json.load(f)

    def _build_label_map(self, data, params):
        # Collect all unique behaviors
        behaviors = set()
        for animal_data in data["animals"].values():
            for event in animal_data["behaviors"]:
                behaviors.add(event["type"])
        behaviors = ["none"] + sorted(behaviors)
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self, data, src_path, raw_row, params):
        sequences = []
        session_id = data["session_id"]
        fps = data.get("fps", params["fps"])

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

Add custom parameters in `_defaults` and use them in your methods:

```python
class MyConverter(CustomLabelConverter):
    _defaults = dict(
        group_from="filename",
        fps=30.0,
        background_label="none",
        # Your custom parameters:
        time_offset=0.0,
        scale_factor=1.0,
        animal_id_file=None,
        ignore_behaviors=None,  # List of behaviors to skip
    )

    def _extract_annotations(self, data, src_path, raw_row, params):
        offset = params["time_offset"]
        scale = params["scale_factor"]
        ignore = params.get("ignore_behaviors") or []

        annotations = []
        for _, row in data.iterrows():
            behavior = row["behavior"]
            if behavior in ignore:
                continue

            start = (row["start"] + offset) * scale
            stop = (row["stop"] + offset) * scale
            annotations.append((start, stop, behavior))

        return [{
            "sequence_name": src_path.stem,
            "annotations": annotations,
        }]
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
snippet_003.mp4,mouse_A12,2024-01-16
```

**Converter:**
```python
class MappedAnimalConverter(CustomLabelConverter):
    src_format = "mapped_animal"
    label_kind = "behavior"
    label_format = "mapped_animal_v1"

    _defaults = dict(
        group_from="filename",
        fps=30.0,
        background_label="none",
        animal_mapping_file=None,  # Path to CSV mapping
    )

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        # Load mapping once during init
        mapping_file = self.params.get("animal_mapping_file")
        if mapping_file:
            import pandas as pd
            self.animal_mapping = pd.read_csv(mapping_file).set_index("video_file")
        else:
            self.animal_mapping = None

    def _load_source_file(self, src_path):
        return pd.read_csv(src_path)

    def _build_label_map(self, data, params):
        behaviors = ["none"] + sorted(data["Behavior"].unique())
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self, data, src_path, raw_row, params):
        # Look up animal ID from mapping
        video_name = src_path.name
        if self.animal_mapping is not None and video_name in self.animal_mapping.index:
            animal_id = self.animal_mapping.loc[video_name, "animal_id"]
            date = self.animal_mapping.loc[video_name, "date"]
        else:
            animal_id = "unknown"
            date = None

        annotations = [
            (row["Start (s)"], row["Stop (s)"], row["Behavior"])
            for _, row in data.iterrows()
        ]

        metadata = {"animal_id": animal_id}
        if date:
            metadata["date"] = date

        return [{
            "sequence_name": f"{date}_{animal_id}_{video_name}",
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

Edit [\_\_init\_\_.py](__init__.py):

```python
# Add import
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

### 2. Register Source Files

```python
from mosaic.core import Dataset

dataset = Dataset("/path/to/dataset")

# Add your annotation file to raw index
dataset.index_tracks_raw(
    path="/path/to/annotations.csv",
    src_format="my_custom_format",  # Must match converter's src_format
    group="experiment_A",
)
```

### 3. Convert Labels

```python
# Convert using your converter
dataset.convert_all_labels(
    kind="behavior",
    source_format="my_custom_format",  # Must match converter's src_format
    # Add any custom parameters:
    time_offset=10.0,
    animal_id="mouse_A12",
)
```

### 4. Verify Output

```python
import pandas as pd
import numpy as np

# Check index
labels_idx = dataset.get_root("labels") / "behavior" / "index.csv"
df = pd.read_csv(labels_idx)
print(df)

# Load a sequence
data = np.load(df.iloc[0]["abs_path"])
print(f"Frames: {len(data['frames'])}")
print(f"Behaviors: {data['label_names']}")
print(f"Labels shape: {data['labels'].shape}")
```

## Output Format

Every converter produces `.npz` files with this structure:

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

Add custom metadata:

```python
{
    # ... required fields above ...
    "meta_animal_id": str,          # Your custom fields
    "meta_time_offset": float,
    "meta_experimenter": str,
    # etc.
}
```

Metadata fields are also added to the index CSV with `meta_` prefix.

## Testing Your Converter

### Step 1: Syntax Check

```bash
python -m py_compile my_converter.py
```

### Step 2: Small Test File

Create a minimal annotation file and test:

```python
from mosaic.core import Dataset

dataset = Dataset("/tmp/test_dataset")
dataset.init()  # Create directory structure

# Index test file
dataset.index_tracks_raw(
    path="/path/to/test_annotation.csv",
    src_format="my_custom_format",
    group="test",
)

# Convert
dataset.convert_all_labels(
    source_format="my_custom_format",
    fps=30.0,
)

# Verify
import pandas as pd
labels_idx = dataset.get_root("labels") / "behavior" / "index.csv"
print(pd.read_csv(labels_idx))
```

### Step 3: Check Output

```python
import numpy as np

data = np.load("/tmp/test_dataset/labels/behavior/test__sequence.npz")

print("Keys:", list(data.keys()))
print("Frames:", data["frames"][:10])
print("Labels:", data["labels"][:10])
print("Label names:", data["label_names"])
print("FPS:", data["fps"])
```

## Troubleshooting

### Error: "No converter registered"

Make sure you:
1. Imported your module in `__init__.py`
2. Called `register_label_converter()` on your class
3. Restarted Python (to reload the module)

### Error: "KeyError: 'behavior_name'"

Check that:
- Your label map includes all behaviors in annotations
- Behavior names are spelled consistently
- The background label is in the label map

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

### Debugging Tips

Add print statements in your methods:

```python
def _extract_annotations(self, data, src_path, raw_row, params):
    print(f"Processing: {src_path}")
    print(f"Params: {params}")
    print(f"Data shape: {data.shape if hasattr(data, 'shape') else len(data)}")

    annotations = [...]
    print(f"Extracted {len(annotations)} annotations")

    return [...]
```

## Best Practices

1. **Keep it simple**: Only implement what you need
2. **Test early**: Test with a small file first
3. **Document parameters**: Add docstrings for custom parameters
4. **Handle edge cases**: Empty files, missing columns, etc.
5. **Validate inputs**: Check that required columns exist
6. **Use background label**: Always include a "none"/"other" category
7. **Consistent naming**: Use clear, descriptive sequence names
8. **Add metadata**: Store useful info (animal IDs, dates, etc.)

## Examples in This Directory

- [custom_label_template.py](custom_label_template.py) - Full template with examples
- [calms21_behavior.py](calms21_behavior.py) - CalMS21 format
- [boris_aggregated_csv.py](boris_aggregated_csv.py) - BORIS CSV/TSV
- [boris_pandas_pickle.py](boris_pandas_pickle.py) - BORIS pickle

## Need Help?

1. Check the template: [custom_label_template.py](custom_label_template.py)
2. Look at existing converters for similar formats
3. Review this README for your use case
4. Add print statements to debug

## Summary Workflow

```python
# 1. Create converter (my_converter.py)
class MyConverter(CustomLabelConverter):
    src_format = "my_format"

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
from . import my_converter
my_converter.MyConverter = register_label_converter(
    my_converter.MyConverter
)

# 3. Use in your code
dataset.convert_all_labels(source_format="my_format")
```

That's it! You now have a custom label converter that fits your exact needs.
