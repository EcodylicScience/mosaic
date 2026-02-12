"""
Label converter library for behavior datasets.

This module provides a plugin architecture for converting various label formats
to the standardized behavior dataset format. Converters are automatically
registered via the @register_label_converter decorator.

Adding a New Label Converter
-----------------------------
1. Create a new file in this directory (e.g., boris_behavior.py)
2. Use label_converter_template.py as a starting point
3. Implement the converter class with required attributes:
   - src_format: str (must match tracks_raw/index.csv)
   - label_kind: str (e.g., "behavior", "id_tags")
   - label_format: str (version identifier)
4. Implement the convert() method
5. Decorate with @register_label_converter
6. Import the module here to register it

Available Converters
--------------------
After importing, converters are registered in LABEL_CONVERTERS dict
accessible from mosaic.core.dataset module.

Usage
-----
>>> from mosaic.core import Dataset
>>> dataset = Dataset("/path/to/dataset")
>>>
>>> # Convert CalMS21 labels
>>> dataset.convert_all_labels(
...     kind="behavior",
...     source_format="calms21_npy",
...     group_from="filename"
... )
>>>
>>> # Convert BORIS aggregated CSV/TSV labels
>>> dataset.convert_all_labels(
...     kind="behavior",
...     source_format="boris_aggregated_csv",
...     delimiter="\t",  # Use "\t" for TSV, "," for CSV
...     fps=None,  # Auto-detect from file
... )
>>>
>>> # Convert BORIS Pandas pickle labels
>>> dataset.convert_all_labels(
...     kind="behavior",
...     source_format="boris_pandas_pickle",
...     fps=None,  # Auto-detect from DataFrame
... )
"""

# Import dataset module to get access to register_label_converter decorator
from mosaic.core.dataset import register_label_converter

# Import all converters to trigger registration
# Each converter module should define a class decorated with @register_label_converter
from . import calms21_behavior
from . import boris_aggregated_csv
from . import boris_pandas_pickle

# Register the CalMS21 converter
calms21_behavior.CalMS21BehaviorConverter = register_label_converter(
    calms21_behavior.CalMS21BehaviorConverter
)

# Register BORIS converters
boris_aggregated_csv.BorisAggregatedCSVConverter = register_label_converter(
    boris_aggregated_csv.BorisAggregatedCSVConverter
)

boris_pandas_pickle.BorisPandasPickleConverter = register_label_converter(
    boris_pandas_pickle.BorisPandasPickleConverter
)

__all__ = [
    "calms21_behavior",
    "boris_aggregated_csv",
    "boris_pandas_pickle",
]
