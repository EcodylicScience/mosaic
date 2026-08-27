"""Field prose shared by the two BORIS converters.

:class:`~mosaic.behavior.label_library.boris_aggregated_csv.BorisAggregatedCSVParams`
and
:class:`~mosaic.behavior.label_library.boris_pandas_pickle.BorisPandasPickleParams`
read the same BORIS event columns from two different export formats, so every
field but ``delimiter`` (the CSV/TSV-only one) means the same thing in both.
Declared once here rather than twice, so the two cannot drift.
"""

from __future__ import annotations

FPS_DESCRIPTION = "Frames per second. Unset, auto-detected from the FPS column."

SUBJECT_ID_MAP_DESCRIPTION = (
    "Mapping from BORIS subject names to numeric IDs. Unset, every label is "
    "scene-level, recorded as [-1, -1]."
)

PAIR_BEHAVIORS_DESCRIPTION = "Behavior names that are symmetric pair interactions."

BACKGROUND_LABEL_DESCRIPTION = (
    "The label used when no behavior is active, conventionally at ID 0."
)

NO_FOCAL_SUBJECT_NAME_DESCRIPTION = (
    'The subject name used for "No focal subject" behaviors.'
)

INCLUDE_POINT_EVENTS_DESCRIPTION = "Include instantaneous point events."

OBSERVATION_COL_DESCRIPTION = "Column name for the observation identifier."

SUBJECT_COL_DESCRIPTION = "Column name for the subject."

BEHAVIOR_COL_DESCRIPTION = "Column name for the behavior."

START_COL_DESCRIPTION = "Column name for the start time."

STOP_COL_DESCRIPTION = "Column name for the stop time."

BEHAVIOR_TYPE_COL_DESCRIPTION = "Column name for the behavior type (STATE/POINT)."

FPS_COL_DESCRIPTION = "Column name for the FPS value."

CATEGORY_COL_DESCRIPTION = "Column name for the behavioral category."

MODIFIERS_COL_DESCRIPTION = "Column name for the modifiers."
