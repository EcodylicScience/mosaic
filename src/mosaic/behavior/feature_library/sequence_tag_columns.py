"""SequenceTagColumns -- the sequence tags of the dataset's tag table, as columns."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, ClassVar, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.annotations.tag_table import TagRead, read_tag_table
from mosaic.core.entry import Entry
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    Column,
    ParamColumn,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel, RelatesLevel
from mosaic.core.pipeline.tag_reads import refuse_tag_read
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    TrackInputs,
)
from mosaic.core.spaces import TRACK_SPACES, TrackSpace
from mosaic.core.typed_attribute import TypedAttributeType, TypedAttributeValue

from .helpers import entry_identity, identity_columns
from .registry import register_feature

_FIELDS_DESCRIPTION = (
    "The sequence tags to attach as columns, at least one. Each names a "
    "sequence tag of the dataset's tag table."
)

_FIELD_RENAMES_DESCRIPTION = (
    "A mapping from a sequence tag's name to its output column name. Unset, "
    "every column keeps its tag's name."
)


@final
@register_feature
class SequenceTagColumns:
    """Attach the sequence tags that ``fields`` names to each row of the input.

    A sequence tag has one value for the whole entry, so every row of an entry
    gets the same value whatever its id, and the feature serves tracks without
    tracklets. A column takes the type of its tag: a ``label`` is a ``bool`` that
    is false where unassigned, ``bool`` is pandas ``boolean``, ``int`` is
    ``Int64``, ``float`` is ``float64``, and ``text`` and ``categorical`` are
    text. Any other unassigned value is null, as is every value of an entry that
    the tag table lacks.

    Outputs per row: frame/time/id/group/sequence + one column per field.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.sequence_tag_columns.SequenceTagColumns.Params`.
    """

    category = "tag"
    name = "sequence-tag-columns"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = False
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    requires: ClassVar[tuple[Requirement, ...]] = ()
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.group_col),
            Column(C.seq_col),
            Column(C.id_col),
            ParamColumn("fields", renamed_by="field_renames"),
        ),
        conditional=kept(C.frame_col, C.time_col),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        fields: Annotated[list[str], Declared(_FIELDS_DESCRIPTION)] = Field(
            default_factory=list
        )
        field_renames: Annotated[
            dict[str, str] | None, Declared(_FIELD_RENAMES_DESCRIPTION)
        ] = None

    def __init__(
        self,
        inputs: SequenceTagColumns.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._types: dict[str, TypedAttributeType] = {}
        self._values: dict[Entry, Mapping[str, TypedAttributeValue]] = {}

    def tag_read(self) -> TagRead:
        """The sequence tags this step reads, and the columns it writes."""
        return TagRead(
            sequence_tags=tuple(self.params.fields),
            writes=frozenset(
                {C.group_col, C.seq_col, C.id_col, C.frame_col, C.time_col}
            ),
            renames=tuple(sorted((self.params.field_renames or {}).items())),
        )

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        lookup = dependency_lookups.get("_tags")
        if not lookup:
            msg = f"{self.name}: the run gave no tag table to read"
            raise ValueError(msg)
        (path,) = {*lookup.values()}
        table = read_tag_table(path)
        refuse_tag_read(self.name, table, self.tag_read())
        defined: dict[str, TypedAttributeType] = {
            definition.name: definition.type
            for definition in table.definitions.sequence
        }
        self._types = {name: defined[name] for name in self.params.fields}
        self._values = {
            (sequence.group, sequence.sequence): sequence.tags
            for sequence in table.sequences
        }
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        tags = self._values.get(entry_identity(df), {})
        rename_map = self.params.field_renames or {}

        out = identity_columns(df)
        for field in self.params.fields:
            out[rename_map.get(field, field)] = _column(
                self._types[field], field, tags, df.index
            )
        return out


def _column(
    type_: TypedAttributeType,
    name: str,
    tags: Mapping[str, TypedAttributeValue],
    index: pd.Index,
) -> pd.Series:
    """The column of tag *name* for an entry holding *tags*, one value per row."""
    rows = len(index)
    assigned = name in tags
    value = tags.get(name)
    if type_ == "label":
        return pd.Series(np.full(rows, assigned, dtype=np.bool_), index=index)
    if type_ == "float":
        number = np.nan if value is None else float(value)
        return pd.Series(np.full(rows, number, dtype=np.float64), index=index)
    if type_ == "bool":
        return pd.Series([value] * rows, index=index, dtype="boolean")
    if type_ == "int":
        return pd.Series([value] * rows, index=index, dtype="Int64")
    return pd.Series([value] * rows, index=index, dtype=object)
