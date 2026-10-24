"""Converting a table into body lengths, as a recorded step.

The counterpart of ``scale-to-cm`` for a comparison across animals of different
sizes. Speed in body lengths per second, distance to a neighbour in body
lengths and the area of a group in body lengths squared are what a study of
development reports, because they compare behaviour rather than size.

**One length per sequence.** Positions are scaled by a single number for every
individual of a sequence: the median, over its individuals, of each
individual's median length that ``body-length`` reports. Scaling each
individual by its own length would move the individuals relative to one
another, and every distance between two of them would then be measured in two
units at once.

The length comes from one of two places. An explicit ``length`` is hashed, as
``scale-to-cm`` hashes ``cm_per_pixel``. Otherwise it is read from the
``body-length`` run that ``body_length`` references, whose identity enters this
run's identity through the reference.

The length must be in the units of the positions it divides. A ``body-length``
run over pixel tracks scales pixel tracks, and one over centimetre tracks
scales centimetre tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, final

import numpy as np
import pandas as pd
from pydantic import model_validator

from mosaic.core.pipeline.column_contract import (
    META_KEPT,
    Column,
    Conditional,
    ParamIs,
    Provides,
    Requirement,
    When,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    NON_PAYLOAD_COLS,
    BodyLengthResult,
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.params import Declared, Params

from .registry import register_feature
from .scale_to_cm import LENGTH_COLUMNS, scalable_columns

_LENGTH_DESCRIPTION = (
    "The body length to divide by, in the units of the input positions. "
    "A value that is not positive raises. Unset, the length is read from "
    "the body-length run that body_length references."
)

_BODY_LENGTH_DESCRIPTION = (
    "Reference to the upstream body-length run. The sequence's length is "
    "the median, over its individuals, of their body_length. Read when "
    "length is unset."
)

_MODE_DESCRIPTION = (
    "derive emits only the scaled columns, each under its own name plus "
    "suffix, alongside the metadata. convert returns the whole table with "
    "every length column converted in place, under its own name."
)

_SUFFIX_DESCRIPTION = (
    "Appended to each scaled column's name in derive mode. Names nothing "
    "in convert mode, which refuses a non-default value."
)

_COLUMNS_DESCRIPTION = (
    "Replace the length-bearing classifier with exactly these columns, "
    "scaling every one whether or not it is length-bearing. A name "
    "absent from the table raises."
)

_NAME = "scale-to-bl"


@final
@register_feature
class ScaleToBodyLength:
    """Convert every length-bearing column into body lengths.

    Outputs one row per input row. The modes are those of ``scale-to-cm``:
    ``"derive"`` (the default) emits only the scaled columns under a suffix,
    and ``"convert"`` returns the whole table with every length column divided
    in place, a track-shaped table in body lengths that a track feature can be
    chained onto. The columns scaled are those ``scale-to-cm`` scales.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.scale_to_bl.ScaleToBodyLength.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False  # one entry's length, and refuses a multi-entry frame
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    # The group and sequence find the entry's length, unless ``length`` is set.
    requires: ClassVar[tuple[Requirement, ...]] = (
        *(
            When(Column(name), (ParamIs("length", negated=True),))
            for name in (C.group_col, C.seq_col)
        ),
        LENGTH_COLUMNS,
    )
    provides: ClassVar[Provides] = Provides(
        conditional=(
            *META_KEPT,
            *(
                Conditional(Column(name))
                for name in sorted(NON_PAYLOAD_COLS - C.meta_set())
            ),
        ),
        payload_families=True,
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        length: Annotated[float | None, Declared(_LENGTH_DESCRIPTION)] = None
        body_length: Annotated[
            BodyLengthResult | None, Declared(_BODY_LENGTH_DESCRIPTION)
        ] = None
        mode: Annotated[Literal["derive", "convert"], Declared(_MODE_DESCRIPTION)] = (
            "derive"
        )
        suffix: Annotated[str, Declared(_SUFFIX_DESCRIPTION)] = "_bl"
        columns: Annotated[list[str] | None, Declared(_COLUMNS_DESCRIPTION)] = None

        @model_validator(mode="after")
        def _check(self) -> Self:
            """Refuse a run with no length, or a suffix that names nothing."""
            if self.length is None and self.body_length is None:
                raise ValueError(
                    f"{_NAME} needs a length: reference a body-length run as "
                    'body_length={"feature": "body-length__from__tracks"}, or '
                    "pass length= in the units of the positions."
                )
            if self.length is not None and self.body_length is not None:
                raise ValueError(
                    f"{_NAME}: length and body_length are both set, and only one "
                    "can be the length divided by. Drop one."
                )
            if self.mode == "convert" and self.suffix != "_bl":
                raise ValueError(
                    f"{_NAME}: mode='convert' converts every length column in "
                    f"place and renames none, so suffix={self.suffix!r} names no "
                    "column while still entering the run identity. Drop it."
                )
            return self

    def __init__(
        self,
        inputs: ScaleToBodyLength.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._length_lookup: DependencyLookup | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._length_lookup = dependency_lookups.get("body_length")
        if self.params.length is None and not self._length_lookup:
            # Raised here, once, rather than per entry in apply, where a failure
            # is recorded and the run carries on having written nothing.
            raise ValueError(
                f"{self.name}: the body-length dependency resolved to no entries. "
                "Run body-length first and reference it as "
                'body_length={"feature": "body-length__from__tracks"}.'
            )
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        p = self.params
        names = [str(name) for name in df.columns]
        wanted = p.columns if p.columns is not None else scalable_columns(names)
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise ValueError(
                f"{self.name}: asked to scale {missing}, which this table does not "
                "carry."
            )
        if not wanted:
            raise ValueError(
                f"{self.name}: this table carries no length-bearing column, so there "
                "is nothing to convert. Pass columns= to name them explicitly."
            )

        factor = 1.0 / self._length_for(df)

        if p.mode == "convert":
            out = df.copy()
            for name in wanted:
                out[name] = df[name] * factor
            return out

        out = pd.DataFrame(
            {f"{name}{p.suffix}": df[name] * factor for name in wanted},
            index=df.index,
        )
        meta = C.meta_set() & set(names)
        return out.join(df[sorted(meta)])

    def _length_for(self, df: pd.DataFrame) -> float:
        """The body length of this entry, in the units of its positions."""
        p = self.params
        if p.length is not None:
            if not p.length > 0.0:
                raise ValueError(
                    f"{self.name}: length={p.length} is not a usable body length."
                )
            return float(p.length)

        group, sequence = self._entry(df)
        if self._length_lookup is None:
            raise ValueError(
                f"{self.name} reads the body length from a body-length run and no "
                "lookup was loaded. Run it through run_feature, or pass length=."
            )
        path = self._length_lookup.get((group, sequence))
        if path is None:
            raise ValueError(
                f"{self.name}: body-length has no output for ({group!r}, {sequence!r})."
            )
        lengths = pd.read_parquet(path, columns=[C.id_col, "body_length"])
        per_id = lengths.dropna().groupby(C.id_col)["body_length"].median()
        length = float(per_id.median()) if not per_id.empty else float("nan")
        if not np.isfinite(length) or length <= 0:
            raise ValueError(
                f"{self.name}: body-length for ({group!r}, {sequence!r}) gives "
                f"{length!r}, which is not a usable body length."
            )
        return length

    def _entry(self, df: pd.DataFrame) -> tuple[str, str]:
        """The single ``(group, sequence)`` this frame covers."""
        names = set(str(name) for name in df.columns)
        if not {C.group_col, C.seq_col} <= names:
            raise ValueError(
                f"{self.name} needs {C.group_col!r} and {C.seq_col!r} to find this "
                "sequence's body length, and this table does not carry both."
            )
        groups = {str(value) for value in df[C.group_col].unique()}
        sequences = {str(value) for value in df[C.seq_col].unique()}
        if len(groups) != 1 or len(sequences) != 1:
            raise ValueError(
                f"{self.name} expects one sequence per call, and got "
                f"{sorted(groups)} x {sorted(sequences)}."
            )
        return next(iter(groups)), next(iter(sequences))
