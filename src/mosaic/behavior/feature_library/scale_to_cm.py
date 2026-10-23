"""Converting a pixel table into physical units, as a recorded step.

Tracks are pixels everywhere, because a pixel is what the camera measured and a
centimetre is a pixel plus a claim about the world. A table that holds only
pixels cannot be wrong about that claim.

The scale comes from one of two places. An explicit ``cm_per_pixel`` is hashed,
so two scales passed that way are two addressable results. Without it, the scale
is read from the media row of the video it describes. It is a property of the
camera and the rig. A dataset that mixes rigs records a different value per row.
That scale enters neither the run identifier nor the composition that the cache
compares. After :meth:`~mosaic.core.dataset.Dataset.set_media_calibration`
changes it, a finished run keeps serving its earlier output until the run is
overwritten or the scale is passed as ``cm_per_pixel``.

**Uncalibrated refuses.** A dataset that has not been told its scale cannot be
given one by default. ``1.0`` is a scale, and assuming it would reproduce the
failure this arrangement exists to remove.

**Converting in place is a mode, and it does not move the invariant.** The claim
is about *tracks*, the tables under ``tracks/``, which this feature does not
write and which stay in pixels in either mode. This feature's output is a derived
table, in the unit it computed. Over a centimetre input, ``speed-angvel``
reports cm/s. ``mode="convert"`` makes the derived table *track-shaped* as well.
A track feature can then consume it, and a pipeline can run downstream of the
conversion. The table is addressed by its run identifier like any other output,
and the pixel original stays on disk, unmodified, one input away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, final

import pandas as pd
from pydantic import model_validator

from mosaic.core.pipeline.column_contract import (
    META_KEPT,
    AnyOf,
    Column,
    Conditional,
    ParamColumn,
    ParamIs,
    Prefix,
    Provides,
    Requirement,
    When,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    NON_PAYLOAD_COLS,
    EmitsLevel,
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.params import Declared, Params

from .registry import register_feature

_CM_PER_PIXEL_DESCRIPTION = (
    "Override the dataset's recorded scale. A value that is not "
    "positive raises when the feature runs. Unset, the scale is read "
    "from the media index for the sequence being processed."
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
    "absent from the table raises. In convert mode, naming a subset "
    "leaves the unnamed length columns in pixels inside a table whose "
    "other lengths are centimeters."
)

# Column families that carry a length, and so scale. Read after stripping a
# ``#`` suffix, so ``X``, ``X#wcentroid`` and ``X#head`` are all covered.
#
# Kept in step with ``mosaic.core.track_library.trex.LENGTH_FIELDS``, which is
# the list the TREx converter divides *back out*. A name on one list and not the
# other is a column that goes to pixels and never returns, and the number stays
# plausible the whole way down -- a border distance in pixels compared against a
# threshold in centimetres keeps every frame and says nothing. Two tests pin the
# correspondence in both directions.
_LENGTH_NAMES: frozenset[str] = frozenset(
    {
        "X",
        "Y",
        "SPEED",
        "VX",
        "VY",
        "AX",
        "AY",
        "speed",
        "vx",
        "vy",
        # What TREx additionally reports as a length.
        "SPEED_SMOOTH",
        "SPEED_OLD",
        "ACCELERATION",
        "ACCELERATION_SMOOTH",
        "BORDER_DISTANCE",
        "NEIGHBOR_DISTANCE",
        "midline_segment_length",
        "segment_length",
    }
)
_LENGTH_PREFIXES: tuple[str, ...] = ("poseX", "poseY", "bbox_", "midline_")

# Confidences share the ``pose`` prefix with the coordinates and carry no length.
# ``midline_length`` shares the ``midline_`` prefix with three genuine lengths
# and carries none either: TREx's centimetre conversion for it is commented out
# in ``OutputLibrary.cpp``, so it is pixels sitting beside ``midline_x``,
# ``midline_y`` and ``midline_segment_length``, which are not. The prefix is
# long enough to spare ``midline_segment_length``.
_NOT_LENGTH_PREFIXES: tuple[str, ...] = ("poseP", "midline_length")


def scalable_columns(columns: list[str]) -> list[str]:
    """Which of *columns* carry a length, and therefore scale.

    Angles, counts, probabilities and identifiers do not appear: multiplying a
    heading by a distance ratio would produce a number with no meaning, and the
    silence of that failure is the point -- it would still be a plausible float.
    """
    out: list[str] = []
    for name in columns:
        if name.startswith(_NOT_LENGTH_PREFIXES):
            continue
        base = name.split("#", 1)[0]
        if base in _LENGTH_NAMES or name.startswith(_LENGTH_PREFIXES):
            out.append(name)
    return out


@final
@register_feature
class ScaleToCm:
    """Convert every length-bearing column into centimeters.

    Outputs one row per input row. **Two modes, and they are exclusive rather
    than composable.** ``"derive"`` (the default) emits *only* the scaled
    columns, each under its own name plus a suffix, alongside the metadata: the
    source table remains the authority on what was measured, and this run is the
    authority on what it means in centimetres. ``"convert"`` returns the whole
    table with every length column converted **in place, under its own name**, so
    the result is a track-shaped table in centimetres that a track feature can be
    chained onto.

    Emitting both at once -- ``X`` in pixels beside ``X_cm`` -- is the one thing
    neither mode does. That is one table holding two coordinate systems with
    nothing recording which column is which, which is the failure this whole
    module exists to remove.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.scale_to_cm.ScaleToCm.Params`.
    """

    category = "per-frame"
    name = "scale-to-cm"
    # 0.2: the length classifier gained the six fields TREx reports as lengths
    # and lost ``midline_length``, which TREx never scales. That changes what a
    # *default-params* run emits, and ``scalable_columns`` is a module function
    # rather than a Params field, so no digest can see it. The version is the
    # only mechanism that can. ``mode`` is hashed and needed no bump of its own.
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    temporal_reach = 0  # multiplies each row by a scale
    accepts_overlap = (
        False  # resolves one entry's calibration, and refuses a multi-entry frame
    )
    consumed_roots: tuple[str, ...] = ("media_raw",)
    emits: EmitsLevel = "individual"
    # ``apply`` reads the group and sequence to find the calibration, unless
    # ``cm_per_pixel`` is set. The scaled columns are met by one column that
    # ``columns`` names or by one length column. ``apply`` refuses an input that
    # lacks a column that ``columns`` names.
    requires: ClassVar[tuple[Requirement, ...]] = (
        *(
            When(Column(name), (ParamIs("cm_per_pixel", negated=True),))
            for name in (C.group_col, C.seq_col)
        ),
        AnyOf(
            (
                ParamColumn("columns"),
                *(Column(name) for name in sorted(_LENGTH_NAMES)),
                *(Prefix(prefix) for prefix in _LENGTH_PREFIXES),
            )
        ),
    )
    # ``derive`` writes ``<column><suffix>`` per scaled column beside the
    # identity columns. ``convert`` returns its input with the scaled columns
    # replaced, the bookkeeping columns included. A declared output cannot test
    # the value of ``mode``. In both modes each scaled column begins with the
    # name of the input column it scales.
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
        cm_per_pixel: Annotated[
            float | None, Declared(_CM_PER_PIXEL_DESCRIPTION, unit="cm/px")
        ] = None
        mode: Annotated[Literal["derive", "convert"], Declared(_MODE_DESCRIPTION)] = (
            "derive"
        )
        suffix: Annotated[str, Declared(_SUFFIX_DESCRIPTION)] = "_cm"
        columns: Annotated[list[str] | None, Declared(_COLUMNS_DESCRIPTION)] = None

        @model_validator(mode="after")
        def _a_suffix_names_nothing_in_convert_mode(self) -> Self:
            """Refuse a suffix that would enter the identity and name no column.

            ``suffix`` is hashed. In ``convert`` mode it renames nothing, so two
            spellings would mint two run identifiers for one byte-identical
            table -- the same hole a throughput knob in the digest opens, and
            cheaper to close here than to document.
            """
            if self.mode == "convert" and self.suffix != "_cm":
                raise ValueError(
                    "scale-to-cm: mode='convert' converts every length column "
                    f"in place and renames none, so suffix={self.suffix!r} "
                    "names no column while still entering the run identity -- "
                    "two identifiers for one table. Drop it."
                )
            return self

    def __init__(
        self,
        inputs: ScaleToCm.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._ds = None

    def bind_dataset(self, ds: object) -> None:
        """Called by ``run_feature`` before any apply, and in each worker."""
        self._ds = ds

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
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

        scale = self._scale_for(df)

        if p.mode == "convert":
            # In place, under the same names, with every other column carried
            # through: the output is a track-shaped table in centimetres, which
            # is what a track feature downstream can consume. Column order is
            # the input's, so a reader matching a column by shape rather than by
            # name -- a border-distance column, say -- sees the same first match.
            out = df.copy()
            for name in wanted:
                out[name] = df[name] * scale
            return out

        out = pd.DataFrame(
            {f"{name}{p.suffix}": df[name] * scale for name in wanted},
            index=df.index,
        )
        meta = C.meta_set() & set(names)
        return out.join(df[sorted(meta)])

    def _scale_for(self, df: pd.DataFrame) -> float:
        """The centimetres-per-pixel to apply to this entry.

        An explicit parameter wins, because a caller who states a scale has said
        something the dataset does not know. Otherwise it comes from the media
        row for this sequence, which is where a calibration belongs.
        """
        p = self.params
        if p.cm_per_pixel is not None:
            if not p.cm_per_pixel > 0.0:
                raise ValueError(
                    f"{self.name}: cm_per_pixel={p.cm_per_pixel} is not a usable scale."
                )
            return float(p.cm_per_pixel)

        group, sequence = self._entry(df)
        if self._ds is None:
            raise ValueError(
                f"{self.name} reads the scale from the dataset's media index and was "
                "not bound to a dataset. Run it through run_feature, or pass "
                "cm_per_pixel= explicitly."
            )
        scale = self._ds.media_calibration(group, sequence)
        if scale is None:
            raise ValueError(
                f"{self.name}: sequence ({group!r}, {sequence!r}) has no recorded "
                "cm_per_pixel, so there is no way to say how far a pixel is. Record "
                "one with Dataset.set_media_calibration(...), or pass cm_per_pixel= "
                "to state it for this run. It is not assumed to be 1.0: that is a "
                "scale, not an absence."
            )
        return scale

    def _entry(self, df: pd.DataFrame) -> tuple[str, str]:
        """The single ``(group, sequence)`` this frame covers."""
        names = set(str(name) for name in df.columns)
        if not {C.group_col, C.seq_col} <= names:
            raise ValueError(
                f"{self.name} needs {C.group_col!r} and {C.seq_col!r} to find this "
                "sequence's calibration, and this table carries neither."
            )
        groups = {str(value) for value in df[C.group_col].unique()}
        sequences = {str(value) for value in df[C.seq_col].unique()}
        if len(groups) != 1 or len(sequences) != 1:
            raise ValueError(
                f"{self.name} expects one sequence per call, and got "
                f"{sorted(groups)} x {sorted(sequences)}. A calibration is "
                "per-recording, so a frame spanning several has no single scale."
            )
        return next(iter(groups)), next(iter(sequences))
