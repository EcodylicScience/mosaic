"""Converting a pixel table into physical units, as a recorded step.

Tracks are pixels, everywhere and by construction. That is not because pixels are
the useful unit -- for most questions they are not -- but because a pixel is what
the camera actually measured, and a centimetre is a pixel plus a claim about the
world. Keeping the claim out of the table means a table cannot be silently wrong
about it, which is what happened when one tracker wrote centimetres into the same
columns every other tracker filled with pixels.

So the conversion lives here. The scale comes from the media index, where it sits
beside the video it describes: it is a property of the camera and the rig, so a
dataset that mixes rigs carries a different value per row and nothing has to be
told twice. Running this feature records which scale was used in a run
identifier, so two calibrations are two addressable results rather than one
column that quietly changed meaning.

**Uncalibrated refuses.** A dataset that has not been told its scale cannot be
given one by default: ``1.0`` is a scale, not an absence, and assuming it would
reproduce exactly the failure this whole arrangement exists to remove.

**Converting in place is a mode, and it does not move the invariant.** The claim
is about *tracks*: the tables under ``tracks/``, which this feature never writes
and which stay in pixels whatever mode it runs in. Its own output is a derived
table like any other, and a derived table has always carried whatever unit its
feature computed -- ``speed-angvel`` over a centimetre input reports cm/s and
always did. What ``mode="convert"`` adds is that the derived table is also
*track-shaped*, so a track feature can consume it and a whole pipeline can run
downstream of the conversion rather than around it. That is the case where
"which unit is this?" gets thin, and the answer is the same one this module is
built on: the scale is in the run identifier, so the table is addressable rather
than conventional, and the pixel original is still on disk, unmodified, one
input away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self, final

import pandas as pd
from pydantic import model_validator

from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    Inputs,
    InputStream,
    Params,
    Result,
    TrackInput,
)

from .registry import register_feature

# Column families that carry a length, and so scale. Read after stripping a
# ``#`` suffix, so ``X``, ``X#wcentroid`` and ``X#head`` are all covered.
#
# Kept in step with ``mosaic.core.track_library.trex._LENGTH_FIELDS``, which is
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

    Params:
        cm_per_pixel: Override the dataset's recorded scale. Hashed, so an
            override is part of the run identity. ``None`` (the default) reads
            the value from the media index for the sequence being processed.
        mode: ``"derive"`` for suffixed copies of the length columns only,
            ``"convert"`` for the whole table converted in place. Default
            ``"derive"``.
        suffix: Appended to each scaled column's name in ``"derive"`` mode.
            Default ``"_cm"``. Names nothing in ``"convert"`` mode, which
            refuses a non-default value rather than hashing one.
        columns: Restrict to these columns instead of every length-bearing one.
            In ``"convert"`` mode naming a subset leaves the unnamed length
            columns in pixels *inside a table whose other lengths are
            centimetres* -- a mixed table this feature will not otherwise
            produce.
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
    accepts_overlap = (
        False  # resolves one entry's calibration, and refuses a multi-entry frame
    )
    consumed_roots: tuple[str, ...] = ("media_raw",)
    emits: EmitsLevel = "individual"

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        cm_per_pixel: float | None = None
        mode: Literal["derive", "convert"] = "derive"
        suffix: str = "_cm"
        columns: list[str] | None = None

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
