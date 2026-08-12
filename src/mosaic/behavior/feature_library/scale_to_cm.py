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
"""

from __future__ import annotations

from pathlib import Path
from typing import final

import pandas as pd

from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
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
_LENGTH_NAMES: frozenset[str] = frozenset(
    {"X", "Y", "SPEED", "VX", "VY", "AX", "AY", "speed", "vx", "vy"}
)
_LENGTH_PREFIXES: tuple[str, ...] = ("poseX", "poseY", "bbox_", "midline_")

# Confidences share the ``pose`` prefix with the coordinates and carry no length.
_NOT_LENGTH_PREFIXES: tuple[str, ...] = ("poseP",)


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
    """Emit centimetre copies of every length-bearing column.

    Outputs one row per input row: each scaled column under its own name plus a
    suffix, alongside the metadata columns. The pixel columns are left untouched
    and are not copied through -- the source table remains the authority on what
    was measured, and this run is the authority on what it means in centimetres.

    Params:
        cm_per_pixel: Override the dataset's recorded scale. Hashed, so an
            override is part of the run identity. ``None`` (the default) reads
            the value from the media index for the sequence being processed.
        suffix: Appended to each scaled column's name. Default ``"_cm"``.
        columns: Restrict to these columns instead of every length-bearing one.
    """

    category = "per-frame"
    name = "scale-to-cm"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = (
        False  # resolves one entry's calibration, and refuses a multi-entry frame
    )
    consumed_roots: tuple[str, ...] = ("media_raw",)

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        cm_per_pixel: float | None = None
        suffix: str = "_cm"
        columns: list[str] | None = None

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
