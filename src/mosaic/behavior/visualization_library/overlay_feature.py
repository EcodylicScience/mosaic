"""The annotated video, as a step a graph can end on.

Rendering was the one thing in mosaic with no identity. ``overlay.py`` carries no
``@register_feature`` and there is no render verb, so every real workflow ended
one step short of the artifact a biologist actually looks at -- the pipeline
produced a table, and somebody opened a notebook to turn it into a video. A graph
could not express "and then draw it", which meant the deliverable was outside
whatever ran the analysis and outside whatever recorded it.

**A ``media`` feature, exactly as ``egocentric-crop`` already is.** It writes an
artifact other things read rather than a table other features read, and that is
what the category means; there is deliberately no visualization category, because
a second, non-artifact status mechanism bolted to the side of the first is what
this exists to avoid. Having an identity is the whole point: the video is
addressed by ``run_id`` like everything else, so re-rendering the same overlay
over the same tracks is a cache hit and a changed palette is a new artifact.

It draws whatever it is given. Where a table carries no keypoints -- which is
every centroid-only tracker, T-Rex among them -- the drawing falls back to the
body centre, which is what the renderer underneath already does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import pandas as pd
from pydantic import Field

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import (
    COLUMNS,
    EmitsLevel,
    Inputs,
    Result,
    TrackInput,
)
from mosaic.core.params import Declared, Params

from ..feature_library.registry import register_feature
from .playback import play_video

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.types.feature import DependencyLookup

__all__ = ["Overlay"]

_LABEL_KIND_DESCRIPTION = (
    "Kind of converted labels to draw as ground truth. Unset, no ground truth is drawn."
)

_COLOR_BY_DESCRIPTION = (
    "Name of an input feature to color the drawing by, matched without "
    "regard to case. The reserved value gt colors by the ground-truth label."
)

_LABEL_MAPS_DESCRIPTION = (
    "Names to render per feature, keyed by the label value written as text."
)

_HIDE_UNLABELED_DESCRIPTION = (
    "Skip drawing individuals that lack a label after filtering and mapping."
)

_START_DESCRIPTION = "The first frame to draw."

_END_DESCRIPTION = (
    "The last frame to draw, inclusive. Unset, drawing continues to the end "
    "of the video."
)

_DOWNSCALE_DESCRIPTION = (
    "Downscale factor applied to the frame, where 1.0 is no scaling."
)

_SHOW_INDIVIDUAL_BBOXES_DESCRIPTION = (
    "Draw a bounding box for each individual. Pose points and labels are "
    "drawn regardless."
)

_PAIR_BOX_FEATURE_DESCRIPTION = (
    "Name of the pair-label feature that determines when to draw a union "
    "box around both individuals of a pair."
)

_PAIR_BOX_BEHAVIORS_DESCRIPTION = (
    "Behavior values that trigger drawing a pair-level box."
)

_HIDE_INDIVIDUAL_BBOXES_FOR_PAIR_DESCRIPTION = (
    "Skip drawing an individual's own bounding box when it is already "
    "inside a drawn pair box."
)

_DRAW_OPTIONS_DESCRIPTION = (
    "Per-frame drawing options. Recognized keys are show_labels, "
    "point_radius, bbox_thickness and font_scale."
)

_CAMERA_DESCRIPTION = (
    "Which camera of a synchronized recording to draw. Required when the "
    "sequence spans more than one, because two views are not one "
    "timeline. The media resolver refuses to pick between them rather "
    "than concatenating them into a fabricated timeline."
)


@register_feature
class Overlay:
    """Render one annotated video per sequence, from tracks and drawn labels.

    Reads the tracks table and, for every feature wired into its inputs, that
    feature's per-frame labels -- so a classifier's predictions are drawn by
    naming the classifier as an input rather than by a second configuration
    language. The video lands in the run root as ``<group>__<sequence>.mp4``,
    beside a one-row table recording what was drawn.

    Examples:
        Draw the tracks alone::

            >>> dataset.run_feature(Overlay(), sequences=["hex_3"])

        Draw a classifier's predictions over them::

            >>> Overlay(
            ...     Overlay.Inputs(("tracks", Result(feature="behavior-xgb__from__tracks"))),
            ... )
    """

    category = "media"
    name = "overlay"
    version = "0.1"
    parallelizable = False  # Video I/O is sequential.
    scope_dependent = False
    accepts_overlap = False  # It opens the entry's own video.
    consumed_roots: tuple[str, ...] = ("media_raw",)
    emits: EmitsLevel = "individual"

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        """What to draw, and how.

        Every field changes the pixels, so every field is in the identity -- a
        re-render under a different palette is a different artifact rather than a
        silent overwrite of the first.
        """

        label_kind: Annotated[str | None, Declared(_LABEL_KIND_DESCRIPTION)] = (
            "behavior"
        )
        color_by: Annotated[str | None, Declared(_COLOR_BY_DESCRIPTION)] = None
        label_maps: Annotated[
            dict[str, dict[str, str]], Declared(_LABEL_MAPS_DESCRIPTION)
        ] = Field(default_factory=dict)
        hide_unlabeled: Annotated[bool, Declared(_HIDE_UNLABELED_DESCRIPTION)] = False
        start: Annotated[int, Declared(_START_DESCRIPTION)] = 0
        end: Annotated[int | None, Declared(_END_DESCRIPTION)] = None
        downscale: Annotated[float, Declared(_DOWNSCALE_DESCRIPTION)] = 1.0
        show_individual_bboxes: Annotated[
            bool, Declared(_SHOW_INDIVIDUAL_BBOXES_DESCRIPTION)
        ] = True
        pair_box_feature: Annotated[
            str | None, Declared(_PAIR_BOX_FEATURE_DESCRIPTION)
        ] = None
        pair_box_behaviors: Annotated[
            list[str], Declared(_PAIR_BOX_BEHAVIORS_DESCRIPTION)
        ] = Field(default_factory=list)
        hide_individual_bboxes_for_pair: Annotated[
            bool, Declared(_HIDE_INDIVIDUAL_BBOXES_FOR_PAIR_DESCRIPTION)
        ] = False
        draw_options: Annotated[dict[str, Any], Declared(_DRAW_OPTIONS_DESCRIPTION)] = (
            Field(default_factory=dict)
        )
        camera: Annotated[str | None, Declared(_CAMERA_DESCRIPTION)] = None

    def __init__(
        self,
        inputs: Overlay.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: Overlay.Inputs = (
            Overlay.Inputs(("tracks",)) if inputs is None else inputs
        )
        self.params: Overlay.Params = self.Params.from_overrides(params)
        self._ds: Dataset | None = None
        self._run_root: Path | None = None
        self._tracks_run_id: str | None = None

        self.storage_feature_name: str = self.name
        self.storage_use_input_suffix: bool = True
        self.skip_existing_outputs: bool = False

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds: Dataset) -> None:
        """Receive the dataset before anything is drawn."""
        self._ds = ds

    def set_scope(self, scope: object) -> None:
        """Take the tracks variant the run resolved, so the drawing reads it.

        A dataset holding two recipes for one entry has no defensible default
        between them, and the renderer refuses rather than guessing -- so the
        variant this run was built from has to reach it, or an ordinary
        two-tracker dataset could not be drawn at all.
        """
        variants: tuple[str, ...] = tuple(getattr(scope, "tracks_variants", ()))
        self._tracks_run_id = variants[0] if len(variants) == 1 else None

    # ----------------------- Feature protocol ---------------------

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._run_root = run_root
        return True

    def save_state(self, run_root: Path) -> None:
        pass

    def fit(self, inputs: object) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Render this entry's video, and record what was drawn."""
        if df.empty:
            return pd.DataFrame()
        if self._ds is None or self._run_root is None:
            raise RuntimeError(
                "[overlay] no dataset bound; run this through run_feature, which "
                "is what resolves the media and the run root."
            )

        group = self._cell(df, COLUMNS.group_col)
        sequence = self._cell(df, COLUMNS.seq_col)
        params = self.params
        out_path = self._run_root / f"{make_entry_key(group, sequence)}.mp4"

        written = play_video(
            self._ds,
            group,
            sequence,
            feature_runs=self._feature_runs(),
            label_kind=params.label_kind,
            color_by=params.color_by,
            label_maps=dict(params.label_maps) or None,
            hide_unlabeled=params.hide_unlabeled,
            start=params.start,
            end=params.end,
            downscale=params.downscale,
            draw_options=dict(params.draw_options) or None,
            show_individual_bboxes=params.show_individual_bboxes,
            pair_box_feature=params.pair_box_feature,
            pair_box_behaviors=list(params.pair_box_behaviors) or None,
            hide_individual_bboxes_for_pair=params.hide_individual_bboxes_for_pair,
            output_path=out_path,
            # A rendered artifact is written, never shown: this runs under a
            # queue as readily as under a notebook, and a window opened on a
            # worker is a job that never returns.
            show_window=False,
            tracks_run_id=self._tracks_run_id,
            camera=params.camera,
        )
        if written is None:
            raise RuntimeError(
                f"[overlay] no video was written for {group}/{sequence}; the "
                f"encoder could not be opened for {out_path}"
            )
        return pd.DataFrame(
            [
                {
                    COLUMNS.group_col: group,
                    COLUMNS.seq_col: sequence,
                    "video": written.name,
                    "n_ids": int(df[COLUMNS.id_col].nunique())
                    if COLUMNS.id_col in df.columns
                    else 0,
                    "n_frames": int(df[COLUMNS.frame_col].nunique())
                    if COLUMNS.frame_col in df.columns
                    else 0,
                }
            ]
        )

    # ----------------------- Internal ----------------------------

    def _feature_runs(self) -> dict[str, str | None]:
        """The label sources this overlay draws, as the renderer names them.

        Read off the pinned inputs rather than from a params list, so what is
        drawn is what the run recorded reading -- one declaration, and a graph's
        substitution reaches it for free.
        """
        found: dict[str, str | None] = {}
        for item in self.inputs.root:
            if isinstance(item, Result):
                found[item.feature] = item.run_id
        return found

    @staticmethod
    def _cell(df: pd.DataFrame, column: str) -> str:
        """One entry-identifying cell, or ``""`` where the table carries none."""
        return str(df[column].iloc[0]) if column in df.columns else ""
