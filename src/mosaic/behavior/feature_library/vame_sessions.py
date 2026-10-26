"""VAME's sessions, built from an entry's track table, and its latents on the rows.

VAME reads a session row by row and ignores its frame numbers. Each individual is
therefore reindexed onto contiguous frames. Under low-confidence cleaning, VAME
interpolates each missing point linearly. A run of more than ``max_gap_frames``
frames in which VAME discards a point of either reference keypoint splits the
individual into separate sessions. Egocentric alignment centers and rotates every
keypoint by the two references, and an interpolated reference adds motion to all of
them. A missing point of any other keypoint affects only that keypoint, and VAME
interpolates it however long it runs. Without cleaning, VAME does not interpolate,
and every frame with a missing point splits the individual. A session that cannot
yield latents is excluded with its reason, because VAME fails on a keypoint whose
every point it discards. Rows at gap frames outside every session and exclusion
do not receive a latent, and are counted.

Latent ``i`` of a session that starts at frame ``f0`` encodes the window of frames
``f0 + i`` to ``f0 + i + time_window - 1``. It is placed at the window's center,
frame ``f0 + i + time_window // 2``, as every VAME output that maps latents to
frames places it.

:func:`write_session_tracks` writes each session's arrays to the file that the runner
reads. The other functions read a table and settings and return arrays or a table.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.individuals import iter_individuals, refuse_repeated_rows
from mosaic.core.pipeline._utils import atomic_savez
from mosaic.core.pipeline.types import COLUMNS, PoseConfig
from mosaic.core.pose_columns import (
    KeypointArrays,
    keypoint_arrays,
    reindex_contiguous,
)
from mosaic.core.wording import counted

from .external.vame_protocol import (
    check_distinct_references,
    least_training_frames,
    repeated_names,
)
from .helpers import meta_columns

if TYPE_CHECKING:
    from pandas._typing import Scalar

__all__ = [
    "LATENT_PREFIX",
    "EntrySessions",
    "Exclusion",
    "GapRows",
    "SessionPurpose",
    "SessionSettings",
    "SessionTrack",
    "VameKeypoints",
    "build_sessions",
    "latent_columns",
    "min_session_frames",
    "place_latents",
    "vame_keypoints",
    "write_session_tracks",
]

LATENT_PREFIX: Final = "vame_z"
"""What a latent column is called before its dimension, from 0."""

SessionPurpose: TypeAlias = Literal["fit", "embed"]
"""The use of a session. A fit needs more frames per session than an embedding."""

_CAMERA_COL: Final = "camera"

_NO_INTERPOLATION: Final = "VAME does not interpolate without run_lowconf_cleaning"


def latent_columns(zdims: int) -> list[str]:
    """Name the latent columns of a model of *zdims* dimensions, in order."""
    return [f"{LATENT_PREFIX}{dimension}" for dimension in range(zdims)]


def min_session_frames(time_window: int, purpose: SessionPurpose) -> int:
    """Return the fewest frames of a session that VAME can use for *purpose*.

    An embedding needs one window of *time_window* frames. A fit needs
    ``2 * time_window + 1``, the fewest frames that contain a training window,
    as ``vame_protocol.least_training_frames`` returns.
    """
    if purpose == "fit":
        return least_training_frames(time_window)
    return time_window


@dataclass(frozen=True)
class SessionSettings:
    """The settings that decide VAME's sessions.

    Attributes:
        pose: The selected keypoints and their columns.
        centered_reference_keypoint: The keypoint that egocentric alignment moves
            to the origin, by index or by name.
        orientation_reference_keypoint: The keypoint whose direction alignment
            fixes, by index or by name.
        run_lowconf_cleaning: Whether VAME replaces a point below
            *pose_confidence* by interpolation. Without it, VAME keeps every
            finite point, and a missing point stays missing.
        pose_confidence: The confidence below which low-confidence cleaning
            replaces a point.
        max_gap_frames: The longest run of frames without a kept point of a
            reference keypoint that a session spans for VAME to interpolate.
            Without low-confidence cleaning, every frame with a missing point
            ends its session, and this setting is not read.
        time_window: The frames of one VAME window.
    """

    pose: PoseConfig
    centered_reference_keypoint: int | str
    orientation_reference_keypoint: int | str
    run_lowconf_cleaning: bool
    pose_confidence: float
    max_gap_frames: int
    time_window: int


@dataclass(frozen=True)
class VameKeypoints:
    """The keypoints that VAME reads, in the order of a session's keypoint axis.

    Attributes:
        suffixes: Each keypoint's column suffix, its index as a string.
        names: Each keypoint's name in VAME's project.
        centered: The name of the centered reference keypoint.
        orientation: The name of the orientation reference keypoint.
    """

    suffixes: tuple[str, ...]
    names: tuple[str, ...]
    centered: str
    orientation: str

    @property
    def reference_positions(self) -> tuple[int, int]:
        """The keypoint-axis positions of the centered and orientation references."""
        return self.names.index(self.centered), self.names.index(self.orientation)


@dataclass(frozen=True, eq=False)
class SessionTrack:
    """One session: one individual's keypoints on contiguous frames.

    Attributes:
        individual: The individual's id, or ``None`` for a table without an id
            column.
        track: The keypoints, on every frame from the first to the last. A frame
            without a row has NaN positions and confidence 0.
    """

    individual: Scalar | None
    track: KeypointArrays

    @property
    def first_frame(self) -> int:
        """The session's first frame."""
        return int(self.track.frames[0])

    @property
    def last_frame(self) -> int:
        """The session's last frame."""
        return int(self.track.frames[-1])

    @property
    def n_frames(self) -> int:
        """The frames of the session."""
        return len(self.track.frames)


@dataclass(frozen=True)
class Exclusion:
    """Frames of one individual that VAME cannot use, and the reason.

    Attributes:
        individual: The individual's id, or ``None`` for a table without an id
            column.
        first_frame: The first excluded frame.
        last_frame: The last excluded frame.
        reason: Why VAME cannot use the frames.
    """

    individual: Scalar | None
    first_frame: int
    last_frame: int
    reason: str

    def describe(self) -> str:
        """Return the exclusion as one line: the frames, the id and the reason."""
        frames = _frames_of(self.individual, self.first_frame, self.last_frame)
        return f"{frames}: {self.reason}"


@dataclass(frozen=True)
class GapRows:
    """The rows of an entry at gap frames that no session or exclusion spans.

    A run of gap frames longer than ``max_gap_frames``, or at either end of an
    individual's frames, lies outside every session, and its rows do not receive a
    latent. Without low-confidence cleaning, every run of gap frames lies outside.

    Attributes:
        count: The rows, over every individual of the entry.
        reason: Why VAME cannot use the rows, stated as the condition at their
            frames.
    """

    count: int
    reason: str

    def describe(self) -> str:
        """Return the rows' count and the condition at their frames, as one line."""
        return f"{counted(self.count, 'row')} at frames where {self.reason}"


@dataclass(frozen=True)
class EntrySessions:
    """An entry's sessions, its excluded frames, and its rows outside both.

    The sessions and the exclusions are ordered by the text of the id, then by the
    first frame.

    Attributes:
        sessions: The sessions that VAME can use.
        excluded: The frames that VAME cannot use, with their reasons.
        gap_rows: The rows at gap frames outside every session and exclusion.
    """

    sessions: tuple[SessionTrack, ...]
    excluded: tuple[Exclusion, ...]
    gap_rows: GapRows

    def describe_unused(self) -> list[str]:
        """Describe each exclusion in order, then the gap rows when there are any."""
        lines = [exclusion.describe() for exclusion in self.excluded]
        if self.gap_rows.count:
            lines.append(self.gap_rows.describe())
        return lines


def vame_keypoints(settings: SessionSettings) -> VameKeypoints:
    """Name the selected keypoints, and resolve the two reference keypoints.

    A keypoint is named from ``pose.keypoint_names``, or ``kp<index>`` when the
    pose names none. A reference is a keypoint index or name, and must be one of
    the selected keypoints.

    Raises:
        ValueError: When a selected keypoint lacks a name, two selected keypoints
            share a name, a reference is not a selected keypoint, or both
            references are one keypoint.
    """
    pose = settings.pose
    selected = pose.selected_indices()
    all_names = pose.names(max(selected, default=-1) + 1)
    unnamed = [index for index in selected if not 0 <= index < len(all_names)]
    if unnamed:
        msg = (
            f"vame: the keypoints {unnamed} are selected, and the keypoint names "
            f"cover the keypoints 0 to {len(all_names) - 1}"
        )
        raise ValueError(msg)
    names = tuple(all_names[index] for index in selected)
    repeated = repeated_names(names)
    if repeated:
        msg = (
            f"vame: the selected keypoints {list(names)} need distinct names, and "
            f"{repeated} repeat"
        )
        raise ValueError(msg)
    [centered] = pose.resolve_keypoints(
        [settings.centered_reference_keypoint], who="vame centered_reference_keypoint"
    )
    [orientation] = pose.resolve_keypoints(
        [settings.orientation_reference_keypoint],
        who="vame orientation_reference_keypoint",
    )
    try:
        check_distinct_references(names[centered], names[orientation])
    except ValueError as error:
        msg = f"vame: {error}"
        raise ValueError(msg) from error
    return VameKeypoints(
        suffixes=tuple(str(index) for index in selected),
        names=names,
        centered=names[centered],
        orientation=names[orientation],
    )


def build_sessions(
    df: pd.DataFrame, settings: SessionSettings, purpose: SessionPurpose
) -> EntrySessions:
    """Build VAME's sessions from one entry's track table.

    Each individual's selected keypoints are read with their confidence, from
    ``pose.confidence_prefix`` when every selected keypoint has a score column and
    1 otherwise, and reindexed onto contiguous frames. A point is kept when its
    position is finite and, under low-confidence cleaning, its confidence is not
    below ``pose_confidence``. Under cleaning, a frame where either reference
    keypoint lacks a kept point is a gap frame, and a session keeps the missing
    points of the other keypoints for VAME to interpolate. Without cleaning, a
    frame with any point not kept is a gap frame. A run of gap frames longer than
    ``max_gap_frames`` splits the individual, and without cleaning every gap frame
    splits it. Gap frames at either end of a session are dropped. The rows at gap
    frames outside every session and exclusion are counted in ``gap_rows``.

    A session is excluded when it is shorter than :func:`min_session_frames`, or
    when one of its keypoints lacks a kept point. An individual whose every frame
    is a gap frame is excluded whole.

    Args:
        df: The entry's rows. Left unchanged.
        settings: The settings that decide the sessions.
        purpose: The use of the sessions.

    Returns:
        The sessions and the exclusions, ordered by the text of the id, then the
        first frame, and the rows outside every session and exclusion.

    Raises:
        ValueError: When the table lacks keypoint columns or the columns of a
            selected keypoint; when its rows come from more than one camera;
            when a row lacks an id; when two rows share a frame and an id; when an
            id's rows are a median of more than one frame apart; and as
            :func:`vame_keypoints` raises.
    """
    keypoints = vame_keypoints(settings)
    pose = settings.pose
    _check_table(df, pose, keypoints)
    sessions: list[SessionTrack] = []
    excluded: list[Exclusion] = []
    gap_rows = 0
    for individual, rows in iter_individuals(df):
        row_frames = rows[COLUMNS.frame_col].to_numpy(dtype=np.int64)
        _check_frame_steps(individual, row_frames)
        track = reindex_contiguous(
            keypoint_arrays(
                rows,
                x_prefix=pose.x_prefix,
                y_prefix=pose.y_prefix,
                confidence_prefix=pose.confidence_prefix,
                suffixes=keypoints.suffixes,
            )
        )
        found, lost, gap_count = _split(
            individual, track, row_frames, settings, keypoints, purpose
        )
        sessions.extend(found)
        excluded.extend(lost)
        gap_rows += gap_count
    sessions.sort(key=lambda session: (str(session.individual), session.first_frame))
    excluded.sort(
        key=lambda exclusion: (str(exclusion.individual), exclusion.first_frame)
    )
    return EntrySessions(
        sessions=tuple(sessions),
        excluded=tuple(excluded),
        gap_rows=GapRows(count=gap_rows, reason=_gap_reason(settings, keypoints)),
    )


def place_latents(
    df: pd.DataFrame,
    entry: EntrySessions,
    latents: Sequence[npt.NDArray[np.floating]],
    *,
    time_window: int,
) -> pd.DataFrame:
    """Place each session's latents on the rows of *df* at their windows' centers.

    Latent ``i`` of a session that starts at frame ``f0`` is placed on the row of
    the session's individual at frame ``f0 + i + time_window // 2``. Rows at
    other frames are left out. A center frame inside a gap that VAME interpolated
    lacks a row, and its latent is dropped.

    Args:
        df: The entry's rows, as :func:`build_sessions` read them.
        entry: The sessions built from *df*.
        latents: Each session's latents, in the order of ``entry.sessions``, one
            row per window.
        time_window: The frames of one VAME window.

    Returns:
        The identity columns of the rows that received a latent, in the order of
        *df*, followed by ``vame_z0`` to ``vame_z{zdims-1}`` as float32.

    Raises:
        ValueError: When *latents* lacks an array for a session, an array's rows
            differ from its session's windows, or two arrays differ in width. When
            every row lacks a latent, naming the excluded frames and the gap rows.
    """
    if len(latents) != len(entry.sessions):
        msg = (
            f"vame: {counted(len(latents), 'latent array')} for "
            f"{counted(len(entry.sessions), 'session')}, and each session needs one"
        )
        raise ValueError(msg)
    rows = df.reset_index(drop=True)
    frames = rows[COLUMNS.frame_col].to_numpy(dtype=np.int64)
    positions_of = _row_positions(rows)
    vectors = [np.asarray(values) for values in latents]
    row_latent = np.full(len(rows), -1, dtype=np.int64)
    offset = 0
    for session, values in zip(entry.sessions, vectors, strict=True):
        windows = session.n_frames - time_window + 1
        if values.ndim != 2 or len(values) != windows:
            msg = (
                f"vame: the latents of {_session_frames(session)} have shape "
                f"{values.shape}, and those frames contain "
                f"{counted(windows, 'window')} of time_window={time_window}"
            )
            raise ValueError(msg)
        # The first array passed the shape check in the first iteration.
        if values.shape[1] != vectors[0].shape[1]:
            msg = (
                f"vame: the latents of {_session_frames(session)} have "
                f"{counted(values.shape[1], 'dimension')}, and those of "
                f"{_session_frames(entry.sessions[0])} have {vectors[0].shape[1]}"
            )
            raise ValueError(msg)
        positions = positions_of[session.individual]
        index = frames[positions] - (session.first_frame + time_window // 2)
        inside = (index >= 0) & (index < windows)
        row_latent[positions[inside]] = offset + index[inside]
        offset += windows
    received = np.flatnonzero(row_latent >= 0)
    if received.size == 0:
        unused = "; ".join(entry.describe_unused())
        msg = (
            "vame: no row receives a latent. Sessions embedded: "
            f"{len(entry.sessions)}. Excluded: {unused or 'nothing'}"
        )
        raise ValueError(msg)
    stacked = np.concatenate(vectors)
    placed = stacked[row_latent[received]].astype(np.float32)
    identity = rows.iloc[received][meta_columns(rows)].reset_index(drop=True)
    values_frame = pd.DataFrame(placed, columns=latent_columns(placed.shape[1]))
    return pd.concat([identity, values_frame], axis=1)


def write_session_tracks(
    sessions: Sequence[SessionTrack], directory: Path
) -> dict[str, Path]:
    """Write each session's track to a file in *directory* that the runner reads.

    Session ``i`` is named ``f"s{i:06d}"``, from ``s000000``. Its file is
    ``<name>.npz``, in the format that ``vame_protocol.Session`` describes. The
    order of *sessions* sets VAME's split of a fit's training and test frames. A fit
    passes its entries' sessions ordered by ``(group, sequence)``, and each entry's
    in the order that :func:`build_sessions` returns them.

    Returns:
        Each session's file by its name, in the order of *sessions*.
    """
    written: dict[str, Path] = {}
    for index, session in enumerate(sessions):
        name = f"s{index:06d}"
        path = directory / f"{name}.npz"
        atomic_savez(
            path,
            frames=session.track.frames,
            position=session.track.position,
            confidence=session.track.confidence,
        )
        written[name] = path
    return written


def _check_table(df: pd.DataFrame, pose: PoseConfig, keypoints: VameKeypoints) -> None:
    """Refuse a table that VAME's sessions cannot be built from."""
    if not any(str(column).startswith(pose.x_prefix) for column in df.columns):
        msg = (
            f"vame: the table lacks keypoint columns ({pose.x_prefix}*), and VAME "
            "embeds keypoints"
        )
        raise ValueError(msg)
    wanted = [
        f"{prefix}{suffix}"
        for suffix in keypoints.suffixes
        for prefix in (pose.x_prefix, pose.y_prefix)
    ]
    missing = [column for column in wanted if column not in df.columns]
    if missing:
        msg = (
            f"vame: the table lacks {missing}, the columns of the selected "
            f"keypoints {list(keypoints.suffixes)}"
        )
        raise ValueError(msg)
    if _CAMERA_COL in df.columns:
        cameras = pd.unique(df[_CAMERA_COL])
        if len(cameras) > 1:
            msg = (
                f"vame: the entry contains the rows of the cameras "
                f"{sorted(str(camera) for camera in cameras)}, and VAME reads the "
                "rows of one camera"
            )
            raise ValueError(msg)
    if COLUMNS.id_col in df.columns:
        n_without = int(df[COLUMNS.id_col].isna().sum())
        if n_without:
            verb = "lacks" if n_without == 1 else "lack"
            msg = f"vame: {counted(n_without, 'row')} {verb} an id"
            raise ValueError(msg)
    refuse_repeated_rows(df, "vame")


def _check_frame_steps(
    individual: Scalar | None, row_frames: npt.NDArray[np.int64]
) -> None:
    """Refuse an individual whose rows are a median of more than one frame apart."""
    steps = np.diff(row_frames)
    if steps.size == 0:
        return
    median = float(np.median(steps))
    if median > 1:
        who = "the table" if individual is None else f"id {individual}"
        msg = (
            f"vame: {who} steps a median of {median:g} frames between rows. VAME "
            "reads consecutive frames, and would interpolate the frames between "
            "them"
        )
        raise ValueError(msg)


def _kept_points(
    track: KeypointArrays, settings: SessionSettings
) -> npt.NDArray[np.bool_]:
    """Mark the points that VAME's preprocessing keeps, ``(T, K)``."""
    kept = np.isfinite(track.position).all(axis=2)
    if settings.run_lowconf_cleaning:
        # VAME discards a point whose confidence is below the threshold, and
        # compares in float64. A NaN confidence is not below it.
        low = track.confidence.astype(np.float64) < settings.pose_confidence
        kept &= ~low
    return kept


def _split(
    individual: Scalar | None,
    track: KeypointArrays,
    row_frames: npt.NDArray[np.int64],
    settings: SessionSettings,
    keypoints: VameKeypoints,
    purpose: SessionPurpose,
) -> tuple[list[SessionTrack], list[Exclusion], int]:
    """Split one individual's contiguous track into sessions and exclusions.

    Returns:
        The sessions, the exclusions, and the count of the rows at *row_frames*
        that neither spans.
    """
    if track.frames.size == 0:
        return [], [], 0
    kept = _kept_points(track, settings)
    if settings.run_lowconf_cleaning:
        centered, orientation = keypoints.reference_positions
        usable_frames = np.flatnonzero(kept[:, centered] & kept[:, orientation])
        allowed_gap = settings.max_gap_frames
    else:
        usable_frames = np.flatnonzero(kept.all(axis=1))
        allowed_gap = 0
    if usable_frames.size == 0:
        exclusion = Exclusion(
            individual=individual,
            first_frame=int(track.frames[0]),
            last_frame=int(track.frames[-1]),
            reason=_no_usable_frame(settings, keypoints),
        )
        return [], [exclusion], 0
    breaks = np.flatnonzero(np.diff(usable_frames) > allowed_gap + 1) + 1
    least = min_session_frames(settings.time_window, purpose)
    sessions: list[SessionTrack] = []
    excluded: list[Exclusion] = []
    spanned = np.zeros(track.frames.size, dtype=np.bool_)
    for run in np.split(usable_frames, breaks):
        start, stop = int(run[0]), int(run[-1]) + 1
        spanned[start:stop] = True
        segment = KeypointArrays(
            frames=track.frames[start:stop],
            position=track.position[start:stop],
            confidence=track.confidence[start:stop],
        )
        reason = _exclusion_reason(
            stop - start, kept[start:stop], least, settings, keypoints, purpose
        )
        if reason is None:
            sessions.append(SessionTrack(individual=individual, track=segment))
        else:
            excluded.append(
                Exclusion(
                    individual=individual,
                    first_frame=int(segment.frames[0]),
                    last_frame=int(segment.frames[-1]),
                    reason=reason,
                )
            )
    gap_count = int(np.count_nonzero(~spanned[row_frames - track.frames[0]]))
    return sessions, excluded, gap_count


def _no_usable_frame(settings: SessionSettings, keypoints: VameKeypoints) -> str:
    if settings.run_lowconf_cleaning:
        return f"at every frame, {_reference_lacks(settings, keypoints)}"
    return (
        "every frame lacks a finite point of at least one selected keypoint, and "
        f"{_NO_INTERPOLATION}"
    )


def _gap_reason(settings: SessionSettings, keypoints: VameKeypoints) -> str:
    """Name the gap frames that lie outside every session, for :class:`GapRows`."""
    if settings.run_lowconf_cleaning:
        return (
            f"{_reference_lacks(settings, keypoints)}, in a run longer than "
            f"max_gap_frames={settings.max_gap_frames} or at either end of an id's "
            "frames"
        )
    return f"a selected keypoint lacks a finite point, and {_NO_INTERPOLATION}"


def _reference_lacks(settings: SessionSettings, keypoints: VameKeypoints) -> str:
    """Name the condition of a gap frame under low-confidence cleaning."""
    return (
        f"reference keypoint {keypoints.centered!r} or {keypoints.orientation!r} "
        f"lacks {_kept_point(settings)}"
    )


def _kept_point(settings: SessionSettings) -> str:
    """Name a point that low-confidence cleaning keeps."""
    return (
        "a finite point with confidence at or above "
        f"pose_confidence={settings.pose_confidence:g}"
    )


def _exclusion_reason(
    n_frames: int,
    kept: npt.NDArray[np.bool_],
    least: int,
    settings: SessionSettings,
    keypoints: VameKeypoints,
    purpose: SessionPurpose,
) -> str | None:
    """Say why VAME cannot use a session, or return None when it can."""
    if n_frames < least:
        needed = (
            f"2 * time_window + 1 = {least}, which one training window needs"
            if purpose == "fit"
            else f"time_window={settings.time_window}"
        )
        return f"{counted(n_frames, 'frame')}, fewer than {needed}"
    unkept = [
        name
        for name, any_kept in zip(keypoints.names, kept.any(axis=0))
        if not any_kept
    ]
    if unkept:
        named = "keypoint" if len(unkept) == 1 else "keypoints"
        verb = "lacks" if len(unkept) == 1 else "lack"
        return f"{named} {unkept} {verb} {_kept_point(settings)}"
    return None


def _frames_of(individual: Scalar | None, first_frame: int, last_frame: int) -> str:
    """Name an individual's frames, as ``frames 3 to 9 of id 2``."""
    frames = f"frames {first_frame} to {last_frame}"
    if individual is None:
        return frames
    return f"{frames} of id {individual}"


def _session_frames(session: SessionTrack) -> str:
    return _frames_of(session.individual, session.first_frame, session.last_frame)


def _row_positions(rows: pd.DataFrame) -> dict[Hashable, npt.NDArray[np.intp]]:
    """Map each individual of *rows* to the positions of its rows."""
    if COLUMNS.id_col not in rows.columns:
        return {None: np.arange(len(rows), dtype=np.intp)}
    groups = rows.groupby(COLUMNS.id_col, sort=False).indices
    return {
        individual: np.asarray(positions, dtype=np.intp)
        for individual, positions in groups.items()
    }
