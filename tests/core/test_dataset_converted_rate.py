"""An imported table is timed at its entry's media rate, else the dataset's, else refused.

A SLEAP analysis file records no frame rate. ``convert_all_tracks`` times each entry
at the explicit ``fps`` parameter, else the rate that the entry's indexed media
measured, else ``meta.fps_default``. An entry with none of the three fails alone and
the run reports ``partial``, or ``failed`` when every entry does, instead of writing
a ``time`` column at an assumed rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.track_converter import FrameRateUnknownError
from tests.helpers import (
    MediaClip,
    make_dataset,
    write_media_index,
    write_sleap_analysis_h5,
)

FRAMES = 6


def _dataset(
    tmp_path: Path,
    *,
    media: tuple[MediaClip, ...] = (),
    sequences: tuple[str, ...] = ("s1",),
    fps_default: float | None = None,
) -> Dataset:
    """A dataset with a SLEAP file per sequence in ``tracks_raw``, indexed."""
    dataset = make_dataset(tmp_path / "ds")
    if fps_default is not None:
        dataset.meta["fps_default"] = fps_default
    if media:
        write_media_index(dataset, list(media))
    raw = dataset.get_root("tracks_raw")
    for sequence in sequences:
        write_sleap_analysis_h5(
            raw / f"{sequence}.h5",
            np.random.default_rng(0).random((FRAMES, 1, 1, 2)),
        )
    _ = dataset.index_tracks_raw(
        [raw], patterns=["*.h5"], src_format="sleap_analysis_h5"
    )
    return dataset


def _time_step(dataset: Dataset, sequence: str = "s1") -> float:
    table = dataset.load_tracks("", sequence)
    return float(np.median(np.diff(np.asarray(table["time"], dtype=float))))


def _clip(sequence: str = "s1", *, fps: float, frame_count: int = FRAMES) -> MediaClip:
    return MediaClip(
        sequence=sequence,
        filename=f"{sequence}.mp4",
        fps=fps,
        frame_count=frame_count,
    )


@pytest.mark.parametrize("merge", [False, True], ids=["single-file", "merged"])
def test_media_at_15_fps_times_the_table_without_a_default(
    tmp_path: Path, merge: bool
) -> None:
    dataset = _dataset(tmp_path, media=(_clip(fps=15.0),))

    outcome = dataset.convert_all_tracks(merge_per_sequence=merge)

    assert outcome.ok
    table = dataset.load_tracks("", "s1")
    np.testing.assert_allclose(
        np.asarray(table["time"], dtype=float),
        np.asarray(table["frame"], dtype=float) / 15.0,
    )


@pytest.mark.parametrize("merge", [False, True], ids=["single-file", "merged"])
def test_an_entry_with_no_rate_fails_alone_and_the_run_is_partial(
    tmp_path: Path, merge: bool
) -> None:
    dataset = _dataset(
        tmp_path,
        media=(_clip("s2", fps=15.0),),
        sequences=("s1", "s2"),
    )
    failures: list[tuple[str, BaseException]] = []

    outcome = dataset.convert_all_tracks(
        merge_per_sequence=merge,
        on_failure=lambda path, exc: failures.append((path, exc)),
    )

    assert not outcome.ok
    assert outcome.failed == 1
    ((path, error),) = failures
    assert path.endswith("s1.h5")
    assert isinstance(error, FrameRateUnknownError)
    assert sorted(read_tracks_index(dataset)["sequence"]) == ["s2"]
    assert _time_step(dataset, "s2") == pytest.approx(1 / 15.0)


def test_a_run_whose_every_entry_has_no_rate_converts_nothing(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)

    outcome = dataset.convert_all_tracks()

    assert outcome.failed == 1
    assert not outcome.ok
    assert read_tracks_index(dataset).empty


def test_the_cli_fails_when_the_only_entry_is_refused(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "convert-tracks",
            "-m",
            str(dataset.manifest_path),
            "--source-format",
            "sleap_analysis_h5",
            "--json",
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert (payload["converted"], payload["failed"]) == (0, 1)


def test_the_default_times_an_entry_without_media(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, fps_default=24.0)

    outcome = dataset.convert_all_tracks()

    assert outcome.ok
    assert _time_step(dataset) == pytest.approx(1 / 24.0)


def test_the_media_rate_outranks_the_default(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        media=(_clip(fps=15.0),),
        fps_default=24.0,
    )

    outcome = dataset.convert_all_tracks()

    assert outcome.ok
    assert _time_step(dataset) == pytest.approx(1 / 15.0)


def test_an_explicit_rate_outranks_the_media(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, media=(_clip(fps=15.0),))

    outcome = dataset.convert_all_tracks(params={"fps": 25.0})

    assert outcome.ok
    assert _time_step(dataset) == pytest.approx(1 / 25.0)


def test_the_legacy_default_key_is_an_explicit_rate(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, media=(_clip(fps=15.0),))

    outcome = dataset.convert_all_tracks(params={"fps_default": 20.0})

    assert outcome.ok
    assert _time_step(dataset) == pytest.approx(1 / 20.0)


def test_the_measured_rate_does_not_move_the_variant(tmp_path: Path) -> None:
    with_media = _dataset(tmp_path / "a", media=(_clip(fps=15.0),), fps_default=24.0)
    without_media = _dataset(tmp_path / "b", fps_default=24.0)

    assert with_media.track_conversion_variant(
        "sleap_analysis_h5"
    ) == without_media.track_conversion_variant("sleap_analysis_h5")


# --- the entry's measured rate ---------------------------------------------


def test_the_measured_rate_of_an_entry_without_media_is_unknown(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)

    assert dataset.measured_entry_fps("", "s1") is None


def test_the_measured_rate_of_an_entry_without_a_media_index_is_unknown(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")

    assert dataset.measured_entry_fps("", "s1") is None


def test_the_measured_rate_is_the_clips_rate_when_they_agree(tmp_path: Path) -> None:
    clips = tuple(
        MediaClip(
            sequence="s1",
            filename=f"s1-{order}.mp4",
            video_order=order,
            video_uuid=f"u{order}",
            fps=rate,
            frame_count=FRAMES,
        )
        for order, rate in enumerate((15.0, 15.001))
    )
    dataset = _dataset(tmp_path, media=clips)

    assert dataset.measured_entry_fps("", "s1") == pytest.approx(15.0)


def test_the_measured_rate_is_unknown_when_the_clips_disagree(tmp_path: Path) -> None:
    clips = tuple(
        MediaClip(
            sequence="s1",
            filename=f"s1-{order}.mp4",
            video_order=order,
            video_uuid=f"u{order}",
            fps=rate,
            frame_count=FRAMES,
        )
        for order, rate in enumerate((15.0, 30.0))
    )
    dataset = _dataset(tmp_path, media=clips)

    assert dataset.measured_entry_fps("", "s1") is None


def test_the_measured_rate_is_unknown_when_a_clip_is_unmeasured(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, media=(_clip(fps=0.0),))

    assert dataset.measured_entry_fps("", "s1") is None


@pytest.mark.parametrize(
    ("rates", "expected"),
    [((15.0, 15.0), 15.0), ((15.0, 30.0), None)],
    ids=["same-rate", "different-rates"],
)
def test_the_cameras_of_an_entry_share_a_rate_or_leave_it_unknown(
    tmp_path: Path, rates: tuple[float, float], expected: float | None
) -> None:
    clips = tuple(
        MediaClip(
            sequence="s1",
            filename=f"s1-{camera}.mp4",
            camera=camera,
            video_uuid=f"u-{camera}",
            fps=rate,
            frame_count=FRAMES,
        )
        for camera, rate in zip(("a", "b"), rates, strict=True)
    )
    dataset = _dataset(tmp_path, media=clips)

    assert dataset.measured_entry_fps("", "s1") == expected


# --- a continuous group --------------------------------------------------------


def test_a_member_of_a_continuous_group_is_timed_at_the_media_rate_without_a_shift(
    tmp_path: Path,
) -> None:
    """Only a division is moved onto its recording's axes at conversion.

    A member of a declared group that is not a division starts at its own frame 0
    in a converter's table, and the measured rate times that table. Setting the rate
    does not shift ``time`` by the media frames of the members before it.
    """
    from mosaic.core.pipeline.tracks_raw_index import TracksRawIndexScope

    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset,
        [
            MediaClip(
                sequence=name,
                group="rec",
                filename=f"{name}.mp4",
                fps=15.0,
                frame_count=FRAMES,
            )
            for name in ("s1", "s2")
        ],
    )
    dataset.set_continuous_groups(["rec"])
    raw = dataset.get_root("tracks_raw")
    for name in ("s1", "s2"):
        write_sleap_analysis_h5(
            raw / name / f"{name}.h5",
            np.random.default_rng(0).random((FRAMES, 1, 1, 2)),
        )
    _ = dataset.write_tracks_raw_index(
        [
            TracksRawIndexScope(
                directory=raw / name,
                group="rec",
                sequence=name,
                src_format="sleap_analysis_h5",
            )
            for name in ("s1", "s2")
        ],
        patterns="*.h5",
    )

    outcome = dataset.convert_all_tracks()

    assert outcome.ok
    table = dataset.load_tracks("rec", "s2")
    assert list(table["frame"]) == list(range(FRAMES))
    np.testing.assert_allclose(
        np.asarray(table["time"], dtype=float),
        np.asarray(table["frame"], dtype=float) / 15.0,
    )


# --- the media index is read once ----------------------------------------------


def _count_media_index_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the reads of the media indexes, which each load takes with its routing."""
    reads: list[int] = []
    original = Dataset.media_routing_context

    def counted(
        self: Dataset, index_filename: str = "index.csv"
    ) -> tuple[bool, pd.DataFrame | None]:
        reads.append(1)
        return original(self, index_filename)

    monkeypatch.setattr(Dataset, "media_routing_context", counted)
    return reads


def test_the_media_index_is_read_once_for_many_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(
        tmp_path,
        media=(_clip("s1", fps=15.0), _clip("s2", fps=15.0)),
        sequences=("s1", "s2"),
    )
    reads = _count_media_index_reads(monkeypatch)

    assert dataset.measured_entry_fps("", "s1") == 15.0
    assert dataset.measured_entry_fps("", "s2") == 15.0

    assert len(reads) == 1


def test_a_rewritten_media_index_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, media=(_clip("s1", fps=15.0),))
    reads = _count_media_index_reads(monkeypatch)
    assert dataset.measured_entry_fps("", "s1") == 15.0

    write_media_index(dataset, [_clip("s1", fps=20.0)])

    assert dataset.measured_entry_fps("", "s1") == 20.0
    assert len(reads) == 2
