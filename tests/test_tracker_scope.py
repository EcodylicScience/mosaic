"""What a resolved media scope becomes: one work item per entry, and what it holds.

This path had no direct coverage at all, which is how it stayed true for years
that a sequence a recorder split into clips was tracked from its first clip and
the rest were dropped with a line on stderr.

Two collapses are asserted here, and they are different things. A tracker that
can read several clips as one video gets all of them; one that cannot is
truncated *and says so*. Cameras collapse regardless, because the working
directory is keyed without a camera.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.tracking.common.scope import (
    JoinedSourceMismatchError,
    TrackerWorkItem,
    build_work_items,
)

TREX = "trex"
"""A tracker that declares ``joins_sources``."""

SLEAP = "sleap"
"""One that does not, and must keep behaving exactly as it did."""


@dataclass
class Clip:
    """One row of the synthetic media index."""

    filename: str
    order: int = 0
    group: str = ""
    sequence: str = "sess"
    camera: str = ""
    video_uuid: str = ""
    fps: float = 30.0
    width: int = 640
    height: int = 480
    rotation: int = 0
    frame_count: int = 100


def _facts_cells(clip: Clip) -> dict[str, object]:
    """Flat + JSON facts cells for one analysis-clean media row."""
    facts: MediaFacts = store_facts(
        width=clip.width,
        height=clip.height,
        fps=clip.fps,
        frame_count=clip.frame_count,
        codec="h264",
        duration=clip.frame_count / clip.fps if clip.fps else 0.0,
        video_uuid=clip.video_uuid,
        identity_scheme="video/1" if clip.video_uuid else "",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
        rotation_degrees=clip.rotation,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def _dataset(tmp_path: Path, clips: list[Clip]) -> Dataset:
    """A dataset whose media index holds exactly *clips*."""
    manifest = new_dataset_manifest("scope", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    media_root = ds.get_root(ds.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for clip in clips:
        video = media_root / clip.filename
        if not video.exists():
            _ = video.write_bytes(b"fake")
        rows.append(
            {
                "name": clip.filename,
                "group": clip.group,
                "sequence": clip.sequence,
                "group_safe": clip.group,
                "sequence_safe": clip.sequence,
                "camera": clip.camera,
                "abs_path": ds.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": clip.width,
                "height": clip.height,
                "fps": clip.fps,
                "codec": "h264",
                "media_type": "video",
                "video_order": clip.order,
                **_facts_cells(clip),
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)
    return ds


def _items(ds: Dataset, *, kind: str) -> list[TrackerWorkItem]:
    return build_work_items(ds, ds.resolve_media_scope(None), kind=kind)


# The measured shape of one real session, shortened.
SESSION = [
    Clip(filename="c0.mp4", order=0, video_uuid="uid-0", fps=30.0),
    Clip(filename="c1.mp4", order=1, video_uuid="uid-1", fps=29.95),
    Clip(filename="c2.mp4", order=2, video_uuid="uid-2", fps=31.0),
]


class TestOneClip:
    def test_the_ordinary_sequence_is_untouched(self, tmp_path: Path) -> None:
        ds = _dataset(tmp_path, [Clip(filename="v.mp4", video_uuid="uid-v")])
        (item,) = _items(ds, kind=TREX)
        assert item.n_sources == 1
        assert item.video_path.name == "v.mp4"
        assert item.fps == 30.0

    def test_its_reuse_key_is_still_the_video_uuid(self, tmp_path: Path) -> None:
        """The proof that nothing already on disk is invalidated."""
        ds = _dataset(tmp_path, [Clip(filename="v.mp4", video_uuid="uid-v")])
        (item,) = _items(ds, kind=TREX)
        assert item.source_uid == "uid-v"
        assert item.video_uid == "uid-v"

    def test_an_unmeasured_rate_still_falls_back(self, tmp_path: Path) -> None:
        """Only a *joined* entry refuses a missing rate; one clip defaults."""
        ds = _dataset(tmp_path, [Clip(filename="v.mp4", fps=0.0)])
        (item,) = _items(ds, kind=TREX)
        assert item.fps == 30.0


class TestSeveralClips:
    def test_a_joining_tracker_gets_them_all_in_order(self, tmp_path: Path) -> None:
        ds = _dataset(tmp_path, SESSION)
        (item,) = _items(ds, kind=TREX)
        assert item.n_sources == 3
        assert [p.name for p in item.video_paths] == ["c0.mp4", "c1.mp4", "c2.mp4"]
        assert item.video_uids == ("uid-0", "uid-1", "uid-2")

    def test_a_joining_tracker_warns_about_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = _dataset(tmp_path, SESSION)
        _ = _items(ds, kind=TREX)
        assert capsys.readouterr().err == ""

    def test_a_non_joining_tracker_is_truncated_and_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = _dataset(tmp_path, SESSION)
        (item,) = _items(ds, kind=SLEAP)
        assert item.n_sources == 1
        assert item.video_path.name == "c0.mp4"
        assert item.source_uid == "uid-0"
        assert "3 videos" in capsys.readouterr().err

    def test_the_frame_rate_stays_the_first_clips(self, tmp_path: Path) -> None:
        """Never a mean: the trackers that read it track clip 0 only."""
        ds = _dataset(tmp_path, SESSION)
        (item,) = _items(ds, kind=TREX)
        assert item.fps == 30.0

    def test_a_heterogeneous_rate_is_carried_not_refused(self, tmp_path: Path) -> None:
        """30 / 29.95 / 31 is a real session and must survive the scope."""
        ds = _dataset(tmp_path, SESSION)
        (item,) = _items(ds, kind=TREX)
        assert [f.fps for f in item.source_facts] == [30.0, 29.95, 31.0]


class TestTheReuseKey:
    def test_several_clips_key_on_the_ordered_composition(self, tmp_path: Path) -> None:
        ds = _dataset(tmp_path, SESSION)
        (item,) = _items(ds, kind=TREX)
        assert item.source_uid != ""
        assert item.source_uid != item.video_uid

    def test_reordering_the_clips_changes_it(self, tmp_path: Path) -> None:
        forward = _dataset(tmp_path / "a", SESSION)
        swapped = _dataset(
            tmp_path / "b",
            [
                dataclasses.replace(SESSION[1], order=0),
                dataclasses.replace(SESSION[0], order=1),
                SESSION[2],
            ],
        )
        (first,) = _items(forward, kind=TREX)
        (second,) = _items(swapped, kind=TREX)
        assert first.source_uid != second.source_uid

    def test_adding_a_clip_changes_it(self, tmp_path: Path) -> None:
        two = _dataset(tmp_path / "a", SESSION[:2])
        three = _dataset(tmp_path / "b", SESSION)
        (short,) = _items(two, kind=TREX)
        (long,) = _items(three, kind=TREX)
        assert short.source_uid != long.source_uid

    def test_an_unidentified_clip_empties_it(self, tmp_path: Path) -> None:
        """Unestablishable, which sends the gate to its path fallback."""
        ds = _dataset(
            tmp_path,
            [SESSION[0], dataclasses.replace(SESSION[1], video_uuid="")],
        )
        (item,) = _items(ds, kind=TREX)
        assert item.source_uid == ""


class TestRefusals:
    def test_a_differing_resolution_is_refused_by_name(self, tmp_path: Path) -> None:
        ds = _dataset(
            tmp_path,
            [SESSION[0], dataclasses.replace(SESSION[1], width=1280)],
        )
        with pytest.raises(JoinedSourceMismatchError, match="c1.mp4"):
            _ = _items(ds, kind=TREX)

    def test_an_unmeasured_rate_among_several_is_refused(self, tmp_path: Path) -> None:
        ds = _dataset(
            tmp_path,
            [SESSION[0], dataclasses.replace(SESSION[1], fps=0.0)],
        )
        with pytest.raises(JoinedSourceMismatchError, match="no frame rate"):
            _ = _items(ds, kind=TREX)

    def test_a_non_joining_tracker_is_not_refused(self, tmp_path: Path) -> None:
        """It only ever sees clip 0, so the others cannot make it fail."""
        ds = _dataset(
            tmp_path,
            [SESSION[0], dataclasses.replace(SESSION[1], width=1280)],
        )
        (item,) = _items(ds, kind=SLEAP)
        assert item.n_sources == 1


class TestCameras:
    def test_two_cameras_still_collapse_onto_one_item(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = _dataset(
            tmp_path,
            [
                Clip(filename="cam0.mp4", camera="cam0", video_uuid="uid-a"),
                Clip(filename="cam1.mp4", camera="cam1", video_uuid="uid-b"),
            ],
        )
        items = _items(ds, kind=TREX)
        assert len(items) == 1
        assert items[0].n_sources == 1
        assert "shares one output directory" in capsys.readouterr().err


class TestTheItemInvariant:
    def test_facts_must_be_parallel_to_paths(self) -> None:
        """A short tuple would place later clips on the first clip's rate."""
        facts = store_facts(
            width=64,
            height=48,
            fps=30.0,
            frame_count=10,
            codec="h264",
            duration=1.0,
            video_uuid="",
            identity_scheme="",
        )
        with pytest.raises(ValueError, match="parallel"):
            _ = TrackerWorkItem(
                group="",
                sequence="s",
                key="s",
                video_paths=(Path("a.mp4"), Path("b.mp4")),
                fps=30.0,
                source_facts=(facts,),
            )

    def test_absent_facts_are_allowed(self) -> None:
        """Absent is not short: the subprocess trackers never open the file."""
        item = TrackerWorkItem(
            group="",
            sequence="s",
            key="s",
            video_paths=(Path("a.mp4"),),
            fps=30.0,
        )
        assert item.facts is None
        assert item.source_uid == ""

    def test_no_paths_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one video path"):
            _ = TrackerWorkItem(
                group="", sequence="s", key="s", video_paths=(), fps=30.0
            )


class TestExpandMediaScope:
    """A group or sequence scope becomes the entry list an op's params take.

    The expansion is what keeps ``--groups`` expressible after the params
    dropped the field: a group named with no sequence means every sequence in
    it, and only the media index knows which those are.
    """

    def _dataset(self, tmp_path: Path) -> Dataset:
        return _dataset(
            tmp_path,
            [
                Clip(filename="a1.mp4", group="A", sequence="one"),
                Clip(filename="a2.mp4", group="A", sequence="two"),
                Clip(filename="b1.mp4", group="B", sequence="one"),
            ],
        )

    def test_a_group_enumerates_its_sequences(self, tmp_path: Path) -> None:
        ds = self._dataset(tmp_path)
        assert ds.expand_media_scope(groups=["A"]) == [("A", "one"), ("A", "two")]

    def test_a_sequence_name_repeated_across_groups_yields_both(
        self, tmp_path: Path
    ) -> None:
        """What the cross product cannot express, and why entries are enumerated."""
        ds = self._dataset(tmp_path)
        assert ds.expand_media_scope(sequences=["one"]) == [("A", "one"), ("B", "one")]

    def test_the_three_selectors_intersect(self, tmp_path: Path) -> None:
        ds = self._dataset(tmp_path)
        expanded = ds.expand_media_scope(
            groups=["A"], entries=[("A", "one"), ("B", "one")]
        )
        assert expanded == [("A", "one")]

    def test_an_unnamed_scope_stays_unscoped(self, tmp_path: Path) -> None:
        """``None`` reaches the op as ``None``, which every op reads as all entries."""
        ds = self._dataset(tmp_path)
        assert ds.expand_media_scope() is None

    def test_an_entry_list_alone_needs_no_media_index(self, tmp_path: Path) -> None:
        """A dataset with no index still accepts an explicit scope."""
        manifest = new_dataset_manifest("no-index", base_dir=tmp_path)
        ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
        assert ds.expand_media_scope(entries=[("A", "one")]) == [("A", "one")]
