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
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner, Result

from mosaic.cli import app
from mosaic.cli.run import split_op_scope
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.scope import Scope
from mosaic.core.media.facts_columns import store_facts
from mosaic.tracking.common.scope import (
    JoinedSourceMismatchError,
    TrackerWorkItem,
    build_work_items,
)
from tests.helpers import MediaClip, write_media_index

TREX = "trex"
"""A tracker that declares ``joins_sources``."""

SLEAP = "sleap"
"""One that does not, and must keep behaving exactly as it did."""


def _dataset(tmp_path: Path, clips: list[MediaClip]) -> Dataset:
    """A dataset whose media index holds exactly *clips*."""
    manifest = new_dataset_manifest("scope", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media_index(ds, clips)
    return ds


def _cli(ds: Dataset, kind: str, params: dict[str, object]) -> Result:
    """``mosaic run --kind`` over *ds*, as a caller types it."""
    return CliRunner().invoke(
        app,
        [
            "run",
            "--manifest",
            str(ds.manifest_path),
            "--kind",
            kind,
            "--params",
            json.dumps(params),
        ],
    )


def _items(ds: Dataset, *, kind: str) -> list[TrackerWorkItem]:
    return build_work_items(ds, ds.resolve_media_scope(None), kind=kind)


# The measured shape of one real session, shortened.
SESSION = [
    MediaClip(filename="c0.mp4", video_order=0, video_uuid="uid-0", fps=30.0),
    MediaClip(filename="c1.mp4", video_order=1, video_uuid="uid-1", fps=29.95),
    MediaClip(filename="c2.mp4", video_order=2, video_uuid="uid-2", fps=31.0),
]


class TestOneClip:
    def test_the_ordinary_sequence_is_untouched(self, tmp_path: Path) -> None:
        ds = _dataset(tmp_path, [MediaClip(filename="v.mp4", video_uuid="uid-v")])
        (item,) = _items(ds, kind=TREX)
        assert item.n_sources == 1
        assert item.video_path.name == "v.mp4"
        assert item.fps == 30.0

    def test_its_reuse_key_is_still_the_video_uuid(self, tmp_path: Path) -> None:
        """The proof that nothing already on disk is invalidated."""
        ds = _dataset(tmp_path, [MediaClip(filename="v.mp4", video_uuid="uid-v")])
        (item,) = _items(ds, kind=TREX)
        assert item.source_uid == "uid-v"
        assert item.video_uid == "uid-v"

    def test_an_unmeasured_rate_still_falls_back(self, tmp_path: Path) -> None:
        """Only a *joined* entry refuses a missing rate; one clip defaults."""
        ds = _dataset(tmp_path, [MediaClip(filename="v.mp4", fps=0.0)])
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
                dataclasses.replace(SESSION[1], video_order=0),
                dataclasses.replace(SESSION[0], video_order=1),
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
                MediaClip(filename="cam0.mp4", camera="cam0", video_uuid="uid-a"),
                MediaClip(filename="cam1.mp4", camera="cam1", video_uuid="uid-b"),
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


class TestResolveScope:
    """A group or sequence scope becomes the entry list an op's params take.

    The enumeration is what keeps ``--groups`` expressible after the params
    dropped the field: a group named with no sequence means every sequence in
    it, and only the media index knows which those are.
    """

    def _dataset(self, tmp_path: Path) -> Dataset:
        """Two groups, a repeated sequence name, and one sequence in two clips.

        ``one`` is written under both groups, and split across two rows under
        ``A``, which is what an enumeration must keep apart and collapse.
        """
        return _dataset(
            tmp_path,
            [
                MediaClip(filename="b1.mp4", group="B", sequence="one"),
                MediaClip(filename="a2.mp4", group="A", sequence="two"),
                MediaClip(filename="a1.mp4", group="A", sequence="one", video_order=0),
                MediaClip(filename="a1b.mp4", group="A", sequence="one", video_order=1),
            ],
        )

    def test_a_group_enumerates_its_sequences(self, tmp_path: Path) -> None:
        """Three media rows under A, and the two entries they belong to.

        ``one`` is split across two clips and is named once. A pair repeated
        per media row is not a wider scope, and an op that acts per entry would
        act on that entry twice.
        """
        ds = self._dataset(tmp_path)
        resolved = ds.resolve_scope(Scope(groups=["A"]))
        assert resolved.entries == {("A", "one"), ("A", "two")}

    def test_a_sequence_name_repeated_across_groups_yields_both(
        self, tmp_path: Path
    ) -> None:
        """What the cross product cannot express, and why entries are enumerated."""
        ds = self._dataset(tmp_path)
        resolved = ds.resolve_scope(Scope(sequences=["one"]))
        assert resolved.entries == {("A", "one"), ("B", "one")}

    def test_entries_beside_groups_is_refused(self) -> None:
        """One way to name a set, not two combined."""
        with pytest.raises(ValidationError, match="cannot be combined"):
            _ = Scope(entries=[("A", "one"), ("B", "one")], groups=["A"])

    def test_an_unnamed_scope_stays_unscoped(self, tmp_path: Path) -> None:
        """An unset selector is what the commands turn into ``entries=None``."""
        ds = self._dataset(tmp_path)
        resolved = ds.resolve_scope()
        assert resolved.entries == set()
        assert resolved.selector.is_unset

    def test_an_entry_list_alone_needs_no_media_index(self, tmp_path: Path) -> None:
        """A dataset with no index still accepts an explicit scope."""
        manifest = new_dataset_manifest("no-index", base_dir=tmp_path)
        ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
        resolved = ds.resolve_scope(Scope(entries=[("A", "one")]))
        assert resolved.entries == {("A", "one")}

    def test_a_group_with_no_rows_is_an_empty_scope(self, tmp_path: Path) -> None:
        """Empty is not unscoped: naming a group that holds nothing runs nothing."""
        ds = self._dataset(tmp_path)
        resolved = ds.resolve_scope(Scope(groups=["absent"]))
        assert resolved.entries == set()
        assert not resolved.selector.is_unset

    def test_a_missing_index_is_reported_where_the_scope_is_read(
        self, tmp_path: Path
    ) -> None:
        """The documented raise, which the commands turn into a message."""
        manifest = new_dataset_manifest("no-index", base_dir=tmp_path)
        ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
        with pytest.raises(FileNotFoundError):
            _ = ds.resolve_scope(Scope(groups=["A"]))


class TestScopeInsideParams:
    """``mosaic run --kind`` takes its scope inside ``--params``.

    The command declares no scope flags and refuses ``--entries``. The scope
    keys are read out of the params object and resolved before the op's own
    model validates.
    """

    def _dataset(self, tmp_path: Path) -> Dataset:
        """Three entries, indexed in the reverse of their sorted order.

        Every assertion below reads an ordered list out of an op's params. The
        two orders disagree here, and an ordered assertion therefore names one
        of them.
        """
        return _dataset(
            tmp_path,
            [
                MediaClip(filename="b1.mp4", group="B", sequence="one"),
                MediaClip(filename="a2.mp4", group="A", sequence="two"),
                MediaClip(filename="a1.mp4", group="A", sequence="one"),
            ],
        )

    def test_params_naming_neither_key_are_untouched(self) -> None:
        """An op with no scope keeps params its model accepts."""
        params: dict[str, object] = {"data": "datasets/pose/data.yaml", "epochs": 3}
        settings, scope = split_op_scope(params)
        assert settings == params
        assert scope.is_unset

    def test_a_group_becomes_the_selector_an_op_covers(self, tmp_path: Path) -> None:
        """The scope leaves the params and arrives as what ``run_op`` takes."""
        ds = self._dataset(tmp_path)
        settings, scope = split_op_scope({"track_max_speed": 2, "groups": ["A"]})
        assert settings == {"track_max_speed": 2}
        assert scope == Scope(groups=["A"])
        assert sorted(ds.resolve_scope(scope).entries) == [
            ("A", "one"),
            ("A", "two"),
        ]

    def test_an_entry_list_alone_is_sorted_and_collapsed(self, tmp_path: Path) -> None:
        """The params key resolves the way ``--entries`` does, through one selector.

        Read off ``op_entries``, the value an op body receives. An entry named
        twice is one entry, and the list is ordered. A run covering one set of
        entries therefore records one list however a caller wrote them.
        """
        ds = self._dataset(tmp_path)
        _, scope = split_op_scope(
            {"entries": [["B", "one"], ["A", "one"], ["B", "one"]]}
        )
        assert ds.resolve_scope(scope).op_entries == [("A", "one"), ("B", "one")]

    def test_index_order_does_not_reach_the_entry_list(self, tmp_path: Path) -> None:
        """The entry list is sorted, whatever order the index rows are in.

        The resolved scope is a set, and ``op_entries`` sorts it. A re-index
        therefore moves nothing an op records.
        """
        ds = self._dataset(tmp_path)
        _, scope = split_op_scope({"groups": ["A", "B"]})
        assert ds.resolve_scope(scope).op_entries == [
            ("A", "one"),
            ("A", "two"),
            ("B", "one"),
        ]

    def test_a_camera_addressed_entry_list_is_refused(self) -> None:
        """An op's entry list is pairs, and a triple is refused by name.

        The refusal names the form to give instead. A resolved triple covers
        every camera of the entry under a selector that named one of them.
        """
        with pytest.raises(ValueError, match=r"Give \(group, sequence\) pairs"):
            _ = split_op_scope({"entries": [["A", "one", "cam0"]]})

    def test_an_entry_list_beside_a_group_is_refused(self) -> None:
        """The params keys name one selector, the same as the flags do."""
        with pytest.raises(ValidationError, match="cannot be combined"):
            _ = split_op_scope(
                {"groups": ["A"], "entries": [["A", "one"], ["B", "one"]]}
            )

    def test_a_scope_free_op_is_refused_at_the_command_line(
        self, tmp_path: Path
    ) -> None:
        """The refusal a caller meets, driven through the command it types.

        A scope named for an op that takes none used to be refused by the params
        model, which received an ``entries`` key it forbids. The scope no longer
        reaches the model, and this asserts that the command still declines
        rather than training over everything with the narrowing dropped.

        Driven end to end because a unit call to the checker proves only that
        the checker works. What has to hold is that the command reaches it.
        """
        ds = self._dataset(tmp_path)
        result = _cli(
            ds, "train-pose", {"data": "datasets/pose/data.yaml", "groups": ["A"]}
        )

        assert result.exit_code != 0
        assert "train-pose takes no entry scope" in result.output

    def test_a_scope_free_op_runs_when_no_scope_is_named(self, tmp_path: Path) -> None:
        """The other side of the refusal, which cannot then pass by refusing always.

        The op gets past the scope check and fails on its own missing tool
        environment, which is as far as this dataset can take it.
        """
        ds = self._dataset(tmp_path)
        result = _cli(ds, "train-pose", {"data": "datasets/pose/data.yaml"})

        assert "takes no entry scope" not in result.output

    def test_a_scope_free_op_keeps_the_selector_out_of_its_settings(self) -> None:
        """The selector is split off rather than dropped or left in the settings.

        A scope named for an op that takes none has to survive this far, because
        ``run_op`` is what refuses it and it can only refuse what it is handed.
        The end-to-end refusal is asserted above.
        """
        settings, scope = split_op_scope(
            {"data": "datasets/pose/data.yaml", "groups": ["A"]}
        )
        assert settings == {"data": "datasets/pose/data.yaml"}
        assert scope == Scope(groups=["A"])
