"""Declared scan sources, and what a scan is allowed to replace.

A scan used to rewrite the whole index from whatever it had just walked. That
made three things true that should not have been: scanning directory A and then
directory B kept only B; any scan destroyed rows pointing at files outside the
dataset, which is the mechanism that replaced an outside root; and one dataset
could not hold two source formats at once. These tests are the record that none
of them is true any more.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.manifest import (
    LabelsScanSource,
    MediaScanSource,
    RawScanSource,
    TracksScanSource,
)
from mosaic.core.pipeline.scan_claim import ScanClaim
from mosaic.core.pipeline.promotion import promote_correction
from mosaic.core.pipeline.tracks_raw_index import (
    RawEntryNamedTwice,
    read_tracks_raw_index,
)

from tests.helpers import tracktor_dataset

VideoWriter = Callable[..., None]


def make_dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="scans", base_dir=tmp_path / "ds")
    return Dataset(manifest_path=manifest).load()


def indexed_paths(dataset: Dataset) -> set[str]:
    return {
        Path(str(row["abs_path"])).name
        for row in dataset.read_media_index()
        if row.get("abs_path")
    }


class TestScanPreservesWhatItDoesNotClaim:
    def test_scanning_a_second_directory_keeps_the_first(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """The headline hole: a scan is a replace over its claim, not the file."""
        dataset = make_dataset(tmp_path)
        first, second = tmp_path / "a", tmp_path / "b"
        write_cfr_mp4(first / "one.mp4")
        write_cfr_mp4(second / "two.mp4")

        _ = dataset.index_media([first])
        _ = dataset.index_media([second])

        assert indexed_paths(dataset) == {"one.mp4", "two.mp4"}

    def test_a_file_deleted_from_a_scanned_directory_leaves(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """Replace-over-the-claim, the other direction. Absent must mean gone."""
        dataset = make_dataset(tmp_path)
        directory = tmp_path / "a"
        write_cfr_mp4(directory / "one.mp4")
        doomed = directory / "two.mp4"
        write_cfr_mp4(doomed)

        _ = dataset.index_media([directory])
        assert indexed_paths(dataset) == {"one.mp4", "two.mp4"}

        doomed.unlink()
        _ = dataset.index_media([directory])
        assert indexed_paths(dataset) == {"one.mp4"}

    def test_a_row_outside_every_claim_survives(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """The arrangement that replaced an outside root, protected.

        A dataset references a file living elsewhere by absolute ``abs_path``
        from an index that stays inside. A scan of an unrelated directory used
        to delete exactly those rows.
        """
        dataset = make_dataset(tmp_path)
        elsewhere = tmp_path / "nas"
        write_cfr_mp4(elsewhere / "far.mp4")
        _ = dataset.index_media([elsewhere])

        unrelated = tmp_path / "local"
        write_cfr_mp4(unrelated / "near.mp4")
        _ = dataset.index_media([unrelated])

        assert indexed_paths(dataset) == {"far.mp4", "near.mp4"}
        stored = {
            Path(str(row["abs_path"])).name: str(row["abs_path"])
            for row in dataset.read_media_index()
        }
        assert Path(stored["far.mp4"]).is_absolute()

    def test_prune_unsourced_opts_into_the_old_behaviour(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """Someone was using the whole-file rebuild to garbage-collect.

        It is no longer what a scan does by default, so it is available on
        request instead of by accident.
        """
        dataset = make_dataset(tmp_path)
        first, second = tmp_path / "a", tmp_path / "b"
        write_cfr_mp4(first / "one.mp4")
        write_cfr_mp4(second / "two.mp4")
        _ = dataset.index_media([first])

        _ = dataset.index_media([second], prune_unsourced=True)
        assert indexed_paths(dataset) == {"two.mp4"}


class TestASymlinkedSourceIsStillTheScansOwnRow:
    """A scan must own what it walked, however the row spells the path.

    A symlink inside a scanned directory splits the two paths a scan reasons
    about: the walk finds ``<dataset>/media_raw/<entry>/clip.mp4``, but the row
    records the symlink's *target*, because ``_relative_to_root`` resolves before
    testing containment and a target outside the dataset is stored absolute. The
    directory claim covers the first path and the row carries the second, so the
    scan used to preserve rows it had itself just written -- and append the same
    files again on the next pass, forever.

    The farm is a real arrangement, not a contrivance: it is how a fixed
    directory tree gets the ``<group>__<sequence>`` level that
    ``layout="per_sequence"`` reads identity from, without copying the videos.
    """

    @staticmethod
    def farm(dataset: Dataset, targets: list[Path], entry: str) -> Path:
        """Link *targets* into ``media_raw/<entry>/`` and declare it as a source."""
        directory = dataset.get_root("media_raw") / entry
        directory.mkdir(parents=True, exist_ok=True)
        for target in targets:
            (directory / target.name).symlink_to(target)
        dataset.add_scan_source(
            MediaScanSource(id="farm", path="media_raw", layout="per_sequence")
        )
        return directory

    def test_rescanning_a_symlink_farm_does_not_duplicate(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """Three scans, three files. The bug made it 3, then 6, then 9."""
        dataset = make_dataset(tmp_path)
        elsewhere = tmp_path / "nas"
        targets = [elsewhere / f"clip_{n}.mp4" for n in range(3)]
        for target in targets:
            write_cfr_mp4(target)
        _ = self.farm(dataset, targets, "cage__day1")

        counts = []
        for _ in range(3):
            _ = dataset.scan_media()
            counts.append(len(dataset.read_media_index()))

        assert counts == [3, 3, 3], f"the index grew across rescans: {counts}"

    def test_the_duplicates_would_have_reached_the_tracker(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """Why this is not a cosmetic index-tidiness bug.

        A tracker reads an entry's clips as one join, so a duplicated row is a
        clip joined twice. The session's frame count doubles and its footage
        repeats mid-timeline, and nothing raises.
        """
        from mosaic.tracking.common.scope import build_work_items

        dataset = make_dataset(tmp_path)
        elsewhere = tmp_path / "nas"
        targets = [elsewhere / f"clip_{n}.mp4" for n in range(3)]
        for target in targets:
            write_cfr_mp4(target)
        _ = self.farm(dataset, targets, "cage__day1")

        _ = dataset.scan_media()
        _ = dataset.scan_media()

        scope = dataset.resolve_media_scope(None)
        item = build_work_items(dataset, scope, kind="trex").items[0]
        assert item.n_sources == 3, "a clip would be converted more than once"
        assert len({path.name for path in item.video_paths}) == 3

    def test_a_symlinked_file_that_leaves_the_farm_still_leaves(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """Widening the claim must not cost the deletion half of the rule.

        The link is what the scan walks, so removing it is how a file leaves a
        farm -- and ``prune_unsourced`` is what collects the row, since the
        target is under no claim once nothing points at it.
        """
        dataset = make_dataset(tmp_path)
        elsewhere = tmp_path / "nas"
        targets = [elsewhere / f"clip_{n}.mp4" for n in range(3)]
        for target in targets:
            write_cfr_mp4(target)
        directory = self.farm(dataset, targets, "cage__day1")

        _ = dataset.scan_media()
        assert len(dataset.read_media_index()) == 3

        (directory / "clip_2.mp4").unlink()
        _ = dataset.scan_media(prune_unsourced=True)
        assert indexed_paths(dataset) == {"clip_0.mp4", "clip_1.mp4"}

    def test_an_unwalked_external_row_still_survives_a_farm_scan(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """The preserve rule the widening must not eat.

        The widened claim covers what this scan *found*. A row no scan walked --
        an assignment, or a reference to another dataset's video -- appears in no
        walk and must still come through untouched.
        """
        dataset = make_dataset(tmp_path)
        far = tmp_path / "other-dataset"
        write_cfr_mp4(far / "borrowed.mp4")
        _ = dataset.index_media([far])

        elsewhere = tmp_path / "nas"
        targets = [elsewhere / "clip_0.mp4"]
        write_cfr_mp4(targets[0])
        _ = self.farm(dataset, targets, "cage__day1")

        _ = dataset.scan_media()
        _ = dataset.scan_media()

        assert indexed_paths(dataset) == {"borrowed.mp4", "clip_0.mp4"}

    def test_a_symlink_loop_does_not_abort_the_scan(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """A circular link in a farm is skipped, and the clip beside it survives.

        This passes because the *walk* rejects a loop before it can become a
        row -- ``is_file()`` is false for one -- so it never reaches the claim
        builder. Worth pinning anyway, and worth being clear about: it is the
        walk that makes this safe, not the claim builder's guard, and the
        sibling test below is what actually exercises that guard.
        """
        dataset = make_dataset(tmp_path)
        elsewhere = tmp_path / "nas"
        good = elsewhere / "clip_0.mp4"
        write_cfr_mp4(good)
        directory = self.farm(dataset, [good], "cage__day1")
        (directory / "loop_a.mp4").symlink_to(directory / "loop_b.mp4")
        (directory / "loop_b.mp4").symlink_to(directory / "loop_a.mp4")

        _ = dataset.scan_media()
        _ = dataset.scan_media()

        assert indexed_paths(dataset) == {"clip_0.mp4"}

    def test_an_unresolvable_stored_path_is_skipped_not_raised(
        self, tmp_path: Path
    ) -> None:
        """The claim builder's guard, exercised directly.

        A row whose path cannot be resolved is claimed by nobody -- the same
        conservative answer ``_row_claimed`` gives an empty cell -- rather than
        an exception escaping into the caller's scan.

        An embedded NUL is the trigger: ``Path.resolve`` rejects it with
        ``ValueError``. A symlink loop resolves to itself, so it is no trigger.
        """
        dataset = make_dataset(tmp_path)

        claim = dataset.walked_claim(
            [
                {"abs_path": str(tmp_path / "nul\x00name.mp4")},
                {"abs_path": str(tmp_path / "plain.mp4")},
                {"abs_path": ""},
            ]
        )

        assert claim.claims((tmp_path / "plain.mp4").resolve())
        assert len(claim.files) == 1, "only the resolvable row is claimed"

    def test_a_symlinked_tracks_source_does_not_duplicate_either(
        self, tmp_path: Path
    ) -> None:
        """The raw scans share the write path, so they shared the bug."""
        dataset = make_dataset(tmp_path)
        elsewhere = tmp_path / "nas"
        elsewhere.mkdir(parents=True, exist_ok=True)
        target = elsewhere / "session.npy"
        target.write_bytes(b"\x00")

        farm = dataset.get_root("tracks_raw") / "linked"
        farm.mkdir(parents=True, exist_ok=True)
        (farm / target.name).symlink_to(target)
        dataset.add_scan_source(
            TracksScanSource(id="farm", path="tracks_raw/linked", patterns=("*.npy",))
        )

        counts = []
        for _ in range(3):
            _ = dataset.scan_tracks()
            counts.append(len(dataset.read_tracks_raw_index()))

        assert counts == [1, 1, 1], f"the tracks index grew across rescans: {counts}"


class TestFileModeSources:
    def test_a_file_source_indexes_only_what_it_lists(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """No glob expresses "these twelve of two hundred"."""
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        for name in ("pick_a.mp4", "pick_b.mp4", "leave_me.mp4"):
            write_cfr_mp4(folder / name)

        dataset.add_scan_source(
            MediaScanSource(
                id="import-1", path=str(folder), files=("pick_a.mp4", "pick_b.mp4")
            )
        )
        _ = dataset.scan_media()

        assert indexed_paths(dataset) == {"pick_a.mp4", "pick_b.mp4"}

    def test_two_import_batches_in_one_folder_do_not_evict_each_other(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """The case selective import actually produces.

        Both sources sit in the same directory. If the claim were the directory
        rather than the listed files, scanning either would delete the other's
        rows.
        """
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            write_cfr_mp4(folder / name)

        dataset.add_scan_source(
            MediaScanSource(id="batch-1", path=str(folder), files=("a.mp4",))
        )
        dataset.add_scan_source(
            MediaScanSource(id="batch-2", path=str(folder), files=("b.mp4",))
        )
        _ = dataset.scan_media()
        assert indexed_paths(dataset) == {"a.mp4", "b.mp4"}

        _ = dataset.scan_media(only=["batch-1"])
        assert indexed_paths(dataset) == {"a.mp4", "b.mp4"}

    def test_add_source_files_extends_an_import(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        for name in ("a.mp4", "b.mp4"):
            write_cfr_mp4(folder / name)
        dataset.add_scan_source(
            MediaScanSource(id="import-1", path=str(folder), files=("a.mp4",))
        )
        _ = dataset.scan_media()
        assert indexed_paths(dataset) == {"a.mp4"}

        assert dataset.add_source_files("media", "import-1", ["b.mp4"]) == 1
        _ = dataset.scan_media()
        assert indexed_paths(dataset) == {"a.mp4", "b.mp4"}

    def test_remove_source_files_drops_the_row_on_the_next_scan(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        for name in ("a.mp4", "b.mp4"):
            write_cfr_mp4(folder / name)
        dataset.add_scan_source(
            MediaScanSource(id="import-1", path=str(folder), files=("a.mp4", "b.mp4"))
        )
        _ = dataset.scan_media()

        assert dataset.remove_source_files("media", "import-1", ["b.mp4"]) == 1
        _ = dataset.scan_media()
        assert indexed_paths(dataset) == {"a.mp4"}

    def test_a_listed_file_that_vanished_is_reported_and_skipped(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        write_cfr_mp4: VideoWriter,
    ) -> None:
        """A share being unmounted is not a decision to un-import.

        The row leaves because the file did; the declaration stays.
        """
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        write_cfr_mp4(folder / "here.mp4")
        dataset.add_scan_source(
            MediaScanSource(id="i", path=str(folder), files=("here.mp4", "gone.mp4"))
        )
        _ = dataset.scan_media()

        captured = capsys.readouterr()
        assert "gone.mp4" in captured.err
        assert indexed_paths(dataset) == {"here.mp4"}
        assert dataset.scan_sources("media")[0].files == ("here.mp4", "gone.mp4")

    def test_extending_a_directory_source_is_refused(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(MediaScanSource(id="walked", path=str(tmp_path / "n")))
        with pytest.raises(ValueError, match="directory source"):
            _ = dataset.add_source_files("media", "walked", ["a.mp4"])


class TestDeclaration:
    def test_a_source_may_point_outside_the_dataset(self, tmp_path: Path) -> None:
        """The asymmetry with roots, at the dataset boundary.

        ``set_root`` refuses an outside path because that root carries its own
        index. A source is the mechanism for storage elsewhere, so it must not.
        """
        dataset = make_dataset(tmp_path)
        outside = tmp_path / "somewhere-else"
        dataset.add_scan_source(MediaScanSource(id="nas", path=str(outside)))
        assert dataset.scan_sources("media")[0].path == str(outside)

    def test_declaring_a_source_does_not_create_its_directory(
        self, tmp_path: Path
    ) -> None:
        """A load creates roots. It must never create a source."""
        dataset = make_dataset(tmp_path)
        outside = tmp_path / "unmounted"
        dataset.add_scan_source(MediaScanSource(id="nas", path=str(outside)))
        assert not outside.exists()

        reloaded = Dataset(manifest_path=dataset.manifest_path).load(ensure_roots=True)
        assert not outside.exists()
        assert reloaded.scan_sources("media")[0].id == "nas"

    def test_a_declaration_survives_a_reload(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(
            MediaScanSource(
                id="nas", path="/mnt/nas", extensions=(".mp4",), layout="per_sequence"
            )
        )
        reloaded = Dataset(manifest_path=dataset.manifest_path).load()
        source = reloaded.scan_sources("media")[0]
        assert (source.path, source.layout, source.extensions) == (
            "/mnt/nas",
            "per_sequence",
            (".mp4",),
        )
        assert source.added_at

    def test_a_file_source_survives_declaring_a_second_source(
        self, tmp_path: Path
    ) -> None:
        """Adding a source must not re-validate the existing ones from a dump.

        ``model_fields_set`` is what tells a file source apart from one that
        asked for a walk, and a dump-and-reparse names every field including the
        defaults. That made every already-declared file source look as though it
        had declared ``extensions`` too, so the *second* ``add`` failed by
        complaining about the first.
        """
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(
            MediaScanSource(id="import", path="/nas/pilot", files=("a.mp4",))
        )
        dataset.add_scan_source(MediaScanSource(id="clips", path="/nas/clips"))
        dataset.add_scan_source(
            TracksScanSource(id="trex", path="/nas/trex", patterns=("*.npz",))
        )
        assert {s.id for s in dataset.scan_sources("media")} == {"import", "clips"}
        assert dataset.scan_sources("media")[0].mode == "files"

    def test_a_duplicate_id_is_refused(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(MediaScanSource(id="nas", path="/a"))
        with pytest.raises(ValueError, match="already declared"):
            dataset.add_scan_source(MediaScanSource(id="nas", path="/b"))

    def test_a_nested_directory_source_is_refused_naming_both(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(MediaScanSource(id="outer", path="/nas"))
        with pytest.raises(ValueError, match="outer.*inner|inner.*outer"):
            dataset.add_scan_source(MediaScanSource(id="inner", path="/nas/clips"))

    def test_removing_a_source_keeps_its_rows(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        """Undeclaring is a statement about future scans, not a delete."""
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        write_cfr_mp4(folder / "one.mp4")
        dataset.add_scan_source(MediaScanSource(id="nas", path=str(folder)))
        _ = dataset.scan_media()

        orphaned = dataset.remove_scan_source("media", "nas")
        assert orphaned == 1
        assert indexed_paths(dataset) == {"one.mp4"}
        assert dataset.scan_sources("media") == ()

    def test_removing_an_unknown_source_lists_the_real_ones(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(MediaScanSource(id="nas", path="/a"))
        with pytest.raises(KeyError, match="nas"):
            _ = dataset.remove_scan_source("media", "typo")

    def test_scanning_with_nothing_declared_says_so(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        with pytest.raises(ValueError, match="no media scan sources are declared"):
            _ = dataset.scan_media()

    def test_only_restricts_the_pass_without_changing_the_declaration(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        dataset = make_dataset(tmp_path)
        first, second = tmp_path / "a", tmp_path / "b"
        write_cfr_mp4(first / "one.mp4")
        write_cfr_mp4(second / "two.mp4")
        dataset.add_scan_source(MediaScanSource(id="first", path=str(first)))
        dataset.add_scan_source(MediaScanSource(id="second", path=str(second)))

        _ = dataset.scan_media(only=["first"])
        assert indexed_paths(dataset) == {"one.mp4"}

        _ = dataset.scan_media(only=["second"])
        assert indexed_paths(dataset) == {"one.mp4", "two.mp4"}
        assert {s.id for s in dataset.scan_sources("media")} == {"first", "second"}

    def test_a_relative_source_path_resolves_against_the_dataset(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        source = MediaScanSource(id="inside", path="media_raw")
        dataset.add_scan_source(source)
        assert dataset.resolve_source_path(source) == dataset.base_dir / "media_raw"


class TestScanClaimUnit:
    def test_a_directory_claim_covers_its_subtree(self) -> None:
        claim = ScanClaim.over_directories([Path("/nas")])
        assert claim.claims(Path("/nas/a/b.mp4"))
        assert not claim.claims(Path("/other/b.mp4"))

    def test_a_file_claim_covers_only_what_it_lists(self) -> None:
        claim = ScanClaim.over_files([Path("/nas/a.mp4")])
        assert claim.claims(Path("/nas/a.mp4"))
        assert not claim.claims(Path("/nas/b.mp4"))

    def test_claims_union(self) -> None:
        claim = ScanClaim.over_files([Path("/x/a.mp4")]) | ScanClaim.over_directories(
            [Path("/y")]
        )
        assert claim.claims(Path("/x/a.mp4"))
        assert claim.claims(Path("/y/deep/b.mp4"))
        assert not claim.claims(Path("/x/b.mp4"))

    def test_an_empty_claim_covers_nothing(self) -> None:
        assert not ScanClaim().claims(Path("/anything"))
        assert not ScanClaim()


class TestTracksSources:
    def test_two_source_formats_coexist_in_one_index(self, tmp_path: Path) -> None:
        """Before, the second scan's write replaced the first's rows entirely."""
        dataset = make_dataset(tmp_path)
        trex_dir, calms_dir = tmp_path / "trex", tmp_path / "calms"
        trex_dir.mkdir()
        calms_dir.mkdir()
        _ = (trex_dir / "a.npz").write_bytes(b"\x00")
        _ = (calms_dir / "b.npy").write_bytes(b"\x00")

        dataset.add_scan_source(
            TracksScanSource(
                id="trex",
                path=str(trex_dir),
                patterns=("*.npz",),
                src_format="trex_npz",
            )
        )
        dataset.add_scan_source(
            TracksScanSource(
                id="calms",
                path=str(calms_dir),
                patterns=("*.npy",),
                src_format="calms21_npy",
            )
        )
        _ = dataset.scan_tracks()

        rows = dataset.read_tracks_raw_index()
        by_name = {Path(str(r["abs_path"])).name: str(r["src_format"]) for r in rows}
        assert by_name == {"a.npz": "trex_npz", "b.npy": "calms21_npy"}


def touch_files(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"\x00")


def raw_names(dataset: Dataset) -> set[str]:
    return {
        Path(str(row["abs_path"])).name
        for row in dataset.read_tracks_raw_index()
        if row.get("abs_path")
    }


def declare_tracks(dataset: Dataset, *sources: TracksScanSource) -> None:
    """Declare *sources* as the dataset's tracks sources, past the overlap check."""
    dataset.manifest.sources = dataset.manifest.sources.with_kind("tracks", sources)


def count_walks(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record the base of every walk a claim makes."""
    from mosaic.core import dataset as dataset_module

    walked: list[Path] = []
    real = dataset_module.walked_files

    def counting(
        base: Path,
        patterns: Sequence[str],
        *,
        recursive: bool,
        exclude_patterns: Sequence[str] = (),
    ) -> frozenset[Path]:
        walked.append(base)
        return real(
            base, patterns, recursive=recursive, exclude_patterns=exclude_patterns
        )

    monkeypatch.setattr(dataset_module, "walked_files", counting)
    return walked


class TestASourceClaimFollowsItsFilter:
    """A tracks or labels directory source claims the files its walk would take.

    Two sources may share a folder with disjoint patterns, so a claim over the
    whole subtree would replace or drop the other source's rows.
    """

    @pytest.mark.parametrize(
        "source_type", [TracksScanSource, LabelsScanSource], ids=["tracks", "labels"]
    )
    def test_a_directory_source_claims_what_its_patterns_accept(
        self, tmp_path: Path, source_type: type[RawScanSource]
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.csv", base / "deep" / "b.csv", base / "c.npz")
        source = source_type(
            id="s", path=str(base), patterns=("*.csv",), src_format="any"
        )

        claim = dataset.source_claim(source)

        assert claim.claims((base / "a.csv").resolve())
        assert claim.claims((base / "deep" / "b.csv").resolve())
        assert claim.claims((base / "gone.csv").resolve())
        assert not claim.claims((base / "c.npz").resolve())
        assert not claim.claims((base / "._a.csv").resolve())

    def test_exclude_patterns_and_recursion_narrow_the_claim(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.csv", base / "skip.csv", base / "deep" / "b.csv")
        source = TracksScanSource(
            id="s",
            path=str(base),
            patterns=("*.csv",),
            recursive=False,
            exclude_patterns=("skip*",),
        )

        claim = dataset.source_claim(source)

        assert claim.claims((base / "a.csv").resolve())
        assert not claim.claims((base / "skip.csv").resolve())
        assert not claim.claims((base / "deep" / "b.csv").resolve())

    @pytest.mark.parametrize("which", ["media", "series"])
    def test_a_media_or_series_source_keeps_the_whole_subtree(
        self, tmp_path: Path, which: str
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        base.mkdir()
        source = (
            MediaScanSource(id="s", path=str(base))
            if which == "media"
            else LabelsScanSource(id="s", path=str(base), series="keypoints")
        )

        claim = dataset.source_claim(source)

        assert claim.claims((base / "anything.txt").resolve())

    def test_a_row_stored_relative_is_judged_by_the_filter(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        inbox = dataset.base_dir / "inbox"
        touch_files(inbox / "a.csv", inbox / "b.npz")
        _ = dataset.index_tracks_raw([inbox], patterns=["*.csv", "*.npz"])
        dataset.add_scan_source(
            TracksScanSource(id="s", path="inbox", patterns=("*.csv",))
        )
        stored = {str(row["abs_path"]) for row in dataset.read_tracks_raw_index()}
        assert stored == {"inbox/a.csv", "inbox/b.npz"}

        assert dataset.remove_scan_source("tracks", "s", save=False) == 1

    @pytest.mark.parametrize(
        "narrowed",
        [
            {"patterns": ("*.npz",)},
            {"patterns": ("*.npy", "*.npz"), "exclude_patterns": ("a*",)},
        ],
        ids=["patterns", "exclude"],
    )
    def test_a_rescan_drops_a_row_its_narrowed_filter_rejects(
        self, tmp_path: Path, narrowed: dict[str, tuple[str, ...]]
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy", base / "b.npz")
        source = TracksScanSource(id="s", path=str(base), patterns=("*.npy", "*.npz"))
        dataset.add_scan_source(source)
        _ = dataset.scan_tracks()
        assert raw_names(dataset) == {"a.npy", "b.npz"}
        declare_tracks(dataset, source.model_copy(update=narrowed))

        _ = dataset.scan_tracks()

        assert raw_names(dataset) == {"b.npz"}

    def test_sources_over_one_folder_keep_each_others_rows(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy", base / "deep" / "b.npz", base / "other.txt")
        declare_tracks(
            dataset,
            TracksScanSource(id="npy", path=str(base), patterns=("*.npy",)),
            TracksScanSource(id="npz", path=str(base), patterns=("*.npz",)),
        )

        for only in ([], ["npy"], ["npz"], [], ["npy"]):
            _ = dataset.scan_tracks(only=only)

            assert raw_names(dataset) == {"a.npy", "b.npz"}

    def test_a_deleted_file_walks_no_unrelated_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = make_dataset(tmp_path)
        touch_files(tmp_path / "a" / "x.npy", tmp_path / "a" / "y.npy")
        touch_files(tmp_path / "b" / "z.npy")
        for name in ("a", "b"):
            dataset.add_scan_source(
                TracksScanSource(
                    id=name, path=str(tmp_path / name), patterns=("*.npy",)
                )
            )
        _ = dataset.scan_tracks()
        (tmp_path / "a" / "x.npy").unlink()
        walked = count_walks(monkeypatch)

        _ = dataset.scan_tracks(only=["a"])

        assert raw_names(dataset) == {"y.npy", "z.npy"}
        assert walked == []

    def test_a_scan_with_no_sibling_walks_no_other_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy", base / "old.npz")
        source = TracksScanSource(id="s", path=str(base), patterns=("*.npy", "*.npz"))
        dataset.add_scan_source(source)
        _ = dataset.scan_tracks()
        declare_tracks(dataset, source.model_copy(update={"patterns": ("*.npy",)}))
        walked = count_walks(monkeypatch)

        _ = dataset.scan_tracks()

        assert raw_names(dataset) == {"a.npy"}
        assert walked == []

    def test_walked_claim_is_the_resolved_stored_paths(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        target = tmp_path / "data" / "a.csv"
        touch_files(target)

        claim = dataset.walked_claim([{"abs_path": str(target)}, {"abs_path": ""}])

        assert claim == ScanClaim.over_files([target.resolve()])


class TestASymlinkTheWalkFindsIsTheSourcesRow:
    """The filter judges a path by its own name and place, which a symlink hides.

    A file symlink whose target has another name, and a file reached through a
    symlinked folder, are walked under the source and stored by their target.
    The claim covers each target the walk reaches, so removing the source counts
    and drops their rows.
    """

    @pytest.mark.parametrize("outside", [True, False], ids=["outside", "inside"])
    def test_a_file_symlink_to_another_name(
        self, tmp_path: Path, outside: bool
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        target = (tmp_path / "elsewhere" if outside else base / "store") / "t.dat"
        touch_files(base / "a.npy", target)
        (base / "link.npy").symlink_to(target)
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(base), patterns=("*.npy",))
        )
        _ = dataset.scan_tracks()
        assert raw_names(dataset) == {"a.npy", "t.dat"}
        source = dataset.scan_sources("tracks")[0]

        assert dataset.remove_scan_source("tracks", "s", save=False) == 2
        assert dataset.drop_claimed_rows("tracks", dataset.source_claim(source)) == 2
        assert raw_names(dataset) == set()

    def test_a_file_through_a_symlinked_folder_outside_the_base(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        outside = tmp_path / "elsewhere"
        touch_files(base / "own" / "a.npy", outside / "x.npy")
        (base / "linked").symlink_to(outside, target_is_directory=True)
        dataset.add_scan_source(
            TracksScanSource(
                id="s", path=str(base), patterns=("*/*.npy",), recursive=False
            )
        )
        _ = dataset.scan_tracks()
        assert raw_names(dataset) == {"a.npy", "x.npy"}
        source = dataset.scan_sources("tracks")[0]

        assert dataset.remove_scan_source("tracks", "s", save=False) == 2
        assert dataset.drop_claimed_rows("tracks", dataset.source_claim(source)) == 2
        assert raw_names(dataset) == set()

    def test_a_rescan_keeps_a_symlinked_row_once(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        target = tmp_path / "elsewhere" / "t.dat"
        touch_files(target)
        base.mkdir()
        (base / "link.npy").symlink_to(target)
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(base), patterns=("*.npy",))
        )

        for _ in range(2):
            _ = dataset.scan_tracks()

        assert len(dataset.read_tracks_raw_index()) == 1


class TestARemovalSparesItsSiblings:
    """Removing a source drops only what no other source of its kind claims."""

    @staticmethod
    def two_sources_sharing_a_target(tmp_path: Path) -> Dataset:
        dataset = make_dataset(tmp_path)
        target = tmp_path / "elsewhere" / "t.dat"
        touch_files(tmp_path / "a" / "x.npy", tmp_path / "b" / "y.npy", target)
        for name in ("a", "b"):
            (tmp_path / name / "link.npy").symlink_to(target)
            dataset.add_scan_source(
                TracksScanSource(
                    id=name, path=str(tmp_path / name), patterns=("*.npy",)
                )
            )
        _ = dataset.scan_tracks()
        assert raw_names(dataset) == {"x.npy", "y.npy", "t.dat"}
        return dataset

    def test_a_shared_target_keeps_its_row(self, tmp_path: Path) -> None:
        dataset = self.two_sources_sharing_a_target(tmp_path)
        removal = dataset.source_removal("tracks", "a")

        assert dataset.remove_scan_source("tracks", "a", removal=removal) == 1
        assert (
            dataset.drop_claimed_rows(
                "tracks", removal.claim, series=removal.series, spared=removal.spared
            )
            == 1
        )
        assert raw_names(dataset) == {"y.npy", "t.dat"}

    def test_the_count_alone_spares_the_shared_target(self, tmp_path: Path) -> None:
        dataset = self.two_sources_sharing_a_target(tmp_path)

        assert dataset.undeclare_scan_source("tracks", "a") == 1

    def test_a_removal_built_before_a_sibling_was_declared_is_built_again(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        target = tmp_path / "elsewhere" / "t.dat"
        touch_files(tmp_path / "a" / "x.npy", tmp_path / "b" / "y.npy", target)
        for name in ("a", "b"):
            (tmp_path / name / "link.npy").symlink_to(target)
        dataset.add_scan_source(
            TracksScanSource(id="a", path=str(tmp_path / "a"), patterns=("*.npy",))
        )
        _ = dataset.scan_tracks()
        old = dataset.source_removal("tracks", "a")
        dataset.add_scan_source(
            TracksScanSource(id="b", path=str(tmp_path / "b"), patterns=("*.npy",))
        )

        assert dataset.remove_scan_source("tracks", "a", save=False, removal=old) == 1


RawKind = Literal["tracks", "labels"]


def raw_source(
    kind: RawKind,
    source_id: str,
    path: Path,
    patterns: tuple[str, ...],
    src_format: str | None = None,
) -> TracksScanSource | LabelsScanSource:
    """A tracks or labels directory source, with its kind's default format."""
    if kind == "tracks":
        return TracksScanSource(
            id=source_id,
            path=str(path),
            patterns=patterns,
            src_format=src_format or "calms21_npy",
        )
    return LabelsScanSource(
        id=source_id,
        path=str(path),
        patterns=patterns,
        src_format=src_format or "boris_aggregated_csv",
    )


def raw_rows(dataset: Dataset, kind: RawKind) -> list[dict[str, object]]:
    index = dataset.get_root(f"{kind}_raw") / "index.csv"
    return [dict(row) for row in read_tracks_raw_index(index)]


def raw_index_bytes(dataset: Dataset, kind: RawKind) -> bytes:
    return (dataset.get_root(f"{kind}_raw") / "index.csv").read_bytes()


def scan_kind(dataset: Dataset, kind: RawKind, only: Sequence[str] = ()) -> object:
    if kind == "tracks":
        return dataset.scan_tracks(only=only)
    return dataset.scan_labels(only=only)


@pytest.mark.parametrize("kind", ["tracks", "labels"])
class TestSourcesOverOneFolder:
    """Two tracks or labels sources over one folder are refused only for shared files.

    A file both sources' walks take would be indexed by two recipes, so it is
    refused at declaration and again at a scan, before any index write. Sources
    that take disjoint files share the folder.
    """

    def test_disjoint_patterns_declare_and_each_scans_its_own(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy", base / "deep" / "b.npz")
        npy, npz = (
            ("calms21_npy", "trex_npz")
            if kind == "tracks"
            else ("calms21_npy", "boris_aggregated_csv")
        )
        dataset.add_scan_source(raw_source(kind, "npy", base, ("*.npy",), npy))
        dataset.add_scan_source(raw_source(kind, "npz", base, ("*.npz",), npz))

        _ = scan_kind(dataset, kind, ["npz"])
        assert {
            (Path(str(row["abs_path"])).name, row["src_format"])
            for row in raw_rows(dataset, kind)
        } == {("b.npz", npz)}

        _ = scan_kind(dataset, kind)
        _ = scan_kind(dataset, kind, ["npy"])

        assert {
            (Path(str(row["abs_path"])).name, row["src_format"])
            for row in raw_rows(dataset, kind)
        } == {("a.npy", npy), ("b.npz", npz)}

    def test_overlapping_patterns_are_refused_naming_the_shared_files(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        from mosaic.core.dataset import SourcesShareFiles

        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        shared = [base / f"x{n}.npy" for n in range(7)]
        touch_files(base / "a.npy", *shared)
        dataset.add_scan_source(raw_source(kind, "all", base, ("*.npy",)))

        with pytest.raises(SourcesShareFiles) as refused:
            dataset.add_scan_source(raw_source(kind, "xs", base, ("x*.npy",)))

        message = str(refused.value)
        assert isinstance(refused.value, ValueError)
        assert "'all'" in message and "'xs'" in message
        assert "7 files" in message
        named = [path.name for path in shared if path.name in message]
        assert named == [f"x{n}.npy" for n in range(5)]
        assert "a.npy" not in message
        assert [s.id for s in dataset.scan_sources(kind)] == ["all"]

    def test_a_nested_path_follows_the_same_rule(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        from mosaic.core.dataset import SourcesShareFiles

        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy", base / "day2" / "b.npz", base / "day2" / "c.npy")
        dataset.add_scan_source(raw_source(kind, "outer", base, ("*.npy",)))

        dataset.add_scan_source(raw_source(kind, "inner", base / "day2", ("*.npz",)))
        with pytest.raises(SourcesShareFiles, match="c.npy"):
            dataset.add_scan_source(
                raw_source(kind, "clash", base / "day2", ("*.npy",))
            )

    def test_a_path_that_does_not_exist_keeps_the_structural_refusal(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        from mosaic.core.dataset import SourcesShareFiles

        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        dataset.add_scan_source(raw_source(kind, "npy", base, ("*.npy",)))

        with pytest.raises(ValueError, match="'npy'.*'npz'") as refused:
            dataset.add_scan_source(raw_source(kind, "npz", base / "day2", ("*.npz",)))

        assert not isinstance(refused.value, SourcesShareFiles)

    def test_an_unmounted_pair_does_not_refuse_another_source(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        import shutil

        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy")
        for name in ("npy", "npz"):
            dataset.add_scan_source(raw_source(kind, name, base, (f"*.{name}",)))
        shutil.rmtree(base)

        dataset.add_scan_source(
            raw_source(kind, "other", tmp_path / "other", ("*.npy",))
        )

        assert [s.id for s in dataset.scan_sources(kind)] == ["npy", "npz", "other"]

    def test_a_symlink_both_walks_reach_is_a_shared_file(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        from mosaic.core.dataset import SourcesShareFiles

        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "t.dat")
        (base / "link.npy").symlink_to(base / "t.dat")
        dataset.add_scan_source(raw_source(kind, "dat", base, ("*.dat",)))

        with pytest.raises(SourcesShareFiles, match="t.dat"):
            dataset.add_scan_source(raw_source(kind, "npy", base, ("*.npy",)))

    def test_a_rescan_of_one_source_keeps_the_others_rows(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy", base / "b.npz")
        for name in ("npy", "npz"):
            dataset.add_scan_source(raw_source(kind, name, base, (f"*.{name}",)))
        _ = scan_kind(dataset, kind)
        (base / "c.npy").write_bytes(b"\x00")

        _ = scan_kind(dataset, kind, ["npy"])

        assert {Path(str(row["abs_path"])).name for row in raw_rows(dataset, kind)} == {
            "a.npy",
            "b.npz",
            "c.npy",
        }

    @pytest.mark.parametrize("only", [[], ["all"], ["xs"]], ids=["full", "A", "B"])
    def test_a_later_file_both_take_is_refused_at_scan_before_any_write(
        self, tmp_path: Path, kind: RawKind, only: list[str]
    ) -> None:
        from mosaic.core.dataset import SourcesShareFiles

        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy")
        dataset.add_scan_source(raw_source(kind, "all", base, ("*.npy",)))
        dataset.add_scan_source(raw_source(kind, "xs", base, ("x*.npy",)))
        _ = scan_kind(dataset, kind)
        before = raw_index_bytes(dataset, kind)
        touch_files(base / "b.npy", base / "x1.npy")

        with pytest.raises(SourcesShareFiles, match="'all'.*'xs'.*x1.npy"):
            _ = scan_kind(dataset, kind, only)

        assert raw_index_bytes(dataset, kind) == before


class TestASequenceNamedFromThePath:
    """A source that declares ``sequence_path_pattern`` names sequences by folder."""

    def test_a_declared_source_names_sequences_from_several_folders(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "raw"
        touch_files(
            base / "1" / "batchA" / "day1" / "t1" / "trajectories.csv",
            base / "1" / "batchB" / "day1" / "t1" / "trajectories.csv",
            base / "2" / "batchA" / "day2" / "t7" / "trajectories.csv",
        )
        dataset.add_scan_source(
            TracksScanSource(
                id="hinz",
                path=str(base),
                patterns=("*.csv",),
                src_format="calms21_npy",
                sequence_path_pattern=r"^[^/]+/([^/]+)/(day[^/]+)/([^/]+)/",
                group_path_pattern=r"^([^/]+)/",
            )
        )

        _ = dataset.scan_tracks()

        named = sorted(
            (str(row["group"]), str(row["sequence"]))
            for row in dataset.read_tracks_raw_index()
        )
        assert named == [
            ("1", "batchA__day1__t1"),
            ("1", "batchB__day1__t1"),
            ("2", "batchA__day2__t7"),
        ]

    def test_a_declared_labels_source_names_sequences_from_the_path(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "labels"
        touch_files(base / "a" / "x" / "ann.csv", base / "b" / "y" / "ann.csv")
        dataset.add_scan_source(
            LabelsScanSource(
                id="ann",
                path=str(base),
                patterns=("*.csv",),
                sequence_path_pattern=r"^([^/]+)/([^/]+)/",
            )
        )

        _ = dataset.scan_labels()

        assert sorted(str(row["sequence"]) for row in raw_rows(dataset, "labels")) == [
            "a__x",
            "b__y",
        ]

    def test_a_declared_source_over_a_merging_format_is_refused_naming_it(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "raw"
        touch_files(base / "g" / "hex_7_fish0.npz")
        dataset.add_scan_source(
            TracksScanSource(
                id="trex-by-path",
                path=str(base),
                patterns=("*.npz",),
                src_format="trex_npz",
                sequence_path_pattern=r"^([^/]+)/",
            )
        )

        with pytest.raises(ValueError, match="'trex-by-path'") as caught:
            _ = dataset.scan_tracks()

        assert "trex_npz" in str(caught.value)
        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()


class TestTheStructuralRefusalStays:
    """Media sources and label series over one folder are refused whatever they hold."""

    def test_two_media_sources_over_one_existing_folder(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.mp4", base / "b.avi")
        dataset.add_scan_source(
            MediaScanSource(id="mp4", path=str(base), extensions=(".mp4",))
        )

        with pytest.raises(ValueError, match="'mp4'.*'avi'"):
            dataset.add_scan_source(
                MediaScanSource(id="avi", path=str(base), extensions=(".avi",))
            )

    def test_a_series_source_beside_a_labels_source(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.csv")
        dataset.add_scan_source(
            LabelsScanSource(id="files", path=str(base), patterns=("*.npy",))
        )

        with pytest.raises(ValueError, match="'files'.*'series'"):
            dataset.add_scan_source(
                LabelsScanSource(id="series", path=str(base), series="keypoints")
            )


def raw_file_names(dataset: Dataset, kind: RawKind) -> set[str]:
    return {Path(str(row["abs_path"])).name for row in raw_rows(dataset, kind)}


def prune_kind(dataset: Dataset, kind: RawKind, only: Sequence[str]) -> object:
    if kind == "tracks":
        return dataset.scan_tracks(only=only, prune_unsourced=True)
    return dataset.scan_labels(only=only, prune_unsourced=True)


@pytest.mark.parametrize("kind", ["tracks", "labels"])
class TestARestrictedPruneBesideASibling:
    """``prune_unsourced`` on a restricted scan is refused beside an unscanned sibling.

    A source over one folder with another owns rows the restricted scan does not
    walk, and a prune would read them as claimed by no source.
    """

    @staticmethod
    def two_sources_over_one_folder(
        tmp_path: Path, kind: RawKind, nested: bool
    ) -> Dataset:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        other = base / "day2" if nested else base
        touch_files(base / "a.npy", other / "b.npz")
        dataset.add_scan_source(raw_source(kind, "npy", base, ("*.npy",)))
        dataset.add_scan_source(raw_source(kind, "npz", other, ("*.npz",)))
        _ = scan_kind(dataset, kind)
        assert raw_file_names(dataset, kind) == {"a.npy", "b.npz"}
        return dataset

    @pytest.mark.parametrize("nested", [False, True], ids=["equal", "nested"])
    def test_it_is_refused_and_the_index_is_unchanged(
        self, tmp_path: Path, kind: RawKind, nested: bool
    ) -> None:
        dataset = self.two_sources_over_one_folder(tmp_path, kind, nested)
        before = raw_index_bytes(dataset, kind)

        with pytest.raises(ValueError, match="'npz'.*'npy'|'npy'.*'npz'") as raised:
            _ = prune_kind(dataset, kind, ["npy"])

        assert "pruning unsourced" in str(raised.value)
        assert raw_index_bytes(dataset, kind) == before

    def test_the_other_source_scanned_alone_is_refused_too(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = self.two_sources_over_one_folder(tmp_path, kind, nested=False)
        before = raw_index_bytes(dataset, kind)

        with pytest.raises(ValueError, match="pruning unsourced"):
            _ = prune_kind(dataset, kind, ["npz"])

        assert raw_index_bytes(dataset, kind) == before

    def test_scanning_both_sources_prunes(self, tmp_path: Path, kind: RawKind) -> None:
        dataset = self.two_sources_over_one_folder(tmp_path, kind, nested=False)
        touch_files(tmp_path / "far" / "c.npy")
        dataset.add_scan_source(raw_source(kind, "far", tmp_path / "far", ("*.npy",)))
        _ = scan_kind(dataset, kind)
        assert "c.npy" in raw_file_names(dataset, kind)

        _ = prune_kind(dataset, kind, ["npy", "npz"])

        assert raw_file_names(dataset, kind) == {"a.npy", "b.npz"}

    def test_a_full_scan_prunes(self, tmp_path: Path, kind: RawKind) -> None:
        dataset = self.two_sources_over_one_folder(tmp_path, kind, nested=False)

        _ = prune_kind(dataset, kind, [])

        assert raw_file_names(dataset, kind) == {"a.npy", "b.npz"}

    def test_without_a_sibling_it_prunes_as_before(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = make_dataset(tmp_path)
        touch_files(tmp_path / "one" / "a.npy", tmp_path / "two" / "b.npy")
        for name in ("one", "two"):
            dataset.add_scan_source(raw_source(kind, name, tmp_path / name, ("*.npy",)))
        _ = scan_kind(dataset, kind)
        assert raw_file_names(dataset, kind) == {"a.npy", "b.npy"}

        _ = prune_kind(dataset, kind, ["one"])

        assert raw_file_names(dataset, kind) == {"a.npy"}

    def test_a_sibling_of_the_other_kind_does_not_count(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy")
        other: RawKind = "labels" if kind == "tracks" else "tracks"
        dataset.add_scan_source(raw_source(kind, "mine", base, ("*.npy",)))
        dataset.add_scan_source(raw_source(other, "theirs", base, ("*.npy",)))
        _ = scan_kind(dataset, kind)

        _ = prune_kind(dataset, kind, ["mine"])

        assert raw_file_names(dataset, kind) == {"a.npy"}


@pytest.mark.parametrize("kind", ["tracks", "labels"])
class TestARemovalBesideASiblingOverOneFolder:
    """A removal counts and drops the removed source's rows, not its sibling's."""

    @staticmethod
    def two_sources_over_one_folder(tmp_path: Path, kind: RawKind) -> Dataset:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "data"
        touch_files(base / "a.npy", base / "a2.npy", base / "b.npz")
        dataset.add_scan_source(raw_source(kind, "npy", base, ("*.npy",)))
        dataset.add_scan_source(raw_source(kind, "npz", base, ("*.npz",)))
        _ = scan_kind(dataset, kind)
        assert raw_file_names(dataset, kind) == {"a.npy", "a2.npy", "b.npz"}
        return dataset

    def test_the_count_is_the_removed_sources_rows_alone(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = self.two_sources_over_one_folder(tmp_path, kind)

        assert dataset.remove_scan_source(kind, "npy") == 2

        assert raw_file_names(dataset, kind) == {"a.npy", "a2.npy", "b.npz"}

    def test_the_drop_removes_the_removed_sources_rows_alone(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = self.two_sources_over_one_folder(tmp_path, kind)
        removal = dataset.source_removal(kind, "npz")

        assert dataset.remove_scan_source(kind, "npz", removal=removal) == 1
        assert (
            dataset.drop_claimed_rows(
                kind, removal.claim, series=removal.series, spared=removal.spared
            )
            == 1
        )

        assert raw_file_names(dataset, kind) == {"a.npy", "a2.npy"}

    def test_rows_kept_under_a_sibling_go_on_its_next_scan(
        self, tmp_path: Path, kind: RawKind
    ) -> None:
        dataset = self.two_sources_over_one_folder(tmp_path, kind)
        _ = dataset.remove_scan_source(kind, "npz")

        _ = scan_kind(dataset, kind)

        assert raw_file_names(dataset, kind) == {"a.npy", "a2.npy"}


def stem_named_source(
    source_id: str, path: Path, *, sequence_path_pattern: str | None = None
) -> TracksScanSource:
    """A tracks source of a format that names one sequence per file, by its stem."""
    return TracksScanSource(
        id=source_id,
        path=str(path),
        patterns=("*.h5",),
        src_format="sleap_analysis_h5",
        sequence_path_pattern=sequence_path_pattern,
    )


def index_files(dataset: Dataset) -> dict[str, bytes]:
    """The bytes of the tracks index and of its sequence index."""
    root = dataset.get_root("tracks_raw")
    return {name: (root / name).read_bytes() for name in ("index.csv", "sequences.csv")}


class TestTwoFilesNamingOneEntry:
    """A scan refuses two files of one format that name one entry, before it writes.

    A conversion would keep one of the two and lose the other recording without
    a message.
    """

    def test_two_files_under_one_source_are_refused_naming_both_and_the_source(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "raw"
        touch_files(base / "a" / "trial1.h5", base / "a" / "other.h5")
        dataset.add_scan_source(stem_named_source("lab", base))
        _ = dataset.scan_tracks()
        before = index_files(dataset)
        touch_files(base / "b" / "trial1.h5")

        with pytest.raises(RawEntryNamedTwice) as refused:
            _ = dataset.scan_tracks()

        message = str(refused.value)
        assert str(base / "a" / "trial1.h5") in message
        assert str(base / "b" / "trial1.h5") in message
        assert "'lab'" in message
        assert "trial1" in message
        assert index_files(dataset) == before

    def test_two_sources_are_each_named_for_their_file(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        left, right = tmp_path / "left", tmp_path / "right"
        touch_files(left / "trial1.h5", right / "other.h5")
        dataset.add_scan_source(stem_named_source("left", left))
        dataset.add_scan_source(stem_named_source("right", right))
        _ = dataset.scan_tracks()
        before = index_files(dataset)
        touch_files(right / "trial1.h5")

        with pytest.raises(RawEntryNamedTwice) as refused:
            _ = dataset.scan_tracks(only=["right"])

        message = str(refused.value)
        assert str(left / "trial1.h5") in message and "'left'" in message
        assert str(right / "trial1.h5") in message and "'right'" in message
        assert index_files(dataset) == before

    def test_a_later_source_naming_an_earlier_sources_entry_leaves_every_pass_unwritten(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        first, second = tmp_path / "first", tmp_path / "second"
        touch_files(first / "one.h5", second / "two.h5")
        dataset.add_scan_source(stem_named_source("first", first))
        dataset.add_scan_source(stem_named_source("second", second))
        _ = dataset.scan_tracks()
        before = index_files(dataset)
        touch_files(first / "fresh.h5", second / "fresh.h5")

        with pytest.raises(RawEntryNamedTwice):
            _ = dataset.scan_tracks()

        assert index_files(dataset) == before

    def test_a_later_sources_capture_that_names_nothing_leaves_an_earlier_pass_unwritten(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        first, second = tmp_path / "first", tmp_path / "second"
        touch_files(first / "one.h5")
        dataset.add_scan_source(stem_named_source("first", first))
        _ = dataset.scan_tracks()
        before = index_files(dataset)
        touch_files(first / "fresh.h5", second / "two.h5")
        dataset.add_scan_source(
            stem_named_source(
                "second", second, sequence_path_pattern=r"^(never)?([^/]*)\.h5$"
            )
        )

        with pytest.raises(ValueError, match="'second'.*names nothing") as refused:
            _ = dataset.scan_tracks()

        assert str(second / "two.h5") in str(refused.value)
        assert index_files(dataset) == before

    def test_a_later_sources_unregistered_format_leaves_an_earlier_pass_unwritten(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        first, second = tmp_path / "first", tmp_path / "second"
        touch_files(first / "one.h5")
        dataset.add_scan_source(stem_named_source("first", first))
        _ = dataset.scan_tracks()
        before = index_files(dataset)
        touch_files(first / "fresh.h5", second / "two.h5")
        dataset.add_scan_source(
            TracksScanSource(
                id="second",
                path=str(second),
                patterns=("*.h5",),
                src_format="no_such_format",
            )
        )

        with pytest.raises(ValueError, match="no_such_format"):
            _ = dataset.scan_tracks()

        assert index_files(dataset) == before

    def test_a_sequence_path_pattern_naming_one_entry_from_two_folders_is_refused(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "raw"
        touch_files(base / "x" / "s1" / "t.h5", base / "y" / "s1" / "t.h5")
        pattern = r"^[^/]+/([^/]+)/"
        dataset.add_scan_source(
            stem_named_source("by-folder", base, sequence_path_pattern=pattern)
        )

        with pytest.raises(RawEntryNamedTwice) as refused:
            _ = dataset.scan_tracks()

        message = str(refused.value)
        assert str(base / "x" / "s1" / "t.h5") in message
        assert str(base / "y" / "s1" / "t.h5") in message
        assert "'by-folder'" in message
        assert pattern in message
        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()

    def test_the_index_alone_refuses_two_files_before_it_writes(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "raw"
        touch_files(base / "a" / "trial1.h5", base / "b" / "trial1.h5")

        with pytest.raises(RawEntryNamedTwice, match="no declared source"):
            _ = dataset.index_tracks_raw(
                [base], patterns=["*.h5"], src_format="sleap_analysis_h5"
            )

        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()

    def test_a_stale_row_of_a_removed_source_refuses_the_scan_without_a_prune(
        self, tmp_path: Path
    ) -> None:
        dataset = stale_row_dataset(tmp_path)
        before = index_files(dataset)

        with pytest.raises(RawEntryNamedTwice, match="no declared source") as refused:
            _ = dataset.scan_tracks()

        assert str(tmp_path / "gone" / "trial1.h5") in str(refused.value)
        assert index_files(dataset) == before

    def test_a_prune_clears_the_stale_row_instead_of_refusing(
        self, tmp_path: Path
    ) -> None:
        dataset = stale_row_dataset(tmp_path)

        _ = dataset.scan_tracks(prune_unsourced=True)

        assert {Path(str(row["abs_path"])) for row in raw_rows(dataset, "tracks")} == {
            tmp_path / "live" / "trial1.h5",
            tmp_path / "live" / "other.h5",
        }

    def test_a_file_moved_into_an_earlier_source_is_not_refused_for_its_old_row(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        first, second = tmp_path / "first", tmp_path / "second"
        touch_files(first / "one.h5", second / "moved.h5")
        dataset.add_scan_source(stem_named_source("first", first))
        dataset.add_scan_source(stem_named_source("second", second))
        _ = dataset.scan_tracks()
        (second / "moved.h5").unlink()
        touch_files(first / "moved.h5")

        _ = dataset.scan_tracks()

        assert {Path(str(row["abs_path"])) for row in raw_rows(dataset, "tracks")} == {
            first / "one.h5",
            first / "moved.h5",
        }

    def test_a_rescan_of_a_source_holding_symlinks_is_accepted(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base, outside = tmp_path / "raw", tmp_path / "outside"
        touch_files(base / "inside.h5", outside / "target.h5")
        (base / "via_inside.h5").symlink_to(base / "inside.h5")
        (base / "via_outside.h5").symlink_to(outside / "target.h5")
        dataset.add_scan_source(stem_named_source("links", base))

        _ = dataset.scan_tracks()
        first = index_files(dataset)["index.csv"]
        _ = dataset.scan_tracks()

        assert index_files(dataset)["index.csv"] == first
        assert len(raw_rows(dataset, "tracks")) == 2

    def test_a_promoted_correction_followed_by_a_rescan_is_accepted(
        self, tmp_path: Path
    ) -> None:
        dataset, recording = tracktor_dataset(
            tmp_path / "ds", "tracktor_corrected_tsv", "*.tsv"
        )
        (row,) = raw_rows(dataset, "tracks")
        dataset.add_scan_source(
            TracksScanSource(
                id="tracktor",
                path=str(dataset.base_dir / "raw"),
                patterns=("*.tsv",),
                src_format="tracktor_corrected_tsv",
            )
        )
        _ = promote_correction(
            dataset,
            str(row["group"]),
            str(row["sequence"]),
            recording.tsv,
            src_format="tracktor_corrected_tsv",
            apply=True,
        )

        _ = dataset.scan_tracks()

        names = sorted(
            Path(str(each["abs_path"])).name for each in raw_rows(dataset, "tracks")
        )
        assert names == sorted([recording.tsv.name, "corrected.rev1.tsv"])

    def test_a_merging_format_and_a_multi_sequence_source_are_not_checked(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        merging, multi = tmp_path / "trex", tmp_path / "calms"
        touch_files(
            merging / "hex_7_fish0.npz",
            merging / "hex_7_fish1.npz",
            multi / "a" / "x.npy",
            multi / "a" / "y.npy",
        )
        dataset.add_scan_source(
            TracksScanSource(
                id="trex", path=str(merging), patterns=("*.npz",), src_format="trex_npz"
            )
        )
        dataset.add_scan_source(
            TracksScanSource(
                id="calms",
                path=str(multi),
                patterns=("*.npy",),
                src_format="calms21_npy",
                multi_sequences_per_file=True,
                group_from="parent",
            )
        )

        _ = dataset.scan_tracks()

        assert len(raw_rows(dataset, "tracks")) == 4

    def test_an_enumerable_format_with_one_stem_in_two_folders_is_not_checked(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "calms"
        touch_files(base / "a" / "x.npy", base / "b" / "x.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="calms",
                path=str(base),
                patterns=("*.npy",),
                src_format="calms21_npy",
            )
        )

        _ = dataset.scan_tracks()

        assert len(raw_rows(dataset, "tracks")) == 2

    def test_a_labels_source_with_two_files_naming_one_entry_is_not_refused(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        base = tmp_path / "labels"
        touch_files(base / "a" / "ann.csv", base / "b" / "ann.csv")
        dataset.add_scan_source(
            LabelsScanSource(id="ann", path=str(base), patterns=("*.csv",))
        )

        _ = dataset.scan_labels()

        assert len(raw_rows(dataset, "labels")) == 2


def stale_row_dataset(tmp_path: Path) -> Dataset:
    """A dataset holding a row of a removed source whose entry a live file also names.

    The removed source's ``gone/trial1.h5`` stays indexed, and the live source
    now holds ``live/trial1.h5``, which names the same entry.
    """
    dataset = make_dataset(tmp_path)
    gone, live = tmp_path / "gone", tmp_path / "live"
    touch_files(gone / "trial1.h5", live / "other.h5")
    dataset.add_scan_source(stem_named_source("gone", gone))
    dataset.add_scan_source(stem_named_source("live", live))
    _ = dataset.scan_tracks()
    assert dataset.remove_scan_source("tracks", "gone", save=False) == 1
    touch_files(live / "trial1.h5")
    return dataset


class TestTheSourceLookupsOfTheDataset:
    """``Dataset.raw_index_sources`` and ``Dataset.sources_claiming``."""

    def test_the_sources_that_feed_the_raw_index_leave_out_a_series_source(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        files, series = tmp_path / "files", tmp_path / "series"
        touch_files(files / "a.csv")
        series.mkdir()
        dataset.add_scan_source(
            LabelsScanSource(id="files", path=str(files), patterns=("*.csv",))
        )
        dataset.add_scan_source(
            LabelsScanSource(id="series", path=str(series), series="keypoints")
        )
        dataset.add_scan_source(stem_named_source("tracks", files))

        assert [s.id for s in dataset.raw_index_sources("labels")] == ["files"]
        assert [s.id for s in dataset.raw_index_sources("tracks")] == ["tracks"]

    def test_a_path_is_claimed_by_the_sources_whose_claim_covers_it(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        left, right = tmp_path / "left", tmp_path / "right"
        touch_files(left / "a.h5", right / "b.h5")
        dataset.add_scan_source(stem_named_source("left", left))
        dataset.add_scan_source(stem_named_source("right", right))

        claiming = dataset.sources_claiming("tracks")

        assert claiming(left / "a.h5") == ["left"]
        assert claiming(right / "b.h5") == ["right"]
        assert claiming(tmp_path / "elsewhere.h5") == []
