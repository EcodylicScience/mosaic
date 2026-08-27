"""Declared scan sources, and what a scan is allowed to replace.

A scan used to rewrite the whole index from whatever it had just walked. That
made three things true that should not have been: scanning directory A and then
directory B kept only B; any scan destroyed rows pointing at files outside the
dataset, which is the mechanism that replaced an outside root; and one dataset
could not hold two source formats at once. These tests are the record that none
of them is true any more.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.manifest import MediaScanSource, TracksScanSource
from mosaic.core.pipeline.scan_claim import ScanClaim

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

        TREx joins an entry's clips into one ``.pv``, so a duplicated row is a
        clip handed to the conversion twice: the session's frame count doubles
        and its footage repeats mid-timeline, with nothing raising.
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
        item = build_work_items(dataset, scope, kind="trex")[0]
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

        An embedded NUL is the trigger that works on every supported
        interpreter: ``Path.resolve`` rejects it with ``ValueError`` on 3.12 and
        3.13 alike. The symlink loop below does not, which is why it is a
        separate, version-gated test.
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

    @pytest.mark.skipif(
        sys.version_info >= (3, 13),
        reason="Path.resolve stopped raising on a symlink loop in 3.13",
    )
    def test_a_symlink_loop_is_skipped_on_the_versions_that_raise(
        self, tmp_path: Path
    ) -> None:
        """The other half of the guard, where the interpreter still supplies it.

        Python 3.12 raises ``RuntimeError`` resolving a symlink loop; 3.13
        returns the path unchanged, so the row resolves to itself and is
        claimed. Both are correct for a scan -- the walk cannot read such a file,
        so no row like this is produced -- and pinning the 3.12 behavior keeps
        the guard honest for as long as mosaic supports that interpreter.
        """
        dataset = make_dataset(tmp_path)
        directory = dataset.get_root("media_raw") / "cage__day1"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "a.mp4").symlink_to(directory / "b.mp4")
        (directory / "b.mp4").symlink_to(directory / "a.mp4")

        claim = dataset.walked_claim([{"abs_path": str(directory / "a.mp4")}])

        assert not claim.claims((directory / "a.mp4").absolute())

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
