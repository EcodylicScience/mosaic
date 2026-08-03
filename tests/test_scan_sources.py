"""Declared scan sources, and what a scan is allowed to replace.

A scan used to rewrite the whole index from whatever it had just walked. That
made three things true that should not have been: scanning directory A and then
directory B kept only B; any scan destroyed rows pointing at files outside the
dataset, which is the mechanism that replaced an outside root; and one dataset
could not hold two source formats at once. These tests are the record that none
of them is true any more.
"""

from __future__ import annotations

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
        _ = dataset.scan_tracks_raw()

        rows = dataset.read_tracks_raw_index()
        by_name = {Path(str(r["abs_path"])).name: str(r["src_format"]) for r in rows}
        assert by_name == {"a.npz": "trex_npz", "b.npy": "calms21_npy"}
