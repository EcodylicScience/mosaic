"""What a source scan reports when it drops or cannot place files.

A scan used to return a source's missing directory, a file source's listed
files and an unmatched group pattern as an empty index with exit status 0. Each
is now a finding: printed on stderr and returned for ``mosaic scan --json``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.manifest import (
    LabelsScanSource,
    MediaScanSource,
    TracksScanSource,
)
from mosaic.core.pipeline.tracks_raw_index import read_tracks_raw_index
from mosaic.core.scan_findings import ScanFinding

VideoWriter = Callable[..., None]


def make_dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="findings", base_dir=tmp_path / "ds")
    return Dataset(manifest_path=manifest).load()


def touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"\x00")


def tracks_names(dataset: Dataset) -> set[str]:
    return {Path(row["abs_path"]).name for row in dataset.read_tracks_raw_index()}


def labels_names(dataset: Dataset) -> set[str]:
    index = dataset.get_root("labels_raw") / "index.csv"
    return {Path(row["abs_path"]).name for row in read_tracks_raw_index(index)}


def groups(dataset: Dataset) -> dict[str, str]:
    return {
        Path(row["abs_path"]).name: row["group"]
        for row in dataset.read_tracks_raw_index()
    }


def problems(findings: tuple[ScanFinding, ...]) -> list[str]:
    return [finding.problem for finding in findings]


class TestAMissingPathIsReported:
    def test_a_dangling_link_names_the_source_and_its_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = make_dataset(tmp_path)
        link = tmp_path / "dangling"
        link.symlink_to(tmp_path / "nowhere" / "real")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(link), patterns=("*.npy",))
        )

        scanned = dataset.scan_tracks()

        (finding,) = scanned.findings
        assert (finding.kind, finding.source, finding.problem) == (
            "tracks",
            "s",
            "path_missing",
        )
        assert finding.path == str(link)
        assert "link" in finding.message
        err = capsys.readouterr().err
        assert "'s'" in err
        assert str(link) in err

    def test_a_directory_that_never_existed_is_reported_for_labels_too(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = make_dataset(tmp_path)
        missing = tmp_path / "never"
        dataset.add_scan_source(
            LabelsScanSource(id="ann", path=str(missing), patterns=("*.csv",))
        )

        scanned = dataset.scan_labels()

        (finding,) = scanned.findings
        assert (finding.kind, finding.source, finding.problem) == (
            "labels",
            "ann",
            "path_missing",
        )
        assert finding.path == str(missing)
        assert str(missing) in capsys.readouterr().err

    def test_a_media_scan_reports_the_same_finding(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = make_dataset(tmp_path)
        missing = tmp_path / "unmounted"
        dataset.add_scan_source(MediaScanSource(id="nas", path=str(missing)))

        scanned = dataset.scan_media()

        (finding,) = scanned.findings
        assert (finding.kind, finding.source, finding.problem) == (
            "media",
            "nas",
            "path_missing",
        )
        assert finding.path == str(missing)
        assert str(missing) in capsys.readouterr().err

    def test_a_present_path_reports_nothing(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "a.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), patterns=("*.npy",))
        )

        assert dataset.scan_tracks().findings == ()

    def test_a_missing_file_source_is_one_finding_not_one_per_file(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(
            TracksScanSource(
                id="s", path=str(tmp_path / "never"), files=("a.npy", "b.npy")
            )
        )

        assert problems(dataset.scan_tracks().findings) == ["path_missing"]

    def test_a_listed_file_that_has_gone_is_a_finding(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "here.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), files=("here.npy", "gone.npy"))
        )

        scanned = dataset.scan_tracks()

        (finding,) = scanned.findings
        assert finding.problem == "listed_file_missing"
        assert finding.path == str(folder / "gone.npy")
        assert tracks_names(dataset) == {"here.npy"}


class TestAMissingPathKeepsItsRows:
    """An unmounted share is not a decision to drop what was indexed from it."""

    def test_tracks_rows_survive_the_directory_vanishing_and_returning(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "a.npy", folder / "b.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), patterns=("*.npy",))
        )
        _ = dataset.scan_tracks()
        away = tmp_path / "away"
        _ = folder.rename(away)

        gone = dataset.scan_tracks()

        assert tracks_names(dataset) == {"a.npy", "b.npy"}
        assert problems(gone.findings) == ["path_missing"]

        _ = away.rename(folder)
        back = dataset.scan_tracks()

        assert tracks_names(dataset) == {"a.npy", "b.npy"}
        assert back.findings == ()

    def test_a_dangling_link_keeps_the_rows_of_its_source(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        real = tmp_path / "real"
        touch(real / "a.npy")
        link = tmp_path / "link"
        link.symlink_to(real)
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(link), patterns=("*.npy",))
        )
        _ = dataset.scan_tracks()
        away = tmp_path / "away"
        _ = real.rename(away)

        gone = dataset.scan_tracks()

        assert tracks_names(dataset) == {"a.npy"}
        assert problems(gone.findings) == ["path_missing"]

    def test_labels_rows_survive_too(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "labels"
        touch(folder / "a.csv")
        dataset.add_scan_source(
            LabelsScanSource(id="ann", path=str(folder), patterns=("*.csv",))
        )
        _ = dataset.scan_labels()
        _ = folder.rename(tmp_path / "away")

        gone = dataset.scan_labels()

        assert labels_names(dataset) == {"a.csv"}
        assert problems(gone.findings) == ["path_missing"]

    def test_a_files_source_keeps_its_rows_when_its_path_is_missing(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "a.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), files=("a.npy",))
        )
        _ = dataset.scan_tracks()
        _ = folder.rename(tmp_path / "away")

        gone = dataset.scan_tracks()

        assert tracks_names(dataset) == {"a.npy"}
        assert problems(gone.findings) == ["path_missing"]

    def test_a_present_path_still_prunes_what_it_no_longer_holds(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "a.npy", folder / "b.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), patterns=("*.npy",))
        )
        _ = dataset.scan_tracks()
        (folder / "b.npy").unlink()

        scanned = dataset.scan_tracks()

        assert tracks_names(dataset) == {"a.npy"}
        assert scanned.findings == ()

    def test_a_missing_source_does_not_spare_the_rows_of_another(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        kept, gone = tmp_path / "kept", tmp_path / "gone"
        touch(kept / "a.npy", gone / "b.npy")
        for name, folder in (("kept", kept), ("gone", gone)):
            dataset.add_scan_source(
                TracksScanSource(id=name, path=str(folder), patterns=("*.npy",))
            )
        _ = dataset.scan_tracks()
        (kept / "a.npy").unlink()
        _ = gone.rename(tmp_path / "away")

        _ = dataset.scan_tracks()

        assert tracks_names(dataset) == {"b.npy"}

    def test_media_rows_survive_the_directory_vanishing_and_returning(
        self, tmp_path: Path, write_cfr_mp4: VideoWriter
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "nas"
        write_cfr_mp4(folder / "one.mp4")
        dataset.add_scan_source(MediaScanSource(id="nas", path=str(folder)))
        _ = dataset.scan_media()
        away = tmp_path / "away"
        _ = folder.rename(away)

        gone = dataset.scan_media()

        assert {
            Path(str(row["abs_path"])).name for row in dataset.read_media_index()
        } == {"one.mp4"}
        assert problems(gone.findings) == ["path_missing"]

        _ = away.rename(folder)
        back = dataset.scan_media()

        assert {
            Path(str(row["abs_path"])).name for row in dataset.read_media_index()
        } == {"one.mp4"}
        assert back.findings == ()


class TestAFileSourceIndexesItsListedFiles:
    def test_a_tracks_file_source_indexes_exactly_its_files(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "a.npy", folder / "b.npy", folder / "c.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), files=("a.npy", "c.npy"))
        )

        scanned = dataset.scan_tracks()

        assert tracks_names(dataset) == {"a.npy", "c.npy"}
        assert scanned.findings == ()

    def test_a_labels_file_source_indexes_exactly_its_files(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "labels"
        touch(folder / "a.csv", folder / "b.csv")
        dataset.add_scan_source(
            LabelsScanSource(id="ann", path=str(folder), files=("b.csv",))
        )

        scanned = dataset.scan_labels()

        assert labels_names(dataset) == {"b.csv"}
        assert scanned.findings == ()

    def test_a_listed_file_is_indexed_whatever_its_suffix(self, tmp_path: Path) -> None:
        """A file source has no patterns: it claims what it lists."""
        dataset = make_dataset(tmp_path / "one")
        folder = tmp_path / "raw"
        touch(
            folder / "fly_tracked_fixed_final.tsv",
            folder / "fish.slp",
            folder / "notes.txt",
        )
        dataset.add_scan_source(
            TracksScanSource(
                id="tracktor",
                path=str(folder),
                files=("fly_tracked_fixed_final.tsv",),
                src_format="tracktor_corrected_tsv",
            )
        )
        dataset.add_scan_source(
            TracksScanSource(
                id="sleap",
                path=str(folder),
                files=("fish.slp",),
                src_format="sleap_analysis_h5",
            )
        )

        scanned = dataset.scan_tracks()

        assert tracks_names(dataset) == {"fly_tracked_fixed_final.tsv", "fish.slp"}
        assert scanned.findings == ()

    def test_a_labels_file_source_indexes_a_suffix_outside_the_defaults(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "labels"
        touch(folder / "a.json")
        dataset.add_scan_source(
            LabelsScanSource(id="ann", path=str(folder), files=("a.json",))
        )

        scanned = dataset.scan_labels()

        assert labels_names(dataset) == {"a.json"}
        assert scanned.findings == ()

    def test_a_listed_file_that_a_rule_skips_is_reported(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "a.npy", folder / "._b.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), files=("a.npy", "._b.npy"))
        )

        scanned = dataset.scan_tracks()

        (finding,) = scanned.findings
        assert finding.problem == "listed_file_skipped"
        assert finding.path == str(folder / "._b.npy")
        assert tracks_names(dataset) == {"a.npy"}

    def test_a_directory_walk_is_unchanged(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        folder = tmp_path / "raw"
        touch(folder / "a.npy", folder / "sub" / "b.npy", folder / "notes.txt")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(folder), patterns=("*.npy",))
        )

        scanned = dataset.scan_tracks()

        assert tracks_names(dataset) == {"a.npy", "b.npy"}
        assert scanned.findings == ()


class TestGroupPatterns:
    def test_a_path_pattern_groups_by_the_folder_under_the_source_root(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "release"
        touch(
            root / "Behavior_data" / "strainA" / "x.npy",
            root / "Behavior_data" / "strainB" / "y.npy",
        )
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                group_path_pattern=r"Behavior_data/([^/]+)/",
            )
        )

        scanned = dataset.scan_tracks()

        assert groups(dataset) == {"x.npy": "strainA", "y.npy": "strainB"}
        assert scanned.findings == ()

    def test_a_path_pattern_does_not_see_the_path_above_the_source_root(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "Behavior_data" / "release"
        touch(root / "strainA" / "x.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                group_path_pattern=r"Behavior_data/([^/]+)/",
            )
        )

        scanned = dataset.scan_tracks()

        (finding,) = scanned.findings
        assert finding.problem == "group_path_pattern_unmatched"
        assert finding.pattern == r"Behavior_data/([^/]+)/"
        assert groups(dataset) == {"x.npy": ""}

    def test_a_path_pattern_groups_the_files_of_a_multi_sequence_source(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "release"
        touch(root / "strainA" / "x.npy", root / "strainB" / "y.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                multi_sequences_per_file=True,
                group_path_pattern=r"^([^/]+)/",
            )
        )

        scanned = dataset.scan_tracks()

        assert groups(dataset) == {"x.npy": "strainA", "y.npy": "strainB"}
        assert scanned.findings == ()

    def test_a_path_pattern_applies_to_a_labels_source(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "labels"
        touch(root / "g1" / "a.csv")
        dataset.add_scan_source(
            LabelsScanSource(
                id="ann",
                path=str(root),
                patterns=("*.csv",),
                group_path_pattern=r"^([^/]+)/",
            )
        )

        _ = dataset.scan_labels()

        index = dataset.get_root("labels_raw") / "index.csv"
        assert [row["group"] for row in read_tracks_raw_index(index)] == ["g1"]

    def test_a_path_pattern_applies_to_a_file_source(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "g1" / "a.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                files=("g1/a.npy",),
                group_path_pattern=r"^([^/]+)/",
            )
        )

        _ = dataset.scan_tracks()

        assert groups(dataset) == {"a.npy": "g1"}

    def test_a_stem_pattern_that_matches_no_file_names_the_pattern(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The reported case: a folder pattern given as a stem pattern."""
        dataset = make_dataset(tmp_path)
        root = tmp_path / "release"
        touch(root / "Behavior_data" / "strainA" / "x.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                group_pattern=r"Behavior_data/([^/]+)/",
            )
        )

        scanned = dataset.scan_tracks()

        (finding,) = scanned.findings
        assert (finding.source, finding.problem) == ("s", "group_pattern_unmatched")
        assert finding.pattern == r"Behavior_data/([^/]+)/"
        assert r"Behavior_data/([^/]+)/" in capsys.readouterr().err

    def test_a_stem_pattern_that_matches_a_file_reports_nothing(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "hex_1.npy", root / "plain.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                group_pattern=r"^(hex)_",
            )
        )

        scanned = dataset.scan_tracks()

        assert groups(dataset) == {"hex_1.npy": "hex", "plain.npy": ""}
        assert scanned.findings == ()

    def test_a_pattern_over_no_files_does_not_also_claim_to_match_none(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(tmp_path / "never"),
                group_path_pattern=r"([^/]+)/",
            )
        )

        assert problems(dataset.scan_tracks().findings) == ["path_missing"]

    def test_the_two_patterns_cannot_both_be_declared(self) -> None:
        with pytest.raises(ValueError, match="group_pattern"):
            _ = TracksScanSource(
                id="s", path="/x", group_pattern="(a)", group_path_pattern="(b)/"
            )

    def test_a_path_pattern_needs_a_capture_group(self) -> None:
        with pytest.raises(ValueError, match="capture group"):
            _ = TracksScanSource(id="s", path="/x", group_path_pattern="strain/")

    def test_a_path_pattern_that_is_not_a_regex_is_refused(self) -> None:
        with pytest.raises(ValueError, match="regular expression"):
            _ = TracksScanSource(id="s", path="/x", group_path_pattern="(strain")
