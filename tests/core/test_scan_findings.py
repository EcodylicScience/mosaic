"""What a source scan reports when it drops or cannot place files.

A source's missing directory, a file source's listed files that a rule skips and
a file a group pattern misses are each a finding, not an empty index with exit
status 0: printed on stderr and returned for ``mosaic scan --json``.
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
from mosaic.core.scan_findings import FindingLog, ScanFinding
from tests.helpers import write_trex_npz

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
        assert finding.message.endswith("so no file is indexed")
        assert tracks_names(dataset) == set()

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
        """A folder pattern given as a stem pattern matches nothing and says so."""
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
        assert finding.message.endswith("so no file is indexed")
        assert tracks_names(dataset) == set()
        assert r"Behavior_data/([^/]+)/" in capsys.readouterr().err

    def test_a_stem_pattern_that_matches_every_file_reports_nothing(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "hex_1.npy", root / "hex_2.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                group_pattern=r"^(hex)_",
            )
        )

        scanned = dataset.scan_tracks()

        assert groups(dataset) == {"hex_1.npy": "hex", "hex_2.npy": "hex"}
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


class TestAnUnmatchedRowNamesItsLine:
    def test_a_row_that_names_no_entry_is_recorded_printed_and_dumped(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = FindingLog(kind="labels", source="design.csv")

        log.report("row_unmatched", "row names no entry: 'x'", line=7)

        (finding,) = log.findings
        assert (finding.problem, finding.line) == ("row_unmatched", 7)
        assert capsys.readouterr().err == (
            "[WARN] labels source 'design.csv': row names no entry: 'x'\n"
        )
        assert ScanFinding.model_validate(finding.model_dump()) == finding
        assert finding.model_dump()["line"] == 7

    def test_a_finding_without_a_line_dumps_a_null_line(self) -> None:
        log = FindingLog(kind="tracks", source="s")

        log.report("path_missing", "gone")

        (finding,) = log.findings
        assert finding.line is None
        assert finding.model_dump()["line"] is None


PATTERN_KINDS = {"group_pattern": r"^(hex)_", "group_path_pattern": r"^(hex)/"}


def declare_with_a_miss(
    dataset: Dataset, root: Path, kind: str, *, misses: int = 1
) -> None:
    """A tracks source of two files the pattern matches and *misses* it misses."""
    if kind == "group_pattern":
        touch(root / "hex_1.npy", root / "hex_2.npy")
        touch(*(root / f"plain_{i}.npy" for i in range(misses)))
    else:
        touch(root / "hex" / "a.npy", root / "hex" / "b.npy")
        touch(*(root / "other" / f"m_{i}.npy" for i in range(misses)))
    pattern = PATTERN_KINDS[kind]
    dataset.add_scan_source(
        TracksScanSource(
            id="s",
            path=str(root),
            patterns=("*.npy",),
            group_pattern=pattern if kind == "group_pattern" else None,
            group_path_pattern=pattern if kind == "group_path_pattern" else None,
        )
    )


class TestAFileThePatternMissesIsNotIndexed:
    @pytest.mark.parametrize("kind", sorted(PATTERN_KINDS))
    def test_a_miss_is_skipped_and_counted_in_one_finding(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        declare_with_a_miss(dataset, root, kind)

        scanned = dataset.scan_tracks()

        (finding,) = scanned.findings
        assert (finding.kind, finding.source, finding.problem) == (
            "tracks",
            "s",
            "group_pattern_missed",
        )
        assert finding.count == 1
        missed = (
            root / "plain_0.npy" if kind == "group_pattern" else root / "other/m_0.npy"
        )
        assert finding.sample == (str(missed),)
        assert "so 1 file is not indexed" in finding.message
        assert finding.pattern == PATTERN_KINDS[kind]
        assert tracks_names(dataset) == (
            {"hex_1.npy", "hex_2.npy"}
            if kind == "group_pattern"
            else {"a.npy", "b.npy"}
        )
        assert finding.message in capsys.readouterr().err

    def test_a_stem_pattern_sees_the_sequence_the_converter_names(
        self, tmp_path: Path
    ) -> None:
        """TRex strips ``_fish0``: ``^hex_7$`` matches the sequence, not the stem."""
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_trex_npz(root / "hex_7_fish0.npz")
        write_trex_npz(root / "hex_7_fish1.npz")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npz",),
                src_format="trex_npz",
                group_pattern=r"^(hex_7)$",
            )
        )

        scanned = dataset.scan_tracks()

        assert scanned.findings == ()
        assert tracks_names(dataset) == {"hex_7_fish0.npz", "hex_7_fish1.npz"}
        assert set(groups(dataset).values()) == {"hex_7"}

    def test_the_two_patterns_are_still_refused_together_by_the_indexer(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "a.npy")

        with pytest.raises(ValueError, match="not both"):
            _ = dataset.index_tracks_raw(
                [root],
                patterns=["*.npy"],
                src_format="calms21_npy",
                group_pattern="(a)",
                group_path_pattern="(b)/",
                group_path_root=root,
            )

    def test_a_multi_sequence_source_with_group_from_misses_nothing(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "s1" / "x.npy", root / "s2" / "y.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                multi_sequences_per_file=True,
                group_from="parent",
            )
        )

        scanned = dataset.scan_tracks()

        assert scanned.findings == ()
        assert groups(dataset) == {"x.npy": "s1", "y.npy": "s2"}

    def test_an_indexer_given_a_group_rule_misses_nothing(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "a.npy", root / "b.npy")

        _ = dataset.index_tracks_raw(
            [root],
            patterns=["*.npy"],
            src_format="calms21_npy",
            group_from_path=lambda path: "g" if path.stem == "a" else "",
        )

        assert groups(dataset) == {"a.npy": "g", "b.npy": ""}

    def test_a_labels_source_skips_a_miss_too(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "labels"
        touch(root / "g1" / "a.csv", root / "g1" / "b.csv", root / "loose" / "c.csv")
        dataset.add_scan_source(
            LabelsScanSource(
                id="ann",
                path=str(root),
                patterns=("*.csv",),
                group_path_pattern=r"^(g\d)/",
            )
        )

        scanned = dataset.scan_labels()

        (finding,) = scanned.findings
        assert (finding.kind, finding.problem, finding.count) == (
            "labels",
            "group_pattern_missed",
            1,
        )
        assert finding.sample == (str(root / "loose" / "c.csv"),)
        assert labels_names(dataset) == {"a.csv", "b.csv"}

    def test_a_labels_indexer_given_a_group_rule_misses_nothing(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "labels"
        touch(root / "a.csv", root / "b.csv")

        _ = dataset.index_labels_raw(
            [root],
            patterns=["*.csv"],
            group_from_path=lambda path: "g" if path.stem == "a" else "",
        )

        assert labels_names(dataset) == {"a.csv", "b.csv"}

    def test_seven_misses_count_seven_and_name_five(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        declare_with_a_miss(dataset, root, "group_pattern", misses=7)

        scanned = dataset.scan_tracks()

        (finding,) = scanned.findings
        assert finding.count == 7
        assert finding.sample == tuple(str(root / f"plain_{i}.npy") for i in range(5))
        assert "so 7 files are not indexed" in finding.message

    def test_a_pattern_that_matches_no_file_is_the_only_finding(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "a.npy", root / "b.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s", path=str(root), patterns=("*.npy",), group_pattern=r"^(hex)_"
            )
        )

        scanned = dataset.scan_tracks()

        assert problems(scanned.findings) == ["group_pattern_unmatched"]
        assert tracks_names(dataset) == set()

    def test_a_rescan_drops_the_row_an_earlier_scan_indexed_with_no_group(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_trex_npz(root / "plain_fish0.npz")
        write_trex_npz(root / "hex_1_fish0.npz")
        dataset.add_scan_source(
            TracksScanSource(
                id="s", path=str(root), patterns=("*.npz",), src_format="trex_npz"
            )
        )
        _ = dataset.scan_tracks()
        _ = dataset.convert_all_tracks()
        converted = sorted(dataset.get_root("tracks").rglob("*.parquet"))
        assert len(converted) == 2
        _ = dataset.remove_scan_source("tracks", "s")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npz",),
                src_format="trex_npz",
                group_pattern=r"^(hex)_",
            )
        )

        scanned = dataset.scan_tracks()

        assert tracks_names(dataset) == {"hex_1_fish0.npz"}
        assert [finding.count for finding in scanned.findings] == [1]
        assert all(path.exists() for path in converted)

    def test_a_rescan_drops_the_row_of_a_missed_symlink_to_outside_the_source(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "hex_1.npy", tmp_path / "elsewhere" / "plain.npy")
        (root / "plain.npy").symlink_to(tmp_path / "elsewhere" / "plain.npy")
        dataset.add_scan_source(
            TracksScanSource(id="s", path=str(root), patterns=("*.npy",))
        )
        _ = dataset.scan_tracks()
        assert tracks_names(dataset) == {"hex_1.npy", "plain.npy"}
        _ = dataset.remove_scan_source("tracks", "s")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                group_pattern=r"^(hex)_",
            )
        )

        scanned = dataset.scan_tracks()

        assert [finding.count for finding in scanned.findings] == [1]
        assert tracks_names(dataset) == {"hex_1.npy"}

    def test_an_indexed_symlink_keeps_one_row_across_rescans(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(tmp_path / "elsewhere" / "hex_2.npy", root / "plain.npy")
        (root / "hex_2.npy").symlink_to(tmp_path / "elsewhere" / "hex_2.npy")
        dataset.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(root),
                patterns=("*.npy",),
                group_pattern=r"^(hex)_",
            )
        )

        _ = dataset.scan_tracks()
        _ = dataset.scan_tracks()

        assert [row["group"] for row in dataset.read_tracks_raw_index()] == ["hex"]

    @pytest.mark.parametrize("patterned_first", [True, False])
    def test_a_missed_symlink_into_another_source_leaves_that_sources_row(
        self, tmp_path: Path, patterned_first: bool
    ) -> None:
        dataset = make_dataset(tmp_path)
        touch(tmp_path / "a" / "hex_1.npy", tmp_path / "b" / "owned.npy")
        (tmp_path / "a" / "plain.npy").symlink_to(tmp_path / "b" / "owned.npy")
        patterned = TracksScanSource(
            id="a",
            path=str(tmp_path / "a"),
            patterns=("*.npy",),
            group_pattern=r"^(hex)_",
        )
        plain = TracksScanSource(id="b", path=str(tmp_path / "b"), patterns=("*.npy",))
        for source in (patterned, plain) if patterned_first else (plain, patterned):
            dataset.add_scan_source(source)

        for _ in range(2):
            _ = dataset.scan_tracks()

            assert tracks_names(dataset) == {"hex_1.npy", "owned.npy"}

    @pytest.mark.parametrize("patterned_first", [True, False])
    def test_a_missed_symlink_to_a_file_another_source_walks_to_leaves_its_row(
        self, tmp_path: Path, patterned_first: bool
    ) -> None:
        dataset = make_dataset(tmp_path)
        target = tmp_path / "elsewhere" / "z.dat"
        touch(tmp_path / "a" / "hex_1.npy", target)
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "plain.npy").symlink_to(target)
        (tmp_path / "b" / "link.npy").symlink_to(target)
        patterned = TracksScanSource(
            id="a",
            path=str(tmp_path / "a"),
            patterns=("*.npy",),
            group_pattern=r"^(hex)_",
        )
        plain = TracksScanSource(id="b", path=str(tmp_path / "b"), patterns=("*.npy",))
        for source in (patterned, plain) if patterned_first else (plain, patterned):
            dataset.add_scan_source(source)

        for _ in range(2):
            _ = dataset.scan_tracks()

            assert tracks_names(dataset) == {"hex_1.npy", "z.dat"}

    def test_a_missed_symlink_to_a_file_another_source_rejects_drops_its_row(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        touch(tmp_path / "a" / "hex_1.npy", tmp_path / "b" / "notes.txt")
        (tmp_path / "a" / "plain.npy").symlink_to(tmp_path / "b" / "notes.txt")
        _ = dataset.index_tracks_raw([tmp_path / "a"], patterns=["*.npy"])
        assert tracks_names(dataset) == {"hex_1.npy", "notes.txt"}
        dataset.add_scan_source(
            TracksScanSource(
                id="a",
                path=str(tmp_path / "a"),
                patterns=("*.npy",),
                group_pattern=r"^(hex)_",
            )
        )
        dataset.add_scan_source(
            TracksScanSource(id="b", path=str(tmp_path / "b"), patterns=("*.npy",))
        )

        _ = dataset.scan_tracks()

        assert tracks_names(dataset) == {"hex_1.npy"}


HINZ_SEQUENCE_PATTERN = r"^[^/]+/([^/]+)/(day[^/]+)/([^/]+)/segm/trajectories\.csv$"
HINZ_GROUP_PATTERN = r"^([^/]+)/"


def write_hinz_layout(root: Path, *trials: str) -> None:
    """One ``trajectories.csv`` per ``<N>/<batch>/day<d>/<trial>`` of *trials*."""
    touch(*(root / trial / "segm" / "trajectories.csv" for trial in trials))


def named(dataset: Dataset) -> dict[str, tuple[str, str]]:
    """Each indexed file's ``(group, sequence)``, by its path below the dataset."""
    return {
        row["abs_path"]: (row["group"], row["sequence"])
        for row in dataset.read_tracks_raw_index()
    }


def scan_hinz(
    dataset: Dataset,
    root: Path,
    log: FindingLog,
    *,
    sequence_path_pattern: str | None = HINZ_SEQUENCE_PATTERN,
    group_pattern: str | None = None,
    group_path_pattern: str | None = HINZ_GROUP_PATTERN,
) -> Path:
    return dataset.index_tracks_raw(
        [root],
        patterns=["*.csv"],
        src_format="calms21_npy",
        sequence_path_pattern=sequence_path_pattern,
        group_pattern=group_pattern,
        group_path_pattern=group_path_pattern,
        group_path_root=root,
        findings=log,
    )


class TestASequenceIsNamedFromThePath:
    def test_the_capture_groups_name_the_sequence_and_the_first_folder_the_group(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_hinz_layout(
            root, "1/batchA/day1/t1", "1/batchB/day1/t1", "2/batchA/day2/t7"
        )
        log = FindingLog("tracks", "hinz")

        _ = scan_hinz(dataset, root, log)

        assert sorted(named(dataset).values()) == [
            ("1", "batchA__day1__t1"),
            ("1", "batchB__day1__t1"),
            ("2", "batchA__day2__t7"),
        ]
        assert log.findings == []

    def test_a_pattern_missing_some_files_reports_the_count_and_a_sample(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_hinz_layout(root, "1/batchA/day1/t1", "1/batchA/day1/t2")
        stray = root / "1" / "batchA" / "notes" / "trajectories.csv"
        touch(stray)
        log = FindingLog("tracks", "hinz")

        _ = scan_hinz(dataset, root, log)

        (finding,) = log.findings
        assert (finding.kind, finding.source, finding.problem) == (
            "tracks",
            "hinz",
            "sequence_path_pattern_missed",
        )
        assert finding.count == 1
        assert finding.sample == (str(stray),)
        assert finding.pattern == HINZ_SEQUENCE_PATTERN
        assert "so 1 file is not indexed" in finding.message
        assert finding.message in capsys.readouterr().err
        assert len(named(dataset)) == 2

    def test_a_sample_names_five_of_seven_misses(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_hinz_layout(root, "1/batchA/day1/t1")
        touch(*(root / "1" / "x" / f"m{i}" / "trajectories.csv" for i in range(7)))
        log = FindingLog("tracks", "hinz")

        _ = scan_hinz(dataset, root, log)

        (finding,) = log.findings
        assert finding.count == 7
        assert len(finding.sample) == 5

    def test_a_pattern_missing_every_file_is_the_only_finding(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "a.csv", root / "b.csv")
        log = FindingLog("tracks", "hinz")

        _ = scan_hinz(dataset, root, log)

        (finding,) = log.findings
        assert finding.problem == "sequence_path_pattern_unmatched"
        assert finding.pattern == HINZ_SEQUENCE_PATTERN
        assert "none of the 2" in finding.message
        assert named(dataset) == {}

    def test_a_stem_group_pattern_searches_the_sequence_the_path_names(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_hinz_layout(
            root, "1/batchA/day1/t1", "1/batchA/day1/t2", "2/batchB/day1/t3"
        )
        stray = root / "1" / "batchA" / "notes" / "trajectories.csv"
        touch(stray)
        log = FindingLog("tracks", "hinz")

        _ = scan_hinz(
            dataset,
            root,
            log,
            group_pattern=r"^(batchA)__",
            group_path_pattern=None,
        )

        assert sorted(named(dataset).values()) == [
            ("batchA", "batchA__day1__t1"),
            ("batchA", "batchA__day1__t2"),
        ]
        sequence_miss, group_miss = log.findings
        assert sequence_miss.problem == "sequence_path_pattern_missed"
        assert sequence_miss.count == 1
        assert group_miss.problem == "group_pattern_missed"
        assert group_miss.count == 1
        assert group_miss.sample == (
            str(root / "2/batchB/day1/t3/segm/trajectories.csv"),
        )
        assert "matched 2 of 3" in group_miss.message

    def test_a_group_pattern_that_misses_every_named_file_is_unmatched(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_hinz_layout(root, "1/batchA/day1/t1")
        touch(root / "1" / "notes" / "trajectories.csv")
        log = FindingLog("tracks", "hinz")

        _ = scan_hinz(
            dataset, root, log, group_pattern=r"^(zzz)", group_path_pattern=None
        )

        assert [finding.problem for finding in log.findings] == [
            "sequence_path_pattern_missed",
            "group_pattern_unmatched",
        ]
        assert "none of the 1" in log.findings[1].message

    def test_a_labels_scan_applies_the_pattern(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "labels"
        write_hinz_layout(root, "1/batchA/day1/t1")
        touch(root / "1" / "loose.csv")
        log = FindingLog("labels", "ann")

        _ = dataset.index_labels_raw(
            [root],
            patterns=["*.csv"],
            sequence_path_pattern=HINZ_SEQUENCE_PATTERN,
            group_path_pattern=HINZ_GROUP_PATTERN,
            group_path_root=root,
            findings=log,
        )

        index = dataset.get_root("labels_raw") / "index.csv"
        assert [
            (row["group"], row["sequence"]) for row in read_tracks_raw_index(index)
        ] == [("1", "batchA__day1__t1")]
        (finding,) = log.findings
        assert (finding.kind, finding.problem, finding.count) == (
            "labels",
            "sequence_path_pattern_missed",
            1,
        )

    def test_a_capture_that_is_not_an_entry_name_is_refused(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_hinz_layout(root, "1/batchA/day1/t1")

        with pytest.raises(
            ValueError, match="sequence may not contain a forward slash"
        ):
            _ = scan_hinz(
                dataset,
                root,
                FindingLog("tracks", "hinz"),
                sequence_path_pattern=r"^([^/]+/[^/]+)/",
            )

        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()


class TestACaptureThatNamesNothingIsRefused:
    @pytest.mark.parametrize(
        ("pattern", "folder"),
        [(r"(a)?(b)", "b"), (r"(x*)", "g")],
        ids=["optional group not taking part", "group matching empty"],
    )
    def test_the_refusal_names_the_file_the_source_and_the_pattern_before_a_write(
        self, tmp_path: Path, pattern: str, folder: str
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / folder / "t.csv")

        with pytest.raises(ValueError) as caught:
            _ = dataset.index_tracks_raw(
                [root],
                patterns=["*.csv"],
                src_format="calms21_npy",
                sequence_path_pattern=pattern,
                group_path_root=root,
                findings=FindingLog("tracks", "hinz"),
            )

        message = str(caught.value)
        assert str(root / folder / "t.csv") in message
        assert "'hinz'" in message
        assert pattern in message
        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()

    def test_a_pattern_without_a_capture_group_is_refused_before_a_write(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "t.csv")

        with pytest.raises(ValueError, match="capture group"):
            _ = dataset.index_tracks_raw(
                [root],
                patterns=["*.csv"],
                src_format="calms21_npy",
                sequence_path_pattern=r"t\.csv$",
                group_path_root=root,
            )

        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()


class TestTheSequencePatternIsRefusedBesideARuleItCannotServe:
    def test_it_is_refused_beside_multi_sequences_per_file(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "a" / "t.csv")

        with pytest.raises(ValueError, match="multi_sequences_per_file") as caught:
            _ = dataset.index_tracks_raw(
                [root],
                patterns=["*.csv"],
                src_format="calms21_npy",
                multi_sequences_per_file=True,
                group_from="parent",
                sequence_path_pattern=r"^([^/]+)/",
                group_path_root=root,
                findings=FindingLog("tracks", "multi"),
            )

        assert "'multi'" in str(caught.value)
        assert r"^([^/]+)/" in str(caught.value)
        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()

    def test_a_pattern_that_is_not_a_regular_expression_is_refused_before_a_write(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "a" / "t.csv")

        with pytest.raises(ValueError, match="not a regular expression") as caught:
            _ = dataset.index_tracks_raw(
                [root],
                patterns=["*.csv"],
                src_format="calms21_npy",
                sequence_path_pattern="(unclosed",
                group_path_root=root,
                findings=FindingLog("tracks", "broken"),
            )

        assert "'broken'" in str(caught.value)
        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()

    def test_it_is_refused_without_a_root_to_search_below(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        touch(root / "a" / "t.csv")

        with pytest.raises(ValueError, match="group_path_root") as caught:
            _ = dataset.index_tracks_raw(
                [root],
                patterns=["*.csv"],
                src_format="calms21_npy",
                sequence_path_pattern=r"^([^/]+)/",
                findings=FindingLog("tracks", "rootless"),
            )

        assert "'rootless'" in str(caught.value)
        assert r"^([^/]+)/" in str(caught.value)

        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()

    def test_it_is_refused_beside_a_format_that_merges_files_per_sequence(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path)
        root = tmp_path / "raw"
        write_trex_npz(root / "g" / "hex_7_fish0.npz")

        with pytest.raises(ValueError, match="trex_npz") as caught:
            _ = dataset.index_tracks_raw(
                [root],
                patterns=["*.npz"],
                src_format="trex_npz",
                sequence_path_pattern=r"^([^/]+)/",
                group_path_root=root,
                findings=FindingLog("tracks", "trex"),
            )

        message = str(caught.value)
        assert "'trex'" in message
        assert r"^([^/]+)/" in message
        assert "several files" in message
        assert str(root) in message
        assert not (dataset.get_root("tracks_raw") / "index.csv").exists()


class TestACountIsOptional:
    def test_a_finding_without_a_count_dumps_null_and_an_empty_sample(self) -> None:
        finding = ScanFinding(
            kind="tracks", source="s", problem="path_missing", message="gone"
        )

        dumped = finding.model_dump(mode="json")

        assert dumped["count"] is None
        assert dumped["sample"] == []
