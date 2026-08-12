"""The manifest format on its own: schema, migration, round trip, serialization.

These exercise ``mosaic.core.manifest`` with no ``Dataset`` anywhere, which is
the point of the module existing separately. The load-bearing claims are that
reading never writes, that retiring a field destroys nothing, and that roots and
sources are validated in opposite directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mosaic.core.manifest import (
    MANIFEST_VERSION,
    NOTES_MAX_CHARS,
    TAGS_MAX_COUNT,
    DatasetManifest,
    DatasetTag,
    LabelsScanSource,
    ManifestVersionError,
    MediaScanSource,
    ScanSources,
    TracksScanSource,
    backfill_roots,
    default_roots,
    legacy_tracking_roots,
    manifest_text,
    migrate_to_current,
    new_manifest,
    overlapping_sources,
    read_manifest,
    resolve_manifest_path,
    validate_root_inside,
    write_manifest,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOT, TRACKING_ROOTS

# A version-1 manifest exactly as `new_dataset_manifest` used to write one,
# including the five fields version 2 no longer models.
V1_MANIFEST = """\
# ==========================================================
# DATASET MANIFEST (extensible YAML)
# ==========================================================
name: legacy-dataset
version: 0.1.0
uuid: 6b1f3f9e-3c2c-4c1b-9a2e-71f1c0c9a0d1
created_at: '2025-01-02T03:04:05+00:00'
index_format: group/sequence
format: yaml
roots:
  media_raw: media_raw
  tracks_raw: tracks_raw
  features: features
dataset_type: continuous
segment_duration: 1H
time_column: timestamp
meta:
  fps_default: 30.0
"""

RETIRED_KEYS = (
    "format",
    "index_format",
    "dataset_type",
    "segment_duration",
    "time_column",
)


def write_v1(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    _ = path.write_text(V1_MANIFEST, encoding="utf-8")
    return path


class TestMigration:
    def test_a_version_1_file_reads_as_the_current_version(
        self, tmp_path: Path
    ) -> None:
        manifest = read_manifest(write_v1(tmp_path))
        assert manifest.manifest_version == MANIFEST_VERSION
        assert manifest.migrated_from == 1
        assert manifest.name == "legacy-dataset"
        assert manifest.meta["fps_default"] == 30.0

    def test_reading_does_not_touch_the_file(self, tmp_path: Path) -> None:
        """A read-only mount must work, and looking must not rewrite.

        The whole migration is in memory. Version 2 reaches disk on the next
        save and not before, so a dataset nobody writes to stays readable by
        whatever wrote it.
        """
        path = write_v1(tmp_path)
        before = path.read_bytes()
        _ = read_manifest(path)
        assert path.read_bytes() == before

    def test_a_manifest_already_current_reports_no_migration(self) -> None:
        manifest = migrate_to_current({"manifest_version": MANIFEST_VERSION})
        assert manifest.migrated_from is None

    def test_a_newer_manifest_raises_rather_than_being_misread(self) -> None:
        with pytest.raises(ManifestVersionError, match="declares version 99"):
            _ = migrate_to_current({"manifest_version": 99})

    def test_a_missing_version_reads_as_1(self) -> None:
        assert migrate_to_current({"name": "x"}).migrated_from == 1

    def test_migration_backfills_the_roots_a_legacy_file_lacks(
        self, tmp_path: Path
    ) -> None:
        manifest = read_manifest(write_v1(tmp_path))
        for key in (TRACKING_ROOT, *TRACKING_ROOTS, "labels_raw"):
            assert manifest.roots[key] == default_roots[key]

    def test_a_declared_legacy_root_is_never_repointed(self) -> None:
        manifest = migrate_to_current({"roots": {"trex": "tracks_raw/trex"}})
        assert manifest.roots["trex"] == "tracks_raw/trex"
        assert legacy_tracking_roots(manifest.roots) == {"trex": "tracks_raw/trex"}


class TestRetirementDestroysNothing:
    def test_every_retired_key_survives_a_round_trip(self, tmp_path: Path) -> None:
        path = write_v1(tmp_path)
        write_manifest(path, read_manifest(path))
        written = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in RETIRED_KEYS:
            assert key in written, f"{key} was destroyed by the round trip"
        assert written["dataset_type"] == "continuous"
        assert written["index_format"] == "group/sequence"

    def test_an_unknown_key_survives_and_lands_after_the_modeled_ones(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "dataset.yaml"
        _ = path.write_text("name: x\nweather: sunny\n", encoding="utf-8")
        manifest = read_manifest(path)
        assert manifest.preserved["weather"] == "sunny"
        write_manifest(path, manifest)
        keys = list(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert keys.index("weather") > keys.index("roots")

    def test_identity_survives_a_round_trip(self, tmp_path: Path) -> None:
        path = write_v1(tmp_path)
        original = read_manifest(path)
        write_manifest(path, original)
        assert read_manifest(path).uuid == original.uuid
        assert read_manifest(path).created_at == original.created_at


class TestSerialization:
    def test_a_json_manifest_round_trips_without_a_stored_format(
        self, tmp_path: Path
    ) -> None:
        """The suffix chooses the writer, so nothing can disagree with the name.

        Version 1 stored a ``format`` field beside a file whose extension said
        the same thing, and the two could drift.
        """
        path = tmp_path / "dataset.json"
        write_manifest(path, new_manifest("as-json"))
        assert json.loads(path.read_text(encoding="utf-8"))["name"] == "as-json"
        assert read_manifest(path).name == "as-json"

    def test_yaml_comments_are_lost_but_notes_are_not(self, tmp_path: Path) -> None:
        """The documented trade-off, asserted rather than only described.

        A YAML dump cannot carry comments through, so the header is regenerated
        and a hand-typed comment does not survive. ``notes`` is the field that
        does, which is why the header points at it.
        """
        path = tmp_path / "dataset.yaml"
        _ = path.write_text(
            "# a comment somebody typed\nname: x\nnotes: prose that matters\n",
            encoding="utf-8",
        )
        write_manifest(path, read_manifest(path))
        text = path.read_text(encoding="utf-8")
        assert "a comment somebody typed" not in text
        assert "prose that matters" in text

    def test_multi_line_notes_are_written_as_a_block(self, tmp_path: Path) -> None:
        manifest = new_manifest("x")
        manifest.notes = "first line\nsecond line\n"
        assert "notes: |" in manifest_text(manifest)
        path = tmp_path / "dataset.yaml"
        write_manifest(path, manifest)
        assert read_manifest(path).notes == "first line\nsecond line\n"

    def test_empty_sections_are_omitted_rather_than_written_as_placeholders(
        self,
    ) -> None:
        text = manifest_text(new_manifest("bare"))
        for absent in ("sources:", "notes:", "tags:", "meta:"):
            assert absent not in text

    def test_the_header_is_regenerated_on_every_write(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.yaml"
        write_manifest(path, new_manifest("x"))
        assert "mosaic dataset manifest (v2)" in path.read_text(encoding="utf-8")

    def test_a_written_manifest_reads_back_identical(self, tmp_path: Path) -> None:
        manifest = new_manifest("full")
        manifest.notes = "some notes"
        manifest.tags = (
            DatasetTag(name="n", type="int", type_constraints={"min": 1}, value=4),
        )
        manifest.meta = {"fps_default": 25.0}
        manifest.sources = ScanSources(
            media=(MediaScanSource(id="a", path="/nas", extensions=(".mp4",)),),
            tracks=(TracksScanSource(id="b", path="/t", patterns=("*.npz",)),),
            labels=(LabelsScanSource(id="c", path="/l", files=("x.csv",)),),
        )
        path = tmp_path / "dataset.yaml"
        write_manifest(path, manifest)
        again = read_manifest(path)
        assert again.name == manifest.name
        assert again.notes == manifest.notes
        assert again.tags == manifest.tags
        assert again.meta == manifest.meta
        assert again.sources == manifest.sources

    def test_a_file_mode_source_writes_no_walk_knobs(self, tmp_path: Path) -> None:
        """Writing them would produce a file this code refuses to read back.

        The model rejects a source declaring both a file list and a discovery
        knob, so the serializer must not emit one.
        """
        manifest = new_manifest("x")
        manifest.sources = ScanSources(
            media=(MediaScanSource(id="imp", path="/nas", files=("a.mp4",)),)
        )
        path = tmp_path / "dataset.yaml"
        write_manifest(path, manifest)
        written = yaml.safe_load(path.read_text(encoding="utf-8"))
        source = written["sources"]["media"][0]
        assert source["files"] == ["a.mp4"]
        for absent in ("recursive", "extensions", "patterns"):
            assert absent not in source
        assert read_manifest(path).sources.media[0].files == ("a.mp4",)


class TestRootsAndSourcesAreValidatedInOppositeDirections:
    def test_a_root_may_not_leave_the_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="would resolve outside the dataset"):
            _ = validate_root_inside(tmp_path, "/somewhere/else", "media_raw")

    def test_an_inside_absolute_root_is_fine(self, tmp_path: Path) -> None:
        assert validate_root_inside(tmp_path, tmp_path / "m", "media") == tmp_path / "m"

    def test_a_source_is_expected_to_point_outside(self) -> None:
        """The rule roots obey does not apply to sources, deliberately.

        A source names storage elsewhere; that is what it is for. Its files are
        recorded by absolute abs_path into an index that stays inside, which is
        the whole arrangement that replaced an outside root.
        """
        source = MediaScanSource(id="nas", path="/Volumes/behavior-nas/cage_a")
        assert source.path == "/Volumes/behavior-nas/cage_a"

    def test_the_root_error_points_at_sources(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="declare a scan source"):
            _ = validate_root_inside(tmp_path, "/elsewhere", "media_raw")

    def test_backfill_fills_absent_keys_and_repoints_none(self) -> None:
        roots = backfill_roots({"trex": "tracks_raw/trex"})
        assert roots["trex"] == "tracks_raw/trex"
        assert roots[TRACKING_ROOT] == TRACKING_ROOT

    def test_the_default_roots_cannot_be_mutated_by_a_caller(self) -> None:
        """It is the template every new dataset starts from.

        A caller mutating it would repoint every dataset created afterwards in
        the same process, which is the kind of bug that shows up a week later in
        someone else's data.
        """
        with pytest.raises(TypeError):
            default_roots["media_raw"] = "elsewhere"  # pyright: ignore[reportIndexIssue]


class TestSourceSchema:
    def test_an_unknown_key_in_a_source_raises(self) -> None:
        with pytest.raises(ValueError, match="extension"):
            _ = MediaScanSource.model_validate(
                {"id": "a", "path": "/x", "extension": [".mp4"]}
            )

    def test_an_unknown_key_at_the_top_level_does_not(self) -> None:
        assert migrate_to_current({"anything": 1}).preserved == {"anything": 1}

    @pytest.mark.parametrize("bad", ["", "-leading", "has space", "x" * 65, "a/b"])
    def test_a_source_id_is_a_token(self, bad: str) -> None:
        with pytest.raises(ValueError, match="source id"):
            _ = MediaScanSource(id=bad, path="/x")

    def test_a_file_list_and_a_walk_knob_cannot_both_be_declared(self) -> None:
        with pytest.raises(ValueError, match="do not apply"):
            _ = MediaScanSource(
                id="a", path="/x", files=("one.mp4",), extensions=(".mp4",)
            )

    def test_an_explicit_recursive_is_refused_beside_a_file_list(self) -> None:
        """Even at its default value, because the manifest would be lying.

        A recipe that says it recurses and then does not is worse than one that
        does not mention recursion at all.
        """
        with pytest.raises(ValueError, match="do not apply"):
            _ = MediaScanSource(id="a", path="/x", files=("one.mp4",), recursive=True)

    @pytest.mark.parametrize("bad", ["/absolute.mp4", "../escape.mp4", "  "])
    def test_a_listed_file_stays_under_the_source_path(self, bad: str) -> None:
        with pytest.raises(ValueError, match="source file"):
            _ = MediaScanSource(id="a", path="/x", files=(bad,))

    def test_a_file_listed_twice_raises(self) -> None:
        with pytest.raises(ValueError, match="listed twice"):
            _ = MediaScanSource(id="a", path="/x", files=("a.mp4", "a.mp4"))

    def test_extensions_are_normalized_to_lowercase_with_a_dot(self) -> None:
        source = MediaScanSource(id="a", path="/x", extensions=("MP4", ".AVI"))
        assert source.extensions == (".mp4", ".avi")

    def test_the_two_grouping_rules_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="exactly one can take effect"):
            _ = TracksScanSource(
                id="a",
                path="/x",
                multi_sequences_per_file=True,
                group_from="filename",
                group_pattern="^x",
            )

    def test_group_from_without_multi_sequences_raises(self) -> None:
        with pytest.raises(ValueError, match="use group_pattern instead"):
            _ = TracksScanSource(id="a", path="/x", group_from="filename")

    def test_two_sources_of_one_kind_may_not_share_an_id(self) -> None:
        with pytest.raises(ValueError, match="share the id"):
            _ = ScanSources(
                media=(
                    MediaScanSource(id="dup", path="/a"),
                    MediaScanSource(id="dup", path="/b"),
                )
            )

    def test_the_same_id_in_two_kinds_is_fine(self) -> None:
        sources = ScanSources(
            media=(MediaScanSource(id="nas", path="/a"),),
            tracks=(TracksScanSource(id="nas", path="/b"),),
        )
        assert sources.media[0].id == sources.tracks[0].id

    def test_mode_is_derived_from_whether_files_are_listed(self) -> None:
        assert MediaScanSource(id="a", path="/x").mode == "directory"
        assert MediaScanSource(id="a", path="/x", files=("y.mp4",)).mode == "files"


class TestSelect:
    def make(self) -> ScanSources:
        return ScanSources(
            media=(
                MediaScanSource(id="one", path="/a"),
                MediaScanSource(id="two", path="/b"),
            )
        )

    def test_no_restriction_returns_everything_declared(self) -> None:
        assert len(self.make().select("media")) == 2

    def test_only_restricts_without_reordering(self) -> None:
        selected = self.make().select("media", ["two"])
        assert [s.id for s in selected] == ["two"]

    def test_an_unknown_id_raises_and_lists_the_real_ones(self) -> None:
        with pytest.raises(KeyError, match="one"):
            _ = self.make().select("media", ["three"])

    def test_a_kind_with_nothing_declared_selects_nothing(self) -> None:
        assert self.make().select("labels") == ()


class TestOverlap:
    def resolve(self, path: str) -> Path:
        return Path(path)

    def test_two_directory_sources_may_not_nest(self) -> None:
        pair = overlapping_sources(
            [
                MediaScanSource(id="outer", path="/nas"),
                MediaScanSource(id="inner", path="/nas/clips"),
            ],
            self.resolve,
        )
        assert pair == ("outer", "inner")

    def test_disjoint_directory_sources_are_fine(self) -> None:
        assert (
            overlapping_sources(
                [
                    MediaScanSource(id="a", path="/nas/one"),
                    MediaScanSource(id="b", path="/nas/two"),
                ],
                self.resolve,
            )
            is None
        )

    def test_two_import_batches_under_one_directory_coexist(self) -> None:
        """The case selective import actually produces.

        Importing some of a folder's files, then some more, must not require a
        new directory or a merged declaration. Their claimed sets are exact and
        disjoint, so nothing is ambiguous.
        """
        assert (
            overlapping_sources(
                [
                    MediaScanSource(id="batch1", path="/nas", files=("a.mp4", "b.mp4")),
                    MediaScanSource(id="batch2", path="/nas", files=("c.mp4",)),
                ],
                self.resolve,
            )
            is None
        )

    def test_two_import_batches_claiming_one_file_are_refused(self) -> None:
        pair = overlapping_sources(
            [
                MediaScanSource(id="batch1", path="/nas", files=("a.mp4",)),
                MediaScanSource(id="batch2", path="/nas", files=("a.mp4",)),
            ],
            self.resolve,
        )
        assert pair == ("batch1", "batch2")

    def test_a_file_source_inside_a_directory_source_is_refused(self) -> None:
        pair = overlapping_sources(
            [
                MediaScanSource(id="walked", path="/nas"),
                MediaScanSource(id="picked", path="/nas/sub", files=("a.mp4",)),
            ],
            self.resolve,
        )
        assert pair == ("walked", "picked")


class TestTags:
    def test_a_tag_round_trips_with_everything_it_declares(
        self, tmp_path: Path
    ) -> None:
        tag = DatasetTag(
            name="cohort",
            type="categorical",
            type_constraints={"options": ["spring", "fall"]},
            value="spring",
            description="Which cohort",
            display_order=3,
        )
        manifest = new_manifest("x")
        manifest.tags = (tag,)
        path = tmp_path / "dataset.yaml"
        write_manifest(path, manifest)
        assert read_manifest(path).tags == (tag,)

    def test_a_label_tag_writes_no_value(self, tmp_path: Path) -> None:
        manifest = new_manifest("x")
        manifest.tags = (DatasetTag(name="pilot", type="label"),)
        path = tmp_path / "dataset.yaml"
        write_manifest(path, manifest)
        written = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "value" not in written["tags"][0]

    @pytest.mark.parametrize(
        ("type_", "value"), [("bool", False), ("int", 0), ("text", "")]
    )
    def test_a_falsy_value_is_still_written(
        self, tmp_path: Path, type_: str, value: object
    ) -> None:
        """A False, a 0 and an empty string are values, not absences.

        Testing the type rather than the truthiness of the value is what keeps
        them from being written as though nothing was set.
        """
        manifest = new_manifest("x")
        manifest.tags = (
            DatasetTag.model_validate({"name": "t", "type": type_, "value": value}),
        )
        path = tmp_path / "dataset.yaml"
        write_manifest(path, manifest)
        assert read_manifest(path).tags[0].value == value

    def test_tags_are_written_by_display_order_then_name(self) -> None:
        manifest = new_manifest("x")
        manifest.tags = (
            DatasetTag(name="zulu", type="label", display_order=0),
            DatasetTag(name="alpha", type="label", display_order=1),
            DatasetTag(name="bravo", type="label", display_order=0),
        )
        assert [t.name for t in manifest.ordered_tags()] == ["bravo", "zulu", "alpha"]

    def test_a_value_outside_its_constraints_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above max"):
            _ = DatasetTag(
                name="n", type="int", type_constraints={"min": 1, "max": 5}, value=9
            )

    def test_the_error_names_the_tag(self) -> None:
        with pytest.raises(ValueError, match="tag 'n'"):
            _ = DatasetTag(name="n", type="label", value="something")

    def test_two_tags_differing_only_by_case_are_refused(self) -> None:
        with pytest.raises(ValueError, match="differ only by case"):
            _ = DatasetManifest(
                tags=(
                    DatasetTag(name="Cohort", type="label"),
                    DatasetTag(name="cohort", type="label"),
                )
            )

    def test_lookup_is_case_insensitive(self) -> None:
        manifest = DatasetManifest(tags=(DatasetTag(name="Cohort", type="label"),))
        assert manifest.tag("COHORT") is not None
        assert manifest.tag("missing") is None

    def test_the_tag_count_is_capped(self) -> None:
        many = tuple(
            DatasetTag(name=f"t{i}", type="label") for i in range(TAGS_MAX_COUNT + 1)
        )
        with pytest.raises(ValueError, match="over the 200 limit"):
            _ = DatasetManifest(tags=many)


class TestNotes:
    def test_notes_over_the_limit_are_refused_on_construction(self) -> None:
        with pytest.raises(ValueError, match="over the"):
            _ = DatasetManifest(notes="x" * (NOTES_MAX_CHARS + 1))

    def test_notes_at_the_limit_are_accepted(self) -> None:
        assert (
            len(DatasetManifest(notes="x" * NOTES_MAX_CHARS).notes) == NOTES_MAX_CHARS
        )


class TestResolveManifestPath:
    def test_a_directory_is_probed_for_the_known_names(self, tmp_path: Path) -> None:
        target = tmp_path / "dataset.yml"
        _ = target.write_text("name: x\n", encoding="utf-8")
        assert resolve_manifest_path(tmp_path) == target

    def test_yaml_wins_over_yml_and_json(self, tmp_path: Path) -> None:
        for name in ("dataset.json", "dataset.yml", "dataset.yaml"):
            _ = (tmp_path / name).write_text("name: x\n", encoding="utf-8")
        assert resolve_manifest_path(tmp_path).name == "dataset.yaml"

    def test_an_empty_directory_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No manifest found"):
            _ = resolve_manifest_path(tmp_path)

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _ = resolve_manifest_path(tmp_path / "nope.yaml")

    def test_an_empty_manifest_file_reads_as_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.yaml"
        _ = path.write_text("", encoding="utf-8")
        assert read_manifest(path).name == "unnamed"

    def test_a_manifest_that_is_not_a_mapping_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.yaml"
        _ = path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a mapping"):
            _ = read_manifest(path)


class TestContinuousGroups:
    """The declaration that a group's sequences divide one recording."""

    def test_absent_means_every_group_is_independent(self, tmp_path: Path) -> None:
        """The default has to be the discrete dataset, which is every dataset."""
        path = tmp_path / "dataset.yaml"
        _ = path.write_text("manifest_version: 2\nname: d\n", encoding="utf-8")
        manifest = read_manifest(path)
        assert manifest.continuous_groups == ()
        assert not manifest.is_continuous_group("anything")

    def test_it_survives_a_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.yaml"
        _ = path.write_text("manifest_version: 2\nname: d\n", encoding="utf-8")
        manifest = read_manifest(path).model_copy(
            update={"continuous_groups": ("trialA", "trialB")}
        )
        write_manifest(path, manifest)
        back = read_manifest(path)
        assert back.continuous_groups == ("trialA", "trialB")
        assert back.is_continuous_group("trialA")
        assert not back.is_continuous_group("trialC")

    def test_adding_it_did_not_move_the_manifest_version(self) -> None:
        """Additive and optional, so an older file reads unchanged.

        A version bump would make every existing manifest a migration, for a key
        none of them carry.
        """
        assert DatasetManifest().manifest_version == MANIFEST_VERSION
        assert DatasetManifest().continuous_groups == ()

    def test_the_empty_group_cannot_be_continuous(self) -> None:
        """A continuous group *is* the recording, so it has to be named."""
        with pytest.raises(ValidationError, match="empty group"):
            _ = DatasetManifest(continuous_groups=("",))

    def test_a_group_is_not_declared_twice(self) -> None:
        with pytest.raises(ValidationError, match="more than once"):
            _ = DatasetManifest(continuous_groups=("g", "g"))
