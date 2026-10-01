"""Which clips mosaic's own reader must read through their join, and finding it.

``MultiVideoReader`` refuses clips whose frame rates disagree and stays strict. A
consumer reading such a recording reads its ``export-joined`` file instead, looked
up by the same rule a tracker uses. These tests place a file at the join's
address rather than encoding one: the lookup reads names, never pixels.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from typing import Final

import pytest
from mosaic_media import MediaFacts

from mosaic.core.dataset import Dataset, ResolvedMedia, ResolvedScopeEntry
from mosaic.core.pipeline.joined_export import (
    JoinedExportMissingError,
    JoinedExportParams,
    current_join,
    join_to_read,
    joined_export_path,
    joined_recipe_hash,
    joined_source_uid,
    missing_joins,
    needs_join,
)
from tests.helpers import clip_facts, make_dataset

# Long enough that 30 beside 31 fps drifts past the half-frame allowance
# `rate_uniform` grants; ten frames each would read as uniform.
CLIP = 300


def _clip(fps: float, uid: str) -> MediaFacts:
    return clip_facts(fps=fps, frame_count=CLIP, video_uuid=uid)


MIXED = (_clip(30.0, "u-a"), _clip(31.0, "u-b"))
UNIFORM = (_clip(30.0, "u-a"), _clip(30.0, "u-b"))
MIXED_PROFILES = (
    _clip(30.0, "u-a"),
    dataclasses.replace(_clip(31.0, "u-b"), codec_name="av1"),
)
"""Clips at two rates and in two codecs, which a join copies only by re-encoding."""

_REENCODE: Final = """--params '{"reencode": true}'"""


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path, roots=["media_raw", "media"])


def _entry(tmp_path: Path, facts: tuple[MediaFacts, ...]) -> ResolvedScopeEntry:
    paths = [tmp_path / f"clip{i}.mp4" for i in range(len(facts))]
    for path in paths:
        path.touch()
    return ResolvedScopeEntry(
        group="",
        sequence="sess",
        camera="",
        resolved=ResolvedMedia(paths=paths, facts=list(facts)),
    )


def _place_join(ds: Dataset, facts: tuple[MediaFacts, ...], recipe: str) -> Path:
    path = joined_export_path(ds, joined_source_uid(list(facts)), recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


class TestNeedsJoin:
    def test_clips_at_different_rates_need_one(self, tmp_path: Path) -> None:
        entry = _entry(tmp_path, MIXED)
        assert needs_join(entry.resolved.paths, entry.resolved.facts)

    def test_clips_at_one_rate_do_not(self, tmp_path: Path) -> None:
        entry = _entry(tmp_path, UNIFORM)
        assert not needs_join(entry.resolved.paths, entry.resolved.facts)

    def test_a_single_clip_does_not(self, tmp_path: Path) -> None:
        entry = _entry(tmp_path, MIXED[:1])
        assert not needs_join(entry.resolved.paths, entry.resolved.facts)

    def test_a_store_sequence_does_not(self, tmp_path: Path) -> None:
        """A store is read natively, and its reader compares no rate."""
        stores: list[Path] = []
        for name in ("s0", "s1"):
            store = tmp_path / name
            store.mkdir()
            _ = (store / "metadata.yaml").write_text(
                "__store:\n  class: VideoImgStore\n"
            )
            stores.append(store)
        assert not needs_join(stores, list(MIXED))


class TestJoinToRead:
    def test_a_uniform_entry_reads_its_clips(self, ds: Dataset, tmp_path: Path) -> None:
        assert (
            join_to_read(ds, _entry(tmp_path, UNIFORM), asker="extract-frames") is None
        )

    def test_a_mixed_rate_entry_reads_its_current_join(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        joined = _place_join(ds, MIXED, joined_recipe_hash(JoinedExportParams()))
        entry = _entry(tmp_path, MIXED)
        assert join_to_read(ds, entry, asker="extract-frames") == joined

    def test_a_reencode_join_answers_as_well(self, ds: Dataset, tmp_path: Path) -> None:
        recipe = joined_recipe_hash(JoinedExportParams(reencode=True))
        joined = _place_join(ds, MIXED, recipe)
        assert (
            join_to_read(ds, _entry(tmp_path, MIXED), asker="extract-frames") == joined
        )

    def test_a_missing_join_is_refused_naming_the_command_and_the_reason(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(JoinedExportMissingError) as excinfo:
            _ = join_to_read(ds, _entry(tmp_path, MIXED), asker="extract-frames")
        message = str(excinfo.value)
        assert message.startswith("[extract-frames] (, sess)")
        assert "differ in frame rate" in message
        assert '--kind export-joined --entries ":sess"' in message
        assert _REENCODE not in message

    def test_a_missing_join_of_clips_in_two_profiles_is_built_by_reencoding(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        """The command the refusal names is one that joins these clips."""
        entry = _entry(tmp_path, MIXED_PROFILES)
        with pytest.raises(JoinedExportMissingError) as excinfo:
            _ = join_to_read(ds, entry, asker="extract-frames")
        assert f'--entries ":sess" {_REENCODE}' in str(excinfo.value)


class TestMissingJoins:
    def test_every_unjoined_camera_is_reported_and_nothing_is_raised(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()
        mixed = _entry(mixed_dir, MIXED)
        uniform = _entry(tmp_path, UNIFORM)

        (missing,) = missing_joins(ds, [mixed, uniform], asker="extract-frames")

        assert (missing.group, missing.sequence, missing.camera) == ("", "sess", "")
        assert "export-joined" in missing.reason

    def test_nothing_is_reported_once_the_join_exists(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = _place_join(ds, MIXED, joined_recipe_hash(JoinedExportParams()))
        assert (
            missing_joins(ds, [_entry(tmp_path, MIXED)], asker="extract-frames") == []
        )


def test_a_clip_set_without_identity_cannot_be_addressed(ds: Dataset) -> None:
    clips = (_clip(30.0, ""), _clip(31.0, "u-b"))
    with pytest.raises(JoinedExportMissingError, match="content identity"):
        _ = current_join(ds, "", "sess", clips, asker="extract-frames", why="unused.")
