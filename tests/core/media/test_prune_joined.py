"""Tests for reclaiming joined exports an earlier op version made.

Stub files throughout: the pruner reads a join's name, size and mtime, never its
frames, so a real concatenation would only make these slow. The media index is
real, because what decides whether a join's clips still belong to an entry is
``Dataset.resolve_media`` over that index.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mosaic_media import MediaProbeError
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset, ResolvedMedia
from mosaic.core.media.prune_joined import JoinedPruneClass, deletable
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.joined_export import (
    JoinedExportParams,
    joined_recipe_hash,
    joined_source_uid,
)
from tests.helpers import MediaClip, make_dataset, write_media_index

UUID_A = "11111111-1111-4111-8111-111111111111"
UUID_B = "22222222-2222-4222-8222-222222222222"
UUID_C = "33333333-3333-4333-8333-333333333333"

CURRENT = joined_recipe_hash(JoinedExportParams())
CURRENT_REENCODE = joined_recipe_hash(JoinedExportParams(reencode=True))
EARLIER = hash_params(
    {"op_version": "0.1", "params": JoinedExportParams(reencode=True).identity_dump()}
)

runner = CliRunner()


def _two_clips(sequence: str = "sess") -> list[MediaClip]:
    return [
        MediaClip(
            sequence=sequence, filename="a.mp4", video_order=0, video_uuid=UUID_A
        ),
        MediaClip(
            sequence=sequence, filename="b.mp4", video_order=1, video_uuid=UUID_B
        ),
    ]


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """One two-clip entry, which a join can hold, and one single-clip entry."""
    dataset = make_dataset(
        (tmp_path / "dataset").resolve(), roots=["media_raw", "media", "tracks"]
    )
    write_media_index(
        dataset,
        [
            *_two_clips(),
            MediaClip(sequence="solo", filename="c.mp4", video_uuid=UUID_C),
        ],
    )
    return dataset


def _uid(ds: Dataset, sequence: str = "sess") -> str:
    return joined_source_uid(list(ds.resolve_media("", sequence).facts))


def _put(ds: Dataset, name: str, content: bytes = b"join") -> Path:
    """Write a stub file directly under the joined kind directory."""
    root = ds.get_root("media") / "joined"
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    _ = path.write_bytes(content)
    return path


def _join(ds: Dataset, recipe: str, content: bytes = b"join") -> Path:
    return _put(ds, f"{_uid(ds)}.{recipe}.joined.mp4", content)


def _verdicts(ds: Dataset) -> dict[str, JoinedPruneClass]:
    """Each path's class, from a dry run with the age window open."""
    report = ds.prune_joined_exports(apply=False, min_age_hours=0.0)
    return {entry.path.name: entry.verdict for entry in report.entries}


# --- the motivating case ------------------------------------------------------


def test_a_join_an_earlier_version_made_is_removed_by_one_apply(ds: Dataset) -> None:
    """The stranded 0.1 join beside the current one, as the upgrade left it."""
    live = _join(ds, CURRENT_REENCODE)
    stale = _join(ds, EARLIER, b"older join")

    dry = ds.prune_joined_exports(apply=False, min_age_hours=0.0)
    assert {e.path.name: e.verdict for e in dry.entries} == {
        live.name: "live",
        stale.name: "superseded",
    }
    assert dry.files_deleted == [stale]
    assert stale.exists(), "a dry run deleted something"

    applied = ds.prune_joined_exports(apply=True, min_age_hours=0.0)
    assert applied.applied
    assert applied.bytes_reclaimed == len(b"older join")
    assert not stale.exists()
    assert live.exists()

    again = ds.prune_joined_exports(apply=True, min_age_hours=0.0)
    assert not again.applied
    assert again.files_deleted == []


def test_a_superseded_join_with_no_current_one_is_still_superseded(
    ds: Dataset,
) -> None:
    """No tracker reads it either way, so deleting it costs nothing read."""
    stale = _join(ds, EARLIER)
    assert _verdicts(ds) == {stale.name: "superseded"}


def test_deletion_authority_is_the_superseded_class_alone() -> None:
    kept: tuple[JoinedPruneClass, ...] = ("live", "competing", "unsourced", "stray")
    assert deletable("superseded")
    assert not any(deletable(verdict) for verdict in kept)


# --- what is kept -------------------------------------------------------------


def test_two_current_joins_are_competing_and_both_kept(ds: Dataset) -> None:
    """Both are valid, so which one a tracker reads is a person's decision."""
    default = _join(ds, CURRENT)
    reencoded = _join(ds, CURRENT_REENCODE)

    report = ds.prune_joined_exports(apply=True, min_age_hours=0.0, include_stray=True)

    assert {e.path.name: e.verdict for e in report.entries} == {
        default.name: "competing",
        reencoded.name: "competing",
    }
    assert default.exists() and reencoded.exists()
    assert {e.entry for e in report.of("competing")} == {"/sess"}


def test_a_join_of_clips_no_entry_resolves_to_survives_every_flag(
    ds: Dataset,
) -> None:
    """It may be the last copy of a session whose clips are gone."""
    orphan = _put(ds, f"{'f' * 64}.{EARLIER}.joined.mp4")

    report = ds.prune_joined_exports(apply=True, min_age_hours=0.0, include_stray=True)

    assert [e.verdict for e in report.entries] == ["unsourced"]
    assert orphan.exists()


def test_an_entry_that_cannot_be_resolved_leaves_its_joins_kept(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolution failure can prevent a deletion and never cause one."""
    stale = _join(ds, EARLIER)
    resolve = ds.resolve_media

    def failing(group: str, sequence: str) -> ResolvedMedia:
        if sequence == "sess":
            raise MediaProbeError("a clip requires a transcode it does not have")
        return resolve(group, sequence)

    monkeypatch.setattr(ds, "resolve_media", failing)
    report = ds.prune_joined_exports(apply=True, min_age_hours=0.0)

    assert [e.verdict for e in report.entries] == ["unsourced"]
    assert stale.exists()
    assert len(report.unresolved) == 1
    assert report.unresolved[0].startswith("/sess: ")


def test_a_multi_camera_entry_is_skipped_rather_than_unresolved(
    tmp_path: Path,
) -> None:
    """The op cannot join one, so its failure to resolve is not news."""
    dataset = make_dataset(
        (tmp_path / "dataset").resolve(), roots=["media_raw", "media", "tracks"]
    )
    write_media_index(
        dataset,
        [
            MediaClip(sequence="rig", filename="l.mp4", camera="left"),
            MediaClip(sequence="rig", filename="r.mp4", camera="right"),
        ],
    )

    report = dataset.prune_joined_exports(apply=False)

    assert report.considered
    assert report.unresolved == []


# --- strays -------------------------------------------------------------------


def test_the_ops_working_files_are_strays_deleted_only_on_request(
    ds: Dataset,
) -> None:
    """A kept partial, a concat listing and a normalised clip are not joins."""
    stem = f"{_uid(ds)}.{CURRENT}.joined"
    strays = [
        _put(ds, f"{stem}.partial.mp4"),
        _put(ds, f"{stem}.concat.txt"),
        _put(ds, f"{stem}.normalised3.mp4"),
    ]

    plain = ds.prune_joined_exports(apply=True, min_age_hours=0.0)
    assert {e.verdict for e in plain.entries} == {"stray"}
    assert all(path.exists() for path in strays)

    swept = ds.prune_joined_exports(apply=True, min_age_hours=0.0, include_stray=True)
    assert sorted(swept.files_deleted) == sorted(strays)
    assert not any(path.exists() for path in strays)


def test_a_subdirectory_and_a_symlink_survive_every_flag(ds: Dataset) -> None:
    live = _join(ds, CURRENT)
    subdirectory = ds.get_root("media") / "joined" / "inspect"
    subdirectory.mkdir()
    link = ds.get_root("media") / "joined" / f"{_uid(ds)}.{EARLIER}.joined.mp4"
    link.symlink_to(live)

    report = ds.prune_joined_exports(apply=True, min_age_hours=0.0, include_stray=True)

    assert _verdicts(ds)[link.name] == "stray"
    assert _verdicts(ds)[subdirectory.name] == "stray"
    assert link.is_symlink() and subdirectory.is_dir() and live.exists()
    assert report.files_deleted == []


# --- the age window -----------------------------------------------------------


def test_a_file_younger_than_the_window_is_held_back(ds: Dataset) -> None:
    stale = _join(ds, EARLIER)

    held = ds.prune_joined_exports(apply=True)
    assert held.held_for_age == 1
    assert held.files_deleted == []
    assert stale.exists()

    later = datetime.now(timezone.utc) + timedelta(hours=25)
    released = ds.prune_joined_exports(apply=True, now=later)
    assert released.files_deleted == [stale]
    assert not stale.exists()


def test_an_old_stray_is_swept_and_a_young_one_is_not(ds: Dataset) -> None:
    """An in-flight partial is exactly the shape the stray sweep removes."""
    young = _put(ds, f"{_uid(ds)}.{CURRENT}.joined.partial.mp4")
    old = _put(ds, f"{_uid(ds)}.{CURRENT_REENCODE}.joined.partial.mp4")
    past = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
    os.utime(old, (past, past))

    report = ds.prune_joined_exports(apply=True, include_stray=True)

    assert report.files_deleted == [old]
    assert young.exists()


# --- gates --------------------------------------------------------------------


def test_a_dataset_with_no_joined_directory_is_considered_and_empty(
    ds: Dataset,
) -> None:
    report = ds.prune_joined_exports(apply=True)
    assert report.considered
    assert report.entries == []


def test_a_single_root_dataset_is_pruned_rather_than_declined(tmp_path: Path) -> None:
    """Unlike prune-media: here `media` holds the originals and can hold joins."""
    dataset = make_dataset((tmp_path / "dataset").resolve(), roots=["media", "tracks"])
    write_media_index(dataset, _two_clips())
    stale = _join(dataset, EARLIER)

    report = dataset.prune_joined_exports(apply=True, min_age_hours=0.0)

    assert report.considered
    assert report.files_deleted == [stale]


def test_a_dataset_with_no_media_root_declines(tmp_path: Path) -> None:
    dataset = make_dataset((tmp_path / "dataset").resolve(), roots=["tracks"])
    report = dataset.prune_joined_exports(apply=True)
    assert not report.considered
    assert report.declined == "no-media-root"


def test_a_root_nested_in_the_joined_directory_declines(tmp_path: Path) -> None:
    base = (tmp_path / "dataset").resolve()
    dataset = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={
            "media": str(base / "media"),
            "media_raw": str(base / "media/joined/raw"),
        },
    )
    dataset.ensure_roots()
    dataset.save()

    report = dataset.prune_joined_exports(apply=True)

    assert report.declined == "nested-root"


# --- the command --------------------------------------------------------------


def test_the_command_is_a_dry_run_by_default(ds: Dataset) -> None:
    stale = _join(ds, EARLIER)

    result = runner.invoke(
        app, ["prune-joined", "-m", str(ds.manifest_path), "--min-age-hours", "0"]
    )

    assert result.exit_code == 0, result.stdout
    assert "would delete 1 file(s)" in result.stdout
    assert stale.exists()
    # The recipes come from the installed mosaic, which can differ from the
    # worker's, so the human report prints them.
    assert CURRENT in result.stdout and CURRENT_REENCODE in result.stdout


def test_the_json_document_is_one_value_on_stdout(ds: Dataset) -> None:
    _ = _join(ds, EARLIER)
    _ = _put(ds, "notes.txt")

    result = runner.invoke(
        app,
        ["prune-joined", "-m", str(ds.manifest_path), "--min-age-hours", "0", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["considered"] is True
    assert payload["applied"] is False
    assert payload["counts"] == {"superseded": 1, "stray": 1}
    assert payload["files_deleted_count"] == 1
    assert sorted(payload["current_recipes"]) == sorted([CURRENT, CURRENT_REENCODE])


def test_a_decline_exits_zero_and_says_why(tmp_path: Path) -> None:
    dataset = make_dataset((tmp_path / "dataset").resolve(), roots=["tracks"])

    result = runner.invoke(app, ["prune-joined", "-m", str(dataset.manifest_path)])

    assert result.exit_code == 0
    assert "declined" in result.stdout
    assert "would delete" not in result.stdout
