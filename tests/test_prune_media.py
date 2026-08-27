"""Tests for the transcode-derivative pruner.

Fixture-driven throughout, with stub files rather than encodes: the pruner never
reads a derivative's content, so a real ffmpeg run would only make these slow and
skippable. What it *does* read is the filename, so every stub is named under the
scheme the transcode op writes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic_media.transcode import Target

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
from mosaic.core.media.prune import PruneClass, parse_derivative_name
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)

UUID_A = "11111111-1111-4111-8111-111111111111"
UUID_B = "22222222-2222-4222-8222-222222222222"


def _live_recipe(ds: Dataset, target: Target = "analysis") -> str:
    """The recipe a run of *target* would name its output after, right now.

    Read through the same call the pruner makes rather than hard-coded: the
    recipe folds the verdict thresholds, which are environment-driven, so a
    literal here would pin this suite to one machine's configuration. The two
    targets hash differently -- both the target and the encoding are terms -- so
    a file named for one target under the other's recipe is genuinely superseded.
    """
    from mosaic_media import CHROME_149
    from mosaic_media.transcode import ANALYSIS_ENCODING, PLAYBACK_ENCODING

    from mosaic.core.pipeline.transcode import TranscodeParams, transcode_recipe_hash
    from mosaic.media_probe_config import media_thresholds

    encoding = ANALYSIS_ENCODING if target == "analysis" else PLAYBACK_ENCODING
    return transcode_recipe_hash(
        TranscodeParams(entries=[("", "")], target=target),
        encoding,
        CHROME_149,
        media_thresholds(),
    )


def _row(**cells: object) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(cells)
    return row


def _write(index_path: Path, rows: list[dict[str, object]]) -> None:
    write_media_index_rows(index_path, frame_from_rows(rows))


def _verdicts(ds: Dataset) -> dict[str, PruneClass]:
    """Map each reconciled path's basename to the class the pruner gave it.

    Always a dry run with the age window open: the classification is what is
    under test, and the flags only decide which classes get acted on.
    """
    report = ds.prune_media(apply=False, min_age_hours=0.0)
    return {entry.path.name: entry.verdict for entry in report.entries}


@pytest.fixture
def pruned_dataset(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> Dataset:
    """Two originals, one linked derivative each under the current recipe.

    The starting point every case below perturbs: nothing here is prunable, so a
    run over it must report no work at all. Two originals rather than one because
    liveness is a union over rows -- a bug that reads one row's cell as the whole
    answer stays invisible with a single original.
    """
    base = (tmp_path / "dataset").resolve()
    dataset = make_media_dataset(base)
    recipe = _live_recipe(dataset)
    transcode_root = base / "media" / "transcode"
    transcode_root.mkdir(parents=True, exist_ok=True)

    originals: list[dict[str, object]] = []
    derivatives: list[dict[str, object]] = []
    for uuid, name in ((UUID_A, "a.mp4"), (UUID_B, "b.mp4")):
        original = base / "media_raw" / name
        original.touch()
        derivative = transcode_root / f"{uuid}.{recipe}.analysis.mp4"
        derivative.write_bytes(b"stub")
        originals.append(
            _row(
                name=name,
                group="g",
                sequence="entry",
                abs_path=dataset.relative_to_root(str(original)),
                video_uuid=uuid,
                analysis_derivative_path=f"transcode/{derivative.name}",
            )
        )
        derivatives.append(
            _row(
                name=derivative.name,
                group="g",
                sequence="entry",
                abs_path=dataset.relative_to_root(str(derivative)),
                source_video_uuid=uuid,
                recipe_hash=recipe,
            )
        )
    _write(dataset.get_root("media_raw") / "index.csv", originals)
    _write(dataset.get_root("media") / "index.csv", derivatives)
    return dataset


def _add_derivative(
    ds: Dataset,
    *,
    uuid: str,
    recipe: str,
    target: Target = "analysis",
    row: bool = True,
) -> Path:
    """Put an unreferenced derivative on disk, optionally with its index row."""
    transcode_root = ds.get_root("media") / "transcode"
    transcode_root.mkdir(parents=True, exist_ok=True)
    path = transcode_root / f"{uuid}.{recipe}.{target}.mp4"
    path.write_bytes(b"stub-old")
    if row:
        index_path = ds.get_root("media") / "index.csv"
        rows = [_row(**dict(existing)) for existing in read_media_index(index_path)]
        rows.append(
            _row(
                name=path.name,
                group="g",
                sequence="entry",
                abs_path=ds.relative_to_root(str(path)),
                source_video_uuid=uuid,
                recipe_hash=recipe,
            )
        )
        _write(index_path, rows)
    return path


# --- the motivating case ------------------------------------------------------


def test_a_retuned_recipe_strands_the_old_derivative_and_one_apply_removes_it(
    pruned_dataset: Dataset,
) -> None:
    """The state the pruner exists for, end to end.

    Retuning a recipe writes a new derivative and overwrites the link cell.
    ``_set_back_link`` drops only the row matching the *new* path, so the old
    file and its row both survive with nothing addressing them. The still-linked
    derivative beside it is what proves the run removed the right one.
    """
    stale = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01")
    live = (
        pruned_dataset.get_root("media")
        / "transcode"
        / f"{UUID_A}.{_live_recipe(pruned_dataset)}.analysis.mp4"
    )
    assert _verdicts(pruned_dataset)[stale.name] == "superseded"

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0)

    assert report.applied and report.changed
    assert not stale.exists(), "the stranded derivative survived"
    assert live.exists(), "the linked derivative was deleted"
    assert report.rows_dropped == 1
    assert report.bytes_reclaimed == len(b"stub-old")
    rows = read_media_index(pruned_dataset.get_root("media") / "index.csv")
    assert {row["recipe_hash"] for row in rows} == {_live_recipe(pruned_dataset)}


def test_a_clean_dataset_is_no_work_and_no_write(pruned_dataset: Dataset) -> None:
    """No work means no write, so a second run leaves both indexes untouched."""
    paths = [
        pruned_dataset.get_root("media_raw") / "index.csv",
        pruned_dataset.get_root("media") / "index.csv",
    ]
    before = [path.read_bytes() for path in paths]

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0, relink=True)

    assert not report.changed
    assert not report.applied, "applied must mean something was done"
    assert not report.backups, "an unchanged index was backed up anyway"
    assert [path.read_bytes() for path in paths] == before


def test_a_dry_run_writes_nothing_but_names_the_file(pruned_dataset: Dataset) -> None:
    stale = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01")
    paths = [
        pruned_dataset.get_root("media_raw") / "index.csv",
        pruned_dataset.get_root("media") / "index.csv",
    ]
    before = [path.read_bytes() for path in paths]

    report = pruned_dataset.prune_media(apply=False, min_age_hours=0.0)

    assert report.changed and not report.applied
    assert stale.exists()
    assert [path.read_bytes() for path in paths] == before
    assert stale in report.files_deleted, "a dry run must still name what it would do"


# --- the classes that are never deleted ---------------------------------------


def test_an_unreferenced_derivative_under_a_live_recipe_is_kept(
    pruned_dataset: Dataset,
) -> None:
    """The interrupted-registration state: file written, link never reached.

    Deleting it would be *safe* -- the next run re-encodes over it either way,
    because the reuse gate needs the link as well as the file -- but pointless.
    ``--relink`` is the repair that makes it worth keeping.
    """
    orphan = _add_derivative(
        pruned_dataset,
        uuid=UUID_A,
        recipe=_live_recipe(pruned_dataset, "playback"),
        target="playback",
    )
    assert _verdicts(pruned_dataset)[orphan.name] == "relinkable"

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0)

    assert orphan.exists()
    assert not report.files_deleted


def test_relink_adopts_it_and_a_second_run_calls_it_live(
    pruned_dataset: Dataset,
) -> None:
    orphan = _add_derivative(
        pruned_dataset,
        uuid=UUID_A,
        recipe=_live_recipe(pruned_dataset, "playback"),
        target="playback",
    )

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0, relink=True)

    assert report.links_relinked == [f"transcode/{orphan.name}"]
    linked = {
        row["video_uuid"]: row["playback_derivative_path"]
        for row in read_media_index(pruned_dataset.get_root("media_raw") / "index.csv")
    }
    assert linked[UUID_A] == f"transcode/{orphan.name}"
    assert linked[UUID_B] == "", "relink wrote a cell on the wrong row"
    assert _verdicts(pruned_dataset)[orphan.name] == "live"


def test_relink_never_displaces_a_link_that_already_names_a_file(
    pruned_dataset: Dataset,
) -> None:
    """Adopting an orphan by overwriting a live cell would strand what it displaced.

    Both files carry the current analysis recipe, so only the uuid differs -- the
    orphan is named after B's uuid while B's cell already names B's derivative.
    A relink that keyed on the target column alone would overwrite it.
    """
    recipe = _live_recipe(pruned_dataset)
    existing = (
        pruned_dataset.get_root("media")
        / "transcode"
        / f"{UUID_B}.{recipe}.analysis.mp4"
    )
    duplicate = _add_derivative(pruned_dataset, uuid=UUID_B, recipe=recipe, row=False)
    assert duplicate == existing  # same name; the fixture's file is the live one

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0, relink=True)

    assert not report.links_relinked
    linked = {
        row["video_uuid"]: row["analysis_derivative_path"]
        for row in read_media_index(pruned_dataset.get_root("media_raw") / "index.csv")
    }
    assert linked[UUID_B] == f"transcode/{existing.name}"


def test_a_derivative_whose_source_is_gone_survives_every_flag(
    pruned_dataset: Dataset,
) -> None:
    """It may be the only surviving copy of an archived video.

    ``index_media`` rebuilds the whole originals index from a directory scan, so
    archiving a video drops its row *and* its link in one pass. The derivative is
    then unreferenced, carries a dead recipe, and is indistinguishable from
    garbage by every test except this one -- and it cannot be re-encoded, because
    the source is not there.
    """
    orphan = _add_derivative(
        pruned_dataset, uuid="99999999-9999-4999-8999-999999999999", recipe="deadbeef01"
    )
    assert _verdicts(pruned_dataset)[orphan.name] == "unsourced"

    report = pruned_dataset.prune_media(
        apply=True, min_age_hours=0.0, relink=True, include_stray=True
    )

    assert orphan.exists()
    assert orphan not in report.files_deleted


def test_a_linked_file_with_no_row_is_refused_in_both_directions(
    pruned_dataset: Dataset,
) -> None:
    """Deleting it breaks a working read path; clearing the link breaks the same one.

    mosaic-api opens this exact cell to serve playback and fails loud on a
    missing file, so neither half is safe. Rebuilding the row needs a probe,
    which is a different command's job.
    """
    index_path = pruned_dataset.get_root("media") / "index.csv"
    rows = [_row(**dict(row)) for row in read_media_index(index_path)]
    kept = [row for row in rows if str(row["source_video_uuid"]) != UUID_A]
    _write(index_path, kept)
    recipe = _live_recipe(pruned_dataset)
    linked = (
        pruned_dataset.get_root("media")
        / "transcode"
        / f"{UUID_A}.{recipe}.analysis.mp4"
    )
    assert _verdicts(pruned_dataset)[linked.name] == "unrowed"

    _ = pruned_dataset.prune_media(
        apply=True, min_age_hours=0.0, relink=True, include_stray=True
    )

    assert linked.exists()
    cells = {
        row["video_uuid"]: row["analysis_derivative_path"]
        for row in read_media_index(pruned_dataset.get_root("media_raw") / "index.csv")
    }
    assert cells[UUID_A] == f"transcode/{linked.name}"


def test_a_row_with_no_recipe_is_foreign_and_the_index_round_trips(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The guard against reading an originals index as a derivative index.

    ``resolve_media_root`` answers ``media_raw`` the moment that root is *set*,
    so adding one to a dataset that never had it reinterprets an index still full
    of originals. Only the transcode job fills ``recipe_hash``; a probe always
    leaves it empty and a re-probe is forbidden from writing it. Every row here
    is an original, so an ``--apply`` with every flag on must be a no-op.
    """
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    index_path = ds.get_root("media") / "index.csv"
    rows: list[dict[str, object]] = []
    for uuid, name in ((UUID_A, "a.mp4"), (UUID_B, "b.mp4")):
        original = base / "media" / name
        original.touch()
        rows.append(
            _row(
                name=name,
                group="g",
                sequence="entry",
                abs_path=ds.relative_to_root(str(original)),
                video_uuid=uuid,
                video_order=1,
            )
        )
    _write(index_path, rows)
    (ds.get_root("media_raw") / "index.csv").parent.mkdir(parents=True, exist_ok=True)
    _write(ds.get_root("media_raw") / "index.csv", [])
    before = index_path.read_bytes()

    report = ds.prune_media(
        apply=True, min_age_hours=0.0, relink=True, include_stray=True
    )

    assert report.counts().get("foreign") == 2
    assert index_path.read_bytes() == before
    assert all((base / "media" / name).exists() for name in ("a.mp4", "b.mp4"))


def test_a_row_outside_the_kind_directory_is_refused(pruned_dataset: Dataset) -> None:
    """A pre-content-address derivative sits directly under the media root.

    The pruner's blast radius stops at the kind directory, so it can see the row
    but must not follow it -- the one-off sweep is what reaches those files.
    """
    legacy = pruned_dataset.get_root("media") / "g__entry.analysis.mp4"
    legacy.write_bytes(b"legacy")
    index_path = pruned_dataset.get_root("media") / "index.csv"
    rows = [_row(**dict(row)) for row in read_media_index(index_path)]
    rows.append(
        _row(
            name=legacy.name,
            group="g",
            sequence="entry",
            abs_path=pruned_dataset.relative_to_root(str(legacy)),
            source_video_uuid=UUID_A,
            recipe_hash="legacy",
        )
    )
    _write(index_path, rows)

    report = pruned_dataset.prune_media(
        apply=True, min_age_hours=0.0, relink=True, include_stray=True
    )

    assert [entry.path for entry in report.of("outside_kind_directory")] == [
        legacy.resolve()
    ]
    assert legacy.exists()
    assert len(read_media_index(index_path)) == 3


# --- rows and files that are safe to drop -------------------------------------


def test_a_row_addressing_nothing_is_dropped(pruned_dataset: Dataset) -> None:
    """An empty ``abs_path`` cell addresses no file, so dropping it strands none.

    It is also what makes ``reprobe-media`` abort, so leaving it in place keeps a
    different command unusable.
    """
    index_path = pruned_dataset.get_root("media") / "index.csv"
    rows = [_row(**dict(row)) for row in read_media_index(index_path)]
    rows.append(_row(name="ghost", group="g", sequence="entry", recipe_hash="r"))
    _write(index_path, rows)

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0)

    assert report.counts().get("unaddressed") == 1
    assert report.rows_dropped == 1
    assert len(read_media_index(index_path)) == 2


def test_a_dangling_link_is_cleared_only_under_relink(
    pruned_dataset: Dataset,
) -> None:
    recipe = _live_recipe(pruned_dataset)
    gone = (
        pruned_dataset.get_root("media")
        / "transcode"
        / f"{UUID_A}.{recipe}.analysis.mp4"
    )
    gone.unlink()
    assert _verdicts(pruned_dataset)[gone.name] == "dangling"

    _ = pruned_dataset.prune_media(apply=True, min_age_hours=0.0)
    raw_index = pruned_dataset.get_root("media_raw") / "index.csv"
    still = {
        r["video_uuid"]: r["analysis_derivative_path"]
        for r in read_media_index(raw_index)
    }
    assert still[UUID_A], "a dangling cell was cleared without --relink"

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0, relink=True)

    assert report.links_cleared == [f"transcode/{gone.name}"]
    cleared = {
        r["video_uuid"]: r["analysis_derivative_path"]
        for r in read_media_index(raw_index)
    }
    assert cleared[UUID_A] == ""
    assert cleared[UUID_B], "the other original's link was cleared too"


# --- strays, and the in-flight encode -----------------------------------------


def test_an_interrupted_encodes_working_file_is_a_stray(
    pruned_dataset: Dataset,
) -> None:
    """``run_transcode`` holds a hidden temp beside the destination for the encode.

    A SIGKILL strands one. It is never a derivative -- the leading dot makes the
    name fail the parse -- so it is never deleted by an ordinary run, and
    ``--include-stray`` is an explicit request rather than a side effect.
    """
    transcode_root = pruned_dataset.get_root("media") / "transcode"
    temp = (
        transcode_root
        / f".{UUID_A}.{_live_recipe(pruned_dataset)}.analysis.abcd1234.mp4"
    )
    temp.write_bytes(b"partial")
    assert parse_derivative_name(temp.name) is None
    assert _verdicts(pruned_dataset)[temp.name] == "stray"

    _ = pruned_dataset.prune_media(apply=True, min_age_hours=0.0)
    assert temp.exists(), "a working file was swept without --include-stray"

    _ = pruned_dataset.prune_media(apply=True, min_age_hours=0.0, include_stray=True)
    assert not temp.exists()


def test_a_subdirectory_and_a_symlink_survive_every_flag(
    pruned_dataset: Dataset,
) -> None:
    """The kind directory is not exclusively owned; a future kind may nest here."""
    transcode_root = pruned_dataset.get_root("media") / "transcode"
    nested = transcode_root / "audio"
    nested.mkdir()
    link = transcode_root / "shortcut.mp4"
    link.symlink_to(
        transcode_root / f"{UUID_A}.{_live_recipe(pruned_dataset)}.analysis.mp4"
    )

    _ = pruned_dataset.prune_media(
        apply=True, min_age_hours=0.0, relink=True, include_stray=True
    )

    assert nested.is_dir()
    assert link.is_symlink()


def test_a_file_younger_than_the_window_is_held_back(pruned_dataset: Dataset) -> None:
    """The age window is what keeps a prune from racing a running encode."""
    stale = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01")

    report = pruned_dataset.prune_media(apply=True, min_age_hours=24.0)

    assert stale.exists()
    assert report.held_for_age == 1
    assert not report.files_deleted

    old = (datetime.now(timezone.utc) - timedelta(hours=48)).timestamp()
    os.utime(stale, (old, old))
    report = pruned_dataset.prune_media(apply=True, min_age_hours=24.0)
    assert not stale.exists()
    assert report.held_for_age == 0


# --- blast radius, gates, determinism -----------------------------------------


def test_every_other_kind_under_media_survives(pruned_dataset: Dataset) -> None:
    """Confinement to the kind directory, asserted against the real siblings.

    ``frames`` defaults to ``media/frames`` and the crop visualizers fall back to
    ``media/egocentric_crops`` / ``media/interaction_crops``, so the media root
    holds other kinds whose contents are nothing to do with transcoding.
    """
    media = pruned_dataset.get_root("media")
    witnesses = [
        media / "frames" / "kmeans" / "index.csv",
        media / "egocentric_crops" / "g__entry" / "0.png",
        media / "interaction_crops" / "clip.mp4",
        media / "index.csv",
    ]
    for path in witnesses:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"keep")
    _ = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01")

    _ = pruned_dataset.prune_media(
        apply=True, min_age_hours=0.0, relink=True, include_stray=True
    )

    assert all(path.exists() for path in witnesses)


def test_no_sequences_projection_is_written_or_disturbed(
    pruned_dataset: Dataset,
) -> None:
    """A derivative has no composition, so pruning one moves no digest.

    ``media/`` never gets a ``sequences.csv`` -- the composition roots are a
    closed set that excludes it -- and a forward-link cell is not a term of a
    media composition, so clearing one cannot move ``media_raw``'s projection
    either. This is why the write path deliberately re-projects nothing.
    """
    raw_projection = pruned_dataset.get_root("media_raw") / "sequences.csv"
    raw_projection.write_text("group,sequence,composition\n,entry,seeded\n")
    gone = (
        pruned_dataset.get_root("media")
        / "transcode"
        / f"{UUID_A}.{_live_recipe(pruned_dataset)}.analysis.mp4"
    )
    gone.unlink()
    before = raw_projection.read_bytes()

    _ = pruned_dataset.prune_media(apply=True, min_age_hours=0.0, relink=True)

    assert raw_projection.read_bytes() == before
    assert not (pruned_dataset.get_root("media") / "sequences.csv").exists()


def test_the_file_predicate_does_not_consult_the_row(pruned_dataset: Dataset) -> None:
    """A run interrupted between the row drop and the unlink finishes its own work.

    That is only true if a file's fate is decided by its name and the links, never
    by whether a row still describes it -- so dropping the rows by hand must not
    change the verdict.
    """
    stale = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01", row=False)
    assert _verdicts(pruned_dataset)[stale.name] == "superseded"

    report = pruned_dataset.prune_media(apply=True, min_age_hours=0.0)

    assert not stale.exists()
    assert report.rows_dropped == 0


def test_two_runs_agree(pruned_dataset: Dataset) -> None:
    """Filesystem order is not stable, so the walk sorts and the report must too."""
    _ = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01")
    _ = _add_derivative(pruned_dataset, uuid=UUID_B, recipe="deadbeef02")

    first = pruned_dataset.prune_media(apply=False, min_age_hours=0.0).payload()
    second = pruned_dataset.prune_media(apply=False, min_age_hours=0.0).payload()

    assert first == second


def test_a_single_root_dataset_declines(tmp_path: Path) -> None:
    """No ``media_raw`` means ``media/index.csv`` is the originals index.

    Declining is reported apart from a dry run finding nothing: "would prune 0"
    invites a re-run with ``--apply``, and here that would never be right.
    """
    base = (tmp_path / "dataset").resolve()
    ds = Dataset(
        manifest_path=base / "dataset.yaml", roots={"media": str(base / "media")}
    )
    ds.ensure_roots()
    ds.save()
    index_path = ds.get_root("media") / "index.csv"
    _write(index_path, [_row(name="a", video_uuid=UUID_A, recipe_hash="r")])
    before = index_path.read_bytes()

    report = ds.prune_media(apply=True, min_age_hours=0.0, include_stray=True)

    assert not report.considered
    assert report.declined == "single-root"
    assert not report.changed and not report.applied
    assert index_path.read_bytes() == before


def test_one_directory_under_two_root_names_declines(tmp_path: Path) -> None:
    """Two names for one directory is one index, and the run writes two.

    `_prune_media` writes a whole-file projection of the originals index and
    another of the derivatives index. Against one path the second erases the
    first, entire, with no error -- and neither projection describes the other's
    rows. Not a lock problem: `index_lock` holds a sidecar, so the two writes
    under one re-entrant lock would keep their grip.
    """
    base = (tmp_path / "dataset").resolve()
    shared = base / "media"
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={"media": str(shared), "media_raw": str(base / "." / "media")},
    )
    ds.ensure_roots()
    ds.save()

    report = ds.prune_media(apply=True, min_age_hours=0.0)

    assert report.declined == "one-index"


def test_a_root_nested_in_the_kind_directory_refuses(tmp_path: Path) -> None:
    """Roots are free-form strings, so a manifest may legally nest them."""
    base = (tmp_path / "dataset").resolve()
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={
            "media": str(base / "media"),
            "media_raw": str(base / "media" / "transcode" / "originals"),
        },
    )
    ds.ensure_roots()
    ds.save()

    report = ds.prune_media(apply=True, min_age_hours=0.0)

    assert report.declined == "nested-root"


def test_a_dataset_with_no_media_root_declines(tmp_path: Path) -> None:
    base = (tmp_path / "dataset").resolve()
    ds = Dataset(manifest_path=base / "dataset.yaml", roots={"tracks": str(base / "t")})
    ds.ensure_roots()
    ds.save()

    assert ds.prune_media(apply=True).declined == "no-media-root"


# --- the name parser ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (f"{UUID_A}.abc123.analysis.mp4", (UUID_A, "abc123", "analysis")),
        (f"{UUID_A}.abc123.playback.mp4", (UUID_A, "abc123", "playback")),
        (f".{UUID_A}.abc123.analysis.mp4", None),
        (f"{UUID_A}.abc123.analysis.mkv", None),
        (f"{UUID_A}.abc123.thumbnail.mp4", None),
        (f"{UUID_A}.analysis.mp4", None),
        ("index.csv", None),
    ],
)
def test_the_name_parser(name: str, expected: tuple[str, str, str] | None) -> None:
    assert parse_derivative_name(name) == expected


# --- the command surface ------------------------------------------------------


def _runner() -> CliRunner:
    # click <8.2 needs mix_stderr=False to split streams; >=8.2 splits by default.
    try:
        return CliRunner(mix_stderr=False)  # pyright: ignore[reportCallIssue]
    except TypeError:
        return CliRunner()


def test_the_command_is_a_dry_run_by_default(pruned_dataset: Dataset) -> None:
    stale = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01")

    result = _runner().invoke(
        app,
        [
            "prune-media",
            "-m",
            str(pruned_dataset.manifest_path),
            "--min-age-hours",
            "0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "would delete" in result.stdout
    assert stale.exists(), "the default invocation deleted something"
    # The recipes are the one input an operator cannot see and that can silently
    # differ from the worker's, so the human report has to print them.
    assert _live_recipe(pruned_dataset) in result.stdout


def test_the_json_document_is_one_value_on_stdout(pruned_dataset: Dataset) -> None:
    _ = _add_derivative(pruned_dataset, uuid=UUID_A, recipe="deadbeef01")

    result = _runner().invoke(
        app,
        [
            "prune-media",
            "-m",
            str(pruned_dataset.manifest_path),
            "--min-age-hours",
            "0",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["considered"] is True
    assert payload["applied"] is False
    assert payload["counts"]["superseded"] == 1


def test_a_decline_exits_zero_and_says_why(tmp_path: Path) -> None:
    """Declining is not a failure, and it must not read as "try --apply"."""
    base = (tmp_path / "dataset").resolve()
    ds = Dataset(
        manifest_path=base / "dataset.yaml", roots={"media": str(base / "media")}
    )
    ds.ensure_roots()
    ds.save()

    result = _runner().invoke(app, ["prune-media", "-m", str(ds.manifest_path)])

    assert result.exit_code == 0
    assert "declined" in result.stdout
    assert "would delete" not in result.stdout
