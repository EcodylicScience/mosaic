"""How a command is told what to cover, and what it refuses to be told.

One vocabulary serves both arms. ``--entries``, ``--groups`` and
``--sequences`` name a scope for a feature run and for an op run alike, and a
selector written inside ``--params`` is refused: those keys are fields on no
feature and on no op, and a run that accepted one would take a narrowing its
own model never validated.

A refused scope is written once, by ``check_scope_takes``, in the
``Scope(...)`` its library, planner and API callers construct. Each command
appends the flags it offers, which differ: ``mosaic run`` has three and
``mosaic pipeline`` has ``--entry``. Both are asserted here so the two cannot
drift from the one sentence they share.

Every case here drives the real command. A unit call to the selector builder
proves the builder works and says nothing about whether the command gets to it,
which is the half that was wrong before these flags existed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import read_tracks_index, write_tracks_row
from mosaic.core.pipeline.writers import write_parquet_atomic
from tests.helpers import MediaClip, make_dataset, write_media_index

runner = CliRunner()

SOURCE_VARIANT = "convert-trex_npz.0.2-aaaaaaaaaa"
"""The tracks recipe the seeded tables answer to."""

ENTRIES = (("A", "one"), ("A", "two"), ("B", "one"))
"""Two groups, and a sequence name repeated across them."""

INDEXED = tuple(reversed(ENTRIES))
"""The order the media index rows are written in.

The reverse of their sorted order, so a resolution that returned rows in the
order it read them would name a different list from one that sorted.
"""


def _table(group: str, sequence: str, rows: int = 200) -> pd.DataFrame:
    """A minimal table on a known frame rate, which is all the op re-grids."""
    frames = np.arange(rows)
    return pd.DataFrame(
        {
            "frame": frames.astype(np.int64),
            "time": (frames / 34.679).astype(np.float64),
            "id": np.zeros(rows, dtype=np.int64),
            "group": [group] * rows,
            "sequence": [sequence] * rows,
            "X": frames * 0.2,
            "Y": np.sin(frames / 25.0) * 10.0,
            "frame_rate": np.full(rows, 34.679),
        }
    )


@pytest.fixture
def scoped(tmp_path: Path) -> Dataset:
    """Three entries across two groups, indexed and tracked."""
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset,
        [
            MediaClip(
                filename=f"{group}_{sequence}.mp4", group=group, sequence=sequence
            )
            for group, sequence in INDEXED
        ],
    )
    root = tracks_variant_root(dataset.get_root("tracks"), SOURCE_VARIANT)
    for group, sequence in ENTRIES:
        frame = _table(group, sequence)
        out_path = root / f"{group}__{sequence}.parquet"
        _ = write_parquet_atomic(frame, out_path)
        write_tracks_row(
            dataset,
            run_id=SOURCE_VARIANT,
            group=group,
            sequence=sequence,
            out_path=out_path,
            producer="convert-trex_npz",
            std_format="mosaic_v1",
            n_rows=len(frame),
            consumed_source_roots=("tracks_raw",),
        )
    return dataset


def _regrid(dataset: Dataset, *flags: str) -> Result:
    """``mosaic run --kind resample-tracks`` over *dataset*, as a caller types it."""
    return runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(dataset.manifest_path),
            "--kind",
            "resample-tracks",
            "--params",
            json.dumps({"target_fps": 30.0}),
            *flags,
        ],
    )


def _regridded(dataset: Dataset) -> list[tuple[str, str]]:
    """Which entries the op wrote a re-gridded table for."""
    index = read_tracks_index(dataset)
    produced = index[index["producer"] == "resample-tracks"]
    return sorted((str(row.group), str(row.sequence)) for row in produced.itertuples())


# --- the flags -----------------------------------------------------------------


def test_an_op_run_covers_the_entries_the_flag_names(scoped: Dataset) -> None:
    """``--entries`` narrows an op run, where it used to be refused by name.

    Asserted on the tables the op wrote. An accepted-and-discarded flag runs
    over every entry and still exits zero.
    """
    result = _regrid(scoped, "--entries", "A:one")

    assert result.exit_code == 0, result.output
    assert _regridded(scoped) == [("A", "one")]


def test_an_op_run_covers_the_group_the_flag_names(scoped: Dataset) -> None:
    """``--groups`` names a cross product the media index enumerates."""
    result = _regrid(scoped, "--groups", "A")

    assert result.exit_code == 0, result.output
    assert _regridded(scoped) == [("A", "one"), ("A", "two")]


def test_an_op_run_covers_the_sequence_the_flag_names(scoped: Dataset) -> None:
    """``--sequences`` crosses every group, which makes it a product."""
    result = _regrid(scoped, "--sequences", "one")

    assert result.exit_code == 0, result.output
    assert _regridded(scoped) == [("A", "one"), ("B", "one")]


def test_an_unscoped_op_run_covers_everything(scoped: Dataset) -> None:
    """The control the three above need: an unset selector still means all."""
    result = _regrid(scoped)

    assert result.exit_code == 0, result.output
    assert _regridded(scoped) == sorted(ENTRIES)


def test_a_feature_run_covers_the_group_the_flag_names(scoped: Dataset) -> None:
    """The feature arm takes the same three flags, and had only ``--entries``."""
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(scoped.manifest_path),
            "--feature",
            "speed-angvel",
            "--groups",
            "A",
        ],
    )

    assert result.exit_code == 0, result.output
    storage = scoped.get_root("features") / "speed-angvel__from__tracks"
    written = sorted(path.name for path in storage.rglob("*.parquet"))
    assert written == ["A__one.parquet", "A__two.parquet"]


def test_a_repeated_entry_is_one_entry(scoped: Dataset) -> None:
    """The flag builds a selector, and a selector is a set of entries.

    A run covering one set of entries records one list however a caller wrote
    them, keeping a duplicated token off the identity.
    """
    result = _regrid(
        scoped, "--entries", "B:one", "--entries", "A:one", "--entries", "B:one"
    )

    assert result.exit_code == 0, result.output
    assert _regridded(scoped) == [("A", "one"), ("B", "one")]


def test_index_order_does_not_reach_what_an_op_covers(scoped: Dataset) -> None:
    """The entries an op covers do not depend on the order the index holds them.

    The fixture writes its media index in the reverse of its sorted order, and
    the flags below name their groups in the reverse of theirs. Neither reaches
    what the run covers.
    """
    result = _regrid(scoped, "--groups", "B", "--groups", "A")

    assert result.exit_code == 0, result.output
    assert _regridded(scoped) == [("A", "one"), ("A", "two"), ("B", "one")]


# --- overwrite -----------------------------------------------------------------


def test_an_op_run_accepts_overwrite_and_recomputes(scoped: Dataset) -> None:
    """``--overwrite`` was refused with ``--kind`` while six ops ignored it.

    The re-gridded table is replaced with a file the op would never write, and
    the flag has to put a readable one back. Asserted on the content rather
    than on the exit code, which an accepted-and-dropped flag also returns.
    """
    assert _regrid(scoped, "--entries", "A:one").exit_code == 0
    index = read_tracks_index(scoped)
    produced = index[index["producer"] == "resample-tracks"]
    written = scoped.resolve_path(str(produced.iloc[0]["abs_path"]))
    _ = Path(written).write_bytes(b"not a parquet file")

    assert _regrid(scoped, "--entries", "A:one").exit_code == 0
    assert Path(written).read_bytes() == b"not a parquet file", (
        "a reuse must leave the table it already wrote"
    )

    result = _regrid(scoped, "--entries", "A:one", "--overwrite")

    assert result.exit_code == 0, result.output
    assert len(pd.read_parquet(written)) > 0


# --- what is refused -----------------------------------------------------------


def test_entries_beside_groups_is_refused_naming_both_forms(scoped: Dataset) -> None:
    """One selector, two ways of writing it, and the message names both."""
    result = _regrid(scoped, "--entries", "A:one", "--groups", "A")

    assert result.exit_code == 1
    assert "entries" in result.output
    assert "groups" in result.output


def test_a_scope_key_in_params_is_refused_for_an_op(scoped: Dataset) -> None:
    """The keys an op run used to take its scope from, now named as flags."""
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(scoped.manifest_path),
            "--kind",
            "resample-tracks",
            "--params",
            json.dumps({"target_fps": 30.0, "entries": [["A", "one"]]}),
        ],
    )

    assert result.exit_code == 1
    assert "--entries" in result.output
    assert _regridded(scoped) == [], "a refused run must write nothing"


def test_a_scope_key_in_params_is_refused_for_a_feature(scoped: Dataset) -> None:
    """Refused on the feature arm too, where it was an unknown params field."""
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(scoped.manifest_path),
            "--feature",
            "speed-angvel",
            "--params",
            json.dumps({"groups": ["A"]}),
        ],
    )

    assert result.exit_code == 1
    assert "--groups" in result.output


def test_a_scope_free_op_is_refused_a_scope_at_the_command_line(
    scoped: Dataset,
) -> None:
    """A training step reads a prepared directory and covers no entry.

    Driven through the command because a unit call to the checker proves only
    that the checker works. What has to hold is that the flags reach it.
    """
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(scoped.manifest_path),
            "--kind",
            "train-pose",
            "--params",
            json.dumps({"data": "datasets/pose/data.yaml"}),
            "--groups",
            "A",
        ],
    )

    assert result.exit_code != 0
    assert "train-pose takes no entry scope" in result.output


def test_a_scope_free_op_is_not_refused_when_no_scope_is_named(
    scoped: Dataset,
) -> None:
    """The other side, which keeps the refusal above from passing by refusing always.

    The op gets past the scope check and fails on its own missing tool
    environment, which is as far as this dataset takes it.
    """
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(scoped.manifest_path),
            "--kind",
            "train-pose",
            "--params",
            json.dumps({"data": "datasets/pose/data.yaml"}),
        ],
    )

    assert "takes no entry scope" not in result.output


# --- how a refusal reads ---------------------------------------------------------


def test_a_refused_scope_names_the_flags_this_command_offers(scoped: Dataset) -> None:
    """The checker answers in ``Scope(...)``, which nobody types at a terminal.

    Both halves are asserted. The consequence the checker states is what decides
    whether to narrow or to proceed, and no flag list replaces it -- a rewrite
    that dropped it would leave a message saying which flags exist and not why
    they matter here.
    """
    result = runner.invoke(
        app,
        [
            "run",
            "--manifest",
            str(scoped.manifest_path),
            "--kind",
            "transcode",
            "--params",
            json.dumps({"target": "analysis"}),
        ],
    )

    assert result.exit_code == 1
    assert "re-encode every video in the dataset" in result.output
    assert "--entries group:sequence" in result.output
    assert "--groups / --sequences" in result.output


def test_a_pipeline_refusal_names_the_flag_that_command_offers(
    scoped: Dataset, tmp_path: Path
) -> None:
    """``mosaic pipeline`` offers ``--entry``, and used to answer with a traceback.

    ``plan_pipeline`` raises the refusal and no verb caught it, so the message
    reached a terminal under a stack trace or not at all.
    """
    recipe = tmp_path / "recipe.json"
    _ = recipe.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": [
                    {"id": "export", "type": "op", "kind": "export-store", "params": {}}
                ],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--manifest",
            str(scoped.manifest_path),
            "--recipe",
            f"@{recipe}",
        ],
    )

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        "a refusal must be a message, not a traceback"
    )
    assert "covers one entry and this scope resolves 3" in result.output
    assert "--entry group:sequence" in result.output
