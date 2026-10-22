"""Test ``mosaic prerequisites``. The verb prints the runs that an op needs, and runs none.

Each printed command is run back through ``mosaic run`` with the op body replaced.
``mosaic run`` parses the command's flags and builds its scope as a real run does,
and the scope is then resolved against the dataset.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from typer.testing import CliRunner

import mosaic.core.pipeline.ops as ops_module
from mosaic.cli import app
from mosaic.cli.prerequisites import NoCommandCause
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.joined_export import JoinedExportParams, joined_run_id
from mosaic.core.pipeline.preprocess import PreprocessParams, preprocess_identity
from mosaic.core.pipeline.store_export import StoreExportParams
from mosaic.core.scope import Scope
from mosaic.tracking import PrerequisiteKind, RefusalCause
from mosaic.tracking.litpose.params import LitposeParams
from tests.helpers import (
    MediaClip,
    assert_tree_unchanged,
    entry_clips,
    invoke_ok,
    make_dataset,
    point_at_a_store,
    record_variant_recipe,
    tree_snapshot,
    write_media_index,
)

_NAMELESS_CLIPS = [
    MediaClip(
        sequence="",
        filename=f"rec-c{order}.mp4",
        video_order=order,
        video_uuid=f"r{order}",
    )
    for order in range(2)
]
"""Two clips of a nameless entry, which an unscoped plan labels ``("", "rec-c0")``."""

_UNJOINABLE = entry_clips("bad", widths=(640, 1280))
"""Two clips of two frame sizes, which no tool reads as one video."""

_ONE_CLIP = MediaClip(sequence="one", filename="one.mp4")
"""One clip, which a tracker reads as it is."""

_PART_REFUSED = [
    *entry_clips("", camera="a", widths=(640, 1280)),
    *entry_clips("", camera="b"),
]
"""A nameless entry whose camera ``a`` is refused, and whose camera ``b`` is read."""

_NAMELESS_CAMERA = (
    "a camera of a nameless entry that the plan keeps only in part, and the scope "
    "of a run covers every camera of an entry"
)

_JOINED = JoinedExportParams(reencode=False)


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Run(_Wire):
    kind: PrerequisiteKind
    run_id: str
    params: dict[str, object]
    entries: list[tuple[str, str]]
    after: list[int]
    command: str


class _Refusal(_Wire):
    group: str
    sequence: str
    camera: str
    cause: RefusalCause
    reason: str


class _NoCommand(_Wire):
    cause: NoCommandCause
    reason: str


class _Omitted(_Wire):
    entries: list[tuple[str, str]]
    reason: str


class _Answer(_Wire):
    """The object that ``--json`` prints."""

    kind: str
    media: str
    runs: list[_Run]
    refusals: list[_Refusal]
    entries: list[tuple[str, str]]
    command: str | None
    no_command: _NoCommand | None
    omitted: _Omitted | None


@dataclass(frozen=True)
class _Received:
    """Hold what ``mosaic run`` hands an op."""

    kind: str
    params: dict[str, object]
    scope: Scope | None


def _dataset(tmp_path: Path, clips: list[MediaClip]) -> Dataset:
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(dataset, clips)
    return dataset


def _answer(ds: Dataset, *arguments: str) -> _Answer:
    result = invoke_ok(
        ["prerequisites", "-m", str(ds.manifest_path), *arguments, "--json"]
    )
    return _Answer.model_validate_json(result.stdout)


def _run_back(command: str, monkeypatch: pytest.MonkeyPatch) -> _Received:
    """Run *command* through ``mosaic run``, and return what the op receives."""
    received: list[_Received] = []

    def receive(
        ds: Dataset,
        kind: str,
        params: dict[str, object],
        *,
        scope: Scope | None = None,
        **_: object,
    ) -> str:
        received.append(_Received(kind, params, scope))
        return "received"

    monkeypatch.setattr(ops_module, "run_op", receive)
    words = shlex.split(command)
    assert words[:2] == ["mosaic", "run"]
    _ = invoke_ok(words[1:])
    (only,) = received
    return only


def _scope_of(ds: Dataset, received: _Received) -> list[Entry] | None:
    """Return the entries that the received scope covers, ``None`` for every entry."""
    return ds.resolve_scope(received.scope).op_entries


# --- the printed commands -------------------------------------------------------


def test_an_unscoped_plan_without_refusals_prints_an_unscoped_op_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The op command omits the scope flags, and covers every entry as the plan did."""
    ds = _dataset(tmp_path, [*entry_clips("sess"), _ONE_CLIP])

    answer = _answer(ds, "--kind", "trex")

    (run,) = answer.runs
    assert (run.kind, run.run_id, run.entries) == (
        "export-joined",
        joined_run_id(_JOINED),
        [("", "sess")],
    )
    assert (answer.refusals, answer.entries) == ([], [("", "one"), ("", "sess")])
    assert answer.command is not None
    assert "--entries" not in answer.command
    op = _run_back(answer.command, monkeypatch)
    assert (op.kind, op.params, _scope_of(ds, op)) == ("trex", {}, None)
    joins = _run_back(run.command, monkeypatch)
    assert JoinedExportParams.model_validate(joins.params) == _JOINED
    assert (joins.kind, _scope_of(ds, joins)) == ("export-joined", [("", "sess")])


def test_a_refusal_scopes_the_op_command_to_the_entries_that_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nameless entry is named by its indexed key, which ``--entries :`` names.

    An unscoped plan names the entry by its first file's stem, which no scope
    resolves.
    """
    ds = _dataset(tmp_path, [*_NAMELESS_CLIPS, *_UNJOINABLE, _ONE_CLIP])

    answer = _answer(ds, "--kind", "trex")

    assert [(r.sequence, r.cause) for r in answer.refusals] == [("bad", "unjoinable")]
    assert answer.entries == [("", "one"), ("", "rec-c0")]
    assert answer.command is not None
    op = _run_back(answer.command, monkeypatch)
    assert _scope_of(ds, op) == [("", ""), ("", "one")]
    (run,) = answer.runs
    joins = _run_back(run.command, monkeypatch)
    assert (joins.kind, _scope_of(ds, joins)) == ("export-joined", [("", "")])


@pytest.mark.parametrize(
    "flags",
    [
        ["--entries", ":sess"],
        ["--sequences", "sess"],
        ["--scope", json.dumps({"entries": [["", "sess"]]})],
    ],
    ids=["entries", "sequences", "scope"],
)
def test_a_scope_flag_scopes_the_op_command_by_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    ds = _dataset(tmp_path, [*entry_clips("sess"), *_UNJOINABLE])

    answer = _answer(ds, "--kind", "trex", *flags)

    assert (answer.refusals, answer.entries) == ([], [("", "sess")])
    assert answer.command is not None
    assert _scope_of(ds, _run_back(answer.command, monkeypatch)) == [("", "sess")]


def test_the_op_command_reads_the_variant_that_the_plan_makes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``media`` joins the given params, and the variant's run makes it."""
    ds = _dataset(tmp_path, [MediaClip(sequence="b", filename="b.mp4", codec="av1")])
    recipe = PreprocessParams(steps=[], codec="h264")
    variant = preprocess_identity(recipe).run_id

    answer = _answer(
        ds, "--kind", "litpose", "--params", json.dumps({"model_path": "lp"})
    )

    assert answer.media == variant
    assert answer.command is not None
    op = _run_back(answer.command, monkeypatch)
    assert op.params == {"model_path": "lp", "media": variant}
    assert LitposeParams.model_validate(op.params).media == variant
    (run,) = answer.runs
    assert (run.kind, run.run_id, run.after) == ("preprocess", variant, [])
    made = _run_back(run.command, monkeypatch)
    assert PreprocessParams.model_validate(made.params) == recipe
    assert _scope_of(ds, made) == [("", "b")]


def test_a_chained_variant_runs_after_the_upstream_that_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path, [MediaClip(frame_count=20)])
    grayscale = PreprocessParams.model_validate({"steps": [{"step": "grayscale"}]})
    upstream = record_variant_recipe(ds, grayscale)
    chained = PreprocessParams.model_validate(
        {"steps": [{"step": "decimate", "every": 2}], "media": upstream}
    )
    variant = record_variant_recipe(ds, chained)

    answer = _answer(ds, "--kind", "trex", "--params", json.dumps({"media": variant}))

    assert [(run.run_id, run.after) for run in answer.runs] == [
        (upstream, []),
        (variant, [0]),
    ]
    for run, recipe in zip(answer.runs, (grayscale, chained), strict=True):
        made = _run_back(run.command, monkeypatch)
        assert (made.kind, _scope_of(ds, made)) == ("preprocess", [("", "sess")])
        assert PreprocessParams.model_validate(made.params) == recipe
    assert answer.command is not None
    assert _run_back(answer.command, monkeypatch).params == {"media": variant}


def test_the_export_of_a_store_names_the_store_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(
        tmp_path,
        [
            MediaClip(sequence="store", filename="store.mp4", video_uuid="uid-store"),
            *entry_clips("sess"),
        ],
    )
    _ = point_at_a_store(ds, "store", ds.get_root("media_raw") / "store.store")

    answer = _answer(ds, "--kind", "trex")

    joins, exports = answer.runs
    assert (joins.kind, exports.kind) == ("export-joined", "export-store")
    made = _run_back(exports.command, monkeypatch)
    assert StoreExportParams.model_validate(made.params) == StoreExportParams()
    assert (made.kind, _scope_of(ds, made)) == ("export-store", [("", "store")])
    assert _scope_of(ds, _run_back(joins.command, monkeypatch)) == [("", "sess")]


def test_a_refusal_scopes_the_extract_frames_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _dataset(tmp_path, [*entry_clips("sess"), *_UNJOINABLE, _ONE_CLIP])
    params = {"n_frames": 3}

    answer = _answer(ds, "--kind", "extract-frames", "--params", json.dumps(params))

    assert [(r.sequence, r.cause) for r in answer.refusals] == [("bad", "unjoinable")]
    assert answer.command is not None
    op = _run_back(answer.command, monkeypatch)
    assert (op.kind, op.params) == ("extract-frames", params)
    assert _scope_of(ds, op) == [("", "one"), ("", "sess")]


def test_the_op_command_omits_the_cameras_of_a_nameless_entry_kept_in_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--entries :`` would name the refused camera too."""
    ds = _dataset(tmp_path, [*_PART_REFUSED, _ONE_CLIP])

    answer = _answer(ds, "--kind", "infer-pose")

    assert [(r.camera, r.cause) for r in answer.refusals] == [("a", "unjoinable")]
    assert answer.entries == [("", "b-c0"), ("", "one")]
    assert answer.omitted == _Omitted(
        entries=[("", "b-c0")],
        reason=f"The infer-pose command omits each of these entries, because each is "
        f"{_NAMELESS_CAMERA}.",
    )
    assert answer.command is not None
    assert _scope_of(ds, _run_back(answer.command, monkeypatch)) == [("", "one")]


# --- a withheld op command ------------------------------------------------------


def test_a_scope_of_refused_entries_withholds_the_op_command(tmp_path: Path) -> None:
    """A command without a scope covers the whole dataset."""
    ds = _dataset(tmp_path, _UNJOINABLE)

    answer = _answer(ds, "--kind", "trex")

    assert answer.command is None
    assert answer.no_command == _NoCommand(
        cause="every_entry_refused",
        reason="The trex command is withheld, because every entry is refused.",
    )


def test_a_nameless_entry_kept_in_part_withholds_the_op_command(
    tmp_path: Path,
) -> None:
    """The plan keeps one camera of the entry, which no run's scope names alone."""
    ds = _dataset(tmp_path, _PART_REFUSED)

    answer = _answer(ds, "--kind", "infer-pose")

    assert (answer.entries, answer.command, answer.omitted) == (
        [("", "b-c0")],
        None,
        None,
    )
    assert answer.no_command == _NoCommand(
        cause="nameless_cameras",
        reason="The infer-pose command is withheld, because each entry left is "
        f"{_NAMELESS_CAMERA}.",
    )


@pytest.mark.parametrize(
    ("clips", "flags", "cause", "because"),
    [
        (
            [],
            [],
            "no_indexed_media",
            "the media index of the dataset is absent or empty",
        ),
        (
            entry_clips("sess"),
            ["--groups", "absent"],
            "empty_scope",
            "the scope selects an empty set of entries",
        ),
    ],
    ids=["no-media", "flags-name-nothing"],
)
def test_an_empty_scope_withholds_the_op_command(
    tmp_path: Path,
    clips: list[MediaClip],
    flags: list[str],
    cause: NoCommandCause,
    because: str,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    if clips:
        write_media_index(ds, clips)

    answer = _answer(ds, "--kind", "trex", *flags)

    assert (answer.runs, answer.refusals, answer.command) == ([], [], None)
    assert answer.no_command == _NoCommand(
        cause=cause, reason=f"The trex command is withheld, because {because}."
    )


# --- the human output -----------------------------------------------------------


def test_the_human_output_names_each_run_refusal_and_command(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, [*entry_clips("sess"), *_UNJOINABLE])
    answer = _answer(ds, "--kind", "trex")

    result = invoke_ok(["prerequisites", "-m", str(ds.manifest_path), "--kind", "trex"])

    (run,) = answer.runs
    (refusal,) = answer.refusals
    assert answer.command is not None
    assert result.stdout.splitlines()[:7] == [
        "kind: trex",
        "media: the entry media",
        "runs: 1",
        f"  1. export-joined {run.run_id}",
        f"     {run.command}",
        "refusals: 1",
        "  :bad (unjoinable)",
    ]
    assert refusal.reason in result.stdout
    assert result.stdout.endswith(f"command:\n  {answer.command}\n")


def test_the_human_output_lists_the_entries_that_the_op_command_omits(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path, [*_PART_REFUSED, _ONE_CLIP])
    answer = _answer(ds, "--kind", "infer-pose")

    result = invoke_ok(
        ["prerequisites", "-m", str(ds.manifest_path), "--kind", "infer-pose"]
    )

    assert answer.omitted is not None
    assert result.stdout.endswith(
        f"command:\n  {answer.command}\nomitted: 1\n  :b-c0\n"
        f"     {answer.omitted.reason}\n"
    )


def test_the_human_output_says_why_the_op_command_is_withheld(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, _UNJOINABLE)

    result = invoke_ok(["prerequisites", "-m", str(ds.manifest_path), "--kind", "trex"])

    assert result.stdout.endswith(
        "command:\n  The trex command is withheld, because every entry is refused.\n"
    )


# --- refusals ------------------------------------------------------------------


def test_a_missing_kind_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    result = CliRunner().invoke(app, ["prerequisites", "-m", str(ds.manifest_path)])

    assert result.exit_code == 2
    assert "Missing option '--kind'" in result.output


def test_a_scope_that_the_op_refuses_is_refused_with_the_flags(tmp_path: Path) -> None:
    """``preprocess`` needs a scope. ``mosaic run`` refuses its unscoped command."""
    ds = _dataset(tmp_path, entry_clips("sess"))
    recipe = json.dumps({"steps": [], "codec": "h264"})

    result = CliRunner().invoke(
        app,
        [
            "prerequisites",
            "-m",
            str(ds.manifest_path),
            "--kind",
            "preprocess",
            "--params",
            recipe,
        ],
    )

    assert result.exit_code == 1
    assert "preprocess must be told which entries to cover" in result.output
    assert "At the command line: --entries group:sequence" in result.output


def test_the_verb_writes_nothing(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, [*entry_clips("sess"), *_UNJOINABLE])
    before = tree_snapshot(ds.base_dir)

    answer = _answer(ds, "--kind", "trex")

    assert [run.kind for run in answer.runs] == ["export-joined"]
    assert_tree_unchanged(ds.base_dir, before)
