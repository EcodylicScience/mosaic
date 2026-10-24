"""``mosaic pipeline score``: score a finished recipe on held-out data added afterwards.

The command a held-out protocol ends with. The agent fits on the training
groups, and the operator runs this to learn how the answer does on data that no
fit has seen. It works on a copy of the study, so the reports never enter a
dataset the agent can read, and it refuses to run when adding the held-out data
would change the run id of any step of the recipe (:mod:`mosaic.core.pipeline.graph.score`
gives the steps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    MinFreeMemoryOption,
    ParallelModeOption,
    StepScopeOption,
    WorkersOption,
    emit_json,
    exit_refused,
    fail,
    load_json_arg,
    log,
    parallel_settings,
    parse_step_scopes,
    stdout_to_stderr,
)
from mosaic.cli.pipeline import (
    JsonOption,
    ManifestOption,
    RecipeOption,
    fail_unrunnable,
    recipe_argument,
)
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.refusal import Refusal
from mosaic.core.strict_model import terse
from mosaic.user_paths import user_path


def _json_file(value: str, what: str) -> object:
    """Read *value* as a path to a JSON file, or as ``@file``, ``@-`` or inline JSON."""
    if not value.startswith(("@", "{", "[")):
        if not user_path(value).is_file():
            fail(f"{what}: {value!r} is not a file and not inline JSON.")
        value = f"@{value}"
    parsed = load_json_arg(value)
    if parsed is None:
        fail(f"{what} is empty.")
    return parsed


def _perspective(
    value: str | None,
) -> int | Literal["pooled", "mean_probabilities"] | None:
    if value is None:
        return None
    if value in ("pooled", "mean_probabilities"):
        return value
    try:
        return int(value)
    except ValueError:
        fail(
            f"--perspective {value!r} is not an integer, 'pooled' or "
            "'mean_probabilities'."
        )


def score_command(
    manifest: ManifestOption,
    recipe: RecipeOption,
    heldout: Annotated[
        str,
        typer.Option(
            "--heldout",
            help=(
                "The held-out data: a file or a directory to add as a tracks and "
                "a labels source of the copy, or a group already in the dataset "
                "whose role is test or sealed, which needs nothing added."
            ),
        ),
    ],
    answers: Annotated[
        str,
        typer.Option(
            "--answers",
            help=(
                'The steps to score, as a JSON file or inline JSON: {"models": '
                '{"<name>": ["<step id>", ...]}}. One name with several steps is '
                "reported as mean and standard deviation."
            ),
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help=(
                "Where the copy of the dataset goes, and the scores are written "
                "(test_scores.json and test_scores.txt). Refused when it exists, "
                "unless --reuse."
            ),
        ),
    ],
    labels_path: Annotated[
        Path | None,
        typer.Option(
            "--labels-path",
            help="Where the held-out labels are, when not beside the tracks.",
        ),
    ] = None,
    src_format: Annotated[
        str | None,
        typer.Option(
            "--src-format",
            help=(
                "The raw format of the held-out files. It must be the format the "
                "recipe's convert steps read, which is where it is taken from."
            ),
        ),
    ] = None,
    patterns: Annotated[
        str | None,
        typer.Option(
            "--patterns",
            help="With a directory, comma-separated globs selecting its files.",
        ),
    ] = None,
    multi_sequences_per_file: Annotated[
        bool,
        typer.Option(
            "--multi-sequences-per-file",
            help="One held-out file holds several sequences.",
        ),
    ] = False,
    group_from: Annotated[
        str | None,
        typer.Option(
            "--group-from", help="'filename' or 'parent'. Multi-sequence files only."
        ),
    ] = None,
    group_pattern: Annotated[
        str | None,
        typer.Option("--group-pattern", help="Regex extracting the group from a path."),
    ] = None,
    perspective: Annotated[
        str | None,
        typer.Option(
            "--perspective",
            help=(
                "Which rows of a pair output the evaluation scores: an individual "
                "as an integer, 'pooled' or 'mean_probabilities'."
            ),
        ),
    ] = None,
    average_over: Annotated[
        list[int] | None,
        typer.Option(
            "--average-over",
            help="A class the macro F1 averages over (repeatable). Unset: every class.",
        ),
    ] = None,
    classes: Annotated[
        list[int] | None,
        typer.Option("--classes", help="A class to report (repeatable), in order."),
    ] = None,
    prediction_column: Annotated[
        str | None,
        typer.Option(
            "--prediction-column", help="The column holding each row's predicted class."
        ),
    ] = None,
    step_scope: StepScopeOption = None,
    expect_plan: Annotated[
        str | None,
        typer.Option(
            "--expect-plan",
            help=(
                "The plan_fingerprint that a --dry-run printed. Refuses the run, "
                "writing nothing more, when the scoring plan has changed since."
            ),
        ),
    ] = None,
    expect_md5: Annotated[
        str | None,
        typer.Option(
            "--expect-md5",
            help=(
                "The MD5 of the held-out file. Refuses an incomplete copy before "
                "anything is done. A file only."
            ),
        ),
    ] = None,
    reuse: Annotated[
        bool,
        typer.Option(
            "--reuse",
            help=(
                "Continue with the copy in --out after a failure or a dry run. The "
                "plan recorded before the held-out data was added is the one compared."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Stop after the comparison of run ids: nothing is run or scored. "
                "The study is never written. A held-out group already in the "
                "dataset is planned in place and writes nothing at all. Held-out "
                "files are added to the copy in --out, which is what the "
                "comparison needs."
            ),
        ),
    ] = False,
    workers: WorkersOption = None,
    parallel_mode: ParallelModeOption = None,
    min_free_memory_fraction: MinFreeMemoryOption = 0.0,
    as_json: JsonOption = False,
) -> None:
    """Score a finished recipe on held-out data that was added afterwards.

    Copies the study to --out, adds the held-out data to the copy, converts it
    with the recipe's own convert steps, appends one evaluate-labels step per
    answer step, and runs the recipe. Every fit of the recipe is kept: the
    command compares the run id of every step before and after the held-out data
    joins, and exits 1 listing 'RUN IDS CHANGED' when one moved, because that
    step would be fitted again on different data. Nothing is run then.

    Give --step-scope as the recipe was run, so each fit stays on the training
    groups. The study is opened read-only and is never written.
    """
    from mosaic.core.pipeline.graph import (
        HeldOutFiles,
        ScoreAnswers,
        ScoreProtocol,
        ScoreRefused,
        score_heldout,
    )
    from mosaic.core.pipeline.graph.score import md5_of_file
    from mosaic.core.pipeline.job import Cancelled, CancelToken, install_signal_handler
    from pydantic import ValidationError

    study = load_dataset(manifest, read_only=True)
    parsed = recipe_argument(recipe, as_json=as_json)
    step_scopes = parse_step_scopes(step_scope)
    try:
        wanted = ScoreAnswers.model_validate(_json_file(answers, "--answers"))
        protocol = ScoreProtocol(
            perspective=_perspective(perspective),
            average_over=average_over,
            classes=classes,
            prediction_column=prediction_column,
        )
    except ValidationError as error:
        fail(f"invalid scoring options: {terse(error)}")

    path = user_path(heldout)
    where: HeldOutFiles | str
    if path.exists():
        if expect_md5 is not None:
            if not path.is_file():
                fail("--expect-md5 names a file; --heldout is a directory.")
            found = md5_of_file(path)
            if found != expect_md5.lower():
                fail(
                    f"{path.name} does not match its expected md5 (an incomplete "
                    f"copy?): found {found}, expected {expect_md5}."
                )
        where = HeldOutFiles(
            tracks=path.resolve(),
            labels=None if labels_path is None else user_path(labels_path).resolve(),
            src_format=src_format,
            patterns=(
                None
                if patterns is None
                else tuple(part.strip() for part in patterns.split(",") if part.strip())
            ),
            multi_sequences_per_file=multi_sequences_per_file,
            group_from=group_from,
            group_pattern=group_pattern,
        )
    else:
        if expect_md5 is not None:
            fail("--expect-md5 needs --heldout to be an existing file.")
        where = heldout

    token = CancelToken()
    install_signal_handler(token)
    try:
        with stdout_to_stderr():
            result = score_heldout(
                study,
                parsed,
                answers=wanted,
                heldout=where,
                out=out,
                protocol=protocol,
                step_scopes=step_scopes,
                reuse=reuse,
                dry_run=dry_run,
                expect_plan=expect_plan,
                parallel=parallel_settings(
                    workers, parallel_mode, min_free_memory_fraction
                ),
                cancel_token=token,
                notify=lambda line: log(f"[score] {line}"),
            )
    except Cancelled:
        raise typer.Exit(code=130) from None
    except ScoreRefused as error:
        fail(str(error))
    except Refusal as refusal:
        exit_refused("", refusal, as_json=as_json)
    except ClassifiedError as error:
        fail_unrunnable(error, as_json=as_json)

    for warning in result.warnings:
        log(f"warning: {warning}")
    if as_json:
        emit_json(result.model_dump(mode="json"))
    elif result.moved:
        pass
    elif dry_run:
        typer.echo(
            "dry run: every step of the recipe keeps its run id. Nothing was run. "
            f"Plan fingerprint {result.plan_fingerprint}"
        )
    else:
        typer.echo(result.table)
        for written in result.files:
            log(f"written: {written}")
    if result.moved:
        log(
            "RUN IDS CHANGED (these steps would be fitted again, not scored as "
            "the agent left them):"
        )
        for moved in result.moved:
            log(str(moved))
        raise typer.Exit(code=1)
