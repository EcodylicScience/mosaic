"""``mosaic run``: execute a feature, an op, or one step of a pipeline request.

This command is the executor's *unit of work* -- the Layer-2 executor shells out
to ``mosaic run --json`` in its own process group. It pre-mints the ULID
``execution_id`` so it can be printed up front (and injected via ``--execution-id``
by the executor), installs a SIGTERM/SIGINT -> cooperative-cancel handler, and
prints ``{execution_id, feature|kind, run_id, status, cache_hit, entries_written}``.

The third form, ``--graph-request <id> --step <id>``, is **step-addressed**: it
names a step of a submitted pipeline and lets the step read the rest out of the
request and the recipe. That is strictly more expressive than spelling the run
out, because several of the arguments that reach a feature's identity have no
flag here at all -- the entry narrowing, the frame filters, the overlap width.

**The request is found from the manifest's parent, and there is deliberately no
second path flag.** A path the queue does not know about is one it cannot
translate for a substrate that mounts the dataset somewhere else, and that would
break precisely where it is hardest to see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from pydantic import ValidationError

from mosaic.cli._context import attempt_facts, load_dataset
from mosaic.cli._features import build_feature
from mosaic.cli._io import (
    RUN_SCOPE_FLAGS_REMEDY,
    MinFreeMemoryOption,
    ParallelModeOption,
    RunEntriesOption,
    RunGroupsOption,
    RunKindOption,
    RunParamsOption,
    RunScopeOption,
    RunSequencesOption,
    WorkersOption,
    command_line_params,
    command_line_scope,
    parallel_settings,
    emit_json,
    exit_refused,
    fail,
    fail_classified,
    load_json_arg,
    log,
    refuse_scope_named_twice,
    stdout_to_stderr,
    with_command_line_scope,
)
from mosaic.cli._render import render_kv
from mosaic.core.strict_model import terse
from mosaic.runlog import attempt_outcome


def run_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    feature: Annotated[
        str | None,
        typer.Option("--feature", help="Feature slug to run, e.g. 'speed-angvel'."),
    ] = None,
    kind: RunKindOption = None,
    graph_request: Annotated[
        str | None,
        typer.Option(
            "--graph-request",
            help=(
                "Run one step of this submitted pipeline request. The request is "
                "read from the dataset that --manifest names."
            ),
        ),
    ] = None,
    step: Annotated[
        str | None,
        typer.Option("--step", help="Which step of --graph-request to run."),
    ] = None,
    params: RunParamsOption = None,
    inputs: Annotated[
        str | None,
        typer.Option(
            "--inputs",
            help='Feature inputs as JSON (default ["tracks"]). Feature runs only.',
        ),
    ] = None,
    groups: RunGroupsOption = None,
    sequences: RunSequencesOption = None,
    entries: RunEntriesOption = None,
    scope_json: RunScopeOption = None,
    tracks_run_id: Annotated[
        str | None,
        typer.Option(
            "--tracks-run-id",
            help=(
                "Which tracks variant to read, e.g. 'trex.0.1-abc123def0'. "
                "Feature runs only; needed when one sequence has two recipes."
            ),
        ),
    ] = None,
    labels_run_id: Annotated[
        str | None,
        typer.Option(
            "--labels-run-id",
            help=(
                "Which labels variant to read, e.g. 'trex.0.1-abc123def0'. "
                "Feature runs only; needed when one sequence has two recipes."
            ),
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Recompute even if a cached run exists.")
    ] = False,
    republish: Annotated[
        bool,
        typer.Option(
            "--republish",
            help=(
                "With --kind: rebuild the op's published outputs from its "
                "finished work without redoing it. Ops without a republish path "
                "refuse it."
            ),
        ),
    ] = False,
    owner: Annotated[
        str,
        typer.Option("--owner", help="Free-form attribution recorded on the attempt."),
    ] = "",
    execution_id: Annotated[
        str | None,
        typer.Option(
            "--execution-id",
            help="Reuse an externally minted ULID (executor unit-of-work).",
        ),
    ] = None,
    workers: WorkersOption = None,
    parallel_mode: ParallelModeOption = None,
    min_free_memory_fraction: MinFreeMemoryOption = 0.0,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json", help="Emit one JSON object on stdout; logs go to stderr."
        ),
    ] = False,
) -> None:
    """Run a feature (--feature), an op (--kind), or one step of a request."""
    named = [feature is not None, kind is not None, graph_request is not None]
    if sum(named) != 1:
        fail("Provide exactly one of --feature, --kind or --graph-request.")
    refuse_scope_named_twice(entries, groups, sequences, scope_json)
    if (graph_request is not None) != (step is not None):
        fail("--graph-request and --step are used together, or not at all.")
    parallel = parallel_settings(workers, parallel_mode, min_free_memory_fraction)
    if parallel is not None and kind is not None:
        fail("--workers is used with --feature or --graph-request, not --kind.")
    if republish and kind is None:
        msg = (
            "--republish rebuilds an op's published outputs, so it is used with "
            "--kind only."
        )
        fail(msg)
    # Below the pairing check, which reports the more actionable fault, and above
    # ``load_dataset``. A step covers the entries its plan resolved: the
    # submission's narrowing minus what is computed and minus what is
    # quarantined. A flag here would name a fourth thing that nothing reads.
    if graph_request is not None and (
        entries is not None
        or groups is not None
        or sequences is not None
        or scope_json is not None
    ):
        fail(
            "--entries / --groups / --sequences / --scope are not supported "
            "with --graph-request. A step covers the entries its plan resolved. "
            "Narrow the submission instead, with "
            "'mosaic pipeline submit --entry group:sequence'."
        )

    from mosaic.core.pipeline._utils import new_execution_id
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.pipeline.ops import ScopeRefused
    from mosaic.core.pipeline.job import CancelToken, Cancelled, install_signal_handler
    from mosaic.core.pipeline.refusal import Refusal

    ds = load_dataset(manifest)

    params_dict = command_line_params(params)
    # The --feature arm and the --kind arm read one selector.
    scope = command_line_scope(entries, groups, sequences, scope_json)

    exec_id = execution_id or new_execution_id()
    token = CancelToken()
    install_signal_handler(token)

    payload: dict[str, object]
    try:
        if graph_request is not None:
            from mosaic.core.pipeline.graph import (
                execute_step,
                load_request,
            )

            # The request is found from the manifest's parent rather than from a
            # flag of its own: a second path is one a queue cannot translate for
            # a substrate that mounts the dataset somewhere else.
            request = load_request(ds.base_dir, graph_request)
            step_id = cast("str", step)
            log(f"[mosaic] execution_id={exec_id} running step {step_id}")
            with stdout_to_stderr():
                outcome = execute_step(
                    ds,
                    request,
                    step_id,
                    execution_id=exec_id,
                    overwrite=overwrite,
                    owner=owner,
                    cancel_token=token,
                    parallel=parallel,
                )
            payload = {
                "execution_id": exec_id,
                "request_id": request.request_id,
                "step": outcome.step_id,
                "run_id": outcome.run_id,
                "status": attempt_outcome("finished", len(outcome.failed_entries)),
                "state": outcome.state,
                "cache_hit": outcome.state == "cached",
                "covered": outcome.covered,
                "target": outcome.target,
                "failed_entries": list(outcome.failed_entries),
            }
        elif feature is not None:
            feat = build_feature(
                feature, load_json_arg(inputs), params_dict, as_json=as_json
            )
            from mosaic.core.pipeline.run import run_feature

            log(f"[mosaic] execution_id={exec_id} running {feature}")
            with stdout_to_stderr():
                result = run_feature(
                    ds,
                    feat,
                    scope=None if scope.is_unset else scope,
                    overwrite=overwrite,
                    tracks_run_id=tracks_run_id,
                    labels_run_id=labels_run_id,
                    execution_id=exec_id,
                    owner=owner,
                    cancel_token=token,
                    parallel_workers=None if parallel is None else parallel.workers,
                    parallel_mode="thread" if parallel is None else parallel.mode,
                    min_free_memory_fraction=(
                        0.0 if parallel is None else parallel.min_free_memory_fraction
                    ),
                )
            # "partial" rather than "finished" when entities were lost. It is
            # deliberately not a new *terminal* status: `partial` is absent from
            # `runlog.TERMINAL_STATUSES` on purpose, because mosaic-api's sweeper
            # treats that set as terminal and would reap a live run. The exit code
            # stays 0, so `terminal_status_for_exit` still records `finished` in
            # the ledger -- with `entries_failed` and the per-entity errors
            # alongside it, which is what a reader needs and what stderr could
            # never carry under the queue.
            payload = {
                "execution_id": result.execution_id,
                "feature": result.feature,
                "run_id": result.run_id,
                "status": attempt_outcome("finished", len(result.failed_entries)),
                "cache_hit": result.cache_hit,
                "failed_entries": list(result.failed_entries),
                "entries_written": result.entries_written,
            }
        else:
            if inputs is not None:
                fail(
                    "--inputs is not supported with --kind (ops declare inputs in Params)."
                )
            if tracks_run_id is not None or labels_run_id is not None:
                fail(
                    "--tracks-run-id / --labels-run-id are not supported with --kind; "
                    "an op produces these rather than reading them."
                )
            op_kind = cast("str", kind)
            from mosaic.core.pipeline.ops import run_op
            from mosaic.tracking import register_ops

            register_ops()
            log(f"[mosaic] execution_id={exec_id} running {op_kind}")
            with stdout_to_stderr():
                run_id = run_op(
                    ds,
                    op_kind,
                    params_dict or {},
                    scope=scope,
                    overwrite=overwrite,
                    republish=republish,
                    execution_id=exec_id,
                    owner=owner,
                    cancel_token=token,
                )
            payload = {
                "execution_id": exec_id,
                "kind": op_kind,
                "run_id": run_id,
                **attempt_facts(ds, exec_id),
            }
    except Cancelled:
        if as_json:
            emit_json({"execution_id": exec_id, "status": "cancelled"})
        else:
            log(f"[mosaic] cancelled {exec_id}")
        raise typer.Exit(code=130) from None
    except Refusal as refusal:
        # A step declining to run, or a tracker refusing a file's codec. A
        # reserved exit code rather than a new terminal status: that set is read
        # by three repositories and mosaic-api's sweeper reaps it, so the ledger
        # row stays ``failed`` and the reason is recorded in ``error_json``, which
        # this attempt's run-log already contains.
        exit_refused(exec_id, refusal, step_id=step or "", as_json=as_json)
    except ScopeRefused as refusal:
        # The checker owns the sentence; this command owns the flags that answer it.
        message = with_command_line_scope(str(refusal), RUN_SCOPE_FLAGS_REMEDY)
        fail_classified(refusal, as_json=as_json, message=message, step=step or "")
    except ClassifiedError as error:
        # Before the arms below, whose bases a classified error may also have,
        # so that --json reports its kind.
        fail_classified(error, as_json=as_json, step=step or "")
    except KeyError as exc:
        fail(str(exc))
    except ImportError as exc:
        fail(
            f"Missing optional dependency for this operation: {exc}. "
            "Install the matching extra (e.g. pip install "
            "'mosaic-behavior[deep-learning]')."
        )
    except FileNotFoundError as exc:
        fail(str(exc))
    except ValidationError as exc:
        fail(f"Invalid params: {terse(exc)}")
    except ValueError as exc:
        # An invalid input chain (a Result that is not track-shaped), and
        # anything else raised as a plain ValueError. The two ValueError
        # subclasses this command answers in its own words -- ValidationError
        # and ScopeRefused -- are caught above. This arm takes the rest, and
        # renders it as a message rather than a traceback.
        fail(str(exc))

    if as_json:
        emit_json(payload)
    else:
        render_kv(payload)
