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

from mosaic.cli._context import attempt_facts, load_dataset
from mosaic.cli._features import build_feature
from mosaic.cli._io import (
    emit_json,
    fail,
    load_json_arg,
    log,
    parse_entries,
    stdout_to_stderr,
)
from mosaic.cli._render import render_kv


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
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind", help="Op kind to run, e.g. 'infer-pose' or 'transcode'."
        ),
    ] = None,
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
    params: Annotated[
        str | None,
        typer.Option(
            "--params", help="Params as inline JSON, @file.json, or @- (stdin)."
        ),
    ] = None,
    inputs: Annotated[
        str | None,
        typer.Option(
            "--inputs",
            help='Feature inputs as JSON (default ["tracks"]). Feature runs only.',
        ),
    ] = None,
    entries: Annotated[
        list[str] | None,
        typer.Option(
            "--entries",
            help="Restrict to group:sequence (repeatable). Feature runs only.",
        ),
    ] = None,
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
    if (graph_request is not None) != (step is not None):
        fail("--graph-request and --step are used together, or not at all.")

    from pydantic import ValidationError

    from mosaic.core.pipeline._utils import new_execution_id
    from mosaic.core.pipeline.graph import REFUSED_EXIT_CODE, StepRefused
    from mosaic.core.pipeline.job import CancelToken, Cancelled, install_signal_handler

    ds = load_dataset(manifest)

    params_value = load_json_arg(params)
    params_dict: dict[str, object] | None = None
    if params_value is not None:
        if not isinstance(params_value, dict):
            fail("--params must be a JSON object.")
        params_dict = cast("dict[str, object]", params_value)

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
                )
            payload = {
                "execution_id": exec_id,
                "request_id": request.request_id,
                "step": outcome.step_id,
                "run_id": outcome.run_id,
                "status": "partial" if outcome.failed_entries else "finished",
                "state": outcome.state,
                "cache_hit": outcome.state == "cached",
                "covered": outcome.covered,
                "target": outcome.target,
                "failed_entries": list(outcome.failed_entries),
            }
        elif feature is not None:
            entry_pairs = parse_entries(entries)
            feat = build_feature(feature, load_json_arg(inputs), params_dict)
            from mosaic.core.pipeline.run import run_feature

            log(f"[mosaic] execution_id={exec_id} running {feature}")
            with stdout_to_stderr():
                result = run_feature(
                    ds,
                    feat,
                    entries=entry_pairs or None,
                    overwrite=overwrite,
                    tracks_run_id=tracks_run_id,
                    labels_run_id=labels_run_id,
                    execution_id=exec_id,
                    owner=owner,
                    cancel_token=token,
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
                "status": "partial" if result.failed_entries else "finished",
                "cache_hit": result.cache_hit,
                "failed_entries": list(result.failed_entries),
                "entries_written": result.entries_written,
            }
        else:
            if entries:
                fail("--entries is not supported with --kind; put scope in --params.")
            if inputs is not None:
                fail(
                    "--inputs is not supported with --kind (ops declare inputs in Params)."
                )
            if tracks_run_id is not None or labels_run_id is not None:
                fail(
                    "--tracks-run-id / --labels-run-id are not supported with --kind; "
                    "an op produces these rather than reading them."
                )
            if overwrite:
                # Refused rather than ignored. An op decides reuse from its own
                # markers and ``run_op`` takes no overwrite at all, so accepting
                # the flag promised a recompute that never happened.
                fail(
                    "--overwrite is not supported with --kind; an op decides reuse "
                    "from its own markers. Clear its run root to recompute it."
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
    except StepRefused as refusal:
        # A reserved exit code rather than a new terminal status: that set is
        # read by three repositories and mosaic-api's sweeper reaps it, so the
        # ledger row stays ``failed`` and the reason travels in ``error_json``,
        # which this attempt's run-log already carries.
        if as_json:
            emit_json(
                {
                    "execution_id": exec_id,
                    "status": "refused",
                    "reason": refusal.reason,
                    "step": refusal.step_id,
                    "error_json": refusal.error_json(),
                }
            )
        else:
            log(f"[mosaic] refused ({refusal.reason}): {refusal}")
        raise typer.Exit(code=REFUSED_EXIT_CODE) from None
    except KeyError as exc:
        fail(str(exc))
    except ImportError as exc:
        fail(
            f"Missing optional dependency for this operation: {exc}. "
            "Install the matching extra (e.g. pip install 'mosaic-behavior[pose]')."
        )
    except FileNotFoundError as exc:
        fail(str(exc))
    except ValidationError as exc:
        fail(f"Invalid params: {exc}")
    except ValueError as exc:
        # e.g. an invalid input chain (a Result that isn't track-shaped). Present
        # it cleanly rather than as a traceback. (ValidationError, a ValueError
        # subclass, is handled above, so this only catches plain ValueErrors.)
        fail(str(exc))

    if as_json:
        emit_json(payload)
    else:
        render_kv(payload)
