"""``mosaic run``: execute a feature or tracking op as a tracked Job-Contract attempt.

This command is the executor's *unit of work* -- the Layer-2 executor shells out
to ``mosaic run --json`` in its own process group. It pre-mints the ULID
``execution_id`` so it can be printed up front (and injected via ``--execution-id``
by the executor), installs a SIGTERM/SIGINT -> cooperative-cancel handler, and
prints ``{execution_id, feature|kind, run_id, status, cache_hit}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from mosaic.cli._context import load_dataset
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
        typer.Option("--kind", help="Tracking-op kind to run, e.g. 'infer-pose'."),
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
    """Run a feature (--feature) or a tracking op (--kind) under the Job Contract."""
    if (feature is None) == (kind is None):
        fail("Provide exactly one of --feature or --kind.")

    from pydantic import ValidationError

    from mosaic.core.pipeline._utils import new_execution_id
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
        if feature is not None:
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
                    execution_id=exec_id,
                    owner=owner,
                    cancel_token=token,
                )
            payload = {
                "execution_id": result.execution_id,
                "feature": result.feature,
                "run_id": result.run_id,
                "status": "finished",
                "cache_hit": result.cache_hit,
            }
        else:
            if entries:
                fail("--entries is not supported with --kind; put scope in --params.")
            if inputs is not None:
                fail(
                    "--inputs is not supported with --kind (ops declare inputs in Params)."
                )
            op_kind = cast("str", kind)
            from mosaic.tracking import run_tracking_op

            log(f"[mosaic] execution_id={exec_id} running {op_kind}")
            with stdout_to_stderr():
                run_id = run_tracking_op(
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
                "status": "finished",
                "cache_hit": None,
            }
    except Cancelled:
        if as_json:
            emit_json({"execution_id": exec_id, "status": "cancelled"})
        else:
            log(f"[mosaic] cancelled {exec_id}")
        raise typer.Exit(code=130) from None
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

    if as_json:
        emit_json(payload)
    else:
        render_kv(payload)
