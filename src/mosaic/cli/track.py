"""``mosaic track <kind>``: run an integrated tracker with flags instead of JSON.

``mosaic run --kind trex --params '{...}'`` can already do this, and is what the
executor shells out to. This is the same path with the params spelled as flags,
because a person typing a tracker invocation should not have to hand-write JSON.

**The flags are derived, never declared.** They come from the op's Pydantic
``Params`` schema, so a tracker that adds a knob gets a flag for it, and one that
renames a knob renames the flag, without this file being touched. That is the
whole reason ``mosaic trex`` is gone: it was 211 hand-wired lines that could
drift from the params they mirrored, and had no equivalent for SLEAP or
Lightning Pose.

**Placement is not a flag.** Which conda environment or binary a tool lives at is
a property of the machine, so it is read from ``MOSAIC_<TOOL>_CONDA_ENV`` /
``MOSAIC_<TOOL>_BIN`` (and ``MOSAIC_TREX_DISPLAY``) rather than typed here. A run
identifier must not depend on where the tool was installed, and a queued job's
recorded params must not carry a path that is valid on exactly one machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from mosaic.cli._context import attempt_facts, load_dataset
from mosaic.cli._io import (
    emit_json,
    fail,
    log,
    parse_entries,
    stdout_to_stderr,
    terse,
)
from mosaic.cli._render import render_kv
from mosaic.core.scope import Scope

__all__ = ["track_command", "tracker_kinds"]


def tracker_kinds() -> list[str]:
    """Every registered op that is an integrated tracker, sorted.

    Read from the roots table rather than a list here, so a tracker that lands
    is runnable through this command on the day its row does.
    """
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
    from mosaic.tracking import register_ops

    register_ops()
    from mosaic.core.pipeline.ops import OPS

    return sorted(
        key
        for key, root in TRACKING_ROOTS.items()
        if root.retention == "tracker" and key in OPS
    )


def _read_value(raw: str) -> object:
    """Read one ``--set key=value`` token.

    JSON first, so a number, a list, a mapping or a null can be given verbatim;
    a bare word that is not valid JSON stays the string it looks like, because
    quoting every string on a command line would be its own annoyance. Pydantic
    validates the result against the field's declared type, so a value of the
    wrong shape is refused by name rather than coerced here.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def track_command(
    kind: Annotated[
        str,
        typer.Argument(help="Which tracker to run, e.g. 'trex', 'sleap', 'litpose'."),
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    set_: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help=(
                "A tracker parameter, as key=value (repeatable). Values are read "
                "as JSON when they parse as JSON, else as a string. Run "
                "'mosaic tracking describe <kind>' for the available keys."
            ),
        ),
    ] = None,
    groups: Annotated[
        list[str] | None, typer.Option("--groups", help="Scope to these groups.")
    ] = None,
    sequences: Annotated[
        list[str] | None, typer.Option("--sequences", help="Scope to these sequences.")
    ] = None,
    entries: Annotated[
        list[str] | None,
        typer.Option(
            "--entries",
            help="Scope to group:sequence pairs (repeatable). A bare token is a "
            "sequence in the empty group.",
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    convert_to_tracks: Annotated[
        bool, typer.Option("--convert-to-tracks/--no-convert-to-tracks")
    ] = True,
    idle_timeout: Annotated[
        float,
        typer.Option(
            "--idle-timeout",
            help="Kill a phase after this many seconds with no output from the tool.",
        ),
    ] = 900,
    max_runtime: Annotated[
        float | None,
        typer.Option("--max-runtime", help="Optional absolute wall-clock ceiling."),
    ] = None,
    owner: Annotated[str, typer.Option("--owner")] = "",
    execution_id: Annotated[
        str | None,
        typer.Option("--execution-id", help="Reuse an externally minted ULID."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit one JSON object on stdout.")
    ] = False,
) -> None:
    """Run an integrated tracker over scoped videos, bridging results into tracks/."""
    from mosaic.core.pipeline._utils import new_execution_id
    from mosaic.core.pipeline.ops import OPS, run_op
    from mosaic.core.pipeline.job import CancelToken, Cancelled, install_signal_handler

    known = tracker_kinds()
    if kind not in known:
        fail(f"Unknown tracker {kind!r}; registered trackers are {', '.join(known)}.")

    ds = load_dataset(manifest)
    fields = OPS[kind].Params.model_json_schema().get("properties", {})

    # --groups and --sequences are a cross product, and params take one entry
    # list. A group named with no sequence means every sequence in it, which
    # only the media index can answer. A missing index and an --entries named
    # beside either of the other two both become a message rather than a
    # traceback, the way every other failure of this command does.
    try:
        scope = Scope(
            entries=parse_entries(entries) or None,
            groups=groups or None,
            sequences=sequences or None,
        )
    except ValidationError as exc:
        fail(f"{kind} run failed: {terse(exc)}")
    try:
        resolved = ds.resolve_scope(scope)
    except FileNotFoundError as exc:
        fail(f"{kind} run failed: {exc}")

    params: dict[str, object] = {
        "entries": resolved.op_entries,
        "overwrite": overwrite,
        "convert_to_tracks": convert_to_tracks,
        "idle_timeout": idle_timeout,
        "max_runtime": max_runtime,
    }
    for token in set_ or []:
        key, separator, value = token.partition("=")
        if not separator:
            fail(f"Invalid --set value {token!r}; expected key=value.")
        if key not in fields:
            available = ", ".join(sorted(fields))
            fail(f"{kind} has no parameter {key!r}. Available: {available}.")
        params[key] = _read_value(value)

    exec_id = execution_id or new_execution_id()
    token_ = CancelToken()
    install_signal_handler(token_)
    log(f"[mosaic] execution_id={exec_id} running {kind}")

    try:
        with stdout_to_stderr():
            run_id = run_op(
                ds, kind, params, execution_id=exec_id, owner=owner, cancel_token=token_
            )
    except Cancelled:
        if as_json:
            emit_json({"execution_id": exec_id, "status": "cancelled"})
        else:
            log(f"[mosaic] cancelled {exec_id}")
        raise typer.Exit(code=130) from None
    except (ImportError, FileNotFoundError) as exc:
        fail(f"{kind} run failed: {exc}")

    payload: dict[str, object] = {
        "execution_id": exec_id,
        "kind": kind,
        "run_id": run_id,
        **attempt_facts(ds, exec_id),
    }
    if as_json:
        emit_json(payload)
    else:
        render_kv(payload)
