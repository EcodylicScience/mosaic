"""``mosaic trex``: run TREx (convert + track) over scoped videos as a tracked job.

Hand-wired because ``run_trex`` has no Pydantic Params model, so it can't ride
the schema-driven ``run --kind`` path. Options map 1:1 to
``mosaic.tracking.run_trex``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from mosaic.cli._context import load_dataset
from mosaic.cli._io import (
    emit_json,
    fail,
    load_json_arg,
    log,
    parse_entries,
    stdout_to_stderr,
)
from mosaic.cli._render import render_kv


def trex_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to the dataset manifest (dataset.yaml)."
        ),
    ],
    groups: Annotated[
        list[str] | None, typer.Option("--groups", help="Scope to these groups.")
    ] = None,
    sequences: Annotated[
        list[str] | None, typer.Option("--sequences", help="Scope to these sequences.")
    ] = None,
    entries: Annotated[
        list[str] | None,
        typer.Option("--entries", help="Scope to group:sequence pairs (repeatable)."),
    ] = None,
    detect_model: Annotated[
        str | None,
        typer.Option("--detect-model", help="Weights path or a prior training run_id."),
    ] = None,
    detect_type: Annotated[str, typer.Option("--detect-type")] = "yolo",
    detect_conf_threshold: Annotated[
        float, typer.Option("--detect-conf-threshold")
    ] = 0.5,
    detect_iou_threshold: Annotated[
        float, typer.Option("--detect-iou-threshold")
    ] = 0.1,
    cm_per_pixel: Annotated[float, typer.Option("--cm-per-pixel")] = 1.0,
    meta_encoding: Annotated[str, typer.Option("--meta-encoding")] = "gray",
    track_max_individuals: Annotated[int, typer.Option("--track-max-individuals")] = 1,
    track_max_speed: Annotated[float, typer.Option("--track-max-speed")] = 80.0,
    track_max_reassign_time: Annotated[
        float, typer.Option("--track-max-reassign-time")
    ] = 2.0,
    track_trusted_probability: Annotated[
        float, typer.Option("--track-trusted-probability")
    ] = 0.1,
    analysis_range: Annotated[
        str | None,
        typer.Option(
            "--analysis-range", help="Two comma-separated frame indices, e.g. 0,1000."
        ),
    ] = None,
    visual_identification_model_path: Annotated[
        str | None, typer.Option("--visual-identification-model-path")
    ] = None,
    auto_train: Annotated[bool, typer.Option("--auto-train/--no-auto-train")] = False,
    timeout: Annotated[
        int, typer.Option("--timeout", help="Per-video subprocess timeout (seconds).")
    ] = 600,
    trex_conda_env: Annotated[str | None, typer.Option("--trex-conda-env")] = None,
    trex_bin: Annotated[str | None, typer.Option("--trex-bin")] = None,
    display: Annotated[str | None, typer.Option("--display")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    convert_to_tracks: Annotated[
        bool, typer.Option("--convert-to-tracks/--no-convert-to-tracks")
    ] = True,
    convert_extra_settings: Annotated[
        str | None,
        typer.Option(
            "--convert-extra-settings", help="Extra convert settings as JSON or @file."
        ),
    ] = None,
    track_extra_settings: Annotated[
        str | None,
        typer.Option(
            "--track-extra-settings", help="Extra track settings as JSON or @file."
        ),
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
    """Run TREx over scoped videos and (by default) bridge results into tracks/."""
    from mosaic.core.pipeline._utils import new_execution_id
    from mosaic.core.pipeline.job import CancelToken, Cancelled, install_signal_handler

    ds = load_dataset(manifest)
    a_range = _parse_range(analysis_range)
    conv_extra = _as_dict(
        load_json_arg(convert_extra_settings), "--convert-extra-settings"
    )
    track_extra = _as_dict(
        load_json_arg(track_extra_settings), "--track-extra-settings"
    )
    entry_pairs = parse_entries(entries)

    exec_id = execution_id or new_execution_id()
    token = CancelToken()
    install_signal_handler(token)
    log(f"[mosaic] execution_id={exec_id} running trex")

    # run_trex has untyped progress_callback/cancel_token params, so its symbol
    # type is partially unknown -- ignore at the import binding.
    from mosaic.tracking import run_trex  # pyright: ignore[reportUnknownVariableType]

    try:
        with stdout_to_stderr():
            run_id = run_trex(
                ds,
                groups=groups,
                sequences=sequences,
                entries=entry_pairs or None,
                detect_model=detect_model,
                detect_type=detect_type,
                detect_conf_threshold=detect_conf_threshold,
                detect_iou_threshold=detect_iou_threshold,
                cm_per_pixel=cm_per_pixel,
                meta_encoding=meta_encoding,
                convert_extra_settings=conv_extra,
                track_max_individuals=track_max_individuals,
                track_max_speed=track_max_speed,
                track_max_reassign_time=track_max_reassign_time,
                track_trusted_probability=track_trusted_probability,
                analysis_range=a_range,
                visual_identification_model_path=visual_identification_model_path,
                auto_train=auto_train,
                track_extra_settings=track_extra,
                timeout=timeout,
                trex_conda_env=trex_conda_env,
                trex_bin=trex_bin,
                display=display,
                overwrite=overwrite,
                convert_to_tracks=convert_to_tracks,
                execution_id=exec_id,
                owner=owner,
                cancel_token=token,
            )
    except Cancelled:
        if as_json:
            emit_json({"execution_id": exec_id, "status": "cancelled"})
        else:
            log(f"[mosaic] cancelled {exec_id}")
        raise typer.Exit(code=130) from None
    except (ImportError, FileNotFoundError) as exc:
        fail(f"TREx run failed: {exc}")

    payload: dict[str, object] = {
        "execution_id": exec_id,
        "kind": "trex",
        "run_id": run_id,
        "status": "finished",
    }
    if as_json:
        emit_json(payload)
    else:
        render_kv(payload)


def _parse_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        fail("--analysis-range must be two comma-separated integers, e.g. 0,1000.")
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        fail("--analysis-range values must be integers, e.g. 0,1000.")


def _as_dict(value: object | None, flag: str) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        fail(f"{flag} must be a JSON object.")
    return cast("dict[str, object]", value)
