"""``mosaic agent``: the guide and the MCP server for an AI agent that uses mosaic.

``guide`` prints the guide shipped with the installed mosaic, then the paths of
this installation. ``install`` points a project's agents at that command, which
keeps the guide they read matched to the mosaic they run, and with ``--mcp``
registers the MCP server in the project's ``.mcp.json``. ``serve`` runs that
server for one client over standard input and output. ``confirm-hook`` answers
the server's confirmations from a Claude Code ``Elicitation`` hook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from mosaic.agent import (
    AGENT_FILES,
    BLOCK_BEGIN,
    BLOCK_END,
    MCP_CONFIG,
    MCP_SERVER_NAME,
    SITE_URL,
    SKILL_DIRECTORY,
    Installation,
    find_installation,
    guide_text,
    install_agent_files,
)
from mosaic.cli._io import emit_json, fail
from mosaic.user_paths import user_path

if TYPE_CHECKING:
    from mosaic.agent.mcp.config import CleanupClass, ServerConfig

agent_app = typer.Typer(
    name="agent",
    help="Print or install the guide for an AI agent that uses mosaic, or serve it MCP tools.",
    no_args_is_help=True,
    add_completion=False,
)


@agent_app.command("guide")
def guide_command() -> None:
    """Print the agent guide for this version, then this installation's paths."""
    try:
        guide = guide_text()
    except FileNotFoundError as exc:
        fail(f"The agent guide is missing from this installation: {exc}")
    installation = find_installation(Path.cwd())
    typer.echo(guide.rstrip())
    typer.echo("")
    typer.echo("\n".join(_installation_lines(installation)))


def _installation_lines(installation: Installation) -> list[str]:
    executable = (
        str(installation.executable)
        if installation.executable is not None
        else "not found; use the interpreter with -m mosaic"
    )
    lines = [
        "## This installation",
        "",
        f"- mosaic executable: {executable}",
        f"- Python interpreter: {installation.interpreter}",
    ]
    if installation.docs is not None:
        lines += [
            f"- Documentation: {installation.docs}",
            f"  A page {SITE_URL}<page>/ is {installation.docs}/<page>.md here, or",
            f"  {installation.docs}/<page>/index.md for a section.",
        ]
    else:
        lines.append(f"- Documentation: {SITE_URL}")
    if installation.manifest is not None:
        lines.append(
            f"- Dataset: {installation.manifest.parent} "
            f"(manifest {installation.manifest})"
        )
    else:
        lines.append("- Dataset: the working directory is not inside one")
    return lines


@agent_app.command("install")
def install_command(
    directory: Annotated[
        Path,
        typer.Argument(help="Project directory. Defaults to the working directory."),
    ] = Path(),
    no_memory: Annotated[
        bool,
        typer.Option(
            "--no-memory",
            help="Write the skill only, and leave CLAUDE.md and AGENTS.md alone.",
        ),
    ] = False,
    mcp: Annotated[
        bool,
        typer.Option(
            "--mcp",
            help="Also register `mosaic agent serve` in the project's .mcp.json.",
        ),
    ] = False,
    datasets: Annotated[
        list[Path] | None,
        typer.Option(
            "--dataset",
            help=(
                "With --mcp, a dataset the server serves: its manifest or its "
                "directory. Repeatable. Defaults to the datasets in and below the "
                "project directory, else the one enclosing it."
            ),
        ),
    ] = None,
) -> None:
    """Point a project's AI coding agents at `mosaic agent guide`.

    Writes .claude/skills/mosaic/SKILL.md, and a marked block in CLAUDE.md and
    AGENTS.md (creating CLAUDE.md when neither exists). With --mcp, also writes
    the `mosaic` server entry of .mcp.json, which Claude Code reads, and keeps
    the file's other servers. Running it again replaces what it wrote.
    """
    target = user_path(directory).resolve()
    if not target.is_dir():
        fail(f"Directory not found: {target}")
    if datasets and not mcp:
        fail("--dataset names the datasets of the MCP server; add --mcp.")
    installation = find_installation(Path.cwd())
    try:
        written = install_agent_files(
            target,
            installation,
            memory=not no_memory,
            mcp_datasets=_served_datasets(target, datasets or []) if mcp else None,
        )
    except ValueError as exc:
        fail(f"install failed: {exc}")

    for entry in written:
        typer.echo(f"{entry.action}: {entry.path}")
    removal = f"To remove, delete {target / SKILL_DIRECTORY}"
    blocks = _files_with_block(target)
    if blocks:
        names = ", ".join(str(path) for path in blocks)
        removal += f" and the lines from {BLOCK_BEGIN} to {BLOCK_END} in {names}"
    if mcp:
        removal += f", and the {MCP_SERVER_NAME} entry of {target / MCP_CONFIG}"
    typer.echo("")
    typer.echo(f"{removal}.")


def _served_datasets(directory: Path, given: list[Path]) -> list[Path]:
    """Return the datasets that the server entry names, as absolute paths.

    *given* when it names any, else the datasets in and below *directory*, else
    the dataset enclosing it.
    """
    from mosaic.agent.mcp.local import discovered_manifests

    if given:
        return [user_path(path).resolve() for path in given]
    found = discovered_manifests(directory)
    if found:
        return [path.resolve() for path in found]
    enclosing = find_installation(directory).manifest
    if enclosing is not None:
        return [enclosing]
    fail(f"No dataset found in, below or above {directory}. Name one with --dataset.")


def _files_with_block(directory: Path) -> list[Path]:
    """The agent memory files in *directory* that hold a pointer block.

    An earlier install may have written one that this run left alone.
    """
    paths = [directory / name for name in AGENT_FILES]
    return [
        path
        for path in paths
        if path.is_file() and BLOCK_BEGIN in path.read_text(encoding="utf-8")
    ]


@agent_app.command("serve")
def serve_command(
    datasets: Annotated[
        list[Path] | None,
        typer.Option(
            "--dataset",
            help="A dataset to serve: its manifest or its directory. Repeatable.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=(
                "A YAML file listing `datasets`, `discover` directories, "
                "`allow_managed_writes`, `allow_cleanup`, the `cleanup` policy, "
                "the parallelism limits `max_workers`, `allow_process_mode` "
                "and `min_free_memory_fraction`, and the `dataset_roots` and "
                "`source_roots` that manage_dataset may write inside. "
                "Its relative paths are read from its directory."
            ),
        ),
    ] = None,
    allow_managed_writes: Annotated[
        bool,
        typer.Option(
            "--allow-managed-writes",
            help=(
                "Let submit_pipeline and cancel_request change a dataset whose "
                "manifest names a managing service."
            ),
        ),
    ] = False,
    allow_cleanup: Annotated[
        list[str] | None,
        typer.Option(
            "--allow-cleanup",
            help=(
                "Let submit_pipeline remove superseded feature runs with cleanup "
                "'superseded', and with retire the runs that a retired recipe "
                "wrote, after the user confirms. Repeatable; the one class is "
                "feature-runs. The policy is the config file's cleanup."
            ),
        ),
    ] = None,
    http: Annotated[
        str | None,
        typer.Option(
            "--http",
            metavar="HOST:PORT",
            help=(
                "Serve one user over streamable HTTP at http://HOST:PORT/mcp "
                "instead of stdio, for a trusted network such as a tailnet. "
                "Needs --token-file. Write an IPv6 host in brackets."
            ),
        ),
    ] = None,
    token_file: Annotated[
        Path | None,
        typer.Option(
            "--token-file",
            help=(
                "With --http: a file of mode 0600 holding the bearer token that "
                "every request must send as 'Authorization: Bearer <token>'."
            ),
        ),
    ] = None,
    allowed_hosts: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-host",
            help=(
                "With --http: a host name clients reach the server by, such as "
                "its tailnet name. Requests naming another host are refused. "
                "Repeatable. Without it, a non-loopback address checks no host."
            ),
        ),
    ] = None,
    tls_cert: Annotated[
        Path | None,
        typer.Option(
            "--tls-cert",
            help="With --http and --tls-key: serve HTTPS with this certificate.",
        ),
    ] = None,
    tls_key: Annotated[
        Path | None,
        typer.Option("--tls-key", help="The private key of --tls-cert."),
    ] = None,
    allow_model_confirm: Annotated[
        bool,
        typer.Option(
            "--allow-model-confirm",
            help=(
                "With --http: accept confirm true from a client that cannot ask "
                "the user, so its model confirms costly work itself. Without it "
                "such a call is refused."
            ),
        ),
    ] = False,
) -> None:
    """Serve mosaic's MCP tools, resources and prompts to one client over stdio.

    An MCP client starts this command and speaks JSON-RPC on its standard input
    and output, so the command writes its own messages to standard error. It
    needs the `mcp` extra. `mosaic agent install --mcp` registers it for Claude
    Code.

    With --http it serves one user over streamable HTTP instead, behind a bearer
    token, for a client on another machine of a trusted network. It does not
    encrypt unless given --tls-cert and --tls-key; put it on a tailnet or behind a
    reverse proxy that does.
    """
    from mosaic.agent.mcp.config import (
        CLEANUP_FEATURE_RUNS,
        ServerConfig,
        ServerConfigInvalid,
        read_server_config,
        served_manifests,
    )
    from mosaic.optional_dependency import MissingOptionalDependency

    try:
        settings = (
            read_server_config(user_path(config))
            if config is not None
            else ServerConfig()
        )
    except ServerConfigInvalid as exc:
        fail(str(exc))
    unknown = sorted(set(allow_cleanup or []) - {CLEANUP_FEATURE_RUNS})
    if unknown:
        fail(f"--allow-cleanup takes {CLEANUP_FEATURE_RUNS}, not {', '.join(unknown)}.")
    manifests = served_manifests([user_path(path) for path in datasets or []], settings)
    if not manifests:
        fail("No dataset to serve. Name one with --dataset or in a --config file.")
    granted: frozenset[CleanupClass] = frozenset(
        [
            *([CLEANUP_FEATURE_RUNS] if allow_cleanup else []),
            *settings.allow_cleanup,
        ]
    )
    if http is not None:
        _serve_http(
            manifests,
            http,
            token_file=token_file,
            allowed_hosts=allowed_hosts or [],
            tls_cert=tls_cert,
            tls_key=tls_key,
            allow_model_confirm=allow_model_confirm,
            allow_managed=allow_managed_writes or settings.allow_managed_writes,
            allow_cleanup=granted,
            settings=settings,
        )
        return
    http_only = [
        flag
        for flag, given in (
            ("--token-file", token_file is not None),
            ("--allowed-host", bool(allowed_hosts)),
            ("--tls-cert", tls_cert is not None),
            ("--tls-key", tls_key is not None),
            ("--allow-model-confirm", allow_model_confirm),
        )
        if given
    ]
    if http_only:
        fail(f"{', '.join(http_only)} applies with --http only.")
    from mosaic.agent.mcp.server import serve

    try:
        serve(
            manifests,
            allow_managed=allow_managed_writes or settings.allow_managed_writes,
            allow_cleanup=granted,
            cleanup_policy=settings.cleanup,
            parallel_limits=settings.parallel_limits,
            dataset_roots=settings.dataset_roots,
            source_roots=settings.source_roots,
        )
    except MissingOptionalDependency as exc:
        fail(str(exc))


def _serve_http(
    manifests: list[Path],
    http: str,
    *,
    token_file: Path | None,
    allowed_hosts: list[str],
    tls_cert: Path | None,
    tls_key: Path | None,
    allow_model_confirm: bool,
    allow_managed: bool,
    allow_cleanup: frozenset[CleanupClass],
    settings: ServerConfig,
) -> None:
    """Check the HTTP options and serve, or exit with what is wrong."""
    from mosaic.agent.mcp.http_auth import (
        TokenFileInvalid,
        parse_http_address,
        read_token_file,
    )
    from mosaic.optional_dependency import MissingOptionalDependency

    try:
        address = parse_http_address(http)
    except ValueError as exc:
        fail(f"--http {exc}")
    if token_file is None:
        fail(
            "--http needs --token-file: a file of mode 0600 holding the token "
            "that every request must send."
        )
    if (tls_cert is None) != (tls_key is None):
        fail("--tls-cert and --tls-key go together.")
    for flag, given in (("--tls-cert", tls_cert), ("--tls-key", tls_key)):
        if given is not None and not user_path(given).is_file():
            fail(f"{flag} {given} is not a file.")
    try:
        token = read_token_file(user_path(token_file))
    except TokenFileInvalid as exc:
        fail(str(exc))
    from mosaic.agent.mcp.server import serve_http

    try:
        serve_http(
            manifests,
            address=address,
            token=token,
            allowed_hosts=allowed_hosts,
            tls_cert=None if tls_cert is None else user_path(tls_cert),
            tls_key=None if tls_key is None else user_path(tls_key),
            allow_model_confirm=allow_model_confirm,
            allow_managed=allow_managed,
            allow_cleanup=allow_cleanup,
            cleanup_policy=settings.cleanup,
            parallel_limits=settings.parallel_limits,
            dataset_roots=settings.dataset_roots,
            source_roots=settings.source_roots,
        )
    except MissingOptionalDependency as exc:
        fail(str(exc))


@agent_app.command("confirm-hook")
def confirm_hook_command(
    approve: Annotated[
        list[str] | None,
        typer.Option(
            "--approve",
            help=(
                "A confirmation condition to accept, such as gpu_lane, "
                "training_op, tracking_op, allow_partial, bind, open_duplicate, "
                "cleanup, score_sealed, hard_cancel, create_dataset or "
                "add_source. Repeatable."
            ),
        ),
    ] = None,
    otherwise: Annotated[
        Literal["ask", "cancel", "decline"],
        typer.Option(
            "--otherwise",
            help=(
                "What to answer a confirmation with a condition not approved: "
                "decline it, cancel it, or ask (print nothing and leave it to "
                "the dialog). decline keeps any other hook from accepting it."
            ),
        ),
    ] = "decline",
    server: Annotated[
        str,
        typer.Option(
            "--server", help="The name under which the client lists mosaic's server."
        ),
    ] = MCP_SERVER_NAME,
) -> None:
    """Answer a confirmation of mosaic's MCP server as a Claude Code Elicitation hook.

    Reads the hook's JSON input on standard input. A confirmation whose
    conditions are all approved is accepted. Any other is answered as
    --otherwise says. Another server's elicitation, or input that is not a
    confirmation, prints nothing and is left to the client's dialog.
    """
    from mosaic.agent.confirm_hook import answer_elicitation
    from mosaic.agent.mcp.spec import CONFIRMATION_CONDITIONS
    from mosaic.core.narrowing import is_str_mapping

    approved = set(approve or [])
    unknown = sorted(approved - set(CONFIRMATION_CONDITIONS))
    if unknown:
        fail(
            f"Unknown confirmation condition(s): {', '.join(unknown)}. Known: "
            f"{', '.join(CONFIRMATION_CONDITIONS)}."
        )
    try:
        hook_input: object = json.loads(sys.stdin.read())
    except ValueError:
        return
    if not is_str_mapping(hook_input):
        return
    answer = answer_elicitation(
        hook_input,
        [condition for condition in CONFIRMATION_CONDITIONS if condition in approved],
        server=server,
        otherwise=otherwise,
    )
    if answer is not None:
        emit_json(answer)
