"""Shared CLI I/O helpers: JSON args, stream-separated output, clean failures.

The ``--json`` contract for every command: stdout carries **exactly one** JSON
value (via :func:`emit_json`); all human/log breadcrumbs go to **stderr** (via
:func:`log`). Machine consumers (the Layer-2 executor, MCP) therefore get a
pristine stdout to parse regardless of any progress chatter.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, NoReturn, overload

import typer

from mosaic.core.entry import parse_entry_tokens
from mosaic.core.pipeline.parallelism import (
    MAX_ADMISSION_MARGIN,
    MAX_WORKERS_CEILING,
    ParallelSettings,
    default_max_workers,
    valid_admission_margin,
)
from mosaic.user_paths import user_path

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.error_kind import ClassifiedError
    from mosaic.core.pipeline.label_series import LabelRevisionRef
    from mosaic.core.pipeline.refusal import Refusal
    from mosaic.core.pipeline.summary_arguments import KeyValue
    from mosaic.core.pipeline.tag_reads import ResolvedTagTable
    from mosaic.core.scope import Scope

EntriesOption = Annotated[
    list[str] | None,
    typer.Option(
        "--entry",
        help="Narrow to a group:sequence entry (repeatable), or to a saved "
        "selection as @name, alone. Default: the whole dataset.",
    ),
]

GroupOption = Annotated[
    list[str] | None,
    typer.Option(
        "--group",
        help="Narrow to every entry of a group (repeatable). Not with --entry.",
    ),
]

TAG_GRAMMAR: Final = (
    "A value is read by its tag's type. The values of one name are alternatives "
    "and different names must all match. NAME alone names a label tag and "
    "matches the entries it is assigned to."
)
"""The sentences that every ``--tag NAME[=VALUE]`` option shares."""

TagOption = Annotated[
    list[str] | None,
    typer.Option(
        "--tag",
        help=(
            "Narrow to the entries holding a sequence tag, as NAME=VALUE "
            f"(repeatable). {TAG_GRAMMAR} With --group, keeps the selected "
            "entries of those groups. Not with --entry."
        ),
    ),
]

RunKindOption = Annotated[
    str | None,
    typer.Option("--kind", help="The op kind, e.g. 'infer-pose'."),
]

RunParamsOption = Annotated[
    str | None,
    typer.Option("--params", help="Params as inline JSON, @file.json, or @- (stdin)."),
]

RunGroupsOption = Annotated[
    list[str] | None, typer.Option("--groups", help="Scope to these groups.")
]

RunSequencesOption = Annotated[
    list[str] | None, typer.Option("--sequences", help="Scope to these sequences.")
]

RunEntriesOption = Annotated[
    list[str] | None,
    typer.Option(
        "--entries",
        help="Scope to group:sequence pairs (repeatable). A bare token is a "
        "sequence in the empty group. @name, alone, scopes to a saved "
        "selection.",
    ),
]

RunScopeOption = Annotated[
    str | None,
    typer.Option(
        "--scope",
        help="Scope as JSON: an inline object, @path.json, or @- for "
        'stdin. {"selection": "name"} alone scopes to a saved selection, and '
        '{"sequence_tags": {"NAME": ["VALUE"]}} alone selects the entries '
        "holding those sequence tags. Excludes --entries / --groups / "
        "--sequences. --params and "
        "--scope cannot both read stdin.",
    ),
]

RUN_SCOPE_FLAGS_REMEDY = (
    "--entries group:sequence (repeatable), or --groups / --sequences, which "
    "combine as a cross product, or --scope with the selector as JSON. Omit "
    "them all to cover everything."
)
"""How to name a scope with the ``mosaic run`` flags, appended to a library refusal.

``check_scope_takes`` answers in ``Scope(...)`` for the library, the graph
planner and mosaic-api alike. A person at a terminal types these flags instead.
"""

StepScopeOption = Annotated[
    list[str] | None,
    typer.Option(
        "--step-scope",
        help=(
            "Narrow one step to part of the request's scope, as STEP=SCOPE "
            'where SCOPE is a JSON selector such as {"groups": ["train"]}, '
            '{"entries": [["g", "s"]]}, {"selection": "name"} for a saved '
            'selection or {"sequence_tags": {"NAME": ["VALUE"]}} for the '
            "entries holding those sequence tags, or @file.json (repeatable). A "
            "bare @file.json holds a JSON object of STEP: SCOPE for several "
            "steps. A selection or a tag selector takes no other key."
        ),
    ),
]

ParallelChoice = Literal["thread", "process"]

WorkersOption = Annotated[
    int | None,
    typer.Option(
        "--workers",
        min=1,
        max=MAX_WORKERS_CEILING,
        help=(
            "Compute this many entries of a parallelizable step at once: a "
            "feature, convert-tracks or resample-tracks. Without it a run "
            "computes them in turn, unless --parallel-mode, "
            "--min-free-memory-fraction or --admission-margin is given, which "
            "takes one worker per physical core. More workers use more memory."
        ),
    ),
]

ParallelModeOption = Annotated[
    ParallelChoice | None,
    typer.Option(
        "--parallel-mode",
        help=(
            "With more than one worker, 'thread' (the default) shares one copy of the "
            "feature. 'process' uses every core but spawns an interpreter per "
            "worker, each holding an entry's table, so it uses far more memory."
        ),
    ),
]

MinFreeMemoryOption = Annotated[
    float,
    typer.Option(
        "--min-free-memory-fraction",
        min=0.0,
        max=0.99,
        help=(
            "With more than one worker, the share of the memory limit that applies, a "
            "container's or else the host's, that must stay available. The run "
            "waits for its in-flight entries while the share is lower, and fails "
            "when it is still lower with none in flight. 0 turns the check off."
        ),
    ),
]


AdmissionMarginOption = Annotated[
    float,
    typer.Option(
        "--admission-margin",
        min=0.0,
        max=MAX_ADMISSION_MARGIN,
        help=(
            "With more than one worker, start an entry beside the ones in flight only "
            "when its predicted memory times this factor fits the memory limit, "
            "learned from the run's own entries. The first entry runs alone and "
            "entries start largest first. 0 (the default) turns the gate off; "
            "otherwise from 1 to 4, such as 1.5."
        ),
    ),
]


def parallel_settings(
    workers: int | None,
    mode: ParallelChoice | None,
    min_free_memory_fraction: float,
    admission_margin: float,
) -> ParallelSettings | None:
    """Return the settings that the parallel flags name, or fail when they conflict.

    Args:
        workers: The ``--workers`` value.
        mode: The ``--parallel-mode`` value.
        min_free_memory_fraction: The ``--min-free-memory-fraction`` value.
        admission_margin: The ``--admission-margin`` value.

    Returns:
        The settings, or ``None`` when no flag asks for more than one worker. A
        run that names a mode, a memory floor or an admission margin and no
        count takes :func:`default_max_workers`, and is serial when that is 1.
    """
    knob_named = (
        mode is not None or min_free_memory_fraction > 0.0 or admission_margin > 0.0
    )
    if workers is not None and workers <= 1:
        if knob_named:
            fail(
                "--parallel-mode, --min-free-memory-fraction and --admission-margin "
                "apply to a run with more than one worker. --workers 1 computes "
                "entries in turn."
            )
        return None
    if not valid_admission_margin(admission_margin):
        fail(f"--admission-margin is 0 (off) or from 1 to 4, not {admission_margin}.")
    if workers is None:
        if not knob_named:
            return None
        workers = default_max_workers()
        if workers <= 1:
            return None
    return ParallelSettings(
        workers=workers,
        mode="thread" if mode is None else mode,
        min_free_memory_fraction=min_free_memory_fraction,
        admission_margin=admission_margin,
    )


def parse_gpu_cards(value: str | None) -> tuple[str, ...] | None:
    """Return the cards that a ``--gpu-cards`` value names, ``None`` when it is absent.

    Args:
        value: One comma-separated value of ``nvidia-smi`` indexes or UUIDs, as
            ``CUDA_VISIBLE_DEVICES`` writes them.

    Returns:
        The names in order, with blanks dropped. An empty tuple for a value that
        names none, which the driver refuses.
    """
    if value is None:
        return None
    return tuple(name.strip() for name in value.split(",") if name.strip())


@contextlib.contextmanager
def stdout_to_stderr() -> Generator[None]:
    """Redirect library stdout chatter (progress/completion prints) to stderr.

    Keeps the ``--json`` contract (one clean JSON value on stdout) intact no
    matter what the compute path prints. Wrap library calls, not the final
    :func:`emit_json`.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def fail(message: str, code: int = 1) -> NoReturn:
    """Print *message* to stderr and exit with *code* (default 1)."""
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


def log(message: str) -> None:
    """Emit a human/log breadcrumb to stderr (keeps stdout clean for ``--json``)."""
    typer.echo(message, err=True)


def emit_json(payload: object) -> None:
    """Emit one JSON value to stdout (the machine-readable ``--json`` output)."""
    typer.echo(json.dumps(payload, indent=2, default=str))


def fail_classified(
    error: ClassifiedError,
    *,
    as_json: bool,
    message: str | None = None,
    step: str = "",
) -> NoReturn:
    """Report a classified error, and exit 1.

    The message goes to stderr. Under ``--json`` the error is also the one JSON
    value on stdout, as its :class:`~mosaic.core.pipeline.error_envelope.ErrorEnvelope`.

    Args:
        error: The error.
        as_json: Whether ``--json`` was given.
        message: The sentence to report in place of ``str(error)``, such as one
            naming this command's flags.
        step: The step to name when *error* names none, or ``""`` when the
            step is unknown.

    Raises:
        typer.Exit: Always, with code 1.
    """
    from mosaic.core.pipeline.error_envelope import error_envelope

    envelope = error_envelope(error)
    if message is not None:
        envelope = envelope.model_copy(update={"message": message})
    if envelope.step is None and step:
        envelope = envelope.model_copy(update={"step": step})
    log(envelope.message)
    if as_json:
        emit_json(envelope.model_dump(mode="json"))
    raise typer.Exit(code=1) from None


def exit_refused(
    execution_id: str, refusal: Refusal, *, step_id: str = "", as_json: bool
) -> NoReturn:
    """Report a refusal, and exit with the code reserved for one.

    The message goes to stderr. Under ``--json`` the refusal is also the one JSON
    value on stdout: the refusal's
    :class:`~mosaic.core.pipeline.error_envelope.ErrorEnvelope`, with the
    execution, ``"status": "refused"``, the step and the ``error_json`` blob that
    a ledger row records beside it. ``step`` is ``""`` when the step is unknown.

    Args:
        execution_id: The attempt that refused.
        refusal: The refusal.
        step_id: The step to name when *refusal* does not name its own, or
            ``""`` for a run outside a pipeline.
        as_json: Whether ``--json`` was given.

    Raises:
        typer.Exit: Always, with ``REFUSED_EXIT_CODE``.
    """
    from mosaic.core.pipeline.error_envelope import error_envelope
    from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE

    log(f"[mosaic] refused ({refusal.reason}): {refusal}")
    if as_json:
        emit_json(
            {
                **error_envelope(refusal).model_dump(mode="json"),
                "execution_id": execution_id,
                "status": "refused",
                "step": refusal.step_id or step_id,
                "error_json": refusal.error_json(step_id),
            }
        )
    raise typer.Exit(code=REFUSED_EXIT_CODE) from None


def load_json_arg(value: str | None) -> object | None:
    """Resolve a ``--params``/``--inputs``-style argument to a JSON value.

    Accepts ``@path.json`` (read a file through
    :func:`~mosaic.core.pipeline.graph.read_json_file`), ``@-`` (read stdin), or
    an inline JSON string. Returns ``None`` when *value* is ``None``. JSON / file
    errors exit cleanly via :func:`fail`.
    """
    if value is None:
        return None
    if value == "@-":
        raw = sys.stdin.read()
        source = "<stdin>"
    elif value.startswith("@"):
        # No shell expands the tilde in `@~/params.json` -- expansion applies to
        # the start of a word, and here the word starts with `@` -- so this form
        # only works if it is expanded here.
        return json_file_arg(user_path(value[1:]))
    else:
        raw = value
        source = "inline JSON"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON ({source}): {exc}")


def json_file_arg(path: Path) -> object:
    """Return the JSON document in the file *path*, exiting cleanly when it does not read."""
    from mosaic.core.pipeline.graph import JsonFileUnreadable, read_json_file

    try:
        return read_json_file(path)
    except JsonFileUnreadable as exc:
        fail(str(exc))


def with_command_line_scope(refusal: str, remedy: str) -> str:
    """*refusal* with the flag spelling of its remedy appended.

    ``check_scope_takes`` is the only place a scope refusal is written, and it
    answers in ``Scope(...)`` because its callers are the library, the graph
    planner and mosaic-api, which all construct one. A person at a terminal
    types flags instead, and each command offers its own spelling of them.

    Appended rather than substituted. The sentence the checker wrote states
    what an unscoped run would cover. That is what decides whether to narrow or
    to proceed, and no flag list replaces it.

    Args:
        refusal: what :func:`~mosaic.core.pipeline.ops.check_scope_takes` said.
        remedy: the flags this command offers, as a phrase.

    Returns:
        The message to print.
    """
    return f"{refusal} At the command line: {remedy}"


def parse_entries(entries: list[str] | None) -> list[tuple[str, str]]:
    """Parse repeated ``group:sequence`` tokens into pairs.

    One grammar, in :func:`mosaic.core.entry.parse_entry_tokens`. This used to
    reject a token with no ``:`` while the ops reading the same values out of
    ``--params`` accepted it as a bare sequence in the empty group, so the two
    ways of naming one entry on one command line disagreed.
    """
    return parse_entry_tokens(entries)


def refuse_scope_in_params(params: dict[str, object]) -> None:
    """Raise unless *params* names settings alone.

    A feature's or an op's model validates ``--params``. A scope belongs to the
    attempt and arrives through the scope flags.

    Args:
        params: The ``--params`` mapping as the caller wrote it.

    Raises:
        ValueError: A key names a selector field instead of a setting.
    """
    from mosaic.core.scope import scope_in_params_refusal

    refusal = scope_in_params_refusal(params, prefix="--")
    if refusal:
        raise ValueError(refusal)


def command_line_params(params: str | None) -> dict[str, object] | None:
    """Return the ``--params`` object, or fail when it is not one or names a scope.

    Args:
        params: The ``--params`` value: inline JSON, ``@file.json`` or ``@-``.

    Returns:
        The mapping, or ``None`` when *params* is ``None``.
    """
    from pydantic import TypeAdapter, ValidationError

    value = load_json_arg(params)
    if value is None:
        return None
    try:
        mapping = TypeAdapter(dict[str, object]).validate_python(value)
    except ValidationError:
        fail("--params must be a JSON object.")
    try:
        refuse_scope_in_params(mapping)
    except ValueError as exc:
        fail(str(exc))
    return mapping


def refuse_scope_named_twice(
    entries: list[str] | None,
    groups: list[str] | None,
    sequences: list[str] | None,
    scope_json: str | None,
) -> None:
    """Fail when ``--scope`` is given beside another scope flag.

    The other flags are ``--entries``, ``--groups`` and ``--sequences``, and
    presence decides. ``--scope '{}'`` beside ``--entries`` names a selector
    twice whatever either of them contains.
    """
    if scope_json is not None and (
        entries is not None or groups is not None or sequences is not None
    ):
        fail(
            "--scope names the whole selector as JSON and cannot be combined "
            "with --entries / --groups / --sequences. Give --scope alone, or "
            "give the flags alone."
        )


TAG_SELECTOR_KEY: Final = "sequence_tags"
"""The key of a JSON scope that selects entries by sequence tag."""

_TAG_SELECTOR_FORM: Final = '{"sequence_tags": {"NAME": ["VALUE"]}}'


@dataclass(frozen=True, slots=True)
class TagChoice:
    """A selection of entries by sequence tag that no revision of the tag table has lowered.

    A command line names the tags and values wanted. The entries that hold them
    are known only once the dataset is open and a revision of its tag table is
    read.

    Attributes:
        where: The wanted values by tag name. A ``label`` takes no values.
        groups: Keep only the selected entries of these groups, or every group
            when empty.
    """

    where: Mapping[str, tuple[KeyValue, ...]]
    groups: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectionName:
    """A saved selection that a command line names and no dataset has resolved yet.

    Attributes:
        name: The selection's name, without the ``@`` of the token form.
    """

    name: str


type ScopeChoice = Scope | SelectionName | TagChoice
"""What a scope flag names: a selector, a saved selection or a tag selection.

A saved selection and a tag selection name entries only once the command has
opened its dataset, and :func:`resolve_choices` resolves both.
"""


def choices_tag_table(
    ds: Dataset,
    scope: ScopeChoice | None,
    step_scopes: Mapping[str, ScopeChoice],
    *,
    revision: int | None = None,
) -> ResolvedTagTable | None:
    """Read the tag table once when any choice of a command selects by tag.

    Args:
        ds: The open dataset.
        scope: The command's scope, or ``None`` when it names none.
        step_scopes: The command's step scopes, by step id.
        revision: The revision to read, or ``None`` for the latest.

    Returns:
        The table that every tag choice is lowered against, or ``None`` when no
        choice names a tag, so that a command without a tag selection reads
        nothing.

    Raises:
        TagSelectionInvalid: No revision is indexed, or *revision* is not.
        SequenceTagTableUnavailable: The table cannot be read as one revision
            of one dataset.
    """
    if not isinstance(scope, TagChoice) and not any(
        isinstance(choice, TagChoice) for choice in step_scopes.values()
    ):
        return None
    from mosaic.core.annotations.tag_selection import selection_tag_table

    return selection_tag_table(ds, revision)


def resolve_scope_choice(
    ds: Dataset, choice: ScopeChoice, *, tags: ResolvedTagTable | None
) -> Scope:
    """Return *choice* as a scope of entries.

    Args:
        ds: The open dataset.
        choice: A scope, which is returned as it is, a saved selection, or a tag
            selection.
        tags: The table from :func:`choices_tag_table`. It is read there once, so
            that the scope and every step scope select from one revision.

    Raises:
        TagSelectionInvalid: The tag selection names an unknown tag, a value its
            type refuses, or no entry.
        SelectionNotFound: The dataset holds no saved selection of that name.
        SelectionUnreadable: The saved selection's file cannot be read.
    """
    if isinstance(choice, SelectionName):
        from mosaic.core.selections import selection_scope

        return selection_scope(ds, choice.name)
    if not isinstance(choice, TagChoice):
        return choice
    from mosaic.core.annotations.tag_selection import tagged_scope

    assert tags is not None
    return tagged_scope(tags, choice.where, groups=choice.groups or None)


def resolve_step_choices(
    ds: Dataset,
    choices: Mapping[str, ScopeChoice],
    *,
    tags: ResolvedTagTable | None,
) -> dict[str, Scope]:
    """Return each step's choice as a scope of entries, as :func:`resolve_scope_choice`."""
    return {
        step_id: resolve_scope_choice(ds, choice, tags=tags)
        for step_id, choice in choices.items()
    }


@dataclass(frozen=True, slots=True)
class ResolvedChoices:
    """A command's scope and step scopes as entries, with what they were chosen from.

    Attributes:
        scope: The command's scope, or ``None`` when it names none.
        step_scopes: Each step's scope, by step id.
        tags: The table the choices read, or ``None`` when no choice named a
            tag.
        selections: The saved selection each scope came from, keyed as
            ``Request.selections`` is: ``""`` for the command's scope and a
            step id for that step's.
    """

    scope: Scope | None
    step_scopes: dict[str, Scope]
    tags: ResolvedTagTable | None
    selections: dict[str, str]

    @property
    def tag_revision(self) -> LabelRevisionRef | None:
        """The revision the tag selections were lowered against, if any."""
        return None if self.tags is None else self.tags.ref


def resolve_choices(
    ds: Dataset,
    scope: ScopeChoice | None,
    step_scope: list[str] | None,
    *,
    as_json: bool,
    revision: int | None = None,
) -> ResolvedChoices:
    """Parse the ``--step-scope`` tokens and resolve every choice of a command.

    The tag table is read once, so that the scope and every step scope select
    from one revision, and each saved selection is read from *ds*. A refused
    choice leaves through :func:`scope_choice_refusals`.

    Args:
        ds: The open dataset.
        scope: The command's scope, or ``None`` when it names none.
        step_scope: The ``--step-scope`` tokens.
        as_json: Whether ``--json`` was given.
        revision: The revision of the tag table to lower against, or ``None`` for
            the latest.
    """
    step_choices = parse_step_scopes(step_scope)
    with scope_choice_refusals(as_json=as_json):
        tags = choices_tag_table(ds, scope, step_choices, revision=revision)
        resolved_scope = (
            None if scope is None else resolve_scope_choice(ds, scope, tags=tags)
        )
        step_scopes = resolve_step_choices(ds, step_choices, tags=tags)
    return ResolvedChoices(
        scope=resolved_scope,
        step_scopes=step_scopes,
        tags=tags,
        selections=_selection_names(scope, step_choices),
    )


def _selection_names(
    scope: ScopeChoice | None, step_choices: Mapping[str, ScopeChoice]
) -> dict[str, str]:
    """The saved selection each choice names, keyed as ``Request.selections`` is."""
    names = {
        step_id: choice.name
        for step_id, choice in step_choices.items()
        if isinstance(choice, SelectionName)
    }
    if isinstance(scope, SelectionName):
        names[""] = scope.name
    return names


@contextlib.contextmanager
def scope_choice_refusals(*, as_json: bool) -> Generator[None]:
    """Report a refused saved or tag selection as a classified error, and exit 1.

    Wrap :func:`choices_tag_table` and the resolutions that read it. A tag
    selection that the request must change, a tag table that cannot be read as
    one revision, and a saved selection that is missing or unreadable each leave
    through :func:`fail_classified`, so ``--json`` prints the error's envelope.

    Args:
        as_json: Whether ``--json`` was given.
    """
    from mosaic.core.annotations.sequence_tags import SequenceTagTableUnavailable
    from mosaic.core.annotations.tag_selection import TagSelectionInvalid
    from mosaic.core.selections import SelectionNotFound, SelectionUnreadable

    try:
        yield
    except (
        TagSelectionInvalid,
        SequenceTagTableUnavailable,
        SelectionNotFound,
        SelectionUnreadable,
    ) as error:
        fail_classified(error, as_json=as_json)


def _selection_token(tokens: list[str] | None) -> SelectionName | None:
    """Return the selection that *tokens* name, or ``None`` when none is a selection.

    A token ``@name`` names a selection, and a sequence of the empty group named
    ``@name`` is written ``:@name``. A selection token excludes every other token.
    """
    chosen = [token for token in tokens or () if token.startswith("@")]
    if not chosen:
        return None
    if len(tokens or ()) > 1:
        fail(
            f"The selection token {chosen[0]} stands alone: it excludes every "
            "other entry token."
        )
    return SelectionName(chosen[0][1:])


def _selection_object(value: Mapping[str, object], flag: str) -> SelectionName | None:
    """Return the selection that the JSON object *value* names, if it names one.

    The ``selection`` key stands alone in its object and holds a name.

    Returns:
        The selection, or ``None`` when *value* has no ``selection`` key.
    """
    if "selection" not in value:
        return None
    name = value["selection"]
    if len(value) > 1 or not isinstance(name, str):
        fail(
            f'{flag}: "selection" holds one saved selection\'s name and stands '
            "alone in its object."
        )
    return SelectionName(name)


def _tag_choice(selector: Mapping[str, object], label: str) -> TagChoice:
    """The tag selection that a JSON *selector* names, failing with *label* when it is not one."""
    from mosaic.core.narrowing import is_object_list, is_str_mapping

    others = sorted(key for key in selector if key != TAG_SELECTOR_KEY)
    if others:
        fail(
            f"{label}: {TAG_SELECTOR_KEY} stands alone in its selector and cannot "
            f"be combined with {', '.join(others)}."
        )
    named = selector[TAG_SELECTOR_KEY]
    if not is_str_mapping(named):
        fail(f"{label}: {TAG_SELECTOR_KEY} must be {_TAG_SELECTOR_FORM}.")
    where: dict[str, tuple[KeyValue, ...]] = {}
    for name, values in named.items():
        found: list[KeyValue] = []
        if not is_object_list(values):
            fail(
                f"{label}: {TAG_SELECTOR_KEY} must give the values of {name!r} as "
                "a JSON list."
            )
        for value in values:
            if not isinstance(value, str | int | float | bool):
                fail(
                    f"{label}: {TAG_SELECTOR_KEY} values of {name!r} must be text, "
                    f"numbers or booleans, not {value!r}."
                )
            found.append(value)
        where[name] = tuple(found)
    return TagChoice(where=where)


def command_line_scope(
    entries: list[str] | None,
    groups: list[str] | None,
    sequences: list[str] | None,
    scope_json: str | None,
) -> ScopeChoice:
    """Return the selector that the scope flags name, or fail with a message.

    ``--groups`` and ``--sequences`` combine as a cross product, and a group
    named with no sequence means every sequence in it, which only an index can
    answer. The caller resolves the selector: a feature against its inputs, an
    op against the media index. ``--scope`` names the whole model as JSON, for a
    caller that formats one instead of typing flags. Both sources meet one
    validation, and a bad selector reads the same however it arrived. A
    selector that addresses cameras is refused after it is built.

    A saved selection is named by ``--entries @name`` alone or by ``--scope`` as
    ``{"selection": "name"}`` alone, and comes back as a :class:`SelectionName`.
    ``--scope`` also takes ``{"sequence_tags": {NAME: [VALUE, ...]}}`` alone in
    its object, which comes back as a :class:`TagChoice`. Each names entries only
    once the dataset is open, and the caller resolves it with
    :func:`resolve_choices`.

    Args:
        entries: The ``--entries`` tokens.
        groups: The ``--groups`` names.
        sequences: The ``--sequences`` names.
        scope_json: The ``--scope`` value: inline JSON, ``@file.json`` or ``@-``.

    Returns:
        The selector, unset when every flag is absent, or the saved or tag
        selection that the flags name.
    """
    from pydantic import ValidationError

    from mosaic.core.narrowing import is_str_mapping
    from mosaic.core.scope import Scope, camera_grain_refusal
    from mosaic.core.strict_model import terse

    try:
        if scope_json is not None:
            scope_value = load_json_arg(scope_json)
            if not is_str_mapping(scope_value):
                fail("--scope must be a JSON object.")
            if (selection := _selection_object(scope_value, "--scope")) is not None:
                return selection
            if TAG_SELECTOR_KEY in scope_value:
                return _tag_choice(scope_value, "--scope")
            scope = Scope.model_validate(scope_value)
        else:
            if (selection := _selection_token(entries)) is not None:
                if groups is not None or sequences is not None:
                    fail(
                        f"The selection token @{selection.name} stands alone: it "
                        "excludes --groups and --sequences."
                    )
                return selection
            scope = Scope(
                entries=parse_entries(entries) or None,
                groups=groups or None,
                sequences=sequences or None,
            )
    except ValidationError as exc:
        fail(f"Invalid scope: {terse(exc)}")
    if refusal := camera_grain_refusal(scope):
        fail(refusal)
    return scope


def tag_wanted(tags: list[str]) -> dict[str, tuple[KeyValue, ...]]:
    """The values wanted of each tag that repeated ``--tag NAME[=VALUE]`` tokens name.

    A token without ``=`` names a ``label``, which takes no values. The text of a
    value is kept as typed: the selection reads it by its tag's type.
    """
    wanted: dict[str, list[KeyValue]] = {}
    for token in tags:
        name, separator, value = token.partition("=")
        name = name.strip()
        if not name:
            fail(f"--tag {token!r} names no tag. Give NAME=VALUE, or NAME for a label.")
        values = wanted.setdefault(name, [])
        if separator:
            values.append(value)
    return {name: tuple(values) for name, values in wanted.items()}


@overload
def entries_scope(
    entries: list[str] | None, groups: list[str] | None = None, *, tags: None = None
) -> Scope | SelectionName | None: ...


@overload
def entries_scope(
    entries: list[str] | None, groups: list[str] | None = None, *, tags: list[str]
) -> ScopeChoice | None: ...


def entries_scope(
    entries: list[str] | None,
    groups: list[str] | None = None,
    *,
    tags: list[str] | None = None,
) -> ScopeChoice | None:
    """Return repeated ``--entry``, ``--group`` or ``--tag`` tokens as a selector.

    ``None`` when none is given. ``--entry`` and ``--group`` are refused together,
    because a selector names entries or groups and not both, and ``--entry`` is
    refused beside ``--tag``. A token ``@name`` names a saved selection, alone,
    and comes back as a :class:`SelectionName`. ``--tag`` names entries only once a revision of the
    tag table is read, so it is returned as a :class:`TagChoice` for the caller to
    lower with :func:`resolve_choices` after the dataset is open. ``--group``
    beside it keeps the selected entries of those groups.

    Args:
        entries: The ``--entry`` tokens.
        groups: The ``--group`` names.
        tags: The ``--tag`` tokens, or ``None`` for a command that takes none.

    Returns:
        The selector, or ``None`` when every flag is absent.
    """
    from mosaic.core.scope import Scope

    if (selection := _selection_token(entries)) is not None:
        if groups or tags:
            fail(
                f"The selection token @{selection.name} given with --entry "
                "excludes --group and --tag."
            )
        return selection
    pairs = parse_entries(entries)
    if tags:
        if pairs:
            fail("Give --entry or --tag, not both.")
        return TagChoice(where=tag_wanted(tags), groups=tuple(groups or ()))
    if pairs and groups:
        fail("Give --entry or --group, not both.")
    if groups:
        return Scope(groups=list(groups))
    return Scope(entries=pairs) if pairs else None


def parse_step_scopes(values: list[str] | None) -> dict[str, ScopeChoice]:
    """Parse repeated ``--step-scope`` tokens into step scopes.

    A token is ``STEP=SCOPE``, where SCOPE is a JSON selector, inline or as
    ``@file.json``, or a bare ``@file.json`` holding a JSON object of ``STEP:
    SCOPE``. A token without a step, a step named twice across the tokens, or
    a value that is not a selector fails, naming the token, and for a file the
    step as ``@file.json [STEP]``. A SCOPE of ``{"selection": "name"}`` alone in
    its object is a :class:`SelectionName`, and one of ``{"sequence_tags": {NAME:
    [VALUE, ...]}}`` alone is a :class:`TagChoice`. The caller resolves each with
    :func:`resolve_choices` once the dataset is open.
    """
    from mosaic.core.narrowing import is_str_mapping

    found: dict[str, ScopeChoice] = {}
    for token in values or ():
        if token.startswith("@"):
            mapping = load_json_arg(token)
            if not is_str_mapping(mapping):
                fail(f"--step-scope {token!r} must hold a JSON object of STEP: SCOPE.")
            pairs = [
                (step_id, selector, f"{token} [{step_id}]")
                for step_id, selector in mapping.items()
            ]
        else:
            step_id, separator, value = token.partition("=")
            if not separator or not step_id.strip() or not value:
                fail(f"--step-scope {token!r} is not STEP=SCOPE.")
            pairs = [(step_id, load_json_arg(value), token)]
        for step_id, selector, named in pairs:
            step_id = step_id.strip()
            if not step_id:
                fail(f"--step-scope {named!r} names a step with no id.")
            if step_id in found:
                fail(f"--step-scope names step {step_id!r} twice.")
            found[step_id] = _step_scope(selector, named)
    return found


def parse_pins(values: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--pin STEP=RUN_ID`` tokens into a pin for each step.

    A token without a step or a run id, or a step named twice, exits 2 as a
    usage error naming the token.
    """
    found: dict[str, str] = {}
    for token in values or ():
        step_id, separator, run_id = token.partition("=")
        step_id, run_id = step_id.strip(), run_id.strip()
        if not separator or not step_id or not run_id:
            fail(f"--pin {token!r} is not STEP=RUN_ID.", code=2)
        if step_id in found:
            fail(f"--pin names step {step_id!r} twice.", code=2)
        found[step_id] = run_id
    return found


def _step_scope(selector: object, token: str) -> ScopeChoice:
    """The scope that *selector* names, failing with *token* when it names none."""
    from pydantic import ValidationError

    from mosaic.core.narrowing import is_str_mapping
    from mosaic.core.scope import Scope

    if not is_str_mapping(selector):
        fail(f"--step-scope {token!r}: SCOPE must be a JSON object.")
    if (
        selection := _selection_object(selector, f"--step-scope {token!r}")
    ) is not None:
        return selection
    if TAG_SELECTOR_KEY in selector:
        return _tag_choice(selector, f"--step-scope {token!r}")
    try:
        narrowed = Scope.model_validate(selector)
    except ValidationError as exc:
        fail(f"--step-scope {token!r} is not a scope: {exc}")
    if narrowed.addresses_cameras:
        fail(
            f"--step-scope {token!r} names cameras. A step scope narrows "
            "(group, sequence) entries."
        )
    return narrowed
