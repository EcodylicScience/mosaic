#!/usr/bin/env -S python3 -B
"""Nudge bare heavy commands to a serializing launcher, as a PreToolUse
reminder.

Nudges do not block. The reminder is added via
hookSpecificOutput.additionalContext without a permissionDecision. The command
still runs under the normal permission flow, and a false positive costs a
harmless nudge.

Two signals shape the nudge:

  * The kind of command.
      - A SUITE run goes through the recommended launcher (`remote` where
        installed, otherwise `heavy`). It is a whole test suite.
      - A BUILD run goes through `heavy`. It is a compiled build or typecheck
        (cargo, tsc, make, gradle, ...).
      - A LIGHT run is a single interpreted test narrowed to one file or node
        id (pytest `...::test_x`, a jest `*.spec.ts`, `go test -run X`, ...).
        It does not need a launcher by itself and is not offloaded. The hook
        stays silent for these by default.
      - The hook leaves every other command alone (git status, a one-file lint,
        installs).

  * Whether another session holds the global lock, read live through a
    non-blocking flock probe. The probe takes precedence over a stale `current`
    file. While a heavy job runs elsewhere, a LIGHT command gets a soft,
    optional reminder to run it under `heavy`, where its results and timing
    may be more reliable.
"""
import os
import re
import shutil
import time
from pathlib import Path
from typing import Literal

from heavy_lock import (
    lock_directory,
    lock_held_by_other,
    lock_paths,
    parse_holder_record,
    read_holder,
)
from hook_payload import read_payload
from pre_tool_use_hook import add_context, bash_command, deny
from shell_command import cut_at_operators, joined_continuations, program_name, strip_quoted

Kind = Literal["suite", "build", "light"]

# Kinds from heaviest to lightest.
KINDS_BY_WEIGHT: tuple[Kind, ...] = ("suite", "build", "light")

# This sentence is appended to every nudge that recommends `remote`. The hook
# cannot tell whether a command's writes are needed in the local tree, and
# names the alternative instead.
LOCAL_WRITES_SENTENCE = (
    "A command whose writes you need in the local tree (a formatter, a `--fix` or "
    "`--write` mode, a snapshot update, a code generator) runs under `heavy` instead."
)

# This sentence is appended to a build nudge where `remote` is installed.
LONG_RUN_SENTENCE = "A run past about 2 minutes goes to `remote`."

# --- command classification --------------------------------------------------

# Leading VAR=val assignments, which may precede an invocation. The value
# stops at a quote, a `$`, a backtick or a `(`. Command substitutions then stay
# outside it, and the launcher patterns do not blame the `remote` in
# `url=$(git remote get-url origin)` for a later trailer.
ENV_ASSIGNMENT = r"""(?:[A-Za-z_]\w*=[^\s'"$`(]*\s+)*"""

# COMMAND_POSITION matches the position of a program name that is being run: at
# the start of the command or just after a shell operator, optionally behind
# those assignments.
# A word boundary alone is not enough. `\bmake\b` also finds "make" inside a
# commit message, and `\bremote\b` also finds the subcommand of `git remote
# -v`. Anchoring on this prefix limits a match to an invocation. Newlines
# separate steps as `;` does, and multi-line Bash calls are routine (`cd <dir>`
# then the runner). The newline therefore belongs in the operator class. The
# pattern is not multi-line, and `^` alone matches only the first step.
# `(` and `{` open a subshell and a brace group, and a command starts inside
# each of them as it does after a pipe. Without them, `(heavy make) | tail` did
# not match a launcher, and the hook stayed silent. It gave neither the deny of
# the ungrouped form nor a nudge. The mask check exists to stop that shape.
# They only add start positions. The word that follows still has to be a
# launcher or a tool, and `(git remote -v) | grep origin` stays unmatched.
# The whitespace run is bounded instead of `\s*`. With `\n` among the
# operators, a plain `\s*` makes every newline in a blank-line run a start
# position. The run is swallowed from there and given back one character at a
# time, which is quadratic, and this is a PreToolUse hook on every Bash call.
# 40,000 blank lines took 41s, against 0.089s bounded. Blank-line runs are
# unaffected however long they are, because a newline is itself an operator.
# The last newline before the program is a start position with almost no
# whitespace left to cover. The bound cuts off a single unbroken run of more
# than 40 spaces or tabs between an operator and the program. That is
# indentation, and 40 columns of it is deeper than any command needs. The bound
# works on every Python version. A possessive `\s*+` needs Python 3.11, and
# regexes that this hook cannot compile fail every Bash call in the repository.
COMMAND_POSITION = r"(?:^|[;&|\n({]|&&|\|\|)\s{0,40}" + ENV_ASSIGNMENT

# LAUNCHER matches the serializing launchers. `heavy` takes the global local
# lock, and `remote` offloads to the task machine or falls back to `heavy`. The
# token must end on whitespace or end-of-string. Word boundaries do not work,
# because `\b` also matches before `=` and before `-`. The end-of-token rule
# keeps the assignment
# in `remote=origin; git fetch $remote` and the program in `remote-sync.sh`
# from reading as this launcher. The group is named to let a deny reason name
# the launcher that the session typed, because `group(0)` includes the leading
# operator and whitespace. Python rejects a repeated group name within one
# pattern, and each pattern built from this fragment embeds it exactly once.
LAUNCHER = COMMAND_POSITION + r"(?P<launcher>heavy|remote)(?=\s|$)"
LAUNCHER_RE = re.compile(LAUNCHER)

# RUNNER_PREFIX matches transparent prefixes, the wrappers that run a tool one
# or two tokens after command position. `uv run pytest` is a pytest run,
# `python -m pytest` is a pytest run, and `npx vite build` is a build. Do not
# simplify this escape away to a bare command-position anchor. These
# repositories run their suites as `uv run pytest` and `npx vitest`, and only
# the escape matches the runs that this hook exists for. The escape stays
# narrow, because the prefix itself must be in command position. Classification
# also runs on the quote-stripped skeleton, and a tool named inside a quoted
# argument does not match.
#
# Two runners are absent. `ssh host make` burns the cores of another machine,
# and a local lock does not protect those cores. `watchexec make` is a watch
# mode that does not exit, and this hook does not nudge one (the same reason
# that `--watch` is skipped below). Bare positionals are absorbed only for
# `docker run`, where the image or service name stands between the runner and
# the tool. Limiting them to `docker run` keeps `sudo apt install make`
# unmatched.
#
# The option group absorbs the arguments that a runner takes before the tool:
# flags (`uv run --project <dir> pytest`), a numeric flag (`nice -19 make`),
# the bare `--` separator (`npm exec -- vitest run`), and `timeout`'s duration.
# The repetition counts are bounded, and the pattern cannot backtrack
# pathologically on any input.
RUNNER_PREFIX = (
    r"(?:(?:uv\s+run|uvx|poetry\s+run|pdm\s+run|hatch\s+run|pipenv\s+run|npx|bunx|"
    r"npm\s+exec|pnpm\s+exec|pnpm\s+dlx|yarn\s+dlx|yarn|"
    r"python(?:3(?:\.\d+)?)?\s+-m|"
    r"docker\s+(?:compose\s+)?run(?:\s+--?[\w=.:/-]+){0,4}\s+[^-\s]\S*|"
    r"timeout\s+[\d.]+[smhd]?|stdbuf|taskset|xargs|env|nohup|caffeinate|parallel|"
    r"sudo|time|nice)\s+"
    r"(?:(?:--|-\d+|--?[A-Za-z][\w-]*(?:=\S+)?)\s+(?:[^-\s]\S*\s+)?){0,3}){0,2}"
)

# INVOCATION matches the position where a tool name counts as being run:
# command position, optionally behind transparent prefixes.
INVOCATION = COMMAND_POSITION + RUNNER_PREFIX

# SUITE_PATTERNS match whole test suites that no narrowing makes light. They
# are always SUITE runs. `cargo test` belongs here, because it recompiles the
# workspace before running one test.
SUITE_PATTERNS = [
    INVOCATION + p
    for p in (
        r"cargo\s+(?:test|bench)\b",
        r"ctest\b",
        r"bazel\s+test\b",
        r"nox\b", r"tox\b",
    )
]

# BUILD_PATTERNS match compiled builds and typecheckers. They are always BUILD
# runs.
BUILD_PATTERNS = [
    INVOCATION + p
    for p in (
        r"cargo\s+(?:build|check|clippy)\b",
        r"tsc\b",
        r"gradle\b", r"\./gradlew\b", r"mvn\b",
        r"make\b", r"cmake\s+--build\b",
        r"bazel\s+build\b",
        r"webpack\b", r"vite\s+build\b", r"next\s+build\b",
        r"go\s+build\b",
        r"npm\s+(?:run\s+)?build\b", r"yarn\s+build\b", r"pnpm\s+(?:run\s+)?build\b",
    )
]

# Interpreted test runners, where narrowing to one file/test makes the run
# light. Each alternation is grouped before the anchor is prepended. The anchor
# then governs every branch and not just the first.
TEST_RUNNERS = [
    (name, INVOCATION + pattern)
    for name, pattern in (
        ("pytest", r"pytest\b"),
        ("gotest", r"go\s+test\b"),
        ("jsx", r"(?:jest|vitest|mocha|playwright\s+test)\b"),
        ("npmtest", r"(?:npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+(?:run\s+)?test)\b"),
    )
]

# Fast type-checkers and linters are not heavy in any of these codebases, and
# the hook must not nag about them: basedpyright, pyright, mypy, ruff. The
# patterns above leave them out, and the test suite locks this in. `tsc` stays
# above, because TypeScript compiles can be heavy across a monorepo.

# Read-only inspection commands that mention a runner, such as grepping a saved
# test log, `echo make` or `sed -n /pytest/p`, stay silent without a list of
# inspectors. The mention is in the arguments of the inspector, which
# are neither command position nor behind a runner prefix. It therefore does
# not classify. The code omits a list of inspector names, because the anchors
# already decide every verdict.


def leading_program(cmd: str) -> str:
    """Return the basename of the first token past leading VAR=val
    assignments."""
    for tok in cmd.strip().split():
        if re.fullmatch(r"\w+=\S+", tok):
            continue
        return program_name(tok)
    return ""


# The command of the launcher extends to the first of these. Pipes are absent,
# because they do not end the command. They are the trailer that the mask check
# looks for. The `&` needs care. The `&` in a redirection joins a
# command to its output and does not end the command. Separate guards exclude
# different forms: the lookbehind covers `2>&-` and `2>& 1`, the `>`
# lookahead covers `&>log`, and the digit lookahead covers `0<&3`. Its `&`
# follows a `<`, which the lookbehind does not see.
#
# The `;` needs the same care, for the same reason. Bash requires one before
# the `}` of a brace group, and one before a `)` closes a subshell whose last
# command was written with a terminator. Neither ends the command of the
# launcher. Each closes the group around the command, and a `;` immediately
# before `}` or `)` is excluded. Excluding it here, as well as from the trailer
# test below, lets the scan cross that `;` and find a pipe on the far side of
# the `}`. `{ heavy make; } | tail` masks an exit code as the ungrouped form
# does.
LAUNCHER_COMMAND_END = r"&&|\|\||;(?!\s*[})])|\n|(?<!>)&(?![\d>])"

# Top-level step boundaries are the same set plus the pipe, built from the one
# definition so the redirection rule cannot drift between them.
STEP_SEPARATOR = re.compile(LAUNCHER_COMMAND_END + r"|\|")


def top_level_segments(cmd: str) -> list[tuple[str, str]]:
    """Each top-level step as a (raw, skeleton) pair, cut at the same offsets.

    Splitting reads the quote-stripped skeleton through cut_at_operators, and
    operators inside a quoted argument do not start a step. Both halves are
    kept, because each consumer needs a different one, and neither may be
    recomputed from the other:

      * `classify` reads the raw half for narrowing. The quoted file or node id
        that makes a run narrow is the text removed by the skeleton.
      * The skeleton half must be the slice decided over the whole command.
        Re-stripping a fragment gives a different answer whenever the text
        that blanked it lies outside the fragment. `cat <<'EOF' | tee doc.md`
        puts the heredoc introducer in the first step and its body in the
        second, and the second alone looks like a `make` on its own line.
    """
    parts = cut_at_operators(cmd, STEP_SEPARATOR)
    # Each half is trimmed separately. Downstream code scans the raw text for
    # narrowing and the skeleton for tools, and does not index one by the
    # other. The halves need to be the same step, and their offsets may differ.
    return [(raw.strip(), skel.strip()) for raw, skel in parts if raw.strip()]


def launcher_led(segment: tuple[str, str]) -> bool:
    """True when one top-level step is already routed through a launcher.

    The launcher must lead the step, named as a command or invoked as a script
    by path. Naming one of these files anywhere else, as in `git add
    .../tools/heavy && make -j8`, edits the file and does not run under it.
    The bare run then gets its full nudge.
    """
    raw, skeleton = segment
    if LAUNCHER_RE.match(raw):
        return True
    return leading_program(skeleton) in ("heavy", "remote")


def matches_any(patterns: list[str], cmd: str) -> bool:
    return any(re.search(p, cmd) for p in patterns)


def detect_runner(cmd: str) -> str | None:
    for name, pat in TEST_RUNNERS:
        if re.search(pat, cmd):
            return name
    return None


def is_narrowed(cmd: str, runner: str) -> bool:
    """Return True when a test-runner call targets one file or test instead of
    the suite."""
    if runner == "pytest":
        return ("::" in cmd) or bool(
            re.search(r"(?:^|[\s=/])(?:test_[\w.-]+|[\w.-]+_test)\.py\b", cmd)
        )
    if runner == "gotest":
        if re.search(r"-run\b", cmd):
            return True
        # a specific package path (but not the whole-module ./...) is narrowed
        return ("./..." not in cmd) and bool(re.search(r"(?:^|\s)\./\S+", cmd))
    if runner == "jsx":
        return bool(re.search(r"\.(?:test|spec)\.[jt]sx?\b", cmd)) or bool(
            re.search(r"(?:^|\s)[\w./-]+\.[jt]sx?\b", cmd)
        )
    if runner == "npmtest":
        # Arguments after `--` go to the runner. Only a clear single-file token
        # narrows the run.
        return bool(re.search(r"\.(?:test|spec)\.[jt]sx?\b", cmd)) or ("::" in cmd) or bool(
            re.search(r"\btest_[\w.-]+\.py\b", cmd)
        )
    return False


def classify(segment: tuple[str, str]) -> Kind | None:
    """Classify a step as 'suite' or 'build' (always nag), 'light' (nag only
    under contention) or None (ignore).

    Tool detection runs on the caller's skeleton, the way the mask check does,
    and does not recompute one. Operators like `;`, `&&` or `|` inside a quoted
    argument then do not create a command position. `git commit -m "fix flaky
    test; make the runner deterministic"` and `grep -rn -E "pytest|vitest|make"
    .` stay unclassified. Narrowing reads the raw text, because
    the file or node id that makes a run light is often the quoted argument
    that stripping removes.
    """
    raw, skeleton = segment
    if matches_any(SUITE_PATTERNS, skeleton):
        return "suite"
    if matches_any(BUILD_PATTERNS, skeleton):
        return "build"
    runner = detect_runner(skeleton)
    if runner is None:
        return None
    return "light" if is_narrowed(raw, runner) else "suite"


# --- live lock state ---------------------------------------------------------

def contention_label(current_file: Path) -> str:
    """Human-readable description of the running job, or '' if unknown."""
    record = parse_holder_record(read_holder(current_file))
    if record is None:
        return ""
    cmd = record.label
    if len(cmd) > 60:
        cmd = cmd[:57] + "..."
    secs = int(time.time()) - record.started_epoch
    age = f", running ~{secs}s" if secs >= 0 else ""
    return f'"{cmd}"{age}'


# --- main --------------------------------------------------------------------

# Output-masking trailers after a launcher run. Pipes into a viewer and `;`
# trailers make the shell report the status of the trailer (almost always 0)
# instead of the status of the command. Failed builds and tests then look
# green. `&&` is not a trailer. `A && B` short-circuits and still reports the
# failure of A. It only leaves B unlocked, which the step reminder covers. Both
# regexes use a tempered token that stops at every operator ending the command
# of the launcher. Pipes and `;` of later steps are then not blamed on the
# launcher. `heavy cargo build & git remote -v | grep origin` backgrounds the
# wrapped run and then greps, and the hook allows it. The `&` of a redirection
# (`2>&1`) does not end a command and is crossed freely. Both regexes run on
# the quote-stripped skeleton (see strip_quoted), and they do not see quotes.
UNTIL_COMMAND_END = r"(?:(?!" + LAUNCHER_COMMAND_END + r")[\s\S])*?"
# A launcher whose first argument asks for its help does not run a command,
# and its trailer does not mask a result. MASKABLE_LAUNCHER skips it.
MASKABLE_LAUNCHER = LAUNCHER + r"(?!\s+(?:-h|--help)(?:[\s;&|]|$))"
MASK_PIPE_RE = re.compile(
    MASKABLE_LAUNCHER
    + UNTIL_COMMAND_END
    + r"\|\s*"
    + (
        r"(?:tee|tail|head|grep|egrep|fgrep|rg|ag|cat|bat|less|more|awk|sed|wc|sort|uniq|"
        r"column|jq|cut|tr)\b"
    )
)
# The `\S` after the `;` must not be the closing `}` or `)` of a group. That
# character is group syntax, because `{ heavy make; }` has nowhere else to put
# the `;`. The `;` is therefore not read as a trailer, and the only hard block
# of this hook does not deny a serialized run.
MASK_SEMI_RE = re.compile(MASKABLE_LAUNCHER + UNTIL_COMMAND_END + r";\s*(?![})])\S")


def mask_deny_reason(launcher: str) -> str:
    """Return the mask-deny message, naming the launcher that the command used.

    Naming the launcher keeps a `remote` run on the task machine.
    """
    return (
        f"Blocked: this `{launcher}` command pipes into a viewer or has a `;` trailer, "
        f"and the shell would report the trailer's exit code. Run `{launcher} --log "
        f"FILE.log <command>` with nothing after it, then read FILE in a separate command."
    )


# WAIT_ARGUMENT matches a `--wait` first argument after the launcher.
WAIT_ARGUMENT = re.compile(r"\s+--wait(?:[\s=]|$)")


def wait_deny_reason(launcher: str) -> str:
    """Return the deny message for a masked `--wait`, which waits on a run
    that is already logging and must not be re-run."""
    return (
        f"Blocked: run `{launcher} --wait FILE` with nothing after it. It exits with "
        f"the run's code and prints exit-status=<N>. Read FILE in a separate command."
    )


def unwrapped_step_reminder(launcher: str) -> str:
    """Return the multi-step nudge, naming the launcher that the command
    already routes through.

    The nudge keeps a sequence led by `remote` on the task machine, because a
    re-run as `heavy bash -c '...'` pulls it back onto this machine. The body
    speaks of "the steps routed through a launcher" instead of the named one,
    because commands may route different steps through different launchers.
    Only the wrap instruction has to pick one.
    """
    return (
        f"This command runs more than one step, and only the steps routed through a "
        f"launcher are serialized. Wrap the whole sequence as one job: `{launcher} "
        f"bash -c '<step1> && <step2>'`. This run is allowed."
    )


def command_launcher(segments: list[tuple[str, str]]) -> str:
    """Return the launcher that a wrapped command routes through, for a message
    that must name it.

    Read from the steps, because a launcher name that is an argument and not
    the program must not name it. The first step that is routed through a
    launcher names it, by a launcher token or by invoking its script. Commands
    without such a step fall back to `heavy`.
    """
    for raw, skeleton in segments:
        match = LAUNCHER_RE.match(raw)
        if match is not None:
            return match.group("launcher")
        program = leading_program(skeleton)
        if program in ("heavy", "remote"):
            return program
    return "heavy"


def recommended_launcher() -> str:
    """`remote` wherever it is installed on the PATH, otherwise `heavy`.

    Unconfigured, `remote` falls back to running the command under `heavy`
    here. It therefore serves every repository, and this check does not read
    configuration.
    """
    return "remote" if shutil.which("remote") is not None else "heavy"


def placement(launcher: str, contended: bool) -> str:
    """Return the place where `<launcher> <command>` puts a full run, as the
    clause after "so it".

    `remote` runs it on the task machine and, where none is configured, falls
    back to the global lock on this machine.
    """
    local = "queues behind that job" if contended else "runs under the global lock"
    if launcher == "remote":
        return f"runs on the task machine, or {local} where none is configured,"
    return local


def heaviest_kind(segments: list[tuple[str, str]]) -> Kind | None:
    """The heaviest classification across the steps.

    Every step is judged separately, and a light step does not mask a suite
    beside it. `pytest tests/test_x.py::test_y && pytest -n auto` is a suite
    run. Classifying the whole string instead finds the node id, calls the
    command narrowed, and stays silent through the suite that follows.

    The readers of this answer ask whether the bare run ahead is worth a nudge,
    and whether a step beside a launcher runs outside the lock. Each of them
    acts on a "suite" or "build" verdict.
    """
    kinds = {classify(segment) for segment in segments}
    return next((kind for kind in KINDS_BY_WEIGHT if kind in kinds), None)


def main() -> None:
    payload = read_payload()
    if payload is None:
        return  # A malformed payload. Leave the tool call alone.
    command = bash_command(payload)
    if command is None or not command.strip():
        return
    # A backslash-newline continues the command. Joined, the pipe in `heavy
    # make \` then `| tail` is a trailer of the launcher's command. The
    # skeleton leaves one in a quote or a heredoc body in place.
    cmd = joined_continuations(command)

    # Split once, on top-level operators only, and judge wrapped-ness per
    # segment. Commands count as routed through a launcher when any of their
    # steps is. The reminder below concerns the other steps, whichever operator
    # separates them.
    segments = top_level_segments(cmd)
    if any(launcher_led(segment) for segment in segments):
        # An output-masking trailer, a pipe into a viewer or a `;` trailer, is
        # always an error, because `--log` already captures the output. Block
        # it with a deny whose reason names the bare form.
        skeleton = strip_quoted(cmd)
        mask = MASK_PIPE_RE.search(skeleton) or MASK_SEMI_RE.search(skeleton)
        if mask is not None:
            launcher = mask.group("launcher")
            if WAIT_ARGUMENT.match(skeleton, mask.end("launcher")):
                deny(wait_deny_reason(launcher))
            else:
                deny(mask_deny_reason(launcher))
            return
        # Heavy steps left unwrapped beside the launcher run outside the lock,
        # regardless of the operator that separates them. The nudge asks to
        # wrap the whole sequence as one job. Only suite and build steps count.
        # Narrowed single tests do not need the lock by themselves, and the
        # nudge does not tell a one-test run that it is fighting for cores.
        # Launcher-led steps do not need a skip and do not have one. They lack
        # a top-level operator, and their tool stands one token past the
        # launcher. Launchers are not runner prefixes, and such steps classify
        # as None by themselves.
        if heaviest_kind(segments) in ("suite", "build"):
            add_context(unwrapped_step_reminder(command_launcher(segments)))
        return

    # Watch and interactive modes do not exit and keep the lock until killed.
    # Skip.
    if re.search(r"--watch\b|--watchAll\b|\bnodemon\b", cmd):
        return

    # The kind is judged per step instead of over the whole command. The
    # leading token of the first step does not describe the later steps, and
    # reading it that way silenced the build in `ls && make -j8`.
    kind = heaviest_kind(segments)
    if kind is None:
        return  # Not a test or build command. Leave it alone.

    # The bare-run nudges below tell the session to use one launcher. Stay
    # silent when that launcher is not installed. Contributors who did not opt
    # in are then not nagged about a missing tool, and those who installed only
    # `remote` still are about suites. The mask-deny above does not depend on
    # this check, because it fires only on a command already invoking a
    # launcher. Builds and light runs are not offloaded, and their reminder
    # names `heavy`.
    recommended = recommended_launcher()
    launcher = recommended if kind == "suite" else "heavy"
    skill = f"{launcher}-task"
    if shutil.which(launcher) is None:
        return

    paths = lock_paths(lock_directory(os.environ, Path.home()))
    contended = lock_held_by_other(paths.lock_file)
    label = contention_label(paths.current_file) if contended else ""
    who = f" ({label})" if label else ""
    opening = (
        f"Another session is running a heavy job under the global lock right now{who}. "
        if contended
        else ""
    )

    if kind == "light":
        if not contended:
            # Single quick tests with the lock free do not need a reminder.
            return
        reminder = (
            f"{opening}This light run does not require `{launcher}`. Running it as "
            f"`{launcher} <command>` gives more reliable results and timing. This run "
            f"is allowed."
        )
    else:  # suite or build
        reminder = (
            f"{opening}This is a full test-suite, build or compile run. This run is "
            f"allowed. Load `/{skill}` and run the next one as `{launcher} <command>` so "
            f"it {placement(launcher, contended)} instead of competing for the cores."
        )

    if launcher == "remote":
        reminder = f"{reminder} {LOCAL_WRITES_SENTENCE}"
    elif kind == "build" and recommended == "remote":
        reminder = f"{reminder} {LONG_RUN_SENTENCE}"
    add_context(reminder)


if __name__ == "__main__":
    main()
