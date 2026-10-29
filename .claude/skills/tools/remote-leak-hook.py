#!/usr/bin/env -S python3 -B
"""Keep the remote configuration out of tracked files, as a PreToolUse guard.

`remote` is configured per repository in `.claude/settings.local.json` under
`env`: `REMOTE_TASK_HOST`, an SSH destination, and `REMOTE_TASK_DIR`, a path on
that machine. Both describe a private machine and belong in that one untracked
file.

The hook runs on `Bash`, `Write`, and `Edit`, and denies these routes:

  * A git command that would put the settings file into the index or a stash.
    The file is gitignored in every repository. Staging it takes either its
    name as a pathspec or a force. `git add -f` stages ignored files whatever
    the pathspec, and `-f .`, `-f -A` and `-f .claude` all stage it without
    naming it. `git stash --all` sweeps them in without any force. The rule
    reads both the pathspec and those flags, through a quote or a paren (`bash
    -c "git add -f ."`), and reads a long flag in any abbreviation that git
    accepts (`--forc`, `--al`). `git update-index` adds every path that it
    names, and naming the file to it is enough. With `--index-info`, or with
    `--stdin` and `--add`, it adds paths read from standard input, and it is
    denied regardless of the paths on its command line. A pathspec read with
    `--pathspec-from-file` comes from elsewhere in the command, and the whole
    command is searched for the file then. A glob names the file when its
    last two components, lexically normalized, match the file's path
    (`.claude/settings.local.*`, `$PWD/.claude/*.local.json`). A heredoc
    body is a document, and its words stage nothing, when its delimiter is
    quoted and every command on its introducer line that reads a body is
    `cat` or `tee` without an output process substitution. A body is read by
    the command of its introducer and by each command downstream of that one
    in a pipeline. Any other heredoc body counts as command text. The rule
    fires whether or not `remote` is configured here, because it does not
    need configuration to know the protected file.
  * A `cp`, `mv`, `install`, `rsync`, `scp`, or `ln` giving the settings file a
    second path, by name or by a glob, in any token position so a `sudo` or
    `time` prefix does not hide it. Only the original path is gitignored. The
    second one is an ordinary file that can be staged freely.
  * A shell command that writes a configured value into a file, via `>`, `>>`,
    `>&` followed by a file name, or `tee`, piped or not. The value scan
    reads the command as written and with each `$'...'` span decoded.
  * A git command that would record a configured value in history or a ref: a
    commit message, a tag annotation, a note, a merge message. Those cannot be
    edited away, and a host name in one of them is a likelier accident than
    one in a redirection. The rule reads a message that spans lines inside
    quotes or a heredoc, one glued to `-m`, one written as `$'...'` with
    escapes, and one read from standard input (`-F -`, `--file /dev/stdin`,
    `/proc/<pid>/fd/0`) out of another stage of the command.
  * A `Write` or an `Edit` placing a configured value in any file other than
    this project's `.claude/settings.local.json`. The values belong in that
    file, and without that exemption they could not be corrected or rotated.

Content scanning catches accidents. It does not stop an adversary. Values
assembled across two edits, split over a line break, encoded, written by `sed
-i` or a `python3 -c` one-liner, or read out of the settings file by a command
that does not name it, get into a file unchallenged. `cat <file> > <copy>`
names the file but no copy command, and passes. So does a copy of the
directory that contains the settings file, such as `cp -r .claude backup/` or
`tar czf c.tgz .claude`. That last one could be matched on the fixed name
`.claude`, and is not, because the same directory contains `settings.json` and
`skills/`. Those are copied in the ordinary course of work, and the
guard leaves that work alone. The staging rule escapes the same
trap only because `-f` is itself the signal, with no legitimate use against an
ignored path.

Ref names built from a value pass for a narrower reason than it seems.
`tag`, `notes`, and `merge` segments are scanned in full, and `git notes add -m
ok <host>` is denied. `git checkout -b deploy-to-<host>` passes because of the
leading boundary. The value follows a hyphen, inside a longer token, and the
scan does read it. `git commit -F <file>` records the content of a file that
this hook cannot see. A glob needs its `.claude` component to match, and a
bare-name glob run from inside `.claude/`, such as `cd .claude && git add
*.local.json`, passes. Paths that `xargs` passes to git, as in `find ... |
xargs git update-index --add`, appear nowhere in the command, and pass. So
does a heredoc that `cat` or `tee` writes to a script, which a later command
runs, as in `cat > x.sh <<'EOF' && bash x.sh`. `NotebookEdit` is
outside the tool matcher and is not scanned at all.

Configuration is read from the environment and from the settings file, because
hook processes are not guaranteed to inherit the session environment. The guard
must see the values to deny a leak. In an unconfigured repository the three
value rules lack a value to scan for and stay silent. The two file rules do not
depend on configuration. Every failure degrades to silence, and so does every
input, however large, because guards that raise or run long enough to look
like a hang block the tool call in question. Every rule here uses regexes
that cannot backtrack across the command.
"""
import os
import posixpath
import re
from bisect import bisect_left
from functools import cache
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from claude_settings import SETTINGS_LOCAL_PATH, SettingsError, environment_block
from hook_payload import read_payload
from json_value import Json
from lexical_paths import normal_join
from pre_tool_use_hook import bash_command, deny, tool_input
from remote_configuration import DIRECTORY_CHARACTERS, REQUIRED_KEYS
from shell_command import (
    decoded_ansi_c,
    joined_continuations,
    operator_spans,
    program_name,
    skeleton_and_introducers,
    strip_unquoted_data,
)

# SETTINGS_LOCAL_NAME is the bare name of the settings file, as a command line
# mentions it.
SETTINGS_LOCAL_NAME = SETTINGS_LOCAL_PATH.name

# A token containing one of these characters is a glob, and names the settings
# file when it matches the file's path.
GLOB_CHARACTERS = frozenset("*?[")

# Globs whose last two components are longer than this are not matched.
# Translating a glob takes time quadratic in its unclosed brackets, fifty
# thousand of which take a minute. A glob of the settings file's path needs
# far fewer characters.
MAXIMUM_GLOB_LENGTH = 256

# The paths through which a git option reads standard input, after lexical
# normalization. POSIX keeps a leading `//`, and the pattern accepts it.
STANDARD_INPUT_PATH = re.compile(r"-|//?dev/stdin|//?dev/fd/0|//?proc/[^/]+/fd/0")

# Programs that read a heredoc body as data and run none of it. A body read by
# any other program counts as command text.
DATA_READERS = frozenset({"cat", "tee"})

# The opener of an output process substitution. It runs a program on the data
# written into it. An input one, `<(`, cannot receive a heredoc body.
OUTPUT_PROCESS_SUBSTITUTION = ">("

# A leading variable assignment, which precedes the program of a command.
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*", re.DOTALL)

# Commands that give a file a second path, where the copy is no longer covered
# by the gitignore rule that protects the original. Matched in any token
# position. `sudo cp <file> x` and `time cp <file> x` are copies of the
# settings file, and a rule that reads only the leading word misses them. `scp`
# and `ln` are here for the same reason as the rest. `scp` is a copy command a
# letter away from `cp`, in a hook about moving work to another machine. `ln`
# gives the same bytes a second name that git will stage.
COPY_COMMANDS = frozenset({"cp", "mv", "install", "rsync", "scp", "ln"})

# git subcommands that write text into history or into a ref, where it cannot
# be edited away afterwards. Commit messages are the likeliest of these by far,
# but tag annotations, notes, and merge messages are as permanent and as
# public.
HISTORY_SUBCOMMANDS = frozenset({"commit", "tag", "notes", "merge"})

# Configured values shorter than this are dropped instead of scanned for.
# Three-character host aliases also occur in ordinary prose, and dropping them
# keeps unrelated edits allowed. Dropping one value leaves the other scanned
# normally.
MINIMUM_SCANNED_LENGTH = 4

# PRECEDING_CHARACTERS lists the characters that may come before a value and
# still make it part of something else. Values flanked by one of these belong
# to a longer token and are not the value. An alias `task` does not occur in
# `heavy-task` or in `src/task/runner.py`, and the configured directory does
# not occur in `/mnt/backup<dir>`, a different absolute path that ends the same
# way.
#
# The `@` is absent. `REMOTE_TASK_HOST` is an SSH destination, and
# `deploy@<host>` is its likeliest written form. That is the same host with a
# login in front. The cost is that a Python decorator `@task` fires for a word
# alias. Such an alias already fires on the bare word, and the deny reasons
# name the way out.
#
# The `/` has to stay, or an ordinary-word alias fires on every path component
# that spells it (`src/task/runner.py`). The scheme is carved back out of it
# separately, by the `(?<=://)` alternative in contains_value.
# `ssh://<host>/srv` is a clone URL naming the private host, the case this
# guard exists for, and `://` is narrow enough to leave an ordinary path
# component alone.
PRECEDING_CHARACTERS = "A-Za-z0-9._/-"

# TOKEN_CONTINUATION lists the characters that may come after a value and still
# make it part of something else. Two characters are absent from this one, for
# opposite reasons.
#
# The period is absent, because it ends a sentence as readily as it continues a
# name. Trailing periods are punctuation unless more of a name follows them
# (the second lookahead below). The alias in "the host is buildbox." is still
# the alias, while `box.example` inside `box.example.com` is a different name.
#
# The slash is absent, because a longer path is not a different place.
# `buildbox-staging` is a different host from `buildbox` and is nobody's
# secret, but `<dir>/logs/out.txt` is the configured directory with something
# after it. The extra element follows the value and does not conceal it.
TOKEN_CONTINUATION = "A-Za-z0-9_@-"

# SEGMENT_SEPARATORS mark where one command ends and the next begins. They are
# matched in the quote-stripped skeleton, outside quotes, comments and heredoc
# bodies. A `|&` is one separator. The `&` of a redirection (`2>&1`, `<&0`,
# `&>`) and the `|` of `>|` separate nothing. The staging, copying and
# git-recording rules work segment by segment, apart from a message read from
# standard input. Text in a later command then does not count against a git
# call. The redirection rule is the exception and reads the whole string. It
# asks only whether a value and a redirection are both present, without pairing
# them. Values in one command and redirections in another then deny together.
# That over-denies, the safe direction for a guard, and the deny reason names
# the way out.
#
# No rule scans with a pattern that could backtrack. An earlier version of the
# staging rule used two tempered quantifiers around an alternation, which cost
# seconds at ten thousand characters and minutes at fifty thousand; a hook that
# takes minutes has blocked the tool call just as surely as one that raises.
SEGMENT_SEPARATORS = re.compile(r"[;\n]|(?<!>)\|&?|(?<![<>])&(?!>)")

# Matches redirections into a file: `>`, `>>`, `>|`, `>&` followed by a file
# name, or `tee` with or without a pipe in front of it. `>&` followed by a
# descriptor number or `-` is excluded, because `2>&1`, `>&2` and `>&-`
# duplicate or close a descriptor and write nothing to disk. This runs only on
# a command already known to contain a configured value.
REDIRECTION_RE = re.compile(r">(?!&)|>&(?!\s*(?:\d+|-)(?:[\s;&|()<>'\"`]|$))|\btee\b")

STAGES_DENY_REASON = (
    "Blocked: this would put `.claude/settings.local.json` into the index or a "
    "stash, by naming it, by `git add -f`, or by `git stash --all`. Leave the file "
    "untracked, and stage the rest of the work by explicit path without `-f`. If "
    "the command only mentions the file's name, as a commit message can, reword "
    "the mention."
)

COPIES_DENY_REASON = (
    "Blocked: this copies or moves `.claude/settings.local.json` to another path, "
    "where git can stage it. Read the file in place, and keep any backup outside "
    "the repository."
)

# The three value rules end with the same escape hatch, because all three fail
# the same way. Host aliases that are ordinary words are indistinguishable from
# those words used as words, and matching rules cannot tell them apart. Saying
# so, and naming the durable fix, keeps a false positive from becoming a dead
# end.
FALSE_POSITIVE_ADVICE = (
    "If the match is an ordinary word that equals the host alias, rename the alias "
    "in your SSH config to a word absent from prose, and update `REMOTE_TASK_HOST` "
    "in `.claude/settings.local.json` to match."
)

REDIRECTION_DENY_REASON = (
    "Blocked: this command writes the value of `REMOTE_TASK_HOST` or "
    "`REMOTE_TASK_DIR` into a file. Name the key, or read the value from the "
    "environment when the command runs. "
) + FALSE_POSITIVE_ADVICE

HISTORY_DENY_REASON = (
    "Blocked: this would write the value of `REMOTE_TASK_HOST` or `REMOTE_TASK_DIR` "
    "into git history or a ref. Describe the change without the value, naming the "
    "key where needed. "
) + FALSE_POSITIVE_ADVICE

WRITES_DENY_REASON = (
    "Blocked: this writes the value of `REMOTE_TASK_HOST` or `REMOTE_TASK_DIR` into "
    "a file other than this project's `.claude/settings.local.json`. Name the key "
    "instead. "
) + FALSE_POSITIVE_ADVICE


def contains_value(text: str, values: list[str]) -> bool:
    """True when a configured value stands in text as a whole token.

    Hits flanked by a character that the value could itself contain belong to
    a longer token and are not the value. PRECEDING_CHARACTERS and
    TOKEN_CONTINUATION list the characters that count on each side, and say
    why the two lists differ. A period continues the value only when a
    character that a configured value may contain follows it; the directory's
    class, the wider of the two, stands for both values.
    """
    for value in values:
        pattern = (
            f"(?:(?<=://)|(?<![{PRECEDING_CHARACTERS}])){re.escape(value)}"
            f"(?![{TOKEN_CONTINUATION}])(?!\\.[{DIRECTORY_CHARACTERS}])"
        )
        if re.search(pattern, text):
            return True
    return False


def read_configuration() -> list[str]:
    """The configured values worth scanning for, longest-lived source first.

    The environment is read first and the settings file second, because Claude
    Code populates the environment from that file's `env` block. Hook processes
    that inherited an empty environment still see the values. Values shorter
    than MINIMUM_SCANNED_LENGTH are dropped. Settings files that cannot be read
    are skipped without raising.
    """
    values: list[str] = []
    for key in REQUIRED_KEYS:
        from_environment = os.environ.get(key)
        if isinstance(from_environment, str):
            values.append(from_environment)
    project_directory = os.environ.get("CLAUDE_PROJECT_DIR")
    if project_directory:
        try:
            block = environment_block(Path(project_directory) / SETTINGS_LOCAL_PATH)
        except SettingsError:
            block = {}
        for key in REQUIRED_KEYS:
            from_file = block.get(key)
            if isinstance(from_file, str):
                values.append(from_file)
    return [value for value in values if len(value) >= MINIMUM_SCANNED_LENGTH]


class Segment(NamedTuple):
    """One command of a command line, in two views of equal length.

    `raw` is the command as written, heredoc bodies included. `words` shows
    the words that the shell runs, and is read for command words and flags.
    On a line that command_segments reads as a document, it blanks comments
    and the heredoc bodies whose delimiter is quoted, and keeps quoted text.
    On any other line it is the raw text. Names of the settings file are read
    from `words`. Values are read from `raw`, because a heredoc body can feed
    a command the message that it records.
    """

    raw: str
    words: str


def command_segments(command: str) -> list[Segment]:
    """Split the command at the operators that separate one command from the
    next.

    Line continuations are joined first, and the join keeps the flag of `git
    add \\` then ` -f .` in the segment of the command. Separators are found in
    the quote-stripped skeleton. A newline inside a quoted message then does
    not end a segment, and a heredoc body stays in the last segment of the
    line of its introducer.

    A line reads as a document, and its segments keep the `words` view, when
    every command on it that reads a heredoc body reads it as data
    (reads_as_data). Any other line keeps its raw text as `words`, because a
    heredoc body read by any other program counts as command text. A command
    with blank words, such as an empty one between two operators, does not
    count, and a line of nothing else keeps its raw text.
    """
    joined = joined_continuations(command)
    skeleton, introducers = skeleton_and_introducers(joined)
    words_view = strip_unquoted_data(joined)
    segments: list[Segment] = []
    spans: list[tuple[int, int]] = []
    for start, end in operator_spans(skeleton, SEGMENT_SEPARATORS):
        spans.append((start, end))
        if joined[end : end + 1] not in ("\n", ""):
            continue
        line = [
            Segment(joined[first:last], words_view[first:last]) for first, last in spans
        ]
        receivers = body_receivers(spans, skeleton, introducers)
        counted = [
            (segment, receives)
            for segment, receives in zip(line, receivers, strict=True)
            if segment.words.strip()
        ]
        if counted and all(
            reads_as_data(segment) for segment, receives in counted if receives
        ):
            segments.extend(line)
        else:
            segments.extend(Segment(segment.raw, segment.raw) for segment in line)
        spans = []
    return segments


def body_receivers(
    spans: list[tuple[int, int]], skeleton: str, introducers: list[int]
) -> list[bool]:
    """Return, for each command of one line, whether it reads a heredoc body.

    A body is read by the command that holds its introducer and passed on to
    each command downstream of that one in a pipeline. A command after `&&`,
    `||`, `;` or a background `&` reads none of it.
    """
    receivers: list[bool] = []
    receiving = False
    for start, end in spans:
        index = bisect_left(introducers, start)
        holds_introducer = index < len(introducers) and introducers[index] < end
        receiving = receiving or holds_introducer
        receivers.append(receiving)
        receiving = receiving and passes_body_on(skeleton, end)
    return receivers


def passes_body_on(skeleton: str, position: int) -> bool:
    """Return True when the separator at a position of the skeleton is a pipe,
    `|` or `|&`, which hands a heredoc body to the command after it."""
    return (
        skeleton[position : position + 1] == "|"
        and skeleton[position + 1 : position + 2] != "|"
    )


def reads_as_data(segment: Segment) -> bool:
    """Return True when a command reads a heredoc body as data and runs none
    of it: a data reader (`cat`, `tee`) without an output process
    substitution, which could hand its input to another program."""
    program = program_name(command_word(shell_tokens(segment.words)))
    return program in DATA_READERS and OUTPUT_PROCESS_SUBSTITUTION not in segment.raw


def command_word(tokens: list[str]) -> str:
    """Return the first token past leading variable assignments, or an empty
    string."""
    return next((token for token in tokens if not ASSIGNMENT.fullmatch(token)), "")


def shell_tokens(segment: str) -> list[str]:
    """Return the segment's whitespace-separated tokens, with shell quoting,
    grouping and separators stripped from each end.

    `bash -c "git add -f ."`, `sh -c 'git add -f .'` and `(git add -f .)` all
    stage an ignored file, and a comparison against the raw token misses every
    one of them, because the quote or the paren is stuck to the word. Stripping
    also closes the same hole one token later, where trailing quotes keep
    `--all"` from matching the flag test.

    A `;`, `&` or `|` glued to a word inside a quoted script is stripped too,
    so the separator does not hide the flag, as in `"git stash --all; ls"`.

    The set covers command substitution and escaping too. `$(git add -f .)` and
    `\\git add -f .` both stage, and stripping `$` and `\\` along with the
    backtick and the paren keeps the class consistent.

    It costs one asymmetry in the other direction. `grep "git add -f"
    README.md` now denies. That is already the answer for `echo git add -f .`,
    and the position of the quote is an arbitrary place for the two to differ.
    They therefore agree, in the direction that denies the staging.
    """
    return [token.strip("'\"()`$\\;&|") for token in segment.split()]


def git_invocations(segment: str) -> list[list[str]]:
    """Return the tokens of each git invocation in the segment, each run from
    its `git` word up to the next one.

    A quoted script (`bash -c "git status -- a; git add -f ."`) is one segment
    holding several git commands. Reading each from its own `git` word keeps
    one command's `--` and subcommand from standing for the next one's.
    """
    invocations: list[list[str]] = []
    for token in shell_tokens(segment):
        if program_name(token) == "git":
            invocations.append([token])
        elif invocations:
            invocations[-1].append(token)
    return invocations


def onward(found: list[bool]) -> list[bool]:
    """Return, for each git invocation of a segment, whether the value is true
    for it or for a later one.

    A word `git` inside an argument (`-m "update git hooks"`) also starts an
    invocation, and the flags after it belong to the command before it.
    Reading a flag onward keeps it counted for that command. It also counts
    for an earlier command of a quoted script, which only over-denies.
    """
    onward_found: list[bool] = []
    later = False
    for value in reversed(found):
        later = later or value
        onward_found.append(later)
    return onward_found[::-1]


def flag_onward(
    invocations: list[list[str]], short: str | None, long_form: str
) -> list[bool]:
    """Return, for each git invocation, whether it or a later one carries a
    flag, read as carries_flag reads it."""
    return onward([carries_flag(tokens, short, long_form) for tokens in invocations])


def abbreviates(token: str, long_form: str) -> bool:
    """Return True when a token is a long option or a prefix of it, from one
    letter past `--` up, with or without an `=value`.

    git accepts any unambiguous prefix of a long option. An ambiguous prefix
    counts here too. git refuses it, and counting it only over-denies.
    """
    name = token.partition("=")[0]
    return len(name) > 2 and name.startswith("--") and long_form.startswith(name)


def short_cluster(token: str) -> str:
    """Return the letters of a short-option token, `fA` for `-fA`, or an
    empty string for any other token."""
    return token[1:] if token.startswith("-") and not token.startswith("--") else ""


def carries_flag(tokens: list[str], short: str | None, long_form: str) -> bool:
    """Return True when a short flag is present, alone or in a cluster, or its
    long form is, whole or abbreviated.

    A flag without a short form passes None. Scanning stops at `--`, past
    which every token is a pathspec. A file named `-f` then does not read as
    a force.
    """
    for token in tokens:
        if token == "--":
            return False
        if abbreviates(token, long_form):
            return True
        if short is not None and short in short_cluster(token):
            return True
    return False


def option_arguments(tokens: list[str], short: str, long_form: str) -> list[str]:
    """Return the arguments given to an option that takes one.

    The short form takes its argument glued (`-mtext`), glued to the end of a
    cluster (`-amtext`), or as the next token. The long form, whole or
    abbreviated, takes it after `=` or as the next token. Scanning stops at
    `--`, as in carries_flag.
    """
    arguments: list[str] = []
    for index, token in enumerate(tokens):
        if token == "--":
            break
        following = tokens[index + 1 : index + 2]
        if abbreviates(token, long_form):
            _, equals, value = token.partition("=")
            arguments.extend([value] if equals else following)
            continue
        letters = short_cluster(token)
        position = letters.find(short)
        if position >= 0:
            glued = letters[position + 1 :]
            arguments.extend([glued] if glued else following)
    return arguments


def reads_standard_input(path: str) -> bool:
    """Return True when a path names standard input, once lexically
    normalized (`/dev//stdin`, `/proc/<any>/fd/0`)."""
    return STANDARD_INPUT_PATH.fullmatch(posixpath.normpath(path)) is not None


def names_settings_local(text: str, tokens: list[str]) -> bool:
    """Return True when a text names the settings file, by name or by a glob
    that matches its path.

    A glob is normalized lexically, and its last two components are matched
    against `.claude/settings.local.json` component by component. A directory
    in front of them does not matter. `$PWD/.claude/*.local.json` and
    `.claude/x/../settings.local.*` match, and `*.json` does not.
    """
    if SETTINGS_LOCAL_NAME in text:
        return True
    for token in tokens:
        if GLOB_CHARACTERS.isdisjoint(token):
            continue
        last_two = PurePosixPath(*PurePosixPath(posixpath.normpath(token)).parts[-2:])
        if (
            len(last_two.parts) == 2
            and len(str(last_two)) <= MAXIMUM_GLOB_LENGTH
            and SETTINGS_LOCAL_PATH.match(str(last_two))
        ):
            return True
    return False


def adds_paths_from_input(invocations: list[list[str]]) -> list[bool]:
    """Return, for each git invocation, whether as `git update-index` it adds
    paths that it reads from standard input: with `--index-info`, or with
    `--stdin` and `--add`, each read onward."""
    index_info = flag_onward(invocations, None, "--index-info")
    add = flag_onward(invocations, None, "--add")
    standard_input = flag_onward(invocations, None, "--stdin")
    return [
        entries or (adds and reads)
        for entries, adds, reads in zip(index_info, add, standard_input, strict=True)
    ]


def stages_settings_local(command: str) -> bool:
    """Return True when a git command would put the settings file into the
    index or a stash.

    The file gets in by name, or by forcing ignored files in wholesale with
    `git add -f` or `git stash --all`, and the flags matter as much as the
    pathspec. Legitimate `git add` calls on un-ignored paths need neither.
    Reading the flags therefore rarely denies a legitimate command. `git
    update-index` adds a named path whatever the gitignore rules say, and
    naming the file to it is enough.

    Needs no configuration: the file is protected by name, so an unconfigured
    repository -- or a hook that could not read the values -- still refuses to
    track it.
    """

    @cache
    def command_names_file() -> bool:
        return names_settings_local(command, shell_tokens(command))

    for segment in command_segments(command):
        invocations = git_invocations(segment.words)
        if not invocations:
            continue
        segment_names_file = names_settings_local(
            segment.words, shell_tokens(segment.words)
        )
        reads_pathspec_file = flag_onward(invocations, None, "--pathspec-from-file")
        forces = flag_onward(invocations, "f", "--force")
        stashes_ignored = flag_onward(invocations, "a", "--all")
        adds_from_input = adds_paths_from_input(invocations)
        for index, tokens in enumerate(invocations):
            if reads_pathspec_file[index]:
                names_file = command_names_file()
            else:
                names_file = segment_names_file
            if "add" in tokens and (names_file or forces[index]):
                return True
            if "stash" in tokens and (names_file or stashes_ignored[index]):
                return True
            if "commit" in tokens and names_file:
                return True
            if "update-index" in tokens and (names_file or adds_from_input[index]):
                return True
    return False


def copies_settings_local(command: str) -> bool:
    """True when a command copies or moves the settings file to another path.

    The gitignore rule protects one path and leaves the file's contents
    unprotected. Copies anywhere else are ordinary tracked-by-default files.
    Like the staging rule, this rule does not need configuration.
    """
    for segment in command_segments(command):
        if any(
            program_name(token) in COPY_COMMANDS for token in shell_tokens(segment.words)
        ) and names_settings_local(segment.words, shell_tokens(segment.words)):
            return True
    return False


def readings(text: str) -> tuple[str, str]:
    """Return the text as written and with each `$'...'` span decoded, the two
    readings that a value scan checks.

    Decoding separates a value from the escape before it, as in
    `$'x\\n<value>'`. The text as written keeps a value that decoding joins to
    a neighbor, such as one after a `$'...'` that is literal inside double
    quotes.
    """
    return text, decoded_ansi_c(text)


def redirects_value(command: str, values: list[str]) -> bool:
    """Return True when a command redirects a configured value into a file.

    The value scan checks both readings of the command. The redirection scan
    reads the command as written, where an escaped `>` is data.
    """
    if not any(contains_value(reading, values) for reading in readings(command)):
        return False
    return bool(REDIRECTION_RE.search(command))


def message_contains_value(text: str, values: list[str]) -> bool:
    """Return True when either reading of a git command's text contains a
    configured value, standing alone or glued to `-m`."""
    for reading in readings(text):
        messages = option_arguments(shell_tokens(reading), "m", "--message")
        if contains_value(reading, values) or any(
            contains_value(message, values) for message in messages
        ):
            return True
    return False


def records_value_in_git(command: str, values: list[str]) -> bool:
    """Return True when git would record a configured value in history or a
    ref.

    A message read from standard input (`-F -`, `--file=/dev/stdin`) comes
    from a pipe or a heredoc elsewhere in the command, and the whole command
    is scanned then, once however many commands read it. Any other message
    is read from its segment. The scan checks both readings of the text, and
    reads a message glued to `-m` (`-m<text>`, `-am<text>`) apart from its
    option.
    """

    @cache
    def command_contains_value() -> bool:
        return message_contains_value(command, values)

    for segment in command_segments(command):
        invocations = git_invocations(segment.words)
        reads_onward = onward(
            [
                any(
                    reads_standard_input(path)
                    for path in option_arguments(tokens, "F", "--file")
                )
                for tokens in invocations
            ]
        )
        reads_input = [
            reads
            for tokens, reads in zip(invocations, reads_onward, strict=True)
            if any(token in HISTORY_SUBCOMMANDS for token in tokens)
        ]
        if any(reads_input) and command_contains_value():
            return True
        if not all(reads_input) and message_contains_value(segment.raw, values):
            return True
    return False


def is_settings_local(file_path: str) -> bool:
    """Return True when a path is the settings file of this project.

    With CLAUDE_PROJECT_DIR set, the comparison is between absolute paths.
    Settings files nested elsewhere in the tree are then not mistaken for the
    project's file. Relative paths are resolved against the project directory,
    because the hook's working directory is not guaranteed to be the project
    directory. Without a project directory, the comparison falls back to the
    tail of the path.
    """
    project_directory = os.environ.get("CLAUDE_PROJECT_DIR")
    if not project_directory:
        return file_path.endswith(str(SETTINGS_LOCAL_PATH))
    try:
        project = Path(project_directory).absolute()
        return normal_join(project, file_path) == normal_join(project, SETTINGS_LOCAL_PATH)
    except (OSError, ValueError):
        return file_path.endswith(str(SETTINGS_LOCAL_PATH))


def writes_value(tool_name: str, arguments: dict[str, Json], values: list[str]) -> bool:
    """Return True when a Write or an Edit would place a configured value in
    another file.

    This project's `.claude/settings.local.json` is exempt.
    """
    if tool_name == "Write":
        written: Json = arguments.get("content")
    elif tool_name == "Edit":
        written = arguments.get("new_string")
    else:
        return False
    if not isinstance(written, str):
        return False
    file_path: Json = arguments.get("file_path")
    if isinstance(file_path, str) and is_settings_local(file_path):
        return False
    return contains_value(written, values)


def main() -> None:
    payload = read_payload()
    if payload is None:
        return  # malformed payload -- never interfere with the tool call
    tool_name: Json = payload.get("tool_name")
    if not isinstance(tool_name, str):
        return
    command = bash_command(payload) or ""

    # First, whether or not anything is configured. The settings file itself
    # must not be staged, and must not be copied to a path outside the
    # gitignore rule.
    if command:
        if stages_settings_local(command):
            deny(STAGES_DENY_REASON)
            return
        if copies_settings_local(command):
            deny(COPIES_DENY_REASON)
            return

    values = read_configuration()
    if not values:
        return  # nothing configured here, so nothing to leak

    if command:
        if redirects_value(command, values):
            deny(REDIRECTION_DENY_REASON)
            return
        if records_value_in_git(command, values):
            deny(HISTORY_DENY_REASON)
            return
    if writes_value(tool_name, tool_input(payload), values):
        deny(WRITES_DENY_REASON)
        return


if __name__ == "__main__":
    main()
