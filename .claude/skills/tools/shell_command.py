"""Read a shell command line as the shell reads it."""

import re

# A line continuation: a backslash-newline whose backslash is not itself
# escaped. An even run of backslashes before a newline escapes itself and
# leaves the newline standing.
CONTINUATION = re.compile(r"(?<!\\)(?:\\\\)*\\\n")


# HEREDOC_INTRODUCER matches the `<<WORD` / `<<-WORD` / `<<'WORD'` / `<<\WORD`
# that opens a heredoc. Its first three groups match a quoted delimiter and its
# fourth an unquoted one. Two lookalikes must not match it, and the pattern
# alone rules out neither. Herestrings like `<<<'x'` are rejected before the
# scan. An anchor one character later reads the trailing `<<` as an introducer
# and takes `x` as a delimiter that does not arrive. Arithmetic like `$((a <<
# b))` is rejected by blanking the whole `$(( ))` span first, which also stops
# a bitwise `|` inside one from reading as a pipe.
HEREDOC_INTRODUCER = re.compile(
    r"<<-?\s*(?:'([^']*)'|\"([^\"]*)\"|\\([A-Za-z_][\w-]*)|([A-Za-z_][\w-]*))"
)


def arithmetic_end(cmd: str, start: int) -> int:
    """Return the index just past the first of the two parentheses closing an
    arithmetic expansion.

    Counting starts inside the outer parenthesis. The outer `)` of `$((a))`
    is then left standing, and the skeleton keeps a stray `)`. A bare `)` is
    neither an operator nor a word, and readers of the skeleton pass over it.

    Unbalanced parentheses blank the rest of the command. Such a command cannot
    run at all. Blanking is also the direction that cannot manufacture a
    command position out of text that was never code.
    """
    depth = 0
    for i in range(start + 1, len(cmd)):
        if cmd[i] == "(":
            depth += 1
        elif cmd[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(cmd)


def heredoc_body_end(cmd: str, start: int, delimiter: str, dashed: bool) -> int:
    """Return the index of the newline ending the terminator line, or the end
    of the command.

    Missing delimiters blank the rest of the command. That direction is safe,
    because blanking only ever removes operators.
    """
    i, n = start, len(cmd)
    while i <= n:
        newline = cmd.find("\n", i)
        line_end = n if newline < 0 else newline
        line = cmd[i:line_end]
        if (line.lstrip("\t") if dashed else line) == delimiter:
            return line_end
        if newline < 0:
            return n
        i = newline + 1
    return n


# A `#` starts a comment when it starts a word: first in the command, or
# after a blank or one of the characters that end a word.
COMMENT_FOLLOWS = " \t\n;&|()<>"


# The characters that an unquoted backslash makes literal and the skeleton
# blanks: quotes, operator characters, `#` and `$`. An escaped one opens no
# span, separates no commands and starts no comment or expansion.
ESCAPE_BLANKS = "'\";&|()<>#$"


# Command substitutions nest at most this deep inside double quotes. A deeper
# one is read as an unterminated quote, and the recursion stays bounded.
MAXIMUM_NESTING = 32


def blanked(
    cmd: str, start: int, skeleton: bool, depth: int
) -> tuple[str, int, list[int]]:
    """Return `cmd` from `start` with its data blanked, the index where the
    scan stopped, and the offset of each heredoc introducer that it read.

    Comments are always blanked. With `skeleton` set, quoted spans, arithmetic
    expansions, every heredoc body and each escaped character of ESCAPE_BLANKS
    are blanked as well. Without it, quoted spans and arithmetic expansions are
    kept as written, and so is a heredoc body whose delimiter is unquoted,
    where the shell expands `$(...)`. An unquoted backslash and the character
    after it read as one escaped character. A `#` after it starts no comment,
    and a `'` after an escaped `$` opens a plain single-quoted span. A
    backslash-newline is kept as written. At depth 0 the scan runs to the end
    of the command. Deeper, it reads the body of a `$(...)` inside double
    quotes and returns the index of its closing `)`. A `$(...)` inside double
    quotes is read as a separate command, with its quotes, heredocs and nested
    substitutions, and a `"` inside it does not close the outer quote.
    """
    out: list[str] = []
    introducers: list[int] = []
    pending: list[tuple[str, bool, bool]] = []
    parens = 0
    # The index just past the last escaped character.
    escape_end = -1
    i, n = start, len(cmd)
    while i < n:
        c = cmd[i]
        if c == "\\" and i + 1 < n and cmd[i + 1] != "\n":
            pair = cmd[i : i + 2]
            out.append("  " if skeleton and pair[1] in ESCAPE_BLANKS else pair)
            i += 2
            escape_end = i
            continue
        if c == "\n" and pending:
            delimiter, dashed, quoted = pending.pop(0)
            end = heredoc_body_end(cmd, i + 1, delimiter, dashed)
            out.append(" " * (end - i) if skeleton or quoted else cmd[i:end])
            i = end
            continue
        if c == "$" and cmd.startswith("$((", i):
            end = arithmetic_end(cmd, i + 1)
            out.append(" " * (end - i) if skeleton else cmd[i:end])
            i = end
            # The outer `)` of the expansion is left standing, and this count
            # cancels it.
            parens += 1
            continue
        if c == "<" and cmd.startswith("<<<", i):
            # Herestrings emit all three characters at once. A rescan of the
            # trailing `<<` registered `<<<'x'` as a heredoc named `x`.
            out.append("<<<")
            i += 3
            continue
        if c == "<" and cmd.startswith("<<", i):
            introducer = HEREDOC_INTRODUCER.match(cmd, i)
            if introducer is not None:
                quoted = introducer.group(4) is None
                word = next(group for group in introducer.groups() if group is not None)
                pending.append((word, cmd.startswith("<<-", i), quoted))
                introducers.append(i)
                # Blank the introducer itself. Redirections may precede the
                # command name, and `<<EOF make -j8` runs `make`. Blanking
                # keeps `make` as the first token of the skeleton.
                out.append(" " * (introducer.end() - i))
                i = introducer.end()
                continue
        if c == "#" and i != escape_end and (i == 0 or cmd[i - 1] in COMMENT_FOLLOWS):
            end = cmd.find("\n", i)
            end = n if end < 0 else end
            out.append(" " * (end - i))
            i = end
            continue
        if c == "'" or c == '"':
            escapes = c == '"' or (i > 0 and cmd[i - 1] == "$" and i != escape_end)
            piece = [c]
            j = i + 1
            while j < n:
                if c == '"' and cmd.startswith("$(", j) and not cmd.startswith("$((", j):
                    if depth >= MAXIMUM_NESTING:
                        j = n
                        break
                    inner, close, inner_introducers = blanked(
                        cmd, j + 2, skeleton, depth + 1
                    )
                    introducers.extend(inner_introducers)
                    piece.append("$(" + inner + cmd[close : close + 1])
                    j = close + 1
                    continue
                if escapes and cmd[j] == "\\" and j + 1 < n:
                    piece.append(cmd[j : j + 2])
                    j += 2
                    continue
                if cmd[j] == c:
                    break
                piece.append(cmd[j])
                j += 1
            if j >= n:
                out.append(cmd[i:])
                return "".join(out), n, introducers
            piece.append(c)
            out.append(" " * (j - i + 1) if skeleton else "".join(piece))
            i = j + 1
            continue
        if c == "(":
            parens += 1
        elif c == ")":
            if depth > 0 and parens == 0:
                return "".join(out), i, introducers
            parens -= 1
        out.append(c)
        i += 1
    return "".join(out), n, introducers


def strip_quoted(cmd: str) -> str:
    """Blank quoted spans, comments and heredoc bodies so text that is data
    cannot read as code.

    Shell operators inside any of them are not top-level boundaries, and lines
    inside a heredoc body are not commands however they start. Each blanked
    span becomes an equal run of spaces. Spans are the quotes and their
    contents, a comment up to its newline, or a body from the newline of the
    introducer through its terminator word. The result is the same length as
    the command, and a top-level pipe or `;` outside the spans survives.
    Blanking a heredoc keeps the newline that ends its terminator line, and a
    command after the heredoc still starts a step. A backslash escapes the next
    character inside double quotes and inside `$'...'`, where `\\'` does not
    close the span. Inside plain single quotes it is literal. Outside quotes it
    escapes the next character, and an escaped quote, operator character, `#`
    or `$` is blanked with its backslash. A `$(...)` inside double quotes ends
    where its parenthesis closes, and the quotes and heredocs inside it are
    read as they are at top level.

    A heredoc body is data. `cat > doc.md <<'EOF'` followed by a line reading
    `heavy npm run build | tail` writes that line and runs neither command.

    The masking is best-effort in two places, and both err toward blanking
    more. Unterminated quotes, and substitutions nested deeper than
    MAXIMUM_NESTING, leave the remainder unmasked instead of raising.
    Heredocs whose terminator does not appear blank to the end of the command,
    including ones written with CRLF line endings, where `EOF\\r` is not `EOF`.
    Masking only ever removes boundary and pipe characters, and it cannot
    manufacture a spurious top-level pipe.
    """
    return skeleton_and_introducers(cmd)[0]


def strip_unquoted_data(cmd: str) -> str:
    """Blank comments and the heredoc bodies whose delimiter is quoted, and
    keep quoted spans and arithmetic expansions as written.

    The result has the same length as the command and the same offsets as
    strip_quoted. It shows the words that the shell runs, including the ones
    inside a quoted `bash -c` script and the substitutions that an arithmetic
    expansion or an unquoted heredoc body expands.
    """
    return blanked(cmd, 0, False, 0)[0]


def skeleton_and_introducers(cmd: str) -> tuple[str, list[int]]:
    """Return the skeleton of strip_quoted and the offset of each heredoc
    introducer, in ascending order.

    An introducer inside a `$(...)` inside double quotes counts. A herestring
    and a `<<` inside an arithmetic expansion do not.
    """
    skeleton, _, introducers = blanked(cmd, 0, True, 0)
    return skeleton, introducers


def joined_continuations(command: str) -> str:
    """Return `command` without the backslash-newline of each line
    continuation.

    Continuations are found in the skeleton of strip_quoted, whose offsets
    apply to the command. A backslash-newline inside a quote or a heredoc body
    is data and stays in place.

    A backslash-newline continues the line and does not separate commands.
    `git add \\` then ` -f .` is one command.

    The halves join without a space, as the shell joins them. `git ad\\` then
    `d -f .` runs `git add -f .`. Joining with a space reads that as the three
    words `git`, `ad`, `d`. A space does not help anywhere else either,
    because two separate words already have a space on one side of the break.
    """
    pieces: list[str] = []
    start = 0
    for match in CONTINUATION.finditer(strip_quoted(command)):
        pieces.append(command[start:match.end() - 2])
        start = match.end()
    pieces.append(command[start:])
    return "".join(pieces)


def operator_spans(skeleton: str, operators: re.Pattern[str]) -> list[tuple[int, int]]:
    """Return the (start, end) offsets of the pieces between the matches of
    `operators` in a skeleton.

    The skeleton of strip_quoted has the same length as the command, and the
    offsets apply to the command and to every view of it of that length. An
    operator inside a quote, a comment or a heredoc body cuts nothing, and a
    heredoc body stays in the last piece of its introducer's line. An empty
    piece between two adjacent operators is kept.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in operators.finditer(skeleton):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(skeleton)))
    return spans


def cut_at_operators(command: str, operators: re.Pattern[str]) -> list[tuple[str, str]]:
    """Cut `command` at the operators in its skeleton, and return each piece
    as a (raw, skeleton) pair with its whitespace kept."""
    skeleton = strip_quoted(command)
    return [
        (command[start:end], skeleton[start:end])
        for start, end in operator_spans(skeleton, operators)
    ]


# An ANSI-C quoted span, `$'...'`, inside which a backslash escapes the next
# character.
ANSI_C_SPAN = re.compile(r"\$'((?:[^\\']|\\.)*)'", re.DOTALL)

# The escapes that bash decodes inside `$'...'`: a named character, an octal
# byte, a hexadecimal byte or code point, or a control character.
ANSI_C_ESCAPE = re.compile(
    r"\\(?:([abeEfnrtv\\'\"?])|([0-7]{1,3})|x([0-9A-Fa-f]{1,2})|u([0-9A-Fa-f]{1,4})"
    + r"|U([0-9A-Fa-f]{1,8})|c(.))",
    re.DOTALL,
)

# The characters that the named escapes of ANSI_C_ESCAPE stand for.
ANSI_C_NAMED = {
    "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r",
    "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?",
}


def ansi_c_character(escape: re.Match[str]) -> str:
    """Return the character that one ANSI_C_ESCAPE match stands for.

    A code point past the Unicode range is left as written.
    """
    named, octal, byte, short_point, long_point, control = escape.groups()
    if named is not None:
        return ANSI_C_NAMED[named]
    if control is not None:
        return chr(ord(control) & 0x1F)
    if octal is not None:
        return chr(int(octal, 8))
    code = int(byte or short_point or long_point, 16)
    return chr(code) if code <= 0x10FFFF else escape.group(0)


def decoded_ansi_c(text: str) -> str:
    """Return `text` with each `$'...'` span replaced by the string that bash
    makes of it.

    `$'x\\n<word>'` puts `<word>` on a separate line, where the raw text runs
    it on from `\\n`. Spans are found without regard to the quoting around
    them. A `$'` inside double quotes is literal to the shell, and is decoded
    anyway.
    """
    return ANSI_C_SPAN.sub(
        lambda span: ANSI_C_ESCAPE.sub(ansi_c_character, span.group(1)), text
    )


def program_name(token: str) -> str:
    """Return the program that `token` names, without its directory."""
    return token.rsplit("/", 1)[-1]
