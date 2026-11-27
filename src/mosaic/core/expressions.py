"""Row expressions: arithmetic over table columns, parsed once and run by numpy.

An expression is Python syntax over column names, such as ``speed * 30 / 100`` or
``nan if isnull(x) else x ** 2``. :func:`parse_expression` checks the whole
tree against the accepted nodes, so a refused node fails when the expression is
declared. :func:`names_read` lists the columns an expression reads without
numpy, and :func:`evaluate` imports numpy and computes one value per row.

This module imports the standard library alone at module scope. The column
contract reads names through :func:`names_read` and loads no numpy for it.

The accepted syntax and what each node computes:

* ``+``, ``-``, ``*``, ``/``, ``//``, ``%`` and ``**`` are ``np.add``,
  ``np.subtract``, ``np.multiply``, ``np.true_divide``, ``np.floor_divide``,
  ``np.remainder`` and ``np.power``. Unary ``-`` and ``+`` are ``np.negative``
  and ``np.positive``. ``%`` is Python's floored modulo.
* ``<``, ``<=``, ``>``, ``>=``, ``==`` and ``!=`` are the numpy comparisons. A
  chain such as ``a < b < c`` joins its pairwise comparisons left to right with
  ``np.logical_and``.
* ``&``, ``|`` and ``~`` are ``np.logical_and``, ``np.logical_or`` and
  ``np.logical_not``, and ``x if c else y`` is ``np.where(c, x, y)``. Each
  refuses an operand that is not boolean when it is evaluated.
* An integer or float literal is ``np.float64``, ``True`` and ``False`` are
  ``np.bool_``, and ``pi``, ``nan`` and ``inf`` are constants no column can
  take the name of.
* ``abs``, ``sqrt``, ``exp``, ``log``, ``floor``, ``ceil``, ``sin``, ``cos``,
  ``deg2rad``, ``rad2deg`` and ``arccos`` are the unary ufunc of that name
  (``abs`` is ``np.absolute``), and ``arctan2``, ``minimum`` and ``maximum`` the
  binary one. ``isnull(a)`` and ``notnull(a)`` test for NaN, and are all false
  and all true on an integer or boolean operand. ``fillna(a, v)`` is
  ``np.where(isnull(a), v, a)``.
* Any other name is the column of that name.

Operations run in the order of the syntax tree, so ``speed * 30 / 100`` is
``(speed * 30) / 100``. Numpy's null rule holds: NaN propagates through
arithmetic, a comparison with NaN is false, and ``!=`` with NaN is true. A
division by zero gives ``inf`` or ``nan`` without a warning.

A column is read by its dtype. A numpy float, integer or boolean column is read
as it is. A nullable pandas number becomes float64 with NaN for NA. A nullable
pandas ``boolean`` column without NA is read as ``np.bool_``, and one with an NA
is refused, since a numpy boolean array cannot hold a null. A column of any
other dtype is refused.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray
    from pandas.api.extensions import ExtensionArray

    type ColumnValues = NDArray[np.generic] | pd.Series[Any] | ExtensionArray
    """What :func:`evaluate` reads a column from: a numpy array, a pandas series
    or a pandas extension array such as ``Int64`` or ``boolean``."""

    type Value = NDArray[np.generic] | np.generic
    """What an operation of an expression computes: an array, or a numpy scalar."""

__all__ = [
    "CONSTANTS",
    "ExpressionError",
    "ParsedExpression",
    "evaluate",
    "names_read",
    "parse_expression",
]


class ExpressionError(ValueError):
    """An expression that cannot be parsed, or cannot be evaluated on its columns."""


@dataclass(frozen=True, slots=True)
class ParsedExpression:
    """An expression whose tree holds accepted nodes only.

    Two parsed expressions are equal when their text is.

    Attributes:
        text: The expression as written.
        tree: Its syntax tree, checked by :func:`parse_expression`.
    """

    text: str
    tree: ast.Expression = field(compare=False, repr=False)


CONSTANTS: Final = frozenset({"pi", "nan", "inf"})
"""The constant names. Each is also the name of the attribute of ``numpy`` that
holds it, and no column can take one of them."""

_ARITHMETIC: Final[dict[type[ast.operator], str]] = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "true_divide",
    ast.FloorDiv: "floor_divide",
    ast.Mod: "remainder",
    ast.Pow: "power",
}
_BOOLEAN: Final[dict[type[ast.operator], str]] = {
    ast.BitAnd: "logical_and",
    ast.BitOr: "logical_or",
}
_REFUSED_OPERATORS: Final[dict[type[ast.operator], str]] = {
    ast.MatMult: "@",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitXor: "^",
}
_COMPARISONS: Final[dict[type[ast.cmpop], str]] = {
    ast.Lt: "less",
    ast.LtE: "less_equal",
    ast.Gt: "greater",
    ast.GtE: "greater_equal",
    ast.Eq: "equal",
    ast.NotEq: "not_equal",
}
_REFUSED_COMPARISONS: Final[dict[type[ast.cmpop], str]] = {
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}
_UNARY_FUNCTIONS: Final = {
    "abs": "absolute",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "floor": "floor",
    "ceil": "ceil",
    "sin": "sin",
    "cos": "cos",
    "deg2rad": "deg2rad",
    "rad2deg": "rad2deg",
    "arccos": "arccos",
}
_BINARY_FUNCTIONS: Final = ("arctan2", "minimum", "maximum")
_ARITY: Final = {
    **dict.fromkeys(_UNARY_FUNCTIONS, 1),
    **dict.fromkeys(_BINARY_FUNCTIONS, 2),
    "isnull": 1,
    "notnull": 1,
    "fillna": 2,
}
"""The accepted function names, each with the number of arguments it takes."""

_COMPREHENSIONS: Final = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_CONTAINERS: Final[dict[type[ast.expr], str]] = {
    ast.List: "list",
    ast.Tuple: "tuple",
    ast.Set: "set",
    ast.Dict: "dict",
}


def _refuse(text: str, what: str) -> ExpressionError:
    return ExpressionError(
        f"Expression {text!r} uses {what}, which an expression does not accept."
    )


def parse_expression(text: str) -> ParsedExpression:
    """Parses *text* and checks every node of it against the accepted syntax.

    Args:
        text: One expression, such as ``speed * 30 / 100``.

    Returns:
        The parsed expression.

    Raises:
        ExpressionError: *text* is not an expression, or holds a node that an
            expression does not accept. The message names the expression and the
            first refused node in the source.
    """
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ExpressionError(f"Expression {text!r} is not valid: {error}.") from error
    _check(tree.body, text)
    return ParsedExpression(text, tree)


def _check(node: ast.expr, text: str) -> None:
    """Raises :class:`ExpressionError` for the first refused node under *node*."""
    match node:
        case ast.BinOp(left=left, op=op, right=right):
            if type(op) in _REFUSED_OPERATORS:
                raise _refuse(text, f"the operator '{_REFUSED_OPERATORS[type(op)]}'")
            if type(op) not in _ARITHMETIC and type(op) not in _BOOLEAN:
                raise _refuse(text, f"the operator {type(op).__name__}")
            _check(left, text)
            _check(right, text)
        case ast.UnaryOp(op=op, operand=operand):
            if isinstance(op, ast.Not):
                raise _refuse(text, "'not'")
            _check(operand, text)
        case ast.BoolOp(op=op, values=values):
            for value in values:
                _check(value, text)
            raise _refuse(text, "'and'" if isinstance(op, ast.And) else "'or'")
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            _check(left, text)
            for op, comparator in zip(ops, comparators, strict=True):
                if type(op) in _REFUSED_COMPARISONS:
                    raise _refuse(text, f"'{_REFUSED_COMPARISONS[type(op)]}'")
                _check(comparator, text)
        case ast.IfExp(test=test, body=body, orelse=orelse):
            _check(body, text)
            _check(test, text)
            _check(orelse, text)
        case ast.Constant(value=value):
            if isinstance(value, bool) or (
                isinstance(value, (int, float)) and not isinstance(value, complex)
            ):
                return
            raise _refuse(text, _constant_kind(value))
        case ast.Name():
            return
        case ast.Call(func=func, args=args, keywords=keywords):
            _check_call(node, func, args, keywords, text)
        case ast.Attribute():
            raise _refuse(text, f"attribute access '.{node.attr}'")
        case ast.Subscript():
            raise _refuse(text, "a subscript")
        case ast.Starred():
            raise _refuse(text, "a starred argument")
        case ast.Lambda():
            raise _refuse(text, "a lambda")
        case ast.NamedExpr():
            raise _refuse(text, "an assignment expression")
        case ast.JoinedStr() | ast.FormattedValue():
            raise _refuse(text, "an f-string")
        case _ if isinstance(node, _COMPREHENSIONS):
            raise _refuse(text, "a comprehension")
        case _ if type(node) in _CONTAINERS:
            raise _refuse(text, f"a {_CONTAINERS[type(node)]}")
        case _:
            raise _refuse(text, f"the node {type(node).__name__}")


def _constant_kind(value: object) -> str:
    """Names a literal that an expression does not accept."""
    if isinstance(value, str):
        return "a string literal"
    if isinstance(value, bytes):
        return "a bytes literal"
    if isinstance(value, complex):
        return "a complex literal"
    if value is None:
        return "None"
    if value is Ellipsis:
        return "Ellipsis"
    return f"a literal of type {type(value).__name__}"


def _check_call(
    node: ast.Call,
    func: ast.expr,
    args: list[ast.expr],
    keywords: list[ast.keyword],
    text: str,
) -> None:
    if isinstance(func, ast.Attribute):
        raise _refuse(text, f"attribute access '.{func.attr}'")
    if not isinstance(func, ast.Name):
        raise _refuse(text, f"the call '{ast.unparse(func)}(...)'")
    if func.id not in _ARITY:
        raise _refuse(
            text, f"a call of '{func.id}', which is not a function of expressions"
        )
    if keywords:
        raise _refuse(text, "a keyword argument")
    for arg in args:
        if isinstance(arg, ast.Starred):
            raise _refuse(text, "a starred argument")
    if len(args) != _ARITY[func.id]:
        raise _refuse(
            text,
            f"'{func.id}' with {len(args)} arguments, where it takes {_ARITY[func.id]}",
        )
    for arg in args:
        _check(arg, text)


def names_read(parsed: ParsedExpression) -> tuple[str, ...]:
    """The columns *parsed* reads, in the order the source first names them.

    ``pi``, ``nan``, ``inf`` and the names of called functions are not columns,
    and a name that repeats is listed once.
    """
    called = {
        id(node.func) for node in ast.walk(parsed.tree) if isinstance(node, ast.Call)
    }
    reads = sorted(
        (node.lineno, node.col_offset, node.id)
        for node in ast.walk(parsed.tree)
        if isinstance(node, ast.Name)
        and id(node) not in called
        and node.id not in CONSTANTS
    )
    return tuple(dict.fromkeys(name for _, _, name in reads))


def evaluate(
    parsed: ParsedExpression, columns: Mapping[str, ColumnValues], n_rows: int
) -> NDArray[np.generic]:
    """Computes one value per row of *parsed* over *columns*.

    Each column the expression reads is converted by its dtype before any
    operation runs. A scalar result is broadcast to *n_rows* values. The result
    has the dtype numpy gives the operations on the converted columns. Literals
    are float64, so ``x * 2`` on a float32 column gives float64, and a comparison
    gives bool.

    Args:
        parsed: The expression.
        columns: The table's columns by name, each with *n_rows* values. A numpy
            array or a pandas array or series.
        n_rows: The number of rows.

    Returns:
        An array of *n_rows* values.

    Raises:
        ExpressionError: A column is missing, has a dtype an expression does not
            read, or has the wrong length, or ``&``, ``|``, ``~`` or a condition takes
            an operand that is not boolean, or numpy refuses an operation.
    """
    import numpy as np

    text = parsed.text
    read: dict[str, Value] = {}
    for name in names_read(parsed):
        if name not in columns:
            raise ExpressionError(
                f"Expression {text!r} reads the column {name!r}, which the table does not have."
            )
        read[name] = _read_column(text, name, columns[name], n_rows)

    def compute(node: ast.expr) -> Value:
        match node:
            case ast.Constant(value=bool() as flag):
                return np.bool_(flag)
            case ast.Constant(value=int() | float() as number):
                return np.float64(number)
            case ast.Name(id=name):
                if name in CONSTANTS:
                    constant: float = getattr(np, name)
                    return np.float64(constant)
                return read[name]
            case ast.BinOp(left=left, op=op, right=right):
                x, y = compute(left), compute(right)
                if type(op) in _BOOLEAN:
                    _need_boolean(
                        text, "&" if isinstance(op, ast.BitAnd) else "|", x, y
                    )
                    return _apply(_BOOLEAN[type(op)], x, y)
                return _apply(_ARITHMETIC[type(op)], x, y)
            case ast.UnaryOp(op=op, operand=operand):
                x = compute(operand)
                if isinstance(op, ast.Invert):
                    _need_boolean(text, "~", x)
                    return _apply("logical_not", x)
                return _apply("negative" if isinstance(op, ast.USub) else "positive", x)
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                operands = [compute(left), *(compute(each) for each in comparators)]
                pairs = [
                    _apply(_COMPARISONS[type(op)], operands[index], operands[index + 1])
                    for index, op in enumerate(ops)
                ]
                return reduce(lambda a, b: _apply("logical_and", a, b), pairs)
            case ast.IfExp(test=test, body=body, orelse=orelse):
                condition = compute(test)
                _need_boolean(text, "the condition of 'if else'", condition)
                return np.where(condition, compute(body), compute(orelse))
            case ast.Call(func=ast.Name(id=function), args=args):
                return _call(function, [compute(arg) for arg in args])
            case _:
                raise AssertionError(ast.dump(node))  # parse_expression admits no other

    try:
        with np.errstate(all="ignore"):
            result = compute(parsed.tree.body)
    except ExpressionError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise ExpressionError(
            f"Expression {text!r} could not be evaluated: {error}."
        ) from error
    if np.ndim(result) == 0:
        return np.full(n_rows, result)
    return np.asarray(result)


def _apply(function: str, *operands: Value) -> Value:
    """The numpy function called *function*, applied to *operands*."""
    import numpy as np

    operation: np.ufunc = getattr(np, function)
    result: Value = operation(*operands)
    return result


def _need_boolean(text: str, where: str, *operands: Value) -> None:
    """Refuses an operand that is not boolean for ``&``, ``|``, ``~`` or a condition."""
    import numpy as np

    for operand in operands:
        kind = np.asarray(operand).dtype.kind
        if kind != "b":
            raise ExpressionError(
                f"Expression {text!r} gives {where} a {np.asarray(operand).dtype} operand, "
                f"which must be boolean. Compare it first, for example with '> 0'."
            )


def _call(function: str, values: list[Value]) -> Value:
    """Applies the function of an expression to its evaluated arguments."""
    import numpy as np

    if function in _UNARY_FUNCTIONS:
        return _apply(_UNARY_FUNCTIONS[function], *values)
    if function in _BINARY_FUNCTIONS:
        return _apply(function, *values)
    if function == "isnull":
        return _isnull(values[0])
    if function == "notnull":
        return _apply("logical_not", _isnull(values[0]))
    return np.where(_isnull(values[0]), values[1], values[0])


def _isnull(value: Value) -> Value:
    """``np.isnan`` of a float, and all false on an integer or boolean."""
    import numpy as np

    if np.asarray(value).dtype.kind == "f":
        return _apply("isnan", value)
    scalar: Value = np.full(np.shape(value), False)[()]
    return scalar


def _read_column(
    text: str, name: str, column: ColumnValues, n_rows: int
) -> NDArray[np.generic]:
    """Converts a column to the numpy array an operation reads, by its dtype."""
    import numpy as np

    dtype: object = getattr(column, "dtype", None)
    kind: object = getattr(dtype, "kind", None)
    if isinstance(dtype, np.dtype):
        if dtype.kind not in "fiub":
            raise _unreadable(text, name, f"has dtype {dtype}")
        values: NDArray[np.generic] = np.asarray(column)
    elif kind in ("i", "u", "f", "b"):
        import pandas as pd
        from pandas.api.extensions import ExtensionArray

        if not isinstance(column, pd.Series | ExtensionArray):
            raise _unreadable(text, name, f"has dtype {dtype}")
        if kind == "b":
            if bool(np.any(column.isna())):
                raise _unreadable(
                    text,
                    name,
                    "is a boolean column with missing values, which numpy cannot hold",
                )
            values = column.to_numpy(dtype=bool)
        else:
            values = column.to_numpy(dtype="float64", na_value=np.nan)
    else:
        raise _unreadable(text, name, f"has dtype {dtype}")
    if values.shape != (n_rows,):
        raise ExpressionError(
            f"Expression {text!r} reads the column {name!r} with shape {values.shape}, "
            f"where the table has {n_rows} rows."
        )
    return values


def _unreadable(text: str, name: str, why: str) -> ExpressionError:
    return ExpressionError(
        f"Expression {text!r} reads the column {name!r}, which {why}. An expression "
        f"reads numeric and boolean columns."
    )
