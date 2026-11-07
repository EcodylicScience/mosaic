"""A converter loads a supplied file without unpickling it.

A raw track or label file comes from an upload, so a converter that loads one
through ``np.load(..., allow_pickle=True)``, ``pd.read_hdf``,
``pd.read_pickle``, ``pd.HDFStore`` or ``pickle`` runs whatever code the file
carries. Every such load sits in ``core/track_library/safe_load.py``, behind the
checks that decide a file is safe to read or the trust flag that says it is not
a concern. The converters call that module.

The scan reads each file's syntax tree, so a docstring that names ``read_hdf``
does not trip it, and an alias or a keyword spelling is caught as well as the
plain call.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.helpers import SOURCE_ROOT

SCANNED: Final = (
    SOURCE_ROOT / "core" / "track_library",
    SOURCE_ROOT / "behavior" / "label_library",
)
"""The directories whose modules read supplied files."""

EXEMPT: Final = SOURCE_ROOT / "core" / "track_library" / "safe_load.py"
"""The one module that may unpickle, because it checks first."""

PICKLE_MODULES: Final = frozenset(
    {"pickle", "_pickle", "cPickle", "dill", "cloudpickle", "joblib"}
)
UNPICKLING_CALLS: Final = frozenset({"load", "loads"})
UNPICKLING_NAMES: Final = frozenset(
    {"HDFStore", "Unpickler", "read_hdf", "read_pickle"}
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One spelling that unpickles, and the line it sits on."""

    line: int
    what: str


def _pickle_aliases(tree: ast.Module) -> frozenset[str]:
    """The names that *tree* binds to a pickle module."""
    bound = set(PICKLE_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in PICKLE_MODULES
            )
    return frozenset(bound)


def _allows_pickle(call: ast.Call) -> bool:
    """Whether *call* passes ``allow_pickle`` as anything but a literal ``False``."""
    return any(
        keyword.arg == "allow_pickle"
        and not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is False
        )
        for keyword in call.keywords
    )


def _referenced_name(node: ast.AST) -> str | None:
    """The name that *node* spells, for a name, an attribute or an imported name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.alias):
        return node.name.rpartition(".")[2]
    return None


def unpickling_spellings(source: str) -> list[Finding]:
    """Every spelling in *source* that can unpickle a file, in line order.

    Flags a call that passes ``allow_pickle`` as anything but a literal ``False``,
    whatever it calls or is named, a ``load`` or ``loads`` off a pickle module, and
    any reference to ``read_hdf``, ``read_pickle``, ``HDFStore`` or ``Unpickler`` by
    name, attribute or import, which covers a subclass of ``pickle.Unpickler``.
    """
    tree = ast.parse(source)
    pickles = _pickle_aliases(tree)
    found: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _allows_pickle(node):
            found.append(Finding(node.lineno, "a call with allow_pickle"))
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in UNPICKLING_CALLS
            and isinstance(node.value, ast.Name)
            and node.value.id in pickles
        ):
            found.append(Finding(node.lineno, f"{node.value.id}.{node.attr}"))
        elif isinstance(node, ast.ImportFrom) and node.module in PICKLE_MODULES:
            found.extend(
                Finding(node.lineno, f"{node.module}.{alias.name}")
                for alias in node.names
                if alias.name in UNPICKLING_CALLS
            )
        elif (name := _referenced_name(node)) in UNPICKLING_NAMES:
            found.append(Finding(getattr(node, "lineno", 0), str(name)))
    return sorted(found, key=lambda finding: finding.line)


def _converter_modules() -> list[Path]:
    """Every module under the scanned directories except the exempt one."""
    return [
        path
        for directory in SCANNED
        for path in sorted(directory.rglob("*.py"))
        if path != EXEMPT
    ]


def test_no_converter_unpickles_a_file() -> None:
    offenders = [
        f"{path.relative_to(SOURCE_ROOT.parent.parent)}:{finding.line}: {finding.what}"
        for path in _converter_modules()
        for finding in unpickling_spellings(path.read_text(encoding="utf-8"))
    ]
    report = (
        "A converter loads a supplied file in a way that unpickles it:\n  "
        + "\n  ".join(offenders)
        + "\nLoad it through core/track_library/safe_load.py, which checks the file "
        "first, and put no unpickling call anywhere else."
    )
    assert not offenders, report


def test_the_exempt_module_is_the_one_that_unpickles() -> None:
    """The exemption names a module that exists and needs it."""
    assert EXEMPT.is_file()
    assert unpickling_spellings(EXEMPT.read_text(encoding="utf-8"))


def test_the_scan_covers_the_converters() -> None:
    names = {path.name for path in _converter_modules()}
    assert {"trex.py", "helpers.py", "deeplabcut.py", "calms21_behavior.py"} <= names
    assert "safe_load.py" not in names


@pytest.mark.parametrize(
    "source",
    [
        "np.load(path, allow_pickle=True)",
        "numpy.load(path, allow_pickle=True).item()",
        "np.load(path, allow_pickle=flag)",
        "from numpy import load\nload(path, allow_pickle=True)",
        "from numpy import load as nl\nnl(path, allow_pickle=True)",
        "np.lib.format.read_array(handle, allow_pickle=True)",
        "pd.read_hdf(path)",
        "from pandas import read_hdf",
        "pd.read_pickle(path)",
        "from pandas import read_pickle",
        "store = pd.HDFStore(path)",
        "from pandas import HDFStore",
        "import pickle\npickle.load(handle)",
        "import pickle\npickle.loads(blob)",
        "import pickle as pk\npk.loads(blob)",
        "from pickle import loads",
        "import joblib\njoblib.load(path)",
        "import dill\ndill.loads(blob)",
        "import cloudpickle\ncloudpickle.load(handle)",
        "from joblib import load",
        "import pickle\nclass Reader(pickle.Unpickler): ...",
        "from pickle import Unpickler\nclass Reader(Unpickler): ...",
    ],
)
def test_the_scan_flags_each_spelling(source: str) -> None:
    assert unpickling_spellings(source)


@pytest.mark.parametrize(
    "source",
    [
        "np.load(path, allow_pickle=False)",
        "np.load(path)",
        "json.load(handle)",
        'def f():\n    """Reads with pd.read_hdf, pickle.load and HDFStore."""',
        "# pd.read_hdf(path) and np.load(path, allow_pickle=True)",
        "message = 'read_hdf unpickles'",
        "import pickle\nerror = pickle.UnpicklingError",
    ],
)
def test_the_scan_passes_what_does_not_unpickle(source: str) -> None:
    assert not unpickling_spellings(source)


def test_a_finding_names_the_line_of_the_call() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "",
            "",
            "def read(path):",
            "    return np.load(path, allow_pickle=True)",
        ]
    )

    assert [finding.line for finding in unpickling_spellings(source)] == [5]
