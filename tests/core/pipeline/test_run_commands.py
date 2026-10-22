"""Test the ``mosaic run`` command line that refusals and plans print.

The line names one ``--entries`` per entry, sorted and without repeats, and quotes
each value that a shell would split. The ``<manifest>`` placeholder is printed bare, as
the slot for the manifest path.
"""

from __future__ import annotations

import shlex
import sys

from mosaic.core.pipeline.run_commands import mosaic_run_command

from tests.helpers import SOURCE_ROOT, imports_of


def test_the_command_names_the_kind_and_each_entry() -> None:
    """A nameless group prints as an empty token before the colon."""
    command = mosaic_run_command("export-joined", [("", "sess")])

    assert command == "mosaic run -m <manifest> --kind export-joined --entries :sess"


def test_entries_are_sorted_and_named_once() -> None:
    """The order and the repeats of the entries given leave the command unchanged."""
    entries = iter([("g", "b"), ("", "a"), ("g", "b")])

    command = mosaic_run_command("export-store", entries)

    assert command == (
        "mosaic run -m <manifest> --kind export-store --entries :a --entries g:b"
    )


def test_params_follow_the_entries_quoted() -> None:
    params = '{"reencode": true}'

    command = mosaic_run_command("export-joined", [("g", "s")], params=params)

    assert command.endswith(""" --entries g:s --params '{"reencode": true}'""")


def test_a_command_without_params_names_no_params_flag() -> None:
    assert "--params" not in mosaic_run_command("preprocess", [("g", "s")])


def test_a_manifest_path_and_an_entry_with_spaces_are_quoted() -> None:
    """Splitting the command as a shell does returns each value whole."""
    command = mosaic_run_command(
        "preprocess",
        [("my group", "s")],
        params="@/recipes/a recipe.json",
        manifest="/data/my study/dataset.yaml",
    )

    assert shlex.split(command) == [
        "mosaic",
        "run",
        "-m",
        "/data/my study/dataset.yaml",
        "--kind",
        "preprocess",
        "--entries",
        "my group:s",
        "--params",
        "@/recipes/a recipe.json",
    ]


def test_the_module_imports_the_standard_library_only() -> None:
    """A reader that prints a command loads no third-party package to build it.

    The scan reads function bodies and the ``TYPE_CHECKING`` block too, because an
    import deferred into the builder loads its package on the first command built.
    The one other import names the ``Entry`` alias from :mod:`mosaic.core.entry`,
    which imports the standard library only.
    """
    path = SOURCE_ROOT / "core" / "pipeline" / "run_commands.py"
    outside = {
        imported.module
        for imported in imports_of(path, scope="anywhere", relative=True)
        if imported.module.split(".")[0] not in sys.stdlib_module_names
    }

    assert outside == {"mosaic.core.entry"}
