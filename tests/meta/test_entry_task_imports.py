"""The entry task writes one output, and its module imports nothing that writes more.

A worker runs ``core/pipeline/entry_task.py`` and may write its entry's output and
nothing else: no index, no run-log, no claim. The module therefore names none of the
modules that write them in its own imports, under ``TYPE_CHECKING`` included. Modules
it imports may reach them, and this test does not follow those.
"""

from __future__ import annotations

from tests.helpers import SOURCE_ROOT, imports_reaching, package_of, source_tree

ENTRY_TASK = SOURCE_ROOT / "core" / "pipeline" / "entry_task.py"

FORBIDDEN = (
    "mosaic.core.pipeline.run",
    "mosaic.core.pipeline.job",
    "mosaic.runlog",
    "mosaic.core.pipeline.entry_claim",
    "mosaic.core.pipeline.index",
    "mosaic.core.pipeline.index_csv",
    "mosaic.core.pipeline.index_lock",
    "mosaic.core.pipeline.tracks_index",
    "mosaic.core.pipeline.media_index",
    "mosaic.core.pipeline.labels_index",
    "mosaic.core.pipeline.arena_index",
    "mosaic.core.pipeline.sequence_index",
    "mosaic.core.pipeline.dataset_indexes",
)
"""The run, the run-log, the claims and the index modules."""


def _module_file(module: str) -> str:
    return "/".join(module.removeprefix("mosaic.").split(".")) + ".py"


def test_each_forbidden_module_exists() -> None:
    """A renamed module would leave its name here matching nothing."""
    assert [m for m in FORBIDDEN if not (SOURCE_ROOT / _module_file(m)).is_file()] == []


def test_the_entry_task_imports_no_run_log_claim_or_index_module() -> None:
    offenders = [
        f"{imported.line}: {imported.statement}"
        for imported in imports_reaching(
            source_tree(ENTRY_TASK),
            package_of(ENTRY_TASK),
            literal_packages=("mosaic",),
        )
        if any(imported.into(module) for module in FORBIDDEN)
    ]

    assert offenders == [], (
        f"entry_task.py imports a module that writes beyond its output: {offenders}"
    )
