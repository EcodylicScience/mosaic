"""Defining a feature in a dataset: the check in a child process, then the write.

:func:`define_feature` imports a source only in a child interpreter, which runs
``python -m mosaic.core.pipeline.definitions.declare``. The caller's registries
and ``sys.modules`` stay as they were, whatever the source does at import.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError

from mosaic.core.pipeline.definitions.loader import load_definition
from mosaic.core.pipeline.definitions.sidecar import DefinitionSidecar
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionFinding,
    DefinitionInvalid,
    DefinitionNameTaken,
    DefinitionRecord,
    check_definition_name,
    resolve_stored_slug,
    write_definition,
)
from mosaic.core.pipeline.graph.resolve import feature_class_for_slug

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.types.feature import Feature

__all__ = ["DECLARE_TIMEOUT_SECONDS", "define_feature", "load_feature_definition"]

DECLARE_TIMEOUT_SECONDS: Final = 300
"""Seconds the child that checks a source has before it is killed."""

_DECLARE_MODULE: Final = "mosaic.core.pipeline.definitions.declare"
_STDERR_LINES: Final = 5
"""The lines of a failed child's standard error that its finding quotes."""


class _Refusal(BaseModel):
    """What the child writes when it refuses a source."""

    findings: tuple[DefinitionFinding, ...]


def define_feature(ds: Dataset, name: str, source: str | bytes) -> DefinitionRecord:
    """Check *source* as the definition *name* and store it in *ds*.

    Defining a name again with other source adds a version and makes it current.
    Defining a source the name already holds reuses its version, writes no file but
    ``current`` when ``current`` named another version, and returns ``reused``.

    Args:
        ds: The dataset whose ``feature_definitions`` root holds the definition.
        name: The definition's name, lowercase kebab-case.
        source: The source, as text or as its UTF-8 bytes.

    Raises:
        DefinitionNameInvalid: *name* does not match the grammar.
        DefinitionNameTaken: *name* is the name of a built-in feature.
        DefinitionInvalid: The source fails a check, with every finding.
        DefinitionCorrupt: A stored file of the source's digest holds other bytes.
    """
    check_definition_name(name)
    if feature_class_for_slug(name) is not None:
        raise DefinitionNameTaken(name)
    try:
        data = source.encode("utf-8") if isinstance(source, str) else source
    except UnicodeEncodeError as exc:
        raise DefinitionInvalid.of_source(
            name, f"the source cannot be encoded as UTF-8: {exc}"
        ) from None
    sidecar = _declare_in_child(name, data)
    return write_definition(ds.get_root(DEFINITIONS_ROOT), name, data, sidecar)


def load_feature_definition(ds: Dataset, slug: str) -> type[Feature]:
    """Load the version of a definition in *ds* that *slug* names, and return its class.

    Args:
        ds: The dataset whose ``feature_definitions`` root holds the definition.
        slug: ``def:<name>`` for the current version, or ``def:<name>@<digest>``
            with the full digest or a prefix of at least 12 characters.

    Raises:
        DefinitionNameInvalid: *slug* is not a definition slug.
        DefinitionNotFound: No version matches *slug*.
        DefinitionDigestAmbiguous: The prefix matches more than one version.
        DefinitionCorrupt: A stored file is not what its name says.
        DefinitionInvalid: The source changed a registry or fails a check.
        DefinitionMismatch: The loaded class differs from its sidecar.
        DefinitionToolkitModelMissing: The class names a toolkit model this
            installation lacks.
    """
    root = ds.get_root(DEFINITIONS_ROOT)
    name, digest = resolve_stored_slug(root, slug)
    return load_definition(root, name, digest)


def _declare_in_child(name: str, source: bytes) -> DefinitionSidecar:
    """The sidecar that a child interpreter declares for *source*.

    The child inherits this process's environment and runs with ``-P``. The
    working directory is not on its import path.
    """
    with tempfile.TemporaryDirectory(prefix="mosaic-definition-") as directory:
        path = Path(directory) / "source.py"
        _ = path.write_bytes(source)
        try:
            completed = subprocess.run(
                [sys.executable, "-P", "-m", _DECLARE_MODULE, name, str(path)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=DECLARE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise DefinitionInvalid.of_source(
                name,
                f"checking the source ran past {DECLARE_TIMEOUT_SECONDS} seconds",
            ) from None
    try:
        if completed.returncode == 0:
            return DefinitionSidecar.model_validate_json(completed.stdout)
        refusal = _Refusal.model_validate_json(completed.stdout)
    except ValidationError:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        tail = "\n".join(stderr.splitlines()[-_STDERR_LINES:])
        raise DefinitionInvalid.of_source(
            name,
            f"checking the source ended with exit status {completed.returncode} "
            f"and no result: {tail}",
        ) from None
    raise DefinitionInvalid(name, refusal.findings)
