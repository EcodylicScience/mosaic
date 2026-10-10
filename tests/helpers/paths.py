"""Where the repository and the suite's own data are, wherever a test file sits.

A test that finds the repository by counting parents from its own ``__file__``
breaks when the file moves into a subdirectory, and some break silently: a grep
pointed at a directory that no longer exists finds nothing and passes. Every
path the suite needs outside a test's ``tmp_path`` is derived here instead, from
this module's fixed place in ``tests/helpers/``.

The programs of the external tools are given relative to the package root. A
separation check reads them under the imported package as well as under
:data:`SOURCE_ROOT`, and a test builds an absolute path as ``SOURCE_ROOT /
KPMS_SERVER``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

TESTS_ROOT: Final = Path(__file__).resolve().parent.parent
"""The ``tests`` directory."""

REPO_ROOT: Final = TESTS_ROOT.parent
"""The repository checkout the suite belongs to."""

SOURCE_ROOT: Final = REPO_ROOT / "src" / "mosaic"
"""The ``mosaic`` package's source directory in that checkout."""

GOLDEN_DIR: Final = TESTS_ROOT / "data"
"""The golden files that pin identifiers and converter output."""

FEATURE_RUNNERS: Final = "behavior/feature_library/external"
"""The directory of the feature runners and the modules they share.

It is also the keypoint-MoSeq environment, which ``uv sync`` builds there.
"""

RUNNER_WIRE: Final = f"{FEATURE_RUNNERS}/runner_wire.py"
"""The runner side of the line protocol, shared by every feature runner."""

KPMS_SERVER: Final = f"{FEATURE_RUNNERS}/kpms_server.py"
"""The keypoint-MoSeq runner program."""

KPMS_PROTOCOL: Final = f"{FEATURE_RUNNERS}/kpms_protocol.py"
"""The commands and payloads of the keypoint-MoSeq runner."""

VAME_RUNNER: Final = f"{FEATURE_RUNNERS}/vame_runner.py"
"""The VAME runner program."""

VAME_PROTOCOL: Final = f"{FEATURE_RUNNERS}/vame_protocol.py"
"""The commands and payloads of the VAME runner."""

ULTRALYTICS_RUNNER: Final = "tracking/external/runner/ultralytics_runner.py"
"""The Ultralytics runner program, which the Ultralytics and POLO environments run."""

ULTRALYTICS_PROTOCOL: Final = "tracking/external/runner/ultralytics_protocol.py"
"""The request and response of the Ultralytics runner."""
