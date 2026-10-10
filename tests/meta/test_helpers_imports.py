"""``import tests.helpers`` does not import the feature library.

Every test module and ``tests/conftest.py`` import ``tests.helpers``, in the
controller and in each xdist worker. Importing the feature library costs about
two seconds. A helper imports it inside the function that needs it.
``tests/conftest.py`` imports the feature library only to locate an environment
that a CI job builds.
"""

from __future__ import annotations

import subprocess
import sys

from tests.helpers import REPO_ROOT

_FEATURE_LIBRARY = "mosaic.behavior.feature_library"

_REPORT_FEATURE_LIBRARY_MODULES = f"""
import sys

import tests.helpers

loaded = sorted(
    name
    for name in sys.modules
    if name == "{_FEATURE_LIBRARY}" or name.startswith("{_FEATURE_LIBRARY}.")
)
print(loaded or "none")
"""


def test_importing_the_helpers_does_not_import_the_feature_library() -> None:
    """Imports ``tests.helpers`` in a fresh interpreter and lists the modules loaded.

    The check runs in a fresh interpreter, because this process has imported the
    feature library already. ``-c`` places the working directory, the repository
    root, first on ``sys.path``.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _REPORT_FEATURE_LIBRARY_MODULES],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "none", (
        f"import tests.helpers loads {completed.stdout.strip()}. Move the helper's "
        "feature-library import into the function that uses it. "
        '`python -X importtime -c "import tests.helpers"` names the importer.'
    )
