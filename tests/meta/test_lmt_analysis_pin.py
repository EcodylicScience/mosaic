"""The three documents that name the pinned lmt-analysis commit name the same one.

The install page, the import guide and the environment's README each tell the
user which lmt-analysis commit to check out or to pass as ``lmt_analysis_commit``.
Nothing else ties them together, so a commit changed in one leaves the others
describing an environment the rebuild op does not get.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from tests.helpers import REPO_ROOT

_PINNING_FILES: Final = (
    Path("docs", "installation.md"),
    Path("docs", "guides", "tracking", "import.md"),
    Path("src", "mosaic", "tracking", "external", "lmt-env", "README.md"),
)
_COMMIT: Final = re.compile(r"\b[0-9a-f]{40}\b")


def test_each_document_names_one_lmt_analysis_commit_and_all_name_the_same() -> None:
    """Each file holds exactly one 40-digit commit, and the three are equal."""
    found = {
        str(file): _COMMIT.findall((REPO_ROOT / file).read_text(encoding="utf-8"))
        for file in _PINNING_FILES
    }

    one_each = all(len(commits) == 1 for commits in found.values())
    equal = len({commit for commits in found.values() for commit in commits}) == 1

    assert one_each and equal, (
        "each of these files names exactly one lmt-analysis commit, and the "
        f"commits are equal; found {found}"
    )
