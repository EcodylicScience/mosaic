"""The classifier guide's recipe page stays short enough to read in one pass.

An agent starts a classifier from ``train-a-classifier.md`` alone. The page
gives the recipe and links, and the detail lives on the pages it links to.
"""

from __future__ import annotations

from typing import Final

from tests.helpers import REPO_ROOT

RECIPE_PAGE: Final = (
    REPO_ROOT / "docs" / "guides" / "analysis" / "train-a-classifier.md"
)
MAX_CHARACTERS: Final = 5000


def test_the_classifier_recipe_page_is_under_its_budget() -> None:
    size = len(RECIPE_PAGE.read_text(encoding="utf-8"))
    assert size < MAX_CHARACTERS, (
        f"{RECIPE_PAGE.name} has {size} characters, over {MAX_CHARACTERS}: move "
        "the detail to a reference page it links to"
    )
