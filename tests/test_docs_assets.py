"""The two hero diagrams are one drawing in two palettes.

The README selects between them with `<picture>` and a `prefers-color-scheme`
query, and the documentation site selects between them with Material's
`#only-light` / `#only-dark` suffixes. Two files is the only technique that
works in both places: inside an `<img>`, a media query in the SVG resolves
against the reader's *operating system*, while Material's dark mode is a site
toggle -- so a self-switching single file shows the light drawing on a dark page
for anyone whose system is light.

The cost of two files is that they drift. These tests pin the thing that must
not: same words, same geometry, different colors.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

SVG_NS = "{http://www.w3.org/2000/svg}"
ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
LIGHT = ASSETS / "pipeline-light.svg"
DARK = ASSETS / "pipeline-dark.svg"


@pytest.fixture(scope="module")
def trees() -> tuple[ElementTree.ElementTree, ElementTree.ElementTree]:
    return ElementTree.parse(LIGHT), ElementTree.parse(DARK)


def test_both_variants_exist() -> None:
    assert LIGHT.is_file(), f"{LIGHT} is missing"
    assert DARK.is_file(), f"{DARK} is missing"


def test_same_words(trees: tuple[ElementTree.ElementTree, ...]) -> None:
    """A label edited in one file and not the other is the likeliest drift."""
    light, dark = trees
    assert [e.text for e in light.iter(f"{SVG_NS}text")] == [
        e.text for e in dark.iter(f"{SVG_NS}text")
    ]


def test_same_geometry(trees: tuple[ElementTree.ElementTree, ...]) -> None:
    """Boxes and arrows in the same places, so the two read as one drawing."""
    light, dark = trees
    for tag, attribute in ((f"{SVG_NS}path", "d"), (f"{SVG_NS}rect", "x")):
        assert [e.get(attribute) for e in light.iter(tag)] == [
            e.get(attribute) for e in dark.iter(tag)
        ], f"{tag} {attribute} differs between the two variants"


def test_no_opaque_canvas() -> None:
    """Neither variant paints its own background.

    The page behind the image supplies it, which is what keeps a light drawing
    readable when it lands on a dark page because the reader's system theme and
    the site toggle disagree. A full-bleed background rect would instead show a
    white slab in the middle of a dark page.
    """
    for path in (LIGHT, DARK):
        text = path.read_text(encoding="utf-8")
        assert 'width="100%"' not in text, f"{path.name} paints a full-width fill"


def test_no_pure_black_or_white() -> None:
    """Mid-tones only, so a theme mismatch degrades to low contrast.

    Pure black text on a dark page, or pure white on a light one, is the failure
    mode that makes a mismatched variant unreadable rather than merely dull.
    """
    for path in (LIGHT, DARK):
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("#000", "#fff", "black", "white"):
            assert banned not in text, f"{path.name} uses {banned}"


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


COUNT_REGISTRIES = """
import json
from typer.main import get_command
from mosaic.behavior.feature_library import FEATURES
from mosaic.cli import app
from mosaic.core.pipeline.ops import OPS
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    ensure_track_converters_registered,
)
from mosaic.tracking import register_ops

register_ops()
ensure_track_converters_registered()
print(json.dumps({
    "features": len(FEATURES),
    "ops": len(OPS),
    "track_formats": len(TRACK_CONVERTERS),
    "cli_commands": len(get_command(app).commands),
}))
"""


def registry_sizes() -> dict[str, int]:
    """What a clean import holds -- counted in a subprocess, deliberately.

    The registries are process-global and mutable, and the suite mutates them.
    `test_conversion_strictness.py:52` decorates a fixture converter with
    `@register_track_converter`, so `test_missing_keypoints` is in
    `TRACK_CONVERTERS` for the rest of the process: an in-process count reports
    9 when this test runs after that file and 8 when it runs alone. Counting
    here would make the test pass or fail on collection order, which is exactly
    the kind of answer it exists to disallow.

    A fresh interpreter reports what the README is claiming: the registry a user
    gets, before any test has added to it.
    """
    completed = subprocess.run(
        [sys.executable, "-c", COUNT_REGISTRIES],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    counts: dict[str, int] = json.loads(completed.stdout)
    return counts


def test_readme_counts_match_the_registries() -> None:
    """The README's headline numbers are the ones the code can produce.

    `docs/reference/` is generated and gated, so it cannot drift. The README is
    hand-written and is the first thing anyone reads, which makes a stale number
    there more damaging than one buried in a reference page -- and it is exactly
    what went wrong before: three hand-maintained feature lists claimed "~30",
    "40+" and a subset, against a registry of 44.
    """
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    sizes = registry_sizes()
    for count, noun in (
        (sizes["features"], "registered features"),
        (sizes["ops"], "ops"),
        (sizes["track_formats"], "formats"),
        (sizes["cli_commands"], "CLI commands"),
    ):
        assert f"{count}** {noun}" in readme or f"{count} {noun}" in readme, (
            f"README no longer says {count} {noun}; rerun the counts and update it"
        )
