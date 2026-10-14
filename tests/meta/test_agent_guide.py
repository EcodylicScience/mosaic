"""The agent guide's links resolve, and the guide ships in the wheel.

``src/mosaic/agent/SKILL.md`` names documentation pages by their site URL, because
it is read in a user's project where no relative link resolves. Each URL is
checked against the page under ``docs/`` that the site builds it from.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Final

import pytest

from mosaic.agent import SITE_URL
from tests.helpers import REPO_ROOT, SOURCE_ROOT

GUIDE: Final = SOURCE_ROOT / "agent" / "SKILL.md"
SITE_LINK: Final = re.compile(re.escape(SITE_URL) + r"[^\s)>\]`\"']*")

BUILD_INPUTS: Final = ("src", "pyproject.toml", "README.md", "LICENSE", "NOTICE")
"""What a wheel is built from: the package source and the files pyproject names."""


def page_for(url: str) -> list[Path]:
    """The ``docs/`` files the site may build *url* from."""
    path = url.removeprefix(SITE_URL).partition("#")[0].rstrip(".,;:").strip("/")
    docs = REPO_ROOT / "docs"
    if not path:
        return [docs / "index.md"]
    return [docs / f"{path}.md", docs / path / "index.md"]


def test_every_site_url_in_the_guide_names_a_docs_page() -> None:
    urls = sorted(set(SITE_LINK.findall(GUIDE.read_text(encoding="utf-8"))))
    assert urls, f"{GUIDE} links no page of {SITE_URL}"
    missing = [url for url in urls if not any(page.is_file() for page in page_for(url))]
    assert not missing, f"{GUIDE} links pages that docs/ does not hold: {missing}"


def build_inputs() -> list[str]:
    """The tracked and unignored files a wheel is built from, relative to the root."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        + ["--", *BUILD_INPUTS],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        pytest.skip(f"not a git checkout: {completed.stderr.strip()}")
    return [name for name in completed.stdout.split("\0") if name]


def wheel_command(source: Path, dist: Path) -> list[str]:
    """A command building *source* into a wheel in *dist*, with an isolated backend.

    uv where it is on PATH, as in every CI job, and pip otherwise.
    """
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "build", "--wheel", "--out-dir", str(dist), str(source)]
    probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("neither uv nor pip is available to build a wheel")
    return [sys.executable, "-m", "pip", "wheel", "--no-deps"] + [
        "--wheel-dir",
        str(dist),
        str(source),
    ]


@pytest.mark.slow
def test_the_wheel_ships_the_guide_and_no_developer_guide(tmp_path: Path) -> None:
    """Built from a copy, which keeps the build's outputs out of the checkout."""
    source = tmp_path / "source"
    for name in build_inputs():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(REPO_ROOT / name, target)
    dist = tmp_path / "dist"
    completed = subprocess.run(
        wheel_command(source, dist), capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    (wheel,) = dist.glob("mosaic_behavior-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "mosaic/agent/SKILL.md" in names
    developer_guides = [name for name in names if Path(name).name == "CLAUDE.md"]
    assert not developer_guides, developer_guides
