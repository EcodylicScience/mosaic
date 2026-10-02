"""``get_root`` says which of three faults it hit, and names the repair.

The three used to share one sentence, and it described the least common of them:
"Root 'tracks' is not set in manifest. Available roots: [... 'tracks' ...]" --
which called the key missing and then listed it as available, and sent the reader
to a manifest file that was correct. The state it never mentioned is the common
one: a ``Dataset`` is constructed around a manifest *path*, so until it is loaded
its roots are the empty template.

``open_dataset`` is the other half of the answer. The constructor has to keep
reading nothing -- pointing at a dataset that does not exist yet is how one gets
created -- so the fix for the common intent is a front door beside it, not a
change of behavior behind it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset

# --- One message per state ---------------------------------------------------


def test_a_dataset_that_was_never_loaded_names_how_to_load_it(tmp_path: Path) -> None:
    """The state that used to be invisible, and the one that cost a session."""
    dataset = Dataset(manifest_path=tmp_path / "dataset.yaml")

    with pytest.raises(KeyError, match="declares no roots at all") as caught:
        _ = dataset.get_root("tracks")

    message = str(caught.value)
    assert "open_dataset()" in message
    assert "load()" in message
    assert "roots=" in message
    # Which file to read, which is the fact the lost debugging session lacked.
    assert str(dataset.manifest_path) in message


def test_a_root_declared_empty_lists_only_the_roots_that_hold_a_value(
    tmp_path: Path,
) -> None:
    """Empty means unset, and an unset root is never offered as one that works."""
    dataset = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={"media_raw": "media_raw", "tracks": "", "features": ""},
    )

    with pytest.raises(KeyError, match="declared with an empty value") as caught:
        _ = dataset.get_root("tracks")

    message = str(caught.value)
    assert "Roots that hold a value: media_raw." in message
    # Named once, as the subject. The key that failed never joins a list the
    # reader will scan as "roots that work" -- which is the bug this file pins.
    assert message.count("tracks") == 1


def test_an_undeclared_root_lists_the_declared_ones(tmp_path: Path) -> None:
    """A near-miss spelling shows in the list, and the key itself is not in it."""
    dataset = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={"media_raw": "media_raw", "tracks": "tracks"},
    )

    with pytest.raises(KeyError, match="is not declared by this manifest") as caught:
        _ = dataset.get_root("trakcs")

    message = str(caught.value)
    assert "Declared roots: media_raw, tracks." in message
    assert message.count("trakcs") == 1


# --- The front door ----------------------------------------------------------


def test_open_dataset_reads_the_manifest_the_constructor_only_points_at(
    tmp_path: Path,
) -> None:
    """The two halves of the papercut, side by side."""
    base = tmp_path / "ds"
    manifest = new_dataset_manifest(name="lookup", base_dir=base)

    assert open_dataset(manifest).get_root("tracks") == (base / "tracks").resolve()

    with pytest.raises(KeyError, match="declares no roots at all"):
        _ = Dataset(manifest_path=manifest).get_root("tracks")


def test_open_dataset_accepts_the_dataset_directory(tmp_path: Path) -> None:
    """``resolve_manifest_path`` probes a directory, so a caller may hand one over."""
    base = tmp_path / "ds"
    _ = new_dataset_manifest(name="lookup", base_dir=base)

    assert open_dataset(base).get_root("tracks") == (base / "tracks").resolve()


def test_open_dataset_without_ensure_roots_writes_nothing(tmp_path: Path) -> None:
    """Opening a dataset on a read-only mount must not try to create its roots."""
    base = tmp_path / "ds"
    manifest = new_dataset_manifest(name="lookup", base_dir=base)
    shutil.rmtree(base / "tracks")

    _ = open_dataset(manifest, ensure_roots=False)
    assert not (base / "tracks").exists()

    _ = open_dataset(manifest)
    assert (base / "tracks").is_dir()


def test_open_dataset_raises_for_a_directory_with_no_manifest(tmp_path: Path) -> None:
    """Absent is reported as absent, not as a dataset whose roots are all unset."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No manifest found in directory"):
        _ = open_dataset(empty)
