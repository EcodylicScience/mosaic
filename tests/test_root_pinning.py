"""Roots live inside the dataset; ``abs_path`` does not -- item 9.1, rule P7.

The two halves are one paragraph in the rule and two different guarantees, and
the second is the one worth protecting from a later tidy-up. Open item O2
resolved against shared membership *because* a file elsewhere can be referenced
by absolute ``abs_path`` from an index that is inside -- so removing that would
not tighten P7, it would delete the mechanism O2 chose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import (
    Dataset,
    new_dataset_manifest,
    validate_root_inside,
)


def _dataset(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest(name="pinned", base_dir=tmp_path / "ds")
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


# --- Roots are pinned --------------------------------------------------------


def test_a_root_outside_the_dataset_is_refused(tmp_path: Path) -> None:
    """An outside root puts that root's own index.csv outside the dataset too."""
    ds = _dataset(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(ValueError, match="would resolve outside the dataset"):
        ds.set_root("features", str(elsewhere))


def test_an_absolute_root_inside_the_dataset_is_accepted(tmp_path: Path) -> None:
    """The rule is about where a root is, not how it is spelled.

    Rejecting every absolute would break the portable form's own fixture idiom
    for no gain: the portability pass relativizes an inside-absolute root on its
    next run.
    """
    ds = _dataset(tmp_path)
    inside = ds.base_dir / "extra"

    ds.set_root("features", str(inside))

    assert ds.get_root("features").resolve() == inside.resolve()


def test_a_traversal_out_of_the_dataset_is_refused(tmp_path: Path) -> None:
    """Both sides resolve, or ``..`` leaves while reading as though it stayed."""
    ds = _dataset(tmp_path)

    with pytest.raises(ValueError, match="would resolve outside the dataset"):
        ds.set_root("features", "../sneaky")


def test_creating_a_manifest_with_an_outside_root_is_refused(tmp_path: Path) -> None:
    """The other write boundary, and the branch item 9.1 names.

    ``new_dataset_manifest`` used to keep an outside root absolute -- the one
    mechanism by which a root, and therefore its index, could live outside the
    dataset it describes.
    """
    with pytest.raises(ValueError, match="would resolve outside the dataset"):
        _ = new_dataset_manifest(
            name="escaping",
            base_dir=tmp_path / "ds2",
            roots={"features": str(tmp_path / "outside")},
        )


def test_a_legacy_outside_root_still_resolves(tmp_path: Path) -> None:
    """Validated on write, tolerated on read -- item 2.5's boundary rule.

    A dataset that already holds an outside root must keep loading, or looking at
    a legacy dataset raises. What refuses to *act* on one is the sweeper, which
    declines rather than deleting.
    """
    ds = _dataset(tmp_path)
    elsewhere = tmp_path / "legacy-elsewhere"
    elsewhere.mkdir()
    # On a *tracker* root, because that is where being wrong costs files: the
    # sweeper is the pass that would delete under it. An outside `features` root
    # resolves too, but nothing destructive walks it, so it would be the wrong
    # arrangement to make this claim with.
    ds.roots["trex"] = str(elsewhere)

    assert ds.get_root("trex").resolve() == elsewhere.resolve()
    report = ds.sweep_tracking(apply=True)
    assert not report.considered
    assert report.declined == "root-outside-dataset"
    assert report.removed == []


def test_the_check_is_reusable_and_returns_the_original_spelling() -> None:
    """Relative in, relative out: the validator normalizes nothing."""
    base = Path("/data/ds")

    assert validate_root_inside(base, "features", "features") == Path("features")
    assert validate_root_inside(base, "/data/ds/f", "features") == Path("/data/ds/f")


# --- ``abs_path`` is not ------------------------------------------------------


def test_an_index_row_may_point_at_a_file_outside_the_dataset(tmp_path: Path) -> None:
    """The half O2 depends on, and the one a later tidy-up would break.

    A second dataset references a video living inside a first one by absolute
    ``abs_path`` rather than copying 40 GB, and the future import gesture uses
    the same mechanism. ``relative_to_root`` keeps an outside path absolute
    precisely so that works; making it refuse would delete the arrangement O2
    chose, not enforce P7.
    """
    ds = _dataset(tmp_path)
    other_dataset = tmp_path / "dataset-a" / "media_raw"
    other_dataset.mkdir(parents=True)
    shared = other_dataset / "video1.mp4"
    shared.write_bytes(b"frames")

    stored = ds.relative_to_root(shared)

    assert Path(stored).is_absolute(), "an external file must stay addressable"
    assert ds.resolve_path(stored).resolve() == shared.resolve()


def test_a_file_inside_the_dataset_is_still_stored_relative(tmp_path: Path) -> None:
    """The negative half: external addressing is the exception, not the default."""
    ds = _dataset(tmp_path)
    own = ds.get_root("media_raw") / "video1.mp4"
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_bytes(b"frames")

    stored = ds.relative_to_root(own)

    assert not Path(stored).is_absolute()
    assert ds.resolve_path(stored).resolve() == own.resolve()
