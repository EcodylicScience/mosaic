"""Building the `Dataset` a test runs against.

This is the helper 32 test modules each wrote for themselves, under the names
``_dataset``, ``_make_dataset``, ``_new_dataset``, ``_build_dataset`` and ``ds``.
They were not written because importing a shared one was hard -- 19 modules
already imported from the shared layer -- but because the shared layer had a
*media*-shaped dataset and a fixed two-sequence scenario, and nothing for "a
dataset over these roots", which is the shape they all needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mosaic.core.dataset import Dataset, new_dataset_manifest


def make_dataset(
    base: Path,
    *,
    name: str = "t",
    roots: Sequence[str] | None = None,
    save: bool = True,
    ensure_roots: bool = True,
) -> Dataset:
    """A saved `Dataset` rooted at *base*.

    Args:
        base: The directory the dataset lives in. Created if absent.
        name: The dataset's name in its manifest. Only matters where a test
            asserts on it or needs two datasets to differ.
        roots: The root keys to declare explicitly, or None for the default set
            from `new_dataset_manifest`. A *restricted* root set is load-bearing
            in several suites -- `resolve_media_root` answers differently for a
            dataset with no `media_raw` -- so it is named rather than inferred.
        save: Whether to write the manifest. Defaults True because an unsaved
            manifest makes `base_dir` resolve to the manifest *path* rather than
            its directory, one level too deep, which silently corrupts every
            root-relative `abs_path` the dataset then writes. The four call sites
            that genuinely test the unsaved state pass `save=False` and say why.
        ensure_roots: Whether to create the root directories. A test asserting
            that a root is *absent* passes False.

    Returns:
        The dataset, loaded.
    """
    if roots is None:
        manifest = new_dataset_manifest(name=name, base_dir=base)
        dataset = Dataset(manifest_path=manifest)
    else:
        dataset = Dataset(
            manifest_path=base / "dataset.yaml",
            roots={key: str(base / key) for key in roots},
        )
    _ = dataset.load(ensure_roots=ensure_roots)
    if save:
        dataset.save()
    return dataset
