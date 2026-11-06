"""Shared, dependency-light helpers for tracking ops.

Holds the models-root guard and the dataset fingerprint that the training ops and
the ``convert-points`` op share.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from pydantic import ValidationError

from mosaic.core.annotations.yolo_layout import (
    YOLO_IMAGE_SUFFIXES,
    label_path_for_image,
)
from mosaic.core.json_object import JSON_OBJECT
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.models import (
    PREPARED_DATA_KINDS,
    model_index_path,
    prepared_artifact_cell,
)
from mosaic.core.pipeline.op_identity import parse_op_run_id

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def ensure_models_root(ds: Dataset) -> None:
    """Ensure the dataset has a ``models`` root (default ``models/``)."""
    if not ds.has_root("models"):
        ds.set_root("models", "models")


def fingerprint_dataset(path: Path) -> str:
    """Content digest of the data a training or conversion op reads.

    Two forms, one per shape of input:

    - **A file** is digested by its own bytes (:func:`file_digest`), so a copy
      fingerprints identically wherever it sits. Nothing beside it enters. This
      form used to list every file under the file's *parent*, recursively, which
      made a ``.slp`` or a CVAT XML at a dataset root depend on the whole dataset
      -- including each run's own output, so an identical resubmission minted a
      new identifier and reuse could never hit.
    - **A directory** is digested by a listing of relative paths and sizes (not
      mtimes), so a copied or moved directory fingerprints identically. Sizes
      rather than bytes because a directory of images is too large to read on
      every planning call. ``train-localizer`` and ``train-litpose`` pass one.

    What a file names by path is not followed: a ``.slp`` carries the names of
    the videos its labels sit on, not their pixels. YOLO / POLO ``data.yaml``
    files have :func:`fingerprint_yolo_dataset`, which reads what the YAML
    declares, digesting each split's labels by content and its images by size.

    A missing or unreadable path still yields a digest, the same one for both:
    identity computation is not the place to refuse a dataset, and the tool that
    reads it says which file is wrong and why.
    """
    path = Path(path)
    if path.is_file():
        try:
            return file_digest(path)
        except OSError:
            return hash_params({"listing": []})
    listing: list[str] = []
    if path.exists():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                try:
                    size = f.stat().st_size
                except OSError:
                    size = -1
                listing.append(f"{f.relative_to(path).as_posix()}:{size}")
    return hash_params({"listing": listing})


# The keys a YOLO / POLO ``data.yaml`` uses to name its splits. ``path`` is
# deliberately absent: it is a location, not content.
_SPLIT_KEYS: Final = ("train", "val", "test")


def _split_roots(declared: JsonValue, base: Path) -> list[Path]:
    """Resolve one split declaration to the roots it names.

    A split is a string or a list of strings, each joined to *base* and resolved,
    as ``check_det_dataset`` of Ultralytics 8.4.121 does. A string naming nothing
    that starts with ``../`` is tried again without that prefix.
    """
    if isinstance(declared, str):
        root = (base / declared).resolve()
        if not root.exists() and declared.startswith("../"):
            root = (base / declared[3:]).resolve()
        return [root]
    if isinstance(declared, list):
        return [(base / item).resolve() for item in declared if isinstance(item, str)]
    return []


def _split_images(root: Path, listed: str | None) -> list[tuple[str, Path]]:
    """The images one split root names, each with its location-free key.

    Mirrors ``get_img_files`` of Ultralytics 8.4.121. A directory is globbed
    recursively and its files are kept unresolved, keyed by their path relative
    to *root*, which is matched literally even when its name holds glob
    characters. A ``.txt`` list names one image per line: a line starting with
    ``./`` is placed under the list's directory, any other is used as written,
    and the line is the key. Only lowercased suffixes in ``YOLO_IMAGE_SUFFIXES``
    count, so ``labels.cache`` and disk-cache ``.npy`` files never enter.

    Args:
        root: A directory to glob, or the ``.txt`` list whose text is *listed*.
        listed: The text of the ``.txt`` list, or ``None`` for a directory.

    Returns:
        ``(key, path)`` of each image, unsorted.
    """
    named: list[tuple[str, Path]] = []
    if listed is None:
        pattern = str(Path(glob.escape(str(root))) / "**" / "*.*")
        for found in glob.glob(pattern, recursive=True):
            path = Path(found)
            named.append((path.relative_to(root).as_posix(), path))
    else:
        parent = f"{root.parent}{os.sep}"
        for line in listed.strip().splitlines():
            spelled = line.replace("./", parent, 1) if line.startswith("./") else line
            named.append((line, Path(spelled)))
    return [
        (key, path) for key, path in named if path.suffix.lower() in YOLO_IMAGE_SUFFIXES
    ]


def _walked_roots(
    roots: list[Path],
) -> list[tuple[str | None, list[tuple[str, Path]]]]:
    """Walk each root that exists, as Ultralytics would.

    Returns:
        One ``(listed, images)`` per root that is a directory or a ``.txt``
        list: the text of the list, ``None`` for a directory, and the images of
        :func:`_split_images`. A root that is neither is skipped.
    """
    walked: list[tuple[str | None, list[tuple[str, Path]]]] = []
    for root in roots:
        listed: str | None = None
        if root.is_file():
            try:
                listed = root.read_text(errors="ignore")
            except OSError:
                listed = ""
        elif not root.is_dir():
            continue
        walked.append((listed, _split_images(root, listed)))
    return walked


def _split_listing(roots: list[Path]) -> dict[str, list[str]]:
    """The images, labels and image lists of one split, each sorted.

    An image is recorded by its key and size, its label by its key and the
    label's content digest, ``-`` when the label is absent or unreadable. A
    ``.txt`` root also records its own text.
    """
    images: list[str] = []
    labels: list[str] = []
    lists: list[str] = []
    for listed, found in _walked_roots(roots):
        if listed is not None:
            lists.append(listed)
        for key, image in found:
            try:
                size = image.stat().st_size
            except OSError:
                size = -1
            try:
                label = file_digest(label_path_for_image(image))
            except OSError:
                label = "-"
            images.append(f"{key}:{size}")
            labels.append(f"{key}:{label}")
    return {"images": sorted(images), "labels": sorted(labels), "lists": sorted(lists)}


def _declared_split_roots(
    data_yaml: Path, parsed: dict[str, JsonValue]
) -> dict[str, list[Path]]:
    """The roots each split of a parsed *data_yaml* names, keyed by split.

    The base is the directory of *data_yaml* unless a ``path`` key names another,
    which is joined to it when relative.
    """
    root = parsed.get("path")
    base = data_yaml.parent
    if isinstance(root, str):
        base = Path(root) if Path(root).is_absolute() else data_yaml.parent / root
    return {
        key: _split_roots(declaration, base)
        for key in _SPLIT_KEYS
        if (declaration := parsed.get(key)) is not None
    }


def _text_keyed(value: object) -> object:
    """*value* with every non-bool ``int`` mapping key spelled as its decimal text.

    A ``data.yaml`` keys ``names`` and ``radii`` by class number, and Ultralytics
    reads ``0`` and ``'0'`` as one class. Other key types are left as they are.
    """
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {
            (str(key) if isinstance(key, int) and not isinstance(key, bool) else key): (
                _text_keyed(item)
            )
            for key, item in items
        }
    if isinstance(value, list):
        return [_text_keyed(item) for item in cast("list[object]", value)]
    return value


def _parsed_declaration(data_yaml: Path) -> tuple[str, dict[str, JsonValue] | None]:
    """The text of *data_yaml*, and its parsed mapping when it is one.

    The text is empty when the file cannot be read. The mapping is ``None`` when
    the text does not parse, is not a mapping, or holds something no JSON value
    covers.
    """
    import yaml

    try:
        text = data_yaml.read_text(errors="ignore")
    except OSError:
        text = ""
    try:
        loaded: object = yaml.safe_load(text)
        return text, JSON_OBJECT.validate_python(_text_keyed(loaded))
    except (yaml.YAMLError, ValidationError):
        return text, None


def resolve_training_data(ds: Dataset, reference: str) -> Path:
    """What a training op's data argument names: a path, or a preparation run.

    A training op has always taken a path. It now also takes the run identifier
    of a preparation -- ``prepare-training-data`` or ``convert-points`` -- and
    reads what that run wrote, the way a model reference may be a path or the run
    that trained it.

    The difference matters to identity. The reference string is part of a
    training run's parameters, so it is hashed. A path is a location: the same
    data at two places mints two models, and moving a dataset re-mints every one.
    A run identifier is content, so it names the same data from anywhere, and it
    moves exactly when the annotations behind it do.

    A path wins when it exists, as it does for a model reference. A reference
    that names neither is returned as the path it spells, so the caller's own
    "not found" is what is reported rather than a second one from here.
    """
    candidate = Path(ds.resolve_path(reference))
    if candidate.exists():
        return candidate
    row = prepared_data_row(ds, reference)
    stored = prepared_artifact_cell(row) if row is not None else ""
    return Path(ds.resolve_path(stored)) if stored else candidate


def prepared_data_row(ds: Dataset, reference: str) -> dict[str, str] | None:
    """The index row of the preparation run that *reference* names, as strings.

    ``None`` when *reference* is not the run id of a preparation kind, or its
    index holds no row of it. The last row of the run wins.
    """
    parsed = parse_op_run_id(reference)
    if parsed is None or parsed.kind not in PREPARED_DATA_KINDS:
        return None
    index_path = model_index_path(ds, parsed.kind)
    if not index_path.exists():
        return None
    import pandas as pd

    frame = pd.read_csv(index_path, dtype="string", keep_default_na=False)
    if "run_id" not in frame.columns:
        return None
    match = frame[frame["run_id"] == reference]
    if match.empty:
        return None
    return {str(name): str(value) for name, value in match.iloc[-1].items()}


def fingerprint_yolo_dataset(data_yaml: Path) -> str:
    """Digest a YOLO / POLO training dataset by what its ``data.yaml`` declares.

    Digests the declared content (every key but ``path``, which is a location)
    and, for each declared split, the images and labels Ultralytics 8.4.121
    would read: each image by its relative path and size, each image's label
    file by its content, ``-`` when it is absent. Integer mapping keys such as
    the class numbers of ``names`` and ``radii`` are read as their decimal text.
    A ``.txt`` split also contributes its own text. Nothing else under the
    dataset enters, so a label cache, a disk cache or a run's own output beside
    the data leaves the digest where it was, and a copy or move of the dataset
    fingerprints identically.

    A YAML that declares no splits, names roots that do not exist yet, or does
    not parse still yields a digest: identity computation is not the place to
    refuse a dataset. A YAML that does not parse is digested by its text.

    Args:
        data_yaml: The ``data.yaml`` naming the dataset.

    Returns:
        A 10-character digest, stable across copies and moves.
    """
    data_yaml = Path(data_yaml)
    text, parsed = _parsed_declaration(data_yaml)
    if parsed is None:
        # Two different odd YAMLs still fingerprint differently rather than
        # collapsing onto one digest.
        return hash_params({"file": data_yaml.name, "text": text, "declared": None})

    declared = {key: value for key, value in parsed.items() if key != "path"}
    splits = {
        key: _split_listing(roots)
        for key, roots in _declared_split_roots(data_yaml, parsed).items()
    }
    return hash_params({"declared": declared, "splits": splits})


def remove_label_caches(data_yaml: Path) -> None:
    """Delete the label caches Ultralytics would read for the dataset of *data_yaml*.

    Ultralytics keeps one ``<labels directory>.cache`` per split dataset, named
    after the directory of its first label file. A cache that an earlier run
    left half-written (0 bytes after a crash) makes the next run fail while it
    loads the labels, and a stale one hides a corrected label. The caches are
    found through the same walk the fingerprint takes, so exactly the label
    directories of the declared splits are covered, and nothing else under the
    dataset is touched. A ``data.yaml`` that does not parse names no split and
    removes nothing.

    Ultralytics writes the cache non-atomically, so the race is reduced and not
    gone: two trainings that start together on one dataset can still meet a
    partial cache the other is writing.

    Args:
        data_yaml: The ``data.yaml`` naming the dataset.

    Raises:
        RuntimeError: A cache exists and cannot be removed. The message names
            the file.
    """
    _, parsed = _parsed_declaration(Path(data_yaml))
    if parsed is None:
        return
    caches = {
        label_path_for_image(image).parent.with_suffix(".cache")
        for roots in _declared_split_roots(Path(data_yaml), parsed).values()
        for _, found in _walked_roots(roots)
        for _, image in found
    }
    for cache in sorted(caches):
        try:
            cache.unlink(missing_ok=True)
        except OSError as error:
            raise RuntimeError(
                f"Cannot remove the label cache {cache} before training: {error}. "
                "Remove it by hand."
            ) from error
