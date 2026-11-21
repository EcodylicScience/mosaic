"""A source dataset holding raw rows, and a copy that adopts them through a link."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.manifest import DatasetLink
from mosaic.core.pipeline.artifact_kinds import LINKED_KINDS, LinkedKind
from mosaic.core.pipeline.index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)
from mosaic.core.pipeline.tracks_raw_index import TracksRawIndexScope
from tests.helpers.datasets import make_dataset
from tests.helpers.labels import add_labels
from tests.helpers.media import MediaClip, add_transcode_derivative, write_media_index
from tests.helpers.tag_tables import tag_table, write_tags
from tests.helpers.tracks import write_trex_npz

__all__ = [
    "CONVERTED_ENTRIES",
    "SOURCE_ENTRIES",
    "SUPERSEDED_RECIPE",
    "add_feature_run",
    "converted_source",
    "copy_of",
    "feature_edge",
    "labelled_source",
    "link_forms",
    "raw_source",
    "take_link_offline",
    "tracks_edge",
    "write_run_params",
]

SOURCE_ENTRIES: tuple[tuple[str, str], ...] = (("g", "s1"), ("g", "s2"))
"""The entries :func:`raw_source` holds, in index order."""

CONVERTED_ENTRIES: tuple[tuple[str, str], ...] = (("g", "a"), ("g", "b"))
"""The entries :func:`converted_source` holds, in index order."""

SUPERSEDED_RECIPE = "superseded0"
"""The ``recipe_hash`` of the derivative row of :func:`raw_source` no cell names."""

link_forms = pytest.mark.parametrize(
    "relative", [True, False], ids=["relative-link", "absolute-link"]
)
"""Runs a test once per path form of a link: relative and absolute."""


def raw_source(base: Path) -> Dataset:
    """A saved dataset at *base* holding raw rows of every kind for two sequences.

    Each entry of :data:`SOURCE_ENTRIES` has one video with an analysis and a
    playback derivative, one raw tracks file, one raw labels file and a display
    name. ``media/index.csv`` also holds a derivative row of the first video that
    no cell names, under :data:`SUPERSEDED_RECIPE`, and the ``tags`` series holds
    one revision. Every stored path is relative to *base*.
    """
    source = make_dataset(base, name="source")
    write_media_index(
        source,
        [
            MediaClip(
                group=group,
                sequence=sequence,
                filename=f"{sequence}.mp4",
                video_uuid=f"uid-{sequence}",
            )
            for group, sequence in SOURCE_ENTRIES
        ],
    )
    _ = source.rebuild_sequence_index("media_raw")
    for _group, sequence in SOURCE_ENTRIES:
        for target in ("analysis", "playback"):
            _ = add_transcode_derivative(source, sequence, target=target)
    _add_superseded_derivative(source, SOURCE_ENTRIES[0])

    scopes: list[TracksRawIndexScope] = []
    labels = source.get_root("labels_raw") / "g"
    labels.mkdir(parents=True)
    for group, sequence in SOURCE_ENTRIES:
        tracks = source.get_root("tracks_raw") / group / sequence
        tracks.mkdir(parents=True)
        _ = (tracks / f"{sequence}.npy").write_bytes(sequence.encode())
        scopes.append(
            TracksRawIndexScope(
                directory=tracks, group=group, sequence=sequence, src_format="trex_npz"
            )
        )
        _ = (labels / f"{sequence}.csv").write_text(f"{sequence}\n", encoding="utf-8")
        source.set_display_name(group, sequence, f"Sequence {sequence}")
    _ = source.write_tracks_raw_index(scopes, patterns="*.npy")
    _ = source.index_labels_raw(
        [labels], patterns="*.csv", group_from_path=lambda _path: "g"
    )
    _ = write_tags(
        source,
        tag_table({SOURCE_ENTRIES[0]: {"site": "a"}}, {"site": "text"}),
    )
    return source


def labelled_source(base: Path) -> Dataset:
    """:func:`raw_source` with the ``behavior`` labels of each entry converted.

    Each entry of :data:`SOURCE_ENTRIES` has one event in the variant
    ``LABELS_VARIANT``, recorded as converted from its raw labels file under
    ``labels_raw``.
    """
    source = raw_source(base)
    for group, sequence in SOURCE_ENTRIES:
        _ = add_labels(
            source,
            group,
            sequence,
            [(0, 0, 1, 0)],
            source=source.get_root("labels_raw") / group / f"{sequence}.csv",
            consumed_source_roots=("labels_raw",),
        )
    return source


def converted_source(base: Path) -> Dataset:
    """A saved dataset at *base* whose raw TREx files are converted to tracks.

    Each entry of :data:`CONVERTED_ENTRIES` has one TREx export under ``raw/``,
    indexed in group ``g`` and converted into one tracks variant.
    """
    source = make_dataset(base, name="source")
    raw = source.base_dir / "raw"
    for _group, sequence in CONVERTED_ENTRIES:
        write_trex_npz(raw / f"{sequence}_fish0.npz")
    _ = source.index_tracks_raw(
        [raw],
        patterns=["*.npz"],
        src_format="trex_npz",
        group_from_path=lambda _path: "g",
    )
    _ = source.convert_all_tracks()
    return source


def add_feature_run(
    dataset: Dataset,
    storage: str,
    run_id: str,
    entries: Collection[tuple[str, str]] = SOURCE_ENTRIES,
) -> Path:
    """Record a finished scope-free feature run over *entries* in *dataset*.

    Each entry gets an output file and an index row whose path is relative to
    the dataset, and the run root holds a ``params.json``. The outputs are not
    parquet tables.

    Args:
        dataset: The dataset that holds the run.
        storage: The feature's storage directory name.
        run_id: The run's identifier.
        entries: The ``(group, sequence)`` keys the run covers.

    Returns:
        The run root.
    """
    run_root = feature_run_root(dataset, storage, run_id)
    run_root.mkdir(parents=True)
    _ = (run_root / "params.json").write_text("{}", encoding="utf-8")
    rows: list[FeatureIndexRow] = []
    for group, sequence in entries:
        output = run_root / f"{sequence}.parquet"
        _ = output.write_bytes(sequence.encode())
        rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=storage,
                version="0.1",
                group=group,
                sequence=sequence,
                abs_path=Path(dataset.relative_to_root(output)),
                params_hash="hash",
                n_rows=1,
            )
        )
    index = feature_index(feature_index_path(dataset, storage))
    index.append(rows)
    index.mark_finished(run_id)
    return run_root


def write_run_params(run_root: Path, document: dict[str, object]) -> None:
    """Replace the ``params.json`` of the run at *run_root* with *document*."""
    _ = (run_root / "params.json").write_text(json.dumps(document), encoding="utf-8")


def tracks_edge(variant: str) -> dict[str, object]:
    """The ``_resolved`` entry of a run that read the tracks *variant*."""
    return {"where": "inputs[tracks]", "feature": "tracks", "run_id": variant}


def feature_edge(position: int, storage: str, run_id: str) -> dict[str, object]:
    """The ``_resolved`` entry of a run whose input *position* is a feature run."""
    return {"where": f"inputs[{position}]", "feature": storage, "run_id": run_id}


def _add_superseded_derivative(source: Dataset, entry: tuple[str, str]) -> None:
    """Write a derivative file of *entry* and its row, named by no cell."""
    from mosaic.core.pipeline.transcode import TRANSCODE_KIND_DIRECTORY

    group, sequence = entry
    media = source.get_root("media")
    derivative = (
        media
        / TRANSCODE_KIND_DIRECTORY
        / f"uid-{sequence}.{SUPERSEDED_RECIPE}.playback.mp4"
    )
    _ = derivative.write_bytes(b"superseded")
    index = media / "index.csv"
    rows: list[dict[str, object]] = [dict(row) for row in read_media_index(index)]
    row: dict[str, object] = {key: "" for key in rows[0]}
    row.update(
        {
            "name": derivative.name,
            "group": group,
            "sequence": sequence,
            "abs_path": source.relative_to_root(derivative),
            "source_video_uuid": f"uid-{sequence}",
            "recipe_hash": SUPERSEDED_RECIPE,
        }
    )
    rows.append(row)
    write_media_index_rows(index, frame_from_rows(rows))


def copy_of(
    source: Dataset,
    directory: Path,
    *,
    entries: Collection[tuple[str, str]] | None = None,
    relative: bool = True,
    provides: tuple[LinkedKind, ...] = LINKED_KINDS,
) -> Dataset:
    """A new dataset at *directory* that links *source* and adopts its raw rows.

    The copy gets a fresh ``uuid`` and a link named ``source-<uuid>`` that
    provides *provides*. Its path is relative to *directory* when *relative* is
    true and *source*'s absolute base directory otherwise, the link form that
    decides how the adopted cells are stored.

    Args:
        source: The dataset whose raw rows are adopted.
        directory: Where the copy is made.
        entries: The ``(group, sequence)`` keys to adopt, or ``None`` for every
            row.
        relative: Whether the link names *source* by a relative path.
        provides: The kinds the link provides.

    Returns:
        The saved copy.
    """
    copy = make_dataset(directory, name=f"copy of {source.name}")
    path = (
        source.base_dir.relative_to(copy.base_dir, walk_up=True).as_posix()
        if relative
        else str(source.base_dir)
    )
    link = DatasetLink(id=f"source-{source.uuid}", path=path, provides=provides)
    _ = copy.add_link(link, save=False)
    _ = copy.adopt_raw_rows(source, entries=entries)
    copy.save()
    return copy


def take_link_offline(copy: Dataset) -> Dataset:
    """Make the one link of *copy* unreachable, and return *copy* opened again.

    The linked dataset's manifest is renamed away, so opening the link raises
    ``FileNotFoundError``. The returned dataset is loaded afresh, so it holds no
    link that an earlier lookup opened.
    """
    (link,) = copy.manifest.links
    manifest = copy.resolve_link_path(link) / "dataset.yaml"
    _ = manifest.rename(manifest.with_name("dataset.yaml.away"))
    return Dataset(manifest_path=copy.manifest_path).load()
