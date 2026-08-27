"""Tests that the overlay renderer actually draws, and for which tables.

Every guarantee this surface had was a pinned identifier: the crop features'
``run_id``s sit in the golden corpus, and nothing ever executed a render. That is
how the renderer came to draw a completely blank video for any tracker that
reports positions without keypoints -- T-Rex among them -- without a single test
going red.

Two levels, deliberately. The ``prepare_overlay`` -> ``draw_frame`` tests assert
on pixels drawn onto an in-memory frame, so a failure points at the drawing
decision rather than at a codec. The ``play_video`` tests then run the whole
chain -- imgstore decode, overlay, encode, read back -- because that is what a
user actually invokes, and because the parts either side of the drawing are
exactly where this surface had rotted.

The two scope refusals build a dataset for the same reason: refusing is the whole
point of them, and both raise before a frame is ever read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.visualization_library.helpers import (
    ID_PALETTE,
    LABEL_PALETTE,
    color_for_id,
    color_for_label,
    remap_label_value,
)
from mosaic.behavior.visualization_library.overlay import (
    _remap_overlay_labels,
    draw_frame,
    prepare_overlay,
)
from mosaic.behavior.visualization_library.overlay_feature import Overlay
from mosaic.behavior.visualization_library.playback import play_video
from mosaic.core.dataset import Dataset
from mosaic.core.media.video_io import open_frame_reader
from mosaic.core.pipeline.tracks_index import write_tracks_row

_FRAMES = 4
_SIZE = (64, 48)  # (width, height)

_NO_LABELS: dict[str, dict[str, object]] = {"per_id": {}, "per_pair": {}, "raw": {}}


def _tracks_frame(*, pose: bool) -> pd.DataFrame:
    """A minimal schema-shaped table, with or without keypoint columns.

    Without pose columns is the T-Rex shape: a centroid, and nothing to derive a
    bounding box from.
    """
    frame = np.arange(_FRAMES, dtype=np.int64)
    columns: dict[str, object] = {
        "frame": frame,
        "time": frame / 30.0,
        "id": np.zeros(_FRAMES, dtype=np.int64),
        "group": ["" for _ in frame],
        "sequence": ["seq" for _ in frame],
        "X#wcentroid": np.full(_FRAMES, 32.0),
        "Y#wcentroid": np.full(_FRAMES, 24.0),
    }
    if pose:
        columns["poseX0"] = np.full(_FRAMES, 30.0)
        columns["poseY0"] = np.full(_FRAMES, 22.0)
        columns["poseX1"] = np.full(_FRAMES, 34.0)
        columns["poseY1"] = np.full(_FRAMES, 26.0)
    return pd.DataFrame(columns)


def _drawn_pixels(*, pose: bool) -> int:
    """Draw frame 0 of a synthetic table onto a blank image; count marked pixels."""
    overlay = prepare_overlay(_tracks_frame(pose=pose), _NO_LABELS)
    blank = np.zeros((_SIZE[1], _SIZE[0], 3), dtype=np.uint8)
    drawn = draw_frame(blank, overlay["per_frame"][0], overlay["id_colors"])
    return int(np.count_nonzero(np.any(drawn != blank, axis=-1)))


def _changed_pixels(rendered: list[np.ndarray], source: list[np.ndarray]) -> int:
    """Pixels the overlay altered, summed over the frames the two share."""
    return sum(
        int(np.count_nonzero(np.any(a.astype(np.int16) != b.astype(np.int16), axis=-1)))
        for a, b in zip(rendered, source, strict=False)
    )


def test_a_pose_table_draws_on_the_frame() -> None:
    """The baseline: keypoints put marks on the frame."""
    assert _drawn_pixels(pose=True) > 0


def test_a_centroid_only_table_still_draws() -> None:
    """The regression this file exists for.

    A T-Rex table carries a centroid and no keypoints. The bounding box is
    derived from the pose extent, and the centroid was a label anchor and nothing
    else, so such a table drew nothing at all -- a blank video indistinguishable
    from a tracker that found nothing.
    """
    assert _drawn_pixels(pose=False) > 0, "a centroid-only table drew nothing"


def test_the_overlay_carries_a_centroid_when_there_is_no_pose() -> None:
    """The same defect one level down, where the cause is legible.

    Asserted separately from the pixels because this is the structure the drawing
    code branches on: no ``pose`` key, no ``bbox`` key, and a ``centroid`` that
    was extracted and then never drawn.
    """
    overlay = prepare_overlay(_tracks_frame(pose=False), _NO_LABELS)
    infos = [info for f in overlay["per_frame"].values() for info in f["ids"].values()]
    assert infos
    assert all("pose" not in info and "bbox" not in info for info in infos)
    assert all("centroid" in info for info in infos)


def test_colors_come_from_a_digest_and_not_a_salted_hash() -> None:
    """A render must not change colors because the interpreter was restarted.

    The defect is invisible within any one process: ``hash()`` on a ``str`` is
    salted by ``PYTHONHASHSEED``, so a run agrees with itself and disagrees with
    the next, and two renders of the same tracks disagree about which animal is
    which. Recomputing the slot from ``hashlib`` catches a reversion without
    launching interpreters under different seeds: over this many values a salted
    hash would have to coincide with the digest on every one of them.
    """
    for value in [*range(64), "a", "b", "id-7", ""]:
        digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
        slot = int.from_bytes(digest, "big")
        assert color_for_id(value) == ID_PALETTE[slot % len(ID_PALETTE)]
        assert color_for_label(value) == LABEL_PALETTE[slot % len(LABEL_PALETTE)]


# --- scope refusals, which raise before a frame is read -------------------


def _add_variant(ds: Dataset, variant: str, *, pose: bool) -> None:
    """Write one tracks variant for the ``seq`` entry and register it."""
    out_dir = ds.get_root("tracks") / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seq.parquet"
    table = _tracks_frame(pose=pose)
    table.to_parquet(out_path)
    write_tracks_row(
        ds,
        run_id=variant,
        group="",
        sequence="seq",
        out_path=out_path,
        producer=variant.split(".")[0],
        # What the TREx tracking root declares (`tracking_roots.py`), and what
        # this table actually is: pixels. It said `trex_v1` from before the
        # pixel-native conversion, which claims centimetres -- so the overlay's
        # units guard refused to draw a centroid it had been told was in cm.
        std_format="trex_v2",
        n_rows=len(table),
    )


def _store_dataset(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
    cameras: list[str],
) -> Dataset:
    """A dataset with one imgstore per named camera (``""`` for single-camera)."""
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    search = ds.get_root("media_raw") / "recordings"
    search.mkdir(parents=True, exist_ok=True)
    for serial in cameras:
        extra = (
            {
                "camera_serial": serial,
                "synchronizationuuid": "f064059f9ea046429f227bc7addab1eb",
                "synchronization": "framenumber",
            }
            if serial
            else None
        )
        make_imgstore(
            name=f"seq.{serial}" if serial else "seq",
            nframes=_FRAMES,
            chunksize=4,
            shape=(_SIZE[1], _SIZE[0], 3),
            parent=search,
            extra_metadata=extra,
        )
    ds.index_media([search])
    return ds


def _rendered_and_source(
    ds: Dataset, out_path: Path, variant: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render *variant* to ``out_path`` through ``play_video`` and read both back."""
    written = play_video(
        ds,
        "",
        "seq",
        feature_runs={},
        label_kind=None,
        show_window=False,
        output_path=out_path,
        tracks_run_id=variant,
    )
    assert written is not None and written.is_file()
    with open_frame_reader(written, target="raw") as reader:
        rendered = [frame for _, frame in reader]
    with open_frame_reader(
        ds.resolve_media("", "seq").paths[0], target="raw"
    ) as reader:
        source = [frame for _, frame in reader]
    return rendered, source


def test_a_pose_table_renders_a_marked_video(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """The whole chain: decode an imgstore, draw, encode, read the marks back."""
    pytest.importorskip("imgstore")
    ds = _store_dataset(tmp_path, make_media_dataset, make_imgstore, [""])
    _add_variant(ds, "ultralytics.8.4-aaaaaaaaaa", pose=True)

    rendered, source = _rendered_and_source(
        ds, tmp_path / "pose.mp4", "ultralytics.8.4-aaaaaaaaaa"
    )
    assert len(rendered) == _FRAMES
    assert _changed_pixels(rendered, source) > 0


def test_a_centroid_only_table_renders_a_marked_video(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """The end-to-end form of the regression: T-Rex output must be visible.

    The unit test above pins the drawing decision; this pins that the decision
    survives a real render, which is the thing a user looks at.
    """
    pytest.importorskip("imgstore")
    ds = _store_dataset(tmp_path, make_media_dataset, make_imgstore, [""])
    _add_variant(ds, "trex.0.1-bbbbbbbbbb", pose=False)

    rendered, source = _rendered_and_source(
        ds, tmp_path / "centroid.mp4", "trex.0.1-bbbbbbbbbb"
    )
    assert len(rendered) == _FRAMES
    assert _changed_pixels(rendered, source) > 0, (
        "a centroid-only table rendered an unmarked video"
    )


def test_two_tracks_variants_must_be_named(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """A dataset tracked by two tools is exactly when you want to look at it."""
    pytest.importorskip("imgstore")
    ds = _store_dataset(tmp_path, make_media_dataset, make_imgstore, [""])
    _add_variant(ds, "ultralytics.8.4-aaaaaaaaaa", pose=True)
    _add_variant(ds, "trex.0.1-bbbbbbbbbb", pose=False)

    with pytest.raises(ValueError, match="2 variants"):
        play_video(ds, "", "seq", feature_runs={}, label_kind=None, show_window=False)


def test_a_multicamera_sequence_must_name_a_camera(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """Two synced cameras are one sequence; the renderer must be told which."""
    pytest.importorskip("imgstore")
    from mosaic_media import MediaProbeError

    ds = _store_dataset(tmp_path, make_media_dataset, make_imgstore, ["CAMA", "CAMB"])
    _add_variant(ds, "ultralytics.8.4-aaaaaaaaaa", pose=True)

    with pytest.raises(MediaProbeError, match="camera"):
        play_video(
            ds,
            "",
            "seq",
            feature_runs={},
            label_kind=None,
            show_window=False,
            tracks_run_id="ultralytics.8.4-aaaaaaaaaa",
        )


# --- the same render, as a step a graph can end on -----------------------------


def _feature_dataset(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> Dataset:
    """A store-backed dataset that can also hold a feature run.

    The media factory declares the roots a *render* needs. A render that is a
    feature also writes into `features/`, so that root is added here rather than
    widened onto every media test.
    """
    ds = _store_dataset(tmp_path, make_media_dataset, make_imgstore, [""])
    ds.roots["features"] = str(Path(ds.base_dir) / "features")
    ds.ensure_roots()
    ds.save()
    return ds


def test_the_overlay_feature_writes_a_marked_video_into_its_run_root(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """Rendering has an identity now, so a graph can end on the deliverable.

    Every real workflow used to stop one step short of the artifact a biologist
    looks at: the pipeline produced a table and somebody opened a notebook. The
    video is addressed by ``run_id`` like everything else, which is what makes it
    cacheable, plannable, and something a queue can be asked for.
    """
    pytest.importorskip("imgstore")
    from mosaic.behavior.visualization_library import Overlay
    from mosaic.core.pipeline.index import feature_run_root
    from mosaic.core.pipeline.run import run_feature

    ds = _feature_dataset(tmp_path, make_media_dataset, make_imgstore)
    variant = "trex.0.1-bbbbbbbbbb"
    _add_variant(ds, variant, pose=False)

    result = run_feature(
        ds, Overlay(params={"label_kind": None}), tracks_run_id=variant, track=False
    )

    run_root = feature_run_root(ds, "overlay__from__tracks", result.run_id)
    written = run_root / "seq.mp4"
    assert written.is_file(), sorted(p.name for p in run_root.iterdir())
    with open_frame_reader(written, target="raw") as reader:
        rendered = [frame for _, frame in reader]
    with open_frame_reader(
        ds.resolve_media("", "seq").paths[0], target="raw"
    ) as reader:
        source = [frame for _, frame in reader]
    assert _changed_pixels(rendered, source) > 0


def test_the_overlay_feature_is_a_cache_hit_the_second_time(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """Same tracks, same params, same identifier -- and so no second encode."""
    pytest.importorskip("imgstore")
    from mosaic.behavior.visualization_library import Overlay
    from mosaic.core.pipeline.run import run_feature

    ds = _feature_dataset(tmp_path, make_media_dataset, make_imgstore)
    variant = "trex.0.1-bbbbbbbbbb"
    _add_variant(ds, variant, pose=False)

    first = run_feature(
        ds, Overlay(params={"label_kind": None}), tracks_run_id=variant, track=False
    )
    again = run_feature(
        ds, Overlay(params={"label_kind": None}), tracks_run_id=variant, track=False
    )

    assert again.run_id == first.run_id


def test_a_changed_drawing_is_a_different_artifact(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """Every params field changes the pixels, so every one is in the identity.

    Without that a re-render under different settings would overwrite the first
    at the same address, and nothing on disk would record which one is there.
    """
    pytest.importorskip("imgstore")
    from mosaic.behavior.visualization_library import Overlay
    from mosaic.core.pipeline.run import run_feature

    ds = _feature_dataset(tmp_path, make_media_dataset, make_imgstore)
    variant = "trex.0.1-bbbbbbbbbb"
    _add_variant(ds, variant, pose=False)

    plain = run_feature(
        ds, Overlay(params={"label_kind": None}), tracks_run_id=variant, track=False
    )
    boxless = run_feature(
        ds,
        Overlay(params={"label_kind": None, "show_individual_bboxes": False}),
        tracks_run_id=variant,
        track=False,
    )

    assert boxless.run_id != plain.run_id


# =============================================================================
# Renaming labels
# =============================================================================


def _labelled_overlay(label_value: object) -> dict:
    """An overlay holding one label under the feature name ``behavior``."""
    return {"per_frame": {0: {"ids": {1: {"labels": {"behavior": label_value}}}}}}


def _renamed(overlay: dict) -> object:
    return overlay["per_frame"][0]["ids"][1]["labels"]["behavior"]


def test_a_string_keyed_map_renames_a_numeric_label() -> None:
    """The shape ``Params`` can express has to reach a numeric label.

    ``Overlay.Params.label_maps`` is ``dict[str, dict[str, str]]`` because a
    params model round-trips through JSON, where an object key is a string. The
    label it renames comes from whichever column ``_pick_label_column`` picks,
    and ``label_id``, ``prediction``, ``cluster`` and ``state`` all hold
    integers. Comparing the two directly matched nothing, so a declared map
    renamed nothing and raised nothing.
    """
    params = Overlay.Params(label_maps={"behavior": {"0": "attack", "1": "rest"}})
    overlay = _labelled_overlay(0)
    _remap_overlay_labels(overlay, dict(params.label_maps))
    assert _renamed(overlay) == "attack"


def test_an_integer_keyed_map_still_renames() -> None:
    """``play_video`` takes ``dict[Any, Any]`` and its callers pass integers."""
    overlay = _labelled_overlay(0)
    _remap_overlay_labels(overlay, {"behavior": {0: "attack"}})
    assert _renamed(overlay) == "attack"


def test_a_label_matching_no_key_is_left_alone() -> None:
    overlay = _labelled_overlay(7)
    _remap_overlay_labels(overlay, {"behavior": {"0": "attack"}})
    assert _renamed(overlay) == 7


def test_the_lookup_renames_a_numeric_label_series() -> None:
    """The rule ``build_overlay`` applies to a loaded label series.

    That stage renamed with ``series.map(mapping).fillna(series)``, which
    restored every value the string keys failed to match. A declared map left
    the series exactly as it found it and reported nothing.
    """
    series = pd.Series([0, 1, 7], index=[0, 1, 2])
    mapping = {"0": "attack", "1": "rest"}
    renamed = series.map(lambda v: remap_label_value(v, mapping))
    assert list(renamed) == ["attack", "rest", 7]


def test_an_unhashable_label_is_left_alone() -> None:
    """A pair feature collects several labels for one id, and a list keys nothing."""
    overlay = _labelled_overlay(["attack", "rest"])
    _remap_overlay_labels(overlay, {"behavior": {"0": "x"}})
    assert _renamed(overlay) == ["attack", "rest"]
