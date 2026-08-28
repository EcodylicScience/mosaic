"""Tests for the imgstore export op and the tracker-side path it feeds.

The export exists so that a tool which opens a path itself -- T-Rex, SLEAP,
Lightning Pose -- can read a recording mosaic stores as a directory. Two
properties carry the whole design and are asserted hardest here: the export is
frame-for-frame faithful to the store, and registering it does not change what
mosaic's own readers open.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imgstore")

from mosaic_media.transcode import TranscodeError  # noqa: E402

from mosaic.core.dataset import Dataset  # noqa: E402
from mosaic.core.media.video_io import open_frame_reader  # noqa: E402
from mosaic.core.pipeline.ops import ScopeRefused, run_op  # noqa: E402
from mosaic.core.scope import Scope  # noqa: E402
from mosaic.core.pipeline.store_export import (  # noqa: E402
    EXPORT_TARGET,
    StoreExportParams,
    export_recipe_hash,
)
from mosaic.tracking.common.scope import TrackerWorkItem  # noqa: E402
from mosaic.tracking.common.tool_input import (  # noqa: E402
    StoreExportMissingError,
    resolve_tool_input,
)

_SYNC_UUID = "f064059f9ea046429f227bc7addab1eb"

# The lowest AV1 CRF, so a frame comes back close enough to its source that the
# per-frame tag is still readable. Frames are written with ``fill=True`` for the
# same reason: this writer encodes 4:2:0, whose chroma planes are subsampled 2x2,
# so a one-pixel tag on an otherwise black frame does not survive the round trip
# at any quality. Uniform frames do, which keeps the assertion about the export's
# ordering rather than about the encoder's fidelity.
_LOSSLESS = 0

MakeStore = Callable[..., tuple[Path, list[np.ndarray]]]


def _camera_meta(serial: str, uuid: str) -> dict[str, object]:
    """Motif document-root metadata for one camera of a synced recording."""
    return {
        "camera_serial": serial,
        "synchronizationuuid": uuid,
        "synchronization": "framenumber",
    }


def _store_dataset(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
    *,
    cameras: list[str] | None = None,
    nframes: int = 12,
    chunksize: int = 5,
) -> tuple[Dataset, str, str]:
    """A dataset holding one indexed store per camera, and its (group, sequence).

    Stores are written into ``media_raw`` and indexed from there, which is the
    shape the op requires: a dataset whose originals index is separate from the
    derivative index under ``media``.
    """
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    search = ds.get_root("media_raw") / "recordings"
    search.mkdir(parents=True, exist_ok=True)
    for serial in cameras or [""]:
        name = f"rec.{serial}" if serial else "rec"
        extra = _camera_meta(serial, _SYNC_UUID) if serial else None
        make_imgstore(
            name=name,
            nframes=nframes,
            chunksize=chunksize,
            parent=search,
            fill=True,
            extra_metadata=extra,
        )
    ds.index_media([search])
    row = _originals(ds).iloc[0]
    return ds, str(row["group"]), str(row["sequence"])


def _export(
    ds: Dataset,
    group: str,
    sequence: str,
    camera: str | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Export one entry, or one camera of it when *camera* names one.

    The camera is part of the entry key rather than a field beside it. A
    triple names one camera, and a pair names every camera of the entry.
    """
    selector = (
        Scope(entries=[(group, sequence, camera)])
        if camera is not None
        else Scope(entries=[(group, sequence)])
    )
    _ = run_op(
        ds,
        "export-store",
        StoreExportParams(av1_crf=_LOSSLESS),
        scope=selector,
        overwrite=overwrite,
    )


def _exports(ds: Dataset) -> list[Path]:
    root = ds.get_root("media") / "transcode"
    return sorted(root.glob("*.mp4")) if root.is_dir() else []


def _read_all(path: Path) -> list[np.ndarray]:
    with open_frame_reader(path, target="raw") as reader:
        return [frame for _, frame in reader]


def _tags(frames: list[np.ndarray]) -> list[float]:
    """One representative value per frame, robust to a lossy round trip."""
    return [float(np.median(frame)) for frame in frames]


def _originals(ds: Dataset) -> pd.DataFrame:
    """The originals index, read as text so an empty cell stays an empty string."""
    return pd.read_csv(ds.get_root("media_raw") / "index.csv", dtype=str).fillna("")


def test_an_export_reproduces_every_store_frame_in_order(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """The property the whole op exists for: exported frame i is store frame i."""
    ds, group, sequence = _store_dataset(
        tmp_path, make_media_dataset, make_imgstore, nframes=12, chunksize=5
    )
    _export(ds, group, sequence)

    exports = _exports(ds)
    assert len(exports) == 1

    # Compared against what mosaic reads from the *store*, not against the
    # fixture's internals: the claim is that the two read the same frames in the
    # same order, which is exactly what a tracker comparing tables from both
    # relies on. Crossing chunk 0 -> 1 -> 2 (chunksize 5 over 12 frames) is part
    # of what this covers.
    stored = _tags(_read_all(ds.resolve_media(group, sequence).paths[0]))
    exported = _tags(_read_all(exports[0]))
    assert len(exported) == len(stored) == 12
    assert len(set(stored)) == 12, "the fixture's frames must be distinguishable"
    # A tolerance, because the encode is lossy: the assertion is that frame i
    # came back as frame i, not that the codec is bit-exact.
    assert exported == pytest.approx(stored, abs=3.0)


def test_an_export_does_not_change_what_mosaic_reads(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """Routing stays inert, so the native store read path keeps being exercised.

    This is the regression that protects every in-process consumer -- most
    visibly the Ultralytics tracker, which reads a store directly and would
    silently start reading an mp4 instead if an export changed routing.
    """
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    before = ds.resolve_media(group, sequence).paths
    _export(ds, group, sequence)
    after = ds.resolve_media(group, sequence).paths

    assert after == before
    assert after[0].is_dir(), "a store must still resolve to its directory"


def test_an_export_registers_both_links(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """A back-link row under ``media`` and a forward link on the store's row."""
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    _export(ds, group, sequence)
    export = _exports(ds)[0]

    store_row = _originals(ds).iloc[0]
    assert store_row["analysis_derivative_path"] == f"transcode/{export.name}"
    assert store_row["playback_derivative_path"] == "", (
        "an export claims the analysis link only"
    )

    derivatives = pd.read_csv(ds.get_root("media") / "index.csv", dtype=str).fillna("")
    assert len(derivatives) == 1
    derivative = derivatives.iloc[0]
    assert derivative["source_video_uuid"] == store_row["video_uuid"]
    assert derivative["group"] == group
    assert derivative["sequence"] == sequence
    assert derivative["recipe_hash"]


def test_an_export_records_the_encoder_it_wrote_with(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """An export encodes, so its derivative row names the encoder like any other.
    The value is read off the writer rather than assumed: nothing else on the row
    carries it, since codec is measured and reads "av1" whichever encoder ran."""
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    _export(ds, group, sequence)

    derivatives = pd.read_csv(ds.get_root("media") / "index.csv", dtype=str).fillna("")
    assert derivatives.iloc[0]["encoder"] == "libsvtav1"


def test_a_second_export_reuses_the_first(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """The recipe-addressed name plus the forward link gate the re-encode."""
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    _export(ds, group, sequence)
    export = _exports(ds)[0]
    stamp = export.stat().st_mtime_ns

    _export(ds, group, sequence)

    assert _exports(ds) == [export]
    assert export.stat().st_mtime_ns == stamp, "the export was re-encoded"


def test_overwrite_forces_a_re_export(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An export already on disk is rebuilt rather than reused.

    Counted at ``write_export``, the call that decodes and re-encodes every
    frame. A reused store never calls it. The count is therefore zero for a
    reuse, and the file still being on disk cannot satisfy it.
    """
    import mosaic.core.pipeline.store_export as store_export_module

    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    _export(ds, group, sequence)

    written: list[Path] = []

    def counted(
        store: Path,
        dest: Path,
        *args: object,
        real: Callable[..., object] = store_export_module.write_export,
        **kwargs: object,
    ) -> object:
        """Record the destination, then export it for real.

        *real* is bound as a default so it keeps the open callable type this
        wrapper forwards to. Read as a module attribute it narrows back to the
        concrete signature, which a ``*args`` forward cannot satisfy.
        """
        written.append(dest)
        return real(store, dest, *args, **kwargs)

    monkeypatch.setattr(store_export_module, "write_export", counted)

    _export(ds, group, sequence)
    assert written == [], "the reuse gate must skip a store already linked"

    _export(ds, group, sequence, overwrite=True)
    assert len(written) == 1, "overwrite must re-export the store it would reuse"


def test_two_entries_are_refused_naming_both(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """One export covers one entry, and the refusal says which two were given.

    The arity used to be carried by a singular ``entry`` field, which a caller
    could only satisfy one way. It is the ``scope_takes`` declaration now, and
    one shared checker raises for every op that states an arity.
    """
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)

    with pytest.raises(ScopeRefused, match="covers one entry") as caught:
        _ = run_op(
            ds,
            "export-store",
            StoreExportParams(),
            scope=Scope(entries=[(group, sequence), ("other", "entry")]),
        )

    message = str(caught.value)
    assert sequence in message and "entry" in message


def test_a_camera_triple_exports_that_camera_and_a_pair_exports_every_one(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """The camera is part of the entry key, and a pair means every camera.

    Both halves in one test, because the pair's meaning is only visible beside
    the triple's: a triple that exported everything would pass a test that
    asserted the pair alone.
    """
    ds, group, sequence = _store_dataset(
        tmp_path, make_media_dataset, make_imgstore, cameras=["CAMA", "CAMB"]
    )

    _export(ds, group, sequence, camera="CAMA")
    assert len(_exports(ds)) == 1, "a triple exports the one camera it names"

    _export(ds, group, sequence)
    assert len(_exports(ds)) == 2, "a pair exports every camera of the entry"


def test_two_triples_of_one_entry_export_both_named_cameras(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """Two cameras of one entry are one entry to the arity check and two stores.

    The case the ``Scope.cameras`` set exists for, and the one an
    implementation reading a single camera would get wrong: it passes
    ``exactly-one`` because the check counts pairs, and it exports both.
    """
    ds, group, sequence = _store_dataset(
        tmp_path, make_media_dataset, make_imgstore, cameras=["CAMA", "CAMB", "CAMC"]
    )

    _ = run_op(
        ds,
        "export-store",
        StoreExportParams(av1_crf=_LOSSLESS),
        scope=Scope(entries=[(group, sequence, "CAMA"), (group, sequence, "CAMB")]),
    )

    assert len(_exports(ds)) == 2, "both named cameras, and not the third"


def test_a_camera_the_entry_does_not_have_is_named_as_such(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """A mistyped camera reads as a mistyped camera, not as a plain video.

    Both failures reach one empty store list. The message has to tell them
    apart, or a caller who typed CAMZ goes looking at the media type.
    """
    ds, group, sequence = _store_dataset(
        tmp_path, make_media_dataset, make_imgstore, cameras=["CAMA", "CAMB"]
    )

    with pytest.raises(TranscodeError, match="no imgstore is recorded under camera"):
        _export(ds, group, sequence, camera="CAMZ")


def test_each_camera_of_a_recording_exports_separately(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """Two synced cameras are one sequence, and each store gets its own export.

    The filenames are keyed on each store's own ``video_uuid``, so they cannot
    collide even though the two rows share a (group, sequence).
    """
    ds, group, sequence = _store_dataset(
        tmp_path, make_media_dataset, make_imgstore, cameras=["CAMA", "CAMB"]
    )
    originals = _originals(ds)
    assert set(originals["camera"]) == {"CAMA", "CAMB"}
    assert originals["sequence"].nunique() == 1

    _export(ds, group, sequence)

    assert len(_exports(ds)) == 2
    links = set(_originals(ds)["analysis_derivative_path"])
    assert len(links) == 2, "each camera links to its own export"


def test_exporting_one_camera_leaves_the_other_unexported(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """``camera`` selects, and selecting does not rename what it does export."""
    ds, group, sequence = _store_dataset(
        tmp_path, make_media_dataset, make_imgstore, cameras=["CAMA", "CAMB"]
    )
    _export(ds, group, sequence, camera="CAMA")

    assert len(_exports(ds)) == 1
    rows = _originals(ds)
    linked = rows[rows["analysis_derivative_path"] != ""]
    assert list(linked["camera"]) == ["CAMA"]

    # camera is excluded from the recipe, so exporting the rest afterwards must
    # leave the first camera's file exactly where it already is.
    first = _exports(ds)[0]
    _export(ds, group, sequence)
    assert first in _exports(ds)
    assert len(_exports(ds)) == 2


def test_the_recipe_ignores_scope_and_tracks_the_encode() -> None:
    """No coverage field to reach the recipe hash, and an encode knob in it.

    The coverage used to be two params fields marked ``HASH_EXCLUDE``. It is
    an argument to the run now. The recipe therefore cannot read it at all.
    The coverage enters ``export_run_id`` instead, through the identities of
    the stores exported.
    """
    declared = set(StoreExportParams.model_fields)
    assert not declared & {"entry", "entries", "camera", "cameras"}
    base = StoreExportParams()
    assert export_recipe_hash(base) != export_recipe_hash(
        StoreExportParams(av1_crf=_LOSSLESS)
    )


def test_a_plain_video_is_refused_rather_than_exported(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
    requires_ffmpeg: None,
) -> None:
    """An entry with nothing to export errors rather than re-encoding a video.

    A plain video is already the thing a tool can open, so exporting one would be
    a pointless re-encode -- and, worse, would register a derivative that makes
    the original look like it needed one.
    """
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    source = ds.get_root("media_raw") / "plain" / "clip.mp4"
    write_cfr_mp4(source)
    ds.index_media([source.parent])
    row = _originals(ds).iloc[0]

    with pytest.raises(TranscodeError, match="no imgstore rows"):
        _export(ds, str(row["group"]), str(row["sequence"]))
    del requires_ffmpeg


# --- the tracker-side boundary -------------------------------------------


def _work_item(ds: Dataset, group: str, sequence: str) -> TrackerWorkItem:
    resolved = ds.resolve_media(group, sequence)
    return TrackerWorkItem(
        group=group,
        sequence=sequence,
        key=sequence,
        video_paths=(resolved.paths[0],),
        fps=resolved.facts[0].fps,
        source_facts=(resolved.facts[0],),
    )


def test_a_tool_input_resolves_a_store_to_its_export(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    _export(ds, group, sequence)

    resolved = resolve_tool_input(ds, _work_item(ds, group, sequence), kind="trex")
    assert resolved == _exports(ds)[0]
    assert resolved.is_file()


def test_a_tool_input_without_an_export_names_the_command(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """The failure has to tell the user what to run, not just that it failed."""
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    with pytest.raises(StoreExportMissingError, match="export-store"):
        resolve_tool_input(ds, _work_item(ds, group, sequence), kind="trex")


def test_a_tool_input_passes_a_plain_video_through(
    scenario_dataset_with_media: Dataset,
) -> None:
    """A video file is handed to the tool untouched -- no export, no lookup."""
    ds = scenario_dataset_with_media
    rows = pd.read_csv(
        ds.get_root(ds.resolve_media_root()) / "index.csv", dtype=str
    ).fillna("")
    row = rows.iloc[0]
    item = _work_item(ds, str(row["group"]), str(row["sequence"]))
    assert resolve_tool_input(ds, item, kind="trex") == item.video_path


def test_a_tool_input_reports_a_link_whose_file_is_gone(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: MakeStore,
) -> None:
    """A deleted export is a distinct failure from one that never existed."""
    ds, group, sequence = _store_dataset(tmp_path, make_media_dataset, make_imgstore)
    _export(ds, group, sequence)
    _exports(ds)[0].unlink()

    with pytest.raises(StoreExportMissingError, match="does not exist"):
        resolve_tool_input(ds, _work_item(ds, group, sequence), kind="trex")


def test_the_export_target_is_the_analysis_link() -> None:
    """Pinned: the column an export claims is what routing and pruning read."""
    assert EXPORT_TARGET == "analysis"
