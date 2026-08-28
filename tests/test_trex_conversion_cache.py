"""The shared TREx conversion: what is reused, what is refused, and what it costs.

A conversion is the expensive half of a TREx run and the tracking that follows it
is cheap, so a track-parameter sweep should convert once. What makes that safe
rather than merely fast is the address: a slot is named by both the convert-phase
digest *and* the source content identity, so a hit is proof that this ``.pv`` was
made from these pixels under these detection settings. This file pins that
proof, and the four ways it can be lost -- a loosened key, a settings file that
does not reach the tracking argv, a published slot that is rewritten, and a
sweeper that reclaims one still in use.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive

import mosaic.tracking.trex.dataset_runs as dr
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.markers import (
    phase_fields,
    read_phase_marker,
    write_phase_marker,
)
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.core.pipeline.ops import OPS
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.tracking.trex.conversion_cache import (
    CONVERSION_STEM,
    CONVERT_KIND,
    conversion_run_id,
)
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.trex.run import TRexConvertResult, TRexTrackResult
from mosaic.tracking.trex.version import TREX_VERSION

# --- fixtures --------------------------------------------------------------


def _facts_cells(video_uuid: str) -> dict[str, object]:
    facts: MediaFacts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid=video_uuid,
        identity_scheme="video/1" if video_uuid else "",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def write_media(ds: Dataset, *, sequence: str, uid: str) -> None:
    """Rewrite the media index to hold one sequence with content identity *uid*."""
    media_root = ds.get_root(ds.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    video = media_root / f"{sequence}.mp4"
    if not video.exists():
        _ = video.write_bytes(b"fake")
    pd.DataFrame(
        [
            {
                "name": video.name,
                "group": "",
                "sequence": sequence,
                "group_safe": "",
                "sequence_safe": sequence,
                "camera": "",
                "abs_path": ds.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": 0,
                **_facts_cells(uid),
            }
        ]
    ).to_csv(media_root / "index.csv", index=False)


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """One sequence, ``vid1``, whose media carries a content identity."""
    manifest = new_dataset_manifest("cache", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media(dataset, sequence="vid1", uid="uid-vid1")
    return dataset


@dataclass
class FakeTrex:
    """Stand-ins for the two phases, recording what each was asked to do."""

    converted: list[Path] = field(default_factory=list)
    tracked: list[Path] = field(default_factory=list)
    convert_kwargs: list[dict[str, object]] = field(default_factory=list)
    track_kwargs: list[dict[str, object]] = field(default_factory=list)
    write_settings: bool = True
    on_convert: Callable[[Path], None] | None = None

    def convert(
        self,
        video_path: Path | list[Path],
        output_dir: Path,
        *,
        output_name: str | None = None,
        **kwargs: object,
    ) -> TRexConvertResult:
        given = (
            [Path(video_path)]
            if isinstance(video_path, (str, Path))
            else [Path(p) for p in video_path]
        )
        self.converted.append(given[0])
        self.convert_kwargs.append(dict(kwargs))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = output_name if output_name is not None else given[0].stem
        pv = out / f"{stem}.pv"
        _ = pv.write_bytes(b"pv")
        settings = out / f"{stem}.settings"
        if self.write_settings:
            _ = settings.write_text("detect_type = yolo\n")
        # TREx writes one of these at the end of every conversion whatever it is
        # asked for, which is why publishing has to remove it rather than rely on
        # a flag.
        _ = (out / f"{stem}.results").write_bytes(b"conversion results")
        if self.on_convert is not None:
            self.on_convert(out)
        return TRexConvertResult(
            pv_path=pv,
            settings_path=settings,
            background_path=None,
            stdout="",
            stderr="",
        )

    def track(
        self, pv_path: Path, output_dir: Path, **kwargs: object
    ) -> TRexTrackResult:
        self.tracked.append(Path(pv_path))
        self.track_kwargs.append(dict(kwargs))
        out = Path(output_dir)
        stem = Path(pv_path).stem
        data = out / "data"
        data.mkdir(parents=True, exist_ok=True)
        npz = data / f"{stem}_id0.npz"
        np.savez(
            npz,
            frame=np.arange(6),
            time=np.arange(6) / 30.0,
            cm_per_pixel=np.array([1.0]),
            **{
                "X#wcentroid": np.arange(6, dtype=float),
                "Y#wcentroid": np.arange(6, dtype=float),
            },
            poseX0=np.arange(6, dtype=float),
            poseY0=np.arange(6, dtype=float),
        )
        results = out / f"{stem}.results"
        _ = results.write_bytes(b"results")
        return TRexTrackResult(
            npz_paths=[npz],
            results_path=results,
            settings_path=out / f"{stem}.settings",
            stdout="",
            stderr="",
        )


@pytest.fixture
def trex(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeTrex]:
    fake = FakeTrex()
    monkeypatch.setattr(dr, "run_trex_convert", fake.convert)
    monkeypatch.setattr(dr, "run_trex_track", fake.track)
    yield fake


def slot_of(ds: Dataset, uid: str = "uid-vid1") -> Path:
    """The one published slot, found by walking rather than by recomputing it."""
    root = ds.get_root(CONVERT_KIND)
    return next(p for p in root.rglob(uid) if p.is_dir())


def slots(ds: Dataset) -> list[Path]:
    root = ds.get_root(CONVERT_KIND)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("uid-*") if p.is_dir())


# --- the address -----------------------------------------------------------


def test_the_slot_run_id_is_the_convert_phase_hash(ds: Dataset, trex: FakeTrex) -> None:
    """One digest, not two that can drift.

    The conversion run root's name and the ``params_hash`` every convert marker
    records are the same value by construction. A second hash function would let
    the gate and the address disagree, and every slot would then miss forever
    while looking correct.
    """
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    parsed = parse_op_run_id(slot.parent.name)
    assert parsed is not None
    assert parsed.kind == CONVERT_KIND
    assert parsed.version == TREX_VERSION

    marker = read_phase_marker(slot, "convert")
    assert marker is not None
    assert parsed.digest == marker.params_hash


def test_the_slot_name_carries_the_source_and_the_recipe(
    ds: Dataset, trex: FakeTrex
) -> None:
    """Both terms in the path is what makes a published slot immutable."""
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    assert slot.name == "uid-vid1"
    assert slot.parent.name.startswith(f"{CONVERT_KIND}.{TREX_VERSION}-")


def test_a_track_only_change_reuses_the_conversion(ds: Dataset, trex: FakeTrex) -> None:
    """The feature: a sweep converts once and tracks many times."""
    _ = dr.run_trex(ds, TrexParams(track_max_speed=50))
    _ = dr.run_trex(ds, TrexParams(track_max_speed=120))
    _ = dr.run_trex(ds, TrexParams(track_max_speed=200))

    assert len(trex.converted) == 1, "a track-only change must not reconvert"
    assert len(trex.tracked) == 3
    assert len(slots(ds)) == 1


def test_a_convert_change_mints_a_new_slot(ds: Dataset, trex: FakeTrex) -> None:
    """A cache keyed too loosely would serve a `.pv` made under other detection."""
    _ = dr.run_trex(ds, TrexParams(detect_conf_threshold=0.5))
    _ = dr.run_trex(ds, TrexParams(detect_conf_threshold=0.9))

    assert len(trex.converted) == 2
    assert len(slots(ds)) == 2


def test_a_replaced_source_video_mints_a_different_slot(
    ds: Dataset, trex: FakeTrex
) -> None:
    """Same path, new bytes: never the same conversion, and never rewritten."""
    _ = dr.run_trex(ds, TrexParams())
    first = slot_of(ds, "uid-vid1")
    write_media(ds, sequence="vid1", uid="uid-replaced")
    _ = dr.run_trex(ds, TrexParams())

    assert len(trex.converted) == 2
    assert first.exists(), "a published slot is never rebuilt in place"
    assert (first / f"{CONVERSION_STEM}.pv").exists()
    assert len(slots(ds)) == 2


def test_media_without_a_uid_converts_in_place_and_says_so(
    tmp_path: Path, trex: FakeTrex, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path is a mutable key, so media with no content identity is not cached."""
    manifest = new_dataset_manifest("nouid", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media(dataset, sequence="vid1", uid="")

    run_id = dr.run_trex(dataset, TrexParams())
    assert slots(dataset) == []
    marker = read_phase_marker(dataset.get_root("trex") / run_id / "vid1", "convert")
    assert marker is not None
    assert "_tracking/trex/" in marker.recorded_output
    assert "reprobe-media" in capsys.readouterr().err


def test_a_cache_hit_and_a_fresh_convert_produce_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse must not leak into identity.

    A dataset that converted and one that hit the cache have to agree on the run
    identifier, on the tracks variant, and on the table itself. If any of the
    path or hit/miss decision reached ``trex_settings``, a sweep's runs would
    stop being comparable -- which is the entire purpose of the sweep.
    """
    warm_ids: list[str] = []
    tables: list[pd.DataFrame] = []
    for name in ("cold", "warm"):
        fake = FakeTrex()
        monkeypatch.setattr(dr, "run_trex_convert", fake.convert)
        monkeypatch.setattr(dr, "run_trex_track", fake.track)
        base = tmp_path / name
        base.mkdir()
        dataset = Dataset(manifest_path=new_dataset_manifest(name, base_dir=base)).load(
            ensure_roots=True
        )
        write_media(dataset, sequence="vid1", uid="uid-vid1")
        if name == "warm":
            # Prime the cache with a run at other tracking settings, so the run
            # under test is a hit rather than a conversion.
            _ = dr.run_trex(dataset, TrexParams(track_max_speed=999))
        warm_ids.append(dr.run_trex(dataset, TrexParams(track_max_speed=50)))
        table = sorted(dataset.get_root("tracks").rglob("*.parquet"))
        tables.append(pd.read_parquet(table[-1]))
        if name == "warm":
            assert len(fake.converted) == 1, "the second run must be a cache hit"

    assert warm_ids[0] == warm_ids[1]
    pd.testing.assert_frame_equal(tables[0], tables[1])


# --- the settings file, which is what carries detection into tracking -------


def test_the_track_argv_names_the_conversions_settings_file(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The sharpest regression this change can have.

    TREx looks for a settings file at ``<output_dir>/<pv stem>.settings`` and not
    beside the ``.pv``, so once a conversion is shared the implicit lookup finds
    nothing -- silently, because a merely composed name that does not exist is
    not an error. Everything the conversion recorded would fall back to what
    ``detect_type`` implies, and the run would say nothing about it.
    """
    _ = dr.run_trex(ds, TrexParams())
    settings_path = trex.track_kwargs[0]["settings_path"]
    assert isinstance(settings_path, Path)
    assert settings_path.is_absolute(), "a relative -s resolves under -d"
    assert settings_path == slot_of(ds) / f"{CONVERSION_STEM}.settings"
    assert settings_path.exists()


def test_no_settings_file_means_no_dash_s(ds: Dataset, trex: FakeTrex) -> None:
    """A named-but-missing ``-s`` is an error, so it is named only when it exists.

    The pre-marker directories a dataset may already hold never had one, and they
    must keep tracking exactly as they did.
    """
    trex.write_settings = False
    _ = dr.run_trex(ds, TrexParams())
    assert trex.track_kwargs[0]["settings_path"] is None


def test_a_conversion_without_settings_is_not_published(
    ds: Dataset, trex: FakeTrex, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused as a *shared* artifact, but never thrown away.

    Sharing a ``.pv`` whose parameters cannot be recovered would make every later
    run track under defaults. Discarding hours of detector inference over a
    missing sidecar would be worse, so the conversion stays with the run that
    made it -- which is exactly where it lived before this cache existed.
    """
    trex.write_settings = False
    run_id = dr.run_trex(ds, TrexParams())
    assert slots(ds) == [], (
        "an unpublished slot must not be left behind: an empty directory carries "
        "no marker, which the sweeper refuses as `foreign` permanently"
    )
    kept = ds.get_root("trex") / run_id / "vid1" / f"{CONVERSION_STEM}.pv"
    assert kept.exists(), "the conversion must survive, unshared"
    assert "instead of sharing it" in capsys.readouterr().err


def test_a_cache_hit_without_a_settings_file_reconverts(
    ds: Dataset, trex: FakeTrex
) -> None:
    """A `.pv` whose parameters are gone is a miss, never a silent hit."""
    _ = dr.run_trex(ds, TrexParams())
    (slot_of(ds) / f"{CONVERSION_STEM}.settings").unlink()
    shutil.rmtree(ds.get_root("trex"))

    _ = dr.run_trex(ds, TrexParams(track_max_speed=120))
    assert len(trex.converted) == 2


# --- publish ----------------------------------------------------------------


def test_the_conversion_results_file_is_not_published(
    ds: Dataset, trex: FakeTrex
) -> None:
    """TREx writes one unconditionally; it must not sit beside a shared `.pv`.

    A results load with no explicit path falls back to the *input* folder, which
    under this layout is the slot. Deleting it at publish makes that unreachable
    by construction rather than by argument.
    """
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    assert list(slot.glob("*.results")) == []
    assert list(slot.glob("*.results.meta")) == []
    assert (slot / f"{CONVERSION_STEM}.pv").exists()


def test_the_slot_holds_only_what_a_later_run_needs(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The published shape, pinned. A diff is intended and re-pinned, or a defect."""
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    assert sorted(p.name for p in slot.iterdir()) == [
        ".mosaic-convert.json",
        f"{CONVERSION_STEM}.pv",
        f"{CONVERSION_STEM}.settings",
    ]


def test_a_torn_publish_leaves_no_marker(ds: Dataset, trex: FakeTrex) -> None:
    """Reuse needs the marker *and* the output, so a partial slot is never served."""
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    (slot / ".mosaic-convert.json").unlink()
    shutil.rmtree(ds.get_root("trex"))

    _ = dr.run_trex(ds, TrexParams(track_max_speed=120))
    assert len(trex.converted) == 2, "a slot with no marker must not be bound"


def test_no_staging_directory_survives_a_clean_run(ds: Dataset, trex: FakeTrex) -> None:
    _ = dr.run_trex(ds, TrexParams())
    assert list(slot_of(ds).glob(".incoming-*")) == []


# --- adoption ---------------------------------------------------------------


def _legacy_conversion(ds: Dataset, trex: FakeTrex) -> Path:
    """Run once, then move the conversion back into the run directory.

    This is the shape every dataset converted before the cache existed is in: a
    `.pv` and its settings beside the tracking output, with a convert marker
    naming them.
    """
    run_id = dr.run_trex(ds, TrexParams())
    work_dir = ds.get_root("trex") / run_id / "vid1"
    slot = slot_of(ds)
    for name in (f"{CONVERSION_STEM}.pv", f"{CONVERSION_STEM}.settings"):
        shutil.move(str(slot / name), str(work_dir / name))
    marker = read_phase_marker(work_dir, "convert")
    assert marker is not None
    write_phase_marker(
        work_dir,
        marker.model_copy(
            update={
                "recorded_output": ds.relative_to_root(
                    work_dir / f"{CONVERSION_STEM}.pv"
                )
            }
        ),
    )
    shutil.rmtree(ds.get_root(CONVERT_KIND))
    return work_dir / f"{CONVERSION_STEM}.pv"


def test_adoption_hard_links_rather_than_copies(ds: Dataset, trex: FakeTrex) -> None:
    """28 GB appears in the cache at zero bytes, or it does not appear."""
    local_pv = _legacy_conversion(ds, trex)
    before = len(trex.converted)

    _ = dr.run_trex(ds, TrexParams())
    assert len(trex.converted) == before, "an adopted conversion is not redone"

    slot_pv = slot_of(ds) / f"{CONVERSION_STEM}.pv"
    assert slot_pv.exists()
    assert slot_pv.stat().st_ino == local_pv.stat().st_ino
    assert local_pv.stat().st_nlink == 2

    # And the run's own convert marker now names the slot. This is not
    # bookkeeping: the sweeper derives "still in use" from these markers, so an
    # adopted slot nothing repointed at is unpinned and reclaimable the moment
    # it ages, while the run that adopted it goes on reading it.
    work_dir = local_pv.parent
    marker = read_phase_marker(work_dir, "convert")
    assert marker is not None
    assert ds.resolve_path(marker.recorded_output) == slot_pv


def test_adoption_is_idempotent(ds: Dataset, trex: FakeTrex) -> None:
    """Running again must not republish over a live slot."""
    local_pv = _legacy_conversion(ds, trex)
    _ = dr.run_trex(ds, TrexParams())
    _ = dr.run_trex(ds, TrexParams(track_max_speed=120))

    assert len(slots(ds)) == 1
    assert local_pv.stat().st_nlink == 2


def test_a_failed_link_keeps_the_run_correct(
    ds: Dataset, trex: FakeTrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No copy fallback: the run simply uses the conversion where it is."""
    local_pv = _legacy_conversion(ds, trex)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(18, "Cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    before = len(trex.converted)
    run_id = dr.run_trex(ds, TrexParams())

    assert len(trex.converted) == before, "the local conversion is still used"
    assert local_pv.stat().st_nlink == 1
    marker = read_phase_marker(ds.get_root("trex") / run_id / "vid1", "convert")
    assert marker is not None
    assert ds.resolve_path(marker.recorded_output) == local_pv


def test_a_marker_without_provenance_is_reused_but_not_adopted(
    ds: Dataset, trex: FakeTrex
) -> None:
    """ "Unknown is not mismatched" is right for reuse and wrong for a cache key.

    A directory adopted from before markers existed records no ``params_hash``
    and no ``source_uid``. Promoting it into a durable shared address would key
    28 GB on a guess.
    """
    local_pv = _legacy_conversion(ds, trex)
    work_dir = local_pv.parent
    marker = read_phase_marker(work_dir, "convert")
    assert marker is not None
    write_phase_marker(
        work_dir,
        marker.model_copy(
            update={"params_hash": "", "source_uid": "", "backfilled": True}
        ),
    )
    before = len(trex.converted)

    _ = dr.run_trex(ds, TrexParams())
    assert len(trex.converted) == before, "it is still reused in place"
    assert slots(ds) == [], "and never promoted into the cache"


# --- overwrite --------------------------------------------------------------


def test_overwrite_reuses_the_conversion(ds: Dataset, trex: FakeTrex) -> None:
    """``overwrite`` means re-track, not re-convert 28 GB."""
    _ = dr.run_trex(ds, TrexParams())
    _ = dr.run_trex(ds, TrexParams(), overwrite=True)

    assert len(trex.converted) == 1
    assert len(trex.tracked) == 2


def test_overwrite_never_touches_a_slot(ds: Dataset, trex: FakeTrex) -> None:
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    stamp = (slot / f"{CONVERSION_STEM}.pv").stat().st_mtime_ns

    _ = dr.run_trex(ds, TrexParams(), overwrite=True)
    assert (slot / f"{CONVERSION_STEM}.pv").stat().st_mtime_ns == stamp


# --- the argv the conversion must never carry -------------------------------


def test_the_convert_argv_never_carries_auto_train(ds: Dataset, trex: FakeTrex) -> None:
    """``-task convert`` with ``auto_train`` loads a results file unconditionally.

    That path sets TRex's load flag whether or not the conversion wrote anything,
    and a results load with no explicit path reaches into the input folder -- the
    shared slot. ``auto_train`` is a tracking setting and only the tracking argv
    may carry it.
    """
    _ = dr.run_trex(ds, TrexParams(auto_train=True))
    for kwargs in trex.convert_kwargs:
        assert "auto_train" not in kwargs
        extra = kwargs.get("extra_settings") or {}
        assert isinstance(extra, dict)
        assert "auto_train" not in extra
        assert "load" not in extra
    tracked = trex.track_kwargs[0]["params"]
    assert isinstance(tracked, TrexParams)
    assert tracked.auto_train is True


# --- what the conversion root is, and is not --------------------------------


def test_the_conversion_root_is_not_an_op() -> None:
    """A storage root, never a command.

    Registering it would put it in ``mosaic track``'s verb list and make the
    graph declare it as writing ``tracks/``, which it does not.
    """
    assert CONVERT_KIND not in OPS
    assert TRACKING_ROOTS[CONVERT_KIND].retention == "conversion"


def test_the_conversion_root_declares_a_phase() -> None:
    """An empty ``phase_outputs`` reads as never-complete, which is deletable."""
    assert TRACKING_ROOTS[CONVERT_KIND].phase_outputs
    assert TRACKING_ROOTS[CONVERT_KIND].phases == ("convert",)


def test_the_index_row_names_the_slot_directory(ds: Dataset, trex: FakeTrex) -> None:
    """The sweeper keys on the basename of ``abs_path``.

    Pointing it at the ``.pv`` would make every slot read as ``unrowed`` -- which
    is refused, so the cache would grow without bound while reporting cleanly.
    """
    _ = dr.run_trex(ds, TrexParams())
    index = pd.read_csv(ds.get_root(CONVERT_KIND) / "index.csv")
    assert len(index) == 1
    row = index.iloc[0]
    assert Path(str(row["abs_path"])).name == "uid-vid1"
    assert str(row["group"]) in ("", "nan")
    assert str(row["sequence"]) == "uid-vid1"


def test_the_slot_name_is_a_pure_function_of_the_convert_settings() -> None:
    """``conversion_run_id`` addresses; it does not sample anything else.

    The same convert-phase settings must always name the same slot, and a
    changed one must always name a different slot. Anything else here -- a
    timestamp, a path, a counter -- would either scatter one conversion across
    many addresses or collapse two recipes onto one.
    """
    settings = {"detect_type": "yolo", "detect_conf_threshold": 0.5}
    assert conversion_run_id(settings) == conversion_run_id(dict(settings))
    assert conversion_run_id(settings) != conversion_run_id(
        {**settings, "detect_conf_threshold": 0.9}
    )
    parsed = parse_op_run_id(conversion_run_id(settings))
    assert parsed is not None
    assert parsed.digest == hash_params(settings)


# --- contention -------------------------------------------------------------


def _unpublish(slot: Path) -> None:
    """Leave the slot directory in place but make it read as never published.

    The state a conversion killed partway leaves behind: a directory, possibly
    some debris, and no completion marker.
    """
    (slot / ".mosaic-convert.json").unlink(missing_ok=True)
    for name in (f"{CONVERSION_STEM}.pv", f"{CONVERSION_STEM}.settings"):
        (slot / name).unlink(missing_ok=True)


def test_a_held_slot_is_waited_for_and_the_entry_claim_survives(
    ds: Dataset, trex: FakeTrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Waiting on a peer's conversion must not cost this run its own directory.

    Two things are pinned here. The conversion is **not** repeated -- that is the
    point of a shared slot. And this entry's own claim keeps being re-stamped
    while the wait goes on: a claim carries its expiry, so a run that waited out
    a multi-hour conversion in silence would be read as abandoned -- stolen by
    the next execution, and reclaimed by a concurrent sweep as ``expired_claim``,
    which is deletable.
    """
    from mosaic.core.pipeline.markers import (
        new_inflight,
        read_inflight,
        write_inflight,
    )

    # Run once to build the real slot, then put it back into the state a peer
    # mid-conversion would leave: present, claimed, nothing published.
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    published_marker = read_phase_marker(slot, "convert")
    assert published_marker is not None
    _unpublish(slot)
    write_inflight(
        slot,
        new_inflight(
            execution_id="peer-exec",
            host="peer",
            pid=1,
            phase="convert",
            idle_seconds=3600,
        ),
    )
    shutil.rmtree(ds.get_root("trex"))
    trex.converted.clear()

    seen_expiry: list[str] = []

    def fake_sleep(_seconds: float) -> None:
        work_dir = next(ds.get_root("trex").glob("*/vid1"))
        entry_claim = read_inflight(work_dir)
        assert entry_claim is not None, "the entry's own claim vanished while waiting"
        seen_expiry.append(entry_claim.expires_at)
        if len(seen_expiry) < 2:
            return
        # The peer finishes: publish the slot and drop its claim.
        _ = (slot / f"{CONVERSION_STEM}.pv").write_bytes(b"pv")
        _ = (slot / f"{CONVERSION_STEM}.settings").write_text("detect_type = yolo\n")
        write_phase_marker(slot, published_marker)
        (slot / ".mosaic-inflight.json").unlink(missing_ok=True)

    monkeypatch.setattr(dr.time, "sleep", fake_sleep)
    _ = dr.run_trex(ds, TrexParams())

    assert len(seen_expiry) >= 2, "the wait loop was never exercised"
    assert trex.converted == [], "a held slot must be waited for, not duplicated"
    assert seen_expiry[0] != seen_expiry[-1], (
        "the entry's claim was never re-stamped while waiting, so a long wait "
        "would let a concurrent sweep delete this run's working directory"
    )


def test_a_crashed_conversion_leaves_no_staging_tree(
    ds: Dataset, trex: FakeTrex
) -> None:
    """A slot holding a partial `.pv` and no marker is refused, never reclaimed.

    So the staging tree has to go on the failure path too, or one crashed
    conversion makes its slot permanently unreclaimable and permanently unusable.
    """

    def explode(_out: Path) -> None:
        raise RuntimeError("detector died")

    trex.on_convert = explode
    with pytest.raises(RuntimeError):
        _ = dr.run_trex(ds, TrexParams())

    leftovers = list(ds.get_root(CONVERT_KIND).rglob(".incoming-*"))
    assert leftovers == [], f"staging survived a crash: {leftovers}"
    assert slots(ds) == [], "and the empty slot went with it"


def test_a_stale_staging_tree_is_cleared_by_the_next_conversion(
    ds: Dataset, trex: FakeTrex
) -> None:
    """Every ``.incoming-*``, not only this execution's.

    One left by a run that died on another host, or before this cleanup existed,
    would otherwise sit in the slot forever -- and a slot carrying debris and no
    marker is refused by the sweeper rather than reclaimed.
    """
    _ = dr.run_trex(ds, TrexParams())
    slot = slot_of(ds)
    _unpublish(slot)
    stale = slot / ".incoming-someone-else"
    stale.mkdir(parents=True)
    _ = (stale / f"{CONVERSION_STEM}.pv").write_bytes(b"partial")
    shutil.rmtree(ds.get_root("trex"))
    trex.converted.clear()

    _ = dr.run_trex(ds, TrexParams())

    assert not stale.exists()
    assert len(trex.converted) == 1, "an unpublished slot must be reconverted"
    assert (slot_of(ds) / f"{CONVERSION_STEM}.pv").exists()


def test_visual_identification_probabilities_are_not_read_as_tracks(
    ds: Dataset, trex: FakeTrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``auto_train`` writes a ``_vi_probs.npz`` beside the per-id exports.

    It holds a ``probs`` matrix and nothing else -- no positions, no
    ``cm_per_pixel`` -- so a bridge that globs ``data/*.npz`` hands it to the
    converter, which refuses it for a calibration it was never going to carry.
    The entry then fails *after* TRex tracked it successfully, and a run using
    visual identification publishes nothing at all.

    The membership test is the individual-id suffix, not the directory.
    """
    from mosaic.core.pipeline.tracks_index import read_tracks_index

    # Wrap what the fixture installed. Rebinding ``trex.track`` would not work:
    # the fixture patched the module with the already-bound method.
    tracked = dr.run_trex_track

    def track_with_vi_probs(pv_path: Path, output_dir: Path, **kwargs: object):
        result = tracked(pv_path, output_dir, **kwargs)
        np.savez(
            Path(output_dir) / "data" / f"{Path(pv_path).stem}_vi_probs.npz",
            probs=np.zeros((6, 2), dtype=float),
        )
        return result

    monkeypatch.setattr(dr, "run_trex_track", track_with_vi_probs)

    variant = dr.run_trex(ds, TrexParams(auto_train=True))

    published = read_tracks_index(ds)
    assert len(published) == 1, "the entry published, so visual identity is usable"
    assert published.iloc[0]["run_id"] == variant
    assert int(published.iloc[0]["n_rows"]) == 6, (
        "the per-id export is what became the table"
    )

    trex_index = pd.read_csv(ds.get_root("trex") / "index.csv")
    assert int(trex_index.iloc[0]["n_ids"]) == 1, (
        "n_ids counts individuals, and the probability matrix is not one"
    )


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("conversion_id0", True),
        ("vid1_fish12", True),
        # TRex names exports `<output_prefix>_id<N>`; an empty prefix leaves the
        # tail as the whole name, which is still one individual's export.
        ("fish0", True),
        ("bee3", True),
        ("conversion_vi_probs", False),  # what auto_train writes
        ("conversion", False),
    ],
)
def test_which_trex_exports_are_one_individual(stem: str, expected: bool) -> None:
    """The membership test is the id suffix, not the directory it sits in."""
    from mosaic.core.track_library.trex import is_per_individual_export

    assert is_per_individual_export(f"{stem}.npz") is expected


def test_naming_the_keypoints_re_tracks_but_reuses_the_conversion(
    ds: Dataset, trex: FakeTrex
) -> None:
    """``detect_keypoint_count`` is a track key, and the distinction is the point.

    It changes which columns are *exported*, never what was detected -- so it
    has to move the tracking identity (the table really is different) while
    leaving the ``.pv`` alone. Getting that backwards would make asking for
    keypoints re-run detection over the whole video, which is the expensive
    phase and the one the shared conversion cache exists to avoid.
    """
    first = dr.run_trex(ds, TrexParams())
    converted_once = list(trex.converted)
    assert len(converted_once) == 1

    second = dr.run_trex(ds, TrexParams(detect_keypoint_count=7))

    assert second != first, "the exported columns changed, so the run must"
    assert trex.converted == converted_once, (
        "the detection pass was repeated for a change that only renames columns"
    )
    assert len(trex.tracked) == 2, "and the tracking pass was not"


def test_the_keypoint_count_reaches_the_tracking_call(
    ds: Dataset, trex: FakeTrex
) -> None:
    """It is the tracking phase that names the columns, not the conversion."""
    _ = dr.run_trex(ds, TrexParams(detect_keypoint_count=7))

    tracked = trex.track_kwargs[0]["params"]
    assert isinstance(tracked, TrexParams)
    assert tracked.detect_keypoint_count == 7
    assert "detect_keypoint_count" not in phase_fields(TrexParams, "convert"), (
        "the conversion must not name the exported columns"
    )
