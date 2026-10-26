"""``dataset_summary``: a dataset's manifest fields, counts, track schemas and frame rates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from mosaic.core import dataset_summary as dataset_summary_module
from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.dataset_summary import (
    SUMMARY_KINDS,
    DatasetCounts,
    DatasetSummary,
    FrameRateCount,
    DividedRecordingSummary,
    TrackSchemaCount,
    UndividedRecordingSummary,
    dataset_summary,
)
from mosaic.core.manifest import MANAGED_BY_KEY, DatasetTag
from mosaic.core.pipeline.graph import run_pipeline
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.media_index import frame_from_rows, write_media_index_rows
from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline import tracks_index as tracks_index_module
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.tracking import dataset_inventory
from tests.helpers import (
    MediaClip,
    speed_recipe,
    UPDATE_GOLDEN_ENV,
    add_tracks_table,
    add_tracks_variant,
    make_dataset,
    read_golden,
    regenerate_command,
    register_trained_model,
    track_table,
    updating_golden,
    write_golden,
    write_media_index,
)

GOLDEN = "dataset_summary_golden.json"
"""The ``mosaic describe --json`` document of the ``populated`` fixture."""

VARYING_FIELDS = ("manifest", "uuid", "created_at")
"""The summary fields that differ between two builds of one fixture."""

PIXEL_VARIANT = "convert-trex.0.2-1111111111"
LEGACY_VARIANT = "convert-trex.0.1-2222222222"
MODEL_KIND = "train-pose"
MODEL_RUN_ID = "train-pose.0.1-0123456789"


def _index_media(ds: Dataset, rates: list[object]) -> None:
    """Write an originals media index with one row per frame rate cell."""
    rows: list[dict[str, object]] = [
        {"group": "", "sequence": f"seq_{i}", "fps": rate}
        for i, rate in enumerate(rates)
    ]
    write_media_index_rows(
        ds.get_root("media_raw") / "index.csv", frame_from_rows(rows)
    )


@pytest.fixture
def populated(tmp_path: Path) -> Dataset:
    """Two tracks variants of two schemas, a feature run, a trained model and media."""
    ds = make_dataset(tmp_path / "ds", name="populated")
    add_tracks_variant(ds, PIXEL_VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    _ = run_pipeline(ds, speed_recipe())
    add_tracks_variant(ds, LEGACY_VARIANT, "seq_c")
    run_root = model_run_root(ds, MODEL_KIND, MODEL_RUN_ID)
    run_root.mkdir(parents=True)
    weights = run_root / "best.pt"
    _ = weights.write_bytes(b"weights")
    register_trained_model(ds, MODEL_KIND, MODEL_RUN_ID, weights)
    _index_media(ds, [30.0, 30.0, 29.97, ""])
    return ds


def test_the_counts_schemas_and_frame_rates_are_read_from_the_indexes(
    populated: Dataset,
) -> None:
    summary = dataset_summary(
        populated, found=dataset_inventory(populated, kinds=SUMMARY_KINDS)
    )

    assert summary.counts == DatasetCounts(
        sequences=3,
        recordings=7,
        divided_recordings=0,
        videos=4,
        tracks_variants=2,
        feature_runs=1,
        models=1,
    )
    assert summary.track_schemas == (
        TrackSchemaCount(name="mosaic_v1", tables=2),
        TrackSchemaCount(name="trex_v1", tables=1),
    )
    assert summary.frame_rates == (
        FrameRateCount(fps=29.97, videos=1),
        FrameRateCount(fps=30.0, videos=2),
    )
    assert summary.inventory_errors == ()


def test_a_media_row_shorter_than_its_header_counts_without_a_rate(
    populated: Dataset,
) -> None:
    index = populated.get_root("media_raw") / "index.csv"
    with index.open("a", encoding="utf-8") as stream:
        _ = stream.write(",seq_short\n")

    summary = dataset_summary(
        populated, found=dataset_inventory(populated, kinds=SUMMARY_KINDS)
    )

    assert summary.counts.videos == 5
    assert summary.frame_rates == (
        FrameRateCount(fps=29.97, videos=1),
        FrameRateCount(fps=30.0, videos=2),
    )


def test_rates_within_the_store_tolerance_are_counted_as_one_rate(
    tmp_path: Path,
) -> None:
    """Each store reports the rate measured from its timestamps.

    Stores of one 20 fps camera measure 20.0000797 and 20.0001013, which differ
    by less than the store tolerance. 29.97 and 30 differ by more.
    """
    ds = make_dataset(tmp_path / "ds")
    _index_media(ds, [20.0001013, 20.0000797, 20.0, 30.0, 29.97])

    summary = dataset_summary(ds, found=dataset_inventory(ds, kinds=SUMMARY_KINDS))

    assert summary.frame_rates == (
        FrameRateCount(fps=20.0, videos=3),
        FrameRateCount(fps=29.97, videos=1),
        FrameRateCount(fps=30.0, videos=1),
    )


def test_a_kind_the_inventory_did_not_report_counts_as_unknown(
    populated: Dataset,
) -> None:
    found = inventory(populated, kinds=["feature", "tracks-variant"])

    counts = dataset_summary(populated, found=found).counts

    assert counts.models is None
    assert counts.feature_runs == 1


def test_the_manifest_fields_are_echoed(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="echoed")
    ds.set_notes("pilot sessions")
    ds.define_tag(DatasetTag(name="cohort", type="text", value="spring"))
    write_media_index(ds, [MediaClip(group="night", sequence="s1")])
    ds.set_continuous_groups(["night"])
    ds.set_splits("test", ["held"])
    ds.set_splits("train", ["a"])

    summary = dataset_summary(open_dataset(ds.manifest_path, ensure_roots=False))

    assert summary.manifest == str(ds.manifest_path.absolute())
    assert summary.name == "echoed"
    assert summary.uuid == ds.uuid
    assert summary.notes == "pilot sessions"
    assert [tag.name for tag in summary.tags] == ["cohort"]
    assert summary.continuous_groups == ("night",)
    assert summary.splits == {"a": "train", "held": "test"}
    assert (summary.counts.recordings, summary.counts.divided_recordings) == (1, 1)
    assert summary.managed_by is None
    assert not summary.api_managed
    assert summary.counts.sequences == 0
    assert summary.counts.videos == 1
    assert summary.frame_rates == (FrameRateCount(fps=30.0, videos=1),)


def test_every_recording_is_listed_and_a_divided_one_with_its_first_and_last_sequence(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(group="night", sequence="s10", filename="s10.mp4"),
            MediaClip(group="night", sequence="s3", filename="s3.mp4"),
            MediaClip(group="night", sequence="s2", filename="s2.mp4"),
            MediaClip(group="arena", sequence="s1", filename="arena_s1.mp4"),
        ],
    )
    add_tracks_variant(ds, PIXEL_VARIANT, "s1")
    ds.set_continuous_groups(["night"])

    summary = dataset_summary(open_dataset(ds.manifest_path, ensure_roots=False))

    assert (summary.counts.recordings, summary.counts.divided_recordings) == (3, 1)
    assert summary.counts.sequences == 1
    assert summary.recordings == (
        UndividedRecordingSummary(name="s1", kind="undivided"),
        UndividedRecordingSummary(name="arena:s1", kind="undivided"),
        DividedRecordingSummary(
            name="night",
            kind="divided",
            n_sequences=3,
            first_sequence="night:s2",
            last_sequence="night:s10",
        ),
    )


def test_a_divided_recording_is_ordered_by_its_tracks_extents(
    tmp_path: Path,
) -> None:
    """The summary orders members as an overlapped run reads them.

    ``s1`` sorts first by name and starts later on the recording's frame axis.
    """
    ds = make_dataset(tmp_path / "ds")
    add_tracks_table(ds, "convert.a", track_table("night", "s1", start_frame=40))
    add_tracks_table(ds, "convert.a", track_table("night", "s2", n_rows=40))
    ds.set_continuous_groups(["night"])

    summary = dataset_summary(open_dataset(ds.manifest_path, ensure_roots=False))

    (night,) = summary.recordings
    assert isinstance(night, DividedRecordingSummary)
    assert (night.first_sequence, night.last_sequence) == ("night:s2", "night:s1")


def test_a_member_with_two_variants_leaves_its_recording_in_name_order(
    tmp_path: Path,
) -> None:
    """No variant is chosen for a summary, and the read does not refuse."""
    ds = make_dataset(tmp_path / "ds")
    for run_id in ("convert.a", "convert.b"):
        add_tracks_table(ds, run_id, track_table("night", "s2", start_frame=40))
        add_tracks_table(ds, run_id, track_table("night", "s10"))
    add_tracks_table(ds, "convert.a", track_table("day", "s3", start_frame=40))
    add_tracks_table(ds, "convert.a", track_table("day", "s4"))
    ds.set_continuous_groups(["night", "day"])

    summary = dataset_summary(open_dataset(ds.manifest_path, ensure_roots=False))

    by_name = {
        recording.name: (recording.first_sequence, recording.last_sequence)
        for recording in summary.recordings
        if isinstance(recording, DividedRecordingSummary)
    }
    assert by_name == {
        "night": ("night:s2", "night:s10"),
        "day": ("day:s4", "day:s3"),
    }


def _index_reads_describing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recordings: int
) -> int:
    """Return how many times describing *recordings* divided recordings reads the index."""
    ds = make_dataset(tmp_path / f"ds{recordings}")
    names = [f"night{n}" for n in range(recordings)]
    for name in names:
        add_tracks_table(ds, "convert.a", track_table(name, "s1"))
    ds.set_continuous_groups(names)
    reopened = open_dataset(ds.manifest_path, ensure_roots=False)
    found = inventory(reopened, kinds=SUMMARY_KINDS)
    reads = 0
    real = tracks_index_module.read_tracks_index

    def counting(dataset: Dataset) -> pd.DataFrame:
        nonlocal reads
        reads += 1
        return real(dataset)

    with monkeypatch.context() as patch:
        for module in (tracks_index_module, dataset_summary_module):
            patch.setattr(module, "read_tracks_index", counting)
        _ = dataset_summary(reopened, found=found)
    return reads


def test_the_tracks_index_is_read_once_however_many_recordings_are_divided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _index_reads_describing(tmp_path, monkeypatch, 1)

    assert _index_reads_describing(tmp_path, monkeypatch, 3) == one


def test_a_declared_group_without_entries_is_no_recording(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(group="night", sequence="s1")])
    ds.set_continuous_groups(["night"])
    write_media_index(ds, [MediaClip(group="day", sequence="s1")])

    summary = dataset_summary(open_dataset(ds.manifest_path, ensure_roots=False))

    assert summary.continuous_groups == ("night",)
    assert (summary.counts.recordings, summary.counts.divided_recordings) == (1, 0)
    assert [(recording.name, recording.kind) for recording in summary.recordings] == [
        ("day:s1", "undivided")
    ]


def test_a_managing_service_is_read_from_the_preserved_key(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    with ds.manifest_path.open("a", encoding="utf-8") as handle:
        _ = handle.write(f"{MANAGED_BY_KEY}: mosaic-api\n")

    summary = dataset_summary(open_dataset(ds.manifest_path, ensure_roots=False))

    assert summary.managed_by == "mosaic-api"
    assert summary.api_managed


def test_a_summary_without_splits_still_validates() -> None:
    """The field is additive with a default, so an older document still reads."""
    old: dict[str, object] = {
        "manifest": "m",
        "name": "n",
        "uuid": None,
        "manifest_version": 3,
        "version": "0.1",
        "created_at": None,
        "notes": "",
        "tags": [],
        "libraries": [],
        "continuous_groups": [],
        "managed_by": None,
        "api_managed": False,
        "counts": {
            "sequences": 0,
            "videos": 0,
            "tracks_variants": None,
            "feature_runs": None,
            "models": None,
        },
        "track_schemas": [],
        "frame_rates": [],
        "inventory_errors": [],
    }
    assert DatasetSummary.model_validate(old).splits == {}


def test_the_version_on_disk_is_reported_before_migration(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    payload = yaml.safe_load(ds.manifest_path.read_text(encoding="utf-8"))
    payload["manifest_version"] = 2
    _ = ds.manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    summary = dataset_summary(open_dataset(ds.manifest_path, ensure_roots=False))

    assert summary.manifest_version == 2


def test_the_summary_dumps_as_json_and_reads_back(populated: Dataset) -> None:
    summary = dataset_summary(populated)

    assert type(summary).model_validate_json(summary.model_dump_json()) == summary


def test_a_table_missing_from_disk_is_left_out_of_every_count(
    populated: Dataset,
) -> None:
    index = read_tracks_index(populated)
    legacy = index.loc[index["run_id"] == LEGACY_VARIANT, "abs_path"].iloc[0]
    populated.resolve_path(str(legacy)).unlink()

    summary = dataset_summary(populated)

    assert summary.counts.sequences == 2
    assert summary.track_schemas == (TrackSchemaCount(name="mosaic_v1", tables=2),)


def _wire(populated: Dataset) -> dict[str, object]:
    """Return the summary's JSON document with the fields that vary per build masked."""
    document = dataset_summary(
        populated, found=dataset_inventory(populated, kinds=SUMMARY_KINDS)
    ).model_dump(mode="json")
    return {**document, **dict.fromkeys(VARYING_FIELDS, "<varies>")}


def test_the_json_document_matches_golden(populated: Dataset) -> None:
    if updating_golden():
        write_golden(GOLDEN, _wire(populated))
        pytest.skip(f"{UPDATE_GOLDEN_ENV}=1: rewrote {GOLDEN}")

    assert _wire(populated) == read_golden(GOLDEN), (
        f"The summary document changed. If the change is intended, run "
        f"`{regenerate_command(__name__)}` and review the diff."
    )
