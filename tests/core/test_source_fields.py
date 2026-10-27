"""One builder of scan sources for the CLI and the MCP, and the locked writers beside it."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import open_dataset
from mosaic.core.manifest import (
    LabelsScanSource,
    MediaDivision,
    MediaScanSource,
    TracksScanSource,
)
from mosaic.core.source_fields import (
    SourceFields,
    SourceFieldsInvalid,
    scan_source_from_fields,
)
from tests.helpers import make_dataset


def test_each_kind_is_built_with_its_own_fields() -> None:
    tracks = scan_source_from_fields(
        SourceFields(
            kind="tracks",
            path="/data/calms",
            patterns=("*.npy",),
            src_format="calms21_npy",
            multi_sequences_per_file=True,
            group_from="filename",
        )
    )
    media = scan_source_from_fields(
        SourceFields(kind="media", path="/data/videos", extensions=("mp4",))
    )
    listed = scan_source_from_fields(
        SourceFields(kind="labels", path="/data/labels", id="ann", files=("a.csv",))
    )

    assert isinstance(tracks, TracksScanSource)
    assert (tracks.id, tracks.multi_sequences_per_file) == ("calms", True)
    assert isinstance(media, MediaScanSource)
    assert media.extensions == (".mp4",)
    assert isinstance(listed, LabelsScanSource)
    assert (listed.id, listed.mode) == ("ann", "files")


def test_a_media_source_takes_a_division() -> None:
    divided = scan_source_from_fields(
        SourceFields(
            kind="media",
            path="/data/stores",
            divide_frames=60000,
            recording_identity="camera",
            recording_name="cam7",
        )
    )
    whole = scan_source_from_fields(SourceFields(kind="media", path="/data/stores"))

    assert isinstance(divided, MediaScanSource)
    assert divided.division == MediaDivision(
        frames=60000, identity="camera", name="cam7"
    )
    assert isinstance(whole, MediaScanSource)
    assert whole.division is None


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        (
            SourceFields(kind="media", path="/v", patterns=("*.mp4",)),
            "patterns does not apply to a media source",
        ),
        (
            SourceFields(kind="tracks", path="/t", layout="stem"),
            "layout does not apply to a tracks source",
        ),
        (
            SourceFields(kind="tracks", path="/t", files=("a.npy",), patterns=("*",)),
            "a file list and patterns",
        ),
        (
            SourceFields(kind="tracks", path="/t", series="keypoints"),
            "series applies to labels",
        ),
        (
            SourceFields(kind="labels", path="/l", series="k", src_format="x"),
            "series and src_format",
        ),
        (
            SourceFields(kind="tracks", path="/t", src_format=" "),
            "src_format must not be empty",
        ),
        (
            SourceFields(kind="tracks", path="/t", divide_frames=60000),
            "divide_frames does not apply to a tracks source",
        ),
        (
            SourceFields(kind="labels", path="/l", recording_name="r"),
            "recording_name does not apply to a labels source",
        ),
        (
            SourceFields(kind="media", path="/v", recording_identity="camera"),
            "recording_identity applies only beside divide_frames",
        ),
        (
            SourceFields(kind="media", path="/v", divide_frames=0),
            "greater than 0",
        ),
        (
            SourceFields(
                kind="media", path="/v", divide_frames=600, recording_name="a__b"
            ),
            "contains '__'",
        ),
    ],
)
def test_fields_that_do_not_make_one_source_are_refused(
    fields: SourceFields, fragment: str
) -> None:
    with pytest.raises(SourceFieldsInvalid, match=fragment):
        _ = scan_source_from_fields(fields)


def test_a_locked_declaration_keeps_what_another_writer_saved(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    other = open_dataset(ds.manifest_path)
    other.manifest.notes = "written by another process"
    other.save()

    ds.declare_scan_source(
        TracksScanSource(
            id="calms",
            path=str(tmp_path),
            patterns=("*.npy",),
            src_format="calms21_npy",
        )
    )
    ds.update_tags({"species": "mouse"})
    orphaned = ds.undeclare_scan_source("tracks", "calms")

    reread = open_dataset(ds.manifest_path).manifest
    assert reread.notes == "written by another process"
    assert [(tag.name, tag.type, tag.value) for tag in reread.tags] == [
        ("species", "text", "mouse")
    ]
    assert not reread.sources.tracks
    assert orphaned == 0


def test_tags_take_the_type_of_their_first_value_and_keep_it(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    ds.update_tags({"n": 2, "rate": 1.5, "pilot": None, "ok": True})
    with pytest.raises(ValueError):
        ds.update_tags({"n": "two"})
    with pytest.raises(KeyError, match="no tag named 'absent'"):
        ds.update_tags({}, remove=["absent"])

    types = {tag.name: tag.type for tag in open_dataset(ds.manifest_path).manifest.tags}
    assert types == {"n": "int", "rate": "float", "pilot": "label", "ok": "bool"}
