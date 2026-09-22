"""Every inference bridge writes the schema its tracking root declares.

Nothing asserted this before, and all three producers were violating it. Each
declares ``output_schema="mosaic_v1"``, which *requires* ``X``/``Y`` and defines
them as the individual's body centre; none of them wrote those columns. The
bridge validated with ``strict=False``, where a missing required column is
printed and not raised, so the report went out under a line reading ``completed``
and the table was written and indexed:

    [schema:mosaic_v1] ... {'missing_required': ['X', 'Y'], ...}
    [infer-pose] completed run_id=infer-pose.0.2-870e02e197 (1/1)

Downstream that is not an incomplete table but a silently useless one: the
overlay's ``centroid_cols`` yields ``None``, ``nearest-neighbor`` and the
social-force chain find no column, and the two crop features fall back *to* the
centre that is missing.

The frames below are built by the real producers -- ``pose_columns`` and
``POINT_COLUMNS`` from the wire protocol, and ``localizer_detections_to_dataframe``
itself -- rather than by a copy of their column lists here, so a producer that
renames a column fails this rather than drifting past it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracking_roots import tracking_output_schema
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import ensure_track_schema
from mosaic.tracking.external.runner.ultralytics_protocol import (
    POINT_COLUMNS,
    pose_columns,
)
from mosaic.tracking.ops.infer import _bridge_df_to_tracks
from mosaic.tracking.pose_training.localizer_inference import (
    localizer_detections_to_dataframe,
)

_N_KEYPOINTS = 3
_N_ROWS = 4


def _dataset(base: Path, kind: str) -> Dataset:
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={
            "tracks": str(base / "tracks"),
            "_tracking": str(base / "_tracking"),
            kind: str(base / "_tracking" / kind),
            "media_raw": str(base / "media_raw"),
            "models": str(base / "models"),
        },
    )
    ds.ensure_roots()
    ds.save()
    return ds


def _pose_predictions() -> pd.DataFrame:
    """What the Ultralytics runner publishes for a pose model.

    Each keypoint sits somewhere different, so the body centre is the mean of
    all three and equals none of them -- a bridge that copied one keypoint
    through would satisfy the schema and fail the value assertion below.
    """
    columns = pose_columns(_N_KEYPOINTS)
    values: dict[str, list[float]] = {"frame": [float(i) for i in range(_N_ROWS)]}
    values["id"] = [0.0] * _N_ROWS
    for k in range(_N_KEYPOINTS):
        values[f"poseX{k}"] = [float(2 * k)] * _N_ROWS
        values[f"poseY{k}"] = [float(6 * k)] * _N_ROWS
        values[f"poseP{k}"] = [0.9] * _N_ROWS
    return pd.DataFrame(values)[columns]


def _point_predictions() -> pd.DataFrame:
    """What the POLO runner publishes: one located point per detection."""
    return pd.DataFrame(
        {
            "frame": list(range(_N_ROWS)),
            "detection_id": [0] * _N_ROWS,
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
            "confidence": [0.9] * _N_ROWS,
            "class_id": [0] * _N_ROWS,
            "class_name": ["bee"] * _N_ROWS,
        }
    )[list(POINT_COLUMNS)]


def _localizer_predictions() -> pd.DataFrame:
    """What mosaic's own heatmap localizer emits, through its real builder."""
    return localizer_detections_to_dataframe(
        [
            [{"x": 1.0 + i, "y": 5.0 + i, "confidence": 0.9, "class_id": 0}]
            for i in range(_N_ROWS)
        ],
        class_names=["bee"],
    )


_PRODUCERS = {
    "infer-pose": _pose_predictions,
    "infer-points": _point_predictions,
    "infer-localizer": _localizer_predictions,
}


def _bridge(tmp_path: Path, kind: str) -> pd.DataFrame:
    """Run one kind's predictions through the bridge and read the table back."""
    ds = _dataset(tmp_path, kind)
    seq_dir = ds.get_root(kind) / "run" / "vid1"
    seq_dir.mkdir(parents=True, exist_ok=True)
    video = ds.get_root("media_raw") / "vid1.mp4"
    video.write_bytes(b"v")
    model = ds.get_root("models") / kind / "best.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"w")

    variant = f"{kind}.9.9-aaaaaaaaaa"
    written = _bridge_df_to_tracks(
        ds,
        _PRODUCERS[kind](),
        "",
        "vid1",
        tracks_variant=variant,
        producer_run_id=variant,
        kind=kind,
        seq_dir=seq_dir,
        video_path=video,
        model_pt=model,
        overwrite=True,
    )
    assert written == _N_ROWS

    rows = read_tracks_index(ds)
    assert len(rows) == 1
    return pd.read_parquet(ds.resolve_path(str(rows.iloc[0]["abs_path"])))


@pytest.mark.parametrize("kind", sorted(_PRODUCERS))
def test_the_bridged_table_satisfies_the_schema_the_root_declares(
    tmp_path: Path, kind: str
) -> None:
    """The invariant, stated once per producer, over the declaration itself.

    ``strict=True`` rather than reading the report, so this fails the way the
    bridge now does rather than one assertion removed from it.
    """
    table = _bridge(tmp_path, kind)

    _, report = ensure_track_schema(
        table, tracking_output_schema(kind), strict=True, source=kind
    )

    assert report["missing_required"] == []
    assert report["missing_prefixes"] == []
    assert report["forbidden_present"] == []


@pytest.mark.parametrize("kind", sorted(_PRODUCERS))
def test_every_bridged_table_carries_a_finite_body_centre(
    tmp_path: Path, kind: str
) -> None:
    """Present is not enough -- an all-NaN column would pass a name check."""
    table = _bridge(tmp_path, kind)

    assert table["X"].notna().all()
    assert table["Y"].notna().all()


def test_the_pose_body_centre_is_the_mean_of_the_keypoints(tmp_path: Path) -> None:
    """A pose model localizes landmarks and no centre, so the centre is their mean.

    The same rule every tracker bridge already applied, and the reason this is
    derived at the bridge rather than added to the wire protocol: an Ultralytics
    pose model does predict a box, but reading it would be a protocol change on
    both sides of the subprocess boundary for an answer the keypoints already
    give.
    """
    table = _bridge(tmp_path, "infer-pose")

    assert (table["X"] == 2.0).all()  # mean(0, 2, 4)
    assert (table["Y"] == 6.0).all()  # mean(0, 6, 12)


@pytest.mark.parametrize("kind", ["infer-points", "infer-localizer"])
def test_a_point_producers_position_is_renamed_not_duplicated(
    tmp_path: Path, kind: str
) -> None:
    """These two already reported the body centre; only its name was wrong.

    Renamed rather than copied, because one position under two names in one
    table invites a reader to take the one the schema does not define.
    """
    table = _bridge(tmp_path, kind)

    assert not {"x", "y"} & set(table.columns)
    assert table["X"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert table["Y"].tolist() == [5.0, 6.0, 7.0, 8.0]


def test_a_table_with_no_position_at_all_is_refused(tmp_path: Path) -> None:
    """The bridge derives a centre; it does not fabricate one.

    A producer emitting neither keypoints nor a position has nothing to name, so
    validation must report the real shortfall rather than an invented column --
    and now raises rather than printing under a line that says ``completed``.
    """
    from mosaic.core.schema import TrackSchemaError

    ds = _dataset(tmp_path, "infer-pose")
    seq_dir = ds.get_root("infer-pose") / "run" / "vid1"
    seq_dir.mkdir(parents=True, exist_ok=True)
    video = ds.get_root("media_raw") / "vid1.mp4"
    video.write_bytes(b"v")
    model = ds.get_root("models") / "pose" / "best.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"w")

    with pytest.raises(TrackSchemaError, match="X"):
        _ = _bridge_df_to_tracks(
            ds,
            pd.DataFrame({"frame": range(_N_ROWS), "confidence": [0.9] * _N_ROWS}),
            "",
            "vid1",
            tracks_variant="infer-pose.9.9-aaaaaaaaaa",
            producer_run_id="infer-pose.9.9-aaaaaaaaaa",
            kind="infer-pose",
            seq_dir=seq_dir,
            video_path=video,
            model_pt=model,
            overwrite=True,
        )
