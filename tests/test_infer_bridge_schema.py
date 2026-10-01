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

import inspect
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import tracks_index
from mosaic.core.pipeline.placement import EntryAxis
from mosaic.core.pipeline.tracking_roots import tracking_output_schema
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import ensure_track_schema
from mosaic.tracking.common import bridge as bridge_module
from mosaic.tracking.common.bridge import BridgeCounts
from mosaic.tracking.external.runner.ultralytics_protocol import (
    POINT_COLUMNS,
    pose_columns,
)
from mosaic.tracking.ops import infer as infer_module
from mosaic.tracking.ops.infer import _bridge_df_to_tracks
from mosaic.tracking.pose_training.localizer_inference import (
    LocalizerDetection,
    LocalizerFrame,
    localizer_detections_to_dataframe,
)
from tests.helpers import clip_facts

_N_KEYPOINTS = 3
_N_ROWS = 4
_TWO_BLIND_FRAMES = (LocalizerFrame(0, ()), LocalizerFrame(1, ()))


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
            LocalizerFrame(
                i, ({"x": 1.0 + i, "y": 5.0 + i, "confidence": 0.9, "class_id": 0},)
            )
            for i in range(_N_ROWS)
        ],
        class_names=["bee"],
    )


_PRODUCERS = {
    "infer-pose": _pose_predictions,
    "infer-points": _point_predictions,
    "infer-localizer": _localizer_predictions,
}


_NO_DETECTIONS = {
    "infer-pose": lambda: _pose_predictions().iloc[0:0],
    "infer-points": lambda: _point_predictions().iloc[0:0],
    "infer-localizer": lambda: localizer_detections_to_dataframe(_TWO_BLIND_FRAMES),
}
"""Each producer's predictions for a video without a detection.

The runner writes its full column set without a row. The localizer's frames come
from its real builder, given two frames without a detection.
"""


def _publish(
    tmp_path: Path, kind: str, predictions: pd.DataFrame
) -> tuple[BridgeCounts, pd.DataFrame]:
    """Run *predictions* through *kind*'s bridge, and read the published table back."""
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
        predictions,
        "",
        "vid1",
        tracks_variant=variant,
        producer_run_id=variant,
        kind=kind,
        seq_dir=seq_dir,
        consumed_media=[video],
        model_pt=model,
        timing=clip_facts(),
        axis=EntryAxis(),
        frames_read=None,
    )

    rows = read_tracks_index(ds)
    assert len(rows) == 1
    table = pd.read_parquet(ds.resolve_path(str(rows.iloc[0]["abs_path"])))
    assert int(rows.iloc[0]["n_rows"]) == len(table)
    return written, table


def _bridge(tmp_path: Path, kind: str) -> pd.DataFrame:
    """Run one kind's predictions through the bridge and read the table back."""
    written, table = _publish(tmp_path, kind, _PRODUCERS[kind]())
    assert written.n_rows == _N_ROWS
    return table


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


@pytest.mark.parametrize("kind", sorted(_NO_DETECTIONS))
def test_predictions_without_a_row_publish_an_empty_table(
    tmp_path: Path, kind: str
) -> None:
    """A video without a detection has a result, and it is published.

    The table is empty and contains every column that the schema requires.
    """
    written, table = _publish(tmp_path, kind, _NO_DETECTIONS[kind]())

    assert (written.n_rows, written.n_ids) == (0, 0)
    assert table.empty
    _, report = ensure_track_schema(
        table, tracking_output_schema(kind), strict=True, source=kind
    )
    assert report["missing_required"] == []


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


def test_the_localizer_table_has_the_same_types_with_and_without_a_row() -> None:
    """An empty localizer table is typed as a full one, column by column.

    Both are written as predictions and published as tracks. Two types for one
    column would give two parquet schemas for one producer's output.
    """
    detection: LocalizerDetection = {
        "x": 1.0,
        "y": 2.0,
        "confidence": 0.9,
        "class_id": 0,
    }
    full = localizer_detections_to_dataframe(
        [LocalizerFrame(0, (detection,))], class_names=["bee"]
    )
    empty = localizer_detections_to_dataframe(_TWO_BLIND_FRAMES, class_names=["bee"])

    assert empty.empty
    assert list(full.columns) == list(POINT_COLUMNS)
    assert empty.dtypes.to_dict() == full.dtypes.to_dict()


def _spy_on_tracks_rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record each tracks-index row that the bridge writes, in place of writing it.

    The writer is replaced in every module on the inference path that binds it.
    Each call is bound to the writer's signature with its defaults applied. An
    argument passed at its default and one left out record the same value.
    """
    signature = inspect.signature(tracks_index.write_tracks_row)
    calls: list[dict[str, object]] = []

    def spy(*args: object, **kwargs: object) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))

    for module in (infer_module, bridge_module):
        if "write_tracks_row" in vars(module):
            monkeypatch.setattr(module, "write_tracks_row", spy)
    return calls


def test_a_fixed_frame_publishes_the_pinned_row_and_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin every argument of the tracks row and every cell of the table.

    The frame lacks ``id``, whose fallback runs, and ``time``, which is each
    frame's time at the 30 fps of *timing*. It has two keypoints, and the body
    center is derived from them.
    """
    kind = "infer-pose"
    ds = _dataset(tmp_path, kind)
    seq_dir = ds.get_root(kind) / "run" / "g__s"
    seq_dir.mkdir(parents=True, exist_ok=True)
    video = ds.get_root("media_raw") / "s.mp4"
    video.write_bytes(b"v")
    model = ds.get_root("models") / kind / "best.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"w")
    rows = _spy_on_tracks_rows(monkeypatch)
    variant = "infer-pose.9.9-aaaaaaaaaa"
    keypoints = {
        "poseX0": [1.0, 2.0, 3.0],
        "poseY0": [10.0, 20.0, 30.0],
        "poseP0": [0.9, 0.8, 0.7],
        "poseX1": [5.0, 6.0, 7.0],
        "poseY1": [14.0, 24.0, 34.0],
        "poseP1": [0.6, 0.5, 0.4],
    }

    _ = _bridge_df_to_tracks(
        ds,
        pd.DataFrame({"frame": [0, 1, 2], **keypoints}),
        "g",
        "s",
        tracks_variant=variant,
        producer_run_id="infer-pose.9.9-bbbbbbbbbb",
        kind=kind,
        seq_dir=seq_dir,
        consumed_media=[video],
        model_pt=model,
        timing=clip_facts(),
        axis=EntryAxis(),
        frames_read=5,
    )

    out_path = ds.get_root("tracks") / variant / "g__s.parquet"
    assert len(rows) == 1
    row = rows[0]
    assert row.pop("ds") is ds
    assert row == {
        "run_id": variant,
        "group": "g",
        "sequence": "s",
        "out_path": out_path,
        "producer": kind,
        "std_format": "mosaic_v1",
        "n_rows": 3,
        "producer_run_id": "infer-pose.9.9-bbbbbbbbbb",
        "source": seq_dir,
        "source_md5": "",
        "consumed_source_roots": ("media_raw", "models"),
        "records_media": True,
        "media_frames": None,
        "frames_read": 5,
    }
    pd.testing.assert_frame_equal(
        pd.read_parquet(out_path),
        pd.DataFrame(
            {
                "frame": [0, 1, 2],
                **keypoints,
                "group": ["g"] * 3,
                "sequence": ["s"] * 3,
                "id": [0, 0, 0],
                "time": [0.0, 1 / 30, 2 / 30],
                "X": [3.0, 4.0, 5.0],
                "Y": [12.0, 22.0, 32.0],
            }
        ),
    )


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
            consumed_media=[video],
            model_pt=model,
            timing=clip_facts(),
            axis=EntryAxis(),
            frames_read=None,
        )
