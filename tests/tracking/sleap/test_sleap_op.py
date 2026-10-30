"""The registered ``sleap`` op.

Its resource class, its run id against ``run_sleap`` called directly, the
throughput knobs kept out of that run id, model identity by content and order,
the shape of a run whose last model is an embedding model, and the count of
appearance vectors without a tracks row that its index row records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import mosaic.tracking.sleap.dataset_runs as dataset_runs
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.markers import INFLIGHT_MARKER_NAME
from mosaic.core.pipeline.ops import OPS, describe_op, run_op
from mosaic.core.scope import Scope
from mosaic.tracking.common.bridge import BridgeCounts
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.sleap.run import SleapNotFoundError
from mosaic.tracking.sleap.version import SLEAP_NN_EMBEDDING_MIN

from tests.helpers import (
    EMBEDDING_TRACKER_SETTINGS,
    FakeSleap,
    install_fake_sleap,
    scope_over,
    stub_media_dataset,
    write_sleap_model,
    write_sleap_model_set,
)


def test_sleap_registered_as_gpu_convert_op():
    assert "sleap" in OPS
    d = describe_op("sleap")
    assert d["category"] == "convert"
    assert {"model_paths", "use_flow"} <= set(d["params_schema"]["properties"])
    from mosaic.core.pipeline.ops import op_resource_class

    # declared "gpu" despite category "convert" (SLEAP inference wants the GPU)
    assert op_resource_class("sleap") == "gpu"


def test_sleap_op_run_id_matches_standalone_run_sleap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # SleapOp must produce the same content run_id as calling run_sleap directly for the
    # same settings. Scope to a missing sequence so the run short-circuits (empty media)
    # after the model resolves and the environment is probed.
    from mosaic.tracking import run_sleap
    from mosaic.tracking.sleap.params import SleapParams

    _ = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    model = write_sleap_model(tmp_path / "model")
    absent = ("", "nonexistent")
    direct = run_sleap(ds, SleapParams(model_paths=[str(model)]), scope_over(absent))
    via_op = run_op(
        ds,
        "sleap",
        {"model_paths": [str(model)]},
        scope=Scope(entries=[absent]),
    )
    assert direct == via_op
    assert direct.startswith("sleap.1.6-")


def test_sleap_params_exclude_throughput_from_run_id():
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.sleap.params import SleapParams

    a = SleapParams(
        model_paths=["m"],
        batch_size=4,
        device=None,
        idle_timeout=900,
        max_runtime=None,
        convert_to_tracks=True,
    )
    b = SleapParams(
        model_paths=["m"],
        batch_size=16,
        device="cpu",
        idle_timeout=30,
        max_runtime=60,
        convert_to_tracks=False,
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    c = SleapParams(model_paths=["m"], peak_threshold=0.5)
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


def test_sleap_model_identity_is_content_not_path(tmp_path):
    # Two model directories with identical weights mint the same model_id (and so
    # the same run_id); different weights mint a different one. "Name the weights,
    # not the path they sat at."
    from mosaic.tracking.model_refs import resolve_model_set

    a = write_sleap_model(tmp_path / "a" / "model", b"same-weights")
    b = write_sleap_model(tmp_path / "b" / "model", b"same-weights")
    c = write_sleap_model(tmp_path / "c" / "model", b"other-weights")

    id_a = resolve_model_set(None, [str(a)], "sleap").model_id
    id_b = resolve_model_set(None, [str(b)], "sleap").model_id
    id_c = resolve_model_set(None, [str(c)], "sleap").model_id
    assert id_a == id_b  # same content, different paths -> same identity
    assert id_a != id_c  # different content -> different identity


def test_sleap_model_order_is_significant(tmp_path):
    # Top-down passes two directories (centroid, then centered-instance); the order
    # is not interchangeable, so it must reach identity.
    from mosaic.tracking.model_refs import resolve_model_set

    d1 = write_sleap_model(tmp_path / "centroid", b"centroid")
    d2 = write_sleap_model(tmp_path / "instance", b"instance")
    forward = resolve_model_set(None, [str(d1), str(d2)], "sleap").model_id
    reverse = resolve_model_set(None, [str(d2), str(d1)], "sleap").model_id
    assert forward != reverse


def _assert_nothing_minted(ds: Dataset, fake: FakeSleap) -> None:
    from mosaic.tracking.sleap.dataset_runs import sleap_index_path

    assert fake.tracked == []
    assert not sleap_index_path(ds).exists()
    assert list(ds.get_root("tracks").rglob("params.json")) == []
    assert list(ds.base_dir.rglob("run_params.json")) == []


@pytest.mark.parametrize(
    ("heads", "settings", "message"),
    [
        (
            ("embedding", "bottomup"),
            {},
            r"position 1 of 2, and an embedding model goes last",
        ),
        (
            ("bottomup", "embedding", "embedding"),
            {},
            r"more than one embedding model",
        ),
        (("embedding",), {}, r"needs a detection model before it"),
        (("centroid", "embedding"), {}, r"needs a detection model before it"),
        (
            ("centered_instance", "embedding"),
            {},
            r"needs a detection model before it",
        ),
        (
            ("bottomup", "single_instance", "embedding"),
            {},
            r"needs a detection model before it",
        ),
        (
            ("multi_class_bottomup", "embedding"),
            {},
            r"retracks the identity classes",
        ),
        (
            ("centroid", "multi_class_topdown", "embedding"),
            {},
            r"retracks the identity classes",
        ),
        (
            ("bottomup", "embedding"),
            {"tracking": False},
            r"only for tracked detections",
        ),
        (
            ("bottomup", "embedding"),
            {"use_flow": True},
            r"does not take optical flow",
        ),
        (
            ("bottomup", "embedding"),
            {"convert_to_tracks": False},
            r"only beside a published tracks table",
        ),
        (
            ("bottomup", "embedding"),
            {"features": "keypoints"},
            r"features='embeddings', scoring_method='cosine_sim' and "
            r"candidates_method='local_queues'",
        ),
        (
            ("bottomup", "embedding"),
            {"scoring_method": "oks"},
            r"features='embeddings', scoring_method='cosine_sim'",
        ),
        (
            ("bottomup", "embedding"),
            {"candidates_method": "fixed_window"},
            r"features='embeddings', scoring_method='cosine_sim'",
        ),
        (
            ("bottomup",),
            {"features": "embeddings"},
            r"features='embeddings' compares appearance vectors",
        ),
    ],
    ids=[
        "not-last",
        "two-embedding-models",
        "no-detection-model",
        "centroid-without-centered-instance",
        "centered-instance-without-centroid",
        "two-detection-models",
        "multi-class-bottomup",
        "multi-class-topdown",
        "untracked",
        "optical-flow",
        "no-tracks-table",
        "features",
        "scoring-method",
        "candidates-method",
        "embeddings-without-an-embedding-model",
    ],
)
def test_an_embedding_run_of_the_wrong_shape_is_refused_before_minting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    heads: tuple[str, ...],
    settings: dict[str, object],
    message: str,
) -> None:
    """A run that cannot write appearance tables is refused before it is named.

    The refusal leaves no run root, no tracks variant and no run index, and
    probes no SLEAP environment.
    """
    fake = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])
    models = write_sleap_model_set(tmp_path, *heads)

    with pytest.raises(ValueError, match=message):
        _ = run_op(
            ds,
            "sleap",
            {"model_paths": models, **EMBEDDING_TRACKER_SETTINGS, **settings},
        )

    _assert_nothing_minted(ds, fake)
    assert fake.probed == 0


def test_a_refusal_names_each_embedding_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])
    models = write_sleap_model_set(tmp_path, "bottomup", "embedding", "embedding")

    with pytest.raises(ValueError, match="more than one embedding model") as caught:
        _ = run_op(ds, "sleap", {"model_paths": models, **EMBEDDING_TRACKER_SETTINGS})

    assert repr(models[1]) in str(caught.value)
    assert repr(models[2]) in str(caught.value)


@pytest.mark.parametrize(
    "heads",
    [("bottomup", "embedding"), ("centroid", "centered_instance", "embedding")],
    ids=["bottomup", "top-down"],
)
def test_an_embedding_model_set_runs_after_one_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, heads: tuple[str, ...]
) -> None:
    """An embedding model last, after its detection models, mints a sleap run."""
    fake = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])
    models = write_sleap_model_set(tmp_path, *heads)

    run_id = run_op(
        ds,
        "sleap",
        {"model_paths": models, **EMBEDDING_TRACKER_SETTINGS},
        scope=Scope(entries=[("", "nonexistent")]),
    )

    assert run_id.startswith("sleap.1.6-")
    assert fake.probed == 1


def test_a_run_without_an_embedding_model_probes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])

    _ = run_op(
        ds,
        "sleap",
        {"model_paths": write_sleap_model_set(tmp_path, "bottomup")},
        scope=Scope(entries=[("", "nonexistent")]),
    )

    assert fake.probed == 1


@pytest.mark.parametrize("heads", [("bottomup",), ("bottomup", "embedding")])
def test_an_environment_without_the_analysis_export_is_refused_before_minting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, heads: tuple[str, ...]
) -> None:
    """A run whose export would fail after inference is refused before it starts.

    The refusal leaves no claim, no run root and no tracks variant, and names
    the piece the environment lacks.
    """
    fake = install_fake_sleap(monkeypatch, FakeSleap(has_analysis_export=False))
    ds = stub_media_dataset(tmp_path, ["vid1"])
    models = write_sleap_model_set(tmp_path, *heads)
    settings = EMBEDDING_TRACKER_SETTINGS if "embedding" in heads else {}

    with pytest.raises(SleapNotFoundError, match="save_analysis_h5"):
        _ = run_op(ds, "sleap", {"model_paths": models, **settings})

    assert fake.probed == 1
    _assert_nothing_minted(ds, fake)
    assert list(ds.base_dir.rglob(INFLIGHT_MARKER_NAME)) == []


def test_an_embedding_run_is_refused_below_the_sleap_nn_release_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_sleap(monkeypatch, FakeSleap(version="0.3.4"))
    ds = stub_media_dataset(tmp_path, ["vid1"])
    models = write_sleap_model_set(tmp_path, "bottomup", "embedding")

    with pytest.raises(ValueError, match="sleap-nn '0.3.4'") as caught:
        _ = run_op(ds, "sleap", {"model_paths": models, **EMBEDDING_TRACKER_SETTINGS})

    assert SLEAP_NN_EMBEDDING_MIN in str(caught.value)
    assert fake.probed == 1
    _assert_nothing_minted(ds, fake)


def test_an_embedding_model_without_a_vector_length_is_refused_before_minting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])
    detector = write_sleap_model_set(tmp_path, "bottomup")
    reid = write_sleap_model(
        tmp_path / "reid", b"reid", head="embedding", merged=True, embedding_dim=None
    )

    with pytest.raises(ValueError, match="embedding_dim"):
        _ = run_op(
            ds,
            "sleap",
            {"model_paths": [*detector, str(reid)], **EMBEDDING_TRACKER_SETTINGS},
        )

    _assert_nothing_minted(ds, fake)


def test_sleap_unresolvable_model_raises(tmp_path):
    from mosaic.tracking.model_refs import resolve_model_set

    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(tmp_path / "missing")], "sleap")
    empty = tmp_path / "no_ckpt"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(empty)], "sleap")


def _unmatched_cells(ds: Dataset) -> list[str]:
    """Return the ``n_vectors_unmatched`` cells of the sleap index, as written."""
    index = pd.read_csv(
        dataset_runs.sleap_index_path(ds), dtype=str, keep_default_na=False
    )
    return index["n_vectors_unmatched"].tolist()


def test_a_run_without_an_embedding_model_records_a_blank_unmatched_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh bridge and a reused one both leave the count blank."""
    fake = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])
    params = SleapParams(model_paths=[str(write_sleap_model(tmp_path / "model"))])

    _ = dataset_runs.run_sleap(ds, params)
    assert _unmatched_cells(ds) == [""]

    _ = dataset_runs.run_sleap(ds, params)
    assert len(fake.converted) == 1
    assert _unmatched_cells(ds) == [""]


def test_the_unmatched_count_of_a_bridge_reaches_the_index_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])

    def measured(*_args: object, **_kwargs: object) -> BridgeCounts:
        return BridgeCounts(n_rows=6, n_ids=1, n_vectors_unmatched=2)

    monkeypatch.setattr(dataset_runs, "bridge_analysis_h5_to_tracks", measured)
    _ = dataset_runs.run_sleap(
        ds, SleapParams(model_paths=[str(write_sleap_model(tmp_path / "model"))])
    )

    assert _unmatched_cells(ds) == ["2"]


def test_the_index_row_names_every_model_of_the_set_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])
    models = write_sleap_model_set(tmp_path, "centroid", "centered_instance")

    _ = dataset_runs.run_sleap(ds, SleapParams(model_paths=models))

    index = pd.read_csv(
        dataset_runs.sleap_index_path(ds), dtype=str, keep_default_na=False
    )
    assert list(index["model_type"]) == ["centroid+centered_instance"]


def test_the_index_row_of_one_model_names_its_type_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_fake_sleap(monkeypatch)
    ds = stub_media_dataset(tmp_path, ["vid1"])
    models = write_sleap_model_set(tmp_path, "bottomup")

    _ = dataset_runs.run_sleap(ds, SleapParams(model_paths=models))

    index = pd.read_csv(
        dataset_runs.sleap_index_path(ds), dtype=str, keep_default_na=False
    )
    assert list(index["model_type"]) == ["bottomup"]
