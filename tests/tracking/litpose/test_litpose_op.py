"""The registered ``litpose`` op.

Its resource class, its run id against ``run_litpose`` called directly, the
throughput knobs kept out of that run id, and model identity by config and weights.
"""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.ops import OPS, describe_op, run_op
from mosaic.core.scope import Scope

from tests.helpers import scope_over, stub_media_dataset, write_litpose_model


def test_litpose_registered_as_gpu_convert_op():
    assert "litpose" in OPS
    d = describe_op("litpose")
    assert d["category"] == "convert"
    assert {"model_path", "litpose_overrides"} <= set(d["params_schema"]["properties"])
    from mosaic.core.pipeline.ops import op_resource_class

    # declared "gpu" despite category "convert" (LP video inference needs the GPU)
    assert op_resource_class("litpose") == "gpu"


def test_litpose_op_run_id_matches_standalone_run_litpose(tmp_path):
    # LitposeOp must produce the same content run_id as calling run_litpose directly
    # for the same settings. Scope to a missing sequence so the run short-circuits
    # (empty media) after the model resolves but before any litpose binary is used.
    from mosaic.tracking import run_litpose
    from mosaic.tracking.litpose.params import LitposeParams

    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    model = write_litpose_model(tmp_path / "lp_model")
    absent = ("", "nonexistent")
    direct = run_litpose(ds, LitposeParams(model_path=str(model)), scope_over(absent))
    via_op = run_op(
        ds,
        "litpose",
        {"model_path": str(model)},
        scope=Scope(entries=[absent]),
    )
    assert direct == via_op
    assert direct.startswith("litpose.2.3-")


def test_litpose_params_exclude_throughput_from_run_id():
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.litpose.params import LitposeParams

    a = LitposeParams(
        model_path="m",
        precision="fp32",
        idle_timeout=900,
        max_runtime=None,
        convert_to_tracks=True,
    )
    b = LitposeParams(
        model_path="m",
        precision="fp16",
        idle_timeout=30,
        max_runtime=60,
        convert_to_tracks=False,
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    c = LitposeParams(
        model_path="m", litpose_overrides={"data.image_resize_dims.height": 256}
    )
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


def test_litpose_model_identity_is_content_not_path(tmp_path):
    # Two model directories with identical config + weights mint the same model_id
    # (and so the same run_id); different weights mint a different one.
    from mosaic.tracking.model_refs import resolve_model_set

    a = write_litpose_model(tmp_path / "a", weights=b"same-weights")
    b = write_litpose_model(tmp_path / "b", weights=b"same-weights")
    c = write_litpose_model(tmp_path / "c", weights=b"other-weights")

    id_a = resolve_model_set(None, [str(a)], "litpose").model_id
    id_b = resolve_model_set(None, [str(b)], "litpose").model_id
    id_c = resolve_model_set(None, [str(c)], "litpose").model_id
    assert id_a == id_b  # same content, different paths -> same identity
    assert id_a != id_c  # different weights -> different identity


def test_litpose_config_is_part_of_identity(tmp_path):
    # config.yaml shapes the output (resize dims, keypoint names), so it reaches
    # identity: same weights + different config -> different run.
    from mosaic.tracking.model_refs import resolve_model_set

    a = write_litpose_model(tmp_path / "a")
    b = write_litpose_model(tmp_path / "b", keypoint_names=("nose", "tail", "mid"))
    assert (
        resolve_model_set(None, [str(a)], "litpose").model_id
        != resolve_model_set(None, [str(b)], "litpose").model_id
    )


def test_litpose_unresolvable_model_raises(tmp_path):
    from mosaic.tracking.model_refs import resolve_model_set

    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(tmp_path / "missing")], "litpose")
    # a checkpoint but no config.yaml
    no_config = tmp_path / "no_config"
    ckpt = no_config / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"w")
    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(no_config)], "litpose")
    # a config.yaml but no checkpoint
    no_ckpt = tmp_path / "no_ckpt"
    no_ckpt.mkdir()
    (no_ckpt / "config.yaml").write_text("model: {}\n")
    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(no_ckpt)], "litpose")
