"""The registered ``sleap`` op.

Its resource class, its run id against ``run_sleap`` called directly, the
throughput knobs kept out of that run id, and model identity by content and order.
"""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.ops import OPS, describe_op, run_op
from mosaic.core.scope import Scope

from tests.helpers import scope_over, stub_media_dataset, write_sleap_model


def test_sleap_registered_as_gpu_convert_op():
    assert "sleap" in OPS
    d = describe_op("sleap")
    assert d["category"] == "convert"
    assert {"model_paths", "use_flow"} <= set(d["params_schema"]["properties"])
    from mosaic.core.pipeline.ops import op_resource_class

    # declared "gpu" despite category "convert" (SLEAP inference wants the GPU)
    assert op_resource_class("sleap") == "gpu"


def test_sleap_op_run_id_matches_standalone_run_sleap(tmp_path):
    # SleapOp must produce the same content run_id as calling run_sleap directly for the
    # same settings. Scope to a missing sequence so the run short-circuits (empty media)
    # after the model resolves but before any sleap binary is used.
    from mosaic.tracking import run_sleap
    from mosaic.tracking.sleap.params import SleapParams

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


def test_sleap_unresolvable_model_raises(tmp_path):
    from mosaic.tracking.model_refs import resolve_model_set

    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(tmp_path / "missing")], "sleap")
    empty = tmp_path / "no_ckpt"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_model_set(None, [str(empty)], "sleap")
