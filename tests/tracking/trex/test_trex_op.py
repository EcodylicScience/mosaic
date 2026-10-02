"""The registered ``trex`` op.

Its resource class, its run id against ``run_trex`` called directly, the
throughput knobs kept out of that run id, and the weights a training run id
resolves to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.ops import OPS, describe_op, run_op
from mosaic.core.scope import Scope

from tests.helpers import FakeTrex, install_fake_trex, scope_over, stub_media_dataset


def test_trex_registered_as_gpu_convert_op():
    assert "trex" in OPS
    d = describe_op("trex")
    assert d["category"] == "convert"
    assert {"detect_model", "track_max_individuals"} <= set(
        d["params_schema"]["properties"]
    )
    from mosaic.core.pipeline.ops import op_resource_class

    # declared "gpu" despite category "convert" (TREx needs the GPU for YOLO detect)
    assert op_resource_class("trex") == "gpu"


def test_trex_op_run_id_matches_standalone_run_trex(tmp_path):
    # TrexOp must produce the same content run_id as calling run_trex directly for the same
    # settings, so existing TREx tracks stay cache-valid after the op refactor. Scope to a
    # missing sequence so the run short-circuits (empty media) before any trex binary is used.
    from mosaic.tracking import run_trex
    from mosaic.tracking.trex.params import TrexParams

    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    absent = ("", "nonexistent")
    direct = run_trex(ds, TrexParams(), scope_over(absent))
    via_op = run_op(ds, "trex", {}, scope=Scope(entries=[absent]))
    assert direct == via_op
    assert direct.startswith("trex.")


def test_trex_params_exclude_throughput_from_run_id():
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.trex.params import TrexParams

    a = TrexParams(
        detect_model="m.pt",
        idle_timeout=900,
        max_runtime=None,
        convert_to_tracks=True,
    )
    b = TrexParams(
        detect_model="m.pt",
        idle_timeout=30,
        max_runtime=60,
        convert_to_tracks=False,
    )
    assert hash_params(a.identity_dump()) == hash_params(b.identity_dump())
    c = TrexParams(detect_model="other.pt")
    assert hash_params(c.identity_dump()) != hash_params(a.identity_dump())


def test_run_trex_resolves_detect_model_run_id_to_weights(tmp_path, monkeypatch):
    """run_trex must resolve a training run_id (detect_model) to its best.pt for TREx.

    Regression: previously the raw run_id string was passed to the trex ``-m`` flag,
    so the train->track handoff (``detect_model=<train run_id>``) gave TREx a
    non-existent model path.
    """
    from pathlib import Path

    from mosaic.core.pipeline.models import model_index_path, model_run_root
    from mosaic.tracking import run_trex
    from mosaic.tracking.ops.train import TrainedModelIndexRow, trained_model_index
    from mosaic.tracking.trex.params import TrexParams

    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])

    # Seed a trained-model index row + a fake best.pt (as train-points would).
    rid = "train-points-deadbeef01"
    run_root = model_run_root(ds, "train-points", rid)
    weights = run_root / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"pt")
    idx = trained_model_index(model_index_path(ds, "train-points"))
    idx.ensure()
    idx.append(
        [
            TrainedModelIndexRow(
                run_id=rid,
                kind="train-points",
                base_model="",
                base_run_id="",
                best_model_path=ds.relative_to_root(weights),
                metrics_path="",
                n_epochs=1,
                status="finished",
                abs_path=Path(ds.relative_to_root(run_root)),
            )
        ]
    )
    idx.mark_finished(rid)

    # Record the weights the conversion receives, then stop before tracking.
    class _Stop(Exception):
        pass

    def stop(_output_dir: Path) -> None:
        raise _Stop()

    fake = install_fake_trex(monkeypatch, FakeTrex(on_convert=stop))

    with pytest.raises(_Stop):
        run_trex(
            ds,
            TrexParams(detect_model=rid, detect_type="yolo"),
            scope_over(("", "vid1")),
        )

    # resolved run_id -> absolute best.pt
    (convert_kwargs,) = fake.convert_kwargs
    assert convert_kwargs["detect_model_path"] == weights
