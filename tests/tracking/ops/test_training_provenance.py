"""A trained model says what it was trained on, and prepared data is not a model.

Two gaps, one story. A training run root held weights and nothing that said how
they came to be: the data reference lived only in whatever submitted the run, so
nothing reading the dataset alone could say what a model had seen. And the
directory a model's training data is prepared into sits beside the models under
``models/``, where the inventory judged it by the trained-model rule and reported
every one as a finished row whose weights were missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.dataset_indexes import iter_dataset_indexes
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.tracking.ops.convert import (
    ConvertedDatasetIndexRow,
    converted_dataset_index,
)
from mosaic.tracking.ops.train import (
    TRAINING_PROVENANCE_FILENAME,
    LocalizerTrainParams,
    finalize_training,
    trained_model_index,
)
from tests.helpers import make_dataset

KIND = "train-localizer"
RUN_ID = "train-localizer.0.1-abcdef0123"


def _finalize(ds: Dataset, data_dir: Path) -> Path:
    run_root = model_run_root(ds, KIND, RUN_ID)
    (run_root / "train").mkdir(parents=True)
    weights = run_root / "train" / "best.pt"
    _ = weights.write_bytes(b"weights")
    finalize_training(
        ds,
        KIND,
        RUN_ID,
        run_root,
        LocalizerTrainParams(dataset_dir=str(data_dir)),
        "",
        "",
        "",
        weights,
        run_root / "train" / "results.csv",
        1,
        data_path=data_dir,
        data_fingerprint="f1ngerpr1nt",
    )
    return run_root


def test_a_finished_run_records_what_it_was_trained_on(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="prov")
    data_dir = ds.get_root("models") / "prepare-training-data" / "p-1"
    data_dir.mkdir(parents=True)

    run_root = _finalize(ds, data_dir)

    row = trained_model_index(model_index_path(ds, KIND)).read(run_id=RUN_ID).iloc[0]
    assert row["data_path"] == "models/prepare-training-data/p-1", "root-relative"
    assert row["data_fingerprint"] == "f1ngerpr1nt"

    document = json.loads((run_root / TRAINING_PROVENANCE_FILENAME).read_text())
    assert document["run_id"] == RUN_ID
    assert document["data"] == {
        "path": "models/prepare-training-data/p-1",
        "fingerprint": "f1ngerpr1nt",
    }
    assert document["params"]["dataset_dir"] == str(data_dir)


def test_data_outside_the_dataset_stays_absolute(tmp_path: Path) -> None:
    """External data is provenance of somewhere else, recorded as it was named."""
    ds = make_dataset(tmp_path / "ds", name="prov")
    outside = tmp_path / "elsewhere" / "yolo"
    outside.mkdir(parents=True)

    _ = _finalize(ds, outside)

    row = trained_model_index(model_index_path(ds, KIND)).read(run_id=RUN_ID).iloc[0]
    assert Path(str(row["data_path"])).is_absolute()


def test_the_data_path_is_visible_to_the_portability_passes(tmp_path: Path) -> None:
    """A path column the repair passes cannot see silently stops being portable."""
    ds = make_dataset(tmp_path / "ds", name="prov")
    _ = _finalize(ds, tmp_path / "elsewhere")

    models = [i for i in iter_dataset_indexes(ds) if i.root_key == "models"]
    from mosaic.core.dataset import _INDEX_PATH_COLUMNS  # pyright: ignore[reportPrivateUsage]

    assert models, "the models index is enumerated"
    assert "data_path" in _INDEX_PATH_COLUMNS["models"]


def test_an_index_written_before_the_columns_existed_is_adopted_on_write(
    tmp_path: Path,
) -> None:
    """Adopt on write, tolerate on read: a legacy index widens when it is next written.

    Reading leaves it as it is, so a read-only mount works. The next finished run
    publishes the wider schema and its own row in one write, and the row that was
    already there keeps its integer ``n_epochs`` rather than becoming ``3.0``.
    """
    ds = make_dataset(tmp_path / "ds", name="prov")
    index_path = model_index_path(ds, KIND)
    index_path.parent.mkdir(parents=True)
    legacy_run = "train-localizer.0.1-0000000000"
    legacy = pd.DataFrame(
        [
            {
                "run_id": legacy_run,
                "abs_path": f"models/{KIND}/{legacy_run}",
                "started_at": "",
                "finished_at": "",
                "kind": KIND,
                "base_model": "",
                "base_run_id": "",
                "best_model_path": f"models/{KIND}/{legacy_run}/best.pt",
                "metrics_path": "",
                "n_epochs": 3,
                "status": "finished",
            }
        ]
    )
    legacy.to_csv(index_path, index=False)
    assert "data_path" not in pd.read_csv(index_path).columns, (
        "reading rewrites nothing"
    )

    _ = _finalize(ds, tmp_path / "elsewhere")

    widened = pd.read_csv(index_path, dtype="string", keep_default_na=False)
    assert {"data_path", "data_fingerprint", "base_origin"} <= set(widened.columns)
    old_row = widened[widened["run_id"] == legacy_run].iloc[0]
    assert old_row["data_path"] == "", "unknown, not invented"
    assert old_row["n_epochs"] == "3", "an absent column must not widen the integers"


def _prepared(ds: Dataset, *, keep_artifact: bool) -> str:
    kind, run_id = "convert-points", "convert-points.0.2-aaaaaaaaaa"
    out = model_run_root(ds, kind, run_id)
    out.mkdir(parents=True)
    data_yaml = out / "data.yaml"
    if keep_artifact:
        _ = data_yaml.write_text("names: [bee]\n")
    index = converted_dataset_index(model_index_path(ds, kind))
    index.ensure()
    index.append(
        [
            ConvertedDatasetIndexRow(
                run_id=run_id,
                kind=kind,
                source_format="cvat_points_polo",
                data_yaml=ds.relative_to_root(data_yaml),
                class_names="bee",
                n_train=3,
                n_valid=1,
                n_test=0,
                status="finished",
                abs_path=Path(ds.relative_to_root(out)),
            )
        ]
    )
    index.mark_finished(run_id)
    return run_id


def test_prepared_data_is_not_reported_as_a_damaged_model(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="prov")
    run_id = _prepared(ds, keep_artifact=True)

    assert inventory(ds, kinds=["trained-model"]).records == ()

    found = inventory(ds, kinds=["prepared-dataset"])
    assert [record.run_id for record in found.records] == [run_id]
    assert found.records[0].status == "complete"
    assert found.records[0].ref.kind == "prepared-dataset"


def test_prepared_data_whose_tree_is_gone_is_not_complete(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds", name="prov")
    _ = _prepared(ds, keep_artifact=False)

    found = inventory(ds, kinds=["prepared-dataset"])

    assert found.records[0].status != "complete"
