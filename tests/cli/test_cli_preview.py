"""``mosaic preview``: one entry's table of a tracks, labels variant or a feature run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.inventory import TracksVariantRef
from mosaic.core.pipeline.labels_index import write_labels_row
from mosaic.core.pipeline.preview import ArtifactPreview, preview_artifact
from tests.helpers import add_tracks_variant, invoke_ok, make_dataset

VARIANT = "convert-trex.0.2-1111111111"
REF = f"tracks-variant/{VARIANT}"

runner = CliRunner()


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", std_format="mosaic_v1")
    return ds


def _preview(ds: Dataset, *arguments: str) -> list[str]:
    return ["preview", REF, "--manifest", str(ds.manifest_path), *arguments]


def test_json_is_the_library_preview(dataset: Dataset) -> None:
    result = invoke_ok(
        _preview(dataset, "--entry", "seq_a", "--column", "X", "--limit", "2", "--json")
    )

    printed = ArtifactPreview.model_validate_json(result.stdout)
    expected = preview_artifact(
        dataset, TracksVariantRef(VARIANT), ("", "seq_a"), columns=["X"], limit=2
    )
    assert printed == expected
    assert printed.read_columns == ("X",)
    assert len(printed.rows) == 2


def test_the_plain_output_lists_the_facts_statistics_and_rows(
    dataset: Dataset,
) -> None:
    result = invoke_ok(_preview(dataset, "--entry", ":seq_a", "--limit", "1"))

    assert f"ref: {REF}" in result.stdout
    assert "frames: 0-39" in result.stdout
    assert "column\ttype\tmin\tmax\tmean\tnan_fraction" in result.stdout
    assert "rows printed" in result.stderr


def test_an_unindexed_entry_reports_not_found(dataset: Dataset) -> None:
    result = runner.invoke(app, _preview(dataset, "--entry", "seq_z", "--json"))

    assert result.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(result.stdout)
    assert envelope.kind == "not_found"
    assert envelope.detail == {"ref": REF, "group": "", "sequence": "seq_z"}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--limit", "51"],
        ["--column", "nope"],
    ],
)
def test_an_invalid_argument_reports_invalid(
    dataset: Dataset, arguments: list[str]
) -> None:
    result = runner.invoke(
        app, _preview(dataset, "--entry", "seq_a", *arguments, "--json")
    )

    assert result.exit_code == 1
    assert ErrorEnvelope.model_validate_json(result.stdout).kind == "invalid"


def test_a_ref_of_another_kind_reports_invalid(dataset: Dataset) -> None:
    result = runner.invoke(
        app,
        [
            "preview",
            "trained-model/train-pose/train-pose.0.1-0123456789",
            "--manifest",
            str(dataset.manifest_path),
            "--entry",
            "seq_a",
            "--json",
        ],
    )

    assert result.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(result.stdout)
    assert (envelope.kind, envelope.detail) == ("invalid", {"field": "ref"})


def test_max_columns_and_patterns_reach_the_library(dataset: Dataset) -> None:
    result = invoke_ok(
        _preview(
            dataset,
            "--entry",
            "seq_a",
            "--column",
            "[XY]",
            "--column",
            "frame",
            "--max-columns",
            "2",
            "--listed",
            "read",
            "--json",
        )
    )

    printed = ArtifactPreview.model_validate_json(result.stdout)
    assert printed == preview_artifact(
        dataset,
        TracksVariantRef(VARIANT),
        ("", "seq_a"),
        columns=["[XY]", "frame"],
        max_columns=2,
        listed="read",
    )
    assert printed.read_columns == ("X", "Y")
    assert printed.columns_omitted == ("frame",)
    assert [column.name for column in printed.columns] == ["X", "Y"]


def test_a_labels_variant_prints_its_classes(dataset: Dataset) -> None:
    run_id = "boris.0.1-2222222222"
    out_path = dataset.get_root("labels") / "behavior" / run_id / "seq_a.npz"
    out_path.parent.mkdir(parents=True)
    np.savez_compressed(out_path, labels=np.array([0, 1, 1]))
    write_labels_row(
        dataset,
        run_id=run_id,
        group="",
        sequence="seq_a",
        out_path=out_path,
        producer="convert-labels-calms21",
        label_kind="behavior",
        n_frames=3,
        label_ids="0,1",
        label_names="other,attack",
    )

    result = invoke_ok(
        [
            "preview",
            f"labels-variant/behavior/{run_id}",
            "--manifest",
            str(dataset.manifest_path),
            "--entry",
            "seq_a",
        ]
    )

    assert "coverage: 1.0" in result.stdout
    assert "id\tname\tevents\tfraction" in result.stdout
    assert "1\tattack\t2\t" in result.stdout
