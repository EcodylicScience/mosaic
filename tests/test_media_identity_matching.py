from pathlib import Path

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import (
    media_row_path_key,
    media_row_uuid,
    read_link_cell,
)


def test_read_link_cell_treats_every_absent_form_as_empty() -> None:
    assert read_link_cell({"c": ""}, "c") == ""
    assert read_link_cell({"c": "nan"}, "c") == ""
    assert read_link_cell({"c": "NaN"}, "c") == ""
    assert read_link_cell({"c": float("nan")}, "c") == ""
    assert read_link_cell({}, "c") == ""
    assert read_link_cell({"c": "  a.mp4  "}, "c") == "a.mp4"


def test_media_row_uuid_reads_the_column_or_empty() -> None:
    assert media_row_uuid({"video_uuid": "abc"}) == "abc"
    assert media_row_uuid({"video_uuid": ""}) == ""
    assert media_row_uuid({"video_uuid": float("nan")}) == ""
    assert media_row_uuid({}) == ""


def test_media_row_path_key_is_the_two_leaf_components() -> None:
    assert media_row_path_key({"abs_path": "/data/seqA/clip.mp4"}) == (
        "seqA",
        "clip.mp4",
    )
    assert media_row_path_key({"abs_path": "seqA/clip.mp4"}) == ("seqA", "clip.mp4")


def _dataset_with_media(tmp_path: Path) -> Dataset:
    for sub in ("media_raw", "media"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
        },
    )


def test_carry_forward_matches_a_renamed_original_by_uuid(tmp_path: Path) -> None:
    import csv

    dataset = _dataset_with_media(tmp_path)
    # A real derivative file so the .exists() dangling check passes.
    derivative = tmp_path / "media" / "clip.analysis.mp4"
    _ = derivative.write_bytes(b"x")

    # Prior index: original at old_name.mp4, minted uuid U, with an analysis link.
    prior_path = tmp_path / "prior_index.csv"
    with prior_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["abs_path", "video_uuid", "analysis_derivative_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "abs_path": "old_name.mp4",
                "video_uuid": "uuid-U",
                "analysis_derivative_path": "clip.analysis.mp4",
            }
        )

    # Fresh rows: the SAME video (uuid-U) now at a renamed path.
    fresh: list[dict[str, object]] = [
        {"abs_path": "renamed.mp4", "video_uuid": "uuid-U"}
    ]
    dataset._carry_forward_derivative_links(fresh, prior_path)

    assert fresh[0]["analysis_derivative_path"] == "clip.analysis.mp4"


def test_carry_forward_falls_back_to_path_for_an_unminted_pair(
    tmp_path: Path,
) -> None:
    import csv

    dataset = _dataset_with_media(tmp_path)
    derivative = tmp_path / "media" / "clip.analysis.mp4"
    _ = derivative.write_bytes(b"x")

    prior_path = tmp_path / "prior_index.csv"
    with prior_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["abs_path", "video_uuid", "analysis_derivative_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "abs_path": "seqA/clip.mp4",
                "video_uuid": "",
                "analysis_derivative_path": "clip.analysis.mp4",
            }
        )

    # Unminted fresh row at the SAME two-leaf path -> matches by path fallback.
    fresh: list[dict[str, object]] = [{"abs_path": "seqA/clip.mp4", "video_uuid": ""}]
    dataset._carry_forward_derivative_links(fresh, prior_path)

    assert fresh[0]["analysis_derivative_path"] == "clip.analysis.mp4"
