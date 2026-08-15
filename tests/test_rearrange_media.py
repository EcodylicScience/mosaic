"""Item 6.3: the rearrangement gesture, and what it refuses.

The gesture is preview-first, so most of what is asserted here is what a caller
learns *before* anything moves. The two blocks are the cases where proceeding
silently destroys something no recipe can rebuild, and the one hard refusal is
the case no force flag should be able to ask for by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.media.rearrange import Arrangement, rearrange_media


def _reverse(ds: Dataset, sequence: str = "seq_a") -> Arrangement:
    return Arrangement(
        group="", sequence=sequence, order_by_name={"b.mp4": 0, "a.mp4": 1}
    )


def _order(ds: Dataset, sequence: str = "seq_a") -> list[str]:
    """The committed clip order, by basename."""
    rows = [
        row for row in ds.read_media_index() if str(row.get("sequence", "")) == sequence
    ]
    rows.sort(key=lambda row: int(float(str(row.get("video_order", "0")) or 0)))
    return [Path(str(row["abs_path"])).name for row in rows]


@pytest.mark.usefixtures("requires_ffmpeg")
class TestPreviewIsTheDefault:
    def test_a_preview_writes_nothing(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        ds = scenario_dataset_with_media
        before = _order(ds)

        report = rearrange_media(ds, [_reverse(ds)])

        assert not report.applied
        assert _order(ds) == before, "a preview committed the arrangement"

    def test_applying_commits_the_order(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        ds = scenario_dataset_with_media
        before = _order(ds)

        report = rearrange_media(ds, [_reverse(ds)], apply=True)

        assert report.applied
        assert _order(ds) == list(reversed(before))

    def test_an_empty_request_is_a_no_op_with_the_full_schema(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from mosaic.core.pipeline.provenance import PROVENANCE_COLUMNS

        report = rearrange_media(scenario_dataset_with_media, [], apply=True)

        assert not report.applied
        assert list(report.reached.columns) == PROVENANCE_COLUMNS


@pytest.mark.usefixtures("requires_ffmpeg")
class TestThePreviewCarriesTheBlastRadius:
    def test_a_reached_feature_is_named_before_anything_moves(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """The preview is item 6.1's walk, run against the sequences about to move."""
        from tests.test_provenance import CropLike

        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, CropLike())

        report = rearrange_media(ds, [_reverse(ds)])

        reached = report.reached
        assert "prov-crop__from__tracks" in set(reached["name"]), (
            "the preview did not name the feature the reorder reaches"
        )
        assert not report.applied


@pytest.mark.usefixtures("requires_ffmpeg")
class TestBlocks:
    def _add_labels(self, ds: Dataset, sequence: str = "seq_a") -> Path:
        index_path = ds.get_root("labels") / "behavior" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        _ = index_path.write_text(
            "kind,label_format,group,sequence,abs_path\n"
            f"behavior,individual_pair_v1,,{sequence},labels/behavior/{sequence}.npz\n"
        )
        return index_path

    def test_converted_labels_block_a_reorder(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """Their frame indices are sequence-global; the remap is item 9.3's."""
        ds = scenario_dataset_with_media
        _ = self._add_labels(ds)
        before = _order(ds)

        report = rearrange_media(ds, [_reverse(ds)], apply=True)

        assert not report.applied
        assert not report.would_proceed
        assert any("converted labels" in reason for reason in report.blocked)
        assert _order(ds) == before, "a blocked reorder was committed anyway"

    def test_force_overrides_and_the_report_still_says_what_it_ran_over(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """A forced run is not a silent one."""
        ds = scenario_dataset_with_media
        _ = self._add_labels(ds)
        before = _order(ds)

        report = rearrange_media(ds, [_reverse(ds)], apply=True, force=True)

        assert report.applied and report.forced
        assert report.blocked, "a forced run forgot what it overrode"
        assert _order(ds) == list(reversed(before))

    def test_a_sequence_without_labels_is_not_blocked(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        ds = scenario_dataset_with_media
        _ = self._add_labels(ds, sequence="seq_b")

        report = rearrange_media(ds, [_reverse(ds)])

        assert report.would_proceed, (
            "another sequence's labels blocked this one's reorder"
        )


@pytest.mark.usefixtures("requires_ffmpeg")
class TestTheScopeRefusal:
    """Not a block: no force flag should be able to merge two sequences."""

    def test_a_shared_directory_refuses(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        ds = scenario_dataset_with_media
        # Point seq_b's row at seq_a's directory, the shape a hand-edited index
        # or a flat legacy layout produces.
        from mosaic.core.pipeline.media_index import (
            frame_from_rows,
            load_media_index_frame,
            write_media_index_rows,
        )

        index_path = ds.get_root("media_raw") / "index.csv"
        frame = load_media_index_frame(index_path)
        rows = frame.to_dict(orient="records")
        intruder = dict(rows[0])
        intruder["sequence"] = "seq_b"
        intruder["abs_path"] = str(rows[0]["abs_path"])
        intruder["name"] = "intruder.mp4"
        write_media_index_rows(index_path, frame_from_rows([*rows, intruder]))

        with pytest.raises(ValueError, match="also holds media for"):
            _ = rearrange_media(ds, [_reverse(ds)])

    def test_an_unknown_sequence_refuses(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        with pytest.raises(ValueError, match="no media rows"):
            _ = rearrange_media(
                scenario_dataset_with_media,
                [Arrangement(group="", sequence="nope", order_by_name={})],
            )


@pytest.mark.usefixtures("requires_ffmpeg")
class TestTheReadabilityRule:
    """Block a regression, never a sequence that was already broken.

    Driven through the verdicts rather than through crafted media: what is under
    test is the asymmetry, and the comparison itself is asserted against real
    frame rates in the uniformity suite.
    """

    def _verdicts(self, readable: bool) -> dict[str, object]:
        from mosaic_media import PropertyMismatch

        from mosaic.core.media.uniformity import UniformityVerdict

        mismatch = (
            None if readable else PropertyMismatch(field="fps", first=29.0, other=28.0)
        )
        return {"": UniformityVerdict(mismatch=mismatch, unmeasured=())}

    def _patch(
        self,
        ds: Dataset,
        monkeypatch: pytest.MonkeyPatch,
        *,
        before: bool,
        after: bool,
    ) -> None:
        def fake(
            group: str,
            sequence: str,
            *,
            order_by_name: object = None,
            index_filename: str = "index.csv",
        ) -> dict[str, object]:
            readable = before if order_by_name is None else after
            return {} if readable else self._verdicts(readable=False)

        monkeypatch.setattr(ds, "sequence_uniformity", fake)

    def test_a_regression_blocks(
        self, scenario_dataset_with_media: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = scenario_dataset_with_media
        self._patch(ds, monkeypatch, before=True, after=False)

        report = rearrange_media(ds, [_reverse(ds)], apply=True)

        assert not report.applied
        assert any("would not after this order" in reason for reason in report.blocked)

    def test_an_already_unreadable_sequence_does_not_block(
        self, scenario_dataset_with_media: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refusing here would strand it -- reordering may be how it gets fixed."""
        ds = scenario_dataset_with_media
        self._patch(ds, monkeypatch, before=False, after=False)

        report = rearrange_media(ds, [_reverse(ds)], apply=True)

        assert report.applied, "a sequence that was already broken was stranded"
        assert not report.blocked

    def test_a_repair_does_not_block(
        self, scenario_dataset_with_media: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = scenario_dataset_with_media
        self._patch(ds, monkeypatch, before=False, after=True)

        report = rearrange_media(ds, [_reverse(ds)], apply=True)

        assert report.applied and not report.blocked
