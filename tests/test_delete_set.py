"""Item 6.4: the scoped delete set, and the two carve-outs it can never contain.

The carve-outs are asserted as properties of the *walk* rather than of a filter:
item 6.1 does not enumerate frames or labels, so no branch here can forget them.
Asserting it at this layer is what makes that structural guarantee visible to
whoever changes the walk later.
"""

from __future__ import annotations

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.delete_set import delete_set
from mosaic.core.pipeline.media_index import MediaIndexScope


def _reorder(ds: Dataset, sequence: str = "seq_a") -> None:
    _ = ds.write_media_index(
        [
            MediaIndexScope(
                directory=ds.get_root("media_raw") / sequence,
                group="",
                sequence=sequence,
                order_by_name={"b.mp4": 0, "a.mp4": 1},
            )
        ],
        extensions=(".mp4",),
    )


@pytest.mark.usefixtures("requires_ffprobe")
class TestDryRunFirst:
    def test_a_preview_deletes_nothing(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from tests.test_provenance import _CropLike

        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, _CropLike())
        output = ds.get_root("features") / "prov-crop__from__tracks" / result.run_id
        before = sorted(path.name for path in output.glob("*.parquet"))
        assert before, "the fixture produced no outputs"

        _reorder(ds)
        report = delete_set(ds, [("", "seq_a")], "media_raw")

        assert not report.applied
        assert report.candidates, "the reorder reached nothing"
        assert sorted(path.name for path in output.glob("*.parquet")) == before

    def test_applying_removes_the_output_and_its_row(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from tests.test_provenance import _CropLike

        from mosaic.core.pipeline.index import feature_index, feature_index_path
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, _CropLike())
        storage = "prov-crop__from__tracks"
        output = ds.get_root("features") / storage / result.run_id / "seq_a.parquet"
        assert output.exists()

        _reorder(ds)
        report = delete_set(ds, [("", "seq_a")], "media_raw", apply=True)

        assert report.applied
        assert not output.exists(), "the reached output survived"
        rows = feature_index(feature_index_path(ds, storage)).read(validate_paths=False)
        remaining = {
            (str(group), str(sequence))
            for group, sequence in zip(rows["group"], rows["sequence"], strict=True)
        }
        assert ("", "seq_a") not in remaining, "the row that named it survived"

    def test_an_unreached_sequence_keeps_its_output(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """H3 case 2's neighbour: scoping is what keeps this honest."""
        from tests.test_provenance import _CropLike

        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, _CropLike())
        storage = "prov-crop__from__tracks"
        untouched = ds.get_root("features") / storage / result.run_id / "seq_b.parquet"
        assert untouched.exists()

        _reorder(ds)
        _ = delete_set(ds, [("", "seq_a")], "media_raw", apply=True)

        assert untouched.exists(), "a sequence nothing changed lost its output"


@pytest.mark.usefixtures("requires_ffprobe")
class TestTheCarveOuts:
    """H4: neither can appear, and neither is filtered out to achieve that."""

    def test_no_frame_set_is_ever_a_candidate(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        ds = scenario_dataset_with_media
        _reorder(ds)
        report = delete_set(ds, [("", "seq_a")], "media_raw")

        assert not any(c.kind == "frames" for c in report.candidates)
        assert not any(d.kind == "frames" for d in report.declined)

    def test_no_label_file_is_ever_a_candidate(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        ds = scenario_dataset_with_media
        index_path = ds.get_root("labels") / "behavior" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        _ = index_path.write_text(
            "kind,label_format,group,sequence,abs_path\n"
            "behavior,individual_pair_v1,,seq_a,labels/behavior/seq_a.npz\n"
        )
        _reorder(ds)
        report = delete_set(ds, [("", "seq_a")], "media_raw")

        assert not any(c.kind == "labels" for c in report.candidates)
        assert not any("labels" in c.abs_path for c in report.candidates)


@pytest.mark.usefixtures("requires_ffprobe")
class TestDeclines:
    def test_a_partly_reached_scope_dependent_run_is_declined(
        self, scenario_dataset_with_media: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whole or not at all.

        Deleting one entry of a fit leaves the rest describing a fit that
        included it, with nothing on disk saying so.
        """
        from tests.test_provenance import _CropLike

        from mosaic.core.pipeline import delete_set as delete_set_mod
        from mosaic.core.pipeline.fit_scope import FitScope
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, _CropLike())
        _reorder(ds)

        def wider(_run_root: object) -> FitScope:
            return FitScope(
                scope_dependent=True,
                entries=(("", "seq_a"), ("", "seq_b")),
                tracks_variants=(),
                labels_variants=(),
                fitted_at="",
                identity_scheme="",
            )

        monkeypatch.setattr(delete_set_mod, "read_fit_scope", wider)
        report = delete_set(ds, [("", "seq_a")], "media_raw", apply=True)

        assert not report.applied
        assert any("scope-dependent" in d.reason for d in report.declined)

    def test_declines_are_reported_rather_than_omitted(
        self, scenario_dataset_with_media: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "Would delete 0" must not read as "nothing was affected"."""
        from tests.test_provenance import _CropLike

        from mosaic.core.pipeline import delete_set as delete_set_mod
        from mosaic.core.pipeline.fit_scope import FitScope
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, _CropLike())
        _reorder(ds)
        monkeypatch.setattr(
            delete_set_mod,
            "read_fit_scope",
            lambda _root: FitScope(
                scope_dependent=True,
                entries=(("", "seq_a"), ("", "seq_b")),
                tracks_variants=(),
                labels_variants=(),
                fitted_at="",
                identity_scheme="",
            ),
        )

        report = delete_set(ds, [("", "seq_a")], "media_raw")

        assert not report.candidates
        assert report.declined
        assert report.considered > 0, (
            "a refusal was reported as though nothing had been reached"
        )


@pytest.mark.usefixtures("requires_ffprobe")
class TestTheSafeguard:
    def test_a_path_outside_the_roots_raises(
        self, scenario_dataset_with_media: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ported from ``Pipeline.clean``: it raises rather than skipping.

        A candidate pointing outside is evidence the set was computed wrongly, so
        the rest of it cannot be trusted either.
        """
        from tests.test_provenance import _CropLike

        from mosaic.core.pipeline import delete_set as delete_set_mod
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, _CropLike())
        _reorder(ds)

        real = ds.resolve_path

        def escape(stored: str) -> object:
            from pathlib import Path

            if stored.endswith("seq_a.parquet"):
                return Path("/tmp/somewhere-else.parquet")
            return real(stored)

        monkeypatch.setattr(ds, "resolve_path", escape)
        with pytest.raises(RuntimeError, match="outside this dataset"):
            _ = delete_set_mod.delete_set(ds, [("", "seq_a")], "media_raw", apply=True)


@pytest.mark.usefixtures("requires_ffprobe")
class TestUnknownIsNeverDeleted:
    """Fail closed means decline, not delete.

    An absent record is not evidence of change. Deleting on it would be deleting
    on a guess -- and unlike the refusal in ``run_feature``, where the cost of
    being wrong is CPU, here it is the file.
    """

    def test_an_unknown_verdict_is_declined_and_the_output_survives(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from tests.test_provenance import _CropLike

        from mosaic.core.pipeline.index import feature_index, feature_index_path
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, _CropLike())
        storage = "prov-crop__from__tracks"
        output = ds.get_root("features") / storage / result.run_id / "seq_a.parquet"

        # Blank the recorded side while the declaration stays, the shape a row
        # written before item 5.1 carries.
        index = feature_index(feature_index_path(ds, storage))
        frame = index.read(validate_paths=False)
        frame["consumed_composition"] = ""
        frame.to_csv(index.path, index=False)

        _reorder(ds)
        report = delete_set(ds, [("", "seq_a")], "media_raw", apply=True)

        assert not any(c.kind == "features" for c in report.candidates), (
            "an output whose provenance is unrecorded was deleted on a guess"
        )
        assert any(d.kind == "features" for d in report.declined), (
            "the refusal was silent rather than reported"
        )
        assert output.exists(), "the file went"
