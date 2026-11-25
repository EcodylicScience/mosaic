"""The scoped delete set, and the two carve-outs it can never contain.

The carve-outs are asserted as properties of the *walk* rather than of a filter:
the walk does not enumerate frames or labels, so no branch here can forget them.
Asserting it at this layer makes that structural guarantee visible to whoever
changes the walk later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.delete_set import (
    delete_set,
)
from mosaic.core.pipeline.fit_scope import FitScope, write_fit_scope
from mosaic.core.pipeline.index import (
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.media_index import MediaIndexScope
from mosaic.core.pipeline.composition import SequenceComposition
from mosaic.core.pipeline.sequence_index import (
    read_sequence_index,
    write_sequence_compositions,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.types.results import Result
from mosaic.core.scope import Scope

from tests.helpers import (
    CONVERTED_ENTRIES,
    add_feature_run,
    CropLike,
    assert_tree_unchanged,
    convert_tracks_from_raw,
    converted_source,
    copy_of,
    tree_snapshot,
)


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


@pytest.mark.usefixtures("requires_ffmpeg")
class TestDryRunFirst:
    def test_a_preview_deletes_nothing(
        self, scenario_dataset_with_media: Dataset
    ) -> None:

        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, CropLike())
        output = ds.get_root("features") / "prov-crop__from__tracks" / result.run_id
        before = sorted(path.name for path in output.glob("*.parquet"))
        assert before, "the fixture produced no outputs"

        _reorder(ds)
        report = delete_set(ds, "media_raw", scope=Scope(entries=[("", "seq_a")]))

        assert not report.applied
        assert report.candidates, "the reorder reached nothing"
        assert sorted(path.name for path in output.glob("*.parquet")) == before

    def test_applying_removes_the_output_and_its_row(
        self, scenario_dataset_with_media: Dataset
    ) -> None:

        from mosaic.core.pipeline.index import feature_index, feature_index_path
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, CropLike())
        storage = "prov-crop__from__tracks"
        output = ds.get_root("features") / storage / result.run_id / "seq_a.parquet"
        assert output.exists()

        _reorder(ds)
        report = delete_set(
            ds, "media_raw", scope=Scope(entries=[("", "seq_a")]), apply=True
        )

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

        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, CropLike())
        storage = "prov-crop__from__tracks"
        untouched = ds.get_root("features") / storage / result.run_id / "seq_b.parquet"
        assert untouched.exists()

        _reorder(ds)
        _ = delete_set(
            ds, "media_raw", scope=Scope(entries=[("", "seq_a")]), apply=True
        )

        assert untouched.exists(), "a sequence nothing changed lost its output"


@pytest.mark.usefixtures("requires_ffmpeg")
class TestTheCarveOuts:
    """H4: neither can appear, and neither is filtered out to achieve that."""

    def test_no_frame_set_is_ever_a_candidate(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        ds = scenario_dataset_with_media
        _reorder(ds)
        report = delete_set(ds, "media_raw", scope=Scope(entries=[("", "seq_a")]))

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
        report = delete_set(ds, "media_raw", scope=Scope(entries=[("", "seq_a")]))

        assert not any(c.kind == "labels" for c in report.candidates)
        assert not any("labels" in c.abs_path for c in report.candidates)


def _record_fit_over_both_entries(ds: Dataset, result: Result[str]) -> None:
    """Record that the run was fitted over both entries of the scenario."""
    write_fit_scope(
        feature_run_root(ds, result.feature, str(result.run_id)),
        FitScope(
            scope_dependent=True,
            entries=(("", "seq_a"), ("", "seq_b")),
            tracks_variants=(),
            labels_variants=(),
            fitted_at="",
            identity_scheme="",
        ),
    )


@pytest.mark.usefixtures("requires_ffmpeg")
class TestDeclines:
    def test_a_partly_reached_scope_dependent_run_is_declined(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """Whole or not at all.

        Deleting one entry of a fit leaves the rest describing a fit that
        included it, with nothing on disk saying so.
        """

        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, CropLike())
        _reorder(ds)
        _record_fit_over_both_entries(ds, result)

        report = delete_set(
            ds, "media_raw", scope=Scope(entries=[("", "seq_a")]), apply=True
        )

        assert not report.applied
        assert any("scope-dependent" in d.reason for d in report.declined)

    def test_declines_are_reported_rather_than_omitted(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """ "Would delete 0" must not read as "nothing was affected"."""

        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, CropLike())
        _reorder(ds)
        _record_fit_over_both_entries(ds, result)

        report = delete_set(ds, "media_raw", scope=Scope(entries=[("", "seq_a")]))

        assert not report.candidates
        assert report.declined
        assert report.considered > 0, (
            "a refusal was reported as though nothing had been reached"
        )


@pytest.mark.usefixtures("requires_ffmpeg")
class TestTheSafeguard:
    def test_a_path_outside_the_roots_raises(
        self, scenario_dataset_with_media: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ported from ``Pipeline.clean``: it raises rather than skipping.

        A candidate pointing outside is evidence the set was computed wrongly, so
        the rest of it cannot be trusted either.
        """

        from mosaic.core.pipeline import delete_set as delete_set_mod
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, CropLike())
        _reorder(ds)

        real = ds.resolve_path

        def escape(stored: str) -> object:
            from pathlib import Path

            if stored.endswith("seq_a.parquet"):
                return Path("/tmp/somewhere-else.parquet")
            return real(stored)

        monkeypatch.setattr(ds, "resolve_path", escape)
        with pytest.raises(RuntimeError, match="outside this dataset"):
            _ = delete_set_mod.delete_set(
                ds, "media_raw", scope=Scope(entries=[("", "seq_a")]), apply=True
            )


@pytest.mark.usefixtures("requires_ffmpeg")
class TestUnknownIsNeverDeleted:
    """Fail closed means decline, not delete.

    An absent record is not evidence of change. Deleting on it would be deleting
    on a guess -- and unlike the refusal in ``run_feature``, where the cost of
    being wrong is CPU, here it is the file.
    """

    def test_an_unknown_verdict_is_declined_and_the_output_survives(
        self, scenario_dataset_with_media: Dataset
    ) -> None:

        from mosaic.core.pipeline.index import feature_index, feature_index_path
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        result = run_feature(ds, CropLike())
        storage = "prov-crop__from__tracks"
        output = ds.get_root("features") / storage / result.run_id / "seq_a.parquet"

        # Blank the recorded side while the declaration stays, the shape a row
        # written before rows recorded their consumed composition carries.
        index = feature_index(feature_index_path(ds, storage))
        frame = index.read(validate_paths=False)
        frame["consumed_composition"] = ""
        frame.to_csv(index.path, index=False)

        _reorder(ds)
        report = delete_set(
            ds, "media_raw", scope=Scope(entries=[("", "seq_a")]), apply=True
        )

        assert not any(c.kind == "features" for c in report.candidates), (
            "an output whose provenance is unrecorded was deleted on a guess"
        )
        assert any(d.kind == "features" for d in report.declined), (
            "the refusal was silent rather than reported"
        )
        assert output.exists(), "the file went"


def _drift_tracks_raw(ds: Dataset) -> None:
    """Record another composition for every sequence of ``tracks_raw`` in *ds*."""
    frame = read_sequence_index(ds, "tracks_raw")
    _ = write_sequence_compositions(
        ds,
        "tracks_raw",
        compositions={
            (str(row["group"]), str(row["sequence"])): SequenceComposition(
                digest="moved", member_count=int(row["member_count"])
            )
            for _, row in frame.iterrows()
        },
    )


@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "applied"])
def test_a_linked_tracks_row_is_declined_with_its_link_and_the_source_is_left(
    tmp_path: Path, apply: bool
) -> None:
    source = converted_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")
    convert_tracks_from_raw(copy, "conv.0.1-aaaaaaaaaa", "own")
    _drift_tracks_raw(copy)
    link_id = f"source-{source.uuid}"
    before = tree_snapshot(source.base_dir)
    scope = Scope(entries=[*CONVERTED_ENTRIES, ("", "own")])

    report = delete_set(copy, "tracks_raw", scope=scope, apply=apply)

    assert [candidate.sequence for candidate in report.candidates] == ["own"]
    declined = [item for item in report.declined if item.kind == "tracks"]
    assert len(declined) == len(CONVERTED_ENTRIES)
    assert all(link_id in item.reason for item in declined)
    assert report.applied is apply
    own_rows = read_tracks_index(copy)
    assert list(own_rows["sequence"]) == ([] if apply else ["own"])
    assert_tree_unchanged(source.base_dir, before)


def test_a_partial_delete_of_a_run_fitted_in_the_source_is_declined(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    source_root = add_feature_run(source, "demo", "0.1-0123456789")
    write_fit_scope(
        source_root,
        FitScope(
            scope_dependent=True,
            entries=(*CONVERTED_ENTRIES, ("", "own")),
            tracks_variants=(),
            labels_variants=(),
            fitted_at="",
            identity_scheme="",
        ),
    )
    copy = copy_of(source, tmp_path / "copy")
    convert_tracks_from_raw(copy, "conv.0.1-aaaaaaaaaa", "own")
    _ = add_feature_run(copy, "demo", "0.1-0123456789", entries=[("", "own")])
    index = feature_index(feature_index_path(copy, "demo"))
    rows = index.read(validate_paths=False)
    rows["consumed_roots"] = "tracks_raw"
    rows["consumed_composition"] = "recorded"
    rows.to_csv(index.path, index=False)
    _drift_tracks_raw(copy)
    run_output = feature_run_root(copy, "demo", "0.1-0123456789") / "own.parquet"

    report = delete_set(
        copy, "tracks_raw", scope=Scope(entries=[("", "own")]), apply=True
    )

    assert not any(candidate.kind == "features" for candidate in report.candidates)
    assert [
        item.reason
        for item in report.declined
        if item.kind == "features" and "scope-dependent" in item.reason
    ]
    assert run_output.exists()


def _record_consumption(ds: Dataset, storage: str) -> None:
    """Record that the rows of feature *storage* read ``tracks_raw``."""
    index = feature_index(feature_index_path(ds, storage))
    rows = index.read(validate_paths=False)
    rows["consumed_roots"] = "tracks_raw"
    rows["consumed_composition"] = "recorded"
    rows.to_csv(index.path, index=False)


@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "applied"])
def test_a_linked_feature_row_is_declined_with_its_link_and_the_source_is_left(
    tmp_path: Path, apply: bool
) -> None:
    source = converted_source(tmp_path / "source")
    _ = add_feature_run(source, "demo", "0.1-0123456789", entries=CONVERTED_ENTRIES)
    _record_consumption(source, "demo")
    copy = copy_of(source, tmp_path / "copy")
    convert_tracks_from_raw(copy, "conv.0.1-aaaaaaaaaa", "own")
    _ = add_feature_run(copy, "demo", "0.1-0123456789", entries=[("", "own")])
    _record_consumption(copy, "demo")
    _drift_tracks_raw(copy)
    link_id = f"source-{source.uuid}"
    own_output = feature_run_root(copy, "demo", "0.1-0123456789") / "own.parquet"
    before = tree_snapshot(source.base_dir)
    scope = Scope(entries=[*CONVERTED_ENTRIES, ("", "own")])

    report = delete_set(copy, "tracks_raw", scope=scope, apply=apply)

    assert [
        candidate.sequence
        for candidate in report.candidates
        if candidate.kind == "features"
    ] == ["own"]
    declined = [item for item in report.declined if item.kind == "features"]
    assert len(declined) == len(CONVERTED_ENTRIES)
    assert all(link_id in item.reason for item in declined)
    assert own_output.exists() is not apply
    assert_tree_unchanged(source.base_dir, before)
