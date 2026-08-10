"""Item 6.1's walk: which derived artifacts a source change reaches.

Every column the walk reads was recorded in Stage 5 and is asserted elsewhere.
What is asserted here is the *join* -- and above all its two edges: that a change
under one root does not reach what consumed another (H3 case 2's whole point),
and that it does reach a feature run through the tracks table it read, which no
recorded cell on that run can say.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.provenance import PROVENANCE_COLUMNS, reached_by
from mosaic.core.pipeline.types import DependencyLookup, Inputs, InputStream, Params


class _P(Params):
    pass


class CropLike:
    """A per-frame feature that opens video: it declares ``media_raw``.

    Plainly named, along with :class:`PlainFeature`, because five test modules
    build their scenarios on these two -- a leading underscore on a symbol a
    sibling imports is the thing ``reportPrivateUsage`` names.

    The four protocol methods carry the protocol's own parameter names and types
    rather than ``object`` stand-ins. A structural protocol matches on parameter
    *names*, so a stub spelling them differently is not a ``Feature``, and every
    module passing one to ``run_feature`` inherited that error.
    """

    name = "prov-crop"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ("media_raw",)

    def __init__(
        self,
        inputs: Inputs | None = None,
        params: dict[str, object] | _P | None = None,
    ) -> None:
        self.inputs = inputs if inputs is not None else Inputs(("tracks",))
        self.params = params if isinstance(params, _P) else _P.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class PlainFeature(CropLike):
    """The ordinary shape: forty of forty-two features declare no source root."""

    name = "prov-plain"
    consumed_roots: tuple[str, ...] = ()


def _for(frame, kind: str, name: str):
    return frame[(frame["kind"] == kind) & (frame["name"] == name)]


class TestTheSchema:
    def test_nothing_changed_returns_the_full_schema(
        self, scenario_dataset: Dataset
    ) -> None:
        """An empty answer a caller can filter without a ``KeyError``."""
        empty = reached_by(scenario_dataset, [], "media_raw")
        assert list(empty.columns) == PROVENANCE_COLUMNS
        assert empty.empty

    def test_an_unreached_sequence_returns_the_full_schema(
        self, scenario_dataset: Dataset
    ) -> None:
        reached = reached_by(scenario_dataset, [("", "no_such_seq")], "media_raw")
        assert list(reached.columns) == PROVENANCE_COLUMNS
        assert reached.empty


@pytest.mark.usefixtures("requires_ffprobe")
class TestScopingByRoot:
    """H3 case 2, as an assertion about the walk rather than about a delete set."""

    def test_a_media_change_does_not_reach_a_tracks_only_consumer(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """The case that ruled out a single per-sequence hash.

        ``seq_a`` has media and tracks; the plain feature consumed neither root
        by declaration and reaches media through nothing, so a media change must
        not name it. A walk that returned it would throw away a feature chain
        built entirely from uploaded tracks.
        """
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, PlainFeature())

        reached = reached_by(ds, [("", "seq_a")], "media_raw")

        assert "prov-plain__from__tracks" not in set(reached["name"]), (
            "a feature that declared no root was reached by a media change"
        )

    def test_a_tracks_raw_change_does_not_reach_a_media_consumer(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """The mirror: the crop declares ``media_raw``, so ``tracks_raw`` misses it."""
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, CropLike())

        reached = reached_by(ds, [("", "seq_a")], "tracks_raw")
        crop = _for(reached, "features", "prov-crop__from__tracks")

        assert crop.empty, "a media consumer was reached by a tracks_raw change"

    def test_a_media_change_reaches_the_declared_consumer(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, CropLike())

        reached = reached_by(ds, [("", "seq_a")], "media_raw")
        crop = _for(reached, "features", "prov-crop__from__tracks")

        assert len(crop) == 1
        assert list(crop["via"]) == ["direct"]
        assert list(crop["group"]) == [""]
        assert list(crop["sequence"]) == ["seq_a"]


@pytest.mark.usefixtures("requires_ffprobe")
class TestVerdicts:
    """Before the change every row reads ``current``; after, what moved reads
    ``drifted``. One function, two moments -- which is what lets the same call
    serve a preview and an audit."""

    def test_before_a_change_the_membership_is_the_answer(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, CropLike())

        reached = reached_by(ds, [("", "seq_a")], "media_raw")
        crop = _for(reached, "features", "prov-crop__from__tracks")

        assert list(crop["verdict"]) == ["current"], (
            "nothing has moved yet, so the walk reports what would be reached"
        )

    def test_after_a_reorder_the_reached_row_reads_drifted(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from mosaic.core.pipeline.media_index import MediaIndexScope
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = run_feature(ds, CropLike())

        _ = ds.write_media_index(
            [
                MediaIndexScope(
                    directory=ds.get_root("media_raw") / "seq_a",
                    group="",
                    sequence="seq_a",
                    order_by_name={"b.mp4": 0, "a.mp4": 1},
                )
            ],
            extensions=(".mp4",),
        )

        reached = reached_by(ds, [("", "seq_a")], "media_raw")
        crop = _for(reached, "features", "prov-crop__from__tracks")

        assert list(crop["verdict"]) == ["drifted"]
        assert crop.iloc[0]["recorded"] != crop.iloc[0]["current"]
        assert crop.iloc[0]["recorded"] and crop.iloc[0]["current"]


class TestTheTracksArm:
    def test_a_tracks_raw_change_reaches_the_converted_table(
        self, scenario_dataset: Dataset
    ) -> None:
        """The tracks row records which root it read, and this is what asks it."""
        from tests.conftest import add_tracks_variant

        ds = scenario_dataset
        add_tracks_variant(ds, "convert-dlc.0.1-aaaaaaaaaa", "seq_a")

        reached = reached_by(ds, [("", "seq_a")], "tracks_raw")
        tracks = reached[reached["kind"] == "tracks"]

        assert "convert-dlc.0.1-aaaaaaaaaa" in set(tracks["run_id"])

    def test_every_variant_of_an_entry_is_reached(
        self, scenario_dataset: Dataset
    ) -> None:
        """Not the one a selector would resolve to.

        A change under ``tracks_raw`` reaches every table converted from it;
        which one a feature *would* read today is a different question.
        """
        from tests.conftest import add_tracks_variant

        ds = scenario_dataset
        add_tracks_variant(ds, "convert-dlc.0.1-aaaaaaaaaa", "seq_a")
        add_tracks_variant(ds, "convert-dlc.0.2-bbbbbbbbbb", "seq_a")

        reached = reached_by(ds, [("", "seq_a")], "tracks_raw")
        tracks = reached[reached["kind"] == "tracks"]

        assert set(tracks["run_id"]) == {
            "convert-dlc.0.1-aaaaaaaaaa",
            "convert-dlc.0.2-bbbbbbbbbb",
        }


@pytest.mark.usefixtures("requires_ffprobe")
class TestTheTransitiveArm:
    """The arm no recorded cell can supply.

    Forty of forty-two features declare no source root, so their own
    ``consumed_composition`` is empty and the direct arm cannot see them. They
    reach media only through the tracks table they read, and the edge lives in
    the run's ``params.json`` -- the readable copy of a term already in the
    identifier. A walk without this stops at the tracks table and says nothing
    about the chain above it.

    Driven through a tracks variant recording ``media_raw``, which is what the
    TREx bridge writes: it reads the video and its own NPZ, and the derived half
    filters out, leaving ``media_raw`` alone on the row.
    """

    def _tracked_dataset(self, ds: Dataset) -> str:
        from tests.conftest import add_tracks_variant

        variant = "trex.0.1-aaaaaaaaaa"
        add_tracks_variant(ds, variant, "seq_a", consumed_source_roots=("media_raw",))
        return variant

    def _reorder(self, ds: Dataset) -> None:
        from mosaic.core.pipeline.media_index import MediaIndexScope

        _ = ds.write_media_index(
            [
                MediaIndexScope(
                    directory=ds.get_root("media_raw") / "seq_a",
                    group="",
                    sequence="seq_a",
                    order_by_name={"b.mp4": 0, "a.mp4": 1},
                )
            ],
            extensions=(".mp4",),
        )

    def test_a_feature_is_reached_through_the_tracks_it_read(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        variant = self._tracked_dataset(ds)
        _ = run_feature(ds, PlainFeature())

        self._reorder(ds)
        reached = reached_by(ds, [("", "seq_a")], "media_raw")

        tracks = reached[reached["kind"] == "tracks"]
        assert list(tracks["run_id"]) == [variant]
        assert list(tracks["verdict"]) == ["drifted"], (
            "the tracks table's recorded media composition did not move"
        )

        plain = _for(reached, "features", "prov-plain__from__tracks")
        assert len(plain) == 1, (
            "the feature built on a drifted tracks table was not reached"
        )
        assert list(plain["via"]) == ["tracks"]

    def test_an_unmoved_variant_propagates_nothing(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """Only a variant that actually moved is an edge to follow.

        Without the reorder the tracks row still matches, so the feature above it
        is reached by nothing -- the walk must not treat "consumed a variant" as
        "reached by a change".
        """
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = self._tracked_dataset(ds)
        _ = run_feature(ds, PlainFeature())

        reached = reached_by(ds, [("", "seq_a")], "media_raw")

        assert _for(reached, "features", "prov-plain__from__tracks").empty
        assert list(reached[reached["kind"] == "tracks"]["verdict"]) == ["current"]

    def test_a_directly_reached_row_is_not_also_reported_transitively(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """The crop declares ``media_raw`` and reads the same tracks.

        It qualifies both ways; it must appear once, as ``direct``, because that
        is the edge a reader can act on without following another.
        """
        from mosaic.core.pipeline.run import run_feature

        ds = scenario_dataset_with_media
        _ = self._tracked_dataset(ds)
        _ = run_feature(ds, CropLike())

        self._reorder(ds)
        reached = reached_by(ds, [("", "seq_a")], "media_raw")
        crop = _for(reached, "features", "prov-crop__from__tracks")

        assert len(crop) == 1
        assert list(crop["via"]) == ["direct"]
