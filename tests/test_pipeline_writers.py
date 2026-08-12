from __future__ import annotations

import pandas as pd
import pytest

from mosaic.core.pipeline._utils import FeatureMeta
from mosaic.core.pipeline.manifest import CoreSelector
from mosaic.core.pipeline.run import build_feature_meta, build_output_path
from mosaic.core.pipeline.writers import trim_feature_output, write_output


@pytest.fixture
def meta(tmp_path):
    return FeatureMeta(
        group="g1",
        sequence="s1",
        out_path=tmp_path / "g1__s1.parquet",
    )


def _selector(first: int, last: int) -> CoreSelector:
    return CoreSelector(order_col="frame", first=first, last=last, entry_key="g__s2")


class TestTrimFeatureOutput:
    def test_trim_keeps_the_entry_and_drops_its_neighbours(self):
        df = pd.DataFrame({"frame": range(10), "a": range(10)})
        trimmed = trim_feature_output(df, _selector(2, 7))
        assert list(trimmed["frame"]) == [2, 3, 4, 5, 6, 7]

    def test_trim_is_a_noop_when_the_output_is_all_core(self):
        df = pd.DataFrame({"frame": range(5), "a": range(5)})
        assert len(trim_feature_output(df, _selector(0, 4))) == 5

    def test_trim_none_returns_none(self):
        assert trim_feature_output(None, _selector(0, 5)) is None

    def test_trim_survives_a_feature_that_reordered_its_rows(self):
        """The defect the row offsets could not survive.

        A feature sorting by ``(id, frame)`` scatters the entry's rows into one
        block per individual, so a positional window lands on the wrong ones --
        with the right row *count*, which is why it read as correct.
        """
        frames = [7, 8, 9] + list(range(10, 14))
        df = pd.DataFrame({"frame": frames * 2, "id": [0] * 7 + [1] * 7}).sort_values(
            ["id", "frame"]
        )
        trimmed = trim_feature_output(df, _selector(10, 13))
        assert sorted(trimmed["frame"].unique()) == [10, 11, 12, 13]
        assert len(trimmed) == 8

    def test_trim_survives_a_feature_that_dropped_rows(self):
        """A filtered output is shorter than its input, which offsets mis-slice."""
        df = pd.DataFrame({"frame": [7, 9, 11, 13, 21], "a": range(5)})
        trimmed = trim_feature_output(df, _selector(10, 19))
        assert list(trimmed["frame"]) == [11, 13]

    def test_trim_refuses_an_output_with_no_frame_axis(self):
        """A per-sequence summary cannot be told from its neighbours' rows."""
        df = pd.DataFrame({"mean_speed": [1.0]})
        with pytest.raises(ValueError, match="does not carry it"):
            trim_feature_output(df, _selector(0, 5))

    def test_trim_refuses_when_nothing_matches_the_entry(self):
        """Rows addressed to frames the entry does not cover are not "no rows".

        Writing an empty parquet here would read as an entry that legitimately
        produced nothing, hiding a feature that re-based the frame axis.
        """
        df = pd.DataFrame({"frame": [100, 101], "a": [1, 2]})
        with pytest.raises(ValueError, match="kept no rows"):
            trim_feature_output(df, _selector(0, 5))


class TestWriteOutput:
    def test_write_dataframe(self, meta):
        df = pd.DataFrame(
            {"frame": [0, 1], "val": [1.0, 2.0], "sequence": ["s1", "s1"]}
        )
        n_rows = write_output(meta, df)
        assert meta.out_path.exists()
        assert n_rows == 2
        result = pd.read_parquet(meta.out_path)
        assert len(result) == 2

    def test_write_none_creates_empty(self, meta):
        n_rows = write_output(meta, None)
        assert meta.out_path.exists()
        assert n_rows == 0

    def test_failed_write_leaves_no_partial_or_temp(self, meta, monkeypatch):
        """A write that raises mid-way leaves no output and no temp residue."""

        def boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError):
            write_output(meta, pd.DataFrame({"a": [1, 2]}))
        assert not meta.out_path.exists()
        assert list(meta.out_path.parent.glob("*.tmp")) == []

    def test_failed_write_preserves_existing_output(self, meta, monkeypatch):
        """A failed re-write never clobbers a pre-existing valid output."""
        write_output(meta, pd.DataFrame({"a": [1, 2, 3]}))
        assert len(pd.read_parquet(meta.out_path)) == 3

        def boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(RuntimeError):
            write_output(meta, pd.DataFrame({"a": [9]}))
        # Old file intact, temp cleaned up.
        assert len(pd.read_parquet(meta.out_path)) == 3
        assert list(meta.out_path.parent.glob("*.tmp")) == []

    def test_output_permissions_respect_umask(self, meta):
        """Output is not left at mkstemp's private 0600 mode."""
        from mosaic.core.pipeline._utils import _UMASK

        write_output(meta, pd.DataFrame({"a": [1]}))
        mode = meta.out_path.stat().st_mode & 0o777
        assert mode == (0o666 & ~_UMASK)  # umask-respecting, not 0o600


class TestBuildFeatureMeta:
    def test_basic(self, tmp_path):
        meta = build_feature_meta("group1", "seq1", tmp_path)
        assert meta.group == "group1"
        assert meta.sequence == "seq1"
        assert meta.out_path.parent == tmp_path

    def test_empty_group(self, tmp_path):
        meta = build_feature_meta("", "seq1", tmp_path)
        assert "__" not in meta.out_path.name  # no double-underscore prefix


class TestBuildOutputPath:
    def test_with_group(self, tmp_path):
        path = build_output_path("g1", "s1", tmp_path)
        assert path.name.endswith(".parquet")
        assert path.parent == tmp_path

    def test_without_group(self, tmp_path):
        path = build_output_path("", "s1", tmp_path)
        assert "__" not in path.name or path.name.startswith("__") is False
