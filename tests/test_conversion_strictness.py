"""`strict_schema=True` must refuse a bad table, not warn and skip the sequence.

Three conversion paths write a tracks table -- single-file, enumerable expansion,
and merge-per-sequence -- and the flag reached none of them in a way a caller could
act on:

* the merge path hard-coded ``strict=False`` and discarded the report entirely,
  while ``conv_params`` sat in scope two lines above;
* both single-file paths did pass the flag, but they run inside
  ``_convert_rows_individually``, whose bare ``except Exception`` turned the
  resulting ``ValueError`` into a ``[WARN]`` line on **stdout** and a skipped
  sequence;
* ``convert_all_tracks`` returned ``None`` and reported no counts, so the CLI
  emitted ``{"status": "ok"}`` regardless -- a run in which every sequence failed
  to convert exited 0 saying ok.

So a user who asked for strict validation got a warning and a silently smaller
dataset. That is worse than not having the flag, because the run claims success.

The counter-test matters as much as the refusal: a *non-strict* conversion must
still warn and continue, because that is the documented default and the whole
suite depends on it.

Both write paths are reached by forcing ``merge_per_sequence`` rather than by
registering a second, merging converter. A merging converter registered here would
join the registry ``test_track_converters`` parametrizes over and fail its
conformance check for not declaring how a stem names its sequence -- a test double
should not become part of the contract other tests measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)


class _BadParams(TrackConvertParams):
    pass


@register_track_converter
class _MissingKeypointsConverter(TrackConverter[_BadParams]):
    """Emits a frame with no ``poseX*``/``poseY*``, which ``trex_v1`` requires."""

    src_format = "test_missing_keypoints"
    version = "0.1"
    Params = _BadParams

    def convert(
        self, path: Path, params: _BadParams, hints: EntryHints
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": range(4),
                "time": [f / 30.0 for f in range(4)],
                "id": [0] * 4,
                "group": [hints.group] * 4,
                "sequence": [hints.sequence] * 4,
            }
        )


def _indexed(tmp_path: Path, files: int) -> Dataset:
    """A dataset with *files* raw inputs indexed against the bad converter."""
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
    raw = ds.base_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for index in range(files):
        suffix = f"_id{index}" if files > 1 else ""
        np.save(raw / f"myseq{suffix}.npy", np.zeros((2, 2)))
    ds.index_tracks_raw([raw], patterns=["*.npy"], src_format="test_missing_keypoints")
    return ds


@pytest.mark.parametrize("merge", [False, True], ids=["single-file", "merged"])
def test_strict_schema_refuses_instead_of_skipping(tmp_path: Path, merge: bool) -> None:
    ds = _indexed(tmp_path, 2 if merge else 1)

    with pytest.raises(ValueError, match="validation failed"):
        _ = ds.convert_all_tracks(
            params={"strict_schema": True}, merge_per_sequence=merge
        )


@pytest.mark.parametrize("merge", [False, True], ids=["single-file", "merged"])
def test_a_non_strict_conversion_still_warns_and_continues(
    tmp_path: Path, merge: bool
) -> None:
    """The default is permissive, and must stay permissive.

    One converted entry per raw file: the ``_id<N>`` suffix is not an id marker any
    converter recognises, so each file is its own sequence even down the merge
    branch, which then finds one file per group.
    """
    files = 2 if merge else 1
    ds = _indexed(tmp_path, files)

    outcome = ds.convert_all_tracks(merge_per_sequence=merge)

    assert outcome.failed == 0
    assert outcome.converted == files
    assert outcome.ok is True


def test_a_conversion_reports_what_it_did(tmp_path: Path) -> None:
    """``convert_all_tracks`` returned ``None``, so nothing could be reported.

    A caller -- the CLI included -- had no way to tell a run that converted
    everything from one that converted nothing, and said ``{"status": "ok"}``
    either way.
    """
    ds = _indexed(tmp_path, 1)

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed, outcome.ok) == (1, 0, True)
