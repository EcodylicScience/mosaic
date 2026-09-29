"""Resolving the media variant file an entry is read from.

A variant's row records the entry's media composition when the file was
written, and a consumer refuses the row when the entry's composition has moved
since. A blank composition on either side is unknown rather than drift, so the
placement is the last check: it must still map into the frames the entry holds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.preprocess_index import MediaVariantDriftedError
from mosaic.core.pipeline.variant_source import resolve_variant_source

from tests.helpers import MediaClip, add_media_variant, make_dataset, write_media_index

_VARIANT = "preprocess.0.1-aaaaaaaaaa"


def test_a_placement_that_no_longer_fits_the_entry_is_drift_naming_overwrite(
    tmp_path: Path,
) -> None:
    """The entry is indexed again with no content identity and more frames.

    Its current composition is then unknown, so nothing compares as drifted,
    and the variant's 12-frame placement cannot map into the 20 frames the entry
    holds now. The preprocess op reuses a file whose composition is unknown, so
    the remedy names overwrite.
    """
    ds = make_dataset(tmp_path / "ds")
    _ = add_media_variant(ds, _VARIANT, "s", composition="a-composition")
    write_media_index(ds, [MediaClip(sequence="s", frame_count=20)])
    (entry,) = ds.resolve_media_scope(None)

    with pytest.raises(MediaVariantDriftedError, match="with overwrite"):
        _ = resolve_variant_source(ds, _VARIANT, entry)
