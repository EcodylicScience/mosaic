"""Test resolving the media variant file that an entry is read from.

A variant's row records the entry's media composition when the file was written, and a
consumer refuses the row when the entry's composition has changed since. A blank
composition on either side is unknown rather than drift. The placement is therefore the
last check, and it must still map into the frames that the entry has. Each refusal
prints the command that rewrites the variant from its recorded recipe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.preprocess_index import MediaVariantDriftedError
from mosaic.core.pipeline.preprocess_layout import media_variant_recipe_path
from mosaic.core.pipeline.variant_source import (
    VariantLookup,
    preprocess_command,
    unreadable_variant_refusal,
)

from mosaic.tracking.common.scope import build_work_items

from tests.helpers import (
    IndexReads,
    MediaClip,
    add_media_variant,
    count_index_reads,
    make_dataset,
    write_media_index,
)

_VARIANT = "preprocess.0.1-aaaaaaaaaa"


def test_a_placement_that_no_longer_fits_the_entry_is_drift_naming_a_rewrite(
    tmp_path: Path,
) -> None:
    """The entry is indexed again without a content identity and with more frames.

    Its current composition is then unknown and does not compare as drifted. The
    variant's 12-frame placement cannot map into the 20 frames that the entry has
    now. The preprocess op compares the recorded placement too, and a plain run of
    the same recipe rewrites the file.
    """
    ds = make_dataset(tmp_path / "ds")
    _ = add_media_variant(ds, _VARIANT, "s", composition="a-composition")
    write_media_index(ds, [MediaClip(sequence="s", frame_count=20)])
    (entry,) = ds.resolve_media_scope(None)

    with pytest.raises(MediaVariantDriftedError) as drifted:
        _ = VariantLookup.read(ds, _VARIANT, [("", "s")]).resolve(ds, entry)

    message = str(drifted.value)
    assert "rewrites a variant whose placement no longer fits" in message
    assert "overwrite" not in message


def test_the_rewrite_command_names_the_recorded_recipe(tmp_path: Path) -> None:
    """The path is absolute, because ``@file`` is read from the working directory."""
    ds = make_dataset(tmp_path / "ds")
    recipe = media_variant_recipe_path(ds, _VARIANT)
    recipe.parent.mkdir(parents=True)
    _ = recipe.write_text("{}")

    command = preprocess_command(ds, _VARIANT, [("g", "b"), ("", "a")])

    assert command == (
        "    mosaic run -m <manifest> --kind preprocess --entries :a "
        f"--entries g:b --params @{recipe.absolute()}"
    )


def test_a_variant_without_a_recorded_recipe_says_so(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    command = preprocess_command(ds, _VARIANT, [("", "a")])

    first, second = command.splitlines()
    assert first == (
        "    mosaic run -m <manifest> --kind preprocess --entries :a "
        f"--params '<the parameters of {_VARIANT}>'"
    )
    recipe = media_variant_recipe_path(ds, _VARIANT).absolute()
    assert second == (
        f"    {_VARIANT} does not record a recipe at {recipe}. Give the parameters "
        f"that made it."
    )


@pytest.mark.parametrize(
    "unresolved", [{}, {"s": ("", "s")}], ids=["none-unresolved", "one-unresolved"]
)
def test_a_run_that_lost_no_entry_earns_no_unreadable_variant_refusal(
    tmp_path: Path, unresolved: dict[str, tuple[str, str]]
) -> None:
    ds = make_dataset(tmp_path / "ds")

    refusal = unreadable_variant_refusal(
        ds, "sleap", _VARIANT, "sleap.0.1-0000000000", lost=set(), unresolved=unresolved
    )

    assert refusal is None


def test_a_run_that_lost_only_unreadable_entries_is_refused_naming_them(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")

    refusal = unreadable_variant_refusal(
        ds,
        "sleap",
        _VARIANT,
        "sleap.0.1-0000000000",
        lost={"s"},
        unresolved={"s": ("", "s")},
    )

    assert refusal is not None
    assert "run_id=sleap.0.1-0000000000 did not produce tracks" in refusal
    assert f"the media variant {_VARIANT}: s." in refusal


def test_work_items_read_each_index_once_for_their_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    sequences = ("s", "t", "u")
    write_media_index(
        ds,
        [
            MediaClip(
                sequence=sequence,
                filename=f"{sequence}.mp4",
                video_uuid=f"uid-{sequence}",
                width=64,
                height=48,
                frame_count=12,
            )
            for sequence in sequences
        ],
    )
    for sequence in sequences:
        _ = add_media_variant(ds, _VARIANT, sequence)
    scope = ds.resolve_media_scope(None)
    reads = count_index_reads(monkeypatch)

    built = build_work_items(ds, scope, kind="sleap", media=_VARIANT)

    assert [item.sequence for item in built.items] == list(sequences)
    assert built.failures == ()
    assert reads == IndexReads(media_scopes=0, variant_indexes=1, compositions=1)
