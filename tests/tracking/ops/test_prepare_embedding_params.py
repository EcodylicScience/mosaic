"""The parameters of an embedding preparation validate its entry lists."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mosaic.tracking.ops.prepare_embedding import PrepareEmbeddingDataParams


def _params(
    train: list[tuple[str, str]],
    val: list[tuple[str, str]],
    **overrides: object,
) -> PrepareEmbeddingDataParams:
    return PrepareEmbeddingDataParams.model_validate(
        {
            "sleap_run_id": "sleap.1.6-0123456789abcdef",
            "train_entries": [{"group": g, "sequence": s} for g, s in train],
            "val_entries": [{"group": g, "sequence": s} for g, s in val],
            **overrides,
        }
    )


def test_the_defaults_are_the_declared_ones() -> None:
    params = _params([("", "a")], [("", "b")])

    assert params.window_frames == 300
    assert params.frames_per_window == 16
    assert params.min_tracks_present == 2
    assert params.split_tracks_at_gap is None
    assert params.val_windows_per_entry == 60


def test_both_lists_are_stored_sorted() -> None:
    params = _params(
        [("g2", "a"), ("g1", "z"), ("g1", "b")],
        [("", "y"), ("", "x")],
    )

    assert [ref.entry for ref in params.train_entries] == [
        ("g1", "b"),
        ("g1", "z"),
        ("g2", "a"),
    ]
    assert [ref.entry for ref in params.val_entries] == [("", "x"), ("", "y")]


def test_the_order_given_does_not_change_the_identity() -> None:
    first = _params([("", "a"), ("", "b")], [("", "d"), ("", "c")])
    second = _params([("", "b"), ("", "a")], [("", "c"), ("", "d")])

    assert first.identity_dump() == second.identity_dump()


def test_an_entry_in_both_lists_is_refused() -> None:
    with pytest.raises(
        ValidationError, match=r"train_entries and val_entries both name \('g', 'b'\)"
    ):
        _ = _params([("", "a"), ("g", "b")], [("g", "b")])


@pytest.mark.parametrize("field", ["train_entries", "val_entries"])
def test_a_repeated_entry_is_refused(field: str) -> None:
    train = [("", "a"), ("", "a")] if field == "train_entries" else [("", "a")]
    val = [("", "b"), ("", "b")] if field == "val_entries" else [("", "b")]

    with pytest.raises(ValidationError, match=rf"{field} names .* more than once"):
        _ = _params(train, val)


@pytest.mark.parametrize("field", ["train_entries", "val_entries"])
def test_an_empty_list_is_refused(field: str) -> None:
    train: list[tuple[str, str]] = [] if field == "train_entries" else [("", "a")]
    val: list[tuple[str, str]] = [] if field == "val_entries" else [("", "b")]

    with pytest.raises(ValidationError, match=field):
        _ = _params(train, val)


@pytest.mark.parametrize("value", [0, 1])
def test_fewer_than_two_tracks_present_is_refused(value: int) -> None:
    with pytest.raises(ValidationError, match="min_tracks_present"):
        _ = _params([("", "a")], [("", "b")], min_tracks_present=value)


@pytest.mark.parametrize(
    "field",
    [
        "window_frames",
        "frames_per_window",
        "split_tracks_at_gap",
        "val_windows_per_entry",
    ],
)
def test_a_count_below_one_is_refused(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        _ = _params([("", "a")], [("", "b")], **{field: 0})


def test_the_validation_windows_per_entry_enter_the_identity() -> None:
    default = _params([("", "a")], [("", "b")])
    fewer = _params([("", "a")], [("", "b")], val_windows_per_entry=30)
    every = _params([("", "a")], [("", "b")], val_windows_per_entry=None)

    dumps = [each.identity_dump() for each in (default, fewer, every)]
    assert all(dump != dumps[0] for dump in dumps[1:])
    assert dumps[1] != dumps[2]
