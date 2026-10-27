"""A store recording divides into sequences of whole chunks by one pure rule.

Every case here builds its chunk records by hand, without a store on disk. Unless a case
says otherwise, a chunk contains 5 frames at 1 fps and a division contains 2 chunks.
A division then spans 10 seconds, and the gap threshold is 10 seconds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mosaic.core.media.store_division import (
    DIVISION_IDENTITY_SCHEME,
    ChunkRecord,
    DivisionPlan,
    DivisionRefusalReason,
    DivisionRefused,
    DivisionRule,
    MediaRowKey,
    RecordingIdentity,
    RowDivision,
    StoreClock,
    StorePlanInput,
    StoreWindow,
    division_display_name,
    division_rule,
    division_token,
    division_uid,
    plan_division,
)

CHUNKSIZE = 5
RATE = 1.0
T0 = 1_613_762_573.0
"""The first frame time of the first store, 2021-02-19T19:22:53Z."""
START = datetime(2021, 2, 19, 19, 22, 53, tzinfo=UTC)


def rule(
    frames: int = 2 * CHUNKSIZE,
    *,
    chunksize: int = CHUNKSIZE,
    identity: RecordingIdentity = "store",
    rate: float = RATE,
) -> DivisionRule:
    """The rule of *frames* per division over chunks of *chunksize* at *rate*."""
    return division_rule(
        frames, chunksize=chunksize, nominal_interval=1.0 / rate, identity=identity
    )


def store(
    counts: Sequence[int | None],
    *,
    name: str = "rec",
    serial: str = "A",
    uuid: str = "4569de67968740e786e775da50cedf12",
    start: float = T0,
    rate: float = RATE,
    chunksize: int = CHUNKSIZE,
    gaps: Mapping[int, float] | None = None,
    inner_gaps: Mapping[int, float] | None = None,
    anchor: datetime | None = None,
) -> StorePlanInput:
    """A store of one chunk per count, ``None`` for a chunk without an index.

    Every chunk advances the clock by *chunksize* frames, whatever its count.
    ``gaps[c]`` seconds pass before chunk ``c``, and ``inner_gaps[c]`` is the
    widest interval inside chunk ``c``.
    """
    before = gaps or {}
    inside = inner_gaps or {}
    interval = 1.0 / rate
    chunks: list[ChunkRecord] = []
    time = start
    for chunk_id, count in enumerate(counts):
        time += before.get(chunk_id, 0.0)
        path = Path(name) / f"{chunk_id:06d}.mp4"
        if count is None:
            chunks.append(ChunkRecord(chunk_id, path, None, None, None, None))
        else:
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    path=path,
                    frame_count=count,
                    first_time=time,
                    last_time=time + (count - 1) * interval,
                    max_interval=inside.get(chunk_id, interval),
                )
            )
        time += chunksize * interval
    clock = StoreClock(
        anchor_utc=START + timedelta(seconds=start - T0) if anchor is None else anchor,
        anchor_time=start,
        source="created_utc",
        timezone_local=None,
    )
    return StorePlanInput(
        abs_path=f"media/{name}",
        store_uuid=uuid,
        camera_serial=serial,
        chunksize=chunksize,
        fps=rate,
        clock=clock,
        chunks=tuple(chunks),
    )


def windows(plan: DivisionPlan) -> list[tuple[int, int]]:
    """The chunk range of every planned division, in recording order."""
    return [(d.row.chunk_first, d.row.chunk_last) for d in plan.divisions]


def refusal(error: pytest.ExceptionInfo[DivisionRefused]) -> DivisionRefusalReason:
    return error.value.reason


def frozen_rows(plan: DivisionPlan) -> tuple[RowDivision, ...]:
    return tuple(d.row for d in plan.divisions)


# The rule


def test_a_whole_multiple_of_the_chunksize_is_a_rule() -> None:
    made = division_rule(60000, chunksize=10000, nominal_interval=0.05)

    assert (made.frames, made.chunksize, made.chunks_per_division) == (60000, 10000, 6)
    assert made.gap_seconds == pytest.approx(0.5)
    assert made.identity == "store"


def test_an_hour_of_500_second_chunks_is_refused() -> None:
    with pytest.raises(DivisionRefused) as error:
        _ = division_rule(72000, chunksize=10000, nominal_interval=0.05)

    assert refusal(error) == "frames_not_chunk_multiple"


@pytest.mark.parametrize("frames", [0, -10000])
def test_a_division_of_no_frames_is_refused(frames: int) -> None:
    with pytest.raises(DivisionRefused) as error:
        _ = division_rule(frames, chunksize=10000, nominal_interval=0.05)

    assert refusal(error) == "frames_not_chunk_multiple"


def test_a_rule_read_from_a_cell_is_checked_against_its_chunksize() -> None:
    cell = json.dumps(
        {
            "scheme": DIVISION_IDENTITY_SCHEME,
            "frames": 7,
            "chunksize": CHUNKSIZE,
            "identity": "store",
            "gap_seconds": 10.0,
        }
    )

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([store([5, 5])], DivisionRule.from_cell(cell))

    assert refusal(error) == "frames_not_chunk_multiple"


def test_the_rule_cell_is_canonical_json_and_round_trips() -> None:
    made = rule(identity="camera", rate=20.00008)

    cell = made.to_cell()

    assert cell == (
        '{"chunksize":5,"frames":10,"gap_seconds":0.499998000008,'
        '"identity":"camera","scheme":"imgstore-division/1"}'
    )
    assert DivisionRule.from_cell(cell) == made


@pytest.mark.parametrize(
    "cell",
    [
        "not json",
        "[]",
        '{"scheme":"imgstore/1","frames":10,"chunksize":5,"identity":"store","gap_seconds":1.0}',
        '{"scheme":"imgstore-division/1","frames":10,"chunksize":5,"identity":"arena","gap_seconds":1.0}',
        '{"scheme":"imgstore-division/1","frames":"10","chunksize":5,"identity":"store","gap_seconds":1.0}',
        '{"scheme":"imgstore-division/1","frames":10,"chunksize":5,"identity":"store"}',
        '{"scheme":"imgstore-division/1","frames":10,"chunksize":5,"identity":"store","gap_seconds":0}',
    ],
)
def test_a_malformed_rule_cell_is_refused(cell: str) -> None:
    with pytest.raises(ValueError, match="division rule"):
        _ = DivisionRule.from_cell(cell)


@pytest.mark.parametrize("gap_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_a_rule_whose_gap_threshold_is_not_positive_and_finite_is_refused(
    gap_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="gap"):
        _ = DivisionRule(
            frames=10, chunksize=CHUNKSIZE, identity="store", gap_seconds=gap_seconds
        )


# Runs and closure


def test_sequences_count_from_the_first_chunk_and_the_open_one_is_held_back() -> None:
    plan = plan_division([store([5, 5, 5, 5, 5])], rule())

    assert windows(plan) == [(0, 1), (2, 3)]
    assert [(h.chunk_first, h.chunk_last, h.frame_count) for h in plan.held_back] == [
        (4, 4, 5)
    ]
    assert plan.held_back[0].reason == "store_recording"
    assert plan.refused == ()


def test_a_short_last_chunk_ends_the_store() -> None:
    plan = plan_division([store([5, 5, 5, 5, 3])], rule())

    assert windows(plan) == [(0, 1), (2, 3), (4, 4)]
    assert plan.divisions[-1].row.frame_count == 3
    assert plan.held_back == ()


def test_a_trailing_chunk_without_an_index_is_open_and_not_counted() -> None:
    plan = plan_division([store([5, 5, 5, None])], rule())

    assert windows(plan) == [(0, 1)]
    assert [(h.chunk_first, h.chunk_last) for h in plan.held_back] == [(2, 2)]
    assert plan.refused == ()


def test_a_full_sequence_closes_although_its_last_chunk_is_open() -> None:
    plan = plan_division([store([5, 5, 5, 5, None])], rule())

    assert windows(plan) == [(0, 1), (2, 3)]
    assert plan.held_back == ()


def test_include_partial_releases_the_trailing_sequence() -> None:
    plan = plan_division([store([5, 5, 5])], rule(), include_partial=True)

    assert windows(plan) == [(0, 1), (2, 2)]
    assert plan.held_back == ()


def test_a_gap_ends_a_run_and_the_next_run_counts_from_its_first_chunk() -> None:
    plan = plan_division([store([5, 5, 5, 5, 5, 5], gaps={1: 30.0})], rule())

    assert windows(plan) == [(0, 0), (1, 2), (3, 4)]
    assert [d.row.follows_gap for d in plan.divisions] == [False, True, False]
    assert [(h.chunk_first, h.chunk_last) for h in plan.held_back] == [(5, 5)]


def test_an_interval_at_the_threshold_is_not_a_gap() -> None:
    # The step from chunk 0's last frame to chunk 1's first is 1 s plus 9 s.
    plan = plan_division([store([5, 5, 5, 5], gaps={1: 9.0})], rule())

    assert windows(plan) == [(0, 1), (2, 3)]


# Placement


def test_a_division_starts_at_its_first_chunk_on_the_frame_axis() -> None:
    plan = plan_division([store([5, 5, 5, 5, 2])], rule())

    assert [d.row.first_frame for d in plan.divisions] == [0, 10, 20]
    assert [d.row.frame_count for d in plan.divisions] == [10, 10, 2]


def test_by_camera_a_later_store_continues_the_frame_axis_after_a_boundary() -> None:
    first = store([5, 5, 5, 3], name="rec1")
    second = store([5, 5, 5, 5, 5], name="rec2", start=T0 + 100.0)

    plan = plan_division([second, first], rule(identity="camera"))

    assert [(d.row.abs_path, d.row.chunk_first) for d in plan.divisions] == [
        ("media/rec1", 0),
        ("media/rec1", 2),
        ("media/rec2", 0),
        ("media/rec2", 2),
    ]
    assert [d.row.first_frame for d in plan.divisions] == [0, 10, 18, 28]
    assert [d.row.follows_gap for d in plan.divisions] == [False, False, True, False]
    assert [(h.abs_path, h.chunk_first) for h in plan.held_back] == [("media/rec2", 4)]


def test_by_camera_a_later_store_ends_the_store_before_it() -> None:
    first = store([5, 5, 5], name="rec1")
    second = store([5, 5], name="rec2", start=T0 + 100.0)

    plan = plan_division([first, second], rule(identity="camera"))

    assert windows(plan) == [(0, 1), (2, 2), (0, 1)]


def test_an_unindexed_last_chunk_of_an_ended_store_refuses_its_division() -> None:
    # A later store ends the first, so its last chunk is missing its index.
    first = store([5, 5, 5, None], name="rec1")
    second = store([5, 5], name="rec2", start=T0 + 100.0)

    plan = plan_division([first, second], rule(identity="camera"))

    assert [(d.row.abs_path, d.row.chunk_first) for d in plan.divisions] == [
        ("media/rec1", 0),
        ("media/rec2", 0),
    ]
    assert [d.row.first_frame for d in plan.divisions] == [0, 20]
    assert [
        (r.abs_path, r.chunk_first, r.chunk_last, r.reason) for r in plan.refused
    ] == [("media/rec1", 2, 3, "missing_chunk_index")]
    assert plan.held_back == ()


def test_restoring_an_ended_stores_last_index_keeps_the_later_store_in_place() -> None:
    second = store([5, 5], name="rec2", start=T0 + 100.0)
    camera = rule(identity="camera")
    damaged = plan_division([store([5, 5, 5, None], name="rec1"), second], camera)

    restored = plan_division(
        [store([5, 5, 5, 5], name="rec1"), second],
        camera,
        frozen=frozen_rows(damaged),
    )

    assert windows(restored) == [(0, 1), (2, 3), (0, 1)]
    assert [d.row.first_frame for d in restored.divisions] == [0, 10, 20]
    assert restored.refused == ()


# Refusals


def test_a_missing_interior_index_refuses_only_its_division() -> None:
    plan = plan_division([store([5, None, 5, 5, 5])], rule())

    assert windows(plan) == [(2, 3)]
    assert plan.divisions[0].row.first_frame == 10
    assert [(r.chunk_first, r.chunk_last, r.reason) for r in plan.refused] == [
        (0, 1, "missing_chunk_index")
    ]


def test_a_short_interior_chunk_refuses_its_store() -> None:
    plan = plan_division([store([5, 3, 5, 5])], rule())

    assert plan.divisions == ()
    assert plan.held_back == ()
    assert [(r.chunk_first, r.chunk_last, r.reason) for r in plan.refused] == [
        (0, 3, "short_interior_chunk")
    ]


def test_a_short_chunk_before_an_open_one_refuses_its_store() -> None:
    plan = plan_division([store([5, 5, 3, None])], rule())

    assert [r.reason for r in plan.refused] == ["short_interior_chunk"]


def test_a_gap_inside_a_chunk_refuses_its_store() -> None:
    plan = plan_division([store([5, 5, 5], inner_gaps={1: 10.5})], rule())

    assert plan.divisions == ()
    assert [r.reason for r in plan.refused] == ["gap_inside_chunk"]


def test_a_refused_store_still_places_the_stores_after_it() -> None:
    first = store([5, 3, 5], name="rec1")
    second = store([5, 5], name="rec2", start=T0 + 100.0)

    plan = plan_division([first, second], rule(identity="camera"))

    assert [(d.row.abs_path, d.row.first_frame) for d in plan.divisions] == [
        ("media/rec2", 13)
    ]


def test_stores_at_two_rates_are_refused() -> None:
    first = store([5, 5], name="rec1")
    second = store([5, 5], name="rec2", start=T0 + 100.0, rate=1.25)

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([first, second], rule(identity="camera"))

    assert refusal(error) == "rate_mismatch"


@pytest.mark.parametrize(
    "rates",
    [
        (1.0, 1.0004, 0.9996),
        (1.0004, 1.0, 0.9996),
        (0.9996, 1.0004, 1.0),
        (1.0, 0.9996, 1.0004),
    ],
)
def test_the_rate_check_does_not_depend_on_the_order_of_the_stores(
    rates: tuple[float, float, float],
) -> None:
    # Each rate is within the tolerance of 1.0, and the outer two are not.
    stores = [
        store([5, 5], name=f"rec{rate}", start=T0 + 100.0 * position, rate=rate)
        for position, rate in enumerate(sorted(rates))
    ]
    given = sorted(stores, key=lambda candidate: rates.index(candidate.fps))

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division(given, rule(identity="camera"))

    assert refusal(error) == "rate_mismatch"


def test_stores_of_one_rate_measured_twice_are_one_recording() -> None:
    first = store([5, 5], name="rec1")
    second = store([5, 5], name="rec2", start=T0 + 100.0, rate=1.0001)

    plan = plan_division([first, second], rule(identity="camera"))

    assert len(plan.divisions) == 2


def test_overlapping_stores_of_one_camera_are_refused() -> None:
    first = store([5, 5], name="rec1")
    second = store([5, 5], name="rec2", start=T0 + 5.0)

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([first, second], rule(identity="camera"))

    assert refusal(error) == "overlapping_stores"


def test_by_camera_a_store_without_a_serial_is_refused() -> None:
    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([store([5, 5], serial="")], rule(identity="camera"))

    assert refusal(error) == "empty_camera_serial"


def test_by_camera_stores_of_two_cameras_are_refused() -> None:
    first = store([5, 5], name="rec1")
    second = store([5, 5], name="rec2", serial="B", start=T0 + 100.0)

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([first, second], rule(identity="camera"))

    assert refusal(error) == "camera_serial_mismatch"


def test_by_store_a_recording_is_one_store() -> None:
    first = store([5, 5], name="rec1")
    second = store([5, 5], name="rec2", start=T0 + 100.0)

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([first, second], rule())

    assert refusal(error) == "several_stores"


def test_a_chunksize_other_than_the_rules_is_refused() -> None:
    first = store([5, 5], name="rec1")
    second = store([4, 4], name="rec2", start=T0 + 100.0, chunksize=4)

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([first, second], rule(identity="camera"))

    assert refusal(error) == "chunksize_mismatch"


def test_a_store_without_a_uuid_is_refused() -> None:
    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([store([5, 5], uuid="")], rule())

    assert refusal(error) == "empty_store_uuid"


def test_two_divisions_starting_in_one_second_are_refused() -> None:
    # Chunks of 5 frames at 20 fps start a quarter second apart.
    made = rule(CHUNKSIZE, rate=20.0)

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([store([5, 5, 2], rate=20.0)], made)

    assert refusal(error) == "duplicate_token"


def test_a_frozen_division_the_plan_does_not_reproduce_is_refused() -> None:
    planned = plan_division([store([5, 5, 5, 5, 5])], rule())
    moved = replace(planned.divisions[1].row, first_frame=11)

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division(
            [store([5, 5, 5, 5, 5])], rule(), frozen=(planned.divisions[0].row, moved)
        )

    assert refusal(error) == "frozen_mismatch"


def test_a_frozen_division_of_another_store_is_refused() -> None:
    planned = plan_division([store([5, 5], name="rec1")], rule())

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division(
            [store([5, 5], name="rec2")], rule(), frozen=frozen_rows(planned)
        )

    assert refusal(error) == "frozen_mismatch"


def test_a_frozen_division_whose_index_went_missing_is_refused() -> None:
    planned = plan_division([store([5, 5, 5])], rule())

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([store([5, None, 5])], rule(), frozen=frozen_rows(planned))

    assert refusal(error) == "frozen_mismatch"


# Names


@pytest.mark.parametrize(
    ("dirname", "serial", "expected"),
    [
        ("20210219_112253", "23036146", "23036146_20210219_112253"),
        (
            "sound_2_20210930_172126.23739656",
            "23739656",
            "23739656_sound_2_20210930_172126",
        ),
        ("20210219_112253", "", "20210219_112253"),
    ],
)
def test_a_recording_by_store_is_named_by_its_serial_and_base(
    dirname: str, serial: str, expected: str
) -> None:
    plan = plan_division([store([5, 5], name=dirname, serial=serial)], rule())

    assert plan.recording == expected


def test_a_recording_by_camera_is_named_by_its_serial() -> None:
    plan = plan_division([store([5, 5], serial="23036146")], rule(identity="camera"))

    assert plan.recording == "23036146"


def test_an_explicit_name_names_the_recording() -> None:
    plan = plan_division([store([5, 5])], rule(), name="tank_one")

    assert plan.recording == "tank_one"


@pytest.mark.parametrize("name", ["", "tank__one", "tank:one", "tank/one", "tank\\one"])
def test_an_explicit_name_that_cannot_name_a_recording_is_refused(name: str) -> None:
    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([store([5, 5])], rule(), name=name)

    assert refusal(error) == "invalid_name"


def test_a_derived_name_that_cannot_name_a_recording_is_refused() -> None:
    with pytest.raises(DivisionRefused) as error:
        _ = plan_division([store([5, 5], name="tank__one")], rule())

    assert refusal(error) == "invalid_name"


# Frozen divisions


def test_frozen_divisions_keep_their_tokens_and_starts() -> None:
    planned = plan_division([store([5, 5, 5, 5, 5])], rule())
    later_clock = store([5, 5, 5, 5, 5, 5, 5], anchor=START + timedelta(hours=1))

    grown = plan_division([later_clock], rule(), frozen=frozen_rows(planned))

    assert windows(grown) == [(0, 1), (2, 3), (4, 5)]
    assert [d.frozen for d in grown.divisions] == [True, True, False]
    assert [d.row for d in grown.divisions[:2]] == list(frozen_rows(planned))
    assert grown.divisions[2].row.start_utc == START + timedelta(hours=1, seconds=20)


def test_chunks_after_a_released_partial_start_a_new_run() -> None:
    released = plan_division([store([5, 5, 5])], rule(), include_partial=True)

    grown = plan_division(
        [store([5, 5, 5, 5, 5, 5])], rule(), frozen=frozen_rows(released)
    )

    assert windows(grown) == [(0, 1), (2, 2), (3, 4)]
    assert [d.row.first_frame for d in grown.divisions] == [0, 10, 15]
    assert [d.row.follows_gap for d in grown.divisions] == [False, False, False]
    assert [(h.chunk_first, h.chunk_last) for h in grown.held_back] == [(5, 5)]


# Identity


def test_a_division_uid_is_32_hex_characters_and_deterministic() -> None:
    uid = division_uid("4569de67968740e786e775da50cedf12", 0, 5, 60000)

    assert len(uid) == 32
    assert int(uid, 16) >= 0
    assert uid == division_uid("4569de67968740e786e775da50cedf12", 0, 5, 60000)


@pytest.mark.parametrize(
    "changed",
    [
        ("other", 0, 5, 60000),
        ("store", 1, 5, 60000),
        ("store", 0, 6, 60000),
        ("store", 0, 5, 59999),
    ],
)
def test_a_division_uid_moves_with_each_of_its_inputs(
    changed: tuple[str, int, int, int],
) -> None:
    assert division_uid(*changed) != division_uid("store", 0, 5, 60000)


def test_planned_divisions_carry_their_uid_and_row_key() -> None:
    plan = plan_division([store([5, 5, 5, 5, 2])], rule())

    first = plan.divisions[0]
    assert first.uid == division_uid("4569de67968740e786e775da50cedf12", 0, 1, 10)
    assert first.row.key == MediaRowKey("media/rec", 0, 1)
    assert first.row.window == StoreWindow(0, 1)


def test_a_token_floors_its_seconds_and_names_utc() -> None:
    assert division_token(datetime(2021, 2, 19, 19, 22, 53, 999999, tzinfo=UTC)) == (
        "20210219T192253Z"
    )
    local = datetime(2021, 2, 19, 11, 22, 53, tzinfo=UTC).astimezone(
        datetime.now(UTC).astimezone().tzinfo
    )
    assert division_token(local) == "20210219T112253Z"


def test_a_token_of_a_naive_time_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _ = division_token(datetime(2021, 2, 19, 19, 22, 53))


def test_tokens_hold_no_separator_and_sort_in_time_order() -> None:
    plan = plan_division([store([5] * 9, gaps={3: 3600.0}, name="rec")], rule())

    tokens = [d.row.sequence for d in plan.divisions]
    starts = [d.row.start_utc for d in plan.divisions]
    assert all(":" not in token and "__" not in token for token in tokens)
    assert sorted(tokens) == tokens
    assert sorted(starts) == starts
    assert len(set(tokens)) == len(tokens)


def test_a_window_holds_its_first_and_last_chunk() -> None:
    with pytest.raises(ValueError, match="window"):
        _ = StoreWindow(3, 2)


# The Princeton store


def test_the_princeton_store_divides_into_two_sequences_and_holds_one_back() -> None:
    rate = 20.00008
    created = datetime(2021, 2, 19, 19, 22, 53, 32917, tzinfo=UTC)
    princeton = replace(
        store(
            [10000] * 13,
            name="20210219_112253",
            serial="23036146",
            start=1613762573.057566,
            rate=rate,
            chunksize=10000,
            inner_gaps=dict.fromkeys(range(13), 0.0645),
        ),
        clock=StoreClock(
            anchor_utc=created,
            anchor_time=1613762573.057566,
            source="created_utc",
            timezone_local="America/Los_Angeles",
        ),
    )

    plan = plan_division(
        [princeton],
        division_rule(60000, chunksize=10000, nominal_interval=1.0 / rate),
    )

    assert plan.recording == "23036146_20210219_112253"
    assert [
        (d.row.chunk_first, d.row.chunk_last, d.row.sequence, d.row.first_frame)
        for d in plan.divisions
    ] == [(0, 5, "20210219T192253Z", 0), (6, 11, "20210219T201253Z", 60000)]
    assert [d.row.start_source for d in plan.divisions] == ["created_utc"] * 2
    assert [(h.chunk_first, h.chunk_last) for h in plan.held_back] == [(12, 12)]


def test_a_released_partial_division_stays_written_on_a_rescan() -> None:
    released = plan_division([store([5, 5, 5])], rule(), include_partial=True)

    rescanned = plan_division([store([5, 5, 5])], rule(), frozen=frozen_rows(released))

    assert windows(rescanned) == [(0, 1), (2, 2)]
    assert rescanned.held_back == ()


def test_a_frozen_division_of_a_store_now_refused_is_refused() -> None:
    planned = plan_division([store([5, 5, 5, 5])], rule())

    with pytest.raises(DivisionRefused) as error:
        _ = plan_division(
            [store([5, 5, 5, 5], inner_gaps={3: 11.0})],
            rule(),
            frozen=frozen_rows(planned),
        )

    assert refusal(error) == "frozen_mismatch"


@pytest.mark.parametrize(
    ("start", "timezone_local", "label"),
    [
        (START, "America/Los_Angeles", "2021-02-19 11:22:53 PST"),
        (
            datetime(2021, 7, 1, 19, 0, 0, 900000, tzinfo=UTC),
            "America/Los_Angeles",
            "2021-07-01 12:00:00 PDT",
        ),
        (START, None, "2021-02-19 19:22:53Z"),
        (START, "Mars/Olympus_Mons", "2021-02-19 19:22:53Z"),
    ],
    ids=["winter", "summer", "no-zone", "unknown-zone"],
)
def test_a_division_s_label_is_its_start_on_the_local_wall_clock(
    start: datetime, timezone_local: str | None, label: str
) -> None:
    assert division_display_name(start, timezone_local) == label


def test_a_naive_start_has_no_label() -> None:
    with pytest.raises(ValueError, match="needs a timezone"):
        _ = division_display_name(datetime(2021, 2, 19, 19, 22, 53), None)
