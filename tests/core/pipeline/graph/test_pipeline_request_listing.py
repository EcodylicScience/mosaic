"""A dataset's pipeline requests, listed a page at a time from their files."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from mosaic.core.pipeline.graph import (
    Request,
    RequestUnreadable,
    list_requests,
    request_path,
    save_request,
)
import mosaic.core.pipeline.graph.store as store_module
from mosaic.runlog import JsonlRunLog, run_log_path

REQUEST_IDS = (
    "01JC000000000000000000000A",
    "01JC000000000000000000000B",
    "01JC000000000000000000000C",
    "01JC000000000000000000000D",
    "01JC000000000000000000000E",
)
"""Request ids in submission order, as ULIDs sort."""


def _finish(base: Path, execution_id: str, *, lost: int = 0) -> None:
    with JsonlRunLog(run_log_path(base, execution_id), execution_id) as log:
        log.started(kind="feature", target="speed")
        for index in range(lost):
            log.entry_failed(f"g__s{index}", '{"message": "boom"}')
        log.finished()


def _save(base: Path, request_id: str, *steps: str) -> None:
    executions = {step: f"{request_id}-{step}" for step in steps}
    save_request(
        base,
        Request(
            request_id=request_id,
            recipe_digest="d" * 64,
            owner="ana",
            created_at="2026-10-14T00:00:00Z",
            step_executions=executions,
        ),
    )


@pytest.fixture
def requests(tmp_path: Path) -> Path:
    for request_id in REQUEST_IDS:
        _save(tmp_path, request_id, "speed")
    return tmp_path


def _ids(base: Path, *, limit: int | None, after: str | None) -> list[str]:
    return [
        item.request_id for item in list_requests(base, limit=limit, after=after).items
    ]


def test_requests_are_listed_newest_first(requests: Path) -> None:
    page = list_requests(requests)

    assert [item.request_id for item in page.items] == list(reversed(REQUEST_IDS))
    assert page.next_after is None


def test_pages_follow_one_another_until_the_last(requests: Path) -> None:
    first = list_requests(requests, limit=2)
    second = list_requests(requests, limit=2, after=first.next_after)
    third = list_requests(requests, limit=2, after=second.next_after)

    assert [item.request_id for item in first.items] == [
        "01JC000000000000000000000E",
        "01JC000000000000000000000D",
    ]
    assert first.next_after == "01JC000000000000000000000D"
    assert [item.request_id for item in second.items] == [
        "01JC000000000000000000000C",
        "01JC000000000000000000000B",
    ]
    assert [item.request_id for item in third.items] == ["01JC000000000000000000000A"]
    assert third.next_after is None


def test_a_page_that_ends_on_the_last_request_names_no_next_page(
    requests: Path,
) -> None:
    page = list_requests(requests, limit=2, after="01JC000000000000000000000C")

    assert [item.request_id for item in page.items] == [
        "01JC000000000000000000000B",
        "01JC000000000000000000000A",
    ]
    assert page.next_after is None


def test_a_request_submitted_between_pages_leaves_the_next_page_unchanged(
    requests: Path,
) -> None:
    first = list_requests(requests, limit=2)
    expected = _ids(requests, limit=2, after=first.next_after)

    _save(requests, "01JC000000000000000000000F", "speed")

    assert _ids(requests, limit=2, after=first.next_after) == expected
    assert _ids(requests, limit=1, after=None) == ["01JC000000000000000000000F"]


def test_only_the_requests_on_the_page_are_read(
    requests: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[str] = []
    load = store_module.load_request

    def counting_load(base_dir: Path | str, request_id: str) -> Request:
        read.append(request_id)
        return load(base_dir, request_id)

    monkeypatch.setattr(store_module, "load_request", counting_load)

    _ = list_requests(requests, limit=2, after="01JC000000000000000000000D")

    assert read == ["01JC000000000000000000000C", "01JC000000000000000000000B"]


def test_each_item_reports_the_request_and_its_verdict(tmp_path: Path) -> None:
    _save(tmp_path, "01JC000000000000000000000A", "speed", "scale")
    _finish(tmp_path, "01JC000000000000000000000A-speed")
    _finish(tmp_path, "01JC000000000000000000000A-scale", lost=1)
    _save(tmp_path, "01JC000000000000000000000B", "speed")
    _finish(tmp_path, "01JC000000000000000000000B-speed")
    _save(tmp_path, "01JC000000000000000000000C", "speed")

    pending, finished, partial = list_requests(tmp_path).items

    assert (partial.request_id, partial.status, partial.verdict) == (
        "01JC000000000000000000000A",
        "finished",
        "partial",
    )
    assert (partial.recipe_digest, partial.owner, partial.step_count) == (
        "d" * 64,
        "ana",
        2,
    )
    assert partial.created_at == "2026-10-14T00:00:00Z"
    assert partial.terminal
    assert (finished.verdict, finished.terminal) == ("finished", True)
    assert (pending.verdict, pending.terminal) == ("running", False)


def test_a_dataset_without_requests_lists_none(tmp_path: Path) -> None:
    page = list_requests(tmp_path, limit=5)

    assert page.items == ()
    assert page.next_after is None


def test_a_limit_below_one_is_refused(requests: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list_requests(requests, limit=0)


def test_a_temporary_file_beside_the_requests_is_not_listed(requests: Path) -> None:
    temporary = request_path(requests, "01JC000000000000000000000Z").with_name(
        ".01JC000000000000000000000Z-abc.tmp"
    )
    temporary.write_text("{")

    assert _ids(requests, limit=1, after=None) == ["01JC000000000000000000000E"]


def test_a_damaged_request_file_on_the_page_is_reported_by_name(
    requests: Path,
) -> None:
    damaged = request_path(requests, "01JC000000000000000000000Z")
    _ = damaged.write_text('{"bad": 1}')

    with pytest.raises(RequestUnreadable) as raised:
        _ = list_requests(requests, limit=1)

    assert raised.value.error_detail() == {
        "request_id": "01JC000000000000000000000Z",
        "path": str(damaged),
    }


def test_requests_changed_since_an_instant_were_made_or_moved_a_step_after_it(
    tmp_path: Path,
) -> None:
    for request_id in ("01JC000000000000000000000A", "01JC000000000000000000000B"):
        _save(tmp_path, request_id, "speed")
    _finish(tmp_path, "01JC000000000000000000000A-speed")
    with JsonlRunLog(
        run_log_path(tmp_path, "01JC000000000000000000000B-speed"),
        "01JC000000000000000000000B-speed",
    ) as log:
        log.started(kind="feature", target="speed")
    since = datetime.datetime.now(datetime.UTC)
    with JsonlRunLog(
        run_log_path(tmp_path, "01JC000000000000000000000B-speed"),
        "01JC000000000000000000000B-speed",
    ) as log:
        log.finished()
    save_request(
        tmp_path,
        Request(
            request_id="01JC000000000000000000000C",
            recipe_digest="d" * 64,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            step_executions={"speed": "01JC000000000000000000000C-speed"},
        ),
    )

    changed = list_requests(tmp_path, changed_since=since)
    paged = list_requests(tmp_path, changed_since=since, limit=1)

    assert [item.request_id for item in changed.items] == [
        "01JC000000000000000000000C",
        "01JC000000000000000000000B",
    ]
    assert [item.request_id for item in paged.items] == ["01JC000000000000000000000C"]
    assert paged.next_after == "01JC000000000000000000000C"


def test_an_unreadable_request_is_listed_as_changed_where_its_page_reaches_it(
    tmp_path: Path,
) -> None:
    damaged = request_path(tmp_path, "00000000000000000000000000")
    damaged.parent.mkdir(parents=True)
    _ = damaged.write_text('{"bad": 1}')
    since = datetime.datetime.now(datetime.UTC)
    save_request(
        tmp_path,
        Request(
            request_id="01JC000000000000000000000C",
            recipe_digest="d" * 64,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            step_executions={"speed": "01JC000000000000000000000C-speed"},
        ),
    )

    first = list_requests(tmp_path, changed_since=since, limit=1)

    assert [item.request_id for item in first.items] == ["01JC000000000000000000000C"]
    assert first.next_after == "01JC000000000000000000000C"
    with pytest.raises(RequestUnreadable):
        _ = list_requests(tmp_path, changed_since=since)
