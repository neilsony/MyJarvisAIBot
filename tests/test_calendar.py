"""Tests for calendar formatting and the event-creation guards.

The Google client itself isn't exercised here — these cover the logic that
decides what the model sees and what gets written, which is where the bugs
that reach a real calendar would come from.
"""

import asyncio
from pathlib import Path

import pytest

from brain.tools.calendar import (
    _parse_local,
    build_calendar_dispatch,
    calendar_schemas,
    connect_calendar,
    format_events,
    summarize_event,
)


class FakeEvents:
    """Stands in for service.events() — records inserts/patches, serves seeded
    existing events for get()."""

    def __init__(self):
        self.inserted: dict | None = None
        self.patched: dict | None = None
        self._existing: dict[str, dict] = {}

    def seed(self, event_id: str, event: dict) -> None:
        self._existing[event_id] = event

    def insert(self, *, calendarId: str, body: dict):  # noqa: N803 - Google's spelling
        self.inserted = body
        return FakeRequest({**body, "summary": body["summary"], "id": "new-event-id"})

    def list(self, **kwargs):
        return FakeRequest({"items": []})

    def get(self, *, calendarId: str, eventId: str):  # noqa: N803
        if eventId not in self._existing:
            raise LookupError(f"no such event: {eventId}")
        return FakeRequest(self._existing[eventId])

    def patch(self, *, calendarId: str, eventId: str, body: dict):  # noqa: N803
        self.patched = body
        merged = {**self._existing.get(eventId, {}), **body, "id": eventId}
        return FakeRequest(merged)


class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeService:
    def __init__(self):
        self._events = FakeEvents()

    def events(self):
        return self._events


def dispatch_for(service):
    return build_calendar_dispatch(service)


class TestSummarizeEvent:
    def test_timed_event(self):
        got = summarize_event({"summary": "Standup", "start": {"dateTime": "2026-09-21T09:00:00"}})
        assert "Standup" in got and "2026-09-21T09:00:00" in got

    def test_all_day_event_uses_date_not_datetime(self):
        # Google sends `date` for all-day events. Reading only `dateTime` here
        # is what turns a birthday into "at midnight".
        got = summarize_event({"summary": "Birthday", "start": {"date": "2026-09-21"}})
        assert "2026-09-21" in got
        assert "unknown time" not in got

    def test_missing_title_does_not_crash(self):
        assert "(no title)" in summarize_event({"start": {"date": "2026-09-21"}})

    def test_location_is_included_when_present(self):
        got = summarize_event(
            {"summary": "Lunch", "start": {"date": "2026-09-21"}, "location": "Sidecar"}
        )
        assert "Sidecar" in got

    def test_location_omitted_when_absent(self):
        assert "@" not in summarize_event({"summary": "Lunch", "start": {"date": "2026-09-21"}})


class TestParseLocal:
    def test_naive_string_gets_a_timezone_attached(self):
        # The actual bug this exists to fix: Google's API rejects a bare
        # datetime with "Missing time zone definition."
        got = _parse_local("2026-09-21T15:00:00")
        assert got.tzinfo is not None

    def test_wall_clock_time_is_unchanged(self):
        # Attaching a timezone must not shift what time was actually meant —
        # "3pm" stays 3pm, it just also says *which* 3pm now.
        got = _parse_local("2026-09-21T15:00:00")
        assert (got.hour, got.minute, got.second) == (15, 0, 0)

    def test_already_aware_string_is_left_alone(self):
        got = _parse_local("2026-09-21T15:00:00+05:00")
        assert got.utcoffset().total_seconds() == 5 * 3600

    def test_unparseable_string_raises(self):
        with pytest.raises(ValueError):
            _parse_local("whenever works")


class TestFormatEvents:
    def test_empty_says_so_plainly(self):
        assert "Nothing on the calendar" in format_events([])

    def test_one_line_per_event(self):
        events = [
            {"summary": "A", "start": {"date": "2026-09-21"}},
            {"summary": "B", "start": {"date": "2026-09-22"}},
        ]
        assert len(format_events(events).splitlines()) == 2


class TestCreateEvent:
    def test_writes_the_event(self):
        service = FakeService()
        asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "Dentist", "start": "2026-09-21T15:00:00"}
            )
        )
        assert service.events().inserted["summary"] == "Dentist"

    def test_defaults_to_a_one_hour_event(self):
        service = FakeService()
        asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "Dentist", "start": "2026-09-21T15:00:00"}
            )
        )
        body = service.events().inserted
        assert body["start"]["dateTime"].startswith("2026-09-21T15:00:00")
        assert body["end"]["dateTime"].startswith("2026-09-21T16:00:00")

    def test_unparseable_start_is_reported_not_written(self):
        service = FakeService()
        got = asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "Dentist", "start": "next tuesday-ish"}
            )
        )
        assert "Couldn't read" in got
        assert service.events().inserted is None

    def test_backwards_times_are_rejected(self):
        # A real write to a real calendar; better to refuse than to guess.
        service = FakeService()
        got = asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "X", "start": "2026-09-21T15:00:00", "end": "2026-09-21T14:00:00"}
            )
        )
        assert "before it starts" in got
        assert service.events().inserted is None

    def test_location_is_passed_through(self):
        service = FakeService()
        asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "Lunch", "start": "2026-09-21T12:00:00", "location": "Sidecar"}
            )
        )
        assert service.events().inserted["location"] == "Sidecar"

    def test_color_name_is_translated_to_the_google_color_id(self):
        service = FakeService()
        asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "Game night", "start": "2026-09-21T19:00:00", "color": "tomato"}
            )
        )
        assert service.events().inserted["colorId"] == "11"

    def test_color_is_case_insensitive(self):
        service = FakeService()
        asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "X", "start": "2026-09-21T19:00:00", "color": "Basil"}
            )
        )
        assert service.events().inserted["colorId"] == "10"

    def test_unknown_color_is_reported_not_written(self):
        service = FakeService()
        got = asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "X", "start": "2026-09-21T19:00:00", "color": "chartreuse"}
            )
        )
        assert "Unknown color" in got
        assert service.events().inserted is None

    def test_no_color_means_no_colorid_key(self):
        # Omitting it entirely, not sending colorId="" — Google would reject
        # an empty string differently than a missing field.
        service = FakeService()
        asyncio.run(
            dispatch_for(service)["create_calendar_event"](
                {"title": "X", "start": "2026-09-21T19:00:00"}
            )
        )
        assert "colorId" not in service.events().inserted


class TestUpdateEvent:
    def _seeded(self, start="2026-09-21T15:00:00", end="2026-09-21T16:00:00"):
        service = FakeService()
        service.events().seed(
            "evt-1",
            {
                "id": "evt-1",
                "summary": "Original title",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
            },
        )
        return service

    def test_no_event_id_is_reported_not_attempted(self):
        service = FakeService()
        got = asyncio.run(dispatch_for(service)["update_calendar_event"]({}))
        assert "No event id" in got
        assert service.events().patched is None

    def test_title_only_change(self):
        service = self._seeded()
        asyncio.run(
            dispatch_for(service)["update_calendar_event"](
                {"event_id": "evt-1", "title": "New title"}
            )
        )
        body = service.events().patched
        assert body["summary"] == "New title"
        assert "start" not in body and "end" not in body

    def test_no_fields_given_is_reported_not_a_no_op_write(self):
        service = self._seeded()
        got = asyncio.run(
            dispatch_for(service)["update_calendar_event"]({"event_id": "evt-1"})
        )
        assert "Nothing to change" in got
        assert service.events().patched is None

    def test_new_start_without_new_end_preserves_original_duration(self):
        # The actual bug this exists to prevent: a naive patch with only a new
        # start would leave the end at its old absolute time, potentially
        # producing an event that ends before it starts.
        service = self._seeded(start="2026-09-21T15:00:00", end="2026-09-21T16:00:00")
        asyncio.run(
            dispatch_for(service)["update_calendar_event"](
                {"event_id": "evt-1", "start": "2026-09-21T17:00:00"}
            )
        )
        body = service.events().patched
        assert body["start"]["dateTime"].startswith("2026-09-21T17:00:00")
        # Original was a 1-hour meeting; moved start should keep it 1 hour.
        assert body["end"]["dateTime"].startswith("2026-09-21T18:00:00")

    def test_new_start_and_new_end_uses_both_exactly(self):
        service = self._seeded()
        asyncio.run(
            dispatch_for(service)["update_calendar_event"](
                {
                    "event_id": "evt-1",
                    "start": "2026-09-21T09:00:00",
                    "end": "2026-09-21T09:30:00",
                }
            )
        )
        body = service.events().patched
        assert body["start"]["dateTime"].startswith("2026-09-21T09:00:00")
        assert body["end"]["dateTime"].startswith("2026-09-21T09:30:00")

    def test_backwards_result_is_rejected(self):
        service = self._seeded()
        got = asyncio.run(
            dispatch_for(service)["update_calendar_event"](
                {
                    "event_id": "evt-1",
                    "start": "2026-09-21T09:00:00",
                    "end": "2026-09-21T08:00:00",
                }
            )
        )
        assert "before it starts" in got
        assert service.events().patched is None

    def test_unparseable_start_is_reported_not_attempted(self):
        service = self._seeded()
        got = asyncio.run(
            dispatch_for(service)["update_calendar_event"](
                {"event_id": "evt-1", "start": "whenever works"}
            )
        )
        assert "Couldn't read" in got
        assert service.events().patched is None

    def test_unknown_event_id_is_reported_not_a_crash(self):
        service = FakeService()  # nothing seeded
        got = asyncio.run(
            dispatch_for(service)["update_calendar_event"](
                {"event_id": "does-not-exist", "start": "2026-09-21T09:00:00"}
            )
        )
        assert "Couldn't find" in got


class TestWiring:
    def test_schemas_and_dispatch_agree(self):
        advertised = {s["function"]["name"] for s in calendar_schemas()}
        assert advertised == set(dispatch_for(FakeService()))

    def test_no_client_id_means_no_calendar(self, tmp_path: Path):
        # A fresh checkout has no Google setup and must still run.
        got = connect_calendar(None, "secret", "{}", tmp_path / ".env")
        assert got is None

    def test_no_client_secret_means_no_calendar(self, tmp_path: Path):
        got = connect_calendar("id", None, "{}", tmp_path / ".env")
        assert got is None

    def test_client_id_without_token_raises_with_instructions(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="authorize_google"):
            connect_calendar("id", "secret", None, tmp_path / ".env")
