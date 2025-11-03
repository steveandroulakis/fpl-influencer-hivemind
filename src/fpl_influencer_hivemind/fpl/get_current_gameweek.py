"""Retrieve current FPL gameweek information."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .utils import (
    DEFAULT_TIMEZONE,
    FPL_TIMEZONE,
    create_fpl_session,
    get_bootstrap_data,
    parse_fpl_datetime,
    safe_close_session,
)


def find_current_gameweek(
    events: list[dict[str, Any]], reference_time: datetime
) -> dict[str, Any]:
    current_events = [event for event in events if event.get("is_current", False)]
    if len(current_events) == 1:
        event = current_events[0]
        return {
            "id": event["id"],
            "name": event["name"],
            "is_current": True,
            "is_next": False,
            "is_past": False,
            "deadline_time": event.get("deadline_time"),
            "finished": event.get("finished", False),
            "data_checked": event.get("data_checked", False),
            "method": "api_is_current_flag",
        }

    reference_time_utc = reference_time.astimezone(FPL_TIMEZONE)
    sorted_events = sorted(events, key=lambda item: item.get("id", 0))

    for index, event in enumerate(sorted_events):
        deadline_raw = event.get("deadline_time")
        if not deadline_raw:
            continue
        deadline = parse_fpl_datetime(deadline_raw)
        if reference_time_utc < deadline:
            if index == 0:
                return {
                    "id": event["id"],
                    "name": event["name"],
                    "is_current": False,
                    "is_next": True,
                    "is_past": False,
                    "deadline_time": deadline_raw,
                    "finished": event.get("finished", False),
                    "data_checked": event.get("data_checked", False),
                    "method": "deadline_calculation_next",
                }
            previous = sorted_events[index - 1]
            return {
                "id": previous["id"],
                "name": previous["name"],
                "is_current": True,
                "is_next": False,
                "is_past": False,
                "deadline_time": previous.get("deadline_time"),
                "finished": previous.get("finished", False),
                "data_checked": previous.get("data_checked", False),
                "method": "deadline_calculation_current",
            }

    last = sorted_events[-1]
    return {
        "id": last["id"],
        "name": last["name"],
        "is_current": False,
        "is_next": False,
        "is_past": True,
        "deadline_time": last.get("deadline_time"),
        "finished": last.get("finished", False),
        "data_checked": last.get("data_checked", False),
        "method": "deadline_calculation_last",
    }


def format_gameweek_info(
    gameweek: dict[str, Any],
    reference_time: datetime,
    local_tz: Any = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    result = {
        "id": gameweek["id"],
        "name": gameweek["name"],
        "is_current": gameweek["is_current"],
        "is_next": gameweek["is_next"],
        "is_past": gameweek["is_past"],
        "finished": gameweek.get("finished", False),
        "data_checked": gameweek.get("data_checked", False),
        "calculation_method": gameweek.get("method", "unknown"),
        "now_utc": reference_time.astimezone(UTC).isoformat(),
        "now_local": reference_time.astimezone(local_tz).isoformat(),
    }

    deadline_raw = gameweek.get("deadline_time")
    if deadline_raw:
        deadline_utc = parse_fpl_datetime(deadline_raw)
        result["deadline_time_utc"] = deadline_utc.isoformat()
        result["deadline_time_local"] = deadline_utc.astimezone(local_tz).isoformat()
        diff = deadline_utc - reference_time.astimezone(FPL_TIMEZONE)
        if diff.total_seconds() > 0:
            result["time_until_deadline"] = str(diff)
            result["deadline_passed"] = False
        else:
            result["time_since_deadline"] = str(abs(diff))
            result["deadline_passed"] = True
    else:
        result["deadline_time_utc"] = None
        result["deadline_time_local"] = None
        result["deadline_passed"] = None

    return result


async def get_current_gameweek_info(
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    reference = reference_time or datetime.now(FPL_TIMEZONE)
    fpl, session = await create_fpl_session()
    try:
        bootstrap = await get_bootstrap_data(fpl)
        events = bootstrap.get("events") or []
        if not events:
            raise RuntimeError("No gameweek data found in FPL bootstrap response")
        raw = find_current_gameweek(events, reference)
        return format_gameweek_info(raw, reference, DEFAULT_TIMEZONE)
    finally:
        await safe_close_session(session)
