#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fpl>=0.6.35",
#   "aiohttp>=3.9.0",
#   "python-dateutil>=2.8.0",
# ]
# ///

"""
Return the current Gameweek for "now" with optional date override.

Determines the current FPL gameweek by analyzing event data and deadline times.
Supports timezone handling and provides comprehensive gameweek information
including deadline times in both UTC and local timezone.

Usage:
    python get_current_gameweek.py
    python get_current_gameweek.py --date 2025-08-24
    uv run get_current_gameweek.py --datetime "2025-08-24T15:30" --out gameweek.json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from utils import (
    DEFAULT_TIMEZONE,
    FPL_TIMEZONE,
    create_fpl_session,
    get_bootstrap_data,
    handle_keyboard_interrupt,
    parse_fpl_datetime,
    safe_close_session,
    write_json,
)


def parse_date_input(
    date_str: str | None,
    datetime_str: str | None,
    is_utc: bool,
    local_tz: ZoneInfo = DEFAULT_TIMEZONE,
) -> datetime:
    """
    Parse date/datetime input from command line arguments.

    Args:
        date_str: Date string in YYYY-MM-DD format
        datetime_str: Datetime string in YYYY-MM-DDTHH:MM format
        is_utc: Whether to treat input as UTC
        local_tz: Local timezone for naive datetime interpretation

    Returns:
        Timezone-aware datetime in UTC
    """
    if datetime_str:
        try:
            # Parse datetime string
            if "T" not in datetime_str:
                raise ValueError("Datetime must be in YYYY-MM-DDTHH:MM format")

            dt = datetime.fromisoformat(datetime_str)

            if dt.tzinfo is None:
                # Naive datetime - apply timezone
                if is_utc:
                    dt = dt.replace(tzinfo=FPL_TIMEZONE)
                else:
                    dt = dt.replace(tzinfo=local_tz)

            return dt.astimezone(FPL_TIMEZONE)

        except ValueError as e:
            print(f"Error parsing datetime '{datetime_str}': {e}", file=sys.stderr)
            sys.exit(1)

    elif date_str:
        try:
            # Parse date string and set to midnight local time
            dt = datetime.fromisoformat(date_str)

            if is_utc:
                dt = dt.replace(tzinfo=FPL_TIMEZONE)
            else:
                dt = dt.replace(tzinfo=local_tz)

            return dt.astimezone(FPL_TIMEZONE)

        except ValueError as e:
            print(f"Error parsing date '{date_str}': {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Use current time
        return datetime.now(FPL_TIMEZONE)


def find_current_gameweek(
    events: list[dict[str, Any]], reference_time: datetime
) -> dict[str, Any]:
    """
    Find the current gameweek based on reference time and event deadlines.

    Args:
        events: List of event (gameweek) data from FPL API
        reference_time: Reference datetime (should be UTC)

    Returns:
        Dictionary with gameweek information and status flags
    """
    # First, check for is_current flag in API data
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

    # Manual calculation based on deadlines
    reference_time_utc = reference_time.astimezone(FPL_TIMEZONE)

    # Sort events by ID (should be chronological)
    sorted_events = sorted(events, key=lambda x: x["id"])

    for i, event in enumerate(sorted_events):
        if not event.get("deadline_time"):
            continue

        deadline = parse_fpl_datetime(event["deadline_time"])

        # Check if we're before this event's deadline
        if reference_time_utc < deadline:
            # This is the next event to happen
            if i == 0:
                # First event of season, not started yet
                return {
                    "id": event["id"],
                    "name": event["name"],
                    "is_current": False,
                    "is_next": True,
                    "is_past": False,
                    "deadline_time": event["deadline_time"],
                    "finished": event.get("finished", False),
                    "data_checked": event.get("data_checked", False),
                    "method": "deadline_calculation_next",
                }
            else:
                # Previous event is current
                prev_event = sorted_events[i - 1]
                return {
                    "id": prev_event["id"],
                    "name": prev_event["name"],
                    "is_current": True,
                    "is_next": False,
                    "is_past": False,
                    "deadline_time": prev_event.get("deadline_time"),
                    "finished": prev_event.get("finished", False),
                    "data_checked": prev_event.get("data_checked", False),
                    "method": "deadline_calculation_current",
                }

    # If we're here, we're past all deadlines - return the last event
    if sorted_events:
        last_event = sorted_events[-1]
        return {
            "id": last_event["id"],
            "name": last_event["name"],
            "is_current": False,
            "is_next": False,
            "is_past": True,
            "deadline_time": last_event.get("deadline_time"),
            "finished": last_event.get("finished", False),
            "data_checked": last_event.get("data_checked", False),
            "method": "deadline_calculation_past",
        }

    # No events found
    return {
        "id": None,
        "name": "No gameweeks found",
        "is_current": False,
        "is_next": False,
        "is_past": False,
        "deadline_time": None,
        "finished": False,
        "data_checked": False,
        "method": "no_events",
    }


def format_gameweek_info(
    gameweek: dict[str, Any],
    reference_time: datetime,
    local_tz: ZoneInfo = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """
    Format gameweek information with timezone conversions.

    Args:
        gameweek: Raw gameweek data
        reference_time: Reference datetime used for calculation
        local_tz: Local timezone for display

    Returns:
        Formatted gameweek information
    """
    result = {
        "id": gameweek["id"],
        "name": gameweek["name"],
        "is_current": gameweek["is_current"],
        "is_next": gameweek["is_next"],
        "is_past": gameweek["is_past"],
        "finished": gameweek["finished"],
        "data_checked": gameweek["data_checked"],
        "calculation_method": gameweek["method"],
        "now_utc": reference_time.astimezone(FPL_TIMEZONE).isoformat(),
        "now_local": reference_time.astimezone(local_tz).isoformat(),
    }

    # Add deadline information if available
    if gameweek["deadline_time"]:
        deadline_utc = parse_fpl_datetime(gameweek["deadline_time"])
        result["deadline_time_utc"] = deadline_utc.isoformat()
        result["deadline_time_local"] = deadline_utc.astimezone(local_tz).isoformat()

        # Calculate time until/since deadline
        time_diff = deadline_utc - reference_time.astimezone(FPL_TIMEZONE)
        if time_diff.total_seconds() > 0:
            result["time_until_deadline"] = str(time_diff)
            result["deadline_passed"] = False
        else:
            result["time_since_deadline"] = str(abs(time_diff))
            result["deadline_passed"] = True
    else:
        result["deadline_time_utc"] = None
        result["deadline_time_local"] = None
        result["deadline_passed"] = None

    return result


async def get_current_gameweek(args: argparse.Namespace) -> None:
    """
    Main function to fetch and display current gameweek information.

    Args:
        args: Parsed command line arguments
    """
    fpl, session = await create_fpl_session()

    try:
        # Parse reference time
        reference_time = parse_date_input(
            args.date, args.datetime, args.utc, DEFAULT_TIMEZONE
        )

        # Fetch bootstrap data
        print("Fetching gameweek data from FPL API...")
        bootstrap = await get_bootstrap_data(fpl)
        events = bootstrap.get("events", [])

        if not events:
            print("No gameweek data found in FPL API.")
            sys.exit(1)

        # Find current gameweek
        current_gameweek = find_current_gameweek(events, reference_time)

        # Format output
        gameweek_info = format_gameweek_info(
            current_gameweek, reference_time, DEFAULT_TIMEZONE
        )

        # Print friendly summary
        if gameweek_info["is_current"]:
            status = "CURRENT"
        elif gameweek_info["is_next"]:
            status = "UPCOMING"
        elif gameweek_info["is_past"]:
            status = "FINISHED"
        else:
            status = "UNKNOWN"

        print(f"Gameweek {gameweek_info['id']}: {gameweek_info['name']} [{status}]")

        if gameweek_info["deadline_time_local"]:
            deadline_local = datetime.fromisoformat(
                gameweek_info["deadline_time_local"]
            )
            print(
                f"Deadline: {deadline_local.strftime('%A, %B %d, %Y at %I:%M %p %Z')}"
            )

            if gameweek_info["deadline_passed"] is False:
                print(f"Time until deadline: {gameweek_info['time_until_deadline']}")
            elif gameweek_info["deadline_passed"] is True:
                print(f"Time since deadline: {gameweek_info['time_since_deadline']}")

        print(f"Calculation method: {gameweek_info['calculation_method']}")

        # Write JSON output if requested
        if args.out:
            output_file = Path(args.out)
            write_json(gameweek_info, output_file)
        else:
            print("\nFull details:")
            print(json.dumps(gameweek_info, indent=2, default=str))

    except KeyboardInterrupt:
        handle_keyboard_interrupt()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await safe_close_session(session)


def main() -> None:
    """
    Command-line interface for getting current gameweek information.
    """
    parser = argparse.ArgumentParser(
        description="Get current FPL gameweek information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --date 2025-08-24
  %(prog)s --datetime "2025-08-24T15:30"
  %(prog)s --datetime "2025-08-24T15:30" --utc --out gameweek.json
        """,
    )

    # Date/time options (mutually exclusive)
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--date",
        type=str,
        help="Date to check in YYYY-MM-DD format (uses midnight local time)",
    )
    time_group.add_argument(
        "--datetime", type=str, help="Datetime to check in YYYY-MM-DDTHH:MM format"
    )

    # Timezone options
    parser.add_argument(
        "--utc",
        action="store_true",
        help="Interpret supplied date/datetime as UTC (default: America/Los_Angeles)",
    )

    # Output options
    parser.add_argument("--out", type=str, help="Output JSON file path")

    args = parser.parse_args()

    # Validate datetime format
    if args.datetime and "T" not in args.datetime:
        print("Error: --datetime must be in YYYY-MM-DDTHH:MM format", file=sys.stderr)
        sys.exit(1)

    # Run async main
    try:
        asyncio.run(get_current_gameweek(args))
    except KeyboardInterrupt:
        handle_keyboard_interrupt()


if __name__ == "__main__":
    main()
