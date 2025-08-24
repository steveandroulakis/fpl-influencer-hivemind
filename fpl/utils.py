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
Shared utilities for FPL data analysis scripts.

Provides async HTTP session management, FPL API data fetching/caching,
position mapping, team resolution, formatting helpers, and timezone handling.

Designed for eventual migration to src/fpl_influencer_hivemind/fpl_tools/
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
from dateutil.parser import parse as parse_datetime

from fpl import FPL

# Constants
POSITION_MAPPING = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DEFAULT_TIMEZONE = ZoneInfo("America/Los_Angeles")
FPL_TIMEZONE = ZoneInfo("UTC")

# Cache for bootstrap data to avoid repeated API calls
_bootstrap_cache: dict[str, Any] | None = None


async def create_fpl_session() -> tuple[FPL, aiohttp.ClientSession]:
    """
    Create an async FPL session with aiohttp client.

    Returns:
        Tuple of (FPL instance, ClientSession) for managing the connection lifecycle
    """
    session = aiohttp.ClientSession()
    fpl = FPL(session)
    return fpl, session


async def get_bootstrap_data(fpl: FPL, force_refresh: bool = False) -> dict[str, Any]:
    """
    Fetch and cache FPL bootstrap-static data.

    Args:
        fpl: FPL instance
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        Bootstrap data dictionary containing elements, teams, events, etc.
    """
    global _bootstrap_cache

    if _bootstrap_cache is None or force_refresh:
        try:
            # Use the actual API methods available in this version
            teams = await fpl.get_teams()
            players = await fpl.get_players()
            gameweeks = await fpl.get_gameweeks()

            # Convert to bootstrap-like format
            _bootstrap_cache = {
                "teams": [team.__dict__ for team in teams],
                "elements": [player.__dict__ for player in players],
                "events": [gw.__dict__ for gw in gameweeks],
            }
        except Exception as e:
            print(f"Error fetching bootstrap data: {e}", file=sys.stderr)
            sys.exit(1)

    return _bootstrap_cache


def map_position(element_type: int) -> str:
    """
    Map FPL element_type to human-readable position.

    Args:
        element_type: FPL position ID (1-4)

    Returns:
        Position string (GKP, DEF, MID, FWD)
    """
    return POSITION_MAPPING.get(element_type, "UNK")


def normalize_price(now_cost: int) -> float:
    """
    Convert FPL price from integer (55) to float (5.5).

    Args:
        now_cost: Raw price from FPL API

    Returns:
        Price as float with 1 decimal place
    """
    return round(now_cost / 10.0, 1)


def normalize_ownership(selected_by_percent: str | float) -> float:
    """
    Convert ownership percentage to float for sorting.

    Args:
        selected_by_percent: Ownership as string or float

    Returns:
        Ownership as float
    """
    if isinstance(selected_by_percent, str):
        try:
            return float(selected_by_percent)
        except (ValueError, TypeError):
            return 0.0
    return float(selected_by_percent) if selected_by_percent is not None else 0.0


async def find_team_by_name_or_id(
    teams: list[dict[str, Any]], team_identifier: str | int
) -> dict[str, Any] | None:
    """
    Find team by ID or name with fuzzy matching.

    Args:
        teams: List of team dictionaries from bootstrap data
        team_identifier: Team ID (int) or name (str)

    Returns:
        Team dictionary if found, None otherwise
    """
    # If it's an integer or numeric string, treat as team ID
    if isinstance(team_identifier, int) or (
        isinstance(team_identifier, str) and team_identifier.isdigit()
    ):
        team_id = int(team_identifier)
        for team in teams:
            if team["id"] == team_id:
                return team
        return None

    # String-based team name matching (case-insensitive)
    team_name = str(team_identifier).lower().strip()

    # Exact match first
    for team in teams:
        if team["name"].lower() == team_name:
            return team

    # Startswith match
    startswith_matches = [
        team for team in teams if team["name"].lower().startswith(team_name)
    ]
    if len(startswith_matches) == 1:
        return startswith_matches[0]

    # Contains match
    contains_matches = [team for team in teams if team_name in team["name"].lower()]
    if len(contains_matches) == 1:
        return contains_matches[0]

    # Multiple matches - let the caller handle this
    if startswith_matches or contains_matches:
        matches = startswith_matches if startswith_matches else contains_matches
        print(f"Multiple teams match '{team_identifier}':", file=sys.stderr)
        for i, team in enumerate(matches, 1):
            print(f"  {i}. {team['name']} (ID: {team['id']})", file=sys.stderr)
        print("Please rerun with exact team name or ID.", file=sys.stderr)
        return None

    return None


def utc_to_local(utc_dt: datetime, local_tz: ZoneInfo = DEFAULT_TIMEZONE) -> datetime:
    """
    Convert UTC datetime to local timezone.

    Args:
        utc_dt: UTC datetime (should be timezone-aware)
        local_tz: Target timezone (default: America/Los_Angeles)

    Returns:
        Datetime in local timezone
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=FPL_TIMEZONE)
    return utc_dt.astimezone(local_tz)


def local_to_utc(local_dt: datetime, local_tz: ZoneInfo = DEFAULT_TIMEZONE) -> datetime:
    """
    Convert local datetime to UTC.

    Args:
        local_dt: Local datetime (naive or timezone-aware)
        local_tz: Source timezone if datetime is naive

    Returns:
        UTC datetime
    """
    if local_dt.tzinfo is None:
        local_dt = local_dt.replace(tzinfo=local_tz)
    return local_dt.astimezone(FPL_TIMEZONE)


def parse_fpl_datetime(dt_string: str) -> datetime:
    """
    Parse FPL API datetime string to timezone-aware datetime.

    Args:
        dt_string: ISO datetime string from FPL API

    Returns:
        Timezone-aware UTC datetime
    """
    dt = parse_datetime(dt_string)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=FPL_TIMEZONE)
    return dt


def format_table_row(
    data: list[str], widths: list[int], alignment: list[str] | None = None
) -> str:
    """
    Format a single table row with fixed column widths.

    Args:
        data: List of cell values as strings
        widths: List of column widths
        alignment: List of alignment specs ('l', 'r', 'c') or None for left

    Returns:
        Formatted row string
    """
    if alignment is None:
        alignment = ["l"] * len(data)

    cells = []
    for value, width, align in zip(data, widths, alignment, strict=False):
        # Truncate if too long
        if len(value) > width:
            value = value[: width - 3] + "..."

        if align == "r":
            cells.append(value.rjust(width))
        elif align == "c":
            cells.append(value.center(width))
        else:  # 'l' or default
            cells.append(value.ljust(width))

    return " | ".join(cells)


def print_table(
    data: list[dict[str, Any]],
    headers: list[str],
    field_order: list[str] | None = None,
    alignments: dict[str, str] | None = None,
) -> None:
    """
    Print data as a formatted table to stdout.

    Args:
        data: List of row dictionaries
        headers: List of column headers
        field_order: Order of fields to display (uses headers if None)
        alignments: Dict mapping field names to alignment ('l', 'r', 'c')
    """
    if not data:
        print("No data to display.")
        return

    if field_order is None:
        field_order = headers

    if alignments is None:
        alignments = {}

    # Calculate column widths
    widths = []
    for field in field_order:
        header_width = len(field)
        if data:
            max_data_width = max(len(str(row.get(field, ""))) for row in data)
            widths.append(max(header_width, max_data_width, 8))  # Min width of 8
        else:
            widths.append(header_width)

    # Print header
    alignment_list = [alignments.get(field, "l") for field in field_order]
    print(format_table_row(field_order, widths, alignment_list))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    # Print data rows
    for row in data:
        row_data = [str(row.get(field, "")) for field in field_order]
        print(format_table_row(row_data, widths, alignment_list))

    print(f"\nTotal: {len(data)} rows")


def write_csv(
    data: list[dict[str, Any]], filepath: Path, field_order: list[str] | None = None
) -> None:
    """
    Write data to CSV file.

    Args:
        data: List of row dictionaries
        filepath: Path to output CSV file
        field_order: Order of fields in CSV (uses dict keys if None)
    """
    if not data:
        print("No data to write to CSV.")
        return

    if field_order is None:
        field_order = list(data[0].keys())

    try:
        with filepath.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(data)
        print(f"CSV written to: {filepath}")
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)


def write_json(data: list[dict[str, Any]] | dict[str, Any], filepath: Path) -> None:
    """
    Write data to JSON file.

    Args:
        data: Data to write (list of dicts or single dict)
        filepath: Path to output JSON file
    """
    try:
        with filepath.open("w", encoding="utf-8") as jsonfile:
            json.dump(data, jsonfile, indent=2, default=str, ensure_ascii=False)
        print(f"JSON written to: {filepath}")
    except Exception as e:
        print(f"Error writing JSON: {e}", file=sys.stderr)
        sys.exit(1)


async def safe_close_session(session: aiohttp.ClientSession) -> None:
    """
    Safely close aiohttp session with error handling.

    Args:
        session: ClientSession to close
    """
    try:
        await session.close()
    except Exception as e:
        print(f"Warning: Error closing session: {e}", file=sys.stderr)


def handle_keyboard_interrupt() -> None:
    """
    Handle Ctrl+C gracefully.
    """
    print("\n\nOperation cancelled by user.", file=sys.stderr)
    sys.exit(130)


if __name__ == "__main__":
    print(
        "This module provides utilities for FPL scripts. Run the individual scripts instead."
    )
    sys.exit(1)
