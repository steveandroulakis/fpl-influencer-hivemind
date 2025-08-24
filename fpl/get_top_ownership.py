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
List the top FPL players by ownership percentage.

Fetches all player data from the FPL API and ranks by selected_by_percent,
showing comprehensive stats including position, price, points, form, and ICT index.
Supports filtering by availability, minutes played, and flexible output options.

Usage:
    python get_top_ownership.py --limit 200
    python get_top_ownership.py --limit 50 --only-available --format csv
    uv run get_top_ownership.py --limit 100 --min-minutes 1000 --out top100.json
"""

import argparse
import asyncio
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from utils import (
    create_fpl_session,
    get_bootstrap_data,
    handle_keyboard_interrupt,
    map_position,
    normalize_ownership,
    normalize_price,
    print_table,
    safe_close_session,
    write_csv,
    write_json,
)


def parse_sort_fields(sort_str: str | None) -> list[str]:
    """
    Parse comma-separated sort fields.

    Args:
        sort_str: Comma-separated field names or None

    Returns:
        List of field names
    """
    if not sort_str:
        return ["selected_by_percent", "total_points"]

    return [field.strip() for field in sort_str.split(",") if field.strip()]


def create_player_record(
    player: dict[str, Any], teams: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Create standardized player record with all relevant fields.

    Args:
        player: Raw player data from FPL API
        teams: List of team data to lookup team names

    Returns:
        Standardized player record dictionary
    """
    # Find team name
    team_name = "Unknown"
    for team in teams:
        if team["id"] == player.get("team"):
            team_name = team["name"]
            break

    return {
        "id": player.get("id", ""),
        "web_name": player.get("web_name", ""),
        "first_name": player.get("first_name", ""),
        "second_name": player.get("second_name", ""),
        "full_name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
        "team_name": team_name,
        "position": map_position(player.get("element_type", 0)),
        "element_type": player.get("element_type", 0),
        "price": normalize_price(player.get("now_cost", 0)),
        "now_cost": player.get("now_cost", 0),
        "selected_by_percent": normalize_ownership(
            player.get("selected_by_percent", 0)
        ),
        "total_points": player.get("total_points", 0),
        "minutes": player.get("minutes", 0),
        "goals_scored": player.get("goals_scored", 0),
        "assists": player.get("assists", 0),
        "clean_sheets": player.get("clean_sheets", 0),
        "goals_conceded": player.get("goals_conceded", 0),
        "saves": player.get("saves", 0),
        "bonus": player.get("bonus", 0),
        "bps": player.get("bps", 0),
        "form": player.get("form", ""),
        "influence": player.get("influence", ""),
        "creativity": player.get("creativity", ""),
        "threat": player.get("threat", ""),
        "ict_index": player.get("ict_index", ""),
        "expected_points": player.get("expected_points", ""),
        "status": player.get("status", ""),
        "news": player.get("news", ""),
        "chance_of_playing_next_round": player.get("chance_of_playing_next_round", ""),
    }


def filter_players(
    players: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """
    Apply filters to player list based on command line arguments.

    Args:
        players: List of player records
        args: Parsed command line arguments

    Returns:
        Filtered list of player records
    """
    filtered = players

    # Filter by minimum minutes
    if args.min_minutes:
        filtered = [p for p in filtered if p.get("minutes", 0) >= args.min_minutes]

    # Filter by availability status
    if args.only_available:
        filtered = [p for p in filtered if p.get("status") == "a"]

    return filtered


async def get_top_ownership_players(args: argparse.Namespace) -> None:
    """
    Main function to fetch and display top ownership players.

    Args:
        args: Parsed command line arguments
    """
    fpl, session = await create_fpl_session()

    try:
        # Fetch bootstrap data
        print("Fetching player data from FPL API...")
        bootstrap = await get_bootstrap_data(fpl)
        teams = bootstrap["teams"]
        elements = bootstrap["elements"]

        print(f"Processing {len(elements)} players...")

        # Create standardized player records
        player_records = [create_player_record(player, teams) for player in elements]

        # Apply filters
        player_records = filter_players(player_records, args)

        if not player_records:
            print("No players found matching the specified criteria.")
            return

        # Sort players
        sort_fields = parse_sort_fields(args.sort)

        def sort_key(player: dict[str, Any]) -> tuple:
            return tuple(
                player.get(field, 0)
                if isinstance(player.get(field, 0), int | float)
                else str(player.get(field, "")).lower()
                for field in sort_fields
            )

        player_records.sort(key=sort_key, reverse=not args.asc)

        # Limit results
        if args.limit:
            player_records = player_records[: args.limit]

        # Filter fields if specified
        if args.fields:
            field_list = [f.strip() for f in args.fields.split(",")]
            filtered_records = []
            for record in player_records:
                filtered_record = {
                    field: record.get(field, "")
                    for field in field_list
                    if field in record
                }
                filtered_records.append(filtered_record)
            player_records = filtered_records

        print(f"Showing top {len(player_records)} players by ownership...")

        # Output results
        output_file = Path(args.out) if args.out else None

        if args.format == "json" or (
            output_file and output_file.suffix.lower() == ".json"
        ):
            if output_file:
                write_json(player_records, output_file)
            else:
                print(json.dumps(player_records, indent=2, default=str))

        elif args.format == "csv" or (
            output_file and output_file.suffix.lower() == ".csv"
        ):
            if output_file:
                write_csv(player_records, output_file)
            else:
                if player_records:
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=player_records[0].keys())
                    writer.writeheader()
                    writer.writerows(player_records)
                    print(output.getvalue().strip())

        else:  # table format
            if player_records:
                # Default table fields for readability
                table_fields = [
                    "web_name",
                    "team_name",
                    "position",
                    "price",
                    "selected_by_percent",
                    "total_points",
                    "minutes",
                    "form",
                    "ict_index",
                    "status",
                ]

                # Use custom fields if specified
                if args.fields:
                    table_fields = [f.strip() for f in args.fields.split(",")]

                alignments = {
                    "price": "r",
                    "selected_by_percent": "r",
                    "total_points": "r",
                    "minutes": "r",
                    "goals_scored": "r",
                    "assists": "r",
                    "form": "r",
                    "ict_index": "r",
                    "influence": "r",
                    "creativity": "r",
                    "threat": "r",
                }

                print_table(player_records, table_fields, table_fields, alignments)

    except KeyboardInterrupt:
        handle_keyboard_interrupt()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await safe_close_session(session)


def main() -> None:
    """
    Command-line interface for getting top ownership players.
    """
    parser = argparse.ArgumentParser(
        description="List top FPL players by ownership percentage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --limit 200
  %(prog)s --limit 50 --only-available --format csv --out top50.csv
  %(prog)s --limit 100 --min-minutes 1000 --sort "total_points,selected_by_percent"
  %(prog)s --limit 25 --fields "web_name,position,price,selected_by_percent,total_points"
        """,
    )

    # Filtering options
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of players to show (default: 200)",
    )
    parser.add_argument(
        "--min-minutes", type=int, help="Filter players with minimum minutes played"
    )
    parser.add_argument(
        "--only-available",
        action="store_true",
        help="Only show available players (status = 'a')",
    )

    # Output options
    parser.add_argument(
        "--out",
        type=str,
        help="Output file path (format determined by extension: .csv, .json)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--fields", type=str, help="Comma-separated list of fields to include in output"
    )

    # Sorting options
    parser.add_argument(
        "--sort",
        type=str,
        default="selected_by_percent,total_points",
        help="Comma-separated fields to sort by (default: selected_by_percent,total_points)",
    )
    sort_group = parser.add_mutually_exclusive_group()
    sort_group.add_argument(
        "--asc", action="store_true", help="Sort in ascending order"
    )
    sort_group.add_argument(
        "--desc",
        action="store_false",
        dest="asc",
        help="Sort in descending order (default)",
    )
    parser.set_defaults(asc=False)

    args = parser.parse_args()

    # Validate limit
    if args.limit and args.limit < 1:
        print("Error: --limit must be a positive integer", file=sys.stderr)
        sys.exit(1)

    if args.min_minutes and args.min_minutes < 0:
        print("Error: --min-minutes must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Run async main
    try:
        asyncio.run(get_top_ownership_players(args))
    except KeyboardInterrupt:
        handle_keyboard_interrupt()


if __name__ == "__main__":
    main()
