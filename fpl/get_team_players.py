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
List all players for a given Premier League club team ID (1-20).

Fetches player data from the FPL API and displays comprehensive stats including
position, price, points, form, and availability status. Supports multiple output
formats and flexible sorting options.

Usage:
    python get_team_players.py --team "Arsenal"
    python get_team_players.py --team-id 1 --format json --out arsenal.json
    uv run get_team_players.py --team "Liverpool" --sort total_points --desc
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from utils import (
    create_fpl_session,
    find_team_by_name_or_id,
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
        return ["element_type", "total_points"]

    return [field.strip() for field in sort_str.split(",") if field.strip()]


def create_player_record(player: dict[str, Any], team_name: str) -> dict[str, Any]:
    """
    Create standardized player record with all relevant fields.

    Args:
        player: Raw player data from FPL API
        team_name: Name of the player's team

    Returns:
        Standardized player record dictionary
    """
    return {
        "id": player.get("id", ""),
        "web_name": player.get("web_name", ""),
        "first_name": player.get("first_name", ""),
        "second_name": player.get("second_name", ""),
        "full_name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
        "team_name": team_name,
        "position": map_position(player.get("element_type", 0)),
        "element_type": player.get("element_type", 0),
        "status": player.get("status", ""),
        "news": player.get("news", ""),
        "chance_of_playing_next_round": player.get("chance_of_playing_next_round", ""),
        "now_cost": player.get("now_cost", 0),
        "price": normalize_price(player.get("now_cost", 0)),
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
        "selected_by_percent": normalize_ownership(
            player.get("selected_by_percent", 0)
        ),
    }


async def get_team_players(team_identifier: str, args: argparse.Namespace) -> None:
    """
    Main function to fetch and display team players.

    Args:
        team_identifier: Team name or ID
        args: Parsed command line arguments
    """
    fpl, session = await create_fpl_session()

    try:
        # Fetch bootstrap data
        bootstrap = await get_bootstrap_data(fpl)
        teams = bootstrap["teams"]
        elements = bootstrap["elements"]

        # Find the team
        team = await find_team_by_name_or_id(teams, team_identifier)
        if not team:
            print(f"Team '{team_identifier}' not found.", file=sys.stderr)
            sys.exit(1)

        print(f"Fetching players for: {team['name']} (ID: {team['id']})")

        # Filter players for this team
        team_players = [
            player for player in elements if player.get("team") == team["id"]
        ]

        if not team_players:
            print(f"No players found for team {team['name']}.")
            return

        # Create standardized player records
        player_records = [
            create_player_record(player, team["name"]) for player in team_players
        ]

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
                import csv
                import io

                output = io.StringIO()
                if player_records:
                    writer = csv.DictWriter(output, fieldnames=player_records[0].keys())
                    writer.writeheader()
                    writer.writerows(player_records)
                    print(output.getvalue().strip())

        else:  # table format
            if player_records:
                # Default table fields for readability
                table_fields = [
                    "web_name",
                    "position",
                    "price",
                    "total_points",
                    "minutes",
                    "form",
                    "selected_by_percent",
                    "status",
                ]

                # Use custom fields if specified
                if args.fields:
                    table_fields = [f.strip() for f in args.fields.split(",")]

                alignments = {
                    "price": "r",
                    "total_points": "r",
                    "minutes": "r",
                    "goals_scored": "r",
                    "assists": "r",
                    "selected_by_percent": "r",
                    "form": "r",
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
    Command-line interface for getting team players.
    """
    parser = argparse.ArgumentParser(
        description="List all players for a Premier League club team",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --team "Arsenal"
  %(prog)s --team-id 1 --format json --out arsenal.json
  %(prog)s --team "Liverpool" --sort total_points --desc
  %(prog)s --team "Man City" --fields "web_name,position,price,total_points"
        """,
    )

    # Team selection (mutually exclusive)
    team_group = parser.add_mutually_exclusive_group(required=True)
    team_group.add_argument(
        "--team-id", type=int, help="Premier League club team ID (1-20)"
    )
    team_group.add_argument(
        "--team",
        type=str,
        help="Premier League club name (e.g., 'Arsenal', 'Liverpool')",
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
        default="element_type,total_points",
        help="Comma-separated fields to sort by (default: element_type,total_points)",
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

    # Determine team identifier
    team_identifier = args.team_id if args.team_id is not None else args.team

    # Run async main
    try:
        asyncio.run(get_team_players(team_identifier, args))
    except KeyboardInterrupt:
        handle_keyboard_interrupt()


if __name__ == "__main__":
    main()
