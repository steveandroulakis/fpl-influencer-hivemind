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
Get comprehensive information about your FPL team.

Fetches your team details, current squad, gameweek history, transfers, and performance stats.
Uses your FPL entry ID to access all available team data.

Usage:
    python get_my_team.py --entry-id 1178124
    python get_my_team.py --entry-id 1178124 --format json --out my_team.json
    uv run get_my_team.py --entry-id 1178124 --show picks,transfers,history
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
    normalize_price,
    print_table,
    safe_close_session,
    write_csv,
    write_json,
)


async def get_current_picks(
    user, bootstrap_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Get current gameweek picks with player details.

    Args:
        user: FPL User object
        bootstrap_data: Bootstrap data for player lookups

    Returns:
        List of current picks with player information
    """
    try:
        picks_by_gw = await user.get_picks()
        current_gw = user.current_event

        # Get picks for current gameweek
        current_picks = picks_by_gw.get(current_gw, [])
        if not current_picks:
            print(f"No picks found for gameweek {current_gw}")
            return []

        elements = {p["id"]: p for p in bootstrap_data["elements"]}
        teams = {t["id"]: t for t in bootstrap_data["teams"]}

        detailed_picks = []
        for pick in current_picks:
            element_id = pick["element"]
            player = elements.get(element_id, {})
            team = teams.get(player.get("team"), {})

            detailed_pick = {
                "position": pick["position"],
                "element_id": element_id,
                "web_name": player.get("web_name", "Unknown"),
                "team_name": team.get("name", "Unknown"),
                "player_position": map_position(player.get("element_type", 0)),
                "price": normalize_price(player.get("now_cost", 0)),
                "total_points": player.get("total_points", 0),
                "is_captain": pick["is_captain"],
                "is_vice_captain": pick["is_vice_captain"],
                "multiplier": pick["multiplier"],
                "selling_price": normalize_price(pick.get("selling_price", 0)),
            }
            detailed_picks.append(detailed_pick)

        # Sort by position
        detailed_picks.sort(key=lambda x: x["position"])
        return detailed_picks
    except Exception as e:
        print(f"Could not get current picks: {e}")
        return []


async def get_gameweek_history(user) -> list[dict[str, Any]]:
    """
    Get gameweek-by-gameweek history.

    Args:
        user: FPL User object

    Returns:
        List of gameweek history data
    """
    try:
        history = await user.get_gameweek_history()
        return [gw.__dict__ if hasattr(gw, "__dict__") else gw for gw in history]
    except Exception as e:
        print(f"Could not get gameweek history: {e}")
        return []


async def get_recent_transfers(user, limit: int = 10) -> list[dict[str, Any]]:
    """
    Get recent transfers.

    Args:
        user: FPL User object
        limit: Maximum number of transfers to return

    Returns:
        List of recent transfer data
    """
    try:
        transfers = await user.get_transfers()
        # Get the most recent transfers
        recent_transfers = transfers[-limit:] if len(transfers) > limit else transfers
        return [
            transfer.__dict__ if hasattr(transfer, "__dict__") else transfer
            for transfer in recent_transfers
        ]
    except Exception as e:
        print(f"Could not get transfers: {e}")
        return []


async def get_team_value_and_bank(user) -> dict[str, Any]:
    """
    Get current team value and bank balance.

    Args:
        user: FPL User object

    Returns:
        Dictionary with team value information
    """
    try:
        return {
            "team_value": normalize_price(
                user.last_deadline_value * 10
            ),  # Convert to standard price format
            "bank_balance": normalize_price(user.last_deadline_bank * 10),
            "total_value": normalize_price(
                (user.last_deadline_value + user.last_deadline_bank) * 10
            ),
            "total_transfers": user.last_deadline_total_transfers,
        }
    except Exception as e:
        print(f"Could not get team value data: {e}")
        return {}


def format_team_summary(user) -> dict[str, Any]:
    """
    Format basic team summary information.

    Args:
        user: FPL User object

    Returns:
        Dictionary with team summary
    """
    return {
        "entry_id": user.id,
        "manager_name": f"{user.player_first_name} {user.player_last_name}",
        "team_name": user.name,
        "total_points": user.summary_overall_points,
        "overall_rank": user.summary_overall_rank,
        "gameweek_points": user.summary_event_points,
        "gameweek_rank": user.summary_event_rank,
        "favourite_team": user.favourite_team,
        "started_event": user.started_event,
        "current_event": user.current_event,
    }


async def get_my_team_info(entry_id: int, args: argparse.Namespace) -> None:
    """
    Main function to fetch and display team information.

    Args:
        entry_id: FPL entry ID
        args: Parsed command line arguments
    """
    fpl, session = await create_fpl_session()

    try:
        print(f"Fetching team data for entry ID: {entry_id}")

        # Get user data
        user = await fpl.get_user(entry_id)

        # Get bootstrap data for player/team lookups
        bootstrap_data = await get_bootstrap_data(fpl)

        # Collect all requested information
        team_data = {
            "summary": format_team_summary(user),
            "team_value": await get_team_value_and_bank(user),
        }

        show_sections = args.show.split(",") if args.show else ["summary", "picks"]

        if "picks" in show_sections:
            print("Fetching current picks...")
            team_data["current_picks"] = await get_current_picks(user, bootstrap_data)

        if "history" in show_sections:
            print("Fetching gameweek history...")
            team_data["gameweek_history"] = await get_gameweek_history(user)

        if "transfers" in show_sections:
            print("Fetching recent transfers...")
            team_data["recent_transfers"] = await get_recent_transfers(
                user, args.transfer_limit
            )

        # Output results
        output_file = Path(args.out) if args.out else None

        if args.format == "json" or (
            output_file and output_file.suffix.lower() == ".json"
        ):
            if output_file:
                write_json(team_data, output_file)
            else:
                print(json.dumps(team_data, indent=2, default=str))

        elif args.format == "csv" and "picks" in show_sections:
            # CSV output only makes sense for picks data
            if output_file:
                write_csv(team_data.get("current_picks", []), output_file)
            else:
                if team_data.get("current_picks"):
                    output = io.StringIO()
                    writer = csv.DictWriter(
                        output, fieldnames=team_data["current_picks"][0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(team_data["current_picks"])
                    print(output.getvalue().strip())

        else:  # table format
            # Print team summary
            print(f"\n=== {team_data['summary']['team_name']} ===")
            print(f"Manager: {team_data['summary']['manager_name']}")
            print(f"Total Points: {team_data['summary']['total_points']:,}")
            print(f"Overall Rank: {team_data['summary']['overall_rank']:,}")
            print(f"Gameweek Points: {team_data['summary']['gameweek_points']}")
            print(f"Gameweek Rank: {team_data['summary']['gameweek_rank']:,}")

            if team_data["team_value"]:
                print(f"Team Value: £{team_data['team_value']['team_value']}m")
                print(f"Bank: £{team_data['team_value']['bank_balance']}m")
                print(f"Total Value: £{team_data['team_value']['total_value']}m")

            # Print current picks if requested
            if "picks" in show_sections and team_data.get("current_picks"):
                print(
                    f"\n=== Current Squad (GW {team_data['summary']['current_event']}) ==="
                )

                # Split into starting XI and bench
                starting_xi = [
                    p for p in team_data["current_picks"] if p["position"] <= 11
                ]
                bench = [p for p in team_data["current_picks"] if p["position"] > 11]

                print("\nStarting XI:")
                table_fields = [
                    "position",
                    "web_name",
                    "team_name",
                    "player_position",
                    "price",
                    "total_points",
                ]
                alignments = {"position": "r", "price": "r", "total_points": "r"}

                # Add captain/vice-captain indicators
                for pick in starting_xi:
                    if pick["is_captain"]:
                        pick["web_name"] = pick["web_name"] + " (C)"
                    elif pick["is_vice_captain"]:
                        pick["web_name"] = pick["web_name"] + " (VC)"

                print_table(starting_xi, table_fields, table_fields, alignments)

                print("\nBench:")
                print_table(bench, table_fields, table_fields, alignments)

            # Print gameweek history if requested
            if "history" in show_sections and team_data.get("gameweek_history"):
                print("\n=== Recent Gameweek History ===")
                recent_history = team_data["gameweek_history"][-5:]  # Last 5 gameweeks

                if recent_history:
                    history_fields = [
                        "event",
                        "points",
                        "total_points",
                        "rank",
                        "overall_rank",
                        "event_transfers",
                        "event_transfers_cost",
                    ]
                    history_alignments = dict.fromkeys(history_fields, "r")
                    print_table(
                        recent_history,
                        history_fields,
                        history_fields,
                        history_alignments,
                    )

            # Print recent transfers if requested
            if "transfers" in show_sections and team_data.get("recent_transfers"):
                print("\n=== Recent Transfers ===")
                if team_data["recent_transfers"]:
                    transfer_fields = [
                        "event",
                        "element_in",
                        "element_out",
                        "entry",
                        "time",
                    ]
                    print_table(
                        team_data["recent_transfers"],
                        transfer_fields,
                        transfer_fields,
                        {},
                    )
                else:
                    print("No recent transfers found.")

    except Exception as e:
        print(f"Error fetching team data: {e}")
        if "not found" in str(e).lower():
            print(f"Entry ID {entry_id} may not exist or may be private.")
        sys.exit(1)
    finally:
        await safe_close_session(session)


def main() -> None:
    """
    Command-line interface for getting team information.
    """
    parser = argparse.ArgumentParser(
        description="Get comprehensive FPL team information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --entry-id 1178124
  %(prog)s --entry-id 1178124 --show "summary,picks,history,transfers"
  %(prog)s --entry-id 1178124 --format json --out my_team.json
  %(prog)s --entry-id 1178124 --show picks --format csv --out current_squad.csv
        """,
    )

    # Required entry ID
    parser.add_argument(
        "--entry-id", type=int, required=True, help="FPL entry (team) ID"
    )

    # What information to show
    parser.add_argument(
        "--show",
        type=str,
        default="summary,picks",
        help="Comma-separated sections to display: summary,picks,history,transfers (default: summary,picks)",
    )

    # Transfer history limit
    parser.add_argument(
        "--transfer-limit",
        type=int,
        default=10,
        help="Number of recent transfers to show (default: 10)",
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

    args = parser.parse_args()

    # Validate entry ID
    if args.entry_id <= 0:
        print("Error: Entry ID must be a positive integer", file=sys.stderr)
        sys.exit(1)

    # Run async main
    try:
        asyncio.run(get_my_team_info(args.entry_id, args))
    except KeyboardInterrupt:
        handle_keyboard_interrupt()


if __name__ == "__main__":
    main()
