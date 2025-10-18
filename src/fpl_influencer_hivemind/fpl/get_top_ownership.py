"""Return players ordered by ownership percentage."""

from __future__ import annotations

from typing import Any

from .utils import (
    create_fpl_session,
    get_bootstrap_data,
    map_position,
    normalize_ownership,
    normalize_price,
    safe_close_session,
)


def _create_player_record(player: dict[str, Any], teams: dict[int, str]) -> dict[str, Any]:
    team_name = teams.get(player.get("team"), "Unknown")
    return {
        "id": player.get("id"),
        "web_name": player.get("web_name", ""),
        "first_name": player.get("first_name", ""),
        "second_name": player.get("second_name", ""),
        "full_name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
        "team_name": team_name,
        "position": map_position(player.get("element_type", 0)),
        "element_type": player.get("element_type", 0),
        "price": normalize_price(player.get("now_cost", 0)),
        "now_cost": player.get("now_cost", 0),
        "selected_by_percent": normalize_ownership(player.get("selected_by_percent")),
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
        "chance_of_playing_next_round": player.get("chance_of_playing_next_round", 0),
    }


async def get_top_players_by_ownership(limit: int = 150) -> list[dict[str, Any]]:
    fpl, session = await create_fpl_session()
    try:
        bootstrap = await get_bootstrap_data(fpl)
        teams = {team.get("id"): team.get("name") for team in bootstrap.get("teams", [])}
        players = bootstrap.get("elements", [])
        records = [_create_player_record(player, teams) for player in players]
        records.sort(
            key=lambda record: (
                record.get("selected_by_percent", 0.0),
                record.get("total_points", 0),
            ),
            reverse=True,
        )
        return records[:limit]
    finally:
        await safe_close_session(session)
