"""Retrieve structured information about an FPL team."""

from __future__ import annotations

from typing import Any

from .utils import (
    create_fpl_session,
    get_bootstrap_data,
    map_position,
    normalize_price,
    safe_close_session,
)


async def _get_current_picks(user, bootstrap_data: dict[str, Any]) -> list[dict[str, Any]]:
    picks_by_gameweek = await user.get_picks()
    current_event = user.current_event
    current_picks = picks_by_gameweek.get(current_event, [])
    if not current_picks:
        return []

    elements = {player["id"]: player for player in bootstrap_data.get("elements", [])}
    teams = {team["id"]: team for team in bootstrap_data.get("teams", [])}

    detailed = []
    for pick in current_picks:
        element_id = pick.get("element")
        player = elements.get(element_id, {})
        team = teams.get(player.get("team"), {})
        detailed.append(
            {
                "position": pick.get("position"),
                "element_id": element_id,
                "web_name": player.get("web_name", "Unknown"),
                "team_name": team.get("name", "Unknown"),
                "player_position": map_position(player.get("element_type", 0)),
                "price": normalize_price(player.get("now_cost", 0)),
                "total_points": player.get("total_points", 0),
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
                "multiplier": pick.get("multiplier", 1),
                "selling_price": normalize_price(pick.get("selling_price", 0)),
            }
        )
    detailed.sort(key=lambda item: item.get("position", 0))
    return detailed


async def _get_team_value(user) -> dict[str, Any]:
    return {
        "team_value": normalize_price(user.last_deadline_value * 10),
        "bank_balance": normalize_price(user.last_deadline_bank * 10),
        "total_value": normalize_price(
            (user.last_deadline_value + user.last_deadline_bank) * 10
        ),
        "total_transfers": user.last_deadline_total_transfers,
    }


def _format_summary(user) -> dict[str, Any]:
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


async def get_my_team_info(entry_id: int) -> dict[str, Any]:
    fpl, session = await create_fpl_session()
    try:
        user = await fpl.get_user(entry_id)
        bootstrap = await get_bootstrap_data(fpl)
        team_data = {
            "summary": _format_summary(user),
            "team_value": await _get_team_value(user),
            "current_picks": await _get_current_picks(user, bootstrap),
        }
        return team_data
    finally:
        await safe_close_session(session)
