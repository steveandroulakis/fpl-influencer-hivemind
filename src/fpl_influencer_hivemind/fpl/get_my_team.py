"""Retrieve structured information about an FPL team."""

from __future__ import annotations

import logging
from typing import Any

from .utils import (
    create_authenticated_fpl_session,
    create_fpl_session,
    get_bootstrap_data,
    map_position,
    normalize_price,
    safe_close_session,
)

logger = logging.getLogger(__name__)


async def _get_current_picks_public(
    user: Any, bootstrap_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Get picks from public endpoint (no selling_price)."""
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
        # Public endpoint doesn't have selling_price, fall back to current price
        price = normalize_price(player.get("now_cost", 0))
        detailed.append(
            {
                "position": pick.get("position"),
                "element_id": element_id,
                "web_name": player.get("web_name", "Unknown"),
                "team_name": team.get("name", "Unknown"),
                "player_position": map_position(player.get("element_type", 0)),
                "price": price,
                "total_points": player.get("total_points", 0),
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
                "multiplier": pick.get("multiplier", 1),
                "selling_price": price,  # Use current price as fallback
            }
        )
    detailed.sort(key=lambda item: item.get("position", 0))
    return detailed


async def _get_current_picks_authenticated(
    user: Any, bootstrap_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Get picks from authenticated endpoint (includes real selling_price)."""
    team_data = await user.get_team()
    if not team_data:
        return []

    elements = {player["id"]: player for player in bootstrap_data.get("elements", [])}
    teams = {team["id"]: team for team in bootstrap_data.get("teams", [])}

    # Also get picks to know captain/vice-captain status
    picks_by_gameweek = await user.get_picks()
    current_event = user.current_event
    current_picks = picks_by_gameweek.get(current_event, [])
    picks_by_element: dict[int, dict[str, Any]] = {
        p.get("element"): p for p in current_picks
    }

    detailed = []
    for idx, team_player in enumerate(team_data, 1):
        element_id = team_player.get("element")
        player = elements.get(element_id, {})
        team = teams.get(player.get("team"), {})
        pick_info = picks_by_element.get(element_id, {})

        detailed.append(
            {
                "position": pick_info.get("position", idx),
                "element_id": element_id,
                "web_name": player.get("web_name", "Unknown"),
                "team_name": team.get("name", "Unknown"),
                "player_position": map_position(player.get("element_type", 0)),
                "price": normalize_price(player.get("now_cost", 0)),
                "total_points": player.get("total_points", 0),
                "is_captain": pick_info.get("is_captain", False),
                "is_vice_captain": pick_info.get("is_vice_captain", False),
                "multiplier": pick_info.get("multiplier", 1),
                "selling_price": normalize_price(team_player.get("selling_price", 0)),
            }
        )
    detailed.sort(key=lambda item: item.get("position", 0))
    return detailed


async def _get_team_value(user: Any) -> dict[str, Any]:
    # FPL API returns values in tenths of millions (e.g., 1024 = 102.4m)
    # normalize_price divides by 10 to get actual millions
    return {
        "team_value": normalize_price(user.last_deadline_value),
        "bank_balance": normalize_price(user.last_deadline_bank),
        "total_value": normalize_price(
            user.last_deadline_value + user.last_deadline_bank
        ),
        "total_transfers": user.last_deadline_total_transfers,
    }


def _format_summary(user: Any) -> dict[str, Any]:
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
    """Get team info, using authenticated endpoint if credentials available.

    If FPL_EMAIL and FPL_PASSWORD are set, uses authenticated endpoint
    which provides accurate selling_price. Otherwise falls back to public
    endpoint where selling_price equals current price.
    """
    # Try authenticated endpoint first for accurate selling prices
    try:
        fpl, session = await create_authenticated_fpl_session()
        try:
            user = await fpl.get_user(entry_id)
            bootstrap = await get_bootstrap_data(fpl)
            team_data = {
                "summary": _format_summary(user),
                "team_value": await _get_team_value(user),
                "current_picks": await _get_current_picks_authenticated(user, bootstrap),
            }
            logger.info("Using authenticated FPL endpoint (accurate selling prices)")
            return team_data
        finally:
            await safe_close_session(session)
    except ValueError as e:
        # Auth credentials not configured, fall back to public endpoint
        logger.warning(f"FPL auth not configured: {e}. Using public endpoint.")
    except Exception as e:
        # Auth failed, fall back to public endpoint
        logger.warning(f"FPL auth failed: {e}. Falling back to public endpoint.")

    # Fall back to public endpoint (selling_price = current price)
    fpl, session = await create_fpl_session()
    try:
        user = await fpl.get_user(entry_id)
        bootstrap = await get_bootstrap_data(fpl)
        team_data = {
            "summary": _format_summary(user),
            "team_value": await _get_team_value(user),
            "current_picks": await _get_current_picks_public(user, bootstrap),
        }
        logger.info("Using public FPL endpoint (selling_price = current price)")
        return team_data
    finally:
        await safe_close_session(session)
