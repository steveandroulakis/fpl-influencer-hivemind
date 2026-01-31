"""Retrieve structured information about an FPL team."""

from __future__ import annotations

import logging
from typing import Any

from .utils import (
    create_authenticated_fpl_session,
    create_fpl_session,
    create_token_authenticated_fpl_session,
    fetch_my_team_direct,
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
        team = teams.get(player.get("team", 0), {})
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
        team = teams.get(player.get("team", 0), {})
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


async def _get_current_picks_with_direct_fetch(
    user: Any,
    bootstrap_data: dict[str, Any],
    session: Any,
    team_id: int,
) -> list[dict[str, Any]]:
    """Get picks using direct API fetch for selling_price (token auth)."""
    team_data = await fetch_my_team_direct(session, team_id)
    if not team_data:
        return []

    elements = {player["id"]: player for player in bootstrap_data.get("elements", [])}
    teams = {team["id"]: team for team in bootstrap_data.get("teams", [])}

    # Get picks for captain/vice-captain status
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
        team = teams.get(player.get("team", 0), {})
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

    Auth priority:
    1. FPL_EMAIL + FPL_PASSWORD (standard login)
    2. FPL_BEARER_TOKEN (token fallback from browser)
    3. Public endpoint (no auth, selling_price = current price)
    """
    # Try email/password auth first
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
        # Auth credentials not configured
        logger.warning(f"FPL email/password not configured: {e}")
    except Exception as e:
        # Auth failed (likely DataDome blocking)
        logger.warning(f"FPL email/password auth failed: {e}")

    # Try token-based auth as fallback
    try:
        fpl, session = await create_token_authenticated_fpl_session()
        try:
            user = await fpl.get_user(entry_id)
            bootstrap = await get_bootstrap_data(fpl)
            team_data = {
                "summary": _format_summary(user),
                "team_value": await _get_team_value(user),
                "current_picks": await _get_current_picks_with_direct_fetch(
                    user, bootstrap, session, entry_id
                ),
            }
            logger.info("Using token-authenticated FPL endpoint (accurate selling prices)")
            return team_data
        finally:
            await safe_close_session(session)
    except ValueError as e:
        # Token not configured
        logger.warning(f"FPL token not configured: {e}")
    except PermissionError as e:
        # Token expired/invalid
        logger.warning(f"FPL token auth failed: {e}. Token may have expired.")
    except Exception as e:
        # Other token auth failure
        logger.warning(f"FPL token auth failed: {e}")

    # Fall back to public endpoint (selling_price = current price)
    logger.info("Falling back to public FPL endpoint (selling_price = current price)")
    fpl, session = await create_fpl_session()
    try:
        user = await fpl.get_user(entry_id)
        bootstrap = await get_bootstrap_data(fpl)
        team_data = {
            "summary": _format_summary(user),
            "team_value": await _get_team_value(user),
            "current_picks": await _get_current_picks_public(user, bootstrap),
        }
        return team_data
    finally:
        await safe_close_session(session)
