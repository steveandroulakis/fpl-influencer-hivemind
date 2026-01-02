"""Return players ranked by transfer activity (in/out/net) for current gameweek."""

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


def _create_transfer_record(
    player: dict[str, Any], teams: dict[int, str]
) -> dict[str, Any]:
    """Build a slim player record with transfer momentum fields."""
    team_id: int | None = player.get("team")
    team_name = teams.get(team_id, "Unknown") if team_id is not None else "Unknown"
    transfers_in = player.get("transfers_in_event", 0) or 0
    transfers_out = player.get("transfers_out_event", 0) or 0
    return {
        "web_name": player.get("web_name", ""),
        "team_name": team_name,
        "position": map_position(player.get("element_type", 0)),
        "price": normalize_price(player.get("now_cost", 0)),
        "selected_by_percent": normalize_ownership(player.get("selected_by_percent")),
        "transfers_in_event": transfers_in,
        "transfers_out_event": transfers_out,
        "net_transfers": transfers_in - transfers_out,
    }


async def get_transfer_momentum(
    limit: int = 10, min_net: int = 1000
) -> dict[str, list[dict[str, Any]]]:
    """Return top players by transfer momentum metrics.

    Args:
        limit: Number of players to return per category.
        min_net: Minimum absolute net transfers to include (filters noise).

    Returns:
        Dict with three keys:
        - top_transfers_in: Top N by transfers_in_event (descending)
        - top_transfers_out: Top N by transfers_out_event (descending)
        - top_net_transfers: Top N by net_transfers (descending, positive = in demand)
    """
    fpl, session = await create_fpl_session()
    try:
        bootstrap = await get_bootstrap_data(fpl)
        teams: dict[int, str] = {
            t.get("id"): t.get("name")
            for t in bootstrap.get("teams", [])
            if t.get("id") is not None
        }
        players = bootstrap.get("elements", [])

        records = [_create_transfer_record(p, teams) for p in players]

        # Filter by minimum net transfer threshold
        filtered = [r for r in records if abs(r["net_transfers"]) >= min_net]

        # Sort for top transfers IN
        top_in = sorted(filtered, key=lambda r: r["transfers_in_event"], reverse=True)[
            :limit
        ]

        # Sort for top transfers OUT
        top_out = sorted(
            filtered, key=lambda r: r["transfers_out_event"], reverse=True
        )[:limit]

        # Sort for top NET transfers (positive = in demand)
        top_net = sorted(filtered, key=lambda r: r["net_transfers"], reverse=True)[
            :limit
        ]

        return {
            "top_transfers_in": top_in,
            "top_transfers_out": top_out,
            "top_net_transfers": top_net,
        }
    finally:
        await safe_close_session(session)
