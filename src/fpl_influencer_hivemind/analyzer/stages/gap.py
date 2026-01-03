"""Gap analysis stage for the FPL Intelligence Analyzer."""

import json
import logging
from typing import Any

from pydantic import ValidationError

from src.fpl_influencer_hivemind.analyzer.api import (
    AnthropicClient,
    extract_last_json,
    save_debug_content,
)
from src.fpl_influencer_hivemind.analyzer.constants import PL_TEAMS_CONTEXT
from src.fpl_influencer_hivemind.analyzer.models import ChannelAnalysis
from src.fpl_influencer_hivemind.types import GapAnalysis

logger = logging.getLogger(__name__)


def aggregate_influencer_consensus(
    channel_analyses: list[ChannelAnalysis],
) -> dict[str, Any]:
    """Aggregate influencer opinions into consensus data."""
    captain_counts: dict[str, list[str]] = {}
    transfers_in_counts: dict[str, list[str]] = {}
    transfers_out_counts: dict[str, list[str]] = {}
    watchlist_items: list[dict[str, Any]] = []

    for analysis in channel_analyses:
        # Captains
        cap = analysis.captain_choice
        if cap and cap != "Not specified":
            captain_counts.setdefault(cap, []).append(analysis.channel_name)

        # Transfers in
        for player in analysis.transfers_in:
            transfers_in_counts.setdefault(player, []).append(analysis.channel_name)

        # Transfers out
        for player in analysis.transfers_out:
            transfers_out_counts.setdefault(player, []).append(analysis.channel_name)

        # Watchlist
        for item in analysis.watchlist:
            watchlist_items.append(
                {
                    "player": item.get("name", ""),
                    "priority": item.get("priority", ""),
                    "reason": item.get("why", ""),
                    "channel": analysis.channel_name,
                }
            )

    return {
        "captain_counts": captain_counts,
        "transfers_in_counts": transfers_in_counts,
        "transfers_out_counts": transfers_out_counts,
        "watchlist": watchlist_items,
        "total_channels": len(channel_analyses),
    }


def stage_gap_analysis(
    client: AnthropicClient,
    channel_analyses: list[ChannelAnalysis],
    squad_context: dict[str, Any],
    gameweek: int,
    transfer_momentum: dict[str, Any] | None = None,
    commentary: str | None = None,
    previous_errors: list[str] | None = None,
) -> GapAnalysis:
    """Stage 1: identify gaps between my squad and consensus."""
    logger.info("Stage 1: Gap Analysis")

    squad = squad_context["squad"]
    squad_names = {p["name"] for p in squad}
    consensus = aggregate_influencer_consensus(channel_analyses)

    # Build transfer momentum section if available
    momentum_section = ""
    if transfer_momentum:
        top_in = transfer_momentum.get("top_transfers_in", [])
        top_out = transfer_momentum.get("top_transfers_out", [])
        top_net = transfer_momentum.get("top_net_transfers", [])
        if top_in or top_out or top_net:
            in_summary = [
                {
                    "name": p.get("web_name"),
                    "team": p.get("team_name"),
                    "net": p.get("net_transfers"),
                }
                for p in top_in[:5]
            ]
            out_summary = [
                {
                    "name": p.get("web_name"),
                    "team": p.get("team_name"),
                    "net": p.get("net_transfers"),
                }
                for p in top_out[:5]
            ]
            net_summary = [
                {
                    "name": p.get("web_name"),
                    "team": p.get("team_name"),
                    "net": p.get("net_transfers"),
                }
                for p in top_net[:5]
            ]
            momentum_section = f"""
TRANSFER MOMENTUM (Real-time FPL manager activity this gameweek):
Top Transfers IN: {json.dumps(in_summary, indent=2)}
Top Transfers OUT: {json.dumps(out_summary, indent=2)}
Top Net Transfers: {json.dumps(net_summary, indent=2)}

NOTE: Flag players with >100k net transfers NOT mentioned by influencers as potential gaps.
"""

    # Build error feedback section
    error_feedback = ""
    if previous_errors:
        error_feedback = (
            "\n\nPREVIOUS ATTEMPT FAILED WITH ERRORS:\n"
            + "\n".join(f"- {e}" for e in previous_errors)
            + "\n\nFix these issues in your response.\n"
        )

    directive_section = ""
    if commentary:
        directive_section = (
            f"\nUSER DIRECTIVE (HIGH PRIORITY - FOLLOW THIS):\n{commentary}\n"
        )

    prompt = f"""Analyze gaps between my FPL squad and influencer consensus for GW{gameweek}.
{momentum_section}
{directive_section}

MY SQUAD (15 players):
{json.dumps(squad, indent=2)}

SQUAD PLAYER NAMES (do NOT recommend these as transfers IN):
{json.dumps(list(squad_names), indent=2)}

INFLUENCER CONSENSUS:
- Captain choices: {json.dumps(consensus["captain_counts"], indent=2)}
- Transfers IN recommended: {json.dumps(consensus["transfers_in_counts"], indent=2)}
- Transfers OUT recommended: {json.dumps(consensus["transfers_out_counts"], indent=2)}
- Watchlist: {json.dumps(consensus["watchlist"], indent=2)}
- Total channels analyzed: {consensus["total_channels"]}
{error_feedback}
Return JSON matching this schema EXACTLY:
{{
  "players_to_sell": [
    {{"name": "Player Name", "position": "POS", "team": "Team Name"}}
  ],
  "players_missing": [
    {{"name": "Player Name", "position": "POS", "team": "Team Name"}}
  ],
  "risk_flags": [
    {{"player": "Player Name", "risk": "Description of risk"}}
  ],
  "formation_gaps": ["Gap description"],
  "captain_gap": "Player Name or null"
}}

{PL_TEAMS_CONTEXT}

Rules:
1. players_to_sell: ONLY players in MY SQUAD that influencers are selling/benching.
2. players_missing: Popular picks I don't own; MUST NOT include any player from MY SQUAD.
3. captain_gap: If consensus captain is NOT in my squad, set this (top priority); else null.
4. risk_flags: injury/rotation/form/availability risks ONLY (no team-quality speculation).
5. formation_gaps: position imbalances or formation inflexibility.
6. Use transfer momentum as a secondary signal for missing players when relevant.

Return ONLY valid JSON, no markdown fences."""

    system = """You are an FPL analyst identifying gaps between a manager's squad and influencer recommendations.
Use only the provided data (no external knowledge). Return valid JSON only."""

    save_debug_content("stage1_gap_analysis_prompt.txt", prompt)

    response, _ = client.call_opus(prompt=prompt, system=system, max_tokens=2000)

    save_debug_content("stage1_gap_analysis_response.json", response)

    # Parse response
    try:
        cleaned = extract_last_json(response)
        data = json.loads(cleaned)
        return GapAnalysis(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Stage 1 parse error: {e}")
        # Return minimal valid response
        return GapAnalysis(
            players_to_sell=[],
            players_missing=[],
            risk_flags=[],
            formation_gaps=[],
            captain_gap=None,
        )


__all__ = ["aggregate_influencer_consensus", "stage_gap_analysis"]
