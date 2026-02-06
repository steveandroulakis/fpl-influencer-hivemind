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
from src.fpl_influencer_hivemind.types import ScoredGapAnalysis

logger = logging.getLogger(__name__)


_ROTATION_KEYWORDS = frozenset(
    {"rotation", "rotated", "benched", "benching", "minutes", "rested", "resting"}
)


def aggregate_influencer_consensus(
    channel_analyses: list[ChannelAnalysis],
) -> dict[str, Any]:
    """Aggregate influencer opinions into consensus data."""
    captain_counts: dict[str, list[str]] = {}
    transfers_in_counts: dict[str, list[str]] = {}
    transfers_out_counts: dict[str, list[str]] = {}
    watchlist_items: list[dict[str, Any]] = []
    rotation_warnings: dict[str, list[str]] = {}  # player → [channel names]
    chip_consensus: dict[str, list[dict[str, str]]] = {}  # "Wildcard GW24" → [{channel, reasoning}]

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

        # Chip strategy
        for chip_rec in analysis.chip_strategy:
            chip = chip_rec.get("chip", "")
            gw = chip_rec.get("gameweek", "")
            reasoning = chip_rec.get("reasoning", "")
            if chip:
                key = f"{chip} {gw}".strip()
                chip_consensus.setdefault(key, []).append(
                    {"channel": analysis.channel_name, "reasoning": reasoning}
                )

        # Rotation warnings from key_issues_discussed
        for issue in analysis.key_issues_discussed:
            text = (
                f"{issue.get('issue', '')} {issue.get('opinion', '')}"
            ).lower()
            if any(kw in text for kw in _ROTATION_KEYWORDS):
                # Extract player names mentioned in team_selection / watchlist
                # to match rotation warning to specific players
                _tag_rotation_players(
                    text, analysis, rotation_warnings
                )

    return {
        "captain_counts": captain_counts,
        "transfers_in_counts": transfers_in_counts,
        "transfers_out_counts": transfers_out_counts,
        "watchlist": watchlist_items,
        "rotation_warnings": rotation_warnings,
        "chip_consensus": chip_consensus,
        "total_channels": len(channel_analyses),
    }


def _tag_rotation_players(
    issue_text: str,
    analysis: ChannelAnalysis,
    rotation_warnings: dict[str, list[str]],
) -> None:
    """Match rotation issue text to player names from channel analysis."""
    # Check all players mentioned in this channel's context
    all_players = (
        analysis.team_selection
        + analysis.transfers_in
        + [w.get("name", "") for w in analysis.watchlist]
    )
    for player_str in all_players:
        # Extract bare name (strip position tag like "(MID)")
        name = player_str.split("(")[0].strip()
        if not name:
            continue
        # Check if any part of the player's name appears in the issue text
        name_parts = name.lower().split()
        if any(part in issue_text for part in name_parts if len(part) > 2):
            rotation_warnings.setdefault(player_str, []).append(
                analysis.channel_name
            )


def stage_gap_analysis(
    client: AnthropicClient,
    channel_analyses: list[ChannelAnalysis],
    squad_context: dict[str, Any],
    gameweek: int,
    transfer_momentum: dict[str, Any] | None = None,
    commentary: str | None = None,
    previous_errors: list[str] | None = None,
    condensed_players: list[dict[str, Any]] | None = None,
) -> ScoredGapAnalysis:
    """Stage 1: identify gaps between my squad and consensus with severity scores."""
    logger.info("Stage 1: Gap Analysis (with severity scoring)")

    squad = squad_context["squad"]
    squad_names = {p["name"] for p in squad}
    consensus = aggregate_influencer_consensus(channel_analyses)
    total_channels = consensus["total_channels"]

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
                    "form": p.get("form"),
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
                    "form": p.get("form"),
                }
                for p in top_net[:5]
            ]
            momentum_section = f"""
TRANSFER MOMENTUM (Real-time FPL manager activity this gameweek):
Top Transfers IN: {json.dumps(in_summary, indent=2)}
Top Transfers OUT: {json.dumps(out_summary, indent=2)}
Top Net Transfers: {json.dumps(net_summary, indent=2)}

Use transfer momentum to boost severity: +1 if >100k net transfers.
"""

    # Build error feedback section
    error_feedback = ""
    if previous_errors:
        error_feedback = (
            "\n\nPREVIOUS ATTEMPT FAILED WITH ERRORS:\n"
            + "\n".join(f"- {e}" for e in previous_errors)
            + "\n\nFix these issues in your response.\n"
        )

    # Build ownership lookup for tiebreaker
    ownership_section = ""
    if condensed_players:
        ownership_map = {
            p["web_name"]: p.get("selected_by_percent", 0)
            for p in condensed_players[:80]
            if p.get("selected_by_percent", 0) >= 10.0
        }
        if ownership_map:
            ownership_section = f"""
PLAYER OWNERSHIP (selected_by_percent, >=10% only):
{json.dumps(ownership_map, indent=1)}
"""

    directive_section = ""
    if commentary:
        directive_section = (
            f"\nUSER DIRECTIVE (HIGH PRIORITY - FOLLOW THIS):\n{commentary}\n"
        )

    prompt = f"""Analyze gaps between my FPL squad and influencer consensus for GW{gameweek}.
{momentum_section}
{ownership_section}
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
- Rotation warnings: {json.dumps(consensus.get("rotation_warnings", {}), indent=2)}
- Total channels analyzed: {total_channels}
{error_feedback}
Return JSON with SEVERITY SCORES (0-10) for each gap:
{{
  "players_to_sell": [
    {{"name": "Player Name", "position": "POS", "team": "Team Name", "severity": 6.0, "severity_factors": ["3/{total_channels} influencers selling", "poor form"]}}
  ],
  "players_missing": [
    {{"name": "Player Name", "position": "POS", "team": "Team Name", "severity": 8.0, "severity_factors": ["5/{total_channels} influencers", "high form"]}}
  ],
  "risk_flags": [
    {{"player": "Player Name", "risk": "Description of risk"}}
  ],
  "formation_gaps": ["Gap description"],
  "captain_gap": "Player Name or null",
  "captain_severity": 9.0,
  "total_severity": 23.0
}}

SEVERITY SCORING GUIDELINES:
- Base: 1 point per influencer recommending (max {total_channels} for {total_channels} channels)
- Captain gap: +3 bonus (critical for 2x points)
- Injury replacement: +2 if replacing injured starter
- Form differential: +1 if replacement has form > 5.0
- Transfer momentum: +1 if >100k net transfers
- Rotation risk: -1 to players_missing severity if 2+ influencers flag rotation risk for that player
- Tiebreaker: when two gaps have equal severity, rank the player with higher ownership (selected_by_percent) first
- Maximum per gap: 10
- total_severity = sum of all players_to_sell + players_missing + captain_severity

{PL_TEAMS_CONTEXT}

Rules:
1. players_to_sell: ONLY players in MY SQUAD that influencers are selling/benching.
2. players_missing: Popular picks I don't own; MUST NOT include any player from MY SQUAD.
3. captain_gap: If consensus captain is NOT in my squad, set this (top priority); else null.
4. captain_severity: 0 if captain_gap is null; otherwise score based on how critical (usually 6-10).
5. risk_flags: injury/rotation/form/availability risks ONLY (no team-quality speculation).
6. formation_gaps: position imbalances or formation inflexibility.
7. ORDER players_missing and players_to_sell by severity (highest first).

Return ONLY valid JSON, no markdown fences."""

    system = """You are an FPL analyst identifying gaps between a manager's squad and influencer recommendations.
Assign severity scores to prioritize gaps. Use only the provided data (no external knowledge). Return valid JSON only."""

    save_debug_content("stage1_gap_analysis_prompt.txt", prompt)

    response, _ = client.call_opus(prompt=prompt, system=system, max_tokens=2500)

    save_debug_content("stage1_gap_analysis_response.json", response)

    # Parse response
    try:
        cleaned = extract_last_json(response)
        data = json.loads(cleaned)
        return ScoredGapAnalysis(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Stage 1 parse error: {e}")
        # Return minimal valid response
        return ScoredGapAnalysis(
            players_to_sell=[],
            players_missing=[],
            risk_flags=[],
            formation_gaps=[],
            captain_gap=None,
            captain_severity=0.0,
            total_severity=0.0,
        )


__all__ = ["aggregate_influencer_consensus", "stage_gap_analysis"]
