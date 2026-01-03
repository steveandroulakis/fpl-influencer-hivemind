"""Lineup selection stage for the FPL Intelligence Analyzer."""

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
from src.fpl_influencer_hivemind.types import LineupPlan

logger = logging.getLogger(__name__)


def aggregate_influencer_xi(
    channel_analyses: list[ChannelAnalysis],
) -> dict[str, Any]:
    """Aggregate influencer starting XI selections where provided."""
    xi_counts: dict[str, list[str]] = {}
    formation_counts: dict[str, int] = {}

    for analysis in channel_analyses:
        if analysis.formation:
            formation_counts[analysis.formation] = (
                formation_counts.get(analysis.formation, 0) + 1
            )
        for player in analysis.team_selection:
            xi_counts.setdefault(player, []).append(analysis.channel_name)

    return {"xi_counts": xi_counts, "formation_counts": formation_counts}


def stage_lineup_selection(
    client: AnthropicClient,
    post_transfer_squad: list[dict[str, Any]],
    channel_analyses: list[ChannelAnalysis],
    gameweek: int,
    commentary: str | None = None,
    previous_errors: list[str] | None = None,
) -> LineupPlan:
    """Stage 3: select XI, bench, captain from post-transfer squad."""
    logger.info("Stage 3: Lineup Selection")

    squad_names = [p["name"] for p in post_transfer_squad]
    squad_labels = [
        f"{p['name']} ({p['position']})" for p in post_transfer_squad if p.get("name")
    ]

    xi_consensus = aggregate_influencer_xi(channel_analyses)

    # Aggregate captaincy data
    captain_data: list[dict[str, Any]] = []
    for analysis in channel_analyses:
        if analysis.captain_choice and analysis.captain_choice != "Not specified":
            captain_data.append(
                {
                    "captain": analysis.captain_choice,
                    "vice": analysis.vice_captain_choice,
                    "channel": analysis.channel_name,
                    "reasoning": "; ".join(analysis.key_reasoning[:2])
                    if analysis.key_reasoning
                    else "",
                }
            )

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

    prompt = f"""Select starting XI, bench, and captain for GW{gameweek}.

POST-TRANSFER SQUAD (15 players - use ONLY these):
{json.dumps(post_transfer_squad, indent=2)}

VALID PLAYER NAMES (use exact names from this list):
{json.dumps(squad_names, indent=2)}

VALID PLAYER LABELS (use EXACT labels in your output):
{json.dumps(squad_labels, indent=2)}

INFLUENCER CAPTAINCY CHOICES:
{json.dumps(captain_data, indent=2)}
INFLUENCER XI PICKS (players explicitly in their starting XIs):
{json.dumps(xi_consensus["xi_counts"], indent=2)}

FORMATION TRENDS (count of formations mentioned):
{json.dumps(xi_consensus["formation_counts"], indent=2)}
{directive_section}
{error_feedback}
Return JSON matching this schema EXACTLY:
{{
  "starting_xi": ["Player1 (POS)", "Player2 (POS)", "..."],
  "bench": ["Player1 (POS)", "Player2 (POS)", "Player3 (POS)", "Player4 (POS)"],
  "captain": "PlayerName (POS)",
  "vice_captain": "PlayerName (POS)",
  "formation": "3-5-2",
  "reasoning": "Brief explanation"
}}

{PL_TEAMS_CONTEXT}

Rules:
1. starting_xi must have EXACTLY 11 players; bench must have EXACTLY 4 (auto-sub order).
2. Formation must be valid: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD.
3. captain and vice_captain MUST be in starting_xi.
4. ALL 15 players must be used exactly once.
5. Use EXACT labels from VALID PLAYER LABELS list.
6. Player format: "PlayerName (POS)" only.
7. Prefer players with higher influencer XI backer counts when choices are close.
8. starting_xi order preference: GK, DEFs, MIDs, FWDs (helps readability).

Return ONLY valid JSON, no markdown fences."""

    system = """You are an FPL lineup selector. Choose optimal starting XI, bench order, and captain.
Use only the provided squad data (no external knowledge). Return valid JSON only."""

    save_debug_content("stage3_lineup_selection_prompt.txt", prompt)

    response, _ = client.call_opus(prompt=prompt, system=system, max_tokens=2000)

    save_debug_content("stage3_lineup_selection_response.json", response)

    # Parse response
    try:
        cleaned = extract_last_json(response)
        data = json.loads(cleaned)
        return LineupPlan(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Stage 3 parse error: {e}")
        # Return minimal valid lineup
        return LineupPlan(
            starting_xi=[],
            bench=[],
            captain="",
            vice_captain="",
            formation="",
            reasoning="Failed to generate lineup",
        )


__all__ = ["aggregate_influencer_xi", "stage_lineup_selection"]
