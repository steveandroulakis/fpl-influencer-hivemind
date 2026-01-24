"""Quality review module for the FPL Intelligence Analyzer."""

import json
import logging
from typing import Any

from pydantic import ValidationError

from src.fpl_influencer_hivemind.analyzer.api import (
    AnthropicClient,
    extract_last_json,
    save_debug_content,
)
from src.fpl_influencer_hivemind.types import (
    GapAnalysis,
    LineupPlan,
    QualityReview,
    ScoredGapAnalysis,
    TransferPlan,
)

logger = logging.getLogger(__name__)


def holistic_quality_review(
    client: AnthropicClient,
    gap: GapAnalysis | ScoredGapAnalysis,
    transfers: TransferPlan,
    lineup: LineupPlan,
    consensus: dict[str, Any],
    squad_context: dict[str, Any],
    gameweek: int,
) -> QualityReview:
    """Final holistic LLM review of the complete report for quality assessment."""
    logger.info("Running holistic quality review")

    # Build comprehensive context for LLM review
    squad_names = [p.get("name", "") for p in squad_context.get("squad", [])]
    itb = squad_context.get("itb", 0.0)
    free_transfers = squad_context.get("free_transfers", 1)

    # Summarize consensus
    captain_counts = consensus.get("captain_counts", {})
    top_captains = sorted(
        captain_counts.items(), key=lambda x: len(x[1]), reverse=True
    )[:3]
    transfers_in_counts = consensus.get("transfers_in_counts", {})
    top_transfers_in = sorted(
        transfers_in_counts.items(), key=lambda x: len(x[1]), reverse=True
    )[:5]

    prompt = f"""Review this FPL GW{gameweek} recommendation report for INTERNAL CONSISTENCY ONLY.

Critical rules:
- Do NOT use external knowledge about players, teams, positions, or transfers.
- Only compare the data below against itself.
- Trust all names/positions as given (FPL data is authoritative).

## GAP ANALYSIS (Stage 1)
Players to sell: {[p.name for p in gap.players_to_sell]}
Players missing: {[p.name for p in gap.players_missing]}
Risk flags: {[(rf.player, rf.risk) for rf in gap.risk_flags]}
Captain gap: {gap.captain_gap}

## TRANSFER PLAN (Stage 2)
Transfers: {[(t.out_player, "→", t.in_player) for t in transfers.transfers]}
Budget after: £{transfers.new_itb}m
FTs used: {transfers.fts_used}, Hits: {transfers.hit_cost}
Reasoning: {transfers.reasoning}

## LINEUP (Stage 3)
Starting XI: {lineup.starting_xi}
Bench: {lineup.bench}
Captain: {lineup.captain}, Vice: {lineup.vice_captain}
Formation: {lineup.formation}
Reasoning: {lineup.reasoning}

## INFLUENCER CONSENSUS
Top captain picks: {[(c, len(backers)) for c, backers in top_captains]}
Top transfer targets: {[(p, len(backers)) for p, backers in top_transfers_in]}
Total channels: {consensus.get("total_channels", 0)}

## SQUAD CONTEXT
Current squad: {squad_names}
ITB: £{itb}m, Free transfers: {free_transfers}

Return JSON only:
{{
  "confidence_score": 0.0-1.0,
  "quality_notes": ["note1", "note2"],
  "consensus_alignment": "alignment summary",
  "risk_assessment": "risk summary",
  "potential_issues": ["non-fixable issues for user awareness"],
  "recommendation_strength": "strong|moderate|weak",
  "fixable_issues": [
    {{
      "stage": "transfer|lineup",
      "issue": "internal contradiction",
      "fix_instruction": "specific instruction to fix next attempt"
    }}
  ]
}}

Fixable issues (go in fixable_issues):
- Internal contradictions that can be fixed by re-running a stage.
- Examples: sold player in XI; risky captain+vice; high-consensus target ignored without reasoning; gap not addressed.

Non-fixable issues (go in potential_issues):
- Trade-offs or constraints that cannot be resolved within the stage outputs.

Do NOT flag:
- Player names/positions/teams (trust the data).
- Any external facts not present above."""

    system = """You validate FPL reports for INTERNAL CONSISTENCY ONLY.
DO NOT use your own knowledge - only compare the provided data against itself.
Trust all player names, teams, and positions as given. Return valid JSON only."""

    save_debug_content("holistic_review_prompt.txt", prompt)

    try:
        response, _ = client.call_sonnet(
            prompt=prompt, system=system, max_tokens=1500
        )

        save_debug_content("holistic_review_response.json", response)

        cleaned = extract_last_json(response)
        data = json.loads(cleaned)
        return QualityReview(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Holistic review parse error: {e}")
        # Return neutral fallback
        return QualityReview(
            confidence_score=0.5,
            quality_notes=["Automated quality review could not be completed"],
            consensus_alignment="Unable to assess",
            risk_assessment="Unable to assess",
            potential_issues=[],
            recommendation_strength="moderate",
        )


__all__ = ["holistic_quality_review"]
