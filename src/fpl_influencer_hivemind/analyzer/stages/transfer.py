"""Transfer plan stage for the FPL Intelligence Analyzer."""

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
from src.fpl_influencer_hivemind.analyzer.models import (
    ChannelAnalysis,
    PlayerLookupEntry,
    SquadPlayerEntry,
)
from src.fpl_influencer_hivemind.analyzer.normalization import (
    canonicalize_player_label,
    coerce_price,
    normalize_name,
    select_lookup_candidate,
    split_player_label,
)
from src.fpl_influencer_hivemind.analyzer.stages.gap import (
    aggregate_influencer_consensus,
)
from src.fpl_influencer_hivemind.types import GapAnalysis, Transfer, TransferPlan

logger = logging.getLogger(__name__)


def stage_transfer_plan(
    client: AnthropicClient,
    gap: GapAnalysis,
    squad_context: dict[str, Any],
    condensed_players: list[dict[str, Any]],
    channel_analyses: list[ChannelAnalysis],
    gameweek: int,
    commentary: str | None = None,
    previous_errors: list[str] | None = None,
) -> TransferPlan:
    """Stage 2: generate specific transfers to address gaps."""
    logger.info("Stage 2: Transfer Plan")

    squad = squad_context["squad"]
    itb = squad_context["itb"]
    fts = squad_context["free_transfers"]
    club_counts = squad_context["club_counts"]
    squad_names = {p["name"] for p in squad}

    consensus = aggregate_influencer_consensus(channel_analyses)

    # Build error feedback
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

    prompt = f"""Generate specific FPL transfers for GW{gameweek} based on gap analysis.

GAP ANALYSIS:
{gap.model_dump_json(indent=2)}
{directive_section}

MY SQUAD (with selling prices):
{json.dumps(squad, indent=2)}

SQUAD PLAYER NAMES (CANNOT transfer these IN - already owned):
{json.dumps(list(squad_names), indent=2)}

CURRENT CLUB COUNTS (max 3 per club):
{json.dumps(club_counts, indent=2)}

BUDGET: ITB = {itb}m, Free Transfers = {fts}

AVAILABLE PLAYERS (top 150 by form):
{json.dumps(condensed_players[:150], indent=2)}

INFLUENCER TRANSFER RECOMMENDATIONS:
- Transfers IN: {json.dumps(consensus["transfers_in_counts"], indent=2)}
- Transfers OUT: {json.dumps(consensus["transfers_out_counts"], indent=2)}
{error_feedback}
Return JSON matching this schema EXACTLY:
{{
  "transfers": [
    {{
      "out_player": "PlayerName (POS)",
      "out_team": "Team Name",
      "in_player": "PlayerName (POS)",
      "in_team": "Team Name",
      "in_price": 8.5,
      "selling_price": 8.0,
      "cost_delta": 0.5,
      "backers": ["Channel1", "Channel2"]
    }}
  ],
  "total_cost": 0.5,
  "new_itb": 1.4,
  "fts_used": 1,
  "fts_remaining": 0,
  "hit_cost": 0,
  "reasoning": "Brief explanation"
}}

{PL_TEAMS_CONTEXT}

Rules:
1. out_player MUST be in my squad; in_player MUST NOT be in my squad.
2. Position must match: FWD->FWD, MID->MID, DEF->DEF, GKP->GKP.
3. in_player MUST be from AVAILABLE PLAYERS list above (use exact web_name + position).
4. Use the exact price from AVAILABLE PLAYERS list (do not estimate).
5. cost_delta = in_price - selling_price.
6. new_itb = ITB - sum(cost_delta for all transfers) (must be >= 0).
7. Club count after transfers must be <= 3 for any club.
8. hit_cost = max(0, len(transfers) - fts) * 4.
9. If no transfers recommended, return empty transfers array with fts_remaining = {fts}.
10. backers should list influencer channel names when available; else [].

Return ONLY valid JSON, no markdown fences."""

    system = """You are an FPL transfer strategist. Generate specific, valid transfers respecting all FPL rules.
Use only the provided data (no external knowledge). Return valid JSON only."""

    save_debug_content("stage2_transfer_plan_prompt.txt", prompt)

    response, _ = client.call_opus(prompt=prompt, system=system, max_tokens=2000)

    save_debug_content("stage2_transfer_plan_response.json", response)

    # Parse response
    try:
        cleaned = extract_last_json(response)
        data = json.loads(cleaned)
        return TransferPlan(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Stage 2 parse error: {e}")
        return TransferPlan(
            transfers=[],
            total_cost=0.0,
            new_itb=itb,
            fts_used=0,
            fts_remaining=fts,
            hit_cost=0,
            reasoning="Failed to generate transfer plan",
        )


def apply_transfer_pricing(
    transfer_plan: TransferPlan,
    squad_context: dict[str, Any],
    player_lookup: dict[str, list[PlayerLookupEntry]],
) -> tuple[TransferPlan, list[str]]:
    """Recalculate transfer prices from FPL data and validate availability."""
    errors: list[str] = []
    updated_transfers: list[Transfer] = []

    itb = float(squad_context.get("itb", 0.0))
    fts = int(squad_context.get("free_transfers", 0))
    squad_lookup: dict[str, SquadPlayerEntry] = {}

    for player in squad_context.get("squad", []):
        name = str(player.get("name", "")).strip()
        if not name:
            continue
        squad_lookup[normalize_name(name)] = SquadPlayerEntry(
            name=name,
            position=str(player.get("position", "")),
            team=str(player.get("team", "")),
            selling_price=coerce_price(
                player.get("selling_price", player.get("price", 0.0))
            ),
        )

    for transfer in transfer_plan.transfers:
        out_name, out_pos_hint = split_player_label(transfer.out_player)
        out_key = normalize_name(out_name)
        out_info = squad_lookup.get(out_key)
        if not out_info:
            errors.append(f"Transfer OUT '{out_name}' not found in squad context")
            continue

        in_label = canonicalize_player_label(
            transfer.in_player, player_lookup, pos_hint=out_pos_hint
        )
        in_name, in_pos_hint = split_player_label(in_label)
        in_key = normalize_name(in_name)
        candidates = player_lookup.get(in_key, [])
        in_info = select_lookup_candidate(
            candidates, in_name, in_pos_hint or out_info.position
        )
        if not in_info:
            errors.append(
                f"Transfer IN '{in_name}' not found in available player list"
            )
            continue

        if out_info.position != in_info.position:
            errors.append(
                f"Position mismatch: {out_info.position} -> {in_info.position}"
            )
            continue

        in_price = in_info.price
        selling_price = out_info.selling_price
        cost_delta = in_price - selling_price

        updated_transfers.append(
            Transfer(
                out_player=f"{out_info.name} ({out_info.position})",
                out_team=out_info.team,
                in_player=f"{in_info.name} ({in_info.position})",
                in_team=in_info.team,
                in_price=in_price,
                selling_price=selling_price,
                cost_delta=cost_delta,
                backers=transfer.backers,
            )
        )

    total_cost = sum(t.cost_delta for t in updated_transfers)
    new_itb = itb - total_cost
    fts_used = len(updated_transfers)
    fts_remaining = max(0, fts - fts_used)
    hit_cost = max(0, fts_used - fts) * 4

    updated_plan = TransferPlan(
        transfers=updated_transfers,
        total_cost=total_cost,
        new_itb=new_itb,
        fts_used=fts_used,
        fts_remaining=fts_remaining,
        hit_cost=hit_cost,
        reasoning=transfer_plan.reasoning,
    )

    return updated_plan, errors


def compute_post_transfer_squad(
    original_squad: list[dict[str, Any]],
    transfers: list[Transfer],
) -> list[dict[str, Any]]:
    """Apply transfers to produce new squad list."""
    # Create a copy of the squad
    new_squad = [p.copy() for p in original_squad]

    for transfer in transfers:
        out_name = transfer.out_player.split(" (")[0]
        in_name = transfer.in_player.split(" (")[0]
        in_pos = transfer.in_player.split("(")[1].rstrip(")")
        out_key = normalize_name(out_name)

        # Find and replace the outgoing player
        for i, player in enumerate(new_squad):
            if normalize_name(str(player.get("name", ""))) == out_key:
                new_squad[i] = {
                    "name": in_name,
                    "position": in_pos,
                    "team": transfer.in_team,
                    "price": transfer.in_price,
                    "selling_price": transfer.in_price,  # New players sell at buy price
                    "element_id": None,  # Unknown for new players
                }
                break

    return new_squad


__all__ = [
    "apply_transfer_pricing",
    "compute_post_transfer_squad",
    "stage_transfer_plan",
]
