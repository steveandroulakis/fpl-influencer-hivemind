"""Cohesion validation methods for the FPL Intelligence Analyzer."""

import json
import logging
from typing import Any

from src.fpl_influencer_hivemind.analyzer.api import (
    AnthropicClient,
    extract_last_json,
)
from src.fpl_influencer_hivemind.analyzer.normalization import (
    normalize_name,
    normalize_player_label,
)
from src.fpl_influencer_hivemind.types import (
    GapAnalysis,
    LineupPlan,
    TransferPlan,
)

logger = logging.getLogger(__name__)

# Cohesion validation thresholds
CONSENSUS_THRESHOLD = 3  # min influencers for warning
MAJORITY_RATIO = 0.6  # 4/6 = majority, triggers error

# Cache for justification verification
_justification_cache: dict[str, dict[str, object]] = {}


def verify_justification_llm(
    client: AnthropicClient,
    player_name: str,
    reasoning_texts: list[str],
    context: str,
) -> dict[str, object]:
    """Use LLM to verify if a player exclusion is justified in reasoning."""
    # Check cache first
    cache_key = f"{player_name}:{hash(tuple(reasoning_texts))}"
    if cache_key in _justification_cache:
        return _justification_cache[cache_key]

    combined_reasoning = "\n\n".join(reasoning_texts)
    prompt = f"""Assess whether excluding "{player_name}" is justified by the reasoning.

REASONING TEXT:
{combined_reasoning}

CONTEXT:
{context}

Return JSON only:
{{"justified": true/false, "reason": "one short sentence"}}

Rules:
- justified = true if ANY explicit or clearly implied reason is present.
- justified = false only if the player is not mentioned and no rationale applies.
- Be lenient: if there is a reasonable explanation, mark justified."""

    try:
        response, _ = client.call_haiku(
            prompt=prompt,
            system="Analyze transfer reasoning. Return JSON only.",
            max_tokens=200,
        )
        cleaned = extract_last_json(response)
        result = json.loads(cleaned)
    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"Justification check failed for {player_name}: {e}")
        # Assume justified if we can't verify (fail open)
        result = {"justified": True, "reason": "Could not verify"}

    _justification_cache[cache_key] = result
    return result


def compute_player_affordability(
    player_name: str,  # noqa: ARG001  # Reserved for future logging
    player_price: float,
    squad_context: dict[str, Any],
) -> tuple[bool, float]:
    """Check if a player is affordable given current ITB + potential sells.

    Returns (is_affordable, max_budget_available).
    """
    itb = squad_context.get("itb", 0.0)
    squad = squad_context.get("squad", [])

    # Find max sell price from any single player
    max_sell = 0.0
    for p in squad:
        sell_price = p.get("selling_price", p.get("price", 0.0))
        if sell_price > max_sell:
            max_sell = sell_price

    max_budget = itb + max_sell
    is_affordable = player_price <= max_budget

    return is_affordable, max_budget


def validate_gap_to_transfer_cohesion(
    client: AnthropicClient,
    gap: GapAnalysis,
    transfers: TransferPlan,
    consensus: dict[str, Any],  # noqa: ARG001  # Reserved for future use
) -> tuple[list[str], list[str]]:
    """Validate Stage 1 gaps are addressed in Stage 2 transfers."""
    errors: list[str] = []
    warnings: list[str] = []

    # Get transferred-in player names
    transferred_in = {
        normalize_name(t.in_player.split(" (")[0]) for t in transfers.transfers
    }
    transferred_out = {
        normalize_name(t.out_player.split(" (")[0]) for t in transfers.transfers
    }

    reasoning_texts = [transfers.reasoning]

    # Check captain_gap (ERROR if not addressed)
    if gap.captain_gap:
        captain_key = normalize_player_label(gap.captain_gap)
        if captain_key not in transferred_in:
            justification = verify_justification_llm(
                client,
                gap.captain_gap,
                reasoning_texts,
                "This player is the consensus captain pick that the manager doesn't own.",
            )
            if not justification.get("justified", False):
                errors.append(
                    f"COHESION ISSUE: Captain gap '{gap.captain_gap}' not addressed - "
                    "transfer in or justify in reasoning"
                )

    # Check players_missing (WARNING if not addressed)
    for player_ref in gap.players_missing:
        player_key = normalize_player_label(player_ref.name)
        if player_key not in transferred_in:
            justification = verify_justification_llm(
                client,
                player_ref.name,
                reasoning_texts,
                f"High-priority missing player ({player_ref.position}, {player_ref.team}).",
            )
            if not justification.get("justified", False):
                warnings.append(
                    f"COHESION ISSUE: High-priority gap '{player_ref.name}' not in transfers"
                )

    # Check players_to_sell (WARNING if not addressed)
    for player_ref in gap.players_to_sell:
        player_key = normalize_player_label(player_ref.name)
        if player_key not in transferred_out:
            justification = verify_justification_llm(
                client,
                player_ref.name,
                reasoning_texts,
                f"Player identified to sell ({player_ref.position}, {player_ref.team}).",
            )
            if not justification.get("justified", False):
                warnings.append(
                    f"COHESION ISSUE: Player to sell '{player_ref.name}' not transferred out"
                )

    return errors, warnings


def validate_consensus_coverage(
    client: AnthropicClient,
    transfers: TransferPlan,
    consensus: dict[str, Any],
    squad_context: dict[str, Any],
    condensed_players: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Validate transfer plan covers strong influencer consensus."""
    errors: list[str] = []
    warnings: list[str] = []

    transfers_in_counts = consensus.get("transfers_in_counts", {})
    total_channels = consensus.get("total_channels", 6)
    majority_count = int(total_channels * MAJORITY_RATIO)

    # Get transferred-in player names
    transferred_in = {t.in_player.split(" (")[0].lower() for t in transfers.transfers}

    # Get current squad names
    squad_names = {
        normalize_name(str(p.get("name", "")))
        for p in squad_context.get("squad", [])
    }

    reasoning_texts = [transfers.reasoning]

    # Build player price lookup
    price_lookup: dict[str, float] = {}
    for p in condensed_players:
        name = normalize_player_label(str(p.get("web_name", "")))
        price_lookup[name] = p.get("price", 0.0)

    for player, backers in transfers_in_counts.items():
        backer_count = len(backers)
        player_key = normalize_player_label(player)

        # Skip if already in squad or already being transferred in
        if player_key in squad_names or player_key in transferred_in:
            continue

        # Check if recommended by threshold+ influencers
        if backer_count >= CONSENSUS_THRESHOLD:
            # Check affordability
            player_price = price_lookup.get(player_key, 0.0)
            is_affordable, max_budget = compute_player_affordability(
                player, player_price, squad_context
            )

            context = (
                f"Recommended by {backer_count} influencers: {', '.join(backers)}. "
                f"Price: £{player_price}m. "
                f"{'Affordable' if is_affordable else 'Not affordable'} (max budget: £{max_budget}m)."
            )

            justification = verify_justification_llm(
                client, player, reasoning_texts, context
            )

            if not justification.get("justified", False):
                if backer_count >= majority_count:
                    # Majority consensus ignored = ERROR
                    errors.append(
                        f"COHESION ISSUE: Majority ({backer_count}/{total_channels}) "
                        f"recommend '{player}' but not in plan"
                    )
                else:
                    # Strong but not majority = WARNING
                    warnings.append(
                        f"COHESION ISSUE: {backer_count} influencers recommend '{player}' "
                        "but not in plan"
                    )

    return errors, warnings


def validate_risk_contingency(
    gap: GapAnalysis,
    lineup: LineupPlan,
) -> tuple[list[str], list[str]]:
    """Validate risk flags have corresponding contingency in lineup."""
    errors: list[str] = []
    warnings: list[str] = []

    # Build set of risky player names
    risky_players = {
        normalize_player_label(rf.player): rf.risk for rf in gap.risk_flags
    }

    if not risky_players:
        return errors, warnings

    # Get XI and bench names
    xi_names = [normalize_name(p.split(" (")[0]) for p in lineup.starting_xi]
    bench_names = [normalize_name(p.split(" (")[0]) for p in lineup.bench]

    captain_name = (
        normalize_name(lineup.captain.split(" (")[0]) if lineup.captain else ""
    )
    vice_name = (
        normalize_name(lineup.vice_captain.split(" (")[0])
        if lineup.vice_captain
        else ""
    )

    # Check captain risk
    if captain_name in risky_players:
        captain_risk = risky_players[captain_name]
        if vice_name in risky_players:
            # Both captain and vice have risk = ERROR
            vice_risk = risky_players[vice_name]
            errors.append(
                f"COHESION ISSUE: Captain '{lineup.captain}' has risk ({captain_risk}) "
                f"AND vice '{lineup.vice_captain}' has risk ({vice_risk}) - need safe fallback"
            )
        else:
            # Just captain risky, vice is safe = WARNING
            warnings.append(
                f"COHESION ISSUE: Captain '{lineup.captain}' has risk flag ({captain_risk})"
            )

    # Check risky XI players have bench backup
    for i, xi_player in enumerate(xi_names):
        if xi_player in risky_players and xi_player != captain_name:
            risk = risky_players[xi_player]
            # Check if there's a same-position backup in bench[0:2]
            xi_full = lineup.starting_xi[i] if i < len(lineup.starting_xi) else ""
            if "(" in xi_full:
                xi_pos = xi_full.split("(")[1].rstrip(")")
                has_backup = False
                for bench_player in bench_names[:2]:
                    # Check if bench player is same position
                    bench_idx = bench_names.index(bench_player)
                    bench_full = (
                        lineup.bench[bench_idx] if bench_idx < len(lineup.bench) else ""
                    )
                    if "(" in bench_full:
                        bench_pos = bench_full.split("(")[1].rstrip(")")
                        if bench_pos == xi_pos:
                            has_backup = True
                            break
                if not has_backup:
                    warnings.append(
                        f"COHESION ISSUE: Risky player '{xi_full}' ({risk}) - "
                        "no same-position backup in top bench slots"
                    )

    return errors, warnings


__all__ = [
    "CONSENSUS_THRESHOLD",
    "MAJORITY_RATIO",
    "compute_player_affordability",
    "validate_consensus_coverage",
    "validate_gap_to_transfer_cohesion",
    "validate_risk_contingency",
    "verify_justification_llm",
]
