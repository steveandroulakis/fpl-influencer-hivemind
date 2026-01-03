"""Mechanical validation methods for the FPL Intelligence Analyzer."""

from typing import Any

from src.fpl_influencer_hivemind.analyzer.api import AnthropicClient
from src.fpl_influencer_hivemind.analyzer.normalization import normalize_name
from src.fpl_influencer_hivemind.analyzer.validation.cohesion import (
    validate_consensus_coverage,
    validate_gap_to_transfer_cohesion,
    validate_risk_contingency,
)
from src.fpl_influencer_hivemind.types import (
    GapAnalysis,
    LineupPlan,
    TransferPlan,
    ValidationResult,
)


def validate_transfers(
    transfers: TransferPlan,
    original_squad: list[dict[str, Any]],
    post_transfer_squad: list[dict[str, Any]],
) -> list[str]:
    """Validate transfer plan."""
    errors: list[str] = []
    original_names = {
        normalize_name(str(p.get("name", ""))) for p in original_squad
    }

    for t in transfers.transfers:
        out_name = t.out_player.split(" (")[0]
        in_name = t.in_player.split(" (")[0]
        out_key = normalize_name(out_name)
        in_key = normalize_name(in_name)

        # Out player must be in original squad
        if out_key not in original_names:
            errors.append(f"Transfer OUT '{out_name}' not in squad")

        # In player must NOT be in original squad
        if in_key in original_names:
            errors.append(f"Transfer IN '{in_name}' already in squad")

        # Position match
        if "(" in t.out_player and "(" in t.in_player:
            out_pos = t.out_player.split("(")[1].rstrip(")")
            in_pos = t.in_player.split("(")[1].rstrip(")")
            if out_pos != in_pos:
                errors.append(f"Position mismatch: {out_pos} -> {in_pos}")

    # Budget check
    if transfers.new_itb < 0:
        errors.append(f"Budget violated: new_itb={transfers.new_itb}")

    # Club limit check
    club_counts: dict[str, int] = {}
    for p in post_transfer_squad:
        club = p.get("team", "")
        club_counts[club] = club_counts.get(club, 0) + 1
    for club, count in club_counts.items():
        if count > 3:
            errors.append(f"Club limit exceeded: {club} has {count} players")

    return errors


def validate_lineup(
    lineup: LineupPlan,
    post_transfer_squad: list[dict[str, Any]],
) -> list[str]:
    """Validate lineup plan."""
    errors: list[str] = []
    post_names = {
        normalize_name(str(p.get("name", ""))) for p in post_transfer_squad
    }

    xi_names = {normalize_name(p.split(" (")[0]) for p in lineup.starting_xi}

    # Count by position
    pos_count = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in lineup.starting_xi:
        if "(" in p:
            pos = p.split("(")[1].rstrip(")")
            pos_count[pos] = pos_count.get(pos, 0) + 1

    if len(lineup.starting_xi) != 11:
        errors.append(f"XI has {len(lineup.starting_xi)} players, need 11")
    if len(lineup.bench) != 4:
        errors.append(f"Bench has {len(lineup.bench)} players, need 4")

    if pos_count["GKP"] != 1:
        errors.append(f"XI must have exactly 1 GKP, has {pos_count['GKP']}")
    if not (3 <= pos_count["DEF"] <= 5):
        errors.append(f"XI must have 3-5 DEF, has {pos_count['DEF']}")
    if not (2 <= pos_count["MID"] <= 5):
        errors.append(f"XI must have 2-5 MID, has {pos_count['MID']}")
    if not (1 <= pos_count["FWD"] <= 3):
        errors.append(f"XI must have 1-3 FWD, has {pos_count['FWD']}")

    # Captain/vice in XI
    if lineup.captain:
        cap_name = lineup.captain.split(" (")[0]
        if cap_name not in xi_names:
            errors.append(f"Captain '{cap_name}' not in starting XI")

    if lineup.vice_captain:
        vice_name = lineup.vice_captain.split(" (")[0]
        if vice_name not in xi_names:
            errors.append(f"Vice '{vice_name}' not in starting XI")

    # All players exist in post-transfer squad
    for label in lineup.starting_xi + lineup.bench:
        name = label.split(" (")[0]
        if normalize_name(name) not in post_names:
            errors.append(f"'{name}' not in post-transfer squad")

    return errors


def validate_all(
    client: AnthropicClient,
    gap: GapAnalysis,
    transfers: TransferPlan,
    lineup: LineupPlan,
    original_squad: list[dict[str, Any]],
    post_transfer_squad: list[dict[str, Any]],
    consensus: dict[str, Any] | None = None,
    squad_context: dict[str, Any] | None = None,
    condensed_players: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    """Stage 4: programmatic validation of all outputs including cohesion checks."""
    errors: list[str] = []
    warnings: list[str] = []

    # Validate transfers (mechanical)
    transfer_errors = validate_transfers(transfers, original_squad, post_transfer_squad)
    errors.extend(transfer_errors)

    # Validate lineup (mechanical)
    lineup_errors = validate_lineup(lineup, post_transfer_squad)
    errors.extend(lineup_errors)

    # Cohesion validation (if data available)
    if consensus is not None:
        # Gap to transfer cohesion
        gap_errors, gap_warnings = validate_gap_to_transfer_cohesion(
            client, gap, transfers, consensus
        )
        errors.extend(gap_errors)
        warnings.extend(gap_warnings)

        # Consensus coverage
        if squad_context is not None and condensed_players is not None:
            consensus_errors, consensus_warnings = validate_consensus_coverage(
                client, transfers, consensus, squad_context, condensed_players
            )
            errors.extend(consensus_errors)
            warnings.extend(consensus_warnings)

    # Risk contingency validation
    risk_errors, risk_warnings = validate_risk_contingency(gap, lineup)
    errors.extend(risk_errors)
    warnings.extend(risk_warnings)

    # Determine failed stage
    failed_stage: str | None = None
    if errors:
        if any(
            "Transfer" in e or "Budget" in e or "Club limit" in e for e in errors
        ):
            failed_stage = "transfer"
        elif any(
            "XI" in e
            or "Captain" in e
            or "Vice" in e
            or "Bench" in e
            or "not in post-transfer" in e
            for e in errors
        ):
            failed_stage = "lineup"
        elif any("COHESION" in e for e in errors):
            failed_stage = "cohesion"

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        failed_stage=failed_stage,
    )


__all__ = ["validate_all", "validate_lineup", "validate_transfers"]
