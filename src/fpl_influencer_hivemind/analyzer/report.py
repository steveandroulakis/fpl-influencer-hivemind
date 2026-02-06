"""Report generation module for the FPL Intelligence Analyzer."""

from typing import Any

from src.fpl_influencer_hivemind.analyzer.models import (
    ChannelAnalysis,
    DecisionOption,
)
from src.fpl_influencer_hivemind.analyzer.stages.gap import (
    aggregate_influencer_consensus,
)
from src.fpl_influencer_hivemind.types import (
    GapAnalysis,
    QualityReview,
    ScoredGapAnalysis,
)


def generate_consensus_section(channel_analyses: list[ChannelAnalysis]) -> str:
    """Build Section 1 from aggregated channel data."""
    consensus = aggregate_influencer_consensus(channel_analyses)
    total = consensus["total_channels"]

    lines = ["## 1) Consensus, Contrarian & Captaincy Snapshot\n"]

    # Captaincy matrix
    captain_counts = consensus["captain_counts"]
    if captain_counts:
        lines.append("### Captaincy Matrix\n")
        lines.append("| Captain | Backers | Count |")
        lines.append("|---------|---------|-------|")
        for cap, backers in sorted(
            captain_counts.items(), key=lambda x: len(x[1]), reverse=True
        ):
            lines.append(f"| {cap} | {', '.join(backers)} | {len(backers)}/{total} |")
        lines.append("")

    # Universal/Majority transfers in
    transfers_in = consensus["transfers_in_counts"]
    if transfers_in:
        lines.append("### Transfer Targets\n")
        for player, backers in sorted(
            transfers_in.items(), key=lambda x: len(x[1]), reverse=True
        ):
            pct = len(backers) / total * 100
            label = "Universal" if len(backers) == total else f"{pct:.0f}%"
            lines.append(f"- **{player}** ({label}): {', '.join(backers)}")
        lines.append("")

    # Transfers out
    transfers_out = consensus["transfers_out_counts"]
    if transfers_out:
        lines.append("### Players to Sell\n")
        for player, backers in sorted(
            transfers_out.items(), key=lambda x: len(x[1]), reverse=True
        ):
            pct = len(backers) / total * 100
            lines.append(f"- **{player}** ({pct:.0f}%): {', '.join(backers)}")
        lines.append("")

    return "\n".join(lines)


def generate_channel_notes(channel_analyses: list[ChannelAnalysis]) -> str:
    """Build Section 2: Channel-by-Channel Notes."""
    lines = ["## 2) Channel-by-Channel Notes\n"]

    for analysis in channel_analyses:
        lines.append(f"### {analysis.channel_name}")
        lines.append(f"*Confidence: {analysis.confidence}*\n")

        if analysis.formation:
            lines.append(f"- **Formation:** {analysis.formation}")

        if analysis.team_selection:
            lines.append(f"- **Team Selection:** {', '.join(analysis.team_selection)}")

        if analysis.transfers_in:
            lines.append(f"- **Transfers IN:** {', '.join(analysis.transfers_in)}")

        if analysis.transfers_out:
            lines.append(f"- **Transfers OUT:** {', '.join(analysis.transfers_out)}")

        lines.append(f"- **Captain:** {analysis.captain_choice}")
        lines.append(f"- **Vice Captain:** {analysis.vice_captain_choice}")

        if analysis.watchlist:
            watch_strs = [
                f"{w['name']} ({w.get('priority', 'med')})" for w in analysis.watchlist
            ]
            lines.append(f"- **Watchlist:** {', '.join(watch_strs)}")

        if analysis.key_issues_discussed:
            lines.append("- **Key Issues:**")
            for issue in analysis.key_issues_discussed[:3]:
                lines.append(
                    f"  - {issue.get('issue', '')}: {issue.get('opinion', '')}"
                )

        if analysis.key_reasoning:
            lines.append(f"- **Reasoning:** {'; '.join(analysis.key_reasoning[:3])}")

        lines.append("")

    return "\n".join(lines)


def format_gap_section(gap: GapAnalysis | ScoredGapAnalysis) -> str:
    """Format Section 3 from GapAnalysis model with severity display."""
    lines = ["## 3) My Team vs Influencers (Gap Analysis)\n"]

    # Check if this is a scored gap analysis
    is_scored = isinstance(gap, ScoredGapAnalysis)

    if gap.captain_gap:
        lines.append("### CRITICAL: Captain Gap")
        if is_scored and gap.captain_severity > 0:
            lines.append(
                f"Consensus captain **{gap.captain_gap}** is NOT in your squad! "
                f"(Severity: {gap.captain_severity:.0f}/10)\n"
            )
        else:
            lines.append(
                f"Consensus captain **{gap.captain_gap}** is NOT in your squad!\n"
            )

    if gap.players_missing:
        lines.append("### Players I'm Missing (by severity)")
        for p in gap.players_missing:
            team_str = f" - {p.team}" if p.team else ""
            # Check if player has severity info (ScoredPlayerRef)
            if hasattr(p, "severity") and p.severity > 0:
                severity_bar = _severity_bar(p.severity)
                factors_str = ""
                if hasattr(p, "severity_factors") and p.severity_factors:
                    factors_str = f"\n  Factors: {', '.join(p.severity_factors)}"
                lines.append(
                    f"- **{p.name}** ({p.position}){team_str} — Severity: {p.severity:.0f}/10 {severity_bar}{factors_str}"
                )
            else:
                lines.append(f"- **{p.name}** ({p.position}){team_str}")
        lines.append("")

    if gap.players_to_sell:
        lines.append("### Players to Consider Selling")
        for p in gap.players_to_sell:
            team_str = f" - {p.team}" if p.team else ""
            if hasattr(p, "severity") and p.severity > 0:
                severity_bar = _severity_bar(p.severity)
                factors_str = ""
                if hasattr(p, "severity_factors") and p.severity_factors:
                    factors_str = f"\n  Factors: {', '.join(p.severity_factors)}"
                lines.append(
                    f"- **{p.name}** ({p.position}){team_str} — Severity: {p.severity:.0f}/10 {severity_bar}{factors_str}"
                )
            else:
                lines.append(f"- **{p.name}** ({p.position}){team_str}")
        lines.append("")

    if gap.risk_flags:
        lines.append("### Risk Flags")
        for rf in gap.risk_flags:
            lines.append(f"- **{rf.player}**: {rf.risk}")
        lines.append("")

    if gap.formation_gaps:
        lines.append("### Formation Gaps")
        for fg in gap.formation_gaps:
            lines.append(f"- {fg}")
        lines.append("")

    # Show total severity if available
    if is_scored and gap.total_severity > 0:
        lines.append(f"**Total Severity:** {gap.total_severity:.1f}\n")

    return "\n".join(lines)


def _severity_bar(severity: float, max_val: float = 10.0) -> str:
    """Generate a visual severity bar."""
    filled = int((severity / max_val) * 10)
    return "█" * filled + "░" * (10 - filled)


def format_action_plan(decision_options: list[DecisionOption]) -> str:
    """Format Section 4 with 2-3 decision options."""
    lines = ["## 4) Action Plan\n", "### Decision Options (choose 1)\n"]

    for option in decision_options:
        transfers = option.transfers
        lineup = option.lineup

        lines.append(f"#### {option.label}")

        if transfers.transfers:
            lines.append("**Transfers:**")
            for t in transfers.transfers:
                backers = ", ".join(t.backers) if t.backers else "General consensus"
                lines.append(f"- **{t.out_player}** → **{t.in_player}**")
                lines.append(
                    f"  - Sell: {t.selling_price}m, Buy: {t.in_price}m, "
                    f"Delta: {t.cost_delta:+.1f}m"
                )
                lines.append(f"  - Backers: {backers}")
        else:
            lines.append("*No transfers (roll).*")

        lines.append(
            f"**Budget after:** {transfers.new_itb:.1f}m ITB | "
            f"**FTs used:** {transfers.fts_used} | "
            f"**FTs remaining:** {transfers.fts_remaining} | "
            f"**Hit cost:** {transfers.hit_cost}"
        )

        lines.append(
            f"**Captain:** {lineup.captain} | "
            f"**Vice:** {lineup.vice_captain} | "
            f"**Formation:** {lineup.formation}"
        )

        gkp = [p for p in lineup.starting_xi if "(GKP)" in p]
        defs = [p for p in lineup.starting_xi if "(DEF)" in p]
        mids = [p for p in lineup.starting_xi if "(MID)" in p]
        fwds = [p for p in lineup.starting_xi if "(FWD)" in p]

        if gkp:
            lines.append(f"**GKP:** {', '.join(gkp)}")
        if defs:
            lines.append(f"**DEF:** {', '.join(defs)}")
        if mids:
            lines.append(f"**MID:** {', '.join(mids)}")
        if fwds:
            lines.append(f"**FWD:** {', '.join(fwds)}")

        bench_str = ", ".join(
            f"{idx}. {player}" for idx, player in enumerate(lineup.bench, 1)
        )
        lines.append(f"**Bench:** {bench_str}")

        if option.rationale:
            lines.append(f"**Why:** {option.rationale}")

        if transfers.reasoning:
            lines.append(f"**Transfer reasoning:** {transfers.reasoning}")

        lines.append("")

    return "\n".join(lines).strip()


def format_quality_review(review: QualityReview) -> str:
    """Format the quality review section for the report."""
    lines = ["## 5) Quality Assessment\n"]

    # Confidence indicator
    confidence_pct = int(review.confidence_score * 100)
    strength_emoji = {"strong": "🟢", "moderate": "🟡", "weak": "🔴"}.get(
        review.recommendation_strength, "⚪"
    )
    lines.append(
        f"**Confidence:** {confidence_pct}% | "
        f"**Recommendation Strength:** {strength_emoji} {review.recommendation_strength.title()}\n"
    )

    # Consensus alignment
    lines.append(f"### Consensus Alignment\n{review.consensus_alignment}\n")

    # Risk assessment
    lines.append(f"### Risk Assessment\n{review.risk_assessment}\n")

    # Quality notes
    if review.quality_notes:
        lines.append("### Key Observations")
        for note in review.quality_notes:
            lines.append(f"- {note}")
        lines.append("")

    # Potential issues
    if review.potential_issues:
        lines.append("### Potential Issues to Consider")
        for issue in review.potential_issues:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")

    return "\n".join(lines)


def generate_chip_strategy_section(
    chip_consensus: dict[str, list[dict[str, str]]], total_channels: int
) -> str:
    """Build chip strategy consensus section from aggregated chip data."""
    if not chip_consensus:
        return ""

    lines = ["## Chip Strategy Consensus\n"]
    lines.append("| Chip | Backers | Count |")
    lines.append("|------|---------|-------|")

    for key, entries in sorted(
        chip_consensus.items(), key=lambda x: len(x[1]), reverse=True
    ):
        backers = ", ".join(e["channel"] for e in entries)
        lines.append(f"| {key} | {backers} | {len(entries)}/{total_channels} |")

    lines.append("")

    # Add key reasoning quotes (one per chip, from first backer)
    lines.append("**Key Reasoning:**")
    for key, entries in sorted(
        chip_consensus.items(), key=lambda x: len(x[1]), reverse=True
    ):
        reasoning = entries[0].get("reasoning", "")
        if reasoning:
            lines.append(f"- **{key}**: {reasoning}")
    lines.append("")

    return "\n".join(lines)


def generate_ownership_section(
    condensed_players: list[dict[str, Any]],
    squad_names: set[str],
    ownership_threshold: float = 15.0,
    max_shown: int = 10,
) -> str:
    """Build section showing high-ownership players missing from squad."""
    missing: list[dict[str, Any]] = []
    for p in condensed_players:
        name = p.get("web_name", "")
        if not name or name in squad_names:
            continue
        ownership = float(p.get("selected_by_percent", 0))
        if ownership >= ownership_threshold:
            missing.append(p)

    if not missing:
        return ""

    # Sort by ownership desc, take top N
    missing.sort(key=lambda x: float(x.get("selected_by_percent", 0)), reverse=True)
    missing = missing[:max_shown]

    lines = ["## Popular Players You're Missing\n"]
    lines.append("| Player | Pos | Team | Ownership | Price |")
    lines.append("|--------|-----|------|-----------|-------|")
    for p in missing:
        lines.append(
            f"| {p.get('web_name', '')} | {p.get('position', '')} | "
            f"{p.get('team_name', '')} | {float(p.get('selected_by_percent', 0)):.1f}% | "
            f"{float(p.get('price', 0)):.1f}m |"
        )
    lines.append("")

    return "\n".join(lines)


def assemble_report(
    channel_analyses: list[ChannelAnalysis],
    gap: GapAnalysis | ScoredGapAnalysis,
    decision_options: list[DecisionOption],
    gameweek: int,  # noqa: ARG001  # Reserved for future use
    commentary: str | None = None,
    quality_review: QualityReview | None = None,
    condensed_players: list[dict[str, Any]] | None = None,
    squad_names: set[str] | None = None,
) -> str:
    """Assemble final markdown report from stage outputs."""
    sections = []

    if commentary:
        sections.append(f"**User Directive:** {commentary}\n")

    sections.append(generate_consensus_section(channel_analyses))
    sections.append(generate_channel_notes(channel_analyses))

    # Chip strategy section (from consensus data)
    consensus = aggregate_influencer_consensus(channel_analyses)
    chip_section = generate_chip_strategy_section(
        consensus.get("chip_consensus", {}), consensus["total_channels"]
    )
    if chip_section:
        sections.append(chip_section)

    sections.append(format_gap_section(gap))

    # Ownership section
    if condensed_players and squad_names is not None:
        ownership_section = generate_ownership_section(condensed_players, squad_names)
        if ownership_section:
            sections.append(ownership_section)

    sections.append(format_action_plan(decision_options))

    # Add quality review section if available
    if quality_review:
        sections.append(format_quality_review(quality_review))

    return "\n\n".join(sections)


__all__ = [
    "assemble_report",
    "format_action_plan",
    "format_gap_section",
    "format_quality_review",
    "generate_channel_notes",
    "generate_chip_strategy_section",
    "generate_consensus_section",
    "generate_ownership_section",
]
