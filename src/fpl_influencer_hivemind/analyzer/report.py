"""Report generation module for the FPL Intelligence Analyzer."""

from src.fpl_influencer_hivemind.analyzer.models import (
    ChannelAnalysis,
    DecisionOption,
)
from src.fpl_influencer_hivemind.analyzer.stages.gap import (
    aggregate_influencer_consensus,
)
from src.fpl_influencer_hivemind.types import GapAnalysis, QualityReview


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


def format_gap_section(gap: GapAnalysis) -> str:
    """Format Section 3 from GapAnalysis model."""
    lines = ["## 3) My Team vs Influencers (Gap Analysis)\n"]

    if gap.captain_gap:
        lines.append("### CRITICAL: Captain Gap")
        lines.append(
            f"Consensus captain **{gap.captain_gap}** is NOT in your squad!\n"
        )

    if gap.players_missing:
        lines.append("### Players I'm Missing (by priority)")
        for p in gap.players_missing:
            team_str = f" - {p.team}" if p.team else ""
            lines.append(f"- **{p.name}** ({p.position}){team_str}")
        lines.append("")

    if gap.players_to_sell:
        lines.append("### Players to Consider Selling")
        for p in gap.players_to_sell:
            team_str = f" - {p.team}" if p.team else ""
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

    return "\n".join(lines)


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


def assemble_report(
    channel_analyses: list[ChannelAnalysis],
    gap: GapAnalysis,
    decision_options: list[DecisionOption],
    gameweek: int,  # noqa: ARG001  # Reserved for future use
    commentary: str | None = None,
    quality_review: QualityReview | None = None,
) -> str:
    """Assemble final markdown report from stage outputs."""
    sections = []

    if commentary:
        sections.append(f"**User Directive:** {commentary}\n")

    sections.append(generate_consensus_section(channel_analyses))
    sections.append(generate_channel_notes(channel_analyses))
    sections.append(format_gap_section(gap))
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
    "generate_consensus_section",
]
