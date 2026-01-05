"""Orchestrator for the FPL Intelligence Analyzer."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from src.fpl_influencer_hivemind.analyzer.api import (
    AnthropicClient,
    configure_debug,
    save_debug_content,
)
from src.fpl_influencer_hivemind.analyzer.constants import PL_TEAMS_CONTEXT
from src.fpl_influencer_hivemind.analyzer.models import (
    ChannelAnalysis,
    DecisionOption,
    PlayerLookupEntry,
)
from src.fpl_influencer_hivemind.analyzer.normalization import (
    build_player_lookup,
    canonicalize_channel_analysis,
)
from src.fpl_influencer_hivemind.analyzer.quality import holistic_quality_review
from src.fpl_influencer_hivemind.analyzer.report import assemble_report
from src.fpl_influencer_hivemind.analyzer.stages.gap import (
    aggregate_influencer_consensus,
    stage_gap_analysis,
)
from src.fpl_influencer_hivemind.analyzer.stages.lineup import stage_lineup_selection
from src.fpl_influencer_hivemind.analyzer.stages.transfer import (
    apply_transfer_pricing,
    compute_post_transfer_squad,
    stage_transfer_plan,
)
from src.fpl_influencer_hivemind.analyzer.validation.cohesion import (
    validate_consensus_coverage,
    validate_gap_to_transfer_cohesion,
    validate_risk_contingency,
)
from src.fpl_influencer_hivemind.analyzer.validation.mechanical import (
    validate_all,
    validate_lineup,
    validate_transfers,
)
from src.fpl_influencer_hivemind.types import (
    GapAnalysis,
    LineupPlan,
    QualityReview,
    TransferPlan,
)


class FPLIntelligenceAnalyzer:
    """Main class for FPL intelligence analysis using LLM calls."""

    def __init__(self, verbose: bool = False, save_prompts: bool = True):
        """Initialize the analyzer with logging and Anthropic client."""
        self._setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        self.save_prompts = save_prompts
        self.prompts_dir: Path | None = None

        # Initialize Anthropic client
        self.client = AnthropicClient()

    def _setup_logging(self, verbose: bool) -> None:
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _merge_commentary(self, base: str | None, extra: str | None) -> str | None:
        """Merge commentary strings."""
        if base and extra:
            return f"{base}\n{extra}"
        return base or extra

    def load_aggregated_data(self, input_file: str) -> dict[str, Any]:
        """Load and parse the FPL aggregated data JSON file."""
        try:
            with Path(input_file).open() as f:
                data_raw = json.load(f)

            if not isinstance(data_raw, dict):
                raise ValueError("Aggregated data must be a JSON object")

            data = cast("dict[str, Any]", data_raw)

            self.logger.info(f"Loaded aggregated data from {input_file}")

            # Validate required structure
            required_keys = ["fpl_data", "youtube_analysis", "gameweek"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")

            return data

        except FileNotFoundError:
            self.logger.error(f"Input file not found: {input_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in input file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading aggregated data: {e}")
            raise

    def condense_player_list(
        self, top_players: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Condense the top players list to essential data only."""
        condensed_players = []

        for player in top_players:
            condensed_player = {
                "web_name": player.get("web_name", ""),
                "team_name": player.get("team_name", ""),
                "position": player.get("position", ""),
                "price": player.get("price", 0),
                "selected_by_percent": player.get("selected_by_percent", 0),
                "total_points": player.get("total_points", 0),
                "status": player.get("status", "a"),
                "news": player.get("news", ""),
                "chance_of_playing_next_round": player.get(
                    "chance_of_playing_next_round", 100
                ),
            }
            condensed_players.append(condensed_player)

        self.logger.debug(
            f"Condensed {len(top_players)} players to essential data with injury/availability info"
        )
        return condensed_players

    def _strip_lineup_metadata(
        self, current_picks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove lineup-specific fields that don't apply to upcoming gameweek decisions."""
        sanitized: list[dict[str, Any]] = []
        for pick in current_picks:
            sanitized.append(
                {
                    "element_id": pick.get("element_id"),
                    "web_name": pick.get("web_name"),
                    "team_name": pick.get("team_name"),
                    "player_position": pick.get("player_position"),
                    "price": pick.get("price"),
                    "total_points": pick.get("total_points"),
                    "selling_price": pick.get("selling_price"),
                }
            )
        return sanitized

    def format_my_team(
        self, my_team_data: dict[str, Any], free_transfers: int = 1
    ) -> str:
        """Format my team data into a readable description using LLM."""
        try:
            current_picks = my_team_data.get("current_picks", [])
            summary = my_team_data.get("summary", {})
            team_value_info = my_team_data.get("team_value", {})

            # Strip lineup metadata (captain, bench order) - not relevant for upcoming GW
            sanitized_picks = self._strip_lineup_metadata(current_picks)

            # Prepare team data for LLM formatting
            team_context = {
                "team_name": summary.get("team_name", "Unknown"),
                "total_points": summary.get("total_points", 0),
                "overall_rank": summary.get("overall_rank", 0),
                "gameweek_points": summary.get("gameweek_points", 0),
                "team_value": team_value_info.get("team_value", 0),
                "bank_balance": team_value_info.get("bank_balance", 0),
                "free_transfers": free_transfers,
                "squad": sanitized_picks,
            }

            prompt = f"""Format this FPL team data into a clear, readable summary:

{json.dumps(team_context, indent=2)}

Output requirements:
- Clear, concise text (no JSON, no code fences).
- Use short labeled sections in this order: Team, Performance, Squad, Finances.
- Squad section: group all 15 players by position (GKP, DEF, MID, FWD).
- For each player: name, team, price, total points, selling price.
- Finances: team value, bank balance, free transfers.
- Do NOT mention captain, vice-captain, starting XI, or bench.
"""

            response, stop_reason = self.client.call_sonnet(
                prompt=prompt,
                system=(
                    "You are an FPL analyst. Format team data clearly and concisely. "
                    "Use only the provided data and do not invent details."
                ),
            )

            if stop_reason and stop_reason not in {"end_turn", "stop_sequence"}:
                self.logger.warning(
                    "Anthropic call for team formatting ended with stop_reason='%s'",
                    stop_reason,
                )

            return response.strip()

        except Exception as e:
            self.logger.error(f"Error formatting team data: {e}")
            return f"Team formatting failed: {e!s}"

    def analyze_channel(
        self,
        channel_data: dict[str, Any],
        condensed_players: list[dict[str, Any]],
        transcript: str,
        gameweek: int,
        player_lookup: dict[str, list[PlayerLookupEntry]] | None = None,
    ) -> ChannelAnalysis | None:
        """Analyze a single channel's transcript using Sonnet-4."""
        channel_name = channel_data.get("channel_name", "Unknown")
        try:
            video_title = channel_data.get("title", "Unknown")
            transcript_length = len(transcript)

            self.logger.info(
                f"Analyzing channel: {channel_name} (transcript: {transcript_length} chars)"
            )

            # Warn if transcript is unusually short
            if transcript_length < 3000:
                self.logger.warning(
                    f"Short transcript for {channel_name}: only {transcript_length} characters"
                )

            # Create structured prompt for channel analysis
            prompt = f"""Analyze this FPL influencer's video transcript for gameweek {gameweek}.

CHANNEL: {channel_name}
VIDEO: {video_title}

TOP FPL PLAYERS BY OWNERSHIP (first 150, includes injury status):
{json.dumps(condensed_players[:150], indent=1)}

PLAYER STATUS CODES:
- a: available, d: doubtful, i: injured, s: suspended, u: unavailable
- Use "status", "news", and "chance_of_playing_next_round" where relevant

TRANSCRIPT:
{transcript}

Return ONLY valid JSON matching this schema (all keys required):
{{
  "channel_name": "{channel_name}",
  "formation": "3-5-2",
  "team_selection": ["Player (POS)", "..."],
  "transfers_in": ["Player (POS)", "..."],
  "transfers_out": ["Player (POS)", "..."],
  "captain_choice": "Player (POS)",
  "vice_captain_choice": "Player (POS)",
  "key_issues_discussed": [
    {{"issue": "topic", "opinion": "their view"}}
  ],
  "watchlist": [
    {{"name": "Player (POS)", "priority": "high|med|low", "why": "reason"}}
  ],
  "bank_itb": "0.5m",
  "key_reasoning": ["Reason 1", "Reason 2"],
  "confidence": 0.85,
  "transcript_length": {transcript_length}
}}

{PL_TEAMS_CONTEXT}

Rules:
- Player format: "Name (POS)" where POS is GKP/DEF/MID/FWD.
- Normalize names using the top players list when possible; otherwise keep transcript spelling.
- formation: null if not mentioned.
- captain_choice/vice_captain_choice: "Not specified" if not stated.
- bank_itb: string like "0.5m" if stated, otherwise null.
- Use empty arrays for missing lists.
- For short transcripts (<3000 chars), focus on transfers and captain; lower confidence.
- Do not invent players or decisions not stated in the transcript.
"""

            system = """You are an expert FPL analyst extracting structured data from influencer transcripts.
Focus on concrete decisions (team selection, transfers, captaincy, reasoning).
Return valid JSON only, matching the provided schema exactly."""

            # Save prompt if debug mode is on
            save_debug_content(f"{channel_name}_prompt.txt", prompt)

            response, stop_reason = self.client.call_sonnet(
                prompt=prompt, system=system, max_tokens=3500
            )

            # Save response if debug mode is on
            save_debug_content(f"{channel_name}_response.json", response)

            # Parse JSON response
            analysis_data: dict[str, Any] | None = None

            try:
                # Clean up response if it has code fences
                if "```" in response:
                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
                    )
                    if json_match:
                        response = json_match.group(1)

                parsed = json.loads(response)
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError("Expected JSON object", response, 0)
                analysis_data = cast("dict[str, Any]", parsed)

                # Ensure transcript_length is set
                if "transcript_length" not in analysis_data:
                    analysis_data["transcript_length"] = transcript_length

                # Handle null values for required string fields
                if analysis_data.get("captain_choice") is None:
                    analysis_data["captain_choice"] = "Not specified"
                if analysis_data.get("vice_captain_choice") is None:
                    analysis_data["vice_captain_choice"] = "Not specified"

                if player_lookup:
                    analysis_data = canonicalize_channel_analysis(
                        analysis_data, player_lookup
                    )

                analysis = ChannelAnalysis(**analysis_data)

                self.logger.info(
                    f"Successfully analyzed {channel_name} (confidence: {analysis.confidence}, transcript: {transcript_length} chars)"
                )
                return analysis

            except json.JSONDecodeError as e:
                self.logger.error(
                    "JSON parsing failed for %s: %s (stop_reason=%s)\nResponse: %s...",
                    channel_name,
                    e,
                    stop_reason or "unknown",
                    response[:500],
                )
                # Try to create a minimal analysis with defaults
                try:
                    minimal_analysis = ChannelAnalysis(
                        channel_name=channel_name,
                        transcript_length=transcript_length,
                        confidence=0.3,
                        key_reasoning=[
                            f"Failed to parse full analysis - transcript too short or unclear ({transcript_length} chars)"
                        ],
                    )
                    self.logger.warning(f"Using minimal analysis for {channel_name}")
                    return minimal_analysis
                except Exception as fallback_error:
                    self.logger.error(f"Even minimal analysis failed: {fallback_error}")
                    return None

            except ValidationError as e:
                formatted_data = json.dumps(analysis_data or {}, indent=2)[:500]
                self.logger.error(
                    f"Validation failed for {channel_name}: {e}\nData: {formatted_data}..."
                )
                return None

        except Exception as e:
            self.logger.error(f"Error analyzing channel {channel_name}: {e}")
            return None

    def _build_squad_context(
        self,
        my_team_data: dict[str, Any],
        free_transfers: int,
    ) -> dict[str, Any]:
        """Build structured squad context for stage prompts."""
        current_picks = my_team_data.get("current_picks", [])
        team_value_info = my_team_data.get("team_value", {})

        squad: list[dict[str, Any]] = []
        club_counts: dict[str, int] = {}

        for pick in current_picks:
            player = {
                "name": pick.get("web_name", ""),
                "position": pick.get("player_position", ""),
                "team": pick.get("team_name", ""),
                "price": pick.get("price", 0.0),
                "selling_price": pick.get("selling_price", pick.get("price", 0.0)),
                "element_id": pick.get("element_id"),
            }
            squad.append(player)

            team = player["team"]
            club_counts[team] = club_counts.get(team, 0) + 1

        return {
            "squad": squad,
            "itb": team_value_info.get("bank_balance", 0.0),
            "free_transfers": free_transfers,
            "club_counts": club_counts,
        }

    def _run_staged_analysis(
        self,
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        my_team_data: dict[str, Any],
        gameweek: int,
        free_transfers: int,
        player_lookup: dict[str, list[PlayerLookupEntry]],
        transfer_momentum: dict[str, Any] | None = None,
        commentary: str | None = None,
        max_stage_attempts: int = 2,
    ) -> tuple[GapAnalysis, TransferPlan, LineupPlan, QualityReview]:
        """Orchestrate all stages with validation retry loop and final quality review."""
        self.logger.info("Starting multi-stage analysis")

        squad_context = self._build_squad_context(my_team_data, free_transfers)
        original_squad = squad_context["squad"]

        # Compute consensus once for cohesion validation
        consensus = aggregate_influencer_consensus(channel_analyses)

        # Stage 1: Gap Analysis (2 attempts)
        gap: GapAnalysis | None = None
        gap_errors: list[str] = []
        for attempt in range(max_stage_attempts):
            self.logger.info(f"Stage 1 attempt {attempt + 1}/{max_stage_attempts}")
            gap = stage_gap_analysis(
                self.client,
                channel_analyses,
                squad_context,
                gameweek,
                transfer_momentum=transfer_momentum,
                commentary=commentary,
                previous_errors=gap_errors if gap_errors else None,
            )
            # Gap analysis doesn't have strict validation, accept it
            break

        if gap is None:
            raise ValueError("Gap analysis failed")

        # Stage 2: Transfer Plan (2 attempts) - includes cohesion validation
        transfers: TransferPlan | None = None
        transfer_errors: list[str] = []
        post_transfer_squad: list[dict[str, Any]] = []

        for attempt in range(max_stage_attempts):
            self.logger.info(f"Stage 2 attempt {attempt + 1}/{max_stage_attempts}")
            transfers = stage_transfer_plan(
                self.client,
                gap,
                squad_context,
                condensed_players,
                channel_analyses,
                gameweek,
                commentary=commentary,
                previous_errors=transfer_errors if transfer_errors else None,
            )

            transfers, pricing_errors = apply_transfer_pricing(
                transfers, squad_context, player_lookup
            )

            post_transfer_squad = compute_post_transfer_squad(
                original_squad, transfers.transfers
            )

            # Mechanical validation
            transfer_errors = pricing_errors + validate_transfers(
                transfers, original_squad, post_transfer_squad
            )

            # Cohesion validation (gap → transfer consistency)
            cohesion_errors, cohesion_warnings = validate_gap_to_transfer_cohesion(
                self.client, gap, transfers, consensus
            )
            transfer_errors.extend(cohesion_errors)

            # Consensus coverage validation
            consensus_errors, consensus_warnings = validate_consensus_coverage(
                self.client, transfers, consensus, squad_context, condensed_players
            )
            transfer_errors.extend(consensus_errors)

            # Log warnings but don't fail on them
            for warning in cohesion_warnings + consensus_warnings:
                self.logger.warning(warning)

            if not transfer_errors:
                self.logger.info("Stage 2 validation passed")
                break
            else:
                self.logger.warning(f"Stage 2 validation failed: {transfer_errors}")

        if transfers is None:
            raise ValueError("Transfer plan failed")

        # Stage 3: Lineup Selection (2 attempts) - includes risk validation
        lineup: LineupPlan | None = None
        lineup_errors: list[str] = []

        for attempt in range(max_stage_attempts):
            self.logger.info(f"Stage 3 attempt {attempt + 1}/{max_stage_attempts}")
            lineup = stage_lineup_selection(
                self.client,
                post_transfer_squad,
                channel_analyses,
                gameweek,
                commentary=commentary,
                previous_errors=lineup_errors if lineup_errors else None,
            )

            # Mechanical validation
            lineup_errors = validate_lineup(lineup, post_transfer_squad)

            # Risk contingency validation
            risk_errors, risk_warnings = validate_risk_contingency(gap, lineup)
            lineup_errors.extend(risk_errors)

            # Log warnings but don't fail on them
            for warning in risk_warnings:
                self.logger.warning(warning)

            if not lineup_errors:
                self.logger.info("Stage 3 validation passed")
                break
            else:
                self.logger.warning(f"Stage 3 validation failed: {lineup_errors}")

        if lineup is None:
            raise ValueError("Lineup selection failed")

        # Final validation (all checks including cohesion)
        final_validation = validate_all(
            self.client,
            gap,
            transfers,
            lineup,
            original_squad,
            post_transfer_squad,
            consensus=consensus,
            squad_context=squad_context,
            condensed_players=condensed_players,
        )

        if not final_validation.valid:
            self.logger.warning(
                f"Final validation has errors (proceeding anyway): {final_validation.errors}"
            )

        if final_validation.warnings:
            self.logger.info(f"Final validation warnings: {final_validation.warnings}")

        # Holistic quality review (LLM assessment with corrective loop)
        quality_review = holistic_quality_review(
            self.client, gap, transfers, lineup, consensus, squad_context, gameweek
        )
        self.logger.info(
            f"Quality review: confidence={quality_review.confidence_score:.2f}, "
            f"strength={quality_review.recommendation_strength}"
        )

        # Corrective loop: if fixable issues found, re-run affected stages
        if quality_review.fixable_issues:
            self.logger.info(
                f"Quality review found {len(quality_review.fixable_issues)} fixable issues"
            )

            # Group issues by stage
            transfer_fixes = [
                f"QUALITY REVIEW FIX: {fi.issue} - {fi.fix_instruction}"
                for fi in quality_review.fixable_issues
                if fi.stage == "transfer"
            ]
            lineup_fixes = [
                f"QUALITY REVIEW FIX: {fi.issue} - {fi.fix_instruction}"
                for fi in quality_review.fixable_issues
                if fi.stage == "lineup"
            ]

            # Re-run transfer stage if needed
            if transfer_fixes:
                self.logger.info(f"Re-running Stage 2 with {len(transfer_fixes)} fixes")
                transfers = stage_transfer_plan(
                    self.client,
                    gap,
                    squad_context,
                    condensed_players,
                    channel_analyses,
                    gameweek,
                    commentary=commentary,
                    previous_errors=transfer_fixes,
                )
                transfers, _ = apply_transfer_pricing(
                    transfers, squad_context, player_lookup
                )
                post_transfer_squad = compute_post_transfer_squad(
                    original_squad, transfers.transfers
                )

            # Re-run lineup stage if needed (or if transfers changed)
            if lineup_fixes or transfer_fixes:
                self.logger.info(f"Re-running Stage 3 with {len(lineup_fixes)} fixes")
                lineup = stage_lineup_selection(
                    self.client,
                    post_transfer_squad,
                    channel_analyses,
                    gameweek,
                    commentary=commentary,
                    previous_errors=lineup_fixes if lineup_fixes else None,
                )

            # Re-run quality review on corrected output
            self.logger.info("Re-running quality review after corrections")
            quality_review = holistic_quality_review(
                self.client, gap, transfers, lineup, consensus, squad_context, gameweek
            )
            self.logger.info(
                f"Updated quality review: confidence={quality_review.confidence_score:.2f}, "
                f"strength={quality_review.recommendation_strength}, "
                f"remaining issues={len(quality_review.fixable_issues)}"
            )

        return gap, transfers, lineup, quality_review

    def _build_decision_options(
        self,
        gap: GapAnalysis,
        primary_transfers: TransferPlan,
        primary_lineup: LineupPlan,
        squad_context: dict[str, Any],
        original_squad: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        player_lookup: dict[str, list[PlayerLookupEntry]],
        gameweek: int,
        commentary: str | None = None,
    ) -> list[DecisionOption]:
        """Create 2-3 decision options ensuring meaningful differentiation.

        FT-aware logic:
        - Option A: Primary (influencer-aligned) - from staged analysis
        - Option B: Depends on Option A and FT count:
          - If A has transfers → roll (conservative)
          - If A has NO transfers but gaps exist → activate transfer
          - If A has NO transfers and no gaps → roll
        - Option C: FT-aware aggressive option:
          - 0 FTs + gaps → take hit for critical target
          - 1 FT + gaps → chase single target
          - 2+ FTs + multiple gaps → maximize FT usage
        """
        options: list[DecisionOption] = []
        fts = int(squad_context.get("free_transfers", 1))
        has_primary_transfers = len(primary_transfers.transfers) > 0
        has_gaps = bool(gap.players_missing or gap.players_to_sell)
        gap_count = len(gap.players_missing) + len(gap.players_to_sell)
        primary_transfer_count = len(primary_transfers.transfers)

        options.append(
            DecisionOption(
                label="Option A: Primary (influencer-aligned)",
                transfers=primary_transfers,
                lineup=primary_lineup,
                rationale="Best balance of consensus alignment and validation checks.",
            )
        )

        # Option B: differentiate based on what Option A recommends
        if has_primary_transfers:
            # Option A has transfers → Option B is conservative (roll)
            self._add_roll_option(
                options,
                squad_context,
                original_squad,
                channel_analyses,
                gameweek,
                primary_lineup,
                commentary,
            )
        elif has_gaps:
            # Option A has NO transfers but gaps exist → Option B activates a transfer
            self._add_activate_transfer_option(
                options,
                gap,
                squad_context,
                original_squad,
                channel_analyses,
                condensed_players,
                player_lookup,
                gameweek,
                primary_lineup,
                commentary,
            )
        else:
            # No transfers, no gaps → roll is the only alternative
            self._add_roll_option(
                options,
                squad_context,
                original_squad,
                channel_analyses,
                gameweek,
                primary_lineup,
                commentary,
            )

        # Option C: FT-aware aggressive option
        target = gap.captain_gap or (
            gap.players_missing[0].name if gap.players_missing else None
        )

        if fts == 0 and has_gaps:
            # 0 FTs but gaps exist → offer "take hit" option
            self._add_take_hit_option(
                options,
                gap,
                squad_context,
                original_squad,
                channel_analyses,
                condensed_players,
                player_lookup,
                gameweek,
                commentary,
            )
        elif fts >= 2 and gap_count >= 2 and primary_transfer_count < fts:
            # Multiple FTs and gaps, primary didn't use all → offer "maximize FTs" option
            self._add_maximize_fts_option(
                options,
                gap,
                squad_context,
                original_squad,
                channel_analyses,
                condensed_players,
                player_lookup,
                gameweek,
                fts,
                gap_count,
                commentary,
            )
        elif target:
            # Standard aggressive chase for single target
            self._add_aggressive_option(
                options,
                gap,
                squad_context,
                original_squad,
                channel_analyses,
                condensed_players,
                player_lookup,
                gameweek,
                target,
                commentary,
            )

        return options

    def _add_roll_option(
        self,
        options: list[DecisionOption],
        squad_context: dict[str, Any],
        original_squad: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        gameweek: int,
        fallback_lineup: LineupPlan,
        commentary: str | None,
    ) -> None:
        """Add conservative roll-transfer option."""
        conservative_commentary = self._merge_commentary(
            commentary,
            "CONSERVATIVE PLAN: make no transfers and prioritize safe minutes.",
        )
        roll_transfers = TransferPlan(
            transfers=[],
            total_cost=0.0,
            new_itb=float(squad_context.get("itb", 0.0)),
            fts_used=0,
            fts_remaining=int(squad_context.get("free_transfers", 0)),
            hit_cost=0,
            reasoning="Roll the transfer to bank flexibility and avoid hits.",
        )
        conservative_lineup = stage_lineup_selection(
            self.client,
            original_squad,
            channel_analyses,
            gameweek,
            commentary=conservative_commentary,
        )
        lineup_errors = validate_lineup(conservative_lineup, original_squad)
        if lineup_errors:
            self.logger.warning(
                "Conservative lineup failed validation; using primary lineup fallback"
            )
            conservative_lineup = fallback_lineup

        options.append(
            DecisionOption(
                label="Option B: Conservative (roll transfer)",
                transfers=roll_transfers,
                lineup=conservative_lineup,
                rationale="Lower variance: bank FTs and prioritize secure starters.",
            )
        )

    def _add_activate_transfer_option(
        self,
        options: list[DecisionOption],
        gap: GapAnalysis,
        squad_context: dict[str, Any],
        original_squad: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        player_lookup: dict[str, list[PlayerLookupEntry]],
        gameweek: int,
        fallback_lineup: LineupPlan,
        commentary: str | None,
    ) -> None:
        """Add option that activates a transfer when primary recommends none but gaps exist."""
        # Identify top gap target
        top_target = None
        if gap.players_missing:
            top_target = gap.players_missing[0].name
        elif gap.players_to_sell:
            top_target = f"replacement for {gap.players_to_sell[0].name}"

        activate_commentary = self._merge_commentary(
            commentary,
            f"ACTIVATE TRANSFER: gaps identified, address top gap ({top_target}). "
            "Do NOT roll - make exactly 1 transfer to address the most important gap.",
        )
        activate_transfers = stage_transfer_plan(
            self.client,
            gap,
            squad_context,
            condensed_players,
            channel_analyses,
            gameweek,
            commentary=activate_commentary,
        )
        activate_transfers, pricing_errors = apply_transfer_pricing(
            activate_transfers, squad_context, player_lookup
        )

        # If still no transfers after forcing, fall back to roll
        if not activate_transfers.transfers:
            self.logger.warning(
                "Activate transfer option still produced no transfers; falling back to roll"
            )
            self._add_roll_option(
                options,
                squad_context,
                original_squad,
                channel_analyses,
                gameweek,
                fallback_lineup,
                commentary,
            )
            return

        post_activate_squad = compute_post_transfer_squad(
            original_squad, activate_transfers.transfers
        )
        activate_errors = pricing_errors + validate_transfers(
            activate_transfers, original_squad, post_activate_squad
        )

        if activate_errors:
            self.logger.warning(
                "Activate transfer option failed validation: %s; falling back to roll",
                activate_errors,
            )
            self._add_roll_option(
                options,
                squad_context,
                original_squad,
                channel_analyses,
                gameweek,
                fallback_lineup,
                commentary,
            )
            return

        # Generate lineup for post-transfer squad
        activate_lineup = stage_lineup_selection(
            self.client,
            post_activate_squad,
            channel_analyses,
            gameweek,
            commentary=activate_commentary,
        )
        lineup_errors = validate_lineup(activate_lineup, post_activate_squad)
        if lineup_errors:
            self.logger.warning(
                "Activate transfer lineup failed validation; using primary lineup fallback"
            )
            activate_lineup = fallback_lineup

        options.append(
            DecisionOption(
                label="Option B: Activate transfer (address gaps)",
                transfers=activate_transfers,
                lineup=activate_lineup,
                rationale=f"Use FT to address gap: {top_target}.",
            )
        )

    def _add_aggressive_option(
        self,
        options: list[DecisionOption],
        gap: GapAnalysis,
        squad_context: dict[str, Any],
        original_squad: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        player_lookup: dict[str, list[PlayerLookupEntry]],
        gameweek: int,
        target: str,
        commentary: str | None,
    ) -> None:
        """Add aggressive chase option if validation passes."""
        chase_commentary = self._merge_commentary(
            commentary,
            f"AGGRESSIVE PLAN: prioritize bringing in {target} even if it requires a hit.",
        )
        chase_transfers = stage_transfer_plan(
            self.client,
            gap,
            squad_context,
            condensed_players,
            channel_analyses,
            gameweek,
            commentary=chase_commentary,
        )
        chase_transfers, pricing_errors = apply_transfer_pricing(
            chase_transfers, squad_context, player_lookup
        )
        post_chase_squad = compute_post_transfer_squad(
            original_squad, chase_transfers.transfers
        )
        chase_errors = pricing_errors + validate_transfers(
            chase_transfers, original_squad, post_chase_squad
        )

        if chase_errors:
            self.logger.warning(
                "Skipping aggressive option due to transfer validation errors: %s",
                chase_errors,
            )
            return

        chase_lineup = stage_lineup_selection(
            self.client,
            post_chase_squad,
            channel_analyses,
            gameweek,
            commentary=chase_commentary,
        )
        lineup_errors = validate_lineup(chase_lineup, post_chase_squad)
        if lineup_errors:
            self.logger.warning(
                "Skipping aggressive option due to lineup validation errors: %s",
                lineup_errors,
            )
            return

        options.append(
            DecisionOption(
                label="Option C: Aggressive (consensus chase)",
                transfers=chase_transfers,
                lineup=chase_lineup,
                rationale=f"High-conviction pivot toward {target} despite hit risk.",
            )
        )

    def _add_take_hit_option(
        self,
        options: list[DecisionOption],
        gap: GapAnalysis,
        squad_context: dict[str, Any],
        original_squad: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        player_lookup: dict[str, list[PlayerLookupEntry]],
        gameweek: int,
        commentary: str | None,
    ) -> None:
        """Add option to take a hit when 0 FTs but critical gaps exist."""
        top_target = None
        if gap.players_missing:
            top_target = gap.players_missing[0].name
        elif gap.players_to_sell:
            top_target = f"replacement for {gap.players_to_sell[0].name}"

        if not top_target:
            return

        hit_commentary = self._merge_commentary(
            commentary,
            f"TAKE HIT: 0 FTs available but critical gap identified ({top_target}). "
            "Make exactly 1 transfer and accept the -4 hit to address the most urgent gap.",
        )
        hit_transfers = stage_transfer_plan(
            self.client,
            gap,
            squad_context,
            condensed_players,
            channel_analyses,
            gameweek,
            commentary=hit_commentary,
        )
        hit_transfers, pricing_errors = apply_transfer_pricing(
            hit_transfers, squad_context, player_lookup
        )

        if not hit_transfers.transfers:
            self.logger.warning("Take hit option produced no transfers; skipping")
            return

        post_hit_squad = compute_post_transfer_squad(
            original_squad, hit_transfers.transfers
        )
        hit_errors = pricing_errors + validate_transfers(
            hit_transfers, original_squad, post_hit_squad
        )

        if hit_errors:
            self.logger.warning(
                "Skipping take hit option due to validation errors: %s", hit_errors
            )
            return

        hit_lineup = stage_lineup_selection(
            self.client,
            post_hit_squad,
            channel_analyses,
            gameweek,
            commentary=hit_commentary,
        )
        lineup_errors = validate_lineup(hit_lineup, post_hit_squad)
        if lineup_errors:
            self.logger.warning(
                "Skipping take hit option due to lineup errors: %s", lineup_errors
            )
            return

        options.append(
            DecisionOption(
                label="Option C: Take hit (address critical gap)",
                transfers=hit_transfers,
                lineup=hit_lineup,
                rationale=f"Accept -4 hit to address critical gap: {top_target}.",
            )
        )

    def _add_maximize_fts_option(
        self,
        options: list[DecisionOption],
        gap: GapAnalysis,
        squad_context: dict[str, Any],
        original_squad: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        player_lookup: dict[str, list[PlayerLookupEntry]],
        gameweek: int,
        fts: int,
        gap_count: int,
        commentary: str | None,
    ) -> None:
        """Add option to maximize FT usage when multiple FTs and gaps available."""
        use_count = min(fts, gap_count)
        gap_targets = []
        for p in gap.players_missing[:use_count]:
            gap_targets.append(p.name)
        for p in gap.players_to_sell[: use_count - len(gap_targets)]:
            gap_targets.append(f"sell {p.name}")

        max_commentary = self._merge_commentary(
            commentary,
            f"MAXIMIZE FTs: {fts} FTs available, {gap_count} gaps identified. "
            f"Use exactly {use_count} transfers to address: {', '.join(gap_targets)}. "
            "Don't waste FTs - unused transfers beyond 2 are lost.",
        )
        max_transfers = stage_transfer_plan(
            self.client,
            gap,
            squad_context,
            condensed_players,
            channel_analyses,
            gameweek,
            commentary=max_commentary,
        )
        max_transfers, pricing_errors = apply_transfer_pricing(
            max_transfers, squad_context, player_lookup
        )

        if not max_transfers.transfers:
            self.logger.warning("Maximize FTs option produced no transfers; skipping")
            return

        post_max_squad = compute_post_transfer_squad(
            original_squad, max_transfers.transfers
        )
        max_errors = pricing_errors + validate_transfers(
            max_transfers, original_squad, post_max_squad
        )

        if max_errors:
            self.logger.warning(
                "Skipping maximize FTs option due to validation errors: %s", max_errors
            )
            return

        max_lineup = stage_lineup_selection(
            self.client,
            post_max_squad,
            channel_analyses,
            gameweek,
            commentary=max_commentary,
        )
        lineup_errors = validate_lineup(max_lineup, post_max_squad)
        if lineup_errors:
            self.logger.warning(
                "Skipping maximize FTs option due to lineup errors: %s", lineup_errors
            )
            return

        options.append(
            DecisionOption(
                label=f"Option C: Maximize FTs (use {len(max_transfers.transfers)}/{fts})",
                transfers=max_transfers,
                lineup=max_lineup,
                rationale=f"Use available FTs to address multiple gaps: {', '.join(gap_targets)}.",
            )
        )

    def run_analysis(
        self,
        input_file: str,
        output_file: str | None = None,
        free_transfers: int = 1,
        commentary: str | None = None,
    ) -> None:
        """Run the complete FPL intelligence analysis pipeline.

        Args:
            input_file: Path to FPL aggregated data JSON file
            output_file: Optional path to write markdown analysis report
            free_transfers: Number of free transfers available (default: 1)
            commentary: Optional user directive to prioritise within the analysis
        """
        try:
            # Set up prompts directory if output file is specified
            if output_file and self.save_prompts:
                output_path = Path(output_file)
                self.prompts_dir = output_path.parent / f"{output_path.stem}_prompts"
                self.prompts_dir.mkdir(exist_ok=True)
                self.logger.info(f"Debug prompts will be saved to {self.prompts_dir}")
                configure_debug(self.save_prompts, self.prompts_dir)

            # Load aggregated data
            self.logger.info("Starting FPL Intelligence Analysis")
            data = self.load_aggregated_data(input_file)

            # Extract key components
            gameweek = data["gameweek"]["current"]
            top_players = data["fpl_data"]["top_players"]
            my_team_data = data["fpl_data"]["my_team"]
            transfer_momentum = data["fpl_data"].get("transfer_momentum", {})
            video_results = data["youtube_analysis"]["video_results"]
            transcripts = data["youtube_analysis"]["transcripts"]

            self.logger.info(f"Processing gameweek {gameweek} analysis")
            self.logger.info(
                f"Found {len(video_results)} channels with {len(transcripts)} transcripts"
            )

            # Phase 1: Individual channel analysis
            self.logger.info("Phase 1: Analyzing individual channels")
            condensed_players = self.condense_player_list(top_players)
            player_lookup = build_player_lookup(
                top_players, my_team_data, transfer_momentum
            )
            channel_analyses = []

            for video_data in video_results:
                channel_name = video_data.get("channel_name")
                video_id = video_data.get("video_id")
                if video_id and video_id in transcripts:
                    transcript_data = transcripts[video_id]
                    transcript = transcript_data.get("text") or transcript_data.get(
                        "transcript", ""
                    )
                    if not transcript and transcript_data.get("segments"):
                        # Join segment text as a last resort to keep backwards compatibility.
                        transcript = "\n".join(
                            segment.get("text", "")
                            for segment in transcript_data["segments"]
                            if segment.get("text")
                        )
                    analysis = self.analyze_channel(
                        video_data,
                        condensed_players,
                        transcript,
                        gameweek,
                        player_lookup=player_lookup,
                    )
                    if analysis:
                        channel_analyses.append(analysis)
                else:
                    self.logger.warning(
                        f"No transcript found for {channel_name or video_id or 'Unknown'}"
                    )

            if not channel_analyses:
                raise ValueError("No successful channel analyses - cannot proceed")

            self.logger.info(f"Successfully analyzed {len(channel_analyses)} channels")

            # Phase 2: Multi-stage analysis
            self.logger.info("Phase 2: Running multi-stage analysis")
            self.logger.info(f"Free transfers available: {free_transfers}")

            gap, transfers, lineup, quality_review = self._run_staged_analysis(
                channel_analyses,
                condensed_players,
                my_team_data,
                gameweek,
                free_transfers,
                player_lookup,
                transfer_momentum=transfer_momentum,
                commentary=commentary,
            )

            squad_context = self._build_squad_context(my_team_data, free_transfers)
            decision_options = self._build_decision_options(
                gap,
                transfers,
                lineup,
                squad_context,
                squad_context["squad"],
                channel_analyses,
                condensed_players,
                player_lookup,
                gameweek,
                commentary=commentary,
            )

            # Assemble final report from stage outputs
            final_report = assemble_report(
                channel_analyses,
                gap,
                decision_options,
                gameweek,
                commentary=commentary,
                quality_review=quality_review,
            )

            # Add header with metadata
            report_header = (
                f"# FPL Intelligence Analysis - Gameweek {gameweek}\n\n"
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
                f"**Channels Analyzed:** {len(channel_analyses)}  \n"
                f"**Data Source:** {Path(input_file).name}\n\n"
                "---\n\n"
            )

            complete_report = report_header + final_report

            # Output results
            if output_file:
                Path(output_file).write_text(complete_report, encoding="utf-8")
                self.logger.info(f"Analysis report written to: {output_file}")
                print("✅ FPL Intelligence Analysis complete!")
                print(f"📄 Report saved to: {output_file}")
            else:
                print(complete_report)

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise


__all__ = ["FPLIntelligenceAnalyzer"]
