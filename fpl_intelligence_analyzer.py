#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.31.0",
#   "pydantic>=2.7.0",
#   "tenacity>=8.2.3"
# ]
# ///

"""
FPL Intelligence Analyzer

Processes FPL data aggregator output through LLM analysis to generate
transfer and captain recommendations using multiple influencer perspectives.

Usage:
    ./fpl_intelligence_analyzer.py --input fpl_analysis_results_clean.json
    ./fpl_intelligence_analyzer.py --input data.json --output-file analysis.md --verbose
    ./fpl_intelligence_analyzer.py --input data.json --output-file analysis.md --free-transfers 2
    ./fpl_intelligence_analyzer.py --input data.json -o analysis.md -ft 0  # Must take hit or roll
    ./fpl_intelligence_analyzer.py --input data.json --commentary "Plan to wildcard, recommend only wildcard path"

Use `--commentary` to inject a high-priority user directive that the analysis must follow.
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import anthropic
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.fpl_influencer_hivemind.types import (
    GapAnalysis,
    LineupPlan,
    QualityReview,
    Transfer,
    TransferPlan,
    ValidationResult,
)

# Current Premier League teams (2025/2026 season)
PL_TEAMS_2025_26 = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]

# Prompt block for current PL teams context
PL_TEAMS_CONTEXT = """CURRENT PREMIER LEAGUE TEAMS (2025/2026):
Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Chelsea, Crystal Palace,
Everton, Fulham, Liverpool, Manchester City, Manchester United, Newcastle United,
Nottingham Forest, Tottenham Hotspur, West Ham United, Wolverhampton Wanderers,
Burnley, Leeds United, Sunderland

IMPORTANT:
- ALL 20 teams above are current Premier League teams
- DO NOT use training knowledge about team league status
- Trust the FPL API data for players and their teams - it's authoritative
"""


class ChannelAnalysis(BaseModel):
    """Pydantic model for individual channel analysis."""

    channel_name: str
    formation: str | None = None
    team_selection: list[str] = []
    transfers_in: list[str] = []
    transfers_out: list[str] = []
    captain_choice: str = "Not specified"
    vice_captain_choice: str = "Not specified"
    key_issues_discussed: list[dict[str, str]] = []
    watchlist: list[dict[str, str]] = []
    bank_itb: str | None = None
    key_reasoning: list[str] = []
    confidence: float = 0.5
    transcript_length: int = 0


class FPLIntelligenceAnalyzer:
    """Main class for FPL intelligence analysis using LLM calls."""

    def __init__(self, verbose: bool = False, save_prompts: bool = True):
        """Initialize the analyzer with logging and Anthropic client."""
        self.setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        self.save_prompts = save_prompts
        self.prompts_dir: Path | None = None

        # Initialize Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.sonnet_model = "claude-opus-4-5-20251101"  # "claude-sonnet-4-20250514" // claude-sonnet-4-5-20250929
        self.opus_model = "claude-opus-4-5-20251101"  # "claude-opus-4-1-20250805"
        self.haiku_model = "claude-3-5-haiku-20241022"

        # Cohesion validation thresholds
        self.consensus_threshold = 3  # min influencers for warning
        self.majority_ratio = 0.6  # 4/6 = majority, triggers error

        # Cache for justification verification
        self._justification_cache: dict[str, dict[str, object]] = {}

    def setup_logging(self, verbose: bool) -> None:
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def save_debug_content(self, filename: str, content: str) -> None:
        """Save debug content to file if prompts directory is set."""
        if self.save_prompts and self.prompts_dir:
            debug_path = self.prompts_dir / filename
            debug_path.write_text(content, encoding="utf-8")
            self.logger.debug(f"Saved debug content to {debug_path}")

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
        """Remove lineup-specific fields that don't apply to upcoming gameweek decisions.

        The FPL API returns lineup data (captain, bench order) from the previous gameweek,
        which is irrelevant for recommendations about the upcoming gameweek.
        """
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

            response, stop_reason = self._make_anthropic_call(
                model=self.sonnet_model,
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

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _make_anthropic_call(
        self, model: str, prompt: str, system: str, max_tokens: int = 3500
    ) -> tuple[str, str]:
        """Make a call to the Anthropic API with retry logic."""
        try:
            self.logger.debug(f"Making API call to {model}")

            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

            parts: list[str] = []
            for block in message.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            response_text = "".join(parts).strip()

            stop_reason = getattr(message, "stop_reason", "") or ""
            usage = getattr(message, "usage", None)
            output_tokens = getattr(usage, "output_tokens", None) if usage else None
            self.logger.debug(
                "API call successful (stop_reason=%s, output_tokens=%s, response_length=%s)",
                stop_reason,
                output_tokens,
                len(response_text),
            )

            if stop_reason and stop_reason not in {"end_turn", "stop_sequence"}:
                self.logger.warning(
                    "Anthropic message returned stop_reason='%s' (response chars=%s)",
                    stop_reason,
                    len(response_text),
                )

            return response_text, stop_reason

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise

    def analyze_channel(
        self,
        channel_data: dict[str, Any],
        condensed_players: list[dict[str, Any]],
        transcript: str,
        gameweek: int,
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
            self.save_debug_content(f"{channel_name}_prompt.txt", prompt)

            response, stop_reason = self._make_anthropic_call(
                model=self.sonnet_model, prompt=prompt, system=system, max_tokens=3500
            )

            # Save response if debug mode is on
            self.save_debug_content(f"{channel_name}_response.json", response)

            # Parse JSON response
            analysis_data: dict[str, Any] | None = None

            try:
                # Clean up response if it has code fences
                if "```" in response:
                    import re

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

    # =========================================================================
    # Multi-Stage Analysis Methods
    # =========================================================================

    def _extract_last_json(self, response: str) -> str:
        """Extract the last valid JSON object from a response with multiple blocks.

        Models sometimes output multiple JSON blocks when they realize an error
        and self-correct. This extracts the last valid JSON block.
        """
        # Find all JSON blocks in markdown fences
        json_blocks = re.findall(
            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response, re.DOTALL
        )
        if json_blocks:
            # Return the last JSON block (most likely the corrected one)
            return json_blocks[-1]

        # Try to find raw JSON (not in fences)
        # Look for last complete JSON object
        cleaned = response.strip()
        if cleaned.startswith("{"):
            # Find matching closing brace
            brace_count = 0
            end_pos = 0
            for i, char in enumerate(cleaned):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
            if end_pos > 0:
                return cleaned[:end_pos]

        return cleaned

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

    def _compute_post_transfer_squad(
        self,
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

            # Find and replace the outgoing player
            for i, player in enumerate(new_squad):
                if player["name"] == out_name:
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

    def _aggregate_influencer_consensus(
        self,
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
                transfers_out_counts.setdefault(player, []).append(
                    analysis.channel_name
                )

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

    def _stage_gap_analysis(
        self,
        channel_analyses: list[ChannelAnalysis],
        squad_context: dict[str, Any],
        gameweek: int,
        transfer_momentum: dict[str, Any] | None = None,
        previous_errors: list[str] | None = None,
    ) -> GapAnalysis:
        """Stage 1: identify gaps between my squad and consensus."""
        self.logger.info("Stage 1: Gap Analysis")

        squad = squad_context["squad"]
        squad_names = {p["name"] for p in squad}
        consensus = self._aggregate_influencer_consensus(channel_analyses)

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

        prompt = f"""Analyze gaps between my FPL squad and influencer consensus for GW{gameweek}.
{momentum_section}

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

        self.save_debug_content("stage1_gap_analysis_prompt.txt", prompt)

        response, _ = self._make_anthropic_call(
            model=self.opus_model,
            prompt=prompt,
            system=system,
            max_tokens=2000,
        )

        self.save_debug_content("stage1_gap_analysis_response.json", response)

        # Parse response
        try:
            cleaned = self._extract_last_json(response)
            data = json.loads(cleaned)
            return GapAnalysis(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Stage 1 parse error: {e}")
            # Return minimal valid response
            return GapAnalysis(
                players_to_sell=[],
                players_missing=[],
                risk_flags=[],
                formation_gaps=[],
                captain_gap=None,
            )

    def _stage_transfer_plan(
        self,
        gap: GapAnalysis,
        squad_context: dict[str, Any],
        condensed_players: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        gameweek: int,
        previous_errors: list[str] | None = None,
    ) -> TransferPlan:
        """Stage 2: generate specific transfers to address gaps."""
        self.logger.info("Stage 2: Transfer Plan")

        squad = squad_context["squad"]
        itb = squad_context["itb"]
        fts = squad_context["free_transfers"]
        club_counts = squad_context["club_counts"]
        squad_names = {p["name"] for p in squad}

        consensus = self._aggregate_influencer_consensus(channel_analyses)

        # Build error feedback
        error_feedback = ""
        if previous_errors:
            error_feedback = (
                "\n\nPREVIOUS ATTEMPT FAILED WITH ERRORS:\n"
                + "\n".join(f"- {e}" for e in previous_errors)
                + "\n\nFix these issues in your response.\n"
            )

        prompt = f"""Generate specific FPL transfers for GW{gameweek} based on gap analysis.

GAP ANALYSIS:
{gap.model_dump_json(indent=2)}

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
3. cost_delta = in_price - selling_price.
4. new_itb = ITB - sum(cost_delta for all transfers) (must be >= 0).
5. Club count after transfers must be <= 3 for any club.
6. hit_cost = max(0, len(transfers) - fts) * 4.
7. If no transfers recommended, return empty transfers array with fts_remaining = {fts}.
8. backers should list influencer channel names when available; else [].

Return ONLY valid JSON, no markdown fences."""

        system = """You are an FPL transfer strategist. Generate specific, valid transfers respecting all FPL rules.
Use only the provided data (no external knowledge). Return valid JSON only."""

        self.save_debug_content("stage2_transfer_plan_prompt.txt", prompt)

        response, _ = self._make_anthropic_call(
            model=self.opus_model,
            prompt=prompt,
            system=system,
            max_tokens=2000,
        )

        self.save_debug_content("stage2_transfer_plan_response.json", response)

        # Parse response
        try:
            cleaned = self._extract_last_json(response)
            data = json.loads(cleaned)
            return TransferPlan(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Stage 2 parse error: {e}")
            return TransferPlan(
                transfers=[],
                total_cost=0.0,
                new_itb=itb,
                fts_used=0,
                fts_remaining=fts,
                hit_cost=0,
                reasoning="Failed to generate transfer plan",
            )

    def _stage_lineup_selection(
        self,
        post_transfer_squad: list[dict[str, Any]],
        channel_analyses: list[ChannelAnalysis],
        gameweek: int,
        previous_errors: list[str] | None = None,
    ) -> LineupPlan:
        """Stage 3: select XI, bench, captain from post-transfer squad."""
        self.logger.info("Stage 3: Lineup Selection")

        squad_names = [p["name"] for p in post_transfer_squad]

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

        prompt = f"""Select starting XI, bench, and captain for GW{gameweek}.

POST-TRANSFER SQUAD (15 players - use ONLY these):
{json.dumps(post_transfer_squad, indent=2)}

VALID PLAYER NAMES (use exact names from this list):
{json.dumps(squad_names, indent=2)}

INFLUENCER CAPTAINCY CHOICES:
{json.dumps(captain_data, indent=2)}
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
5. Use EXACT names from VALID PLAYER NAMES list.
6. Player format: "PlayerName (POS)".
7. starting_xi order preference: GK, DEFs, MIDs, FWDs (helps readability).

Return ONLY valid JSON, no markdown fences."""

        system = """You are an FPL lineup selector. Choose optimal starting XI, bench order, and captain.
Use only the provided squad data (no external knowledge). Return valid JSON only."""

        self.save_debug_content("stage3_lineup_selection_prompt.txt", prompt)

        response, _ = self._make_anthropic_call(
            model=self.opus_model,
            prompt=prompt,
            system=system,
            max_tokens=2000,
        )

        self.save_debug_content("stage3_lineup_selection_response.json", response)

        # Parse response
        try:
            cleaned = self._extract_last_json(response)
            data = json.loads(cleaned)
            return LineupPlan(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Stage 3 parse error: {e}")
            # Return minimal valid lineup
            return LineupPlan(
                starting_xi=[],
                bench=[],
                captain="",
                vice_captain="",
                formation="",
                reasoning="Failed to generate lineup",
            )

    # =========================================================================
    # Cohesion Validation Methods
    # =========================================================================

    def _verify_justification_llm(
        self,
        player_name: str,
        reasoning_texts: list[str],
        context: str,
    ) -> dict[str, object]:
        """Use LLM to verify if a player exclusion is justified in reasoning."""
        # Check cache first
        cache_key = f"{player_name}:{hash(tuple(reasoning_texts))}"
        if cache_key in self._justification_cache:
            return self._justification_cache[cache_key]

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
            response, _ = self._make_anthropic_call(
                model=self.haiku_model,
                prompt=prompt,
                system="Analyze transfer reasoning. Return JSON only.",
                max_tokens=200,
            )
            cleaned = self._extract_last_json(response)
            result = json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            self.logger.debug(f"Justification check failed for {player_name}: {e}")
            # Assume justified if we can't verify (fail open)
            result = {"justified": True, "reason": "Could not verify"}

        self._justification_cache[cache_key] = result
        return result

    def _compute_player_affordability(
        self,
        player_name: str,  # noqa: ARG002
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
            sell_price = p.get("sell_price", p.get("price", 0.0))
            if sell_price > max_sell:
                max_sell = sell_price

        max_budget = itb + max_sell
        is_affordable = player_price <= max_budget

        return is_affordable, max_budget

    def _validate_gap_to_transfer_cohesion(
        self,
        gap: GapAnalysis,
        transfers: TransferPlan,
        consensus: dict[str, Any],  # noqa: ARG002
    ) -> tuple[list[str], list[str]]:
        """Validate Stage 1 gaps are addressed in Stage 2 transfers."""
        errors: list[str] = []
        warnings: list[str] = []

        # Get transferred-in player names
        transferred_in = {
            t.in_player.split(" (")[0].lower() for t in transfers.transfers
        }
        transferred_out = {
            t.out_player.split(" (")[0].lower() for t in transfers.transfers
        }

        reasoning_texts = [transfers.reasoning]

        # Check captain_gap (ERROR if not addressed)
        if gap.captain_gap:
            captain_name = gap.captain_gap.lower()
            if captain_name not in transferred_in:
                justification = self._verify_justification_llm(
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
            player_name = player_ref.name.lower()
            if player_name not in transferred_in:
                justification = self._verify_justification_llm(
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
            player_name = player_ref.name.lower()
            if player_name not in transferred_out:
                justification = self._verify_justification_llm(
                    player_ref.name,
                    reasoning_texts,
                    f"Player identified to sell ({player_ref.position}, {player_ref.team}).",
                )
                if not justification.get("justified", False):
                    warnings.append(
                        f"COHESION ISSUE: Player to sell '{player_ref.name}' not transferred out"
                    )

        return errors, warnings

    def _validate_consensus_coverage(
        self,
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
        majority_count = int(total_channels * self.majority_ratio)

        # Get transferred-in player names
        transferred_in = {
            t.in_player.split(" (")[0].lower() for t in transfers.transfers
        }

        # Get current squad names
        squad_names = {
            p.get("name", "").lower() for p in squad_context.get("squad", [])
        }

        reasoning_texts = [transfers.reasoning]

        # Build player price lookup
        price_lookup: dict[str, float] = {}
        for p in condensed_players:
            name = p.get("web_name", "").lower()
            price_lookup[name] = p.get("price", 0.0)

        for player, backers in transfers_in_counts.items():
            backer_count = len(backers)
            player_lower = player.lower()

            # Skip if already in squad or already being transferred in
            if player_lower in squad_names or player_lower in transferred_in:
                continue

            # Check if recommended by threshold+ influencers
            if backer_count >= self.consensus_threshold:
                # Check affordability
                player_price = price_lookup.get(player_lower, 0.0)
                is_affordable, max_budget = self._compute_player_affordability(
                    player, player_price, squad_context
                )

                context = (
                    f"Recommended by {backer_count} influencers: {', '.join(backers)}. "
                    f"Price: £{player_price}m. "
                    f"{'Affordable' if is_affordable else 'Not affordable'} (max budget: £{max_budget}m)."
                )

                justification = self._verify_justification_llm(
                    player, reasoning_texts, context
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

    def _validate_risk_contingency(
        self,
        gap: GapAnalysis,
        lineup: LineupPlan,
    ) -> tuple[list[str], list[str]]:
        """Validate risk flags have corresponding contingency in lineup."""
        errors: list[str] = []
        warnings: list[str] = []

        # Build set of risky player names
        risky_players = {rf.player.lower(): rf.risk for rf in gap.risk_flags}

        if not risky_players:
            return errors, warnings

        # Get XI and bench names
        xi_names = [p.split(" (")[0].lower() for p in lineup.starting_xi]
        bench_names = [p.split(" (")[0].lower() for p in lineup.bench]

        captain_name = lineup.captain.split(" (")[0].lower() if lineup.captain else ""
        vice_name = (
            lineup.vice_captain.split(" (")[0].lower() if lineup.vice_captain else ""
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
                            lineup.bench[bench_idx]
                            if bench_idx < len(lineup.bench)
                            else ""
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

    def _validate_transfers(
        self,
        transfers: TransferPlan,
        original_squad: list[dict[str, Any]],
        post_transfer_squad: list[dict[str, Any]],
    ) -> list[str]:
        """Validate transfer plan."""
        errors: list[str] = []
        original_names = {p["name"] for p in original_squad}

        for t in transfers.transfers:
            out_name = t.out_player.split(" (")[0]
            in_name = t.in_player.split(" (")[0]

            # Out player must be in original squad
            if out_name not in original_names:
                errors.append(f"Transfer OUT '{out_name}' not in squad")

            # In player must NOT be in original squad
            if in_name in original_names:
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

    def _validate_lineup(
        self,
        lineup: LineupPlan,
        post_transfer_squad: list[dict[str, Any]],
    ) -> list[str]:
        """Validate lineup plan."""
        errors: list[str] = []
        post_names = {p["name"] for p in post_transfer_squad}

        xi_names = {p.split(" (")[0] for p in lineup.starting_xi}
        bench_names = {p.split(" (")[0] for p in lineup.bench}

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
        all_lineup = xi_names | bench_names
        for name in all_lineup:
            if name not in post_names:
                errors.append(f"'{name}' not in post-transfer squad")

        return errors

    def _validate_all(
        self,
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
        transfer_errors = self._validate_transfers(
            transfers, original_squad, post_transfer_squad
        )
        errors.extend(transfer_errors)

        # Validate lineup (mechanical)
        lineup_errors = self._validate_lineup(lineup, post_transfer_squad)
        errors.extend(lineup_errors)

        # Cohesion validation (if data available)
        if consensus is not None:
            # Gap to transfer cohesion
            gap_errors, gap_warnings = self._validate_gap_to_transfer_cohesion(
                gap, transfers, consensus
            )
            errors.extend(gap_errors)
            warnings.extend(gap_warnings)

            # Consensus coverage
            if squad_context is not None and condensed_players is not None:
                consensus_errors, consensus_warnings = (
                    self._validate_consensus_coverage(
                        transfers, consensus, squad_context, condensed_players
                    )
                )
                errors.extend(consensus_errors)
                warnings.extend(consensus_warnings)

        # Risk contingency validation
        risk_errors, risk_warnings = self._validate_risk_contingency(gap, lineup)
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

    def _holistic_quality_review(
        self,
        gap: GapAnalysis,
        transfers: TransferPlan,
        lineup: LineupPlan,
        consensus: dict[str, Any],
        squad_context: dict[str, Any],
        gameweek: int,
    ) -> QualityReview:
        """Final holistic LLM review of the complete report for quality assessment."""
        self.logger.info("Running holistic quality review")

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

        self.save_debug_content("holistic_review_prompt.txt", prompt)

        try:
            response, _ = self._make_anthropic_call(
                model=self.sonnet_model,
                prompt=prompt,
                system=system,
                max_tokens=1500,
            )

            self.save_debug_content("holistic_review_response.json", response)

            cleaned = self._extract_last_json(response)
            data = json.loads(cleaned)
            return QualityReview(**data)

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Holistic review parse error: {e}")
            # Return neutral fallback
            return QualityReview(
                confidence_score=0.5,
                quality_notes=["Automated quality review could not be completed"],
                consensus_alignment="Unable to assess",
                risk_assessment="Unable to assess",
                potential_issues=[],
                recommendation_strength="moderate",
            )

    def _run_staged_analysis(
        self,
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        my_team_data: dict[str, Any],
        gameweek: int,
        free_transfers: int,
        transfer_momentum: dict[str, Any] | None = None,
        commentary: str | None = None,  # noqa: ARG002
        max_stage_attempts: int = 2,
    ) -> tuple[GapAnalysis, TransferPlan, LineupPlan, QualityReview]:
        """Orchestrate all stages with validation retry loop and final quality review."""
        self.logger.info("Starting multi-stage analysis")

        squad_context = self._build_squad_context(my_team_data, free_transfers)
        original_squad = squad_context["squad"]

        # Compute consensus once for cohesion validation
        consensus = self._aggregate_influencer_consensus(channel_analyses)

        # Stage 1: Gap Analysis (2 attempts)
        gap: GapAnalysis | None = None
        gap_errors: list[str] = []
        for attempt in range(max_stage_attempts):
            self.logger.info(f"Stage 1 attempt {attempt + 1}/{max_stage_attempts}")
            gap = self._stage_gap_analysis(
                channel_analyses,
                squad_context,
                gameweek,
                transfer_momentum=transfer_momentum,
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
            transfers = self._stage_transfer_plan(
                gap,
                squad_context,
                condensed_players,
                channel_analyses,
                gameweek,
                previous_errors=transfer_errors if transfer_errors else None,
            )

            post_transfer_squad = self._compute_post_transfer_squad(
                original_squad, transfers.transfers
            )

            # Mechanical validation
            transfer_errors = self._validate_transfers(
                transfers, original_squad, post_transfer_squad
            )

            # Cohesion validation (gap → transfer consistency)
            cohesion_errors, cohesion_warnings = (
                self._validate_gap_to_transfer_cohesion(gap, transfers, consensus)
            )
            transfer_errors.extend(cohesion_errors)

            # Consensus coverage validation
            consensus_errors, consensus_warnings = self._validate_consensus_coverage(
                transfers, consensus, squad_context, condensed_players
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
            lineup = self._stage_lineup_selection(
                post_transfer_squad,
                channel_analyses,
                gameweek,
                previous_errors=lineup_errors if lineup_errors else None,
            )

            # Mechanical validation
            lineup_errors = self._validate_lineup(lineup, post_transfer_squad)

            # Risk contingency validation
            risk_errors, risk_warnings = self._validate_risk_contingency(gap, lineup)
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
        final_validation = self._validate_all(
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
        quality_review = self._holistic_quality_review(
            gap, transfers, lineup, consensus, squad_context, gameweek
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
                transfers = self._stage_transfer_plan(
                    gap,
                    squad_context,
                    condensed_players,
                    channel_analyses,
                    gameweek,
                    previous_errors=transfer_fixes,
                )
                post_transfer_squad = self._compute_post_transfer_squad(
                    original_squad, transfers.transfers
                )

            # Re-run lineup stage if needed (or if transfers changed)
            if lineup_fixes or transfer_fixes:
                self.logger.info(f"Re-running Stage 3 with {len(lineup_fixes)} fixes")
                lineup = self._stage_lineup_selection(
                    post_transfer_squad,
                    channel_analyses,
                    gameweek,
                    previous_errors=lineup_fixes if lineup_fixes else None,
                )

            # Re-run quality review on corrected output
            self.logger.info("Re-running quality review after corrections")
            quality_review = self._holistic_quality_review(
                gap, transfers, lineup, consensus, squad_context, gameweek
            )
            self.logger.info(
                f"Updated quality review: confidence={quality_review.confidence_score:.2f}, "
                f"strength={quality_review.recommendation_strength}, "
                f"remaining issues={len(quality_review.fixable_issues)}"
            )

        return gap, transfers, lineup, quality_review

    def _generate_consensus_section(
        self, channel_analyses: list[ChannelAnalysis]
    ) -> str:
        """Build Section 1 from aggregated channel data."""
        consensus = self._aggregate_influencer_consensus(channel_analyses)
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
                lines.append(
                    f"| {cap} | {', '.join(backers)} | {len(backers)}/{total} |"
                )
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

    def _generate_channel_notes(self, channel_analyses: list[ChannelAnalysis]) -> str:
        """Build Section 2: Channel-by-Channel Notes."""
        lines = ["## 2) Channel-by-Channel Notes\n"]

        for analysis in channel_analyses:
            lines.append(f"### {analysis.channel_name}")
            lines.append(f"*Confidence: {analysis.confidence}*\n")

            if analysis.formation:
                lines.append(f"- **Formation:** {analysis.formation}")

            if analysis.team_selection:
                lines.append(
                    f"- **Team Selection:** {', '.join(analysis.team_selection)}"
                )

            if analysis.transfers_in:
                lines.append(f"- **Transfers IN:** {', '.join(analysis.transfers_in)}")

            if analysis.transfers_out:
                lines.append(
                    f"- **Transfers OUT:** {', '.join(analysis.transfers_out)}"
                )

            lines.append(f"- **Captain:** {analysis.captain_choice}")
            lines.append(f"- **Vice Captain:** {analysis.vice_captain_choice}")

            if analysis.watchlist:
                watch_strs = [
                    f"{w['name']} ({w.get('priority', 'med')})"
                    for w in analysis.watchlist
                ]
                lines.append(f"- **Watchlist:** {', '.join(watch_strs)}")

            if analysis.key_issues_discussed:
                lines.append("- **Key Issues:**")
                for issue in analysis.key_issues_discussed[:3]:
                    lines.append(
                        f"  - {issue.get('issue', '')}: {issue.get('opinion', '')}"
                    )

            if analysis.key_reasoning:
                lines.append(
                    f"- **Reasoning:** {'; '.join(analysis.key_reasoning[:3])}"
                )

            lines.append("")

        return "\n".join(lines)

    def _format_gap_section(self, gap: GapAnalysis) -> str:
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

    def _format_action_plan(self, transfers: TransferPlan, lineup: LineupPlan) -> str:
        """Format Section 4 from TransferPlan + LineupPlan."""
        lines = ["## 4) Action Plan\n"]

        # Transfers
        lines.append("### Recommended Transfers\n")
        if transfers.transfers:
            for t in transfers.transfers:
                backers = ", ".join(t.backers) if t.backers else "General consensus"
                lines.append(f"- **{t.out_player}** → **{t.in_player}**")
                lines.append(f"  - Sell: {t.selling_price}m, Buy: {t.in_price}m")
                lines.append(f"  - Cost delta: {t.cost_delta:+.1f}m")
                lines.append(f"  - Backers: {backers}")
            lines.append("")
            lines.append(f"**Budget after transfers:** {transfers.new_itb:.1f}m ITB")
            lines.append(
                f"**FTs used:** {transfers.fts_used}, remaining: {transfers.fts_remaining}"
            )
            if transfers.hit_cost > 0:
                lines.append(f"**Hit cost:** -{transfers.hit_cost} points")
        else:
            lines.append("*No transfers recommended - roll the transfer.*")
            lines.append(f"**FTs remaining:** {transfers.fts_remaining}")

        lines.append(f"\n**Reasoning:** {transfers.reasoning}\n")

        # Lineup
        lines.append(f"### Starting XI ({lineup.formation})\n")
        lines.append(f"**Captain:** {lineup.captain}")
        lines.append(f"**Vice Captain:** {lineup.vice_captain}\n")

        # Group by position
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

        lines.append("\n### Bench (auto-sub order)\n")
        for i, p in enumerate(lineup.bench, 1):
            lines.append(f"{i}. {p}")

        lines.append(f"\n**Reasoning:** {lineup.reasoning}")

        return "\n".join(lines)

    def _assemble_report(
        self,
        channel_analyses: list[ChannelAnalysis],
        gap: GapAnalysis,
        transfers: TransferPlan,
        lineup: LineupPlan,
        gameweek: int,  # noqa: ARG002
        commentary: str | None = None,
        quality_review: QualityReview | None = None,
    ) -> str:
        """Assemble final markdown report from stage outputs."""
        sections = []

        if commentary:
            sections.append(f"**User Directive:** {commentary}\n")

        sections.append(self._generate_consensus_section(channel_analyses))
        sections.append(self._generate_channel_notes(channel_analyses))
        sections.append(self._format_gap_section(gap))
        sections.append(self._format_action_plan(transfers, lineup))

        # Add quality review section if available
        if quality_review:
            sections.append(self._format_quality_review(quality_review))

        return "\n\n".join(sections)

    def _format_quality_review(self, review: QualityReview) -> str:
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

    def _legacy_comparative_analysis(
        self,
        channel_analyses: list[ChannelAnalysis],
        condensed_players: list[dict[str, Any]],
        my_team_summary: str,
        gameweek: int,
        commentary: str | None = None,
    ) -> str:
        """Generate final comparative analysis using Opus-4."""
        try:
            self.logger.info("Generating comparative analysis with Opus-4")

            # Prepare channel summaries with enhanced structured data
            summaries = []
            for analysis in channel_analyses:
                # Format key issues
                key_issues = ""
                if analysis.key_issues_discussed:
                    issues_list = [
                        f"'{issue['issue']}': {issue['opinion']}"
                        for issue in analysis.key_issues_discussed
                    ]
                    key_issues = f"\n- Key Issues: {'; '.join(issues_list)}"

                # Format watchlist
                watchlist = ""
                if analysis.watchlist:
                    watch_items = [
                        f"{item['name']} ({item['priority']}: {item['why']})"
                        for item in analysis.watchlist
                    ]
                    watchlist = f"\n- Watchlist: {'; '.join(watch_items)}"

                # Format formation and bank
                formation = (
                    f"\n- Formation: {analysis.formation}" if analysis.formation else ""
                )
                bank = f"\n- Bank ITB: {analysis.bank_itb}" if analysis.bank_itb else ""

                summary = f"""
**{analysis.channel_name}** (Confidence: {analysis.confidence}){formation}
- Team Selection: {", ".join(analysis.team_selection) if analysis.team_selection else "Not specified"}
- Transfers In: {", ".join(analysis.transfers_in) if analysis.transfers_in else "None"}
- Transfers Out: {", ".join(analysis.transfers_out) if analysis.transfers_out else "None"}
- Captain: {analysis.captain_choice}
- Vice Captain: {analysis.vice_captain_choice}{bank}{key_issues}{watchlist}
- General Reasoning: {"; ".join(analysis.key_reasoning) if analysis.key_reasoning else "Not provided"}
"""
                summaries.append(summary)

            combined_summaries = "\n".join(summaries)

            directive_section = ""
            critical_directive_line = ""
            if commentary:
                directive_text = commentary.strip()
                if directive_text:
                    directive_section = (
                        "USER PRIMARY DIRECTIVE (DO NOT IGNORE):\n"
                        f"- {directive_text}\n\n"
                    )
                    critical_directive_line = (
                        f"- PRIMARY USER DIRECTIVE (NON-NEGOTIABLE): {directive_text}\n"
                    )

            # Create comprehensive prompt for Opus-4
            prompt = f"""Generate a comprehensive FPL analysis report for gameweek {gameweek}.

MY CURRENT TEAM:
{my_team_summary}

TOP PLAYERS REFERENCE (first 30 - includes injury/availability data):
{json.dumps(condensed_players[:30], indent=1)}

PLAYER STATUS CODES:
- a: available, d: doubtful, i: injured, s: suspended, u: unavailable
- Consider "status", "news", and "chance_of_playing_next_round" in all recommendations

INFLUENCER ANALYSES:
{combined_summaries}

You are an analyst that turns influencer summaries/transcripts + my FPL squad data into a concise, actionable gameweek report.

{directive_section}CRITICAL ANALYSIS REQUIREMENT:
- ALWAYS check if universal/majority captain choices are in my squad
- If influencers are captaining a player I don't own, this is the #1 gap to highlight and fix!
- If I'm likely bench boosting then my starting XI doesn't matter
- If I'm likely wildcarding then my current squad doesn't matter
- If I'm likely free hitting then my future team or transfers don't matter

STYLE
- MAX 12000 characters and <=200 lines.
- Keep it terse and practical. Use bullets and compact tables. No fluff.
- Cite influencers inline like: (FPL Harry), (Let's Talk FPL, FPL Raptor). Never invent citations.
- Only use the four section headers below. Do not add others.

DEFINITIONS
- Universal = backed by all listed influencers
- Majority = backed by >50% of influencers
- Split = clear disagreement with no >50% majority
- Differential = either mentioned by a single influencer or (if ownership provided) <10% owned

DATA YOU CAN USE (may be partially provided)
- influencers[]: [channel name, formation_strategy?, key_transfers?, captain?, vice?, watchlist[], key_issues_discussed[], chip_talk?, starting_xi?, bench?, notable_rationale?]
- my_team: [squad[15] with positions/teams/prices/sell_prices, free_transfers (IMPORTANT: consider this number (and the number of hits beyond that number) when making transfer recommendations), planned_hits?, team_value, chips_available, risk_tolerance?]
- context (optional): [gw, deadline_utc, fixture_difficulty, minutes/injury flags, projections, blank/double indicators, price_change_risk]

FPL VALIDATION RULES FOR ALL RECOMMENDATIONS
- Obey budget (use sell_prices if provided; otherwise use current prices and label as estimate).
- Max 3 players per real club. E.g. can't recommend transferring in more than 3 players.
- Transfers must be the same position. E.g. you CAN'T transfer a MID for a FWD, you can only transfer a FWD for a FWD.
- Legal starting XI formation (GK x1, DEF 3-5, MID 2-5, FWD 1-3; 11 players total).
- Always provide bench order (1/2/3) and GK decision.
- Show remaining ITB after any proposed moves and the hit cost if applicable.
- If required info is missing, say “not stated” rather than guessing.

--------------------------------------------------
## 1) Consensus, Contrarian & Captaincy Snapshot
- **Universal picks**: brief bullets with why and citations.
- **Majority picks** (>50%): bullets with reasons and which channels back them.
- **Splits / debates**: A vs B with one-line reasons per side, cited.
- **Differentials to watch**: bullets with role (enabler/ceiling), cited.
- **Key cross-channel talking points**: 3-6 bullets summarizing shared themes (injuries, fixture swings, chip timing, formation trends).
- **Captaincy matrix**: a compact table:
  | Captain | Pros | Cons/Risk | Backers |
  Include safe vs upside notes. Add vice-captain considerations if mentioned.

## 2) Channel-by-Channel Notes (every influencer with a summary)
For each influencer name:
- **Ensure you list every single player in the starting XI, if known, and bench players**
- Formation strategy (if mentioned)
- Key transfers + one-line rationale (who OUT → who IN, est. cost)
- Captain (and vice if stated) + rationale
- Watchlist highlights (1-3)
- Top 3 issues discussed
- Chip talk (if any)
Keep this section concise; use bullets. Always cite this influencer after their opinions.

## 3) My Team vs Influencers (Gap Analysis)
- **Ensure you list every single player in my starting XI if known (with position), and bench**
- **Players I have (in starting xi or bench, doesn't matter) that are being benched/sold by influencers**: list with citations.
- **Popular picks I'm missing**: list by PRIORITY with brief why and names of backers.
  CRITICAL: If influencers are captaining a player you don't own, that's #1 priority!
  Order: 1) Universal/majority captains not in squad, 2) High ownership/selected players, 3) Differentials
- **Formation & XI differences**: note mismatches vs consensus trends.
- **Risk flags**: minutes/fitness/rotation or suspension risks in my squad.
- **Money & constraints**: current ITB, FTs, per-club counts that block moves.
Use a small table where helpful:
| Slot | My Player | Influencer View | Risk/Reason | Suggested Alt | Backers |

## 4) Action Plan (This GW + Short-Term)
Start with 1-3 **clear recommended paths** (transfers, captaincy, XI) for what to do this week. Highlight these upfront before going into scenarios.

### Transfers
- Use my current number of free transfers (FTs) from the MY TEAM data.
- Always consider:
  - Rolling transfers (now up to 5 can be banked).
  - Impact of taking hits: (extra transfers beyond free) x 4 points.
  - Budget, ITB, per-club limits, and position requirements.
- If missing a **universal captain choice**, prioritise bringing them in.
- Recommendations should include: OUT → IN, est. cost, new ITB, and which influencers back the move.

### Scenarios
- **0FT**: Must either roll or take a hit. Recommend whether a hit is justified or whether rolling is stronger.
- **1FT**: Suggest best use, but also say if rolling is sensible to set up 2FT next week.
- **2+ FTs (up to 5)**: Suggest optimal sequences. Note that using fewer than available means the rest can still be banked (until 5 max).

### If Considering a Hit (-4 or worse)
- Only suggest high-conviction moves with strong influencer backing and clear expected upside.
- Show hit cost calculation.
- If upside is marginal, recommend avoiding the hit or rolling.

### Starting XI & Bench Order
- List recommended XI by position.
- Show bench order (1/2/3) and highlight any 50/50 calls, with reasoning.

### Future Planning (GW+1 to GW+3)
- Suggest 2-4 priority future targets with reasoning (fixtures, form, price).
- Recommend when to roll to set up a bigger move.
- Call out potential early vs late transfer timing (e.g., due to price rises).

--------------------------------------------------
CRITICAL REQUIREMENTS
- {critical_directive_line}
- MAX 12000 characters and <=200 lines.
- Always attribute influencer opinions by name.
- Factor injuries/rotation into every rec.
- Enforce budget and per-club limits.
- Give conditional advice for 0FT/1FT/2FT and optional hits, if prudent.
- If data missing, state explicitly rather than guessing.
- Report must stay in clean Markdown.
"""

            system = """You are an expert FPL strategist and analyst. Generate comprehensive, actionable FPL advice
by analyzing multiple influencer perspectives alongside detailed player data. Your recommendations should be
specific, well-reasoned, and tailored to the user's current team situation."""

            # Save final prompt if debug mode is on
            self.save_debug_content("final_analysis_prompt.txt", prompt)

            response, stop_reason = self._make_anthropic_call(
                model=self.opus_model, prompt=prompt, system=system, max_tokens=6000
            )

            if stop_reason and stop_reason not in {"end_turn", "stop_sequence"}:
                self.logger.warning(
                    "Anthropic comparative analysis ended with stop_reason='%s'",
                    stop_reason,
                )

            # Save final response if debug mode is on
            self.save_debug_content("final_analysis_response.md", response)

            self.logger.info("Comparative analysis generated successfully")
            return response

        except Exception as e:
            self.logger.error(f"Error generating comparative analysis: {e}")
            return f"# Analysis Error\n\nFailed to generate comparative analysis: {e!s}"

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
                        video_data, condensed_players, transcript, gameweek
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
                transfer_momentum=transfer_momentum,
                commentary=commentary,
            )

            # Assemble final report from stage outputs
            final_report = self._assemble_report(
                channel_analyses,
                gap,
                transfers,
                lineup,
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


def main() -> int:
    """Main entry point for the FPL intelligence analyzer."""
    parser = argparse.ArgumentParser(
        description="FPL Intelligence Analyzer - Generate transfer/captain recommendations from influencer analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input fpl_analysis_results_clean.json
  %(prog)s --input data.json --output-file analysis_report.md --verbose
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to FPL aggregated data JSON file (e.g., fpl_analysis_results_clean.json)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="Path to write markdown analysis report (default: stdout)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-save-prompts",
        action="store_true",
        help="Disable saving prompts and responses to debug files",
    )
    parser.add_argument(
        "--free-transfers",
        "-ft",
        type=int,
        default=1,
        choices=range(0, 6),
        help="Number of free transfers available (0-5, default: 1)",
    )
    parser.add_argument(
        "--commentary",
        help="Optional high-priority user directive for the analysis",
    )

    args = parser.parse_args()

    try:
        analyzer = FPLIntelligenceAnalyzer(
            verbose=args.verbose, save_prompts=not args.no_save_prompts
        )
        analyzer.run_analysis(
            args.input,
            args.output_file,
            args.free_transfers,
            commentary=args.commentary,
        )
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
